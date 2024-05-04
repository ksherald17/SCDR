import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, DistilBertForTokenClassification
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn import KLDivLoss
import logging
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
NUM_EPOCHS = 3
BATCH_SIZE = 8
ALPHA = 0.99
EMA_UPDATE_PERIOD = 10

# Load and preprocess the dataset
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
dataset = load_dataset("conll2003")
NUM_LABELS = dataset['train'].features['ner_tags'].feature.num_classes

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding="max_length", is_split_into_words=True, return_token_type_ids=False)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

def sample_dataset(dataset, sample_size=0.1):
    return dataset.shuffle(seed=42).select(range(int(len(dataset) * sample_size)))

train_dataset = sample_dataset(tokenized_datasets["train"])
val_dataset = sample_dataset(tokenized_datasets["validation"])
test_dataset = sample_dataset(tokenized_datasets["test"])

# Initialize models
teacher1 = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
teacher2 = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
student1 = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS)
student2 = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS)

# Prepare DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher1.to(device)
teacher2.to(device)
student1.to(device)
student2.to(device)

# Optimizers
optimizer_s1 = AdamW(student1.parameters(), lr=5e-5)
optimizer_s2 = AdamW(student2.parameters(), lr=5e-5)

# Loss function
kl_div_loss = KLDivLoss(reduction='batchmean')

# Function to evaluate model
def evaluate_model(model, dataloader, device):
    model.eval()
    total_eval_loss = 0
    total_correct = 0
    total_examples = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
            logits = output.logits
            loss = cross_entropy(logits.view(-1, NUM_LABELS), batch['labels'].view(-1), ignore_index=-100)
            total_eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_labels = batch['labels'] != -100
            total_correct += (predictions == batch['labels']).sum().item()
            total_examples += correct_labels.sum().item()
    accuracy = total_correct / total_examples if total_examples > 0 else 0
    return total_eval_loss / len(dataloader), accuracy

def apply_ema(teacher, student, alpha):
    with torch.no_grad():
        teacher_params = {name: param for name, param in teacher.named_parameters()}
        student_params = {name: param for name, param in student.named_parameters()}
        
        for name, param in teacher_params.items():
            if name in student_params:
                student_param = student_params[name]
                if param.data.shape == student_param.data.shape:
                    param.data.copy_(alpha * param.data + (1 - alpha) * student_param.data)
                else:
                    logging.warning(f"Skipping EMA for {name} due to shape mismatch: {param.data.shape} vs {student_param.data.shape}")
            else:
                logging.warning(f"Student model missing parameter {name} for EMA")

def adjust_confidence_threshold(teacher, dataloader, device, percentile=75):
    all_confidences = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = teacher(**inputs).logits
            confidences = torch.softmax(outputs, dim=-1).max(dim=-1)[0]  # Get the maximum softmax output across classes for each token
            all_confidences.append(confidences)
    
    all_confidences = torch.cat(all_confidences)
    threshold = np.percentile(all_confidences.cpu().numpy(), percentile)
    return threshold

def enhanced_loss_function(outputs, labels, soft_labels, threshold):
    # Calculate the hard label loss using cross-entropy
    hard_loss = F.cross_entropy(outputs, labels, ignore_index=-100)
    
    # Calculate the soft label loss using KL divergence
    soft_loss = F.kl_div(F.log_softmax(outputs, dim=-1), F.softmax(soft_labels, dim=-1), reduction='batchmean')
    
    # Apply a threshold to determine which loss to prioritize
    confidence_mask = (torch.max(soft_labels, dim=-1)[0] > threshold).float()
    
    # Combine losses using the confidence mask
    combined_loss = (confidence_mask * soft_loss + (1 - confidence_mask) * hard_loss).mean()
    
    return combined_loss

# TensorBoard setup
writer = SummaryWriter()

# Training loop
best_val_accuracy = 0.0
early_stopping_counter = 0
early_stopping_rounds = 5
for epoch in range(NUM_EPOCHS):
    student1.train()
    student2.train()
    for i, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']

        # Teacher predictions and soft labels
        with torch.no_grad():
            teacher1_logits = teacher1(**batch).logits
            teacher2_logits = teacher2(**batch).logits
            soft_labels1 = F.softmax(teacher1_logits, dim=-1)
            soft_labels2 = F.softmax(teacher2_logits, dim=-1)
            threshold1 = adjust_confidence_threshold(validation_loader, teacher1, device=device)
            threshold2 = adjust_confidence_threshold(validation_loader, teacher2, device=device)

        # Update students using enhanced loss function with dynamic confidence threshold
        student1_loss = enhanced_loss_function(student1(**batch).logits, labels, soft_labels2, threshold2)
        student2_loss = enhanced_loss_function(student2(**batch).logits, labels, soft_labels1, threshold1)
        
        # Optimizer steps
        optimizer_s1.zero_grad()
        student1_loss.backward()
        optimizer_s1.step()

        optimizer_s2.zero_grad()
        student2_loss.backward()
        optimizer_s2.step()

        # TensorBoard logging for training
        writer.add_scalar('Loss/Student1', student1_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Loss/Student2', student2_loss.item(), epoch * len(train_loader) + i)

        # Apply EMA periodically
        if (i + 1) % EMA_UPDATE_PERIOD == 0:
            apply_ema(teacher1, student1, ALPHA)
            apply_ema(teacher2, student2, ALPHA)

    # Validation step and early stopping
    eval_st1_loss, eval_st1_accuracy = evaluate_model(student1, validation_loader, device)
    eval_st2_loss, eval_st2_accuracy = evaluate_model(student2, validation_loader, device)
    logging.info(f'Epoch {epoch+1}/{NUM_EPOCHS}, St1 [Validation Loss: {eval_st1_loss:.4f}, Accuracy: {eval_st1_accuracy:.3f}] | St2 [Validation Loss: {eval_st2_loss:.4f}, Accuracy: {eval_st2_accuracy:.3f}]')
    writer.add_scalar('Validation_Loss/Student1', eval_st1_loss, epoch)
    writer.add_scalar('Validation_Accuracy/Student1', eval_st1_accuracy, epoch)
    writer.add_scalar('Validation_Loss/Student2', eval_st2_loss, epoch)
    writer.add_scalar('Validation_Accuracy/Student2', eval_st2_accuracy, epoch)

    # Check for early stopping
    if eval_st1_accuracy > best_val_accuracy and eval_st2_accuracy > best_val_accuracy:
        best_val_accuracy = max(eval_st1_accuracy, eval_st2_accuracy)
        torch.save(student1.state_dict(), "student1_best_model.pt")
        torch.save(student2.state_dict(), "student2_best_model.pt")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_rounds:
            logging.info("Early stopping triggered!")
            break

# Evaluate on test set
test_st1_loss, test_st1_accuracy = evaluate_model(student1, test_loader, device)
test_st2_loss, test_st2_accuracy = evaluate_model(student2, test_loader, device)
logging.info(f'St1 [Test Loss: {test_st1_loss:.4f}, Accuracy: {test_st1_accuracy:.3f}] | St2 [Test Loss: {test_st2_loss:.4f}, Accuracy: {test_st2_accuracy:.3f}]')
writer.add_scalar('Test_Loss/Student1', test_st1_loss)
writer.add_scalar('Test_Accuracy/Student1', test_st1_accuracy)
writer.add_scalar('Test_Loss/Student2', test_st2_loss)
writer.add_scalar('Test_Accuracy/Student2', test_st2_accuracy)

# Close TensorBoard writer
writer.close()
