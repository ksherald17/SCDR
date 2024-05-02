import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, DistilBertForTokenClassification
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn.functional import cross_entropy, softmax, log_softmax, kl_div
from torch.nn import KLDivLoss
import logging
import os
from torch.utils.tensorboard import SummaryWriter

# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
ALPHA = 0.99            # EMA coefficient
NUM_EPOCHS = 3          # Number of training epochs
BATCH_SIZE = 8          # Batch size for training
EMA_UPDATE_PERIOD = 10  # Apply EMA updates every 10 batches
PATIENCE = 3             # Patience for early stopping

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

# Sampling a subset of the data for quicker iterations
def sample_dataset(dataset, sample_size=0.1):
    return dataset.shuffle(seed=42).select(range(int(len(dataset) * sample_size)))

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize models
teacher1 = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=NUM_LABELS)
teacher2 = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=NUM_LABELS)
student1 = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=NUM_LABELS)
student2 = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=NUM_LABELS)

# Prepare Dataset & Loaders
sampling = 0.1
train_loader = DataLoader(sample_dataset(tokenized_datasets["train"], sample_size=sampling), batch_size=BATCH_SIZE)
validation_loader = DataLoader(sample_dataset(tokenized_datasets["validation"], sample_size=sampling), batch_size=BATCH_SIZE)
test_loader = DataLoader(sample_dataset(tokenized_datasets["test"], sample_size=sampling), batch_size=BATCH_SIZE)

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

# Apply EMA (Student -> Teacher)
def apply_ema(teacher, student, alpha=ALPHA):
    with torch.no_grad():
        teacher_params = dict(teacher.named_parameters())
        student_params = dict(student.named_parameters())
        for name, param in teacher_params.items():
            if name in student_params:
                param.data.lerp_(student_params[name].data, 1 - alpha) # linear interpolation

def soft_label_cross_entropy(preds, soft_labels, true_labels, confidence_mask):
    # Ensure true labels are class indices for cross_entropy
    if true_labels.dim() > 1:
        true_labels = torch.argmax(true_labels, dim=-1)  # Convert one-hot encoded labels to class indices

    # Calculate the soft label loss using KL divergence
    soft_label_loss = F.kl_div(F.log_softmax(preds, dim=-1), soft_labels, reduction='none').sum(dim=-1)
    
    # Calculate the hard label loss
    true_label_loss = F.cross_entropy(preds.sum(dim=(1, 2)).float(), true_labels.float(), reduction='none')
    
    # Apply confidence mask to the soft label loss component
    combined_loss = confidence_mask * soft_label_loss + (1 - confidence_mask) * true_label_loss
    return combined_loss.mean()

# Generate a confidence mask for tokens where the maximum predicted probability 
# from the teacher model exceeds a given threshold.
def generate_confidence_mask(soft_labels, threshold=0.9):
    max_probs, _ = torch.max(soft_labels, dim=-1)  # Get the max probability for each token/class
    confidence_mask = (max_probs >= threshold).float()  # Compare to threshold and convert to float for masking
    return confidence_mask

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
            loss = cross_entropy(logits.view(-1, NUM_LABELS), batch['labels'].view(-1))
            total_eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_labels = batch['labels'] != -100
            total_correct += (predictions == batch['labels']).sum().item()
            total_examples += correct_labels.sum().item()
    accuracy = total_correct / total_examples if total_examples > 0 else 0
    return total_eval_loss / len(dataloader), accuracy

# Initialize tensorboard writer
writer = SummaryWriter()

# Training loop
best_accuracy = 0.0
patience_counter = 0
for epoch in range(NUM_EPOCHS):
    student1.train()
    student2.train()
    for i, batch in enumerate(train_loader): 
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']
        
        correct_predictions1 = 0
        total_predictions1 = 0
        correct_predictions2 = 0
        total_predictions2 = 0

        # Teacher predictions
        with torch.no_grad():
            teacher1_logits = teacher1(**batch).logits
            teacher2_logits = teacher2(**batch).logits
            soft_labels1 = softmax(teacher1_logits, dim=-1)
            soft_labels2 = softmax(teacher2_logits, dim=-1)
            confidence_mask1 = generate_confidence_mask(soft_labels1)
            confidence_mask2 = generate_confidence_mask(soft_labels2)

        # Update students using soft labels with confidence masking
        student1_loss = soft_label_cross_entropy(student1(**batch).logits, soft_labels2, labels, confidence_mask2)
        student2_loss = soft_label_cross_entropy(student2(**batch).logits, soft_labels1, labels, confidence_mask1)
        
        optimizer_s1.zero_grad()
        student1_loss.backward()
        optimizer_s1.step()

        optimizer_s2.zero_grad()
        student2_loss.backward()
        optimizer_s2.step()

        # Apply EMA to teacher models periodically
        if (i + 1) % EMA_UPDATE_PERIOD == 0:
            apply_ema(teacher1, student1)
            apply_ema(teacher2, student2)
    
        predictions1 = torch.argmax(student1(**batch).logits, dim=-1)
        correct_predictions1 += (predictions1 == batch['labels']).sum().item()
        total_predictions1 += batch['labels'].numel()

        predictions2 = torch.argmax(student2(**batch).logits, dim=-1)
        correct_predictions2 += (predictions2 == batch['labels']).sum().item()
        total_predictions2 += batch['labels'].numel()

        # Logging
        if i % 5 == 0:
            logging.info(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, Train1 Loss: {student1_loss.item():.4f}, Train2 Loss: {student2_loss.item():.4f}, Accuracy1: {correct_predictions1/total_predictions1:.4f}, Accuracy2: {correct_predictions2/total_predictions2:.4f}')
            writer.add_scalar('Train/Student1_Loss', student1_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Train/Student2_Loss', student2_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Train/Student1_Accuracy', correct_predictions1/total_predictions1, epoch * len(train_loader) + i)
            writer.add_scalar('Train/Student2_Accuracy', correct_predictions2/total_predictions2, epoch * len(train_loader) + i)

    # Validation step
    eval_st1_loss, eval_st1_accuracy = evaluate_model(student1, validation_loader, device)
    eval_st2_loss, eval_st2_accuracy = evaluate_model(student2, validation_loader, device)
    writer.add_scalar('Validation/Student1_Loss', eval_st1_loss, epoch)
    writer.add_scalar('Validation/Student2_Loss', eval_st2_loss, epoch)
    writer.add_scalar('Validation/Student1_Accuracy', eval_st1_accuracy, epoch)
    writer.add_scalar('Validation/Student2_Accuracy', eval_st2_accuracy, epoch)
    logging.info(f'Epoch {epoch+1}/{NUM_EPOCHS}, St1 [Validation Loss: {eval_st1_loss:.4f}, Accuracy: {eval_st1_accuracy:.3f}] | St2 [Validation Loss: {eval_st2_loss:.4f}, Accuracy: {eval_st2_accuracy:.3f}]')

    # Early stopping
    if eval_st1_accuracy > best_accuracy or eval_st2_accuracy > best_accuracy:
        best_accuracy = max(eval_st1_accuracy, eval_st2_accuracy)
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        logging.info("Early stopping. Patience limit reached.")
        break

# Test step
test_st1_loss, test_st1_accuracy = evaluate_model(student1, test_loader, device)
test_st2_loss, test_st2_accuracy = evaluate_model(student2, test_loader, device)
logging.info(f'Test - St1 [Loss: {test_st1_loss:.4f}, Accuracy: {test_st1_accuracy:.3f}] | St2 [Loss: {test_st2_loss:.4f}, Accuracy: {test_st2_accuracy:.3f}]')

# Save the models
torch.save(student1.state_dict(), "student1_model.pt")
torch.save(student2.state_dict(), "student2_model.pt")
torch.save(teacher1.state_dict(), "teacher1_model.pt")
torch.save(teacher2.state_dict(), "teacher2_model.pt")

# Close tensorboard writer
writer.close()
