import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, DistilBertForTokenClassification
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn import KLDivLoss
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Setup logging and TensorBoard writer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
writer = SummaryWriter()

# Constants
NUM_LABELS = dataset['train'].features['ner_tags'].feature.num_classes
NUM_EPOCHS = 3
BATCH_SIZE = 8
ALPHA = 0.99
EMA_UPDATE_PERIOD = 10

# Load and preprocess the dataset
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
dataset = load_dataset("conll2003")
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize models
teacher1 = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
teacher2 = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
student1 = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS)
student2 = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS)

# Prepare Dataset & Loaders
train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE)
test_loader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE)

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

# Dynamic confidence threshold function
def adjust_confidence_threshold(validation_loader, model, percentile=75):
    softmax_outputs = []
    model.eval()
    with torch.no_grad():
        for batch in validation_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch).logits
            softmax_outputs.extend(torch.softmax(outputs, dim=-1).max(dim=-1)[0].cpu().numpy())
    return np.percentile(softmax_outputs, percentile)

# Enhanced loss function with denoising
def enhanced_loss_function(outputs, labels, soft_labels, threshold):
    hard_loss = F.cross_entropy(outputs, labels, ignore_index=-100)
    soft_loss = F.kl_div(F.log_softmax(outputs, dim=-1), soft_labels, reduction='batchmean')
    confidence_mask = (torch.max(soft_labels, dim=-1)[0] > threshold).float()
    return (confidence_mask * soft_loss + (1 - confidence_mask) * hard_loss).mean()

def evaluate_model(model, dataloader, device):
    model.eval()
    total_eval_loss = 0
    total_correct = 0
    total_examples = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, NUM_LABELS), batch['labels'].view(-1), ignore_index=-100)
            total_eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_labels = batch['labels'] != -100
            total_correct += (predictions == batch['labels']).sum().item()
            total_examples += correct_labels.sum().item()
    accuracy = total_correct / total_examples if total_examples > 0 else 0
    return total_eval_loss / len(dataloader), accuracy

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
