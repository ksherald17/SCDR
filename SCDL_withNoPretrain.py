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
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import precision_score, recall_score, f1_score
import heapq
from torch.cuda.amp import GradScaler, autocast

# Environment setup
torch.autograd.set_detect_anomaly(True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
NUM_EPOCHS = 1
BATCH_SIZE = 128
ALPHA = 0.99
EMA_UPDATE_PERIOD = 10
WARMUP_EPOCHS = 1
NUM_PRETRAIN_EPOCHS = 1
ACCUMULATION_STEPS = 4  # Number of steps to accumulate gradients

# Initialize a heap to store the top models with their performance
best_models_heap = []

# Load and preprocess the dataset
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
dataset = load_dataset("conll2003")
NUM_LABELS = dataset['train'].features['ner_tags'].feature.num_classes

checkpoint_dir = './checkpoints/scdr'

def random_masking(inputs, labels, tokenizer, mask_probability=0.15):
    """ Randomly masks tokens with a given probability. """
    rand = torch.rand(inputs.shape)
    mask_arr = (rand < mask_probability) & (inputs != tokenizer.pad_token_id) & (inputs != tokenizer.sep_token_id) & (inputs != tokenizer.cls_token_id)
    inputs[mask_arr] = tokenizer.mask_token_id
    labels[~mask_arr] = -100  # We only compute loss on masked tokens
    return inputs, labels

def tokenize_and_align_random_mask_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding="max_length", is_split_into_words=True, return_token_type_ids=False)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their word IDs
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)
    
    input_ids = torch.tensor(tokenized_inputs['input_ids'])
    labels = torch.tensor(labels)
    # Apply random masking
    input_ids, labels = random_masking(input_ids, labels, tokenizer)

    return {'input_ids': input_ids, 'attention_mask': tokenized_inputs['attention_mask'], 'labels': labels}

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding="max_length", is_split_into_words=True, return_token_type_ids=False)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def warmup_lr_scheduler(optimizer, total_steps, warmup_steps, initial_lr):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def sample_dataset(dataset, sample_size=0.5):
    return dataset.shuffle(seed=42).select(range(int(len(dataset) * sample_size)))

tokenized_datasets = dataset.map(tokenize_and_align_random_mask_labels, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_dataset = sample_dataset(tokenized_datasets["train"])
val_dataset = sample_dataset(tokenized_datasets["validation"])
test_dataset = sample_dataset(tokenized_datasets["test"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

teacher1 = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
teacher2 = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
student1 = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS)
student2 = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher1.to(device)
teacher2.to(device)
student1.to(device)
student2.to(device)

optimizer_s1 = AdamW(student1.parameters(), lr=5e-5)
optimizer_s2 = AdamW(student2.parameters(), lr=5e-5)
scheduler_s1 = warmup_lr_scheduler(optimizer_s1, len(train_loader)*NUM_EPOCHS, len(train_loader)*WARMUP_EPOCHS, 5e-5)
scheduler_s2 = warmup_lr_scheduler(optimizer_s2, len(train_loader)*NUM_EPOCHS, len(train_loader)*WARMUP_EPOCHS, 5e-5)

optimizer_t1 = AdamW(teacher1.parameters(), lr=5e-5)
optimizer_t2 = AdamW(teacher2.parameters(), lr=5e-5)

# Pre-training Teacher Models
for epoch in range(NUM_PRETRAIN_EPOCHS):
    teacher1.train()
    teacher2.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs1 = teacher1(**batch)
        loss1 = outputs1.loss
        loss1.backward()
        optimizer_t1.step()
        optimizer_t1.zero_grad()

        outputs2 = teacher2(**batch)
        loss2 = outputs2.loss
        loss2.backward()
        optimizer_t2.step()
        optimizer_t2.zero_grad()

# Save the pre-trained teacher models
torch.save(teacher1.state_dict(), os.path.join(checkpoint_dir, 'teacher1_pretrained.pth'))
torch.save(teacher2.state_dict(), os.path.join(checkpoint_dir, 'teacher2_pretrained.pth'))

torch.cuda.empty_cache()  # Use it judiciously, as too frequent cache clearing can lead to performance degradation.

kl_div_loss = KLDivLoss(reduction='batchmean')

best_test_loss = float('inf')
best_model_path = None

def evaluate_model(model, dataloader, device):
    model.eval()
    total_eval_loss = 0
    all_predictions = []
    all_true_labels = []
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
            logits = output.logits
            loss = F.cross_entropy(logits.view(-1, NUM_LABELS), batch['labels'].view(-1), ignore_index=-100)
            total_eval_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1).view(-1)
            true_labels = batch['labels'].view(-1)
            valid_indices = true_labels != -100  # Exclude padding
            
            valid_predictions = predictions[valid_indices]
            valid_true_labels = true_labels[valid_indices]
            
            all_predictions.extend(valid_predictions.cpu().numpy())
            all_true_labels.extend(valid_true_labels.cpu().numpy())

    # Calculate precision, recall, and F1-score considering potential class imbalance
    precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)

    return total_eval_loss / len(dataloader), precision, recall, f1

def calculate_batch_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    mask = labels != -100
    correct_predictions = (predictions == labels) & mask
    total_correct = correct_predictions.sum().item()
    total = mask.sum().item()
    return (total_correct / total) * 100 if total > 0 else 0

def apply_ema(teacher, student, alpha):
    with torch.no_grad():
        teacher_params = {name: param for name, param in teacher.named_parameters()}
        student_params = {name: param for name, param in student.named_parameters()}
        
        for name, teacher_param in teacher_params.items():
            if name in student_params:
                student_param = student_params[name]
                if teacher_param.data.shape == student_param.data.shape:
                    teacher_param.data.copy_(alpha * teacher_param.data + (1 - alpha) * student_param.data)
                else:
                    logging.warning(f"Skipping EMA for {name} due to shape mismatch: {teacher_param.data.shape} vs {student_param.data.shape}")

def adjust_confidence_threshold(dataloader, model, device, percentile=75):
    all_confidences = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs).logits
            confidences = torch.softmax(outputs, dim=-1).max(dim=-1)[0]
            all_confidences.append(confidences)
    
    all_confidences = torch.cat(all_confidences)
    threshold = np.percentile(all_confidences.cpu().numpy(), percentile)
    return threshold

def enhanced_loss_function(outputs, labels, soft_labels, threshold):
    outputs = outputs.view(-1, outputs.size(-1))
    labels = labels.view(-1)
    
    hard_loss = F.cross_entropy(outputs, labels, ignore_index=-100)
    soft_labels = F.log_softmax(outputs, dim=-1)
    soft_loss = F.kl_div(soft_labels, F.softmax(soft_labels, dim=-1), reduction='batchmean')

    confidence_mask = (torch.max(soft_labels, dim=-1)[0] > threshold).float()
    combined_loss = (confidence_mask * soft_loss + (1 - confidence_mask) * hard_loss).mean()
    
    return combined_loss

def apply_mask_to_inputs(inputs, logits, threshold):
    confidences = torch.max(F.softmax(logits, dim=-1), dim=-1)[0]
    mask = confidences < threshold
    masked_inputs = inputs.clone()
    masked_inputs[mask] = tokenizer.mask_token_id
    return masked_inputs

def save_checkpoint(model, performance, epoch, max_checkpoints=5):
    heapq.heappush(best_models_heap, (performance, f"student_model_epoch_{epoch+1}.pt"))
    
    if len(best_models_heap) > max_checkpoints:
        _, oldest_model_path = heapq.heappop(best_models_heap)
        os.remove(oldest_model_path)

    torch.save(model.state_dict(), best_models_heap[-1][1])

# Training loop
scaler = GradScaler()  # For mixed precision training

best_val_accuracy = 0.0
early_stopping_counter = 0
early_stopping_rounds = 5

for epoch in range(NUM_EPOCHS):
    student1.train()
    student2.train()
    for i, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']
        
        with torch.no_grad():
            teacher1_logits = teacher1(**batch).logits
            teacher2_logits = teacher2(**batch).logits
            soft_labels1 = F.softmax(teacher1_logits, dim=-1)
            soft_labels2 = F.softmax(teacher2_logits, dim=-1)
            threshold1 = adjust_confidence_threshold(validation_loader, teacher1, device)
            threshold2 = adjust_confidence_threshold(validation_loader, teacher2, device)
            
            masked_inputs1 = apply_mask_to_inputs(batch['input_ids'], teacher1_logits, threshold1)
            masked_inputs2 = apply_mask_to_inputs(batch['input_ids'], teacher2_logits, threshold2)
            
            batch['input_ids'] = masked_inputs1 if i % 2 == 0 else masked_inputs2

        optimizer_s1.zero_grad()
        optimizer_s2.zero_grad()

        with autocast():
            student1_loss = enhanced_loss_function(student1(**batch).logits, labels, soft_labels2, threshold2)
            student2_loss = enhanced_loss_function(student2(**batch).logits, labels, soft_labels1, threshold1)

        scaler.scale(student1_loss).backward()
        scaler.scale(student2_loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer_s1)
            scaler.step(optimizer_s2)
            scaler.update()
            optimizer_s1.zero_grad()
            optimizer_s2.zero_grad()

        scheduler_s1.step()
        scheduler_s2.step()

        batch_accuracy1 = calculate_batch_accuracy(student1(**batch).logits, labels)
        batch_accuracy2 = calculate_batch_accuracy(student2(**batch).logits, labels)

        if (i + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Student1 Accuracy: {batch_accuracy1:.2f}%, Student2 Accuracy: {batch_accuracy2:.2f}%")

        if (i + 1) % EMA_UPDATE_PERIOD == 0:
            apply_ema(teacher1, student1, ALPHA)
            apply_ema(teacher2, student2, ALPHA)

    eval_loss, precision, recall, f1 = evaluate_model(student1, validation_loader, device)
    logging.info(f'Epoch {epoch+1}, Validation Loss: {eval_loss:.4f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}')

    # Save checkpoints for both models
    if eval_loss < best_test_loss:
        best_test_loss = eval_loss
        torch.save(student1.state_dict(), f"{checkpoint_dir}/student1_best.pt")
        torch.save(student2.state_dict(), f"{checkpoint_dir}/student2_best.pt")
        best_val_accuracy = max(eval_loss, best_val_accuracy)  # Update best_val_accuracy with the lowest loss

    if eval_loss < best_test_loss:
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_rounds:
            logging.info("Early stopping triggered!")
            break

# Evaluate on test set
test_st1_metrics = evaluate_model(student1, test_loader, device)
test_st1_loss, test_st1_precision, test_st1_recall, test_st1_f1 = test_st1_metrics
test_st2_metrics = evaluate_model(student2, test_loader, device)
test_st2_loss, test_st2_precision, test_st2_recall, test_st2_f1 = test_st2_metrics

logging.info(f'St1 [Test Loss: {test_st1_loss:.4f}, Precision: {test_st1_precision:.3f}, Recall: {test_st1_recall:.3f}, F1-Score: {test_st1_f1:.3f}] | '
             f'St2 [Test Loss: {test_st2_loss:.4f}, Precision: {test_st2_precision:.3f}, Recall: {test_st2_recall:.3f}, F1-Score: {test_st2_f1:.3f}]')

# Compare test losses to determine the best model
if test_st1_loss < test_st2_loss:
    best_model_path = f"{checkpoint_dir}/student1_best.pt"
    print(f"Student 1 is the best model based on testing data. - Loss: {test_st1_loss:.4f}")
else:
    best_model_path = f"{checkpoint_dir}/student2_best.pt"
    print(f"Student 2 is the best model based on testing data. - Loss: {test_st2_loss:.4f}")

# Optionally, load and use the best model
best_model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)

