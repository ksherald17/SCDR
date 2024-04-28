import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, DistilBertForTokenClassification
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn.functional import cross_entropy, softmax, log_softmax
from torch.nn import KLDivLoss
import logging
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

torch.cuda.empty_cache()

# Constants
ALPHA = 0.99  # EMA coefficient
NUM_EPOCHS = 3  # Number of training epochs
BATCH_SIZE = 8  # Batch size for training
EMA_UPDATE_PERIOD = 10 # Apply EMA updates every 10 batchs

# Load and preprocess the dataset
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
dataset = load_dataset("conll2003")
NUM_LABELS = dataset['train'].features['ner_tags'].feature.num_classes

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
    labels = [[label for label in doc] for doc in examples["ner_tags"]]
    new_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        new_labels.append(label_ids)
    tokenized_inputs["labels"] = new_labels
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

sampling = 0.01
train_dataset = sample_dataset(tokenized_datasets["train"], sample_size=sampling) # Sample 10% of each split
eval_dataset = sample_dataset(tokenized_datasets["validation"], sample_size=sampling)
test_dataset = sample_dataset(tokenized_datasets["test"], sample_size=sampling)

# Prepare data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
validation_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)

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

# Function to apply EMA
def apply_ema(teacher, student, alpha=ALPHA):
    with torch.no_grad():
        teacher_params = dict(teacher.named_parameters())
        student_params = dict(student.named_parameters())
        for name, param in teacher_params.items():
            if name in student_params:
                param.data.lerp_(student_params[name].data, 1 - alpha)

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
            loss = output.loss
            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)
            labels = batch['labels']
            mask = labels != -100
            labels = labels[mask]
            predictions = predictions[mask]
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)
            total_eval_loss += loss.item()

    accuracy = total_correct / total_examples if total_examples > 0 else 0
    return accuracy, total_eval_loss / len(dataloader)
# Training loop
for epoch in range(NUM_EPOCHS):
    student1.train()
    student2.train()
    for i, batch in enumerate(train_loader): 
        # Forward pass
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']

        # Teacher predictions
        with torch.no_grad():
            teacher1_pred = teacher1(**batch).logits
            teacher2_pred = teacher2(**batch).logits

        # Student predictions
        student1_pred = student1(**batch).logits
        student2_pred = student2(**batch).logits

        # Compute loss
        student1_loss = (kl_div_loss(log_softmax(student1_pred, dim=-1), softmax(teacher2_pred.detach(), dim=-1)) +
                         cross_entropy(student1_pred.view(-1, NUM_LABELS), labels.view(-1))).mean()
        student2_loss = (kl_div_loss(log_softmax(student2_pred, dim=-1), softmax(teacher1_pred.detach(), dim=-1)) +
                         cross_entropy(student2_pred.view(-1, NUM_LABELS), labels.view(-1))).mean()

        # Backpropagation
        optimizer_s1.zero_grad()
        student1_loss.backward()
        optimizer_s1.step()

        optimizer_s2.zero_grad()
        student2_loss.backward()
        optimizer_s2.step()

        # Apply EMA to teacher models
        if (i + 1) % EMA_UPDATE_PERIOD == 0:
            apply_ema(teacher1, student1)
            apply_ema(teacher2, student2)

    # Validation step
    eval_st1_accuracy, eval_st1_loss = evaluate_model(student1, validation_loader)
    eval_st2_accuracy, eval_st2_loss = evaluate_model(student2, validation_loader) 
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, St1 [Validation Loss: {eval_st1_loss:.4f}, Accuracy: {eval_st1_accuracy:.3f}] | St2 [Validation Loss: {eval_st2_loss:.4f}, Accuracy: {eval_st2_accuracy:.3f}]')

# Save the models
torch.save(student1.state_dict(), "student1_model.pt")
torch.save(student2.state_dict(), "student2_model.pt")
torch.save(teacher1.state_dict(), "teacher1_model.pt")
torch.save(teacher2.state_dict(), "teacher2_model.pt")

