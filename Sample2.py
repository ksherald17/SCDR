import torch
from transformers import BertTokenizerFast, BertForTokenClassification, DistilBertForTokenClassification
from transformers import AdamW, DataCollatorForTokenClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

def main():
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir='./tensorboard_logs')

    # Load tokenizer and dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = load_dataset("conll2003")
    num_labels = dataset['train'].features['ner_tags'].feature.num_classes

    example_data = dataset["train"][0]
    writer.add_text("Sample Data", f"Tokens: {example_data['tokens']}\nTags: {example_data['ner_tags']}")

        # Sampling a subset of the data for quicker iterations
    def sample_dataset(dataset, sample_size=0.1):
        """ Randomly sample sample_size proportion of dataset for each split """
        sampled = {}
        for split in dataset.keys():
            sampled_split = dataset[split].shuffle(seed=42).select(range(int(len(dataset[split]) * sample_size)))
            sampled[split] = sampled_split
        return sampled

    dataset = sample_dataset(dataset, sample_size=0.1)  # Sample 10% of each split


    # Function to tokenize and align labels for NER
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding="max_length", is_split_into_words=True, return_token_type_ids=False)
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Tokenize and prepare all data splits
    dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "pos_tags", "chunk_tags", "id", "ner_tags"])
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]


    # Load teacher and student models
    teacher_model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    student_model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    teacher_model.eval()
    student_model.train()

    # Data collator and DataLoader setup
    data_collator = DataCollatorForTokenClassification(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

    # Optimizer
    optimizer = AdamW(student_model.parameters(), lr=5e-5)
    temperature = 2.0
    best_accuracy = 0

    # Training loop with evaluation and checkpointing
    epochs = 1
    for epoch in range(epochs):
        total_train_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(student_model.device) for k, v in batch.items() if k != 'token_type_ids'}
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher_model(**batch).logits

            student_output = student_model(**batch)
            student_loss = student_output.loss

            # Calculate distillation loss
            soft_teacher_labels = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
            soft_student_logits = torch.nn.functional.log_softmax(student_output.logits / temperature, dim=-1)
            distillation_loss = torch.nn.KLDivLoss(reduction='batchmean')(soft_student_logits, soft_teacher_labels)

            # Combine losses
            loss = student_loss + 0.5 * distillation_loss
            loss.backward()
            optimizer.step()

            # Calculate predictions for accuracy
            logits = student_output.logits.detach()
            predictions = torch.argmax(logits, dim=-1)
            labels = batch['labels']
            mask = labels != -100 # Compute accuracy, considering -100 labels that should be ignored
            correct_predictions += (predictions[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()

            total_train_loss += loss.item()
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)

            # Logging
            if i % 5 == 0:
                current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}, Accuracy: {current_accuracy:.4f}')
                # Optionally, reset for more fine-grained batch accuracy rather than cumulative
                correct_predictions = 0
                total_predictions = 0

        # Validation phase
        student_model.eval()
        eval_accuracy, eval_loss = evaluate_model(student_model, eval_loader, device=student_model.device)
        writer.add_scalar('Validation Loss', eval_loss, epoch)
        writer.add_scalar('Validation Accuracy', eval_accuracy, epoch)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.3f}')

        # Checkpointing based on validation accuracy
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            torch.save(student_model.state_dict(), 'best_student_model.pt')
            print(f"Checkpoint saved: Improved validation accuracy to {best_accuracy:.3f}")

    # Evaluate on test dataset
    student_model.load_state_dict(torch.load('best_student_model.pt'))
    test_accuracy, test_loss = evaluate_model(student_model, test_loader, device=student_model.device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.3f}')

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

if __name__ == "__main__":
    main()
