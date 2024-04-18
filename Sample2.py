import torch
from transformers import BertTokenizerFast, BertForTokenClassification, DistilBertForTokenClassification
from transformers import AdamW, DataCollatorForTokenClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def main():
    # Load fast tokenizer and dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = load_dataset("conll2003")

    # Extract the number of unique NER labels
    num_labels = dataset['train'].features['ner_tags'].feature.num_classes

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

    # Tokenize all data splits and remove unneeded columns
    dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "pos_tags", "chunk_tags", "id", "ner_tags"])

    # Load teacher and student models
    teacher_model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    student_model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    teacher_model.eval()
    student_model.train()

    # Data collator that dynamically pads the inputs and labels
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # DataLoader setup
    train_loader = DataLoader(dataset['train'], batch_size=8, collate_fn=data_collator)

    # Optimizer setup
    optimizer = AdamW(student_model.parameters(), lr=5e-5)
    temperature = 2.0

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_len = 0

        for itr, batch in enumerate(train_loader):
            batch = {k: v.to(student_model.device) for k, v in batch.items() if k != 'token_type_ids'}
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
            optimizer.zero_grad()

            total_loss += loss.item()

            # Convert logits to predicted labels
            predictions = torch.argmax(soft_student_logits, dim=-1).flatten()
            labels = batch['labels'].flatten()

            # Filter out `-100` values from the labels and predictions
            mask = labels != -100
            labels = labels[mask]
            predictions = predictions[mask]

            total_correct += (predictions == labels).sum().item()
            total_len += labels.size(0)

            print(f'[Epoch {epoch+1}/{epochs}] Iteration {itr+1} -> Train Loss: {total_loss/(itr+1):.4f}, Accuracy: {total_correct/total_len:.3f}')

        print(f'End of Epoch {epoch+1}/{epochs} -> Average Loss: {total_loss/(itr+1):.4f}, Accuracy: {total_correct/total_len:.3f}')

if __name__ == "__main__":
    main()
