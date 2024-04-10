import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, TFBertModel
from torch.optim import Adam
import torch.nn.functional as Func
from ExternalDataset import NaverDataset

# 네이버 리뷰 데이터 셋
train_data = pd.read_csv("./nsmc/ratings_train.txt", sep='\t')
test_data = pd.read_csv("./nsmc/ratings_test.txt", sep='\t')
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# 데이터 샘플링 비율 
sampling_rate=0.4
train_sampled = train_data.sample(frac=sampling_rate, random_state=999)
test_sampled = test_data.sample(frac=sampling_rate, random_state=999)

naver_train_dataset = NaverDataset(train_sampled)
train_loader = DataLoader(naver_train_dataset, batch_size=2, shuffle=True)

# RoBerta 모델 from Hugging Face (모델에 관한 정보는 PPT 참조)
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base')
model.to(device)

# Parameter 선정
optimizer = Adam(model.parameters(), lr=1e-5) # SCDR에 제시된 learning rate
itr = 1
p_itr = 500
epochs = 1
total_loss = 0
total_len = 0
total_correct = 0

# Roberta 모델을 토대로 학습하는 과정
model.train()
for epoch in range(epochs):
    
    for text, label in train_loader:
        optimizer.zero_grad()

        encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded, label = encoded.to(device), label.to(device)
        outputs = model(**encoded, labels=label)
        
        loss = outputs.loss
        logits = outputs.logits
        
        pred = torch.argmax(Func.softmax(logits), dim=1)
        correct = pred.eq(label)
        total_correct += correct.sum().item()
        total_len += len(label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if itr % p_itr == 0:
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, total_correct/total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0

        itr+=1

# evaluation
model.eval()

nsmc_eval_dataset = NaverDataset(test_sampled)
eval_loader = DataLoader(nsmc_eval_dataset, batch_size=8, shuffle=False)

total_loss = 0
total_len = 0
total_correct = 0

for text, label in eval_loader:
    
    encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded, label = encoded.to(device), label.to(device)
    outputs = model(**encoded, labels=label)
    
    logits = outputs.logits

    pred = torch.argmax(Func.softmax(logits), dim=1)
    correct = pred.eq(label)
    total_correct += correct.sum().item()
    total_len += len(label)

print('Test accuracy: ', total_correct / total_len)