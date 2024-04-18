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
## batch_size: 2개를 집어넣어서 Loss를 업데이트 할 때 2개를 같이 업데이트 하겠다라는 내용. 각각 데이터들이 어떻게 나오는지 까보자. (1)

# Bert 모델 from Hugging Face (모델에 관한 정보는 PPT 참조)
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
# tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') # 토큰 단위 Id => id 변환 => 모델로 변환되는 과정 확인해보기 (2)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
## model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
model.to(device)

## (3) 모델을 저장할 수 있고, 저장하는 것을 체크포인트라고 함. 제일 좋은 모델 5개
## overfitting이 training data에 대한 답을 외우는 것 패턴을 배우는 것이 아니고 암기하는 수준을 방지 training accuracy는 올라가지만 validation accuracy는 내려감 (올라가다가 떨어짐) 오버피팅 감지하기 위한 수단으로 Validation 둠
## Validation Accuracy를 보기 위해 모델을 저장해야 함. Validation Accuracy가 가장 높았던 지점을 저장해놓고, 오버피팅이 일어나면 더 이상 트레이닝이 의미 없기 때문에 스탑하고 그 시점을 써야 함.
## Generalization 패턴을 배우는 용어 잘 되는 지점 Accuracy가 Validation 기준으로 최고점일 때 
 
# Parameter 선정
optimizer = Adam(model.parameters(), lr=1e-5) # SCDR에 제시된 learning rate
itr = 1
p_itr = 500
epochs = 1
total_loss = 0
total_len = 0
total_correct = 0

# Bert 모델을 토대로 학습하는 과정
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
            print('[Epoch {}/{}] Ite성ation {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, total_correct/total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0

        itr+=1

    ## (4)
    ## epoch이 추가적으로 해봐야하는 부분인데, epoch이 적으면 트레이닝이 안됐는데 스탑 될 가능성 있음. 에폭이 드으면 오버피팅 가능성
    ## Test말고 Validation Data가 따로 있어야 함. (모델이 얼마나 됐는지 별도의 기이터) Validation이 높아지지 않으면 오버피팅 발생하면 스탑하는 코드 
    ## 10 에포크 정도 두고, 올라가지 않으면 스탑 코드 넣기, 그리고 가장 좋은 모델에 대해 테스트 돌리는 코드 넣기

    ## training data / test data 말고 validation data 
    

    ## (5)
    ## 그래프 찍기 탠서보드로 보여주기 . 텐서보드 런하는 방법 확인하기 조금 어려울 수 있음.

    ## (6) Optimizaer -> 교수님 질문 가능성 있음
    ## 모델이 로스 계산하는 방식  / 배치와 같이 전반적으로 알아둡시댜. 입력이 어떻게 들어가는지 정확히 알자.

    ## 배치를 쓰는 이유가 원래는 전체 데이터에 대한 로스를 계산하고 한번에 업데이트 하는데, 메모리 부족으로 넣기 쉽지 않음. 오래걸림. 배치가 전체를 대변하는 그룹. 배치 단위로 넣는 것이 간략한 그림
    ## 수학적으로 전체 트레이닝에 대체적으로 수렴한다고 알려져 있으니 확인해보자.

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