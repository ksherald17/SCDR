from torch.utils.data import Dataset, DataLoader

# 데이터셋 클래스 
class NaverDataset(Dataset):
    '''Naver Sentiment Movie Corpus Dataset'''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label
