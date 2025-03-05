import pandas as pd
import torch
import random
import os
import numpy as np
from torch.optim import Adam
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# 本地数据地址
data_path = 'data/dice/train.tsv'  # 使用完整数据集
label_path = 'data/dice/class.tsv'

# 读取数据
df = pd.read_csv(data_path, sep='\t', header=None)

# 更改列名
new_columns = ['text', 'label']
df = df.rename(columns=dict(zip(df.columns, new_columns)))

# 读取标签
real_labels = []
with open(label_path, 'r') as f:
    for row in f.readlines():
        real_labels.append(row.strip())

print("read data done!")

# 下载的预训练文件路径
BERT_PATH = 'base/base-chinese'

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

print("tokenizer done!")

print("loading dataset...")

class MyDataset(Dataset):
    def __init__(self, df):
        # tokenizer分词后可以被自动汇聚
        self.texts = [tokenizer(text, 
                                padding='max_length',  # 填充到最大长度
                                max_length=35,  # 经过数据分析，最大长度为35
                                truncation=True,
                                return_tensors="pt") 
                      for text in df['text']]
        # Dataset会自动返回Tensor
        self.labels = [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

# 因为要进行分词，此段运行较久，约40s
dataset = MyDataset(df)

# 定义模型
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

print("load dataset done!")

# 训练超参数
epochs = 20
batch_size = 64
lr = 1e-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 1999
save_path = './bert_checkpoint'
bootstrap_samples = 5  # 自助法采样次数

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(random_seed)

def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))

# 自助法采样
def bootstrap_sample(dataset, n_samples):
    indices = np.random.choice(len(dataset), size=n_samples, replace=True)
    return indices

# 开始自助法训练
for sample_idx in range(bootstrap_samples):
    print(f"Bootstrap Sample {sample_idx + 1}/{bootstrap_samples}")
    
    # 生成自助法采样索引
    train_indices = bootstrap_sample(dataset, len(dataset))
    val_indices = list(set(range(len(dataset))) - set(train_indices))  # 未被采样的样本作为验证集
    
    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # 创建DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    
    # 定义模型、损失函数和优化器
    model = BertClassifier().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    # 训练
    best_val_acc = 0
    for epoch_num in range(epochs):
        model.train()
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([batch_size, 35])
            masks = inputs['attention_mask'].to(device)  # torch.Size([batch_size, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)
            
            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += batch_loss.item()
        
        # 验证
        model.eval()
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                input_ids = inputs['input_ids'].squeeze(1).to(device)
                masks = inputs['attention_mask'].to(device)
                labels = labels.to(device)
                output = model(input_ids, masks)
                
                batch_loss = criterion(output, labels)
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc
                total_loss_val += batch_loss.item()
            
            print(f'''Epoch {epoch_num + 1}/{epochs} 
                  | Train Loss: {total_loss_train / len(train_indices): .3f} 
                  | Train Accuracy: {total_acc_train / len(train_indices): .3f} 
                  | Val Loss: {total_loss_val / len(val_indices): .3f} 
                  | Val Accuracy: {total_acc_val / len(val_indices): .3f}''')
            
            # 保存最优模型
            if total_acc_val / len(val_indices) > best_val_acc:
                best_val_acc = total_acc_val / len(val_indices)
                save_model(f'best_sample{sample_idx + 1}.pt')
    
    # 保存最后一个epoch的模型
    save_model(f'last_sample{sample_idx + 1}.pt')