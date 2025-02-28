import pandas as pd
import torch
import random
import os
import numpy as np
from torch.optim import Adam
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from transformers import BertTokenizer
from tqdm import tqdm


# 本地数据地址
train_data_path = 'data/dice/train.txt'
dev_data_path = 'data/dice/dev.txt'
label_path = 'data/dice/class.txt'

# 读取数据
train_df = pd.read_csv(train_data_path, sep='\t', header=None)
dev_df = pd.read_csv(dev_data_path, sep='\t', header=None)

# 更改列名
new_columns = ['text', 'label']  
train_df = train_df.rename(columns=dict(zip(train_df.columns, new_columns)))
dev_df = dev_df.rename(columns=dict(zip(dev_df.columns, new_columns)))

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
                                max_length = 35, 	# 经过数据分析，最大长度为35
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
train_dataset = MyDataset(train_df)
dev_dataset = MyDataset(dev_df)

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
epoch = 20
batch_size = 64
lr = 1e-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 1999
save_path = './bert_checkpoint'

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

    
# 定义模型
model = BertClassifier()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
model = model.to(device)
criterion = criterion.to(device)

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)


# 训练
best_dev_acc = 0
for epoch_num in range(epoch):
    total_acc_train = 0
    total_loss_train = 0
    for inputs, labels in tqdm(train_loader):
        input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
        masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
        labels = labels.to(device)
        output = model(input_ids, masks)

        batch_loss = criterion(output, labels)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = (output.argmax(dim=1) == labels).sum().item()
        total_acc_train += acc
        total_loss_train += batch_loss.item()

    # ----------- 验证模型 -----------
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    # 不需要计算梯度
    with torch.no_grad():
        # 循环获取数据集，并用训练好的模型进行验证
        for inputs, labels in dev_loader:
            input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc
            total_loss_val += batch_loss.item()
        
        print(f'''Epochs: {epoch_num + 1} 
          | Train Loss: {total_loss_train / len(train_dataset): .3f} 
          | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
          | Val Loss: {total_loss_val / len(dev_dataset): .3f} 
          | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}''')
        
        # 保存最优的模型
        if total_acc_val / len(dev_dataset) > best_dev_acc:
            best_dev_acc = total_acc_val / len(dev_dataset)
            save_model('best.pt')
        
    model.train()

# 保存最后的模型，以便继续训练
save_model('last.pt')
# todo 保存优化器
