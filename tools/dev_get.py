import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码

# 读取train.tsv文件
df = pd.read_csv('train.txt', sep='\t', header=None, names=['text', 'label'])

# 统计每个标签的总数
label_counts = df['label'].value_counts()

# 随机抽取20%的数据作为评估数据集
dev_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.2))

# 将剩余的数据作为训练数据集
train_df = df.drop(dev_df.index)

# 保存评估数据集到dev.tsv
dev_df.to_csv('dev.tsv', sep='\t', index=False, header=False)

# 保存训练数据集到train.tsv（如果需要）
train_df.to_csv('train.tsv', sep='\t', index=False, header=False)

# 打印每个标签的总数和抽取的数量
print("标签总数:")
print(label_counts)
print("\n评估数据集中的标签数量:")
print(dev_df['label'].value_counts())