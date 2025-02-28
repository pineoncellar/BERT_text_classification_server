import re

# 定义称呼列表
names = ["七海千秋", "七海", "千秋", "nanami", "娜娜米"]

# 读取数据集
with open('train.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 用于存储处理后的数据
unique_lines = []

# 用于记录已经处理过的句子模式
processed_patterns = set()

# 遍历每一行数据
for line in lines:
    line = line.strip()
    # 替换称呼为占位符，生成句子模式
    pattern = line
    for name in names:
        pattern = re.sub(re.escape(name), '<NAME>', pattern)
    
    # 如果这个句子模式已经处理过，跳过
    if pattern in processed_patterns:
        continue
    
    # 记录这个句子模式
    processed_patterns.add(pattern)
    
    # 将原始句子添加到唯一列表中
    unique_lines.append(line)

# 将处理后的数据写回文件
with open('dataset_cleaned.txt', 'w', encoding='utf-8') as file:
    for line in unique_lines:
        file.write(line + '\n')

print("数据去重完成，结果已保存到 dataset_cleaned.txt")