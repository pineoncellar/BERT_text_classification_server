import re
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码

def process_log_file(input_file, output_file):
    # 修复后的正则表达式
    pattern = re.compile(
        r'的消息，内容：(.*)，分类为：(related|unrelated)'
    )

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            
            line = re.sub(r'\[CQ:reply,id=\d+\]', '', line)
            line = re.sub(r'\[CQ:face,id=\d+\]', '', line)
            line = re.sub(r'\[CQ:at,id=\d+\]', '', line)

            # 跳过空行和生成回复的行
            if not line or '生成回复：' in line or 'CQ:image' in line or 'CQ:vedio' in line or len(line) < 59:
                continue

            if not( 'nanami' in line or '七海' in line or '千秋' in line or '娜娜米' in line):
                print(f"line未检测到关键词，跳过:{line}")
                continue

            # print(f"get line:{line}   len={len(line)}")

            # 尝试匹配消息格式
            match = pattern.search(line)
            if match:
                content = match.group(1).strip()  # 提取消息内容
                label = 1 if match.group(2) == 'related' else 0  # 提取分类标签
                # 写入TSV格式（内容\t标签）
                outfile.write(f'{content}\t{label}\n')
            else:
                # 打印未匹配的行以便调试
                print(f"未匹配的行: {line}")

if __name__ == '__main__':
    input_filename = '2025-02-27.log'  # 确保文件名正确
    output_filename = 'dataset_tmp.tsv'

    if os.path.exists(input_filename):
        process_log_file(input_filename, output_filename)
        print(f'数据集已生成：{output_filename}')
    else:
        print(f'错误：输入文件 {input_filename} 不存在')