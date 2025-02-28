import re

# 修复后的正则表达式
pattern = r'^\[\d{2}:\d{2}:\d{2}\] 收到来自 (?:群聊|私聊)\d+中\d+的消息，内容：(.*)，分类为：(related|unrelated)'
raw_str = "[00:05:18] 收到来自群聊586917668中3848812783的消息，内容：怎么就没有吃掉的选项，分类为：unrelated"

# 使用 re.match 进行匹配
match = re.match(pattern, raw_str)

if match:
    print("match. ")
    print("msg:", match.group(1))  # 提取消息内容
    print("label:", match.group(2))  # 提取分类标签
else:
    print("fail")