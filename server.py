import torch
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
import json


HOST = '0.0.0.0'
PORT = 15974

# 本地数据地址
label_path = 'data/dice/class.txt'

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

# 定义 BERT 分类模型
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

# 加载模型
save_path = 'bert_checkpoint'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertClassifier()
model = model.to(device)
model.load_state_dict(torch.load(os.path.join(save_path, 'best.pt'), map_location=device))
model.eval()

# 自定义 HTTP 请求处理器
class ClassificationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 获取请求内容长度
        content_length = int(self.headers['Content-Length'])
        # 读取请求体
        post_data = self.rfile.read(content_length)
        
        try:
            # 解析 JSON 数据
            data = json.loads(post_data.decode('utf-8'))
            action = data.get('action', '')
            request_data = data.get('data', {})

            # 检查 action 是否正确
            if action != 'text_classification':
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {
                    'error': 'Invalid action',
                    'message': 'The action must be "text_classification".'
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return

            # 获取文本
            text = request_data.get('text', '')
            data_type = request_data.get('type', '')

            if not text:
                # 如果文本为空，返回错误
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {
                    'error': 'No text provided',
                    'message': 'The "text" field is required in the "data" object.'
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return

            # 对输入文本进行分词和编码
            bert_input = tokenizer(text, padding='max_length', max_length=35, truncation=True, return_tensors="pt")
            input_ids = bert_input['input_ids'].to(device)
            masks = bert_input['attention_mask'].unsqueeze(1).to(device)

            # 模型推理
            with torch.no_grad():
                output = model(input_ids, masks)
                pred = output.argmax(dim=1).item()

            # 返回分类结果
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'action': 'text_classification',
                'data': {
                    'type': data_type,
                    'category': real_labels[pred]
                }
            }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            # 处理异常
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'error': 'Internal Server Error',
                'message': str(e)
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))

# 启动 HTTP 服务
def run_server():
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, ClassificationHandler)
    print(f"Starting server on http://{server_address[0]}:{server_address[1]}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()