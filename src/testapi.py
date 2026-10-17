import requests

# 1. 测试上传接口
url_upload = "http://127.0.0.1:8000/upload"

# 注意：这里要换成你电脑里真实存在的 txt 文件路径！
# 如果你用的正斜杠或者反斜杠报错，试着改一下路径格式
file_path = "D:\\github-shangchuangxiangmu\\RAG-基于大模型API的文档问答助手\\ku\\t2.txt" 

try:
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/plain")}
        res = requests.post(url_upload, files=files)
        
    print("上传结果：", res.status_code, res.text)
    
except FileNotFoundError:
    print(f"找不到文件 {file_path}，请把 file_path 改成你真实的 txt 文件绝对路径！")
    exit()

# 2. 测试问答接口
url_ask = "http://127.0.0.1:8000/ask"
data = {
    "question": "阿里巴巴总部在哪里？",
    "top_k": 3
}

print("\n正在思考回答，请稍等...")
res_ask = requests.post(url_ask, json=data)
print("问答结果：", res_ask.status_code, res_ask.text)