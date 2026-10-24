import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from pypdf import PdfReader

# 1. 初始化 FastAPI，关掉自带 docs，用国内 CDN
app = FastAPI(docs_url=None, redoc_url=None)


# 2. 初始化大模型客户端和向量库（复用你 app.py 里的配置）
# 注意：这里留空，你本地测试时把 Key 填进去，但千万别传 GitHub！
client = OpenAI(
    api_key="",  # 你本地测试时把 Key 填进去
    base_url="https://api.deepseek.com"
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")


# 3. 文件切分和读取的辅助函数（跟你 app.py 里一样的逻辑）
def split_text(text: str, chunk_size: int = 300, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

def extract_text(file_path: str, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


# 4. 测试接口
@app.get("/health")
async def health():
    return {"status": "ok"}


# 5. 自定义文档页（解决白屏的代码，原样保留）
@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
  <title>RAG API Docs</title>
  <link rel="stylesheet" href="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: "#swagger-ui"
    });
  </script>
</body>
</html>
""")


# 6. 上传文档接口 (对应你 app.py 的文件上传按钮)
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="只支持 PDF 或 TXT 文件")

    tmp_path = None
    try:
        # 把上传的文件存到临时目录，处理完就删
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        text = extract_text(tmp_path, file.filename)
        if not text.strip():
            raise HTTPException(status_code=400, detail="文件内容为空或无法解析")

        chunks = split_text(text)
        ids = [f"{file.filename}-{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)

        return {"filename": file.filename, "chunks": len(chunks)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# 7. 问答接口 (对应你 app.py 的聊天框)
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    results = collection.query(query_texts=[req.question], n_results=req.top_k)
    docs = results.get("documents", [[]])[0]

    if not docs:
        raise HTTPException(status_code=404, detail="向量库为空，请先上传文档")

    context = "\n\n".join(docs)
    prompt = f"""请严格根据以下文档内容回答问题，不要编造。如果文档中没有相关信息，请回答“文档中未提及”。

文档内容：
{context}

问题：{req.question}
"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"大模型调用失败：{e}")

    return {"question": req.question, "answer": answer, "sources": docs}