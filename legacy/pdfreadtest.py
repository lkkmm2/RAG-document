from pypdf import PdfReader
reader = PdfReader("D:\\github-shangchuangxiangmu\\RAG-基于大模型API的文档问答助手\\ku\\t1.pdf")  # 换成你的 PDF 路径
text = ""
for page in reader.pages:
    text += page.extract_text() or ""
print("总字数：", len(text))
print("前 500 字：")
print(text[:500])