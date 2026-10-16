import streamlit as st
import chromadb
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="AI 文档问答助手", layout="centered")
st.title("📄 我的 AI 文档问答助手")

# 初始化 API 客户端
client = OpenAI(
    api_key="sk-e3dd959dfad247e3ba01494f37101bdf",
    base_url="https://api.deepseek.com"
)

# 初始化 Chroma 向量库
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 侧边栏：上传文档
with st.sidebar:
    st.header("📁 文档上传")
    uploaded_file = st.file_uploader("上传 PDF 或 TXT 文件", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("构建向量库"):
            with st.spinner("正在读取文档并入库..."):
                # 1. 读取文件
                if uploaded_file.name.endswith(".txt"):
                    text = uploaded_file.read().decode("utf-8")
                else:
                    reader = PdfReader(uploaded_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                
                # 2. 按段落切分
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                chunks = []
                current = ""
                for p in paragraphs:
                    if len(current) + len(p) < 300:
                        current += p + "\n"
                    else:
                        if current:
                            chunks.append(current)
                        current = p
                if current:
                    chunks.append(current)
                
                # 3. 存入 Chroma
                try:
                    chroma_client.delete_collection("documents")
                except:
                    pass
                collection = chroma_client.create_collection("documents")
                collection.add(
                    documents=chunks,
                    ids=[f"chunk_{i}" for i in range(len(chunks))]
                )
                st.success(f"✅ 已切分为 {len(chunks)} 段，入库完成！")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 聊天输入
prompt = st.chat_input("请输入你的问题...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("正在检索文档并思考..."):
            # 1. 检索文档
            context = ""
            try:
                collection = chroma_client.get_collection("documents")
                results = collection.query(query_texts=[prompt], n_results=3)
                context = "\n\n".join(results['documents'][0])
            except Exception as e:
                context = "未找到相关文档。"

            # 2. 拼 Prompt，调用大模型
            system_prompt = f"""你是一个文档问答助手。
请严格根据下面提供的上下文回答问题。

上下文：
{context}

规则：
1. 优先从上下文里找答案。
2. 如果上下文确实没有相关信息，直接说“文档中未提及”。
3. 不要自己编造。
"""
            messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages
            )
            answer = response.choices[0].message.content
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})