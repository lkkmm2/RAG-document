# 基于大模型API的文档文档助手（RAG）
本项目是一个基于 RAG（检索增强生成）架构的文档问答应用。支持上传 PDF/TXT 文档，系统会对文档进行切分、向量化入库，并在用户提问时检索相关片段，结合大模型生成准确、不瞎编的回答。

项目此阶段包含前端交互界面（Streamlit）与后端服务接口（FastAPI），实现了前后端分离的 RAG 架构
# 简介
一个基于 Python + Streamlit + 大模型API（DeepSeek） + Chroma 的文档问答助手。支持上传 TXT/PDF 文件，自动切分、向量化入库，通过 RAG 检索增强生成，让大模型基于文档内容回答问题，并显示引用来源。

# 核心功能
**文件解析**：支持上传 `.txt` 和 `.pdf` 格式文档，自动提取纯文本。
**文本切分与向量化**：按定长与重叠策略对文本进行切块（Chunking），使用 Chroma 构建本地向量数据库。
**RAG 检索问答**：基于用户问题检索 Top-K 相关文档片段，拼接 Prompt 并调用 DeepSeek 大模型 API 生成回答。
**防幻觉约束**：在 Prompt 中强制约束模型“仅根据文档内容回答”，并提供 `sources` 溯源，降低模型胡编乱造的概率。
**FastAPI 接口封装**：提供标准的 RESTful API，包含 `/upload` 和 `/ask` 接口，并自带 Swagger UI 在线调试文档。
**前端演示页面**：基于 Streamlit 构建的对话界面（开发中）。

# 技术支持
前端：Streamlit
大模型：DeepSeek API
向量库：Chroma
文档解析：pypdf
语言：Python

# 运行过程
1. 安装依赖：`pip install -r requirements.txt`
2. 配置 API Key（DeepSeek）
3. 启动：`streamlit run app.py`
4. 测试的库来源在目录下的ku文件夹中）代码标注了输入位置



# 项目存在的问题与解决：
1.模型说上下文里没有这方面的知识
先看，从pdf读取了什么内容
pdfreadtest.py代码记录了从pdf文件里面提取到的内容，如果遇到了模型回答不出上下文的问题，可以使用此代码来查看模型从文档里面读取了什么

2.pdf读取不了，不全类型不对




# 目录结构（主要）
```text
RAG-document/
src/
    api.py            # FastAPI 后端接口
    app.py            # Streamlit 前端页面
    testapi .py       # 测试文本接口
    pdfreadtest.py    # 测试pdf文件作为上下文时模型的读取能力 
.gitignore            # 忽略环境文件、向量库文件等
requirements.txt      # 项目依赖项
README.md             # 项目说明文档