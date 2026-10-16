# 基于大模型API的文档文档助手（RAG）

# 简介
一个基于 Python + Streamlit + 大模型API（DeepSeek） + Chroma 的文档问答助手。支持上传 TXT/PDF 文件，自动切分、向量化入库，通过 RAG 检索增强生成，让大模型基于文档内容回答问题，并显示引用来源。

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


# 目录结构
见项目根目录下的说明。
