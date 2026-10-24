"""DocQA —— 可评测、可观测的 RAG 文档问答系统。

分层：

    ingest      文档解析 → 清洗 → 结构化切分（父块/子块）
    embedding   文本向量化（本地 bge / 离线哈希兜底）
    store       Chroma 向量索引 + BM25 稀疏索引
    retrieval   向量召回 + BM25 召回 → RRF 融合 → Cross-Encoder 精排
    llm         大模型客户端（DeepSeek / Mock）
    rag         Prompt 组装、引用校验、流式回答
    agent       ReAct 工具调用循环
    api         FastAPI 接口层
    eval        检索与回答质量评测
"""

__version__ = "2.0.0"
