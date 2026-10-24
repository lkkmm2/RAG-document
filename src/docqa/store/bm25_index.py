"""BM25 稀疏索引（中文用 jieba 分词）。

为什么向量检索之外还要 BM25：
- 向量模型擅长"语义相近"，但对专有名词、编号、条款号这类**字面精确**匹配很弱，
  比如"第三十二条""学籍异动"里的编号和术语，向量容易召回一堆"意思差不多"的段落；
- BM25 正好相反，词面精确命中，但不理解同义改写。
两者互补，所以做混合召回。融合策略见 retrieval/fusion.py。
"""

from __future__ import annotations

import re
from typing import Any

from docqa.logging_setup import get_logger
from docqa.schemas import Chunk

log = get_logger("store.bm25")

# 只保留有区分度的停用词，别把"学籍"这种领域词误杀
STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "是", "在", "有", "为", "对", "从", "到", "中",
    "上", "下", "按", "并", "等", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这个", "那个", "这些", "那些", "什么", "怎么", "如何", "哪些", "哪个", "是否",
    "可以", "需要", "应该", "一个", "以及", "根据", "关于", "对于", "如果", "但是",
    "a", "an", "the", "of", "to", "and", "or", "is", "are", "in", "on", "for",
}

_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_NUM_RE = re.compile(r"^\d+$")


def tokenize(text: str) -> list[str]:
    import jieba

    tokens: list[str] = []
    for raw in jieba.lcut(text.lower()):
        token = raw.strip()
        if not token or token in STOPWORDS:
            continue
        if _PUNCT_RE.match(token):
            continue
        # 单个数字/字母几乎没区分度，丢掉；单个汉字保留（"学""籍"在短 query 里有意义）
        if len(token) == 1 and (_NUM_RE.match(token) or token.isascii()):
            continue
        tokens.append(token)
    return tokens


class BM25Index:
    """内存态 BM25。启动时从 chunks.jsonl 重建，千级 chunk 只要几百毫秒。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: Any = None
        self._ids: list[str] = []
        self._tokens: list[list[str]] = []

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._ids = [c.id for c in chunks]
        self._tokens = [tokenize(c.text) for c in chunks]
        corpus = [tokens or ["_empty_"] for tokens in self._tokens]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b) if corpus else None
        log.info("BM25 索引就绪：%s 篇文档，平均 %.1f 词", len(corpus),
                 (sum(len(t) for t in self._tokens) / len(corpus)) if corpus else 0)

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def ready(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if not self.ready or not query.strip():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"id": self._ids[i], "score": round(float(scores[i]), 6), "rank": rank + 1}
            for rank, i in enumerate(ranked)
            if scores[i] > 0
        ]
