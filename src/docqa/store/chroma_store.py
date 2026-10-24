"""Chroma 向量索引封装。

只做一件事：把「向量 → chunk_id」的近邻查询包成稳定的接口。
embedding 永远由上层算好再传进来（`embedding_function=None`），
这样：
- 换模型只改 embedding 层，不动存储层；
- 不会出现"入库用 A 模型、查询用 B 模型"这种静默错误；
- 单元测试可以用哈希向量跑，不需要下载模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docqa.logging_setup import get_logger

log = get_logger("store.chroma")


class ChromaStore:
    def __init__(self, persist_dir: Path, collection: str, space: str = "cosine") -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection_name = collection
        self.space = space
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": space},
            embedding_function=None,
        )

    # ------------------------------------------------------------------ 写
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        # 分批，避免单次请求过大
        batch = 256
        for start in range(0, len(ids), batch):
            end = start + batch
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def delete(self, where: dict[str, Any]) -> None:
        self._collection.delete(where=where)

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.space},
            embedding_function=None,
        )

    # ------------------------------------------------------------------ 读
    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(self.count(), 1)),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for i, chunk_id in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else 1.0
            # cosine 距离 → 相似度；其它空间退化为负距离，只保证排序正确
            score = 1.0 - distance if self.space == "cosine" else -distance
            hits.append(
                {
                    "id": chunk_id,
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": distance,
                    "score": round(float(score), 6),
                }
            )
        return hits

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:  # noqa: BLE001
            return 0
