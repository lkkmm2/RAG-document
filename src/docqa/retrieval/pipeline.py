"""检索编排：改写 → 双路召回 → 融合 → 精排 → 父块回溯 → 上下文组装。

这是整个项目里最值得讲的一段代码，因为它把 RAG 的"召回质量"拆成了可测量的四步，
每一步都能单独关掉做消融实验（eval 脚本就是这么对比出数字的）。
"""

from __future__ import annotations

from typing import Any

from docqa.config import Settings
from docqa.logging_setup import Timer, get_logger
from docqa.retrieval.fusion import mmr_select, rrf_fuse, weighted_fuse
from docqa.retrieval.query_rewrite import QueryRewriter
from docqa.retrieval.rerank import Reranker
from docqa.schemas import (
    Chunk,
    ContextItem,
    Message,
    RetrievalDebug,
    RetrievalResult,
    ScoredChunk,
)
from docqa.store.bm25_index import BM25Index, tokenize
from docqa.store.chroma_store import ChromaStore
from docqa.store.chunk_repo import ChunkRepository

log = get_logger("retrieval")


class RetrievalPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Any,
        chroma: ChromaStore,
        bm25: BM25Index,
        repo: ChunkRepository,
        reranker: Reranker,
        rewriter: QueryRewriter,
    ) -> None:
        self.settings = settings
        self.cfg = settings.retrieval
        self.embedder = embedder
        self.chroma = chroma
        self.bm25 = bm25
        self.repo = repo
        self.reranker = reranker
        self.rewriter = rewriter
        self._chunk_map: dict[str, Chunk] = {}

    # ------------------------------------------------------------------ 索引生命周期
    def refresh(self) -> None:
        """入库 / 删除文档后重建内存索引。"""
        chunks = self.repo.load_chunks()
        self._chunk_map = {c.id: c for c in chunks}
        self.bm25.build(chunks)
        log.info("检索索引已刷新：%s 个 chunk", len(self._chunk_map))

    @property
    def is_empty(self) -> bool:
        return not self._chunk_map and self.chroma.count() == 0

    def chunk_count(self) -> int:
        return max(len(self._chunk_map), self.chroma.count())

    # ------------------------------------------------------------------ 主流程
    def retrieve(
        self,
        query: str,
        *,
        history: list[Message] | None = None,
        top_k: int | None = None,
        debug: bool = False,
    ) -> RetrievalResult:
        timer = Timer()
        top_k = top_k or self.cfg.final_top_k

        rewritten = self.rewriter.rewrite(query, history) if self.cfg.rewrite_query else query
        timer.mark("rewrite")

        vector_hits: list[dict[str, Any]] = []
        if self.cfg.enable_vector and self.embedder is not None and self.chroma.count() > 0:
            query_vec = self.embedder.embed_query(rewritten)
            vector_hits = self.chroma.query(query_vec, top_k=self.cfg.vector_top_k)
        timer.mark("vector")

        bm25_hits = (
            self.bm25.search(rewritten, top_k=self.cfg.bm25_top_k)
            if self.cfg.enable_bm25
            else []
        )
        timer.mark("bm25")

        # ---- 融合 ----
        if self.cfg.fusion == "weighted":
            fused_pairs = weighted_fuse(
                [
                    [(h["id"], h["score"]) for h in vector_hits],
                    [(h["id"], h["score"]) for h in bm25_hits],
                ],
                weights=[self.cfg.vector_weight, self.cfg.bm25_weight],
            )
        else:
            fused_pairs = rrf_fuse(
                [[h["id"] for h in vector_hits], [h["id"] for h in bm25_hits]],
                k=self.cfg.rrf_k,
                weights=[self.cfg.vector_weight, self.cfg.bm25_weight],
            )
        fused_pairs = fused_pairs[: self.cfg.candidate_top_k]
        timer.mark("fuse")

        # ---- 组装候选（带上各自的中间分，方便前端展示与排查）----
        vec_score = {h["id"]: h["score"] for h in vector_hits}
        bm_score = {h["id"]: h["score"] for h in bm25_hits}
        candidates: list[dict[str, Any]] = []
        for chunk_id, fused_score in fused_pairs:
            chunk = self._chunk_map.get(chunk_id)
            if chunk is None:
                continue
            candidates.append(
                {
                    "id": chunk_id,
                    "text": chunk.text,
                    "chunk": chunk,
                    "score": fused_score,
                    "scores": {
                        "vector": round(vec_score.get(chunk_id, 0.0), 4),
                        "bm25": round(bm_score.get(chunk_id, 0.0), 4),
                        "fused": round(fused_score, 6),
                    },
                }
            )

        # ---- 精排 ----
        if self.reranker.enabled and candidates:
            candidates = self.reranker.rerank(
                rewritten, candidates, top_n=max(self.cfg.candidate_top_k, top_k * 3)
            )
        timer.mark("rerank")

        # ---- 去冗余（可选）----
        if self.cfg.mmr_lambda < 1.0 and len(candidates) > top_k:
            items = [
                (c["id"], c["score"], tokenize(c["text"]))
                for c in candidates
            ]
            keep = mmr_select(items, top_k, self.cfg.mmr_lambda)
            order = {cid: i for i, cid in enumerate(keep)}
            candidates = sorted(
                [c for c in candidates if c["id"] in order], key=lambda c: order[c["id"]]
            )

        # ---- 父块回溯 + 去重 ----
        candidate_ids = [c["id"] for c in candidates]
        contexts = self._build_contexts(candidates, top_k)
        timer.mark("context")

        debug_info = None
        if debug:
            debug_info = RetrievalDebug(
                original_query=query,
                rewritten_query=rewritten,
                vector=self._to_scored(vector_hits, "vector"),
                bm25=self._to_scored(bm25_hits, "bm25"),
                fused=[
                    ScoredChunk(
                        chunk_id=cid,
                        score=round(score, 6),
                        rank=i + 1,
                        source="fused",
                        scores={
                            "vector": round(vec_score.get(cid, 0.0), 4),
                            "bm25": round(bm_score.get(cid, 0.0), 4),
                        },
                    )
                    for i, (cid, score) in enumerate(fused_pairs)
                ],
                rerank=[
                    ScoredChunk(
                        chunk_id=c["id"],
                        score=round(c["score"], 6),
                        rank=i + 1,
                        source="rerank",
                        scores={k: round(v, 4) for k, v in c["scores"].items()},
                    )
                    for i, c in enumerate(candidates)
                ],
                timings_ms=timer.marks,
            )

        log.info(
            "检索完成：%r → 向量%s / BM25 %s / 融合%s / 重排%s / 上下文%s | %s",
            rewritten,
            len(vector_hits),
            len(bm25_hits),
            len(fused_pairs),
            len(candidates),
            len(contexts),
            timer.marks,
        )
        return RetrievalResult(contexts=contexts, candidate_ids=candidate_ids, debug=debug_info)

    # ------------------------------------------------------------------ 内部
    def _build_contexts(
        self, candidates: list[dict[str, Any]], top_k: int
    ) -> list[ContextItem]:
        parents = self.repo.load_parents()
        contexts: list[ContextItem] = []
        seen_parents: set[str] = set()
        used_chars = 0
        budget = self.cfg.max_context_chars

        for candidate in candidates:
            if len(contexts) >= top_k:
                break
            chunk: Chunk = candidate["chunk"]
            # 同一父块的不同子块只保留一条，避免把同一段话喂两遍
            if chunk.parent_id and chunk.parent_id in seen_parents:
                continue
            parent = parents.get(chunk.parent_id)
            text = parent.text if parent else chunk.text
            if used_chars + len(text) > budget and contexts:
                continue
            seen_parents.add(chunk.parent_id or chunk.id)
            used_chars += len(text)
            contexts.append(
                ContextItem(
                    index=len(contexts) + 1,
                    chunk_id=chunk.id,
                    parent_id=chunk.parent_id,
                    doc_name=chunk.doc_name,
                    section=chunk.section,
                    page=parent.page if parent else chunk.page,
                    text=text,
                    score=round(candidate["score"], 6),
                    scores={k: round(v, 4) for k, v in candidate["scores"].items()},
                )
            )
        return contexts

    @staticmethod
    def _to_scored(hits: list[dict[str, Any]], source: str) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk_id=h["id"],
                score=round(float(h.get("score", 0.0)), 6),
                rank=i + 1,
                source=source,
            )
            for i, h in enumerate(hits[:10])
        ]
