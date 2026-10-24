"""服务容器：把各层装配成一个对象，供 API / 脚本 / 前端复用。

手写装配而不是上依赖注入框架，原因很简单：
这个项目的依赖关系是一棵浅树，`build_services()` 20 行就讲清楚了，
上框架反而让人看不出"谁依赖谁"。面试时这一屏就能说清架构。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from docqa.agent.react import ReActAgent
from docqa.agent.tools import build_tools
from docqa.config import Settings, get_settings
from docqa.embedding.embedder import Embedder, build_embedder
from docqa.ingest.pipeline import IngestPipeline
from docqa.llm.client import LLMClient, build_llm
from docqa.logging_setup import get_logger, setup_logging
from docqa.rag.answerer import RAGAnswerer
from docqa.retrieval.pipeline import RetrievalPipeline
from docqa.retrieval.query_rewrite import QueryRewriter
from docqa.retrieval.rerank import Reranker, build_reranker
from docqa.schemas import HealthResponse
from docqa.store.bm25_index import BM25Index
from docqa.store.chroma_store import ChromaStore
from docqa.store.chunk_repo import ChunkRepository

log = get_logger("services")


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    chroma: ChromaStore
    repo: ChunkRepository
    bm25: BM25Index
    reranker: Reranker
    llm: LLMClient
    retriever: RetrievalPipeline
    answerer: RAGAnswerer
    agent: ReActAgent
    ingest: IngestPipeline

    def refresh(self) -> None:
        self.retriever.refresh()

    def health(self) -> HealthResponse:
        stats = self.repo.stats()
        return HealthResponse(
            status="ok",
            version=self.settings.app_version,
            llm_provider=self.llm.name,
            llm_model=self.llm.model,
            embedding_provider=self.settings.embedding.provider,
            embedding_model=getattr(self.embedder, "name", ""),
            rerank_enabled=self.reranker.enabled,
            documents=stats.documents,
            chunks=stats.chunks,
        )


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    setup_logging(settings.api.log_level, settings.api.log_format)

    log.info("启动 DocQA %s", settings.app_version)
    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = build_embedder(settings.embedding, settings.model_dir)
    chroma = ChromaStore(index_dir / "chroma", settings.store.collection, settings.store.space)
    repo = ChunkRepository(index_dir)
    bm25 = BM25Index()
    llm = build_llm(settings.llm)
    reranker = build_reranker(settings.rerank, settings.model_dir)

    for warning in repo.check_compatibility(
        getattr(embedder, "name", ""),
        settings.chunking.chunk_size,
        settings.chunking.chunk_overlap,
    ):
        log.warning(warning)

    rewriter = QueryRewriter(
        llm,
        enabled=settings.retrieval.rewrite_query and llm.name != "mock",
        history_turns=settings.retrieval.rewrite_history_turns,
    )
    retriever = RetrievalPipeline(
        settings,
        embedder=embedder,
        chroma=chroma,
        bm25=bm25,
        repo=repo,
        reranker=reranker,
        rewriter=rewriter,
    )
    retriever.refresh()

    answerer = RAGAnswerer(settings, llm, retriever)
    tools = build_tools(retriever, repo)
    agent = ReActAgent(settings.agent, llm, tools)
    ingest = IngestPipeline(settings, embedder=embedder, chroma=chroma, repo=repo)

    log.info(
        "装配完成：llm=%s(%s) embedding=%s rerank=%s chunks=%s",
        llm.name,
        llm.model,
        getattr(embedder, "name", "?"),
        reranker.enabled,
        retriever.chunk_count(),
    )
    return Services(
        settings=settings,
        embedder=embedder,
        chroma=chroma,
        repo=repo,
        bm25=bm25,
        reranker=reranker,
        llm=llm,
        retriever=retriever,
        answerer=answerer,
        agent=agent,
        ingest=ingest,
    )


@lru_cache(maxsize=1)
def get_services() -> Services:
    return build_services()


def reset_services() -> None:
    """测试用：清掉单例，让下次调用重新装配。"""
    get_services.cache_clear()
    get_settings.cache_clear()
