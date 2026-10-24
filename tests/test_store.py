"""存储层测试：仓库读写、向量索引、删除与兼容性检查。"""

from __future__ import annotations

from docqa.ingest.pipeline import IngestPipeline
from docqa.store.chunk_repo import ChunkRepository


def test_ingest_returns_expected_counts(ingested):
    assert ingested.chunks > 0
    assert ingested.parents > 0
    assert ingested.pages >= 1
    assert ingested.doc_id


def test_repository_lists_and_stats(services, ingested):
    docs = services.repo.list_documents()
    assert any(d.doc_id == ingested.doc_id for d in docs)

    stats = services.repo.stats()
    assert stats.documents >= 1
    assert stats.chunks >= ingested.chunks
    assert stats.chunk_size == services.settings.chunking.chunk_size


def test_reingest_same_content_is_idempotent(services, ingested, sample_text):
    """内容哈希做 doc_id：同一份内容改名再入库，不该产生两份向量。"""
    before = services.chroma.count()
    services.ingest.ingest_bytes("改了个名字.txt", sample_text.encode("utf-8"))
    services.refresh()
    assert services.chroma.count() == before
    assert services.repo.stats().documents >= 1


def test_chroma_query_returns_scored_hits(services):
    hits = services.chroma.query(services.embedder.embed_query("学籍异动"), top_k=3)
    assert hits
    assert 0.0 <= hits[0]["score"] <= 1.0
    assert hits[0]["metadata"]["doc_name"]


def test_soft_delete_removes_document(tmp_path, services):
    """单独造一篇文档来删，避免影响共享索引。"""
    pipeline = IngestPipeline(
        services.settings,
        embedder=services.embedder,
        chroma=services.chroma,
        repo=services.repo,
    )
    response = pipeline.ingest_bytes(
        "临时文件.md", "# 临时\n\n这是一篇用于删除测试的临时文档，内容足够长以满足切分要求。".encode()
    )
    services.refresh()
    assert pipeline.delete(response.doc_id) is True
    services.refresh()
    assert pipeline.delete(response.doc_id) is False
    assert all(d.doc_id != response.doc_id for d in services.repo.list_documents())


def test_compatibility_warning_on_model_change(tmp_path):
    repo = ChunkRepository(tmp_path / "idx")
    repo._write_manifest(
        {"params": {"embedding_model": "old-model", "chunk_size": 100, "chunk_overlap": 10}}
    )
    warnings = repo.check_compatibility("new-model", 200, 20)
    assert len(warnings) == 3
    assert "重新入库" in warnings[0]
