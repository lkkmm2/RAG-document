"""入库编排：解析 → 清洗 → 切分 → 向量化 → 写索引。

关键设计：
- doc_id 用**文件内容哈希**，不是文件名。同一个文件改名后重新上传会被识别为同一篇，
  不会产生两份重复向量；不同文件同名也不会互相覆盖。
- 重新入库时先按 doc_id 删旧向量，再写新的，避免"改了配置但旧向量还在"的脏索引。
- manifest 里记下 embedding 模型和分块参数，启动时比对，防止模型换了索引没重建。
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from datetime import datetime
from pathlib import Path

from docqa.config import Settings
from docqa.ingest.cleaner import clean_pages
from docqa.ingest.loader import LoadedDocument, load_document
from docqa.ingest.splitter import DocumentSplitter
from docqa.logging_setup import get_logger
from docqa.schemas import DocumentInfo, UploadResponse
from docqa.store.chroma_store import ChromaStore
from docqa.store.chunk_repo import ChunkRepository

log = get_logger("ingest.pipeline")

EMBED_BATCH = 64


class IngestPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder,
        chroma: ChromaStore,
        repo: ChunkRepository,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.chroma = chroma
        self.repo = repo
        self.splitter = DocumentSplitter(settings.chunking)

    # ------------------------------------------------------------------ 入口
    def ingest_path(self, path: Path) -> UploadResponse:
        data = path.read_bytes()
        return self.ingest_bytes(path.name, data)

    def ingest_bytes(self, filename: str, data: bytes) -> UploadResponse:
        started = time.perf_counter()
        doc_id = hashlib.sha1(data).hexdigest()[:12]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / Path(filename).name
            tmp_path.write_bytes(data)
            loaded = load_document(
                tmp_path,
                max_pages=self.settings.ingest.max_pages,
                extract_tables=self.settings.ingest.extract_tables,
            )

        if loaded.chars == 0:
            raise ValueError("文档解析后为空，可能是扫描件或文件损坏")

        pages = clean_pages(
            loaded.pages,
            dedupe_header_footer=self.settings.ingest.dedupe_header_footer,
            header_footer_ratio=self.settings.ingest.header_footer_ratio,
        )
        result = self.splitter.split(doc_id, filename, pages)
        if not result.chunks:
            raise ValueError("切分后没有有效内容，请检查文档格式")

        self._write_index(doc_id, filename, result, size_bytes=len(data), pages=len(pages))

        elapsed = round((time.perf_counter() - started) * 1000, 2)
        log.info(
            "入库完成：%s（%s）%s 页 / %s 子块 / %s 父块 / %.0fms",
            filename, doc_id, len(pages), len(result.chunks), len(result.parents), elapsed,
        )
        return UploadResponse(
            doc_id=doc_id,
            doc_name=filename,
            pages=len(pages),
            chunks=len(result.chunks),
            parents=len(result.parents),
            chars=result.chars,
            elapsed_ms=elapsed,
            warnings=loaded.warnings,
        )

    def delete(self, doc_id: str) -> bool:
        self.chroma.delete(where={"doc_id": doc_id})
        return self.repo.delete_document(doc_id)

    # ------------------------------------------------------------------ 写索引
    def _write_index(self, doc_id, filename, result, *, size_bytes: int, pages: int) -> None:
        chunks = result.chunks
        log.info("向量化 %s 个 chunk（模型 %s）...", len(chunks), self.embedder.name)

        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            embeddings.extend(self.embedder.embed_documents([c.text for c in batch]))

        # 先删同 doc 的旧向量，避免重复 upsert 留下孤儿
        self.chroma.delete(where={"doc_id": doc_id})
        self.chroma.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "section": c.section,
                    "page": int(c.page),
                    "order": int(c.order),
                    "parent_id": c.parent_id,
                }
                for c in chunks
            ],
        )

        info = DocumentInfo(
            doc_id=doc_id,
            doc_name=filename,
            pages=pages,
            chunks=len(chunks),
            parents=len(result.parents),
            chars=result.chars,
            size_bytes=size_bytes,
            ingested_at=datetime.now().isoformat(timespec="seconds"),
        )
        params = {
            "embedding_model": getattr(self.embedder, "name", "unknown"),
            "embedding_dim": getattr(self.embedder, "dim", 0),
            "chunk_size": self.settings.chunking.chunk_size,
            "chunk_overlap": self.settings.chunking.chunk_overlap,
            "parent_size": self.settings.chunking.parent_size,
        }
        self.repo.upsert_document(info, chunks, result.parents, params)

    # ------------------------------------------------------------------ 批量
    def ingest_directory(self, directory: Path) -> list[UploadResponse]:
        responses: list[UploadResponse] = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}:
                try:
                    responses.append(self.ingest_path(path))
                except Exception as exc:  # noqa: BLE001
                    log.error("跳过 %s：%s", path.name, exc)
        return responses


def describe(document: LoadedDocument) -> str:
    return f"{document.doc_name}: {len(document.pages)} 页 / {document.chars} 字"
