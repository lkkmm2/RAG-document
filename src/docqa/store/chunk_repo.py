"""Chunk 仓库：整条链路的「事实源」。

为什么不把 chunk 只存在 Chroma 里？
1. 向量库是索引，不是数据库：换 embedding 模型要重建，但原文不该跟着丢；
2. BM25 需要全量文本才能建索引，每次从向量库反查文本很别扭；
3. 评测要对齐 gold_chunk_id，需要能直接读文件做 diff。

所以：chunks.jsonl / parents.jsonl 是事实源，Chroma 是可重建的派生索引。
manifest.json 记录"这份索引是用哪个模型、什么分块参数建的"——
模型换了但索引没重建，是 RAG 项目里最常见、最难查的一类 bug，
把参数写进 manifest 并在启动时比对，能直接把它挡在门口。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from docqa.logging_setup import get_logger
from docqa.schemas import Chunk, DocumentInfo, IndexStats, ParentChunk

log = get_logger("store.repo")

MANIFEST_VERSION = 2


class ChunkRepository:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = index_dir / "chunks.jsonl"
        self.parents_path = index_dir / "parents.jsonl"
        self.manifest_path = index_dir / "manifest.json"
        self._lock = threading.Lock()
        self._parents_cache: dict[str, ParentChunk] | None = None

    # ------------------------------------------------------------------ 读
    def load_chunks(self) -> list[Chunk]:
        return [Chunk.model_validate_json(line) for line in self._iter_lines(self.chunks_path)]

    def load_parents(self) -> dict[str, ParentChunk]:
        if self._parents_cache is None:
            self._parents_cache = {
                p.id: p for p in self._iter_model(self.parents_path, ParentChunk)
            }
        return self._parents_cache

    def get_parents(self, parent_ids: list[str]) -> dict[str, ParentChunk]:
        parents = self.load_parents()
        return {pid: parents[pid] for pid in parent_ids if pid in parents}

    def read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def list_documents(self) -> list[DocumentInfo]:
        manifest = self.read_manifest()
        docs = manifest.get("documents", {})
        return [DocumentInfo.model_validate(v) for v in docs.values()]

    def stats(self) -> IndexStats:
        manifest = self.read_manifest()
        params = manifest.get("params", {})
        docs = manifest.get("documents", {})
        return IndexStats(
            documents=len(docs),
            chunks=sum(d.get("chunks", 0) for d in docs.values()),
            parents=sum(d.get("parents", 0) for d in docs.values()),
            embedding_model=params.get("embedding_model", ""),
            chunk_size=params.get("chunk_size", 0),
            chunk_overlap=params.get("chunk_overlap", 0),
            updated_at=manifest.get("updated_at", ""),
        )

    # ------------------------------------------------------------------ 写
    def upsert_document(
        self,
        doc: DocumentInfo,
        chunks: list[Chunk],
        parents: dict[str, ParentChunk],
        params: dict,
    ) -> None:
        with self._lock:
            old_chunks = [c for c in self.load_chunks() if c.doc_id != doc.doc_id]
            old_parents = {
                k: v for k, v in self.load_parents().items() if v.doc_id != doc.doc_id
            }
            new_chunks = old_chunks + chunks
            new_parents = {**old_parents, **parents}
            self._write_all(new_chunks, new_parents)

            manifest = self.read_manifest() or {"version": MANIFEST_VERSION}
            manifest.setdefault("documents", {})[doc.doc_id] = doc.model_dump()
            manifest["params"] = params
            manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
            manifest["version"] = MANIFEST_VERSION
            self._write_manifest(manifest)
            self._parents_cache = None

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            chunks = self.load_chunks()
            kept = [c for c in chunks if c.doc_id != doc_id]
            if len(kept) == len(chunks):
                return False
            parents = {k: v for k, v in self.load_parents().items() if v.doc_id != doc_id}
            self._write_all(kept, parents)

            manifest = self.read_manifest()
            manifest.get("documents", {}).pop(doc_id, None)
            manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self._write_manifest(manifest)
            self._parents_cache = None
            return True

    def clear(self) -> None:
        with self._lock:
            for path in (self.chunks_path, self.parents_path, self.manifest_path):
                path.unlink(missing_ok=True)
            self._parents_cache = None

    # ------------------------------------------------------------------ 兼容性
    def check_compatibility(self, embedding_model: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """索引参数和当前配置不一致时给出明确警告，而不是让效果悄悄变差。"""
        params = self.read_manifest().get("params", {})
        warnings: list[str] = []
        if not params:
            return warnings
        if params.get("embedding_model") and params["embedding_model"] != embedding_model:
            warnings.append(
                f"索引由 {params['embedding_model']} 建立，当前使用 {embedding_model}，"
                "向量空间不一致，请重新入库（ingest --reset）"
            )
        if params.get("chunk_size") and params["chunk_size"] != chunk_size:
            warnings.append(f"索引分块大小 {params['chunk_size']} 与当前配置 {chunk_size} 不一致")
        if params.get("chunk_overlap") and params["chunk_overlap"] != chunk_overlap:
            warnings.append(
                f"索引重叠 {params['chunk_overlap']} 与当前配置 {chunk_overlap} 不一致"
            )
        return warnings

    # ------------------------------------------------------------------ 内部
    def _write_all(self, chunks: list[Chunk], parents: dict[str, ParentChunk]) -> None:
        tmp_chunks = self.chunks_path.with_suffix(".jsonl.tmp")
        with tmp_chunks.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json() + "\n")
        tmp_chunks.replace(self.chunks_path)

        tmp_parents = self.parents_path.with_suffix(".jsonl.tmp")
        with tmp_parents.open("w", encoding="utf-8") as fh:
            for parent in parents.values():
                fh.write(parent.model_dump_json() + "\n")
        tmp_parents.replace(self.parents_path)

    def _write_manifest(self, manifest: dict) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    @staticmethod
    def _iter_lines(path: Path):
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line

    def _iter_model(self, path: Path, model):
        for line in self._iter_lines(path):
            try:
                yield model.model_validate_json(line)
            except Exception:  # noqa: BLE001 - 单行坏数据不该让整个索引读不出来
                log.warning("跳过坏数据行：%s", line[:80])
