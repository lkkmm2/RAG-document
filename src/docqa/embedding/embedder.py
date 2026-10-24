"""向量化。

两个实现：
- LocalBGEEmbedder：真正的语义向量（bge 中文模型），有模型就优先用它；
- HashingEmbedder：字符 n-gram 哈希，零依赖、确定性，用于单元测试和"没网没模型也要能跑"的兜底。

面试点：为什么 query 要加指令前缀而 passage 不加？
bge 是"非对称检索"训练出来的：训练时 query 侧带指令、passage 侧不带。
推理时保持一致，相似度分布才对得上；不加指令，短 query 和长 passage 的向量会偏到不同区域。
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from docqa.config import EmbeddingSettings
from docqa.logging_setup import get_logger

log = get_logger("embedding")

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# ------------------------------------------------------------------ 本地 bge
class LocalBGEEmbedder:
    def __init__(self, cfg: EmbeddingSettings, base_dir: Path) -> None:
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.name = cfg.model_name
        path = resolve_model_path(cfg.model_name, base_dir, cfg.local_files_only)
        log.info("加载 embedding 模型：%s（%s）", cfg.model_name, path)
        self._model = SentenceTransformer(path, device=cfg.device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        log.info("embedding 维度：%s", self.dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        query = f"{self.cfg.query_instruction}{text}" if self.cfg.query_instruction else text
        vector = self._model.encode(
            [query],
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vector[0], dtype=np.float32).tolist()


# ------------------------------------------------------------------ 哈希兜底
class HashingEmbedder:
    """字符 n-gram → 固定维度哈希向量。

    不是"玩具"：它等价于一个固定权重的 sparse-to-dense 投影，
    对中文词面重合度敏感，足以支撑单元测试和流程验证；
    但它是词面匹配，不理解同义改写，所以生产环境必须换 bge。
    """

    def __init__(self, dim: int = 256, ngrams: tuple[int, ...] = (2, 3)) -> None:
        self.dim = dim
        self.ngrams = ngrams
        self.name = f"hashing-{dim}d"

    def _encode(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        grams: list[str] = []
        for n in self.ngrams:
            grams.extend("".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        grams.extend(tokens)
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


# ------------------------------------------------------------------ 工厂
def resolve_model_path(model_name: str, base_dir: Path, local_files_only: bool) -> str:
    """优先本地目录 → ModelScope → HF 镜像。国内环境这三层基本能兜住。"""
    base_dir.mkdir(parents=True, exist_ok=True)
    local = base_dir / model_name.replace("/", "__")
    if (local / "config.json").exists():
        return str(local)
    if local_files_only:
        raise FileNotFoundError(f"本地未找到模型 {model_name}（{local}），且已设置 local_files_only")

    try:
        from modelscope import snapshot_download  # type: ignore[import-not-found]

        log.info("从 ModelScope 下载 %s ...", model_name)
        path = snapshot_download(model_name, cache_dir=str(base_dir / "_modelscope"))
        return str(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("ModelScope 不可用（%s），改用 HuggingFace 镜像", exc)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download as hf_download

    log.info("从 HuggingFace 镜像下载 %s ...", model_name)
    return str(hf_download(model_name, cache_dir=str(base_dir / "_huggingface")))


def build_embedder(cfg: EmbeddingSettings, base_dir: Path) -> Embedder:
    if cfg.provider == "hashing":
        log.info("使用哈希 embedding（离线兜底模式）")
        return HashingEmbedder()
    try:
        return LocalBGEEmbedder(cfg, base_dir)
    except Exception as exc:  # noqa: BLE001 - 模型拿不到时不能让服务起不来
        log.error("本地 embedding 加载失败，降级为哈希模式：%s", exc)
        return HashingEmbedder()
