"""精排（Cross-Encoder Rerank）。

为什么召回之后还要重排：
- 向量检索是 bi-encoder：query 和 doc 各自编码，最后算余弦。
  好处是可以预先建索引、快；坏处是两者没见过彼此，细粒度差异判断不准。
- Cross-Encoder 把 (query, doc) 拼在一起送进模型，能做逐词交互，
  相关性判断明显更准，但必须实时算，所以只对召回的前 20 条做。

这就是"两阶段检索"：粗排要召回率，精排要准确率。

实现说明（踩过的坑）：
一开始用 `sentence_transformers.CrossEncoder` 加载 bge-reranker-base，
结果 20 条候选打分全挤在 0.730 附近、排序不可用。直接用 transformers 加载同一个模型，
同一组 (query, passage) 的 logits 是 +2.46 / -3.74 / -10.20，区分度完全正常。
说明是封装层与当前 transformers 版本（v5）的兼容问题，不是模型问题。
所以这里直接使用 `AutoModelForSequenceClassification`，少一层黑盒，也方便解释。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from docqa.config import RerankSettings
from docqa.embedding.embedder import resolve_model_path
from docqa.logging_setup import get_logger

log = get_logger("retrieval.rerank")


class Reranker(Protocol):
    enabled: bool

    def rerank(self, query: str, items: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]: ...


class NoopReranker:
    enabled = False

    def rerank(self, query: str, items: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        return items[:top_n]


class CrossEncoderReranker:
    enabled = True

    def __init__(self, cfg: RerankSettings, base_dir: Path) -> None:
        self.cfg = cfg
        self.base_dir = base_dir
        self.model_name = cfg.model_name
        self._tokenizer = None
        self._model = None

    def _ensure_model(self):
        """懒加载：重排模型 1GB 级，不该拖慢服务启动。第一次真正用到时再加载。"""
        if self._model is not None:
            return self._tokenizer, self._model
        import torch  # noqa: F401 - 确认运行时有 torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = resolve_model_path(self.cfg.model_name, self.base_dir, local_files_only=False)
        log.info("加载 rerank 模型：%s（%s）", self.cfg.model_name, path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        return self._tokenizer, self._model

    def rerank(self, query: str, items: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        if not items:
            return []
        try:
            logits = self._score(query, [item["text"] for item in items])
        except Exception as exc:  # noqa: BLE001 - 精排挂了就退回融合排序，不让请求失败
            log.error("rerank 失败，退回融合结果：%s", exc)
            return items[:top_n]

        for item, logit in zip(items, logits, strict=False):
            item["scores"]["rerank_raw"] = round(logit, 4)
            item["scores"]["rerank"] = round(_sigmoid(logit), 4)
            item["score"] = round(_sigmoid(logit), 6)
        ranked = sorted(items, key=lambda x: x["score"], reverse=True)
        return ranked[:top_n]

    def _score(self, query: str, passages: list[str]) -> list[float]:
        import torch

        tokenizer, model = self._ensure_model()
        scores: list[float] = []
        batch_size = max(1, self.cfg.batch_size)
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            features = tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**features).logits.view(-1).float().tolist()
            scores.extend(logits)
        return scores


def _sigmoid(value: float) -> float:
    """logits → (0,1)，只为了前端好展示；排序不受影响。"""
    return 1.0 / (1.0 + pow(2.718281828, -value))


def build_reranker(cfg: RerankSettings, base_dir: Path) -> Reranker:
    if not cfg.enabled or cfg.provider == "none":
        log.info("精排已关闭")
        return NoopReranker()
    return CrossEncoderReranker(cfg, base_dir)
