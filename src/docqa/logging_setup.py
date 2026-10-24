"""结构化日志 + trace_id。

线上排查一个 RAG 请求，最需要的是"这一次请求都发生了什么"：
改写后的 query、向量召回了什么、BM25 召回了什么、融合后什么、重排后什么、花了多久。
trace_id 就是把这些日志串起来的那根线。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from typing import Any

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

_configured = False


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def set_trace_id(trace_id: str | None = None) -> str:
    tid = trace_id or new_trace_id()
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id.get()


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = _trace_id.get()
        return super().format(record)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": _trace_id.get(),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    """幂等初始化。Uvicorn / Streamlit 都会调用它。"""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            TextFormatter(
                "%(asctime)s %(levelname)-7s [%(trace_id)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # 第三方库降噪
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "jieba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"docqa.{name}")


class Timer:
    """毫秒计时器，用来做各阶段耗时统计。"""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> float:
        elapsed = (time.perf_counter() - self._t0) * 1000
        self.marks[name] = round(elapsed, 2)
        return self.marks[name]

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)
