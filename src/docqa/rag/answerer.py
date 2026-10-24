"""问答编排：检索 → Prompt → 生成 → 引用校验。

两个细节值得在面试里主动说：
1. 检索为空时**不调用大模型**，直接返回"未提及"。既省钱又避免模型开始编。
2. 生成完一定要回填 citations 并校验编号：模型偶尔会引用 [7]，
   但上下文只有 4 条——这种"引用了不存在的来源"是最典型的幻觉信号。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from docqa.config import Settings
from docqa.logging_setup import get_logger, get_trace_id
from docqa.rag.prompts import (
    NO_ANSWER_HINT,
    SYSTEM_PROMPT,
    build_citations,
    build_user_prompt,
    validate_citations,
)
from docqa.retrieval.pipeline import RetrievalPipeline
from docqa.schemas import AskRequest, AskResponse, CitationStats, Message

log = get_logger("rag.answerer")


class RAGAnswerer:
    def __init__(self, settings: Settings, llm: Any, retriever: RetrievalPipeline) -> None:
        self.settings = settings
        self.llm = llm
        self.retriever = retriever

    # ------------------------------------------------------------------ 非流式
    def answer(self, request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        retrieval = self.retriever.retrieve(
            request.question,
            history=request.history,
            top_k=request.top_k,
            debug=request.debug,
        )
        contexts = retrieval.contexts

        if not contexts:
            return AskResponse(
                answer=f"{NO_ANSWER_HINT}：当前知识库里没有与这个问题相关的内容。",
                citations=[],
                citation_stats=CitationStats(),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                trace_id=get_trace_id(),
                debug=retrieval.debug,
            )

        messages = self._build_messages(request.question, contexts, request.history)
        result = self.llm.chat(messages)
        answer = result.text or NO_ANSWER_HINT
        stats = validate_citations(answer, contexts)

        if stats.invalid:
            log.warning("模型引用了不存在的编号：%s", stats.invalid)

        return AskResponse(
            answer=answer,
            citations=build_citations(contexts),
            citation_stats=stats,
            usage=result.usage,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            trace_id=get_trace_id(),
            debug=retrieval.debug,
        )

    # ------------------------------------------------------------------ 流式
    def stream(
        self,
        question: str,
        history: list[Message] | None = None,
        top_k: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        retrieval = self.retriever.retrieve(question, history=history, top_k=top_k)
        contexts = retrieval.contexts
        citations = build_citations(contexts)

        yield {
            "type": "retrieval",
            "citations": [c.model_dump() for c in citations],
            "contexts": len(contexts),
            "timings_ms": (retrieval.debug.timings_ms if retrieval.debug else {}),
        }

        if not contexts:
            text = f"{NO_ANSWER_HINT}：当前知识库里没有与这个问题相关的内容。"
            yield {"type": "delta", "content": text}
            yield {
                "type": "done",
                "answer": text,
                "citation_stats": CitationStats().model_dump(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            return

        messages = self._build_messages(question, contexts, history or [])
        buffer: list[str] = []
        try:
            for token in self.llm.stream(messages):
                buffer.append(token)
                yield {"type": "delta", "content": token}
        except Exception as exc:  # noqa: BLE001
            log.error("流式生成中断：%s", exc)
            yield {"type": "error", "message": str(exc)}

        answer = "".join(buffer).strip()
        stats = validate_citations(answer, contexts)
        yield {
            "type": "done",
            "answer": answer,
            "citation_stats": stats.model_dump(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    # ------------------------------------------------------------------ 内部
    def _build_messages(
        self, question: str, contexts, history: list[Message]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history[-self.settings.retrieval.rewrite_history_turns :]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append(
            {
                "role": "user",
                "content": build_user_prompt(
                    question, contexts, max_chars=self.settings.retrieval.max_context_chars
                ),
            }
        )
        return messages
