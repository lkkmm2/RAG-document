"""大模型客户端。

抽象成接口的意义：换模型（DeepSeek / Qwen / 本地 vLLM）只改一个类；
测试用 Mock 就不需要网络和 Key；
Agent 和 RAG 共用同一套调用与用量统计。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from docqa.config import LLMSettings
from docqa.errors import LLMError
from docqa.logging_setup import get_logger
from docqa.schemas import Usage

log = get_logger("llm")


@dataclass
class LLMResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0


class LLMClient(Protocol):
    name: str
    model: str

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult: ...

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]: ...


# ------------------------------------------------------------------ DeepSeek
class DeepSeekClient:
    """OpenAI 兼容接口，所以同一个类也能接 Qwen/智谱/本地 vLLM，改 base_url 即可。"""

    def __init__(self, cfg: LLMSettings) -> None:
        from openai import OpenAI

        self.cfg = cfg
        self.name = "deepseek"
        self.model = cfg.model
        self._client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
            max_retries=2,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            resp = self._call(kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"大模型调用失败：{exc}") from exc

        text = resp.choices[0].message.content or ""
        usage = Usage()
        if getattr(resp, "usage", None):
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
                total_tokens=resp.usage.total_tokens or 0,
            )
        return LLMResult(
            text=text.strip(),
            usage=usage,
            model=self.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _call(self, kwargs: dict[str, Any]):
        try:
            return self._client.chat.completions.create(**kwargs)
        except TypeError:
            # 新版本 SDK 可能要求 max_completion_tokens，做一次兼容重试
            if "max_tokens" not in kwargs:
                raise
            kwargs = dict(kwargs)
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            return self._client.chat.completions.create(**kwargs)

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.cfg.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.cfg.max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"流式调用失败：{exc}") from exc


# ------------------------------------------------------------------ Mock
class MockClient:
    """无 Key 时的降级实现：不联网，但从检索结果里"摘抄"答案，保证链路完整。"""

    name = "mock"
    model = "mock-echo"

    def __init__(self, cfg: LLMSettings | None = None) -> None:
        self.cfg = cfg

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,  # noqa: ARG002
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        prompt = messages[-1]["content"] if messages else ""
        text = (
            self._compose_agent(messages)
            if json_mode
            else self._compose(prompt)
        )
        return LLMResult(
            text=text,
            usage=self._estimate(prompt, text),
            model=self.model,
            latency_ms=5.0,
        )

    def _compose_agent(self, messages: list[dict[str, str]]) -> str:
        """模拟一次 ReAct：第一轮调 search_docs，第二轮给 final_answer。

        这样即使没有 API Key，Agent 的循环、工具调用、观察回灌也能完整演示出来。
        """
        joined = "\n".join(m.get("content", "") for m in messages)
        task = next(
            (m["content"] for m in messages if m.get("content", "").startswith("任务：")),
            "",
        )
        if "观察结果" not in joined:
            query = task.replace("任务：", "").strip()[:60] or "文档内容"
            return json.dumps(
                {
                    "thought": "先检索知识库里有没有相关内容",
                    "action": "search_docs",
                    "action_input": {"query": query, "top_k": 3},
                },
                ensure_ascii=False,
            )
        observation = joined.rsplit("观察结果", 1)[-1]
        snippet = re.sub(r"\s+", " ", observation)[:180].strip("：: ")
        return json.dumps(
            {
                "thought": "已有检索结果，可以作答",
                "action": "final_answer",
                "action_input": {
                    "answer": f"（Mock 模式，未配置 API Key）根据检索结果：{snippet} [1]"
                },
            },
            ensure_ascii=False,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> Iterator[str]:
        text = self._compose(messages[-1]["content"] if messages else "")
        for i in range(0, len(text), 12):
            yield text[i : i + 12]

    # -- 内部 --
    def _compose(self, prompt: str) -> str:
        blocks = re.findall(r"\[(\d+)\]\s*([^\n]*)\n(.{0,200})", prompt, flags=re.S)
        if not blocks:
            return "（Mock 模式）文档中未提及。请在 .env 里配置 DOCQA_LLM__API_KEY 后重试。"
        lines = ["（Mock 模式，未配置 API Key，以下内容直接摘自检索结果）", ""]
        for idx, meta, body in blocks[:2]:
            snippet = re.sub(r"\s+", " ", body).strip()
            lines.append(f"- {snippet[:120]} [{idx}]（{meta.strip()}）")
        return "\n".join(lines)

    @staticmethod
    def _estimate(prompt: str, answer: str) -> Usage:
        p = max(1, len(prompt) // 3)
        c = max(1, len(answer) // 3)
        return Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


def build_llm(cfg: LLMSettings) -> LLMClient:
    if cfg.provider == "mock" or not cfg.api_key:
        if cfg.provider != "mock":
            log.warning("未配置 API Key，自动降级为 Mock 模式（回答不可用于演示）")
        return MockClient(cfg)
    return DeepSeekClient(cfg)
