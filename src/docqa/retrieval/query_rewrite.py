"""多轮对话下的查询改写。

问题：用户问"那第二学期呢"，直接把这句话拿去做向量检索，召回质量很差，
因为"那第二学期"里没有可检索的实体。

两种做法：
- 规则法：把上一轮问题拼进来。简单，但会引入噪声，也解决不了指代；
- LLM 法：让模型把"带指代的追问"改写成"独立可检索的问题"。

这里用 LLM 法，但加了两道闸：
1. 只有出现指代词、或者问题很短时才触发，省一次调用；
2. 改写失败/超时直接退回原问题——改写是优化，不能变成单点故障。
"""

from __future__ import annotations

import re
from typing import Any

from docqa.logging_setup import get_logger
from docqa.schemas import Message

log = get_logger("retrieval.rewrite")

PRONOUNS = (
    "它", "他", "她", "这个", "那个", "这些", "那些", "该", "此", "其",
    "上述", "上面", "前面", "刚才", "其中", "这条", "那条", "这", "那",
)

REWRITE_PROMPT = """你是检索查询改写器。根据对话历史，把用户最后一句问题改写成**一个独立、完整、适合检索的问题**。

要求：
1. 补全所有指代（它/这个/上述/该…），替换成历史里的具体主体；
2. 保留原问题的意图和限定条件，不要增加历史里没有的信息；
3. 只输出改写后的问题本身，不要解释、不要引号、不要编号；
4. 如果原问题已经独立完整，原样输出。

对话历史：
{history}

用户最后一句：{question}

改写结果："""


class QueryRewriter:
    def __init__(self, llm: Any, *, enabled: bool = True, history_turns: int = 4) -> None:
        self.llm = llm
        self.enabled = enabled
        self.history_turns = history_turns
        self._cache: dict[str, str] = {}

    def needs_rewrite(self, question: str, history: list[Message]) -> bool:
        if not self.enabled or not history:
            return False
        text = question.strip()
        if len(text) <= 12:
            return True
        return any(word in text for word in PRONOUNS)

    def rewrite(self, question: str, history: list[Message] | None = None) -> str:
        history = history or []
        if not self.needs_rewrite(question, history):
            return question

        key = question + "||" + "|".join(m.content for m in history[-self.history_turns :])
        if key in self._cache:
            return self._cache[key]

        history_text = "\n".join(
            f"{'用户' if m.role == 'user' else '助手'}：{m.content[:200]}"
            for m in history[-self.history_turns :]
        )
        prompt = REWRITE_PROMPT.format(history=history_text, question=question)
        try:
            result = self.llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=120
            )
            rewritten = _clean(result.text)
        except Exception as exc:  # noqa: BLE001 - 改写失败退回原问题
            log.warning("查询改写失败，使用原问题：%s", exc)
            return question

        if not rewritten or len(rewritten) > 200:
            return question
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = rewritten
        log.info("查询改写：%r -> %r", question, rewritten)
        return rewritten


def _clean(text: str) -> str:
    text = text.strip().split("\n")[0]
    text = re.sub(r"^(改写结果|结果|问题)[:：]\s*", "", text)
    return text.strip().strip("\"“”'‘’ ")
