"""Agent 层测试：解析容错、安全计算器、循环终止。"""

from __future__ import annotations

import pytest

from docqa.agent.react import ReActAgent, parse_action
from docqa.agent.tools import Tool, ToolRegistry, ToolResult, _safe_eval, build_tools
from docqa.config import AgentSettings
from docqa.schemas import Usage


def test_parse_plain_json():
    action = parse_action(
        '{"thought": "查一下", "action": "search_docs", "action_input": {"query": "学籍"}}'
    )
    assert action["action"] == "search_docs"
    assert action["action_input"]["query"] == "学籍"


def test_parse_json_in_code_fence():
    text = '好的，我来检索：\n```json\n{"thought": "t", "action": "list_documents", "action_input": {}}\n```'
    assert parse_action(text)["action"] == "list_documents"


def test_parse_json_with_surrounding_prose():
    text = '我先想想。{"thought": "t", "action": "final_answer", "action_input": {"answer": "42"}} 完毕。'
    action = parse_action(text)
    assert action["action"] == "final_answer"
    assert action["action_input"]["answer"] == "42"


def test_parse_garbage_returns_empty():
    assert parse_action("我无法回答这个问题") == {}
    assert parse_action("") == {}


def test_safe_eval_allows_arithmetic():
    assert _safe_eval("(85+92)/2") == 88.5
    assert _safe_eval("2 ** 3") == 8


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('dir')", "open('a.txt')", "9 ** 999", "[1,2,3]", "a + 1"],
)
def test_safe_eval_blocks_dangerous_input(expression):
    with pytest.raises(Exception):  # noqa: B017 - 拒绝执行才是重点，异常类型不重要
        _safe_eval(expression)


def test_agent_finishes_with_final_answer(services, ingested):  # noqa: ARG001
    tools = build_tools(services.retriever, services.repo)
    agent = ReActAgent(services.settings.agent, services.llm, tools)
    response = agent.run("休学最长可以多久？", max_steps=2)

    assert response.answer
    assert response.steps
    assert response.stopped_reason == "final_answer"
    assert response.trace_id


class _LoopingLLM:
    """永远要求调用工具的假模型，用来验证终止条件能兜住死循环。"""

    name = "looping"
    model = "looping"

    def chat(self, messages, **kwargs):  # noqa: ARG002
        class _Result:
            text = '{"thought": "再查一次", "action": "list_documents", "action_input": {}}'
            usage = Usage()

        return _Result()

    def stream(self, messages, **kwargs):  # noqa: ARG002
        yield ""


def test_agent_stops_when_steps_exhausted():
    registry = ToolRegistry(
        [
            Tool(
                name="list_documents",
                description="d",
                parameters={},
                func=lambda: ToolResult(content="空"),
            )
        ]
    )
    agent = ReActAgent(
        AgentSettings(max_steps=2, no_progress_threshold=2), _LoopingLLM(), registry
    )
    response = agent.run("测试", max_steps=2)

    assert len(response.steps) <= 2
    assert response.stopped_reason in {"no_progress", "max_steps", "max_steps_fallback"}
