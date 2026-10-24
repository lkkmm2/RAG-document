"""ReAct Agent：思考 → 行动 → 观察，循环到给出答案。

为什么手写而不直接用 LangChain/LangGraph：
1. 依赖少、行为可预测，出问题能直接断点；
2. 面试时"这个循环的终止条件是什么、死循环怎么防"必须能答到代码级，
   用框架反而说不清黑盒里发生了什么；
3. 这套循环换成 LangGraph 只是把 while 换成状态图，概念是一样的。

三个工程细节（也是真实面试追问最多的点）：
- 终止条件不能只有"最大步数"，还要有"无进展检测"；
- 工具失败要回灌给模型，让它换策略，而不是抛异常结束；
- 模型输出的 JSON 必须容错解析，格式错一次不该让整个任务失败。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from docqa.agent.tools import ToolRegistry
from docqa.config import AgentSettings
from docqa.logging_setup import get_logger, get_trace_id
from docqa.schemas import AgentResponse, AgentStep, Citation, Usage

log = get_logger("agent.react")

SYSTEM_PROMPT = """你是一个会使用工具的文档研究助手。你可以调用工具检索知识库，然后基于检索结果回答。

可用工具：
{tools}

输出格式：必须是**合法 JSON 对象**，不要加 Markdown 代码块，不要输出多余文字。
{{"thought": "简短推理（这一步要做什么、为什么）", "action": "工具名", "action_input": {{...}}}}

或者，当信息已经足够时：
{{"thought": "已有足够信息", "action": "final_answer", "action_input": {{"answer": "最终回答，保留 [编号] 引用"}}}}

规则：
1. 每次只输出一个动作；
2. 涉及文档内容的问题，必须先调用 search_docs，不要凭印象回答；
3. 工具返回错误时，换参数或换工具重试，不要重复同样的调用；
4. 上下文里没有的内容，明确说"文档中未提及"，不要编造；
5. 最多 {max_steps} 步，请尽快收敛。"""


class ReActAgent:
    def __init__(self, settings: AgentSettings, llm: Any, tools: ToolRegistry) -> None:
        self.cfg = settings
        self.llm = llm
        self.tools = tools

    # ------------------------------------------------------------------ 主流程
    def run(self, task: str, max_steps: int | None = None) -> AgentResponse:
        started = time.perf_counter()
        max_steps = max(1, min(max_steps or self.cfg.max_steps, 8))
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                tools=self.tools.describe(), max_steps=max_steps)},
            {"role": "user", "content": f"任务：{task}"},
        ]

        steps: list[AgentStep] = []
        citations: list[Citation] = []
        usage = Usage()
        seen_signatures: dict[str, int] = {}
        stopped_reason = "max_steps"
        tool_calls = 0
        answer = ""

        for step_no in range(1, max_steps + 1):
            step_started = time.perf_counter()
            try:
                result = self.llm.chat(messages, temperature=self.cfg.temperature, json_mode=True)
            except Exception as exc:  # noqa: BLE001
                log.error("Agent 第 %s 步模型调用失败：%s", step_no, exc)
                steps.append(AgentStep(step=step_no, error=str(exc)))
                stopped_reason = "llm_error"
                break

            usage = _add_usage(usage, result.usage)
            action = parse_action(result.text)
            step = AgentStep(
                step=step_no,
                thought=str(action.get("thought", ""))[:500],
                action=str(action.get("action", "")),
                action_input=_as_dict(action.get("action_input")),
            )

            if not step.action:
                step.error = "模型输出无法解析为动作"
                steps.append(step)
                messages.append({"role": "assistant", "content": result.text})
                messages.append(
                    {"role": "user", "content": "上一条输出不是合法 JSON，请只输出 JSON 对象。"}
                )
                continue

            if step.action == "final_answer":
                answer = str(step.action_input.get("answer", "")).strip() or result.text.strip()
                step.observation = "（任务结束）"
                step.latency_ms = round((time.perf_counter() - step_started) * 1000, 2)
                steps.append(step)
                stopped_reason = "final_answer"
                break

            # ---- 防重复：同样的工具 + 同样的参数，第二次就熔断 ----
            signature = f"{step.action}:{json.dumps(step.action_input, ensure_ascii=False, sort_keys=True)}"
            seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
            if seen_signatures[signature] > self.cfg.no_progress_threshold:
                step.error = "重复调用被拦截"
                step.observation = "你已经用相同参数调用过这个工具，请换一个思路或直接给出结论。"
                steps.append(step)
                stopped_reason = "no_progress"
                break

            if tool_calls >= self.cfg.max_tool_calls:
                step.error = "工具调用次数超限"
                steps.append(step)
                stopped_reason = "too_many_tool_calls"
                break

            # ---- 执行工具 ----
            tool = self.tools.get(step.action)
            if tool is None:
                step.error = f"不存在的工具：{step.action}"
                observation = f"错误：没有名为 {step.action} 的工具，可用工具：{self.tools.names}"
            else:
                tool_calls += 1
                try:
                    tool_result = tool.run(**step.action_input)
                    observation = tool_result.content
                    if tool_result.citations:
                        observation, new_cites = _register_citations(
                            observation, tool_result.citations, citations
                        )
                        citations.extend(new_cites)
                except TypeError as exc:
                    step.error = f"参数不匹配：{exc}"
                    observation = f"错误：工具参数不正确（{exc}），请检查参数名和类型"
                except Exception as exc:  # noqa: BLE001
                    step.error = str(exc)
                    observation = f"错误：工具执行失败（{exc}），请换个方式再试"

            step.observation = observation[:1200]
            step.latency_ms = round((time.perf_counter() - step_started) * 1000, 2)
            steps.append(step)

            messages.append({"role": "assistant", "content": result.text})
            messages.append(
                {
                    "role": "user",
                    "content": f"观察结果（第 {step_no} 步）：\n{observation[:2500]}\n\n"
                    "请继续：要么调用下一个工具，要么用 final_answer 给出最终回答。",
                }
            )

        if not answer:
            answer = self._fallback_answer(task, steps, messages)
            if stopped_reason == "max_steps":
                stopped_reason = "max_steps_fallback"

        elapsed = round((time.perf_counter() - started) * 1000, 2)
        log.info(
            "Agent 结束：%s 步，%s 次工具调用，原因=%s，%.0fms",
            len(steps), tool_calls, stopped_reason, elapsed,
        )
        return AgentResponse(
            answer=answer,
            steps=steps,
            citations=citations,
            usage=usage,
            latency_ms=elapsed,
            trace_id=get_trace_id(),
            stopped_reason=stopped_reason,
        )

    # ------------------------------------------------------------------ 兜底
    def _fallback_answer(self, task: str, steps: list[AgentStep], messages: list[dict[str, str]]) -> str:
        """步数用完还没给结论时，让模型基于已有观察直接收尾。"""
        observations = "\n\n".join(
            f"第{s.step}步 {s.action} 的观察：\n{s.observation}" for s in steps if s.observation
        )
        if not observations.strip():
            return "抱歉，没能完成这个任务（没有获得有效的工具结果）。"
        try:
            result = self.llm.chat(
                [
                    {"role": "system", "content": "你是文档助手，请基于已有观察直接给出结论。"},
                    {
                        "role": "user",
                        "content": f"任务：{task}\n\n已有观察：\n{observations[:4000]}\n\n"
                        "请给出最终回答；信息不足就说明缺什么。保留 [编号] 引用。",
                    },
                ],
                temperature=0.0,
            )
            return result.text.strip()
        except Exception as exc:  # noqa: BLE001
            log.error("兜底总结失败：%s", exc)
            return f"任务未完成（{exc}）。已获得的信息：\n{observations[:800]}"


# ------------------------------------------------------------------ 工具函数
def parse_action(text: str) -> dict[str, Any]:
    """容错解析模型输出：直接 JSON → 去代码块 → 抓第一个平衡括号。"""
    text = (text or "").strip()
    if not text:
        return {}
    for candidate in (text, _strip_code_fence(text), _first_json_object(text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    return match.group(1).strip() if match else ""


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _add_usage(total: Usage, delta: Usage) -> Usage:
    return Usage(
        prompt_tokens=total.prompt_tokens + delta.prompt_tokens,
        completion_tokens=total.completion_tokens + delta.completion_tokens,
        total_tokens=total.total_tokens + delta.total_tokens,
    )


def _register_citations(
    observation: str, new_cites: list[Citation], existing: list[Citation]
) -> tuple[str, list[Citation]]:
    """把每次工具调用内部从 1 开始的编号，改成全局连续编号。

    否则第 2 次 search_docs 返回的 [1] 会和第 1 次的 [1] 撞车，用户点开引用就对不上。
    """
    offset = len(existing)
    mapping = {c.index: offset + i + 1 for i, c in enumerate(new_cites)}
    for cite in new_cites:
        cite.index = mapping[cite.index]

    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return f"[{mapping.get(idx, idx)}]"

    return re.sub(r"\[(\d+)\]", repl, observation), new_cites
