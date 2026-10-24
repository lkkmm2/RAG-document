"""Agent 工具集。

工具设计的三个要点（面试常问"怎么提升工具调用准确率"）：
1. description 写"什么时候用它"，而不是"它是什么"；
2. 参数尽量少、有默认值、有取值范围，模型更容易填对；
3. 返回值要结构化、可截断——工具输出太长会直接把上下文顶爆。
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from docqa.retrieval.pipeline import RetrievalPipeline
from docqa.schemas import Citation
from docqa.store.chunk_repo import ChunkRepository


@dataclass
class ToolResult:
    content: str
    citations: list[Citation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    func: Callable[..., ToolResult]

    def run(self, **kwargs: Any) -> ToolResult:
        return self.func(**kwargs)

    def describe(self) -> str:
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"- {self.name}({params}): {self.description}"


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe(self) -> str:
        return "\n".join(tool.describe() for tool in self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools)


# ------------------------------------------------------------------ 工具实现
def build_tools(retriever: RetrievalPipeline, repo: ChunkRepository) -> ToolRegistry:
    def search_docs(query: str, top_k: int = 4) -> ToolResult:
        top_k = max(1, min(int(top_k or 4), 8))
        result = retriever.retrieve(query, top_k=top_k)
        if not result.contexts:
            return ToolResult(content="（没有检索到相关内容，可以换个说法再试）")
        lines: list[str] = []
        citations: list[Citation] = []
        for item in result.contexts:
            head = f"[{item.index}] 《{item.doc_name}》"
            if item.section:
                head += f" · {item.section}"
            if item.page:
                head += f" · 第{item.page}页"
            lines.append(f"{head}\n{item.text[:800]}")
            citations.append(
                Citation(
                    index=item.index,
                    chunk_id=item.chunk_id,
                    doc_name=item.doc_name,
                    section=item.section,
                    page=item.page,
                    snippet=item.text[:220],
                    score=item.score,
                )
            )
        return ToolResult(content="\n\n".join(lines), citations=citations)

    def list_documents() -> ToolResult:
        docs = repo.list_documents()
        if not docs:
            return ToolResult(content="知识库为空。")
        lines = [f"- {d.doc_name}（{d.chunks} 个片段，{d.chars} 字）" for d in docs]
        return ToolResult(content="知识库中的文档：\n" + "\n".join(lines))

    def read_section(keyword: str, max_chars: int = 1500) -> ToolResult:
        parents = repo.load_parents()
        hits = [p for p in parents.values() if keyword and keyword in p.section]
        if not hits:
            return ToolResult(content=f"没有找到标题包含「{keyword}」的章节。")
        max_chars = max(200, min(int(max_chars or 1500), 3000))
        blocks = [
            f"《{p.doc_name}》 · {p.section} · 第{p.page}页\n{p.text[:max_chars]}"
            for p in hits[:2]
        ]
        return ToolResult(content="\n\n".join(blocks))

    def calculator(expression: str) -> ToolResult:
        try:
            value = _safe_eval(expression)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"计算失败：{exc}（只支持 + - * / // % ** 和括号）")
        return ToolResult(content=f"{expression} = {value}")

    return ToolRegistry(
        [
            Tool(
                name="search_docs",
                description="在知识库中检索与问题相关的文档片段。回答任何关于文档内容的问题前，都应该先调用它。",
                parameters={"query": "要检索的问题，尽量具体", "top_k": "返回片段数，1-8，默认 4"},
                func=search_docs,
            ),
            Tool(
                name="list_documents",
                description="列出知识库里有哪些文档。当你不确定该查哪份文档，或用户问'都有什么资料'时使用。",
                parameters={},
                func=list_documents,
            ),
            Tool(
                name="read_section",
                description="按章节标题关键词，直接读取该章节的完整内容。已知章节名（如'学籍异动'）时比 search_docs 更直接。",
                parameters={"keyword": "章节标题里的关键词", "max_chars": "最多返回多少字"},
                func=read_section,
            ),
            Tool(
                name="calculator",
                description="做四则运算。需要算比例、总分、差值时使用，不要自己心算。",
                parameters={"expression": "算术表达式，例如 (85+92)/2"},
                func=calculator,
            ),
        ]
    )


# ------------------------------------------------------------------ 安全计算器
_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> float:
    """只允许数字和四则运算。

    绝对不能用 eval()：大模型生成的字符串是不可信输入，
    一个 `__import__('os').system(...)` 就能把服务器交出去。
    """
    if len(expression) > 100:
        raise ValueError("表达式过长")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("只支持数字")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("指数过大")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError("不支持的一元运算符")
            return op(_eval(node.operand))
        raise ValueError(f"不支持的语法：{type(node).__name__}")

    return _eval(ast.parse(expression, mode="eval"))
