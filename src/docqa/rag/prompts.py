"""Prompt 与引用校验。

引用不是"排版好看"，它是三个东西：
1. 用户体验：让用户能点开原文核对，而不是盲信模型；
2. 幻觉检测：模型编的编号对不上上下文，说明它在瞎编，可以直接兜住；
3. 可评测：citation_precision 是评测里唯一能自动算的"可信度"指标。
"""

from __future__ import annotations

import re

from docqa.schemas import Citation, CitationStats, ContextItem

SYSTEM_PROMPT = """你是 DocQA，一个严格基于给定文档回答问题的助手。

规则：
1. 只能使用【参考资料】中的信息作答，不得使用你自己的先验知识补充。
2. 每个结论后面用方括号编号标注来源，例如 [1]、[2][3]；编号必须来自参考资料。
3. 如果参考资料不足以回答，直接回答"文档中未提及"，并说明缺少哪方面信息；不要猜测。
4. 不要编造参考资料里没有的数字、日期、条款号、人名。
5. 用中文作答，先给结论，再给依据，简洁清楚。"""

NO_ANSWER_HINT = "文档中未提及"

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,，、]\s*\d+)*)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")


def build_user_prompt(question: str, contexts: list[ContextItem], max_chars: int = 4000) -> str:
    blocks: list[str] = []
    used = 0
    for item in contexts:
        meta = f"《{item.doc_name}》"
        if item.section:
            meta += f" · {item.section}"
        if item.page:
            meta += f" · 第{item.page}页"
        body = item.text.strip()
        block = f"[{item.index}] {meta}\n{body}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    joined = "\n\n".join(blocks) if blocks else "（无检索结果）"
    return f"【参考资料】\n{joined}\n\n【问题】\n{question}\n\n请依据参考资料作答，并在句末标注引用编号。"


def extract_citation_indexes(answer: str) -> list[int]:
    indexes: list[int] = []
    for match in _CITE_RE.finditer(answer):
        for part in re.split(r"[\s,，、]+", match.group(1)):
            if part.isdigit():
                indexes.append(int(part))
    return indexes


def validate_citations(answer: str, contexts: list[ContextItem]) -> CitationStats:
    valid_range = {item.index for item in contexts}
    used = extract_citation_indexes(answer)
    invalid = sorted({idx for idx in used if idx not in valid_range})

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s and len(s.strip()) > 4]
    cited = sum(1 for s in sentences if extract_citation_indexes(s))
    coverage = round(cited / len(sentences), 4) if sentences else 0.0

    return CitationStats(
        total=len(set(used)),
        invalid=invalid,
        coverage=coverage,
        valid=not invalid,
    )


def build_citations(contexts: list[ContextItem]) -> list[Citation]:
    citations: list[Citation] = []
    for item in contexts:
        snippet = re.sub(r"\s+", " ", item.text).strip()
        citations.append(
            Citation(
                index=item.index,
                chunk_id=item.chunk_id,
                doc_name=item.doc_name,
                section=item.section,
                page=item.page,
                snippet=snippet[:220],
                score=item.score,
            )
        )
    return citations
