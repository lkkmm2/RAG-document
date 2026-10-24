"""文本清洗：页眉页脚、页码、断行。

这一步经常被跳过，但它是 RAG 效果的地基：
学生手册、制度文件这类 PDF，每页顶上都印着"XX大学学生手册"，
如果不删，这些字会被切进每个 chunk，把向量"稀释"成平均语义，
检索时任何问题都能匹配到每一页——召回看起来很高，精度全废。
"""

from __future__ import annotations

import re
from collections import Counter

from docqa.ingest.headings import HEADING_START_RE
from docqa.logging_setup import get_logger
from docqa.schemas import PageText

log = get_logger("ingest.cleaner")

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)

# 纯页码：1 / -1- / — 12 — / 第 3 页 / 3/20
PAGE_NO_RE = re.compile(r"^\s*(?:[-—–]\s*)?(?:第\s*)?\d{1,4}\s*(?:页)?(?:\s*/\s*\d{1,4})?\s*(?:[-—–])?\s*$")
# 只有符号的行
SYMBOL_ONLY_RE = re.compile(r"^[\s\-—_=*·•.,，。、:：;；|/\\~`'\"()（）\[\]【】<>《》]+$")
# 句子结尾（用来判断能不能跟下一行拼接）
SENTENCE_END = "。！？；：.!?;:”』】）)」…"


def clean_pages(
    pages: list[PageText],
    *,
    dedupe_header_footer: bool = True,
    header_footer_ratio: float = 0.6,
) -> list[PageText]:
    normalized = [PageText(page=p.page, text=_normalize(p.text)) for p in pages]

    if dedupe_header_footer and len(normalized) >= 4:
        drop = _detect_repeating_lines(normalized, header_footer_ratio)
        if drop:
            normalized = [
                PageText(page=p.page, text=_drop_lines(p.text, drop)) for p in normalized
            ]
            log.info("已移除疑似页眉页脚 %s 种：%s", len(drop), list(drop)[:3])

    return [PageText(page=p.page, text=_join_broken_lines(p.text)) for p in normalized]


def _normalize(text: str) -> str:
    text = text.translate(ZERO_WIDTH)
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _detect_repeating_lines(pages: list[PageText], ratio: float) -> set[str]:
    """在「每页前两行 + 后两行」里找跨页高频重复的行。"""
    counter: Counter[str] = Counter()
    for page in pages:
        lines = [line for line in page.text.split("\n") if line.strip()]
        if not lines:
            continue
        for line in set(lines[:2] + lines[-2:]):
            if 2 <= len(line) <= 40:
                counter[line] += 1
    threshold = max(3, int(len(pages) * ratio))
    return {line for line, count in counter.items() if count >= threshold}


def _drop_lines(text: str, drop: set[str]) -> str:
    kept = [line for line in text.split("\n") if line.strip() not in drop]
    return "\n".join(kept)


def _join_broken_lines(text: str) -> str:
    """把 PDF 抽出来的硬换行接回完整句子。

    规则：上一行不是标题、不以句末标点结尾、长度足够；下一行也不是标题。
    满足就把两行拼起来（中文不加空格）。
    """
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if PAGE_NO_RE.match(stripped) or SYMBOL_ONLY_RE.match(stripped):
            continue
        if not out:
            out.append(stripped)
            continue
        prev = out[-1]
        if (
            prev
            and not prev.endswith(tuple(SENTENCE_END))
            and not HEADING_START_RE.match(prev)
            and not HEADING_START_RE.match(stripped)
            and not prev.startswith("#")
            and len(prev) >= 8
        ):
            joiner = " " if (prev[-1].isascii() and stripped[0].isascii()) else ""
            out[-1] = prev + joiner + stripped
        else:
            out.append(stripped)
    return "\n".join(out)
