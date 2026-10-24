"""结构化切分：标题→章节→子块/父块。

为什么不用 `text[start:start+400]` 这种定长切分：
- 会切断句子，检索命中半句话，模型只能靠猜；
- 会跨越章节，一个 chunk 里混进两章内容，向量被稀释；
- 丢了结构，回答时无法告诉用户"这在第 3 章第 2 节"。

这里的策略是「小块检索、大块生成」：
- 子块（chunk_size≈400）负责被检索，粒度细，召回准；
- 父块（parent_size≈1200）负责被喂给模型，上下文完整，不容易断章取义；
- 检索命中子块后回溯父块，兼顾精度和上下文完整性。

父块是**跨章节**聚合的：制度类文档一章只有一两百字，如果父块也按章切，
父块就和子块一样大，"小块检索大块生成"就退化成了普通切分。
子块仍然只在自己的章节内，保证引用能准确指到"第几章第几节"。
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

from docqa.config import ChunkingSettings
from docqa.ingest.headings import match_heading
from docqa.logging_setup import get_logger
from docqa.schemas import Chunk, PageText, ParentChunk

log = get_logger("ingest.splitter")

_CJK_OR_ALPHA = re.compile(r"[\u4e00-\u9fffA-Za-z]")

# 合并之后仍然短于这个长度的块，基本是页眉/符号残留，直接丢掉
HARD_MIN_CHARS = 8


class PageMap:
    """字符偏移 → 页码。切分跨页时靠它算引用页码。"""

    def __init__(self) -> None:
        self._starts: list[int] = []
        self._pages: list[int] = []

    def add(self, page: int, start: int) -> None:
        self._starts.append(start)
        self._pages.append(page)

    def page_at(self, offset: int) -> int:
        if not self._starts:
            return 0
        idx = bisect_right(self._starts, offset) - 1
        return self._pages[max(idx, 0)]


@dataclass
class SplitResult:
    chunks: list[Chunk] = field(default_factory=list)
    parents: dict[str, ParentChunk] = field(default_factory=dict)
    page_map: PageMap = field(default_factory=PageMap)

    @property
    def chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)


@dataclass
class _Block:
    section: str
    start: int
    end: int


@dataclass
class _Span:
    start: int
    end: int
    section: str


class DocumentSplitter:
    def __init__(self, settings: ChunkingSettings, min_chars: int | None = None) -> None:
        self.cfg = settings
        # min_chars 是"合并阈值"：太短的块不丢，并到相邻块里去
        self.min_chars = min_chars or settings.min_chars
        self.separators = [s for s in settings.separators if s != ""]

    # ------------------------------------------------------------------ 入口
    def split(self, doc_id: str, doc_name: str, pages: list[PageText]) -> SplitResult:
        text, page_map = self._join_pages(pages)
        blocks = self._split_sections(text)
        result = SplitResult(page_map=page_map)
        if not blocks:
            return result

        # 1) 章节内部切分：保证子块不跨章，引用能指到"第几章第几节"
        spans: list[_Span] = []
        for block in blocks:
            block_text = text[block.start : block.end]
            if not block_text.strip():
                continue
            # 只有一行标题、没有正文的块（章标题后紧跟节标题）不单独成块：
            # 它的文字已经进了子块的 section 路径，留着只会产生 8 个字的孤儿块
            body_lines = [line for line in block_text.strip().split("\n") if line.strip()]
            if len(body_lines) == 1 and match_heading(body_lines[0]):
                continue
            for start, end in self._pack(self._atomic_spans(block_text, block.start)):
                spans.append(_Span(start, end, block.section))

        # 2) 碎片合并：几十个字的块单独成块只会污染召回
        spans = self._merge_short(spans, text)

        # 3) 生成子块
        child_idx = 0
        for span in spans:
            chunk_text = text[span.start : span.end].strip()
            if len(chunk_text) < HARD_MIN_CHARS or not self._has_letters(chunk_text):
                continue
            result.chunks.append(
                Chunk(
                    id=f"{doc_id}-{child_idx:05d}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    text=chunk_text,
                    parent_id="",
                    section=span.section,
                    page=page_map.page_at(span.start),
                    order=child_idx,
                    char_start=span.start,
                    char_end=span.end,
                )
            )
            child_idx += 1

        # 4) 父块：跨章聚合，让"大块生成"名副其实
        parent_idx = 0
        for group in self._group_for_parents(result.chunks):
            if not group:
                continue
            parent_id = f"{doc_id}-P{parent_idx:04d}"
            p_start, p_end = group[0].char_start, group[-1].char_end
            result.parents[parent_id] = ParentChunk(
                id=parent_id,
                doc_id=doc_id,
                doc_name=doc_name,
                text=text[p_start:p_end].strip(),
                section=group[0].section,
                page=group[0].page,
            )
            for chunk in group:
                chunk.parent_id = parent_id
            parent_idx += 1

        log.info(
            "切分完成：%s 个章节块，%s 个子块（平均 %.0f 字），%s 个父块（平均 %.0f 字）",
            len(blocks),
            len(result.chunks),
            (result.chars / len(result.chunks)) if result.chunks else 0,
            len(result.parents),
            (
                sum(len(p.text) for p in result.parents.values()) / len(result.parents)
                if result.parents
                else 0
            ),
        )
        return result

    # ------------------------------------------------------------------ 拼接
    def _join_pages(self, pages: list[PageText]) -> tuple[str, PageMap]:
        page_map = PageMap()
        parts: list[str] = []
        cursor = 0
        for page in pages:
            page_map.add(page.page, cursor)
            parts.append(page.text)
            cursor += len(page.text) + 2  # "\n\n"
        return "\n\n".join(parts), page_map

    # ------------------------------------------------------------------ 章节
    def _split_sections(self, text: str) -> list[_Block]:
        blocks: list[_Block] = []
        stack: list[tuple[int, str]] = []
        cur_section = ""
        cur_start = 0
        offset = 0

        for line in text.split("\n"):
            line_start = offset
            offset += len(line) + 1
            heading = match_heading(line)
            if heading is None:
                continue
            level, title = heading
            if line_start > cur_start:
                blocks.append(_Block(cur_section, cur_start, line_start))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_section = " > ".join(t for _, t in stack)
            cur_start = line_start

        if cur_start < len(text):
            blocks.append(_Block(cur_section, cur_start, len(text)))
        return blocks

    # ------------------------------------------------------------------ 原子切分
    def _atomic_spans(
        self, text: str, base: int, seps: list[str] | None = None
    ) -> list[tuple[int, int]]:
        """按分隔符优先级递归切成不超过 chunk_size 的「原子片段」。"""
        seps = self.separators if seps is None else seps
        size = self.cfg.chunk_size
        if len(text) <= size:
            return [(base, base + len(text))] if text.strip() else []

        for idx, sep in enumerate(seps):
            if sep not in text:
                continue
            spans: list[tuple[int, int]] = []
            start = 0
            while True:
                pos = text.find(sep, start)
                if pos < 0:
                    spans.append((start, len(text)))
                    break
                spans.append((start, pos + len(sep)))
                start = pos + len(sep)

            out: list[tuple[int, int]] = []
            for s, e in spans:
                piece = text[s:e]
                if not piece.strip():
                    continue
                if len(piece) <= size:
                    out.append((base + s, base + e))
                else:
                    out.extend(self._atomic_spans(piece, base + s, seps[idx + 1 :]))
            return out

        # 没有任何分隔符可用：硬切兜底
        return [
            (base + i, min(base + i + size, base + len(text)))
            for i in range(0, len(text), size)
        ]

    # ------------------------------------------------------------------ 打包
    def _pack(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """把原子片段贪心装进 chunk_size 的窗口，窗口之间保留 overlap。"""
        if not spans:
            return []
        size = self.cfg.chunk_size
        overlap = max(0, min(self.cfg.chunk_overlap, size // 2))
        windows: list[tuple[int, int]] = []
        i = 0
        while i < len(spans):
            start = spans[i][0]
            end = spans[i][1]
            j = i
            while j + 1 < len(spans) and spans[j + 1][1] - start <= size:
                j += 1
                end = spans[j][1]
            windows.append((start, end))
            if j + 1 >= len(spans):
                break
            target = end - overlap
            k = i
            while k <= j and spans[k][0] < target:
                k += 1
            i = min(max(k, i + 1), j + 1)
        return windows

    # ------------------------------------------------------------------ 碎片合并
    def _merge_short(self, spans: list[_Span], text: str) -> list[_Span]:
        """过短的窗口不丢弃，并进前一个窗口（跨章也并，碎片留着更伤召回）。"""
        if not spans:
            return []
        merged: list[_Span] = [spans[0]]
        for span in spans[1:]:
            previous = merged[-1]
            if len(text[previous.start : previous.end].strip()) < self.min_chars:
                previous.end = span.end
            else:
                merged.append(span)
        if len(merged) > 1 and len(text[merged[-1].start : merged[-1].end].strip()) < self.min_chars:
            merged[-2].end = merged[-1].end
            merged.pop()
        return merged

    # ------------------------------------------------------------------ 父块分组
    def _group_for_parents(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        target = self.cfg.parent_size
        groups: list[list[Chunk]] = []
        current: list[Chunk] = []
        size = 0
        for chunk in chunks:
            current.append(chunk)
            size += len(chunk.text)
            if size >= target:
                groups.append(current)
                current = []
                size = 0
        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------------ 质量过滤
    def _has_letters(self, text: str) -> bool:
        letters = len(_CJK_OR_ALPHA.findall(text))
        return letters / max(len(text), 1) >= 0.3
