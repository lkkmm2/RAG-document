"""文档加载：把 PDF / TXT / Markdown 变成「按页的纯文本」。

设计取舍（面试会问"你为什么不用 XX"）：
- 文本主引擎用 pypdf：速度快，线性扫描，适合几百页的数字化 PDF；
- 表格走 pdfplumber，但做成开关（默认关）：它按坐标布局解析，一页要 1~2 秒，
  几百页会从 10 秒涨到几分钟，收益只在"文档里真有表格"时才明显；
- 不做 OCR：扫描件只做检测 + 告警，把"该用 PaddleOCR/多模态"这件事交给后续迭代，
  比硬塞一个识别率不稳定的 OCR 更诚实。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from docqa.logging_setup import get_logger
from docqa.schemas import PageText

log = get_logger("ingest.loader")

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
TXT_PAGE_CHARS = 2000


@dataclass
class LoadedDocument:
    doc_name: str
    pages: list[PageText]
    tables: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


def load_document(path: Path, *, max_pages: int = 400, extract_tables: bool = False) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix}，仅支持 {sorted(SUPPORTED_SUFFIXES)}")

    if suffix == ".pdf":
        return _load_pdf(path, max_pages=max_pages, extract_tables=extract_tables)
    return _load_text(path)


# ------------------------------------------------------------------ PDF
def _load_pdf(path: Path, *, max_pages: int, extract_tables: bool) -> LoadedDocument:
    from pypdf import PdfReader

    warnings: list[str] = []
    reader = PdfReader(str(path))

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            raise ValueError("PDF 已加密，无法解析") from None

    total_pages = len(reader.pages)
    if total_pages > max_pages:
        warnings.append(f"文档共 {total_pages} 页，超过上限 {max_pages}，只解析前 {max_pages} 页")

    pages: list[PageText] = []
    for idx, page in enumerate(reader.pages[:max_pages]):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - 单页失败不该中断整篇
            log.warning("第 %s 页解析失败：%s", idx + 1, exc)
            text = ""
            warnings.append(f"第 {idx + 1} 页解析失败")
        pages.append(PageText(page=idx + 1, text=text))

    # 扫描件检测：平均每页少于 30 个字符，基本可以判定是图片型 PDF
    avg = sum(len(p.text.strip()) for p in pages) / max(len(pages), 1)
    if avg < 30:
        warnings.append(
            "文档平均每页文字极少（疑似扫描件/图片型 PDF），当前未启用 OCR，检索效果会很差"
        )

    tables = 0
    if extract_tables:
        tables, table_pages, table_warnings = _extract_tables(path, max_pages)
        warnings.extend(table_warnings)
        by_page = {p.page: p for p in pages}
        for page_no, extra in table_pages.items():
            if page_no in by_page:
                by_page[page_no].text = by_page[page_no].text + "\n\n" + extra

    log.info("PDF 解析完成：%s 页，%s 字符，%s 张表", len(pages), sum(len(p.text) for p in pages), tables)
    return LoadedDocument(doc_name=path.name, pages=pages, tables=tables, warnings=warnings)


def _extract_tables(path: Path, max_pages: int) -> tuple[int, dict[int, str], list[str]]:
    """用 pdfplumber 抽表格，转成 Markdown 表格追加到对应页。"""
    warnings: list[str] = []
    by_page: dict[int, list[str]] = {}
    count = 0
    try:
        import pdfplumber
    except ImportError:
        return 0, {}, ["未安装 pdfplumber，已跳过表格抽取"]

    try:
        with pdfplumber.open(str(path)) as pdf:
            for idx, page in enumerate(pdf.pages[:max_pages]):
                for table in page.extract_tables() or []:
                    rows = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in table]
                    rows = [r for r in rows if any(r)]
                    if len(rows) < 2:
                        continue
                    md = _rows_to_markdown(rows)
                    by_page.setdefault(idx + 1, []).append("【表格】\n" + md)
                    count += 1
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"表格抽取失败：{exc}")

    return count, {k: "\n\n".join(v) for k, v in by_page.items()}, warnings


def _rows_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, divider, *body])


# ------------------------------------------------------------------ TXT / MD
def _load_text(path: Path) -> LoadedDocument:
    raw = path.read_bytes()
    text = _decode(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if path.suffix.lower() in {".md", ".markdown"}:
        text = _strip_markdown_noise(text)

    # 纯文本没有页码概念，按固定长度切成"伪页"，让引用至少有个位置感
    pages = [
        PageText(page=i // TXT_PAGE_CHARS + 1, text=text[i : i + TXT_PAGE_CHARS])
        for i in range(0, len(text), TXT_PAGE_CHARS)
    ] or [PageText(page=1, text="")]

    warnings = [] if text.strip() else ["文件内容为空"]
    return LoadedDocument(doc_name=path.name, pages=pages, warnings=warnings)


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _strip_markdown_noise(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)          # 代码块
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)           # 图片
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)       # 链接保留文字
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)  # 标题井号
    return text
