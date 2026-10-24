"""切分器测试：这是全项目最容易出 bug、也最影响效果的一段。"""

from __future__ import annotations

from docqa.config import ChunkingSettings
from docqa.ingest.headings import match_heading
from docqa.ingest.splitter import DocumentSplitter
from docqa.schemas import PageText


def _pages(text: str, page_size: int = 300) -> list[PageText]:
    return [
        PageText(page=i // page_size + 1, text=text[i : i + page_size])
        for i in range(0, len(text), page_size)
    ]


def test_heading_detection():
    assert match_heading("第一章 总则") == (1, "第一章 总则")
    assert match_heading("第二节 学籍异动") == (2, "第二节 学籍异动")
    assert match_heading("一、总则") == (2, "一、总则")
    assert match_heading("（三）考核方式") == (3, "（三）考核方式")
    # 正文不能被误判成标题
    assert match_heading("1. 学生应当在两周内完成注册。") is None
    assert match_heading("第十条 学生可以申请转专业，经学院同意后办理，逾期不予受理。") is None


def test_split_respects_size_and_tracks_section(sample_text: str):
    cfg = ChunkingSettings(chunk_size=120, chunk_overlap=30, parent_size=300, min_chars=10)
    result = DocumentSplitter(cfg).split("doc1", "手册.txt", _pages(sample_text))

    assert result.chunks, "应该切出 chunk"
    assert all(len(c.text) <= cfg.chunk_size for c in result.chunks)
    # 每个 chunk 都能挂到某个章节上，并且父块存在
    assert all(c.parent_id in result.parents for c in result.chunks)
    assert any("学籍管理" in c.section for c in result.chunks)


def test_parent_chunks_cover_children(sample_text: str):
    cfg = ChunkingSettings(chunk_size=100, chunk_overlap=20, parent_size=260, min_chars=10)
    result = DocumentSplitter(cfg).split("doc2", "手册.txt", _pages(sample_text))

    for parent in result.parents.values():
        assert len(parent.text) >= 50
        assert parent.doc_id == "doc2"

    # 子块文本必须能在父块里找到（父子块是包含关系，不是平行关系）
    parents = result.parents
    for chunk in result.chunks[:10]:
        assert chunk.text[:20] in parents[chunk.parent_id].text


def test_page_mapping(sample_text: str):
    cfg = ChunkingSettings(chunk_size=80, chunk_overlap=10, parent_size=200, min_chars=10)
    result = DocumentSplitter(cfg).split("doc3", "手册.txt", _pages(sample_text, page_size=200))
    pages = {c.page for c in result.chunks}
    assert len(pages) > 1, "跨页文档应该映射出多个页码"
    assert min(pages) >= 1


def test_short_document_single_chunk():
    cfg = ChunkingSettings(chunk_size=400, chunk_overlap=50, min_chars=10)
    result = DocumentSplitter(cfg).split("doc4", "短.txt", _pages("这是一段很短的说明文字，用来验证短文档只切一块。"))
    assert len(result.chunks) == 1
    assert result.chunks[0].parent_id in result.parents


def test_ids_are_unique_and_ordered(sample_text: str):
    cfg = ChunkingSettings(chunk_size=100, chunk_overlap=20, min_chars=10)
    result = DocumentSplitter(cfg).split("doc5", "手册.txt", _pages(sample_text))
    ids = [c.id for c in result.chunks]
    assert len(ids) == len(set(ids))
    assert [c.order for c in result.chunks] == sorted(c.order for c in result.chunks)
