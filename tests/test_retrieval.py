"""检索层测试：融合策略、精排降级、引用校验、BM25 中文召回。"""

from __future__ import annotations

from docqa.rag.prompts import build_citations, extract_citation_indexes, validate_citations
from docqa.retrieval.fusion import mmr_select, rrf_fuse, weighted_fuse
from docqa.schemas import ContextItem


def test_rrf_prefers_documents_hit_by_both_routes():
    vector = ["a", "b", "c"]
    bm25 = ["c", "a", "d"]
    fused = dict(rrf_fuse([vector, bm25], k=60))

    # a 和 c 两路都命中，应该排在只被一路命中的 b/d 前面
    assert fused["a"] > fused["b"]
    assert fused["c"] > fused["d"]
    assert fused["a"] > 0


def test_rrf_respects_weights():
    fused = dict(rrf_fuse([["a"], ["b"]], k=60, weights=[1.0, 0.1]))
    assert fused["a"] > fused["b"]


def test_weighted_fuse_normalizes():
    fused = dict(weighted_fuse([[("a", 0.9), ("b", 0.1)], [("b", 100.0), ("c", 1.0)]]))
    # 两路量纲差 100 倍，归一化后不应该出现压倒性偏差
    assert 0 <= fused["b"] <= 2.0
    assert fused["b"] > fused["c"]


def test_mmr_drops_redundant_items():
    items = [
        ("a", 0.9, ["学籍", "注册", "异动"]),
        ("b", 0.85, ["学籍", "注册", "异动"]),  # 和 a 几乎重复
        ("c", 0.5, ["休学", "复学", "期限"]),
    ]
    picked = mmr_select(items, top_n=2, lambda_=0.5)
    assert "a" in picked
    assert "c" in picked, "应该在相关性和多样性之间做权衡，而不是简单取前二"


def _contexts() -> list[ContextItem]:
    return [
        ContextItem(index=1, chunk_id="c1", parent_id="p1", doc_name="手册.txt", text="甲"),
        ContextItem(index=2, chunk_id="c2", parent_id="p2", doc_name="手册.txt", text="乙"),
    ]


def test_citation_extraction_handles_multiple_formats():
    assert extract_citation_indexes("结论[1]，依据[2][3]。") == [1, 2, 3]
    assert extract_citation_indexes("合并写法[1,2]") == [1, 2]
    assert extract_citation_indexes("没有引用") == []


def test_citation_validation_flags_hallucinated_index():
    stats = validate_citations("答案[1]，另有[9]。", _contexts())
    assert stats.invalid == [9]
    assert stats.valid is False
    assert 0 < stats.coverage <= 1


def test_citation_validation_accepts_valid_answer():
    stats = validate_citations("答案[1]。补充[2]。", _contexts())
    assert stats.valid
    assert stats.total == 2
    assert stats.coverage == 1.0


def test_build_citations_truncates_snippet():
    contexts = [
        ContextItem(
            index=1, chunk_id="c1", parent_id="p1", doc_name="d", text="长" * 500, score=0.5
        )
    ]
    citations = build_citations(contexts)
    assert len(citations[0].snippet) <= 220


def test_bm25_finds_chinese_keyword(ingested):  # noqa: ARG001
    from docqa.store.bm25_index import tokenize

    tokens = tokenize("学籍异动怎么申请？")
    assert "学籍" in tokens
    assert "怎么" not in tokens, "疑问词应该被停用词表过滤"
