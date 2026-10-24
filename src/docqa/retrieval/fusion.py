"""多路召回融合。

三个可选策略：
- rrf（默认）：Reciprocal Rank Fusion，只看名次不看分数。
  好处是不用处理"向量分数是余弦相似度、BM25 分数是无界实数"这种量纲问题。
  公式：score(d) = Σ weight_i / (k + rank_i(d))，k 一般取 60。
- weighted：把两路分数各自 min-max 归一化后加权求和。
  好处是保留了"有多相似"的信息，坏处是对分数分布敏感，换个模型就要重新调权重。

工程上默认选 RRF：少一个超参，跨模型迁移更稳。
"""

from __future__ import annotations

from collections.abc import Iterable


def rrf_fuse(
    rankings: Iterable[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """输入多路有序 id 列表，输出融合后的 (id, score) 降序。"""
    rankings = [r for r in rankings]
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 长度必须与召回路数一致")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=False):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def weighted_fuse(
    ranked: list[list[tuple[str, float]]],
    *,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """归一化分数后加权求和。ranked 形如 [[(id, score), ...], ...]。"""
    if weights is None:
        weights = [1.0] * len(ranked)
    merged: dict[str, float] = {}
    for hits, weight in zip(ranked, weights, strict=False):
        if not hits:
            continue
        values = [s for _, s in hits]
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        for doc_id, score in hits:
            normalized = (score - lo) / span
            merged[doc_id] = merged.get(doc_id, 0.0) + weight * normalized
    return sorted(merged.items(), key=lambda item: item[1], reverse=True)


def mmr_select(
    items: list[tuple[str, float, list[str]]],
    top_n: int,
    lambda_: float = 0.7,
) -> list[str]:
    """最大边际相关（MMR）去冗余。

    items: [(id, relevance, tokens)]
    这里的相似度用词面 Jaccard 近似——再算一遍向量太贵，
    而检索结果高度冗余时（同一段话被切成多块），词面重合度已经足够识别。
    """
    if lambda_ >= 1.0 or len(items) <= top_n:
        return [i for i, _, _ in items[:top_n]]

    selected: list[int] = []
    candidates = list(range(len(items)))
    while candidates and len(selected) < top_n:
        best_idx = None
        best_score = float("-inf")
        for idx in candidates:
            relevance = items[idx][1]
            if selected:
                penalty = max(_jaccard(items[idx][2], items[s][2]) for s in selected)
            else:
                penalty = 0.0
            score = lambda_ * relevance - (1 - lambda_) * penalty
            if score > best_score:
                best_score, best_idx = score, idx
        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
    return [items[i][0] for i in selected]


def _jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0
