"""评测脚本：跑单组配置，或做消融对比。

用法：

    python scripts/eval.py --presets vector_only,hybrid          # 只测检索（不花 token）
    python scripts/eval.py --presets hybrid --with-answer        # 连回答质量一起测
    python scripts/eval.py --presets vector_only,hybrid,hybrid_rerank

消融实验的意义：简历上写"混合检索 + 精排把 recall@5 提升了 X"，
前提是你真的能一条条跑出这几个数字。这个脚本就是产出那张表的工具。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from docqa.config import get_settings
from docqa.eval.evaluator import Evaluator
from docqa.logging_setup import setup_logging
from docqa.services import build_services

PRESETS: dict[str, dict] = {
    "vector_only": {"enable_vector": True, "enable_bm25": False, "rerank": False},
    "bm25_only": {"enable_vector": False, "enable_bm25": True, "rerank": False},
    "hybrid": {"enable_vector": True, "enable_bm25": True, "rerank": False},
    "hybrid_rerank": {"enable_vector": True, "enable_bm25": True, "rerank": True},
    "weighted": {"enable_vector": True, "enable_bm25": True, "rerank": False, "fusion": "weighted"},
}


def apply_preset(services, preset: str) -> None:
    cfg = PRESETS[preset]
    services.settings.retrieval.enable_vector = cfg["enable_vector"]
    services.settings.retrieval.enable_bm25 = cfg["enable_bm25"]
    services.settings.retrieval.fusion = cfg.get("fusion", "rrf")
    services.reranker.enabled = cfg["rerank"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocQA 评测")
    parser.add_argument(
        "--presets",
        default="vector_only,hybrid",
        help=f"逗号分隔，可选：{','.join(PRESETS)}",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--with-answer", action="store_true", help="同时评测回答质量（消耗 token）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，调试用")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.api.log_level, settings.api.log_format)

    services = build_services(settings)
    if services.retriever.chunk_count() == 0:
        print("索引为空，请先运行：python -m docqa.cli")
        return 1

    evaluator = Evaluator(settings, services.retriever, services.answerer)
    cases = evaluator.load_cases()
    if args.limit:
        cases = cases[: args.limit]

    reports = []
    for preset in [p.strip() for p in args.presets.split(",") if p.strip()]:
        if preset not in PRESETS:
            print(f"未知 preset：{preset}")
            return 1
        apply_preset(services, preset)
        print(f"\n▶ 运行 {preset} ...")
        report = evaluator.run(
            cases, top_k=args.top_k, with_answer=args.with_answer, name=preset
        )
        path_json, path_md = evaluator.save(report, settings.report_dir)
        print(
            f"  recall@{report.config['top_k']}={report.recall_at_k:.4f}  "
            f"MRR={report.mrr:.4f}  "
            f"keyword_hit={report.keyword_hit:.4f}  "
            f"P95={report.latency_p95_ms:.0f}ms"
        )
        print(f"  报告：{path_md}")
        reports.append(report)

    if len(reports) > 1:
        print("\n" + render_comparison(reports))
        out = settings.report_dir / f"compare-{datetime.now():%Y%m%d-%H%M%S}.md"
        out.write_text(render_comparison(reports), encoding="utf-8")
        print(f"\n对比表已保存：{out}")

    return 0


def render_comparison(reports) -> str:
    header = (
        "| 配置 | recall@4 | recall@10 | recall@20 | MRR | 关键词命中 | "
        "引用准确率 | P50 | P95 |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        rk = r.recall_by_k
        lines.append(
            f"| {r.name} | {rk.get(4, 0):.4f} | {rk.get(10, 0):.4f} | {rk.get(20, 0):.4f} | "
            f"{r.mrr:.4f} | {r.keyword_hit:.4f} | {r.citation_precision:.4f} | "
            f"{r.latency_p50_ms:.0f}ms | {r.latency_p95_ms:.0f}ms |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
