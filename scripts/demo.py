"""一键演示：把整条链路跑一遍，并把过程写成 markdown。

    python scripts/demo.py

输出三段：
1. RAG 问答（带引用）
2. 检索过程（向量 / BM25 / 融合 / 精排 各阶段得分）
3. Agent 多步任务（工具调用轨迹）

结果同时写到 data/eval/reports/demo-<时间>.md，可以直接贴进 README 或简历附件。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from docqa.config import get_settings
from docqa.logging_setup import setup_logging
from docqa.schemas import AskRequest
from docqa.services import build_services

QUESTIONS = [
    "学生申请转专业需要满足什么条件、走什么流程？",
    "休学最长期限是多久？休学期满不办复学会怎么处理？",
    "考试作弊会受到什么处理？",
    "学校对家庭经济困难学生有哪些资助政策？",
]

AGENT_TASK = "这本手册里，学生违纪处分的种类有哪几种？分别对应什么情形？"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocQA 演示")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging("WARNING", "text")
    services = build_services(settings)

    if services.repo.stats().documents == 0:
        print("索引为空，先入库 data/docs ...")
        for path in sorted(settings.docs_path.glob("*")):
            if path.suffix.lower() in {".pdf", ".txt", ".md"}:
                print(f"  入库 {path.name}")
                services.ingest.ingest_path(path)
        services.refresh()

    stats = services.repo.stats()
    lines: list[str] = []
    lines.append("# DocQA 演示记录")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(
        f"- 索引：{stats.documents} 篇文档 / {stats.chunks} 个子块 / {stats.parents} 个父块"
    )
    lines.append(
        f"- 模型：embedding=`{getattr(services.embedder, 'name', '?')}` "
        f"llm=`{services.llm.name}/{services.llm.model}` rerank=`{services.reranker.enabled}`"
    )
    lines.append("")

    # ---------------------------------------------------------------- 1. RAG 问答
    lines.append("## 一、RAG 问答")
    lines.append("")
    for question in QUESTIONS:
        response = services.answerer.answer(AskRequest(question=question, top_k=args.top_k))
        print(f"\n❓ {question}")
        print(f"💬 {response.answer[:400]}")
        print(f"   ⏱ {response.latency_ms:.0f}ms | 引用 {len(response.citations)} 条")

        lines.append(f"### Q：{question}")
        lines.append("")
        lines.append("```text")
        lines.append(response.answer)
        lines.append("```")
        lines.append("")
        lines.append("引用来源：")
        lines.append("")
        for cite in response.citations:
            section = f" · {cite.section}" if cite.section else ""
            lines.append(
                f"- `[{cite.index}]` 《{cite.doc_name}》{section} 第 {cite.page} 页 "
                f"（相关度 {cite.score:.3f}）"
            )
        stats_text = response.citation_stats
        lines.append("")
        lines.append(
            f"> 延迟 {response.latency_ms:.0f}ms · 引用覆盖 "
            f"{(stats_text.coverage if stats_text else 0):.0%} · "
            f"tokens {response.usage.total_tokens}"
        )
        lines.append("")

    # ---------------------------------------------------------------- 2. 检索过程
    lines.append("## 二、检索过程（消融观察）")
    lines.append("")
    probe = QUESTIONS[0] if QUESTIONS else "学籍异动"
    result = services.retriever.retrieve(probe, top_k=args.top_k, debug=True)
    if result.debug:
        debug = result.debug
        print(f"\n🔍 检索过程：{probe}")
        if debug.rewritten_query and debug.rewritten_query != debug.original_query:
            print(f"   改写：{debug.original_query} → {debug.rewritten_query}")
        lines.append(f"问题：**{probe}**")
        lines.append("")
        for key, title in (
            ("vector", "向量召回 Top5"),
            ("bm25", "BM25 召回 Top5"),
            ("fused", "RRF 融合 Top5"),
            ("rerank", "精排 Top5"),
        ):
            rows = getattr(debug, key, []) or []
            print(f"   {title}: {[r.chunk_id for r in rows[:5]]}")
            lines.append(f"**{title}**")
            lines.append("")
            lines.append("| 排名 | chunk | 分数 | 明细 |")
            lines.append("|---|---|---|---|")
            for row in rows[:5]:
                detail = ", ".join(f"{k}={v}" for k, v in row.scores.items())
                lines.append(f"| {row.rank} | `{row.chunk_id}` | {row.score:.4f} | {detail} |")
            lines.append("")
        lines.append(
            "耗时：" + " · ".join(f"{k} {v:.0f}ms" for k, v in debug.timings_ms.items())
        )
        lines.append("")

    # ---------------------------------------------------------------- 3. Agent
    lines.append("## 三、Agent 多步任务")
    lines.append("")
    print(f"\n🤖 Agent 任务：{AGENT_TASK}")
    agent_response = services.agent.run(AGENT_TASK)
    print(f"   答：{agent_response.answer[:300]}")
    print(f"   步数 {len(agent_response.steps)} · 终止 {agent_response.stopped_reason}")
    for step in agent_response.steps:
        print(f"   [{step.step}] {step.action} -> {step.observation[:80]}...")

    lines.append(f"任务：**{AGENT_TASK}**")
    lines.append("")
    lines.append("```text")
    lines.append(agent_response.answer)
    lines.append("```")
    lines.append("")
    lines.append("| 步 | 思考 | 工具 | 观察（截断） |")
    lines.append("|---|---|---|---|")
    for step in agent_response.steps:
        thought = (step.thought or "").replace("|", "\\|")[:40]
        observation = (step.observation or "").replace("|", "\\|").replace("\n", " ")[:60]
        lines.append(f"| {step.step} | {thought} | `{step.action}` | {observation} |")
    lines.append("")
    lines.append(
        f"> 终止原因：{agent_response.stopped_reason} · 延迟 {agent_response.latency_ms:.0f}ms · "
        f"tokens {agent_response.usage.total_tokens}"
    )

    # ---------------------------------------------------------------- 落盘
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    out = settings.report_dir / f"demo-{datetime.now():%Y%m%d-%H%M%S}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 演示记录已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
