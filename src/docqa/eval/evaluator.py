"""评测：把"感觉还行"变成"召回 0.86、引用准确率 0.92"。

指标定义（面试会追问每个指标怎么算）：
- recall@k：gold chunk 是否出现在前 k 条里。检索层最核心的指标——
  召回不到，后面 prompt 写得再花也没用。
- MRR：第一个正确结果的排名倒数，衡量"正确答案排得有多靠前"。
  recall 只看有没有，MRR 还看位置。
- keyword_hit：答案里命中了多少 gold 关键词。是"回答正确性"的廉价代理指标，
  比 LLM 打分稳定、比人工便宜。
- citation_precision：模型引用的 chunk 里，有多少是 gold chunk。
  衡量"引用是否指向真正的依据"，而不是随便引一条格式正确的。
- refusal_accuracy：无答案问题里，模型正确说"未提及"的比例。防幻觉的最后一道闸。
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from docqa.config import Settings
from docqa.logging_setup import get_logger
from docqa.rag.answerer import RAGAnswerer
from docqa.retrieval.pipeline import RetrievalPipeline
from docqa.schemas import AskRequest, EvalCase, EvalReport, EvalRow

log = get_logger("eval")

# 同时报多个截断点的召回率：
# recall@20 接近"召回上限"，recall@4 是"实际能用的精度"。
# 两者的差距就是精排/排序还能榨出的空间。
RECALL_CUTOFFS = (4, 10, 20)


class Evaluator:
    def __init__(
        self,
        settings: Settings,
        retriever: RetrievalPipeline,
        answerer: RAGAnswerer | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.answerer = answerer

    # ------------------------------------------------------------------ 数据
    def load_cases(self, path: Path | None = None) -> list[EvalCase]:
        path = path or self.settings.qa_path
        if not path.exists():
            raise FileNotFoundError(f"评测集不存在：{path}")
        cases: list[EvalCase] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                cases.append(EvalCase.model_validate_json(line))
        log.info("载入评测集：%s 条（%s）", len(cases), path.name)
        return cases

    # ------------------------------------------------------------------ 运行
    def run(
        self,
        cases: list[EvalCase] | None = None,
        *,
        top_k: int | None = None,
        with_answer: bool = False,
        name: str = "run",
    ) -> EvalReport:
        cases = cases or self.load_cases()
        top_k = top_k or self.settings.retrieval.final_top_k
        rows: list[EvalRow] = []

        for i, case in enumerate(cases, start=1):
            row = EvalRow(id=case.id, question=case.question)
            try:
                started = time.perf_counter()
                result = self.retriever.retrieve(case.question, top_k=top_k)
                row.latency_ms = round((time.perf_counter() - started) * 1000, 2)
                retrieved = result.candidate_ids
                row.retrieved_ids = retrieved
                row.recall, row.reciprocal_rank = _retrieval_scores_at(retrieved, case.gold_chunk_ids, top_k)
                for k in RECALL_CUTOFFS:
                    row.recall_by_k[k] = _hit_at(retrieved, case.gold_chunk_ids, k)

                if with_answer and self.answerer is not None:
                    response = self.answerer.answer(
                        AskRequest(question=case.question, top_k=top_k)
                    )
                    row.answer = response.answer
                    row.latency_ms = response.latency_ms
                    row.keyword_hit = _keyword_hit(response.answer, case.gold_keywords)
                    row.citation_precision = _citation_precision(response, case)
            except Exception as exc:  # noqa: BLE001 - 单条失败不该中断整轮评测
                row.error = str(exc)
                log.warning("用例 %s 失败：%s", case.id, exc)

            rows.append(row)
            if i % 10 == 0:
                log.info("已评测 %s/%s", i, len(cases))

        report = _aggregate(rows, cases, name)
        report.config = {
            "top_k": top_k,
            "with_answer": with_answer,
            "fusion": self.settings.retrieval.fusion,
            "enable_vector": self.settings.retrieval.enable_vector,
            "enable_bm25": self.settings.retrieval.enable_bm25,
            "rerank": self.settings.rerank.enabled,
            "chunk_size": self.settings.chunking.chunk_size,
            "chunk_overlap": self.settings.chunking.chunk_overlap,
            "embedding": self.settings.embedding.model_name,
        }
        return report

    # ------------------------------------------------------------------ 保存
    @staticmethod
    def save(report: EvalReport, report_dir: Path) -> tuple[Path, Path]:
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = f"{stamp}-{report.name}"
        json_path = report_dir / f"{stem}.json"
        md_path = report_dir / f"{stem}.md"
        json_path.write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md_path.write_text(render_markdown(report), encoding="utf-8")
        return json_path, md_path


# ------------------------------------------------------------------ 指标计算
def _hit_at(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    gold_set = set(gold)
    return 1.0 if any(cid in gold_set for cid in retrieved[:k]) else 0.0


def _retrieval_scores_at(retrieved: list[str], gold: list[str], k: int) -> tuple[float, float]:
    """返回 (recall@k, reciprocal rank)。RR 只在 top-k 内找第一个命中。"""
    if not gold:
        return 0.0, 0.0
    gold_set = set(gold)
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in gold_set:
            return 1.0, round(1.0 / rank, 4)
    return 0.0, 0.0


def _keyword_hit(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw and kw in answer)
    return round(hits / len(keywords), 4)


def _citation_precision(response, case: EvalCase) -> float:
    if not response.citations:
        return 0.0
    cited = {c.chunk_id for c in response.citations}
    if not case.gold_chunk_ids:
        return 0.0
    gold = set(case.gold_chunk_ids)
    return round(len(cited & gold) / len(cited), 4)


def _aggregate(rows: list[EvalRow], cases: list[EvalCase], name: str) -> EvalReport:
    ok = [r for r in rows if not r.error]
    latencies = [r.latency_ms for r in ok if r.latency_ms > 0]
    case_map = {c.id: c for c in cases}

    # 关键：召回类指标只在「可回答」的用例上平均。
    # 无答案题没有 gold chunk，把它算进分母会让 recall 凭空掉 11%（5/45），
    # 这是很多人做 RAG 评测时无意识引入的偏差。
    answerable = [r for r in ok if case_map.get(r.id) and case_map[r.id].answerable]
    refusals = [r for r in ok if case_map.get(r.id) and not case_map[r.id].answerable]

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    return EvalReport(
        name=name,
        cases=len(rows),
        cases_answerable=len(answerable),
        recall_at_k=mean([r.recall for r in answerable]),
        recall_by_k={
            k: mean([r.recall_by_k.get(k, 0.0) for r in answerable]) for k in RECALL_CUTOFFS
        },
        mrr=mean([r.reciprocal_rank for r in answerable]),
        keyword_hit=mean([r.keyword_hit for r in answerable if r.answer]),
        citation_precision=mean([r.citation_precision for r in answerable if r.answer]),
        refusal_accuracy=mean(
            [1.0 if "未提及" in r.answer else 0.0 for r in refusals if r.answer]
        ),
        latency_p50_ms=round(statistics.median(latencies), 2) if latencies else 0.0,
        latency_p95_ms=_percentile(latencies, 95),
        rows=rows,
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[idx], 2)


# ------------------------------------------------------------------ 渲染
def render_markdown(report: EvalReport) -> str:
    lines = [
        f"# 评测报告：{report.name}",
        "",
        f"- 用例数：{report.cases}（可回答 {report.cases_answerable}）",
        f"- recall@{report.config.get('top_k', 'k')}：**{report.recall_at_k:.4f}**",
        "- 各截断点召回："
        + " · ".join(f"@{k}={v:.4f}" for k, v in sorted(report.recall_by_k.items())),
        f"- MRR：{report.mrr:.4f}",
        f"- 关键词命中：{report.keyword_hit:.4f}",
        f"- 引用准确率：{report.citation_precision:.4f}",
        f"- 拒答准确率：{report.refusal_accuracy:.4f}",
        f"- 延迟 P50 / P95：{report.latency_p50_ms:.0f}ms / {report.latency_p95_ms:.0f}ms",
        "",
        "## 配置",
        "",
        "```json",
        json.dumps(report.config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 逐条结果",
        "",
        "| # | 问题 | recall | RR | 关键词命中 | 引用准确率 |",
        "|---|------|--------|----|-----------|-----------|",
    ]
    for i, row in enumerate(report.rows, start=1):
        question = row.question.replace("|", "\\|")[:40]
        lines.append(
            f"| {i} | {question} | {row.recall:.0f} | {row.reciprocal_rank:.3f} | "
            f"{row.keyword_hit:.2f} | {row.citation_precision:.2f} |"
        )
    return "\n".join(lines) + "\n"
