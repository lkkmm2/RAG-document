"""把「人工写的种子问答」对齐到 chunk id，生成可自动评分的评测集。

为什么需要这一步：
评测要算 recall@k，就必须知道"这个问题对应的正确 chunk 是哪个"。
人工在 900 个 chunk 里点选不现实，所以：
- 人工只写问题 + 标准答案 + 几个**高区分度的关键词**（如"累计不得超过两年"）；
- 脚本按关键词在整个索引里定位 chunk，命中的就是 gold chunk；
- 关键词太通用（命中 > 5 个 chunk）会被判为"不够有区分度"并告警。

这样 gold 标注既准确又可复查：任何一条都能追溯到具体字符串。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docqa.config import get_settings
from docqa.schemas import EvalCase
from docqa.store.chunk_repo import ChunkRepository

MAX_MATCH = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成评测集")
    parser.add_argument("--seed", default="data/eval/seed_qa.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    seed_path = Path(settings.qa_path).parent / "seed_qa.jsonl"
    seed_path = Path(args.seed) if args.seed else seed_path
    out_path = Path(args.out) if args.out else settings.qa_path

    if not seed_path.exists():
        print(f"种子文件不存在：{seed_path}")
        return 1

    repo = ChunkRepository(settings.index_dir)
    chunks = repo.load_chunks()
    if not chunks:
        print("索引为空，请先运行 python -m docqa.cli")
        return 1

    cases: list[EvalCase] = []
    warnings: list[str] = []
    for line in seed_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        seed = json.loads(line)
        case = EvalCase.model_validate(seed)

        if case.answerable and case.gold_keywords:
            matched = [
                c for c in chunks
                if all(kw in c.text for kw in case.gold_keywords)
            ]
            if not matched:
                # 退化成"任一关键词命中"，再不行就告警
                matched = [c for c in chunks if any(kw in c.text for kw in case.gold_keywords)]
            if len(matched) > MAX_MATCH:
                warnings.append(
                    f"{case.id}: 关键词命中 {len(matched)} 个 chunk，区分度不足，建议换更独特的词"
                )
                matched = matched[:MAX_MATCH]
            if matched:
                case.gold_chunk_ids = [c.id for c in matched[:3]]
                case.gold_sections = sorted({c.section for c in matched[:3] if c.section})
            else:
                warnings.append(f"{case.id}: 没有 chunk 命中关键词，会被算作 recall=0")

        cases.append(case)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(case.model_dump_json() + "\n")

    answerable = [c for c in cases if c.answerable]
    linked = [c for c in answerable if c.gold_chunk_ids]
    print(f"写出 {len(cases)} 条（可回答 {len(answerable)} / 不可回答 {len(cases) - len(answerable)}）")
    print(f"成功对齐 gold chunk：{len(linked)}/{len(answerable)}")
    print(f"输出：{out_path}")
    for warning in warnings:
        print(f"  ⚠️  {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
