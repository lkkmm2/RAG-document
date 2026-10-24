"""命令行入口：入库。

用法：
    docqa-ingest                       # 入库 data/docs 下所有文档
    docqa-ingest --path data/docs/x.pdf
    docqa-ingest --reset               # 清空索引后重建
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docqa.config import get_settings
from docqa.logging_setup import setup_logging
from docqa.services import build_services


def main_ingest(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocQA 文档入库")
    parser.add_argument("--path", type=str, help="单个文件路径")
    parser.add_argument("--dir", type=str, default=None, help="目录（默认 data/docs）")
    parser.add_argument("--reset", action="store_true", help="重建索引（清空已有数据）")
    parser.add_argument("--tables", action="store_true", help="启用 pdfplumber 表格抽取（较慢）")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.tables:
        settings.ingest.extract_tables = True
    setup_logging(settings.api.log_level, settings.api.log_format)

    services = build_services(settings)
    if args.reset:
        services.chroma.reset()
        services.repo.clear()
        services.refresh()
        print("已清空索引")

    targets: list[Path] = []
    if args.path:
        targets = [Path(args.path)]
    else:
        directory = Path(args.dir) if args.dir else settings.docs_path
        targets = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}
        )

    if not targets:
        print(f"没有找到可入库的文档（{settings.docs_path}）")
        return 1

    total_chunks = 0
    for path in targets:
        if not path.exists():
            print(f"跳过（不存在）：{path}")
            continue
        try:
            response = services.ingest.ingest_path(path)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {path.name}: {exc}")
            continue
        total_chunks += response.chunks
        print(
            f"✅ {response.doc_name}  {response.pages} 页 / "
            f"{response.chunks} 子块 / {response.parents} 父块 / "
            f"{response.elapsed_ms:.0f}ms"
        )
        for warning in response.warnings:
            print(f"   ⚠️  {warning}")

    services.refresh()
    stats = services.repo.stats()
    print(
        f"\n索引就绪：{stats.documents} 篇文档 / {stats.chunks} 个子块 / "
        f"embedding={stats.embedding_model} / 本次新增 {total_chunks} 块"
    )
    print(f"索引目录：{settings.index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main_ingest())
