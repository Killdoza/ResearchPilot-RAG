"""只读解析冻结语料，输出 Parser / Chunker 的质量报告，不写入索引。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ingest_eval_corpus import (  # noqa: E402
    inspect_local_corpus,
    parse_corpus_for_indexing,
)

from researchpilot.evaluation import load_corpus_manifest, load_experiment_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 ResearchPilot Parser / Chunker 的解析质量。")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_eval_v0.1.corpus.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/e5_structure_aware.yaml")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw/ai_cs_eval_v0.1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/runtime/parse_quality")
    return parser.parse_args(argv)


def build_report(*, experiment_name: str, imported: tuple[dict[str, object], ...]) -> dict[str, object]:
    qualities = [item["parse_quality"] for item in imported if isinstance(item.get("parse_quality"), dict)]
    return {
        "schema_version": "parse_quality_report_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": experiment_name,
        "paper_count": len(imported),
        "total_blocks": sum(int(item["block_count"]) for item in imported),
        "total_parent_chunks": sum(int(item["parent_chunk_count"]) for item in imported),
        "total_child_chunks": sum(int(item["child_chunk_count"]) for item in imported),
        "total_empty_pages": sum(int(quality["empty_page_count"]) for quality in qualities),
        "total_headings": sum(int(quality["heading_count"]) for quality in qualities),
        "total_tables": sum(int(quality["table_count"]) for quality in qualities),
        "total_parse_elapsed_ms": round(sum(float(quality["elapsed_ms"]) for quality in qualities), 3),
        "papers": list(imported),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_corpus_manifest(args.manifest)
        config = load_experiment_config(args.config)
        inspected = inspect_local_corpus(manifest=manifest, raw_dir=args.raw_dir)
        imported, _ = parse_corpus_for_indexing(manifest=manifest, inspected=inspected, config=config)
        report = build_report(experiment_name=config.name, imported=imported)
    except Exception as error:
        print(f"解析质量检查失败：{error}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"{config.name}_{timestamp}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"解析质量检查完成：论文 {report['paper_count']} 篇，Block {report['total_blocks']} 个，"
        f"Parent/Child Chunk {report['total_parent_chunks']}/{report['total_child_chunks']}，报告：{output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
