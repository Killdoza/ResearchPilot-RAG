"""P2-C 预算预检：默认仅输出 JSONL，不调用 DeepSeek。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from import_p2_eval_markdown import p2_pdf_page_counts  # noqa: E402

from researchpilot.evaluation import EvaluationDataError, load_corpus_manifest  # noqa: E402
from researchpilot.p2_dataset import (  # noqa: E402
    P2_PAPER_SPLITS,
    load_p2_cases,
    validate_frozen_p2_manifest,
    validate_p2_cases,
)
from researchpilot.p2_runner import build_p2_dry_run, load_p2_execution_config  # noqa: E402
from researchpilot.p2_runtime import P2RuntimePlanner  # noqa: E402
from researchpilot.scope_resolution import PaperScopeCatalog  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 P2-C 预算 Dry-run（不调用模型）。")
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_p2_v0.2.jsonl")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.corpus.json")
    parser.add_argument("--p2-manifest", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.manifest.json")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/raw/ai_cs_eval_v0.2")
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / "data/runtime/p2_evaluations")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        corpus = load_corpus_manifest(args.corpus)
        cases = load_p2_cases(args.dataset)
        pages = p2_pdf_page_counts(corpus_path=args.corpus, raw_root=args.raw_root)
        validate_p2_cases(cases=cases, corpus=corpus, paper_splits=P2_PAPER_SPLITS, page_counts=pages)
        validate_frozen_p2_manifest(dataset_path=args.dataset, manifest_path=args.p2_manifest)
        selected = tuple(case for case in cases if case.split == args.split)
        if not selected:
            raise EvaluationDataError(f"P2 数据集没有 split={args.split} 的题目。")
        planner = P2RuntimePlanner(catalog=PaperScopeCatalog.from_corpus(corpus))
        args.runtime_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = args.runtime_dir / f"dry_run_p2_{args.split}_{timestamp}.jsonl"
        all_summaries = []
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for config_path in args.config:
                config = load_p2_execution_config(config_path)
                records, summary = build_p2_dry_run(cases=selected, config=config, planner=planner)
                config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
                for record in records:
                    handle.write(
                        json.dumps(
                            {
                                "schema_version": "p2_dry_run_v1",
                                "config": config.retrieval_experiment.name,
                                "config_sha256": config_hash,
                                "record": record.to_dict(),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                all_summaries.append(summary.to_dict())
    except Exception as error:
        print(f"P2 Dry-run 失败：{error}", file=sys.stderr)
        return 2
    print(f"P2 Dry-run 完成：split={args.split}，记录：{output}")
    for summary in all_summaries:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
