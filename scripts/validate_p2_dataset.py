"""独立校验已冻结的 P2 v0.2 数据集和 Manifest。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from import_p2_eval_markdown import p2_pdf_page_counts  # noqa: E402

from researchpilot.evaluation import EvaluationDataError, load_corpus_manifest  # noqa: E402
from researchpilot.p2_dataset import (  # noqa: E402
    P2_PAPER_SPLITS,
    load_p2_cases,
    validate_frozen_p2_manifest,
    validate_p2_cases,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 P2 v0.2 JSONL、语料和冻结 Manifest。")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_p2_v0.2.jsonl")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.corpus.json")
    parser.add_argument("--p2-manifest", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.manifest.json")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/raw/ai_cs_eval_v0.2")
    parser.add_argument("--skip-frozen-hash", action="store_true", help="仅校验 JSONL 语义，不校验冻结哈希。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        corpus = load_corpus_manifest(args.corpus)
        cases = load_p2_cases(args.dataset)
        page_counts = p2_pdf_page_counts(corpus_path=args.corpus, raw_root=args.raw_root)
        validate_p2_cases(cases=cases, corpus=corpus, paper_splits=P2_PAPER_SPLITS, page_counts=page_counts)
        if not args.skip_frozen_hash:
            validate_frozen_p2_manifest(manifest_path=args.p2_manifest, dataset_path=args.dataset)
    except (EvaluationDataError, OSError, UnicodeError) as error:
        print(f"P2 校验失败：{error}", file=sys.stderr)
        return 2
    by_split = {split: sum(case.split == split for case in cases) for split in ("dev", "test")}
    print(f"P2 校验通过：{len(cases)} 题（Dev {by_split['dev']}，Test {by_split['test']}）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
