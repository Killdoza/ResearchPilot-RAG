"""验证冻结的跨论文评测集及其防篡改 Manifest。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.crossdoc import (  # noqa: E402
    DEFAULT_PAPER_SPLITS,
    load_crossdoc_cases,
    validate_crossdoc_cases,
    validate_frozen_crossdoc_manifest,
)
from researchpilot.evaluation import EvaluationDataError, load_corpus_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 AI/CS 跨论文评测集。")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_crossdoc_v0.1.jsonl")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_eval_v0.1.corpus.json")
    parser.add_argument(
        "--crossdoc-manifest",
        type=Path,
        default=PROJECT_ROOT / "eval/manifests/ai_cs_crossdoc_v0.1.manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frozen = validate_frozen_crossdoc_manifest(dataset_path=args.dataset, manifest_path=args.crossdoc_manifest)
        page_counts = frozen.get("pdf_page_counts")
        if not isinstance(page_counts, dict):
            raise EvaluationDataError("跨论文 Manifest 缺少冻结的 pdf_page_counts。")
        cases = load_crossdoc_cases(args.dataset)
        validate_crossdoc_cases(
            cases=cases,
            corpus=load_corpus_manifest(args.manifest),
            paper_splits=DEFAULT_PAPER_SPLITS,
            page_counts=page_counts,
        )
    except (OSError, EvaluationDataError) as error:
        print(f"跨论文评测集校验失败：{error}", file=sys.stderr)
        return 2
    print(
        "跨论文评测集校验通过："
        f"{len(cases)} 条；SHA-256={frozen['dataset_sha256']}；"
        f"Dev={sum(case.split == 'dev' for case in cases)}；"
        f"Test={sum(case.split == 'test' for case in cases)}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
