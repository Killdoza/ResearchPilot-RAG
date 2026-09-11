"""校验 AI/CS 评测 Manifest、JSONL 标注和实验配置，不加载模型或调用 API。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.evaluation import (  # noqa: E402
    EvaluationDataError,
    load_corpus_manifest,
    load_eval_cases,
    load_experiment_config,
    paper_ids_from_manifest,
    validate_dataset_manifest_coverage,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 ResearchPilot AI/CS 评测数据与实验配置。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "eval/manifests/ai_cs_eval_v0.1.corpus.json",
        help="论文 Manifest JSON 路径。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "eval/datasets/ai_cs_eval_v0.1.jsonl",
        help="人工标注 JSONL 路径。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help="待校验的 YAML 实验配置；可重复传入。",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="仅用于验证尚未开始标注的模板；正式实验禁止使用。",
    )
    parser.add_argument(
        "--require-hashes",
        action="store_true",
        help="要求 Manifest 中每篇论文均已填入 SHA-256。",
    )
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = [
            PROJECT_ROOT / "configs/experiments/e0_dense.yaml",
            PROJECT_ROOT / "configs/experiments/h0_current_hybrid.yaml",
        ]
    return args


def validate(
    *, manifest_path: Path, dataset_path: Path, config_paths: list[Path], allow_empty: bool, require_hashes: bool
) -> tuple[int, Counter[str], Counter[str]]:
    manifest = load_corpus_manifest(manifest_path)
    cases = load_eval_cases(dataset_path, allow_empty=allow_empty)
    validate_dataset_manifest_coverage(cases, manifest)
    if require_hashes:
        paper_ids_from_manifest(manifest, require_hashes=True)
    for config_path in config_paths:
        load_experiment_config(config_path)
    return (
        len(cases),
        Counter(case.split for case in cases),
        Counter(case.task_type for case in cases),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        total, split_counts, type_counts = validate(
            manifest_path=args.manifest,
            dataset_path=args.dataset,
            config_paths=args.config,
            allow_empty=args.allow_empty,
            require_hashes=args.require_hashes,
        )
    except EvaluationDataError as error:
        print(f"评测数据校验失败：{error}", file=sys.stderr)
        return 2
    print(f"评测数据校验通过：论文 {len(load_corpus_manifest(args.manifest).papers)} 篇，问题 {total} 条。")
    print(f"按切分统计：{dict(sorted(split_counts.items()))}")
    print(f"按类型统计：{dict(sorted(type_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
