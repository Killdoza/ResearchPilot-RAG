"""评测脚本的轻量边界测试；不加载 Embedding、Reranker 或真实 Chroma。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.evaluation import (  # noqa: E402
    CORPUS_SCHEMA_VERSION,
    CorpusManifest,
    EvaluationDataError,
    EvaluationSummary,
    load_experiment_config,
)


def _load_script(name: str) -> object:
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"researchpilot_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_script("validate_eval_dataset.py")
INGESTOR = _load_script("ingest_eval_corpus.py")
RUNNER = _load_script("run_retrieval_eval.py")
PARSE_QUALITY = _load_script("inspect_parse_quality.py")


class EvaluationScriptTests(unittest.TestCase):
    def test_frozen_dataset_is_valid_without_empty_dataset_escape_hatch(self) -> None:
        total, splits, task_types = VALIDATOR.validate(
            manifest_path=PROJECT_ROOT / "eval/manifests/ai_cs_eval_v0.1.corpus.json",
            dataset_path=PROJECT_ROOT / "eval/datasets/ai_cs_eval_v0.1.jsonl",
            config_paths=[PROJECT_ROOT / "configs/experiments/e0_dense.yaml"],
            allow_empty=True,
            require_hashes=False,
        )
        self.assertEqual(
            (total, dict(splits), dict(task_types)),
            (
                77,
                {"dev": 53, "test": 24},
                {
                    "cross_lingual": 2,
                    "experimental_setup": 22,
                    "keyword_exact": 6,
                    "local_fact": 9,
                    "multi_section": 6,
                    "semantic_explanation": 28,
                    "unanswerable": 4,
                },
            ),
        )
        validated_total, _, _ = VALIDATOR.validate(
            manifest_path=PROJECT_ROOT / "eval/manifests/ai_cs_eval_v0.1.corpus.json",
            dataset_path=PROJECT_ROOT / "eval/datasets/ai_cs_eval_v0.1.jsonl",
            config_paths=[PROJECT_ROOT / "configs/experiments/e0_dense.yaml"],
            allow_empty=False,
            require_hashes=False,
        )
        self.assertEqual(validated_total, 77)

    def test_ingestion_inspection_rejects_empty_manifest_without_writing(self) -> None:
        manifest = CorpusManifest(
            schema_version=CORPUS_SCHEMA_VERSION,
            description="空 Manifest 边界测试。",
            papers=(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(EvaluationDataError, "Manifest 为空"):
                INGESTOR.inspect_local_corpus(manifest=manifest, raw_dir=Path(temp_dir))

    def test_aggregate_results_and_jsonl_are_written_in_expected_formats(self) -> None:
        experiment = load_experiment_config(PROJECT_ROOT / "configs/experiments/e0_dense.yaml")
        summary = EvaluationSummary(
            evaluated_cases=2,
            answerable_cases=2,
            unanswerable_cases=0,
            question_recall_at={1: 0.5, 5: 1.0, 10: 1.0},
            evidence_recall_at={1: 0.5, 5: 1.0, 10: 1.0},
            mrr=0.75,
            ndcg_at={1: 0.5, 5: 0.8, 10: 0.8},
            mean_retrieval_latency_ms=12.5,
            unanswerable_top1_retrieval_rate=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_path, csv_path = RUNNER.write_aggregate_results(
                results_dir=root / "results",
                experiment=experiment,
                split="dev",
                summary=summary,
                overwrite=False,
            )
            jsonl_path = RUNNER.write_jsonl_records(
                runtime_dir=root / "runtime",
                experiment_name=experiment.name,
                split="dev",
                records=({"case_id": "q001", "retrieval": {"trace_id": "trace_1"}},),
            )
            self.assertIn("Question Recall@5", markdown_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(jsonl_path.read_text(encoding="utf-8").strip())["case_id"], "q001")
            self.assertIn("question_recall_at_5", csv_path.read_text(encoding="utf-8"))
            self.assertIn("p95_retrieval_latency_ms", csv_path.read_text(encoding="utf-8"))

    def test_custom_config_replaces_default_validator_configs(self) -> None:
        parsed = VALIDATOR.parse_args(["--config", "custom.yaml", "--allow-empty"])
        self.assertEqual(parsed.config, [Path("custom.yaml")])

    def test_parse_quality_report_aggregates_structural_counts(self) -> None:
        report = PARSE_QUALITY.build_report(
            experiment_name="e5_structure_aware",
            imported=(
                {
                    "block_count": 12,
                    "parent_chunk_count": 5,
                    "child_chunk_count": 18,
                    "parse_quality": {
                        "empty_page_count": 1,
                        "heading_count": 3,
                        "table_count": 2,
                        "elapsed_ms": 20.5,
                    },
                },
            ),
        )
        self.assertEqual(report["total_blocks"], 12)
        self.assertEqual(report["total_tables"], 2)


if __name__ == "__main__":
    unittest.main()
