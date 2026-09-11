"""评测数据契约与页级确定性检索指标测试；不加载模型、不读取真实论文。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.evaluation import (  # noqa: E402
    CorpusManifest,
    EvalCase,
    EvaluationDataError,
    GoldEvidence,
    case_evaluation_to_dict,
    evaluate_case,
    load_eval_cases,
    load_experiment_config,
    paper_ids_from_manifest,
    render_summary_markdown,
    summarize_evaluations,
    validate_dataset_manifest_coverage,
)
from researchpilot.schemas import BlockType, Chunk, ScoredChunk  # noqa: E402

PAPER_A_HASH = "a" * 64
PAPER_B_HASH = "b" * 64


def _manifest() -> CorpusManifest:
    return CorpusManifest.from_dict(
        {
            "schema_version": "ai_cs_corpus_v0.1",
            "description": "测试论文清单",
            "papers": [
                {
                    "paper_key": "paper_a",
                    "title": "Paper A",
                    "area": "nlp",
                    "year": 2025,
                    "source_url": "https://example.com/paper_a.pdf",
                    "local_filename": "paper_a.pdf",
                    "sha256": PAPER_A_HASH,
                },
                {
                    "paper_key": "paper_b",
                    "title": "Paper B",
                    "area": "systems",
                    "year": 2024,
                    "source_url": "https://example.com/paper_b.pdf",
                    "local_filename": "paper_b.pdf",
                    "sha256": PAPER_B_HASH,
                },
            ],
        }
    )


def _case(
    *,
    case_id: str,
    paper_keys: tuple[str, ...],
    task_type: str = "local_fact",
    evidence: tuple[GoldEvidence, ...] = (),
    answerable: bool = True,
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        split="dev",
        paper_keys=paper_keys,
        question="论文的实验设置是什么？",
        question_language="zh",
        task_type=task_type,
        answerable=answerable,
        gold_evidence=evidence,
        expected_answer="论文给出了可核验的实验设置。" if answerable else "论文未提供该信息，不能可靠作答。",
    )


def _result(*, paper_id: str, page: int, score: float) -> ScoredChunk:
    chunk = Chunk.create(
        paper_id=paper_id,
        parent_id="block_demo",
        page_start=page,
        page_end=page,
        section_path=("实验",),
        content=f"第 {page} 页实验内容。",
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test_v1",
    )
    return ScoredChunk(chunk=chunk, source_name="demo.pdf", dense_rank=1, dense_score=score)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.paper_ids = paper_ids_from_manifest(self.manifest)

    def test_page_level_matching_requires_the_same_paper_and_overlapping_page(self) -> None:
        case = _case(
            case_id="q_page_match",
            paper_keys=("paper_a",),
            evidence=(GoldEvidence("paper_a", 3, 3, "方法"),),
        )
        outcome = evaluate_case(
            case,
            (
                _result(paper_id=self.paper_ids["paper_b"], page=3, score=0.99),
                _result(paper_id=self.paper_ids["paper_a"], page=2, score=0.90),
                _result(paper_id=self.paper_ids["paper_a"], page=3, score=0.80),
            ),
            paper_ids=self.paper_ids,
            retrieval_latency_ms=12.5,
        )

        self.assertEqual(outcome.relevant_ranks, (3,))
        self.assertEqual(outcome.evidence_first_ranks, (3,))
        self.assertEqual(outcome.result_count, 3)

    def test_metrics_calculate_question_evidence_recall_mrr_and_unanswerable_rate(self) -> None:
        first_case = _case(
            case_id="q_first",
            paper_keys=("paper_a",),
            evidence=(GoldEvidence("paper_a", 2, 2),),
        )
        second_case = _case(
            case_id="q_second",
            paper_keys=("paper_b",),
            evidence=(GoldEvidence("paper_b", 4, 4), GoldEvidence("paper_b", 5, 5)),
        )
        unanswerable_case = _case(
            case_id="q_unanswerable",
            paper_keys=("paper_a",),
            task_type="unanswerable",
            answerable=False,
        )
        outcomes = (
            evaluate_case(
                first_case,
                (
                    _result(paper_id=self.paper_ids["paper_b"], page=2, score=0.99),
                    _result(paper_id=self.paper_ids["paper_a"], page=2, score=0.90),
                ),
                paper_ids=self.paper_ids,
                retrieval_latency_ms=10.0,
            ),
            evaluate_case(
                second_case,
                (_result(paper_id=self.paper_ids["paper_b"], page=4, score=0.95),),
                paper_ids=self.paper_ids,
                retrieval_latency_ms=20.0,
            ),
            evaluate_case(
                unanswerable_case,
                (_result(paper_id=self.paper_ids["paper_a"], page=1, score=0.80),),
                paper_ids=self.paper_ids,
                retrieval_latency_ms=30.0,
            ),
        )

        summary = summarize_evaluations(outcomes, cutoffs=(1, 5))

        self.assertEqual(summary.evaluated_cases, 3)
        self.assertAlmostEqual(summary.question_recall_at[1], 0.5)
        self.assertAlmostEqual(summary.question_recall_at[5], 1.0)
        self.assertAlmostEqual(summary.evidence_recall_at[1], 1 / 3)
        self.assertAlmostEqual(summary.evidence_recall_at[5], 2 / 3)
        self.assertAlmostEqual(summary.mrr, 0.75)
        self.assertAlmostEqual(summary.mean_retrieval_latency_ms, 20.0)
        self.assertAlmostEqual(summary.p50_retrieval_latency_ms, 20.0)
        self.assertAlmostEqual(summary.p95_retrieval_latency_ms, 29.0)
        self.assertEqual(summary.unanswerable_top1_retrieval_rate, 1.0)
        self.assertEqual(case_evaluation_to_dict(outcomes[0])["relevant_ranks"], (2,))

    def test_invalid_unanswerable_case_and_unknown_manifest_paper_are_rejected(self) -> None:
        with self.assertRaisesRegex(EvaluationDataError, "不可回答问题"):
            _case(
                case_id="q_invalid",
                paper_keys=("paper_a",),
                task_type="unanswerable",
                answerable=False,
                evidence=(GoldEvidence("paper_a", 1, 1),),
            )
        unknown_case = _case(
            case_id="q_unknown",
            paper_keys=("not_in_manifest",),
            evidence=(GoldEvidence("not_in_manifest", 1, 1),),
        )
        with self.assertRaisesRegex(EvaluationDataError, "不存在"):
            validate_dataset_manifest_coverage((unknown_case,), self.manifest)

    def test_duplicate_gold_evidence_is_rejected_and_duplicate_retrieval_does_not_inflate_ndcg(self) -> None:
        with self.assertRaisesRegex(EvaluationDataError, "不能重复"):
            _case(
                case_id="q_duplicate_gold",
                paper_keys=("paper_a",),
                evidence=(GoldEvidence("paper_a", 2, 2), GoldEvidence("paper_a", 2, 2)),
            )
        case = _case(
            case_id="q_duplicate_result",
            paper_keys=("paper_a",),
            evidence=(GoldEvidence("paper_a", 2, 2),),
        )
        outcome = evaluate_case(
            case,
            (
                _result(paper_id=self.paper_ids["paper_a"], page=2, score=0.9),
                _result(paper_id=self.paper_ids["paper_a"], page=2, score=0.8),
            ),
            paper_ids=self.paper_ids,
            retrieval_latency_ms=1.0,
        )
        self.assertEqual(summarize_evaluations((outcome,)).ndcg_at[5], 1.0)

    def test_jsonl_loader_reports_line_errors_and_accepts_explicit_empty_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "cases.jsonl"
            dataset_path.write_text("{bad json}\n", encoding="utf-8")
            with self.assertRaisesRegex(EvaluationDataError, "第 1 行"):
                load_eval_cases(dataset_path)
            dataset_path.write_text("\n", encoding="utf-8")
            self.assertEqual(load_eval_cases(dataset_path, allow_empty=True), ())
            with self.assertRaisesRegex(EvaluationDataError, "评测集为空"):
                load_eval_cases(dataset_path)

    def test_versioned_experiment_configs_and_markdown_report_are_loadable(self) -> None:
        e0 = load_experiment_config(PROJECT_ROOT / "configs/experiments/e0_dense.yaml")
        h0 = load_experiment_config(PROJECT_ROOT / "configs/experiments/h0_current_hybrid.yaml")
        e1 = load_experiment_config(PROJECT_ROOT / "configs/experiments/e1_bm25.yaml")
        e3 = load_experiment_config(PROJECT_ROOT / "configs/experiments/e3_dense_bm25_rrf.yaml")
        e4 = load_experiment_config(PROJECT_ROOT / "configs/experiments/e4_dense_bm25_rrf_rerank.yaml")
        e5 = load_experiment_config(PROJECT_ROOT / "configs/experiments/e5_structure_aware.yaml")
        self.assertEqual(e0.strategy, "dense")
        self.assertFalse(e0.query_expansion)
        self.assertEqual(h0.strategy, "hybrid_union")
        self.assertEqual(h0.fusion, "parent_id_set_union")
        self.assertEqual(e1.strategy, "bm25")
        self.assertEqual(e1.dense_limit_per_query, 0)
        self.assertEqual(e3.fusion, "rrf")
        self.assertEqual(e3.rrf_k, 60)
        self.assertTrue(e4.reranker)
        self.assertEqual(e5.parser_name, "docling")
        self.assertEqual(e5.chunker_version, "structure_aware_v1")
        self.assertFalse(e5.enable_ocr)

        case = _case(
            case_id="q_report",
            paper_keys=("paper_a",),
            evidence=(GoldEvidence("paper_a", 1, 1),),
        )
        outcome = evaluate_case(
            case,
            (_result(paper_id=self.paper_ids["paper_a"], page=1, score=1.0),),
            paper_ids=self.paper_ids,
            retrieval_latency_ms=1.0,
        )
        report = render_summary_markdown(experiment=e0, split="dev", summary=summarize_evaluations((outcome,)))
        self.assertIn("Question Recall@1", report)
        self.assertIn("e0_dense", report)

    def test_loader_parses_a_valid_jsonl_case(self) -> None:
        raw_case = {
            "id": "q_jsonl",
            "split": "dev",
            "paper_keys": ["paper_a"],
            "question": "方法是什么？",
            "question_language": "zh",
            "task_type": "semantic_explanation",
            "answerable": True,
            "gold_evidence": [{"paper_key": "paper_a", "page_start": 2, "page_end": 2, "section_hint": "方法"}],
            "expected_answer": "论文在方法部分解释了核心机制。",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "cases.jsonl"
            dataset_path.write_text(json.dumps(raw_case, ensure_ascii=False) + "\n", encoding="utf-8")
            cases = load_eval_cases(dataset_path)
        self.assertEqual(cases[0].case_id, "q_jsonl")
        self.assertEqual(cases[0].gold_evidence[0].page_start, 2)


if __name__ == "__main__":
    unittest.main()
