"""P1-E 上下文选择、统一回答 JSON 与引用指标测试；不访问模型/Chroma。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.answer_evaluation import (  # noqa: E402
    AnswerCitation,
    GeneratedAnswer,
    evaluate_generated_answer,
    summarize_answer_evaluations,
)
from researchpilot.answer_runner import (  # noqa: E402
    AnswerExperimentConfig,
    ModelCompletion,
    generate_structured_answer,
    load_answer_experiment_config,
)
from researchpilot.context_modes import E5BlockRepository, full_context  # noqa: E402
from researchpilot.evaluation import CorpusManifest, EvalCase, ExperimentConfig, GoldEvidence  # noqa: E402


def _manifest() -> CorpusManifest:
    return CorpusManifest.from_dict(
        {
            "schema_version": "ai_cs_corpus_v0.1",
            "description": "P1-E 测试论文",
            "papers": [
                {
                    "paper_key": "paper_a",
                    "title": "Paper A",
                    "area": "ai",
                    "year": 2025,
                    "source_url": "https://example.com/a.pdf",
                    "local_filename": "a.pdf",
                    "sha256": "a" * 64,
                },
                {
                    "paper_key": "paper_b",
                    "title": "Paper B",
                    "area": "cs",
                    "year": 2025,
                    "source_url": "https://example.com/b.pdf",
                    "local_filename": "b.pdf",
                    "sha256": "b" * 64,
                },
            ],
        }
    )


def _experiment() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "schema_version": "researchpilot_retrieval_experiment_v1",
            "name": "test_lc1",
            "description": "测试全文上下文",
            "ingestion": {
                "parser_name": "docling",
                "parser_version": "2.120.2",
                "enable_ocr": False,
                "chunker_version": "structure_aware_v1",
                "parent_chunk_size": 1200,
                "parent_chunk_overlap": 200,
                "child_chunk_size": 300,
                "child_chunk_overlap": 50,
            },
            "models": {
                "embedding_model": "test-embedding",
                "reranker_model": None,
                "generation_model": None,
            },
            "retrieval": {
                "strategy": "dense",
                "query_expansion": False,
                "query_expansion_count": 0,
                "dense_limit_per_query": 10,
                "bm25_limit_per_query": 0,
                "fusion": "none",
                "reranker": False,
                "result_limit": 10,
                "rrf_k": 0,
            },
        }
    )


class _FakeClient:
    def complete(self, **_: object) -> ModelCompletion:
        return ModelCompletion(
            text=(
                '{"answer":"根据论文，方法有效。","citations":'
                '[{"paper_key":"paper_a","page":1,"context_id":"B0001"}],'
                '"refused":false,"refusal_reason":null}'
            ),
            prompt_tokens=123,
            completion_tokens=45,
        )


class _BadRefusalClient:
    def complete(self, **_: object) -> ModelCompletion:
        return ModelCompletion(
            text=(
                '{"answer":"证据不足。","citations":'
                '[{"paper_key":"paper_a","page":1,"context_id":"B0001"}],'
                '"refused":true,"refusal_reason":"证据不足。"}'
            )
        )


class _MissingRefusalReasonClient:
    def complete(self, **_: object) -> ModelCompletion:
        return ModelCompletion(text=('{"answer":"证据不足。","citations":[],"refused":true,"refusal_reason":null}'))


class LongContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        for paper_key, content_hash, text in (
            ("paper_a", "a" * 64, "A 的方法。"),
            ("paper_b", "b" * 64, "B 的方法。"),
        ):
            payload = {
                "schema_version": "parsed_blocks_v1",
                "paper": {"content_hash": content_hash},
                "blocks": [
                    {
                        "block_id": f"block_{paper_key}",
                        "page": 1,
                        "block_type": "paragraph",
                        "section_path": ["方法"],
                        "text": text,
                        "parser_name": "docling",
                        "parser_version": "2.120.2",
                    }
                ],
            }
            (root / f"{content_hash}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self.repository = E5BlockRepository(artifact_dir=root, corpus=self.manifest)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_lc1_lc2_use_frozen_e5_blocks_without_duplicate_context_ids(self) -> None:
        single = full_context(mode="lc1", block_repository=self.repository, paper_keys=("paper_a",))
        multi = full_context(mode="lc2", block_repository=self.repository, paper_keys=("paper_a", "paper_b"))
        self.assertEqual(len(single.references), 1)
        self.assertEqual(len(multi.references), 2)
        self.assertEqual([item.reference_id for item in multi.references], ["B0001", "B0002"])
        self.assertIn("paper_key=paper_b", multi.render())
        with self.assertRaisesRegex(Exception, "LC1"):
            full_context(mode="lc1", block_repository=self.repository, paper_keys=("paper_a", "paper_b"))

    def test_citation_metrics_use_paper_and_page_overlap_and_refusal_is_strict(self) -> None:
        case = EvalCase(
            case_id="q001",
            split="dev",
            paper_keys=("paper_a",),
            question="方法是什么？",
            question_language="zh",
            task_type="semantic_explanation",
            answerable=True,
            gold_evidence=(GoldEvidence("paper_a", 2, 3, "方法"),),
            expected_answer="方法。",
        )
        answer = GeneratedAnswer(
            answer="回答。",
            citations=(AnswerCitation("paper_a", 3, 3), AnswerCitation("paper_b", 3, 3)),
            refused=False,
            refusal_reason=None,
            prompt_tokens=100,
            completion_tokens=10,
            cost=None,
            retrieval_latency_ms=1.0,
            generation_latency_ms=2.0,
        )
        outcome = evaluate_generated_answer(case=case, answer=answer)
        self.assertAlmostEqual(outcome.citation_precision or 0, 0.5)
        self.assertAlmostEqual(outcome.citation_recall or 0, 1.0)
        self.assertAlmostEqual(outcome.citation_f1 or 0, 2 / 3)
        refused_case = EvalCase(
            case_id="q002",
            split="dev",
            paper_keys=("paper_a",),
            question="没有的信息？",
            question_language="zh",
            task_type="unanswerable",
            answerable=False,
            gold_evidence=(),
            expected_answer="拒答。",
        )
        refused = GeneratedAnswer(
            answer="不能可靠回答。",
            citations=(),
            refused=True,
            refusal_reason="证据不足。",
            prompt_tokens=10,
            completion_tokens=5,
            cost=None,
            retrieval_latency_ms=0,
            generation_latency_ms=1,
        )
        self.assertTrue(evaluate_generated_answer(case=refused_case, answer=refused).correct_refusal)
        summary = summarize_answer_evaluations(
            outcomes=(outcome, evaluate_generated_answer(case=refused_case, answer=refused)),
            answers=(answer, refused),
        )
        self.assertAlmostEqual(summary.correct_refusal_rate or 0, 1.0)

    def test_fake_structured_client_produces_unified_answer_record(self) -> None:
        context = full_context(mode="lc1", block_repository=self.repository, paper_keys=("paper_a",))
        config = AnswerExperimentConfig(
            retrieval_experiment=_experiment(),
            context_mode="lc1",
            artifact_dir=Path(self.temp_dir.name),
            parser_name="docling",
            parser_version="2.120.2",
            max_input_tokens=10_000,
            max_output_tokens=100,
            temperature=0.0,
            input_cost_per_million=None,
            output_cost_per_million=None,
            evaluation_user_id="test_user",
            evaluation_session_id="test_session",
        )
        answer = generate_structured_answer(
            question="方法是什么？", context=context, config=config, client=_FakeClient()
        )
        self.assertEqual(answer.prompt_tokens, 123)
        self.assertEqual(answer.citations[0].page_start, 1)
        self.assertFalse(answer.refused)

    def test_four_checked_in_p1e_configs_share_frozen_e5_parser(self) -> None:
        for filename, mode in (
            ("lc1_full_paper.yaml", "lc1"),
            ("lc2_multi_paper.yaml", "lc2"),
            ("n0_dense_rag_on_e5.yaml", "n0"),
            ("a0_advanced_rag_e5.yaml", "a0"),
        ):
            config = load_answer_experiment_config(
                PROJECT_ROOT / "configs" / "experiments" / filename,
                project_root=PROJECT_ROOT,
            )
            self.assertEqual(config.context_mode, mode)
            self.assertEqual(config.retrieval_experiment.parser_name, "docling")
            self.assertEqual(config.retrieval_experiment.chunker_version, "structure_aware_v1")

    def test_refusal_with_citations_is_recorded_as_an_evaluation_error_not_a_runner_crash(self) -> None:
        context = full_context(mode="lc1", block_repository=self.repository, paper_keys=("paper_a",))
        config = AnswerExperimentConfig(
            retrieval_experiment=_experiment(),
            context_mode="lc1",
            artifact_dir=Path(self.temp_dir.name),
            parser_name="docling",
            parser_version="2.120.2",
            max_input_tokens=10_000,
            max_output_tokens=100,
            temperature=0.0,
            input_cost_per_million=None,
            output_cost_per_million=None,
            evaluation_user_id="test_user",
            evaluation_session_id="test_session",
        )
        answer = generate_structured_answer(
            question="问题？", context=context, config=config, client=_BadRefusalClient()
        )
        case = EvalCase(
            case_id="unanswerable",
            split="dev",
            paper_keys=("paper_a",),
            question="论文未提到的信息？",
            question_language="zh",
            task_type="unanswerable",
            answerable=False,
            gold_evidence=(),
            expected_answer="应拒答。",
        )
        self.assertFalse(evaluate_generated_answer(case=case, answer=answer).correct_refusal)

    def test_refusal_without_reason_is_normalized_instead_of_aborting_a_run(self) -> None:
        context = full_context(mode="lc1", block_repository=self.repository, paper_keys=("paper_a",))
        config = AnswerExperimentConfig(
            retrieval_experiment=_experiment(),
            context_mode="lc1",
            artifact_dir=Path(self.temp_dir.name),
            parser_name="docling",
            parser_version="2.120.2",
            max_input_tokens=10_000,
            max_output_tokens=100,
            temperature=0.0,
            input_cost_per_million=None,
            output_cost_per_million=None,
            evaluation_user_id="test_user",
            evaluation_session_id="test_session",
        )
        answer = generate_structured_answer(
            question="问题？", context=context, config=config, client=_MissingRefusalReasonClient()
        )
        self.assertTrue(answer.refused)
        self.assertEqual(answer.refusal_reason, "模型未提供拒答理由。")


if __name__ == "__main__":
    unittest.main()
