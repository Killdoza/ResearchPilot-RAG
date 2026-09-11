"""P2-C Dry-run：不调用 Retriever、LLM 或 Gold 泄漏到端到端 Planner。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.crossdoc import CrossDocumentEvidence, ExpectedClaim  # noqa: E402
from researchpilot.evaluation import CORPUS_SCHEMA_VERSION, CorpusManifest, CorpusPaper, ExperimentConfig  # noqa: E402
from researchpilot.p2_dataset import P2Case  # noqa: E402
from researchpilot.p2_runner import P2ExecutionConfig, build_p2_dry_run, load_p2_execution_config  # noqa: E402
from researchpilot.p2_runtime import P2RuntimePlanner  # noqa: E402
from researchpilot.query_plan import P2RunMode  # noqa: E402
from researchpilot.scope_resolution import PaperScopeCatalog  # noqa: E402


def _experiment(name: str) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        description="test",
        parser_name="docling",
        parser_version="2",
        enable_ocr=False,
        chunker_version="structure_aware_v1",
        parent_chunk_size=1200,
        parent_chunk_overlap=200,
        child_chunk_size=300,
        child_chunk_overlap=50,
        embedding_model="embed",
        reranker_model="rerank",
        generation_model=None,
        strategy="hybrid_rrf",
        query_expansion=False,
        query_expansion_count=0,
        dense_limit_per_query=10,
        bm25_limit_per_query=10,
        fusion="rrf",
        reranker=True,
        result_limit=10,
        rrf_k=60,
    )


class P2RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        corpus = CorpusManifest(
            CORPUS_SCHEMA_VERSION,
            "test corpus",
            (
                CorpusPaper("paper_a", "Paper A", "rag", 2024, "https://a.example", "a.pdf", "a" * 64),
                CorpusPaper("paper_b", "Paper B", "rag", 2024, "https://b.example", "b.pdf", "b" * 64),
            ),
        )
        self.planner = P2RuntimePlanner(catalog=PaperScopeCatalog.from_corpus(corpus))
        self.case = P2Case(
            case_id="p2_q1",
            source_id="source_1",
            split="dev",
            task_type="cross_document",
            answerable=True,
            paper_keys=("paper_a", "paper_b"),
            required_paper_keys=("paper_a", "paper_b"),
            comparison_dimension="method",
            comparison_allowed=True,
            refusal_reason_target=None,
            question="比较 Paper A 和 Paper B 的方法",
            expected_answer="answer",
            expected_claims=(ExpectedClaim("C1", "a"), ExpectedClaim("C2", "b"), ExpectedClaim("C3", "compare")),
            gold_evidence=(
                CrossDocumentEvidence("e1", ("C1", "C3"), "paper_a", 1, 1, "M", "a"),
                CrossDocumentEvidence("e2", ("C2", "C3"), "paper_b", 1, 1, "M", "b"),
            ),
        )

    def config(self, mode: P2RunMode) -> P2ExecutionConfig:
        return P2ExecutionConfig(_experiment(mode.value), mode, 3, 16000, 1000, 0.0, None, None, "u", "s")

    def test_oracle_and_end_to_end_are_explicitly_distinguished(self) -> None:
        oracle_records, _ = build_p2_dry_run(
            cases=(self.case,), config=self.config(P2RunMode.ORACLE), planner=self.planner
        )
        end_records, _ = build_p2_dry_run(
            cases=(self.case,), config=self.config(P2RunMode.END_TO_END), planner=self.planner
        )
        self.assertEqual(oracle_records[0].query_type_source, "oracle_eval")
        self.assertEqual(oracle_records[0].scope_source, "oracle_eval")
        self.assertEqual(end_records[0].query_type_source, "rule_predicted")
        self.assertEqual(end_records[0].scope_source, "title_resolved")

    def test_budget_accounts_for_one_retry_without_model_call(self) -> None:
        records, summary = build_p2_dry_run(
            cases=(self.case,), config=self.config(P2RunMode.SCOPED), planner=self.planner
        )
        self.assertEqual(records[0].retrieval_calls_upper_bound, 4)
        self.assertEqual(records[0].model_calls_upper_bound, 1)
        self.assertEqual(summary.prompt_tokens_upper_bound, records[0].estimated_prompt_tokens_upper_bound)

    def test_repository_configs_provide_a_complete_budget_contract(self) -> None:
        config = load_p2_execution_config(PROJECT_ROOT / "configs/experiments/e6_end_to_end.yaml")
        self.assertEqual(config.mode, P2RunMode.END_TO_END)
        self.assertEqual(config.max_input_tokens, 16000)
        self.assertEqual(config.max_output_tokens, 1200)


if __name__ == "__main__":
    unittest.main()
