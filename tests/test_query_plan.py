"""P2 规则路由器只测确定性策略，不调用检索器或 LLM。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.query_plan import (  # noqa: E402
    P2RulesRouter,
    P2RunMode,
    QueryPlan,
    QueryTypeSource,
    ScopeSource,
)
from researchpilot.schemas import QueryType, SchemaValidationError  # noqa: E402


class QueryPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = P2RulesRouter()

    def test_cross_document_is_balanced_and_serializable(self) -> None:
        plan = self.router.plan(query_type=QueryType.CROSS_DOCUMENT, target_paper_keys=("paper_a", "paper_b"))
        self.assertEqual(plan.retrieval_policy, "balanced_cross_document")
        self.assertTrue(plan.coverage_gate_required)
        self.assertEqual(plan.minimum_evidence_per_paper, 1)
        self.assertEqual(plan.to_dict()["target_paper_keys"], ["paper_a", "paper_b"])
        self.assertEqual(plan.to_dict()["experiment_mode"], "static")

    def test_general_chat_skips_paper_retrieval(self) -> None:
        plan = self.router.plan(query_type=QueryType.GENERAL_CHAT, target_paper_keys=("ignored",))
        self.assertFalse(plan.needs_retrieval)
        self.assertEqual(plan.dense_limit + plan.sparse_limit, 0)

    def test_keyword_query_gives_sparse_budget_and_invalid_plan_is_rejected(self) -> None:
        plan = self.router.plan(query_type=QueryType.KEYWORD_EXACT, target_paper_keys=("paper_a",))
        self.assertGreater(plan.sparse_limit, plan.dense_limit)
        with self.assertRaisesRegex(SchemaValidationError, "target_paper_keys"):
            QueryPlan(
                query_type=QueryType.LOCAL_FACT,
                target_paper_keys=(),
                retrieval_policy="local",
                dense_limit=1,
                sparse_limit=1,
                rerank_enabled=True,
                block_types=("paragraph",),
                minimum_evidence_per_paper=0,
                allow_one_retry=True,
                coverage_gate_required=False,
                needs_retrieval=True,
            )

    def test_oracle_and_end_to_end_sources_cannot_be_mixed(self) -> None:
        oracle = self.router.plan(
            query_type=QueryType.CROSS_DOCUMENT,
            target_paper_keys=("paper_a", "paper_b"),
            experiment_mode=P2RunMode.ORACLE,
            query_type_source=QueryTypeSource.ORACLE_EVAL,
            scope_source=ScopeSource.ORACLE_EVAL,
        )
        self.assertEqual(oracle.to_dict()["scope_source"], "oracle_eval")
        with self.assertRaisesRegex(SchemaValidationError, "端到端"):
            self.router.plan(
                query_type=QueryType.CROSS_DOCUMENT,
                target_paper_keys=("paper_a", "paper_b"),
                experiment_mode=P2RunMode.END_TO_END,
                query_type_source=QueryTypeSource.ORACLE_EVAL,
                scope_source=ScopeSource.ORACLE_EVAL,
            )


if __name__ == "__main__":
    unittest.main()
