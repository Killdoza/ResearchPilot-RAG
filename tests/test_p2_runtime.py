"""P2-B0 运行时边界测试：Oracle 与端到端入口不能混用。"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.p2_runtime import (  # noqa: E402
    GoldAnnotation,
    P2RuntimePlanner,
    RuntimeRequest,
)
from researchpilot.query_plan import (  # noqa: E402
    P2RunMode,
    QueryTypeSource,
    ScopeSource,
)
from researchpilot.schemas import QueryType, SchemaValidationError  # noqa: E402
from researchpilot.scope_resolution import PaperScopeCatalog, PaperScopeEntry  # noqa: E402


class P2RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = PaperScopeCatalog(
            (
                PaperScopeEntry("self_rag_2023", "Self-RAG: Learning to Retrieve", "Self_RAG.pdf"),
                PaperScopeEntry("r2ag_2024", "R2AG: Retrieval Information", "R2AG.pdf"),
            )
        )
        self.planner = P2RuntimePlanner(catalog=self.catalog)
        self.request = RuntimeRequest(
            question="Self-RAG 和 R2AG 的检索机制有什么不同？",
            session_available_paper_keys=("self_rag_2023", "r2ag_2024"),
        )
        self.gold = GoldAnnotation(
            case_id="CD901",
            gold_query_type=QueryType.CROSS_DOCUMENT,
            required_paper_keys=("self_rag_2023", "r2ag_2024"),
            answerable=True,
            gold_evidence=(),
        )

    def test_oracle_plan_is_explicitly_marked_as_oracle(self) -> None:
        outcome = self.planner.plan_oracle(request=self.request, gold=self.gold)
        assert outcome.plan is not None
        self.assertEqual(outcome.mode, P2RunMode.ORACLE)
        self.assertEqual(outcome.plan.experiment_mode, P2RunMode.ORACLE)
        self.assertEqual(outcome.plan.query_type_source, QueryTypeSource.ORACLE_EVAL)
        self.assertEqual(outcome.plan.scope_source, ScopeSource.ORACLE_EVAL)

    def test_scoped_plan_only_uses_user_selected_scope(self) -> None:
        request = RuntimeRequest(
            question="请比较两篇论文。",
            session_available_paper_keys=("self_rag_2023", "r2ag_2024"),
            user_selected_paper_keys=("self_rag_2023", "r2ag_2024"),
            experiment_mode=P2RunMode.SCOPED,
        )
        outcome = self.planner.plan_scoped(request=request)
        assert outcome.plan is not None
        self.assertEqual(outcome.plan.scope_source, ScopeSource.USER_SELECTED)
        self.assertEqual(outcome.plan.target_paper_keys, request.user_selected_paper_keys)

    def test_end_to_end_resolves_titles_without_gold_parameter(self) -> None:
        parameters = inspect.signature(self.planner.plan_end_to_end).parameters
        self.assertNotIn("gold", parameters)
        self.assertNotIn("required_paper_keys", parameters)
        self.assertNotIn("predicted_query_type", parameters)
        outcome = self.planner.plan_end_to_end(request=self.request)
        assert outcome.plan is not None
        self.assertEqual(outcome.plan.experiment_mode, P2RunMode.END_TO_END)
        self.assertEqual(outcome.plan.scope_source, ScopeSource.TITLE_RESOLVED)
        with self.assertRaises(TypeError):
            self.planner.plan_end_to_end(  # type: ignore[call-arg]
                request=self.request,
                gold=self.gold,
            )

    def test_end_to_end_scope_failure_does_not_guess_gold(self) -> None:
        request = RuntimeRequest(
            question="请比较两篇论文的实验设置。",
            session_available_paper_keys=("self_rag_2023", "r2ag_2024"),
        )
        outcome = self.planner.plan_end_to_end(request=request)
        self.assertIsNone(outcome.plan)
        self.assertTrue(outcome.is_scope_insufficient)

    def test_runtime_request_rejects_gold_field_and_oracle_mode(self) -> None:
        with self.assertRaises(TypeError):
            RuntimeRequest(  # type: ignore[call-arg]
                question="问题",
                session_available_paper_keys=("self_rag_2023",),
                required_paper_keys=("self_rag_2023", "r2ag_2024"),
            )
        with self.assertRaisesRegex(SchemaValidationError, "Oracle 请求"):
            RuntimeRequest(
                question="问题",
                session_available_paper_keys=("self_rag_2023",),
                experiment_mode=P2RunMode.ORACLE,
            )


if __name__ == "__main__":
    unittest.main()
