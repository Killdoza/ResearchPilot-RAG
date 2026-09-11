"""P2-B1 规则 Query Analyzer 的离线边界测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.query_analysis import RuleQueryAnalyzer  # noqa: E402
from researchpilot.query_plan import ScopeSource  # noqa: E402
from researchpilot.schemas import QueryType  # noqa: E402
from researchpilot.scope_resolution import PaperScopeCatalog, PaperScopeEntry  # noqa: E402


class QueryAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = PaperScopeCatalog(
            (
                PaperScopeEntry("self_rag_2023", "Self-RAG: Learning to Retrieve", "Self_RAG.pdf"),
                PaperScopeEntry("r2ag_2024", "R2AG: Retrieval Information", "R2AG.pdf"),
            )
        )
        self.analyzer = RuleQueryAnalyzer(catalog=self.catalog)

    def analyze(self, question: str, selected: tuple[str, ...] = ()):  # type: ignore[no-untyped-def]
        return self.analyzer.analyze(
            question=question,
            session_available_paper_keys=("self_rag_2023", "r2ag_2024"),
            user_selected_paper_keys=selected,
        )

    def test_cross_document_question_requires_resolved_two_paper_scope(self) -> None:
        result = self.analyze("比较 Self-RAG 和 R2AG 的检索信息如何使用？")
        self.assertEqual(result.query_type, QueryType.CROSS_DOCUMENT)
        self.assertEqual(result.scope.source, ScopeSource.TITLE_RESOLVED)
        self.assertEqual(result.scope.paper_keys, ("self_rag_2023", "r2ag_2024"))

    def test_user_selected_scope_has_priority(self) -> None:
        result = self.analyze("Self-RAG 的作者是谁？", selected=("r2ag_2024",))
        self.assertEqual(result.query_type, QueryType.METADATA)
        self.assertEqual(result.scope.source, ScopeSource.USER_SELECTED)
        self.assertEqual(result.scope.paper_keys, ("r2ag_2024",))

    def test_missing_scope_is_a_safe_fallback(self) -> None:
        result = self.analyze("请比较两篇论文的实验设置")
        self.assertIsNone(result.query_type)
        self.assertFalse(result.needs_retrieval)
        self.assertEqual(result.fallback_reason, "scope_insufficient")

    def test_single_paper_comparison_does_not_use_global_candidates(self) -> None:
        result = self.analyze("比较 Self-RAG 和其他论文的实验设置")
        self.assertIsNone(result.query_type)
        self.assertFalse(result.needs_retrieval)
        self.assertEqual(result.fallback_reason, "scope_insufficient:comparison_requires_two_papers")

    def test_unanswerable_language_is_a_risk_not_a_gold_label(self) -> None:
        result = self.analyze("Self-RAG 有没有报告训练成本？")
        self.assertEqual(result.query_type, QueryType.SEMANTIC_EXPLANATION)
        self.assertTrue(result.unanswerable_risk)

    def test_general_chat_never_looks_like_paper_retrieval(self) -> None:
        result = self.analyze("你好")
        self.assertEqual(result.query_type, QueryType.GENERAL_CHAT)
        self.assertFalse(result.needs_retrieval)
        self.assertEqual(result.scope.source, ScopeSource.NONE)


if __name__ == "__main__":
    unittest.main()
