"""运行时 Scope 解析测试：只读取问题、会话目录与用户选择，不读取评测 Gold。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.query_plan import ScopeSource  # noqa: E402
from researchpilot.schemas import SchemaValidationError  # noqa: E402
from researchpilot.scope_resolution import PaperScopeCatalog, PaperScopeEntry  # noqa: E402


class ScopeResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = PaperScopeCatalog(
            (
                PaperScopeEntry("self_rag_2023", "Self-RAG: Learning to Retrieve", "2023_Self_RAG.pdf"),
                PaperScopeEntry("r2ag_2024", "R2AG: Retrieval Information", "2024_R2AG.pdf"),
                PaperScopeEntry("vhelm_2024", "VHELM: Holistic Evaluation", "2024_VHELM.pdf"),
            )
        )

    def test_user_selected_scope_has_priority_over_title_matching(self) -> None:
        resolution = self.catalog.resolve(
            question="Self-RAG 与 R2AG 的机制有什么差异？",
            available_paper_keys=("self_rag_2023", "r2ag_2024", "vhelm_2024"),
            user_selected_paper_keys=("vhelm_2024",),
        )
        self.assertEqual(resolution.paper_keys, ("vhelm_2024",))
        self.assertEqual(resolution.source, ScopeSource.USER_SELECTED)

    def test_title_and_file_aliases_resolve_only_available_papers(self) -> None:
        resolution = self.catalog.resolve(
            question="请比较 Self-RAG 和 R2AG 的检索信息使用方式。",
            available_paper_keys=("self_rag_2023", "r2ag_2024"),
        )
        self.assertEqual(resolution.paper_keys, ("self_rag_2023", "r2ag_2024"))
        self.assertEqual(resolution.source, ScopeSource.TITLE_RESOLVED)

    def test_unknown_question_returns_scope_insufficient_not_a_guess(self) -> None:
        resolution = self.catalog.resolve(
            question="请比较两篇文章的训练策略。",
            available_paper_keys=("self_rag_2023", "r2ag_2024"),
        )
        self.assertEqual(resolution.paper_keys, ())
        self.assertEqual(resolution.source, ScopeSource.NONE)
        self.assertEqual(resolution.fallback_reason, "scope_insufficient:no_title_or_alias_match")

    def test_selection_outside_session_scope_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "当前会话 Scope"):
            self.catalog.resolve(
                question="Self-RAG 是什么？",
                available_paper_keys=("self_rag_2023",),
                user_selected_paper_keys=("r2ag_2024",),
            )

    def test_same_title_is_ambiguous_not_a_cross_document_scope(self) -> None:
        catalog = PaperScopeCatalog(
            (
                PaperScopeEntry("paper_a", "A Shared Title", "a.pdf"),
                PaperScopeEntry("paper_b", "A Shared Title", "b.pdf"),
            )
        )
        resolution = catalog.resolve(
            question="A Shared Title 的方法是什么？",
            available_paper_keys=("paper_a", "paper_b"),
        )
        self.assertEqual(resolution.paper_keys, ())
        self.assertEqual(resolution.source, ScopeSource.NONE)
        self.assertEqual(resolution.fallback_reason, "scope_insufficient:ambiguous_title_or_alias")


if __name__ == "__main__":
    unittest.main()
