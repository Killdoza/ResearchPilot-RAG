"""产品运行时规则路由测试；不加载模型、不读取评测 Gold。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.routing import (  # noqa: E402
    RuntimePaper,
    RuntimeRequest,
    RuntimeRoute,
    RuntimeRouter,
)


class RuntimeRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = RuntimeRouter()
        self.paper_a = RuntimePaper("paper_a", "UICompass.pdf", page_count=12)
        self.paper_b = RuntimePaper("paper_b", "RAG.pdf", page_count=15)

    def _request(
        self,
        question: str,
        *,
        papers: tuple[RuntimePaper, ...] = (),
        selected: tuple[str, ...] = (),
    ) -> RuntimeRequest:
        return RuntimeRequest(
            question=question,
            user_id="user_a",
            session_id="session_a",
            available_papers=papers,
            selected_paper_ids=selected,
        )

    def test_general_chat_never_enters_paper_retrieval(self) -> None:
        decision = self.router.route(self._request("你好", papers=(self.paper_a,)))

        self.assertEqual(decision.route, RuntimeRoute.GENERAL_CHAT)
        self.assertFalse(decision.needs_retrieval)
        self.assertEqual(decision.target_paper_ids, ())

    def test_metadata_uses_selected_scope_without_body_retrieval(self) -> None:
        decision = self.router.route(
            self._request("这篇论文的作者是谁？", papers=(self.paper_a, self.paper_b), selected=("paper_a",))
        )

        self.assertEqual(decision.route, RuntimeRoute.METADATA)
        self.assertEqual(decision.target_paper_ids, ("paper_a",))
        self.assertTrue(decision.needs_retrieval)
        self.assertEqual(decision.query_type.value, "metadata")

    def test_summary_accepts_verb_first_word_order(self) -> None:
        decision = self.router.route(self._request("请总结这篇论文", papers=(self.paper_a,), selected=("paper_a",)))

        self.assertEqual(decision.route, RuntimeRoute.DOCUMENT_SUMMARY)
        self.assertEqual(decision.query_type.value, "document_summary")

    def test_summary_accepts_follow_up_phrase_used_by_product_regression(self) -> None:
        decision = self.router.route(self._request("请再概括一下这篇论文", papers=(self.paper_a,)))

        self.assertEqual(decision.route, RuntimeRoute.DOCUMENT_SUMMARY)
        self.assertEqual(decision.query_type.value, "document_summary")

    def test_experimental_setup_has_dedicated_route(self) -> None:
        decision = self.router.route(
            self._request("论文的实验设置是什么？", papers=(self.paper_a,), selected=("paper_a",))
        )

        self.assertEqual(decision.route, RuntimeRoute.EXPERIMENTAL_SETUP)
        self.assertEqual(decision.query_type.value, "experimental_setup")

    def test_experimental_results_are_not_misclassified_as_setup(self) -> None:
        decision = self.router.route(self._request("论文报告了哪些主要实验结果？", papers=(self.paper_a,)))

        self.assertEqual(decision.route, RuntimeRoute.EXPERIMENTAL_RESULTS)
        self.assertEqual(decision.query_type.value, "experimental_results")

    def test_single_paper_baseline_comparison_is_an_experimental_result(self) -> None:
        decision = self.router.route(
            self._request(
                "LightRAG 与 GraphRAG 的主要胜率比较是什么？",
                papers=(self.paper_a,),
                selected=("paper_a",),
            )
        )

        self.assertEqual(decision.route, RuntimeRoute.EXPERIMENTAL_RESULTS)
        self.assertEqual(decision.target_paper_ids, ("paper_a",))

    def test_method_and_result_relation_uses_multi_section_route(self) -> None:
        decision = self.router.route(
            self._request(
                "图索引、双层检索与实验优势之间有什么联系？",
                papers=(self.paper_a,),
                selected=("paper_a",),
            )
        )

        self.assertEqual(decision.route, RuntimeRoute.MULTI_SECTION)
        self.assertEqual(decision.query_type.value, "multi_section")

    def test_problem_and_method_correspondence_uses_multi_section_route(self) -> None:
        decision = self.router.route(
            self._request(
                "方法设计如何对应论文提出的研究问题？",
                papers=(self.paper_a,),
            )
        )

        self.assertEqual(decision.route, RuntimeRoute.MULTI_SECTION)
        self.assertEqual(decision.query_type.value, "multi_section")

    def test_detail_question_defaults_to_all_current_papers(self) -> None:
        decision = self.router.route(self._request("核心方法是什么？", papers=(self.paper_a, self.paper_b)))

        self.assertEqual(decision.route, RuntimeRoute.DOCUMENT_DETAIL)
        self.assertTrue(decision.needs_retrieval)
        self.assertEqual(decision.target_paper_ids, ("paper_a", "paper_b"))

    def test_comparison_requires_two_papers_in_scope(self) -> None:
        decision = self.router.route(
            self._request("比较这两篇论文的实验结果", papers=(self.paper_a,), selected=("paper_a",))
        )

        self.assertEqual(decision.route, RuntimeRoute.SCOPE_INSUFFICIENT)
        self.assertEqual(decision.reason, "scope_insufficient:comparison_requires_two_papers")

    def test_cross_document_route_remains_explicit(self) -> None:
        decision = self.router.route(self._request("比较两篇论文的实验结果", papers=(self.paper_a, self.paper_b)))

        self.assertEqual(decision.route, RuntimeRoute.CROSS_DOCUMENT)
        self.assertFalse(decision.needs_retrieval)
        self.assertEqual(decision.target_paper_ids, ("paper_a", "paper_b"))

    def test_selected_scope_must_be_visible_in_session_catalog(self) -> None:
        with self.assertRaises(ValueError):
            self._request("方法是什么？", papers=(self.paper_a,), selected=("paper_other",))


if __name__ == "__main__":
    unittest.main()
