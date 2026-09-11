"""Claim/citation and numeric fidelity contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.answer_contract import enforce_answer_contract  # noqa: E402
from researchpilot.claim_verification import extract_numeric_values  # noqa: E402
from researchpilot.evidence_alignment import (  # noqa: E402
    align_claims,
    ensure_annotation_citation_markers,
    ensure_factual_citation_markers,
    format_citation_markers_with_pages,
    normalize_citation_markers,
)
from researchpilot.schemas import Answer, Citation  # noqa: E402


def _citation(evidence: str, *, paper_id: str = "paper_a") -> Citation:
    return Citation(paper_id, "chunk_a", 7, 7, evidence)


class AnswerContractTests(unittest.TestCase):
    def test_parenthesized_and_page_qualified_markers_are_normalized(self) -> None:
        answer = enforce_answer_contract(
            Answer(
                text="结论一（证据 1）。\n结论二 [证据 2 | 第 8 页]。",
                citations=(_citation("结论一"), _citation("结论二")),
                refused=False,
                trace_id="trace_marker_variants",
                latency_ms=1.0,
            )
        )
        assert answer.text == "结论一[证据 1]。\n结论二 [证据 2]。"
        assert all(claim.support_status == "supported" for claim in answer.claims)

    def test_compound_page_qualified_markers_are_normalized(self) -> None:
        citations = (_citation("甲"), Citation("paper_a", "chunk_b", 8, 8, "乙"))
        text = "结论。[证据 1，第 7 页；证据 2，第 8 页]\n补充（证据 1、证据 2）。"

        normalized = normalize_citation_markers(text, len(citations))

        self.assertEqual(normalized, "结论。[证据 1、2]\n补充[证据 1、2]。")

    def test_bare_evidence_labels_are_normalized_but_ranges_are_not_guessed(self) -> None:
        normalized = normalize_citation_markers(
            "方法来自证据 1 和证据 2。范围写作证据 1-8。",
            8,
        )

        self.assertEqual(normalized, "方法来自[证据 1、2]。范围写作证据 1-8。")

    def test_table_headers_and_bold_labels_are_not_factual_claims(self) -> None:
        claims = align_claims(
            "**实验结果**\n\n| 数据集 | 胜率 |\n|---|---|\n| Legal | 85.6% | [证据 1]",
            (_citation("Legal 85.6%"),),
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].support_status, "supported")

    def test_annotation_relationship_gets_paper_attribution_but_note_does_not(self) -> None:
        text = """**论文原文**：
- 论文声称更高效。

**用户批注**：
> 我认为没有控制模型规模。

**二者关系**：
- 原文没有说明是否控制规模。
"""
        normalized = ensure_annotation_citation_markers(text, 2)

        self.assertIn("论文声称更高效。 [证据 1、2]", normalized)
        self.assertNotIn("我认为没有控制模型规模。 [证据", normalized)
        self.assertIn("原文没有说明是否控制规模。 [证据 1、2]", normalized)

        claims = align_claims(
            "用户批注内容为：没有控制模型规模。【用户批注】",
            (_citation("效率"), _citation("模型规模")),
        )
        note_claim = next(claim for claim in claims if "【用户批注】" in claim.text)
        self.assertEqual(note_claim.support_status, "not_applicable")

    def test_annotation_fallback_reuses_at_most_three_observed_citations(self) -> None:
        normalized = ensure_annotation_citation_markers(
            "**论文原文**：\n- 原文结论。[证据 5]\n**二者关系**：\n- 需要核验。",
            10,
        )

        self.assertIn("需要核验。 [证据 5]", normalized)
        self.assertNotIn("证据 1、2、3、4、5、6、7、8、9、10", normalized)

    def test_generated_user_note_line_is_marked_not_applicable(self) -> None:
        normalized = ensure_annotation_citation_markers(
            "**用户批注**：\n- 页码 1：这里没有控制模型规模。",
            2,
        )
        claims = align_claims(normalized, (_citation("效率"), _citation("实验")))

        self.assertIn("【用户批注】", normalized)
        self.assertEqual(claims[0].support_status, "not_applicable")

    def test_uncited_synthesis_reuses_observed_sources_but_boundary_is_not_applicable(self) -> None:
        text = """- 方法事实。[证据 2]
- 方法与结果形成对应关系。

当前证据不足，无法确认独立消融效果。
"""
        normalized = ensure_factual_citation_markers(text, 3)
        claims = align_claims(normalized, (_citation("A"), _citation("方法与结果"), _citation("C")))

        self.assertIn("方法与结果形成对应关系。 [证据 2]", normalized)
        boundary = next(claim for claim in claims if "无法确认" in claim.text)
        self.assertEqual(boundary.support_status, "not_applicable")

    def test_claim_is_bound_to_numbered_evidence(self) -> None:
        answer = Answer("Legal 胜率为 85.6%。[证据 1]", (_citation("Legal 85.6 %"),), False, "t", 1)
        checked = enforce_answer_contract(answer)
        self.assertEqual(checked.claims[0].citation_ids, ("e1",))
        self.assertEqual(checked.claims[0].support_status, "supported")
        self.assertTrue(checked.numeric_fidelity_passed)

    def test_unsupported_number_is_removed_from_answer(self) -> None:
        answer = Answer("Legal 胜率为 99.9%。[证据 1]", (_citation("Legal 85.6%"),), False, "t", 1)
        checked = enforce_answer_contract(answer)
        self.assertNotIn("99.9%", checked.text)
        self.assertFalse(checked.numeric_fidelity_passed)
        self.assertIn("数字核验提示", checked.text)

    def test_unsupported_number_in_a_bullet_is_removed_as_a_whole_claim(self) -> None:
        answer = Answer(
            "- Legal 胜率为 99.9%。[证据 1]\n- 方法采用图检索。[证据 1]",
            (_citation("Legal 胜率为 85.6%，方法采用图检索。"),),
            False,
            "t",
            1,
        )
        checked = enforce_answer_contract(answer)
        self.assertNotIn("99.9%", checked.text)
        self.assertIn("方法采用图检索", checked.text)

    def test_numeric_removal_does_not_leave_dangling_intro(self) -> None:
        answer = Answer(
            "增量更新阶段的效率优势：\n- Token 数为 9999。[证据 1]",
            (_citation("Token 数为 100。"),),
            False,
            "t",
            1,
        )

        checked = enforce_answer_contract(answer)

        self.assertNotIn("9999", checked.text)
        self.assertNotIn("增量更新阶段的效率优势：", checked.text)
        self.assertIn("数字核验提示", checked.text)

    def test_scaled_numbers_are_equivalent(self) -> None:
        self.assertEqual(extract_numeric_values("共有 1.2M tokens"), ("1200000tokens",))
        self.assertEqual(extract_numeric_values("共有 1,200,000 tokens"), ("1200000tokens",))

    def test_refusal_has_empty_contract(self) -> None:
        answer = Answer("证据不足。", (), True, "t", 1, "no evidence")
        checked = enforce_answer_contract(answer)
        self.assertEqual(checked.claims, ())
        self.assertTrue(checked.numeric_fidelity_passed)

    def test_unbound_claim_is_structurally_marked_without_rewriting_answer(self) -> None:
        answer = Answer("这是论文事实。", (_citation("这是论文事实。"),), False, "t", 1)
        checked = enforce_answer_contract(answer)
        self.assertEqual(checked.claims[0].support_status, "unsupported")
        self.assertEqual(checked.text, answer.text)

    def test_grouped_citation_ids_are_all_bound_and_rendered_with_pages(self) -> None:
        citations = (
            _citation("方法证据"),
            Citation("paper_a", "chunk_b", 8, 9, "结果证据"),
        )

        claims = align_claims("方法与结果相互对应。[证据 1、2]", citations)
        rendered = format_citation_markers_with_pages("方法与结果相互对应。[证据 1、2]", citations)

        self.assertEqual(claims[0].citation_ids, ("e1", "e2"))
        self.assertIn("证据 1 · 第 7 页", rendered)
        self.assertIn("证据 2 · 第 8-9 页", rendered)

    def test_observed_model_citation_styles_are_normalized(self) -> None:
        text = "\n".join(
            (
                "甲。[证据1]",
                "乙。[1][2]",
                "丙。<sup>[2]</sup>",
                "丁。【证据 1】",
                "戊。［证据2］",
            )
        )

        normalized = normalize_citation_markers(text, 2)

        self.assertIn("甲。[证据 1]", normalized)
        self.assertIn("乙。[证据 1、2]", normalized)
        self.assertIn("丙。[证据 2]", normalized)
        self.assertIn("丁。[证据 1]", normalized)
        self.assertIn("戊。[证据 2]", normalized)

    def test_plain_bibliography_marker_inside_prose_is_not_rewritten(self) -> None:
        text = "Smith [1] 提出该方法，发表于 [2024]。"
        self.assertEqual(normalize_citation_markers(text, 10), text)

    def test_unused_citations_are_pruned_and_markers_are_compacted(self) -> None:
        citations = (
            _citation("未使用证据"),
            Citation("paper_a", "chunk_b", 8, 8, "实际证据"),
        )
        answer = Answer("结论成立。[2]", citations, False, "t", 1)

        checked = enforce_answer_contract(answer)

        self.assertEqual(checked.text, "结论成立。[证据 1]")
        self.assertEqual(tuple(item.chunk_id for item in checked.citations), ("chunk_b",))
        self.assertEqual(checked.claims[0].citation_ids, ("e1",))

    def test_answer_contract_is_idempotent(self) -> None:
        answer = Answer("Legal 胜率为 99.9%。[1]", (_citation("Legal 85.6%"),), False, "t", 1)

        once = enforce_answer_contract(answer)
        twice = enforce_answer_contract(once)

        self.assertEqual(twice, once)
        self.assertEqual(once.text.count("数字核验提示"), 1)


if __name__ == "__main__":
    unittest.main()
