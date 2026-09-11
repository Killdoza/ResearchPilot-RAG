"""P2 数据契约与受控 Markdown 导入的纯本地测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.evaluation import CorpusManifest, EvaluationDataError  # noqa: E402
from researchpilot.p2_dataset import P2_SCHEMA_VERSION, P2Case, validate_p2_cases  # noqa: E402


def _corpus() -> CorpusManifest:
    return CorpusManifest.from_dict(
        {
            "schema_version": "ai_cs_corpus_v0.1",
            "description": "P2 单元测试语料",
            "papers": [
                {
                    "paper_key": "paper_a",
                    "title": "A",
                    "area": "rag",
                    "year": 2024,
                    "source_url": "https://example.com/a",
                    "local_filename": "a.pdf",
                    "sha256": "a" * 64,
                },
                {
                    "paper_key": "paper_b",
                    "title": "B",
                    "area": "rag",
                    "year": 2024,
                    "source_url": "https://example.com/b",
                    "local_filename": "b.pdf",
                    "sha256": "b" * 64,
                },
            ],
        }
    )


def _answerable(*, comparison_allowed: bool = False) -> P2Case:
    return P2Case.from_dict(
        {
            "schema_version": P2_SCHEMA_VERSION,
            "id": "CD901",
            "source_id": "CD901",
            "split": "dev",
            "task_type": "cross_document",
            "answerable": True,
            "paper_keys": ["paper_a", "paper_b"],
            "required_paper_keys": ["paper_a", "paper_b"],
            "comparison_dimension": "method",
            "comparison_allowed": comparison_allowed,
            "refusal_reason_target": None,
            "question": "差异？",
            "expected_answer": "C1：A\nC2：B\nC3：边界",
            "expected_claims": [
                {"claim_id": "C1", "text": "A"},
                {"claim_id": "C2", "text": "B"},
                {"claim_id": "C3", "text": "边界"},
            ],
            "gold_evidence": [
                {
                    "evidence_id": "E1",
                    "claim_ids": ["C1", "C3"],
                    "paper_key": "paper_a",
                    "page_start": 1,
                    "page_end": 1,
                    "section_hint": "方法",
                    "supporting_text_summary": "A",
                },
                {
                    "evidence_id": "E2",
                    "claim_ids": ["C2", "C3"],
                    "paper_key": "paper_b",
                    "page_start": 2,
                    "page_end": 2,
                    "section_hint": "方法",
                    "supporting_text_summary": "B",
                },
            ],
            "annotation_notes": ["可解释测试。"],
        }
    )


class P2DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = _corpus()
        self.splits = {"paper_a": "dev", "paper_b": "dev"}
        self.pages = {"paper_a": 3, "paper_b": 3}

    def test_answerable_requires_c3_evidence_from_each_required_paper(self) -> None:
        valid = _answerable(comparison_allowed=True)
        validate_p2_cases(cases=(valid,), corpus=self.corpus, paper_splits=self.splits, page_counts=self.pages)
        raw = valid.to_dict()
        raw["gold_evidence"][1]["claim_ids"] = ["C2"]
        with self.assertRaisesRegex(EvaluationDataError, "C3"):
            P2Case.from_dict(raw)

    def test_unanswerable_requires_reason_and_no_fabricated_evidence(self) -> None:
        raw = _answerable().to_dict()
        raw.update(
            {
                "id": "CD902",
                "source_id": "CD902",
                "task_type": "unanswerable",
                "answerable": False,
                "comparison_allowed": False,
                "refusal_reason_target": "incompatible_setup",
                "expected_claims": [],
                "gold_evidence": [],
                "expected_answer": "两篇论文实验设置不同，不能比较。",
            }
        )
        self.assertFalse(P2Case.from_dict(raw).answerable)
        raw["gold_evidence"] = [_answerable().to_dict()["gold_evidence"][0]]
        with self.assertRaisesRegex(EvaluationDataError, "不能伪造"):
            P2Case.from_dict(raw)

    def test_split_and_page_range_are_checked(self) -> None:
        with self.assertRaisesRegex(EvaluationDataError, "错误 split"):
            validate_p2_cases(
                cases=(_answerable(),),
                corpus=self.corpus,
                paper_splits={"paper_a": "dev", "paper_b": "test"},
                page_counts=self.pages,
            )
        with self.assertRaisesRegex(EvaluationDataError, "超出 PDF"):
            validate_p2_cases(
                cases=(_answerable(),),
                corpus=self.corpus,
                paper_splits=self.splits,
                page_counts={"paper_a": 3, "paper_b": 1},
            )

    def test_importer_reads_required_keys_and_comparison_flag(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "import_p2_eval_markdown", PROJECT_ROOT / "scripts" / "import_p2_eval_markdown.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        markdown = """## CD903
* split: dev
* task_type: cross_document
* answerable: true
* paper_keys:
  * paper_a
  * paper_b
* required_paper_keys:
  * paper_a
  * paper_b
* comparison_dimension: method
* comparison_allowed: true
* question: 差异？

### 预期答案
* C1：A。
* C2：B。
* C3：受限比较。

### 页级证据
* evidence_id: E1
  * claim_ids: [C1, C3]
  * paper_key: paper_a
  * page_start: 1
  * page_end: 1
  * section_hint: 方法
  * supporting_text_summary: A
* evidence_id: E2
  * claim_ids: [C2, C3]
  * paper_key: paper_b
  * page_start: 1
  * page_end: 1
  * section_hint: 方法
  * supporting_text_summary: B
"""
        case = module.parse_p2_markdown(markdown)[0]
        self.assertEqual(case.required_paper_keys, ("paper_a", "paper_b"))
        self.assertTrue(case.comparison_allowed)


if __name__ == "__main__":
    unittest.main()
