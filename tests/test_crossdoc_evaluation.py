"""跨论文数据契约、Split 隔离和 Markdown 导入测试；不读取真实 PDF 或模型。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.crossdoc import (  # noqa: E402
    CROSSDOC_SCHEMA_VERSION,
    CrossDocumentCase,
    dataset_sha256,
    load_crossdoc_cases,
    validate_crossdoc_cases,
)
from researchpilot.evaluation import CorpusManifest, EvaluationDataError  # noqa: E402


def _manifest() -> CorpusManifest:
    return CorpusManifest.from_dict(
        {
            "schema_version": "ai_cs_corpus_v0.1",
            "description": "跨论文测试语料",
            "papers": [
                {
                    "paper_key": "paper_a",
                    "title": "Paper A",
                    "area": "nlp",
                    "year": 2025,
                    "source_url": "https://example.com/a.pdf",
                    "local_filename": "a.pdf",
                    "sha256": "a" * 64,
                },
                {
                    "paper_key": "paper_b",
                    "title": "Paper B",
                    "area": "systems",
                    "year": 2025,
                    "source_url": "https://example.com/b.pdf",
                    "local_filename": "b.pdf",
                    "sha256": "b" * 64,
                },
            ],
        }
    )


def _answerable_case(*, split: str = "dev", page_end: int = 2) -> CrossDocumentCase:
    return CrossDocumentCase.from_dict(
        {
            "schema_version": CROSSDOC_SCHEMA_VERSION,
            "id": "CD001",
            "source_id": "CD001",
            "split": split,
            "task_type": "cross_document",
            "answerable": True,
            "paper_keys": ["paper_a", "paper_b"],
            "comparison_dimension": "method",
            "question": "两篇论文的方法差异是什么？",
            "expected_answer": "C1：A。\nC2：B。",
            "expected_claims": [
                {"claim_id": "C1", "text": "A 的结论。"},
                {"claim_id": "C2", "text": "B 的结论。"},
            ],
            "gold_evidence": [
                {
                    "evidence_id": "E1",
                    "claim_ids": ["C1"],
                    "paper_key": "paper_a",
                    "page_start": 1,
                    "page_end": 1,
                    "section_hint": "方法",
                    "supporting_text_summary": "A 的方法。",
                },
                {
                    "evidence_id": "E2",
                    "claim_ids": ["C2"],
                    "paper_key": "paper_b",
                    "page_start": page_end,
                    "page_end": page_end,
                    "section_hint": "方法",
                    "supporting_text_summary": "B 的方法。",
                },
            ],
            "annotation_notes": ["比较方法职责。"],
        }
    )


class CrossDocumentEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.splits = {"paper_a": "dev", "paper_b": "dev"}
        self.pages = {"paper_a": 3, "paper_b": 3}

    def test_valid_case_requires_two_papers_claim_coverage_and_page_ranges(self) -> None:
        case = _answerable_case()
        validate_crossdoc_cases(cases=(case,), corpus=self.manifest, paper_splits=self.splits, page_counts=self.pages)

        with self.assertRaisesRegex(EvaluationDataError, "超出 PDF 实际页数"):
            validate_crossdoc_cases(
                cases=(_answerable_case(page_end=4),),
                corpus=self.manifest,
                paper_splits=self.splits,
                page_counts=self.pages,
            )

    def test_mixed_paper_split_and_invalid_unanswerable_are_rejected(self) -> None:
        with self.assertRaisesRegex(EvaluationDataError, "错误 split"):
            validate_crossdoc_cases(
                cases=(_answerable_case(),),
                corpus=self.manifest,
                paper_splits={"paper_a": "dev", "paper_b": "test"},
                page_counts=self.pages,
            )
        with self.assertRaisesRegex(EvaluationDataError, "不可回答跨论文题"):
            CrossDocumentCase.from_dict(
                {
                    **_answerable_case().to_dict(),
                    "id": "CD002",
                    "task_type": "unanswerable",
                    "answerable": False,
                }
            )

    def test_jsonl_loader_rejects_non_object_claim_and_hashes_frozen_data(self) -> None:
        raw = _answerable_case().to_dict()
        raw["expected_claims"] = ["not_an_object"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "crossdoc.jsonl"
            import json

            path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EvaluationDataError, "expected_claims"):
                load_crossdoc_cases(path)
            path.write_text(json.dumps(_answerable_case().to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(len(dataset_sha256(path)), 64)
            self.assertEqual(load_crossdoc_cases(path)[0].case_id, "CD001")

    def test_markdown_importer_turns_controlled_template_into_cases(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "import_crossdoc_eval_markdown",
            PROJECT_ROOT / "scripts" / "import_crossdoc_eval_markdown.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        markdown = """## CD001
* split: dev
* task_type: cross_document
* answerable: true
* paper_keys:

  * paper_a
  * paper_b
* comparison_dimension: method
* question: 差异是什么？

### 预期答案

* C1：A 的方法。
* C2：B 的方法。

### 页级证据

* evidence_id: E1

  * claim_ids: [C1]
  * paper_key: paper_a
  * page_start: 1
  * page_end: 1
  * section_hint: 方法
  * supporting_text_summary: A

* evidence_id: E2

  * claim_ids: [C2]
  * paper_key: paper_b
  * page_start: 2
  * page_end: 2
  * section_hint: 方法
  * supporting_text_summary: B

### 标注备注

* 比较方法。
"""
        cases = module.parse_crossdoc_markdown(markdown)
        self.assertEqual(cases[0].case_id, "CD001")
        self.assertEqual(cases[0].gold_evidence[1].paper_key, "paper_b")


if __name__ == "__main__":
    unittest.main()
