"""人工 Markdown 评测题导入器的格式与去重测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.evaluation import EvalCase, GoldEvidence  # noqa: E402


def _load_importer() -> object:
    path = PROJECT_ROOT / "scripts" / "import_markdown_eval_draft.py"
    spec = importlib.util.spec_from_file_location("researchpilot_markdown_eval_import", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclass 会按模块名读取 sys.modules；动态加载时需先登记模块。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


IMPORTER = _load_importer()


class MarkdownEvalImportTests(unittest.TestCase):
    def test_parser_handles_multiple_page_ranges(self) -> None:
        evidence = IMPORTER._parse_evidence(paper_key="mapreduce_2004", raw="第 6 页、第 10 页")
        self.assertEqual([(item.page_start, item.page_end) for item in evidence], [(6, 6), (10, 10)])

    def test_import_cases_are_dev_and_receive_next_stable_ids(self) -> None:
        existing = (
            EvalCase(
                case_id="ai_cs_v01_q005",
                split="dev",
                paper_keys=("rag_2020",),
                question="已有问题",
                question_language="zh",
                task_type="unanswerable",
                answerable=False,
                gold_evidence=(),
                expected_answer="论文未提及。",
            ),
        )
        draft = (
            IMPORTER.DraftQuestion(
                paper_key="resnet_2015",
                ordinal=1,
                question="新问题",
                expected_answer="新答案",
                evidence=(GoldEvidence("resnet_2015", 1, 1),),
                task_type="semantic_explanation",
                split="dev",
                answerable=True,
            ),
        )

        imported = IMPORTER.build_import_cases(draft_questions=draft, existing_cases=existing)

        self.assertEqual(imported[0].case_id, "ai_cs_v01_q006")
        self.assertEqual(imported[0].split, "dev")

    def test_existing_questions_are_skipped_and_test_unanswerable_is_preserved(self) -> None:
        existing = (
            EvalCase(
                case_id="ai_cs_v01_q053",
                split="dev",
                paper_keys=("mapreduce_2004",),
                question="已有问题",
                question_language="zh",
                task_type="semantic_explanation",
                answerable=True,
                gold_evidence=(GoldEvidence("mapreduce_2004", 1, 1),),
                expected_answer="已有答案",
            ),
        )
        draft = (
            IMPORTER.DraftQuestion(
                paper_key="mapreduce_2004",
                ordinal=1,
                question="已有问题",
                expected_answer="已有答案",
                evidence=(GoldEvidence("mapreduce_2004", 1, 1),),
                task_type="semantic_explanation",
                split="dev",
                answerable=True,
            ),
            IMPORTER.DraftQuestion(
                paper_key="sam_2023",
                ordinal=5,
                question="论文未报告的随机种子是什么？",
                expected_answer="论文未提供该随机种子。",
                evidence=(),
                task_type="unanswerable",
                split="test",
                answerable=False,
            ),
        )

        imported = IMPORTER.build_import_cases(draft_questions=draft, existing_cases=existing)

        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0].case_id, "ai_cs_v01_q054")
        self.assertEqual(imported[0].split, "test")
        self.assertFalse(imported[0].answerable)
        self.assertEqual(imported[0].gold_evidence, ())


if __name__ == "__main__":
    unittest.main()
