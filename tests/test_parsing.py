"""Parser Adapter 与解析质量报告的单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.parsing import (  # noqa: E402
    DoclingParser,
    ExtractionQuality,
    PdfPlumberParser,
)
from researchpilot.schemas import BlockType  # noqa: E402


class _FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class _FakePdf:
    def __init__(self, texts: list[str | None]) -> None:
        self.pages = [_FakePage(text) for text in texts]

    def __enter__(self) -> _FakePdf:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


class ParsingTests(unittest.TestCase):
    @patch("researchpilot.parsing.pdfplumber.open")
    def test_pdfplumber_parser_returns_page_blocks_and_quality(self, mock_open: object) -> None:
        mock_open.return_value = _FakePdf(["第一节  方法", "", "第三页实验结果"])

        result = PdfPlumberParser(parser_version="test", min_text_length=10).parse(b"demo", paper_id="paper_demo")

        self.assertEqual([block.page for block in result.blocks], [1, 3])
        self.assertEqual(result.quality_report.page_count, 3)
        self.assertEqual(result.quality_report.text_page_count, 2)
        self.assertEqual(result.quality_report.empty_page_count, 1)
        self.assertEqual(result.quality_report.extraction_quality, ExtractionQuality.NORMAL)
        self.assertFalse(result.quality_report.ocr_used)
        self.assertEqual(result.quality_report.to_dict()["parser_version"], "test")

    def test_docling_helpers_preserve_page_and_structure_labels(self) -> None:
        heading = SimpleNamespace(label=SimpleNamespace(value="section_header"))
        table = SimpleNamespace(label=SimpleNamespace(value="table"))
        formula = SimpleNamespace(label=SimpleNamespace(value="formula"))
        caption = SimpleNamespace(label=SimpleNamespace(value="caption"))
        item = SimpleNamespace(prov=[SimpleNamespace(page_no=4)])

        self.assertEqual(DoclingParser._block_type(heading), BlockType.HEADING)
        self.assertEqual(DoclingParser._block_type(table), BlockType.TABLE)
        self.assertEqual(DoclingParser._block_type(formula), BlockType.FORMULA)
        self.assertEqual(DoclingParser._block_type(caption), BlockType.FIGURE_CAPTION)
        self.assertEqual(DoclingParser._page_number(item), 4)
        self.assertIsNone(DoclingParser._page_number(SimpleNamespace(prov=[])))

    def test_docling_text_normalization_uses_markdown_fallback(self) -> None:
        item = SimpleNamespace(text="", export_to_markdown=lambda doc=None: "| A | B |\n|---|---|")
        self.assertIn("| A | B |", DoclingParser._item_text(item))

    def test_docling_formula_uses_orig_when_text_is_empty(self) -> None:
        item = SimpleNamespace(text="", orig="PE(pos, 2i) = sin(pos / 10000)")
        self.assertEqual(DoclingParser._item_text(item), "PE(pos, 2i) = sin(pos / 10000)")


if __name__ == "__main__":
    unittest.main()
