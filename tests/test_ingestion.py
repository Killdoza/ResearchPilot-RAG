"""PDF 解析与页码父子分块的单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.ingestion import (  # noqa: E402
    IngestionConfig,
    PdfIngestionError,
    _build_parent_child_chunks,
    clean_extracted_text,
    ingest_pdf,
)
from researchpilot.parsing import DocumentParsingError, ParseResult, PdfPlumberParser  # noqa: E402
from researchpilot.schemas import BlockType, DocumentBlock  # noqa: E402


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


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = IngestionConfig(
            parent_chunk_size=48,
            parent_chunk_overlap=8,
            child_chunk_size=20,
            child_chunk_overlap=4,
        )

    def test_clean_extracted_text_preserves_paragraphs(self) -> None:
        text = "  第一段   有多个空格。\r\n\r\n\r\n第二段\t仍应保留。  "
        self.assertEqual(clean_extracted_text(text), "第一段 有多个空格。\n\n第二段 仍应保留。")

    @patch("researchpilot.parsing.pdfplumber.open")
    def test_ingest_pdf_keeps_paper_page_and_parent_relationships(self, mock_open: object) -> None:
        mock_open.return_value = _FakePdf(
            [
                "第一页：本文提出一个用于论文问答的检索方法。它使用页码级证据。",
                "第二页：实验在三个数据集上进行，指标包括 Recall 和 MRR。",
                None,
            ]
        )
        result = ingest_pdf(b"fake pdf binary", source_name="sample.pdf", config=self.config)

        self.assertEqual(result.paper.source_name, "sample.pdf")
        self.assertEqual(result.paper.page_count, 3)
        self.assertEqual([block.page for block in result.blocks], [1, 2])
        self.assertIn("第一页", result.head_text)
        self.assertIn("第二页", result.full_text)
        self.assertTrue(result.parent_chunks)
        self.assertTrue(result.child_chunks)
        parent_ids = {chunk.chunk_id for chunk in result.parent_chunks}
        for parent in result.parent_chunks:
            self.assertEqual(parent.paper_id, result.paper.paper_id)
            self.assertEqual(parent.page_start, parent.page_end)
            self.assertTrue(parent.parent_id and parent.parent_id.startswith("block_"))
        for child in result.child_chunks:
            self.assertEqual(child.paper_id, result.paper.paper_id)
            self.assertEqual(child.page_start, child.page_end)
            self.assertIn(child.parent_id, parent_ids)

    @patch("researchpilot.parsing.pdfplumber.open")
    def test_ingestion_is_deterministic_for_same_binary_content(self, mock_open: object) -> None:
        mock_open.return_value = _FakePdf(["可复现性要求相同输入产生相同的论文 ID 和 Chunk ID。"])
        first = ingest_pdf(b"same bytes", source_name="one.pdf", config=self.config)
        mock_open.return_value = _FakePdf(["可复现性要求相同输入产生相同的论文 ID 和 Chunk ID。"])
        second = ingest_pdf(b"same bytes", source_name="renamed.pdf", config=self.config)

        self.assertEqual(first.paper.paper_id, second.paper.paper_id)
        self.assertEqual(
            [chunk.chunk_id for chunk in first.all_chunks],
            [chunk.chunk_id for chunk in second.all_chunks],
        )

    def test_repeated_child_text_gets_distinct_deterministic_ids(self) -> None:
        block = DocumentBlock(
            paper_id="paper_repeated_text",
            page=1,
            block_id="block_paper_repeated_text_p0001",
            block_type=BlockType.PARAGRAPH,
            text="repeat repeat repeat repeat",
            parser_name="test",
            parser_version="1",
        )
        config = IngestionConfig(
            parent_chunk_size=100,
            parent_chunk_overlap=0,
            child_chunk_size=6,
            child_chunk_overlap=0,
        )

        _, children = _build_parent_child_chunks((block,), config)

        self.assertGreaterEqual(len(children), 2)
        self.assertEqual(len(children), len({chunk.chunk_id for chunk in children}))

    @patch("researchpilot.parsing.pdfplumber.open")
    def test_scanned_or_empty_pdf_is_rejected(self, mock_open: object) -> None:
        mock_open.return_value = _FakePdf([None, "   "])
        with self.assertRaisesRegex(PdfIngestionError, "扫描版"):
            ingest_pdf(b"empty text", source_name="scan.pdf", config=self.config)

    def test_invalid_chunking_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(PdfIngestionError, "size > overlap"):
            IngestionConfig(parent_chunk_size=20, parent_chunk_overlap=20)

    def test_unknown_parser_is_rejected(self) -> None:
        with self.assertRaisesRegex(PdfIngestionError, "parser_name"):
            IngestionConfig(parser_name="unknown")

    def test_fallback_parser_is_recorded_without_silently_relabeling_result(self) -> None:
        config = IngestionConfig(
            parser_name="docling",
            parser_version="2.120.2",
            fallback_parser_name="pdfplumber",
            fallback_parser_version="0.11.9",
            chunker_version="structure_aware_v1",
            parent_chunk_size=48,
            parent_chunk_overlap=8,
            child_chunk_size=20,
            child_chunk_overlap=4,
        )

        class _FailingDocling:
            def parse(self, source_bytes: bytes, *, paper_id: str) -> ParseResult:
                raise DocumentParsingError("Docling 测试失败")

        with (
            patch(
                "researchpilot.ingestion.build_document_parser",
                side_effect=[_FailingDocling(), PdfPlumberParser(parser_version="0.11.9")],
            ),
            patch("researchpilot.parsing.pdfplumber.open") as mock_open,
        ):
            mock_open.return_value = _FakePdf(["兼容解析应保留实际 Parser 信息。"])
            result = ingest_pdf(b"fallback binary", source_name="fallback.pdf", config=config)

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.requested_parser_name, "docling")
        self.assertEqual(result.parse_quality.parser_name, "pdfplumber")
        self.assertIn("Docling 测试失败", result.fallback_reason or "")


if __name__ == "__main__":
    unittest.main()
