"""可替换的文档解析层：PDF → 带页码和质量信息的结构化 Block。

本模块不依赖 Streamlit、Chroma 或 Embedding。它刻意与 ``ingestion.py`` 的
Parent-Child 分块解耦：替换 Parser 不会改变检索接口，解析质量也能独立统计。
"""

from __future__ import annotations

import re
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Protocol

import pdfplumber

from .schemas import BlockType, DocumentBlock


class DocumentParsingError(ValueError):
    """文件无法被 Parser 转换为可信文本块时抛出。"""


class ExtractionQuality(str, Enum):
    NORMAL = "normal"
    LOW_TEXT = "low_text"
    FAILED = "failed"


@dataclass(frozen=True)
class ParseQualityReport:
    """一次解析的可序列化质量摘要，供 UI、Manifest 与评测使用。"""

    parser_name: str
    parser_version: str
    ocr_used: bool
    page_count: int
    text_page_count: int
    empty_page_count: int
    character_count: int
    heading_count: int
    table_count: int
    elapsed_ms: float
    extraction_quality: ExtractionQuality

    def __post_init__(self) -> None:
        if not self.parser_name.strip() or not self.parser_version.strip():
            raise DocumentParsingError("解析质量报告必须包含解析器名称和版本。")
        if self.page_count < 1 or self.text_page_count < 0 or self.empty_page_count < 0:
            raise DocumentParsingError("解析质量报告中的页数非法。")
        if self.text_page_count + self.empty_page_count != self.page_count:
            raise DocumentParsingError("解析质量报告中的文本页与空页总数必须等于总页数。")
        if min(self.character_count, self.heading_count, self.table_count) < 0 or self.elapsed_ms < 0:
            raise DocumentParsingError("解析质量报告中的计数或耗时非法。")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["extraction_quality"] = self.extraction_quality.value
        return payload


@dataclass(frozen=True)
class ParseResult:
    """Parser 的稳定输出；后续 Chunker 只依赖该对象。"""

    blocks: tuple[DocumentBlock, ...]
    quality_report: ParseQualityReport

    @property
    def full_text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks).strip()


class DocumentParser(Protocol):
    """所有 Parser 必须返回页码可追溯的统一结构块。"""

    parser_name: str
    parser_version: str

    def parse(self, source_bytes: bytes, *, paper_id: str) -> ParseResult: ...


_INLINE_WHITESPACE = re.compile(r"[^\S\r\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_extracted_text(text: str) -> str:
    """保留段落边界，同时移除常见 PDF 提取冗余空白。"""
    if not isinstance(text, str):
        raise DocumentParsingError("页面文本必须是字符串。")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(_INLINE_WHITESPACE.sub(" ", line).strip() for line in normalized.split("\n"))
    return _EXCESS_NEWLINES.sub("\n\n", normalized).strip()


def _quality_from_blocks(
    *,
    blocks: tuple[DocumentBlock, ...],
    parser_name: str,
    parser_version: str,
    ocr_used: bool,
    page_count: int,
    elapsed_ms: float,
    min_text_length: int,
) -> ParseQualityReport:
    text_pages = {block.page for block in blocks}
    character_count = sum(len(block.text) for block in blocks)
    return ParseQualityReport(
        parser_name=parser_name,
        parser_version=parser_version,
        ocr_used=ocr_used,
        page_count=page_count,
        text_page_count=len(text_pages),
        empty_page_count=page_count - len(text_pages),
        character_count=character_count,
        heading_count=sum(block.block_type is BlockType.HEADING for block in blocks),
        table_count=sum(block.block_type is BlockType.TABLE for block in blocks),
        elapsed_ms=elapsed_ms,
        extraction_quality=(
            ExtractionQuality.NORMAL if character_count >= min_text_length else ExtractionQuality.LOW_TEXT
        ),
    )


class PdfPlumberParser:
    """旧版稳定基线的 Parser Adapter。"""

    parser_name = "pdfplumber"

    def __init__(self, *, parser_version: str | None = None, min_text_length: int = 10) -> None:
        self.parser_version = parser_version or getattr(pdfplumber, "__version__", "unknown")
        self._min_text_length = min_text_length

    def parse(self, source_bytes: bytes, *, paper_id: str) -> ParseResult:
        started = time.perf_counter()
        try:
            with pdfplumber.open(BytesIO(source_bytes)) as pdf:
                page_texts = tuple(clean_extracted_text(page.extract_text() or "") for page in pdf.pages)
        except Exception as exc:
            raise DocumentParsingError(f"pdfplumber 无法读取 PDF：{exc}") from exc
        if not page_texts:
            raise DocumentParsingError("PDF 不包含任何页面。")
        blocks = tuple(
            DocumentBlock(
                paper_id=paper_id,
                page=page_number,
                block_id=f"block_{paper_id}_p{page_number:04d}",
                block_type=BlockType.PARAGRAPH,
                text=page_text,
                parser_name=self.parser_name,
                parser_version=self.parser_version,
                extraction_quality="normal" if page_text else "low_text",
            )
            for page_number, page_text in enumerate(page_texts, start=1)
            if page_text
        )
        return ParseResult(
            blocks=blocks,
            quality_report=_quality_from_blocks(
                blocks=blocks,
                parser_name=self.parser_name,
                parser_version=self.parser_version,
                ocr_used=False,
                page_count=len(page_texts),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                min_text_length=self._min_text_length,
            ),
        )


class DoclingParser:
    """Docling 的最小结构化 PDF Adapter。

    Windows 下显式使用 PyPdfium 后端并关闭 ``torch.compile``，避免默认后端的
    资源路径和 GBK 编码兼容问题。OCR 保留为配置项，本 Sprint 默认关闭。
    """

    parser_name = "docling"

    def __init__(
        self,
        *,
        parser_version: str,
        min_text_length: int = 10,
        enable_ocr: bool = False,
    ) -> None:
        self.parser_version = parser_version
        self._min_text_length = min_text_length
        self._enable_ocr = enable_ocr
        self._converter: object | None = None

    @staticmethod
    def _block_type(item: object) -> BlockType:
        label = getattr(getattr(item, "label", None), "value", "")
        label = str(label).lower()
        if label == "section_header":
            return BlockType.HEADING
        if label == "table":
            return BlockType.TABLE
        if label == "formula":
            return BlockType.FORMULA
        if label == "caption":
            return BlockType.FIGURE_CAPTION
        if label == "reference":
            return BlockType.REFERENCE
        return BlockType.PARAGRAPH

    @staticmethod
    def _item_text(item: object, *, document: object | None = None) -> str:
        text = getattr(item, "text", "")
        if not text and hasattr(item, "export_to_markdown"):
            try:
                text = item.export_to_markdown(doc=document)
            except Exception:
                text = ""
        # Docling 的 FormulaItem 通常将可检索的线性化公式保存在 ``orig``，
        # 而 ``text`` 与 Markdown 导出都可能为空。若忽略该字段，会把公式
        # 静默丢弃，导致“公式/超参数”类证据不完整。
        if not text:
            text = getattr(item, "orig", "")
        return clean_extracted_text(str(text or ""))

    @staticmethod
    def _page_number(item: object) -> int | None:
        provenance = getattr(item, "prov", ()) or ()
        if not provenance:
            return None
        page_number = getattr(provenance[0], "page_no", None)
        return page_number if isinstance(page_number, int) and page_number >= 1 else None

    def _get_converter(self):
        try:
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise DocumentParsingError("未安装 Docling；请安装 requirements.txt 中的 docling 依赖。") from exc
        if self._converter is None:
            options = PdfPipelineOptions()
            options.do_ocr = self._enable_ocr
            options.do_table_structure = True
            options.heading_hierarchy_options.enabled = True
            options.layout_options.engine_options.compile_model = False
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=options,
                        backend=PyPdfiumDocumentBackend,
                    )
                }
            )
        return self._converter

    def parse(self, source_bytes: bytes, *, paper_id: str) -> ParseResult:
        if not source_bytes:
            raise DocumentParsingError("PDF 文件为空。")
        started = time.perf_counter()

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="researchpilot_", suffix=".pdf", delete=False) as handle:
                handle.write(source_bytes)
                temporary_path = Path(handle.name)
            converted = self._get_converter().convert(temporary_path)
            document = converted.document
            page_count = len(document.pages)
            if page_count < 1:
                raise DocumentParsingError("Docling 未返回可用页面。")
            blocks: list[DocumentBlock] = []
            section_stack: list[str] = []
            for index, (item, _) in enumerate(document.iterate_items()):
                page_number = self._page_number(item)
                text = self._item_text(item, document=document)
                if page_number is None or not text:
                    continue
                block_type = self._block_type(item)
                if block_type is BlockType.HEADING:
                    level = getattr(item, "level", 1)
                    level = level if isinstance(level, int) and level >= 1 else 1
                    section_stack = section_stack[: level - 1]
                    section_stack.append(text)
                blocks.append(
                    DocumentBlock(
                        paper_id=paper_id,
                        page=page_number,
                        block_id=f"block_{paper_id}_docling_{index:06d}",
                        block_type=block_type,
                        text=text,
                        section_path=tuple(section_stack),
                        parser_name=self.parser_name,
                        parser_version=self.parser_version,
                        ocr_used=self._enable_ocr,
                    )
                )
        except DocumentParsingError:
            raise
        except Exception as exc:
            raise DocumentParsingError(f"Docling 无法解析 PDF：{exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        normalized_blocks = tuple(blocks)
        return ParseResult(
            blocks=normalized_blocks,
            quality_report=_quality_from_blocks(
                blocks=normalized_blocks,
                parser_name=self.parser_name,
                parser_version=self.parser_version,
                ocr_used=self._enable_ocr,
                page_count=page_count,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                min_text_length=self._min_text_length,
            ),
        )
