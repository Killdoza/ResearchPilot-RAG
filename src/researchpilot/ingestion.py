"""论文 PDF 的确定性解析、清洗与父子分块。

本模块不依赖 Streamlit、Chroma 或 LLM。上层只需传入 PDF 内容，即可获得带
论文 ID、页码和父子关系的领域对象；之后替换为 Docling 等解析器时，索引和检索
接口无需变化。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

from .chunking import build_structure_aware_parent_child_chunks
from .parsing import (
    DoclingParser,
    DocumentParser,
    DocumentParsingError,
    ParseQualityReport,
    PdfPlumberParser,
    clean_extracted_text,
)
from .schemas import BlockType, Chunk, DocumentBlock, Paper, sha256_bytes


class PdfIngestionError(ValueError):
    """PDF 无法读取或未提取到可用文本时抛出。"""


@dataclass(frozen=True)
class IngestionConfig:
    """可记录、可复现的第一版解析与分块参数。"""

    min_document_text_length: int = 10
    head_page_limit: int = 3
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 300
    child_chunk_overlap: int = 50
    parser_name: str = "pdfplumber"
    parser_version: str = "0.11.9"
    fallback_parser_name: str | None = None
    fallback_parser_version: str | None = None
    chunker_version: str = "page_parent_child_v1"
    enable_ocr: bool = False

    def __post_init__(self) -> None:
        if self.min_document_text_length < 1:
            raise PdfIngestionError("min_document_text_length 必须为正整数。")
        if self.head_page_limit < 1:
            raise PdfIngestionError("head_page_limit 必须为正整数。")
        for size, overlap, label in (
            (self.parent_chunk_size, self.parent_chunk_overlap, "parent"),
            (self.child_chunk_size, self.child_chunk_overlap, "child"),
        ):
            if size < 1 or overlap < 0 or overlap >= size:
                raise PdfIngestionError(f"{label} 分块参数必须满足 size > overlap >= 0。")
        if self.parser_name not in {"pdfplumber", "docling"}:
            raise PdfIngestionError("parser_name 目前只支持 pdfplumber 或 docling。")
        if self.fallback_parser_name is not None:
            if self.fallback_parser_name not in {"pdfplumber", "docling"}:
                raise PdfIngestionError("fallback_parser_name 目前只支持 pdfplumber 或 docling。")
            if self.fallback_parser_name == self.parser_name:
                raise PdfIngestionError("fallback_parser_name 不能与 parser_name 相同。")
            if not self.fallback_parser_version or not self.fallback_parser_version.strip():
                raise PdfIngestionError("配置 fallback_parser_name 时必须提供 fallback_parser_version。")
        elif self.fallback_parser_version is not None:
            raise PdfIngestionError("仅配置 fallback_parser_version 无效；请同时配置 fallback_parser_name。")
        if self.chunker_version not in {"page_parent_child_v1", "structure_aware_v1"}:
            raise PdfIngestionError("chunker_version 目前只支持 page_parent_child_v1 或 structure_aware_v1。")
        if not self.parser_version.strip():
            raise PdfIngestionError("parser_version 不能为空。")
        if not isinstance(self.enable_ocr, bool):
            raise PdfIngestionError("enable_ocr 必须是布尔值。")


@dataclass(frozen=True)
class IngestionResult:
    """一次 PDF 解析的完整中间结果。"""

    paper: Paper
    blocks: tuple[DocumentBlock, ...]
    parent_chunks: tuple[Chunk, ...]
    child_chunks: tuple[Chunk, ...]
    head_text: str
    full_text: str
    parse_quality: ParseQualityReport | None = None
    requested_parser_name: str = "pdfplumber"
    requested_parser_version: str = "0.11.9"
    fallback_used: bool = False
    fallback_reason: str | None = None

    @property
    def all_chunks(self) -> tuple[Chunk, ...]:
        """按父块、子块的确定顺序返回所有可索引 Chunk。"""
        return self.parent_chunks + self.child_chunks


_SENTENCE_BOUNDARIES = ("\n\n", "\n", "。", "！", "？", ". ", "; ", ", ", " ")


def _read_source_bytes(pdf_source: Path | bytes | BinaryIO) -> bytes:
    if isinstance(pdf_source, Path):
        return pdf_source.read_bytes()
    if isinstance(pdf_source, bytes):
        return pdf_source
    if hasattr(pdf_source, "getvalue"):
        data = pdf_source.getvalue()
        if isinstance(data, bytes):
            return data
    if not (hasattr(pdf_source, "read") and hasattr(pdf_source, "seek")):
        raise PdfIngestionError("PDF 输入必须是路径、bytes 或可回退的二进制文件对象。")
    original_position = pdf_source.tell() if hasattr(pdf_source, "tell") else None
    pdf_source.seek(0)
    data = pdf_source.read()
    if original_position is not None:
        pdf_source.seek(original_position)
    else:
        pdf_source.seek(0)
    if not isinstance(data, bytes):
        raise PdfIngestionError("PDF 文件对象读取结果必须是 bytes。")
    return data


def _infer_source_name(pdf_source: Path | bytes | BinaryIO, source_name: str | None) -> str:
    if source_name and source_name.strip():
        return source_name.strip()
    if isinstance(pdf_source, Path):
        return pdf_source.name
    candidate = getattr(pdf_source, "name", None)
    if isinstance(candidate, str) and candidate.strip():
        return Path(candidate).name
    raise PdfIngestionError("请提供 source_name，以便形成可追溯的论文来源。")


def _detect_language(text: str) -> str:
    chinese_character_count = sum("\u4e00" <= character <= "\u9fff" for character in text)
    latin_character_count = sum(character.isascii() and character.isalpha() for character in text)
    if chinese_character_count >= 20 and chinese_character_count >= latin_character_count / 3:
        return "zh"
    if latin_character_count >= 20:
        return "en"
    return "unknown"


def _split_text(text: str, *, chunk_size: int, overlap: int) -> tuple[str, ...]:
    """轻量、确定性的文本切分，不依赖 UI 或 LangChain。"""
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return ()

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)
    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        end = target_end
        if target_end < text_length:
            boundary = max(
                (cleaned.rfind(token, start + 1, target_end + 1) for token in _SENTENCE_BOUNDARIES),
                default=-1,
            )
            if boundary > start:
                end = boundary + 1
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
    return tuple(chunks)


def _build_blocks(
    page_texts: tuple[str, ...],
    *,
    paper_id: str,
    config: IngestionConfig,
) -> tuple[DocumentBlock, ...]:
    blocks: list[DocumentBlock] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        if not page_text:
            continue
        blocks.append(
            DocumentBlock(
                paper_id=paper_id,
                page=page_number,
                block_id=f"block_{paper_id}_p{page_number:04d}",
                block_type=BlockType.PARAGRAPH,
                text=page_text,
                parser_name=config.parser_name,
                parser_version=config.parser_version,
            )
        )
    return tuple(blocks)


def build_document_parser(config: IngestionConfig) -> DocumentParser:
    if config.parser_name == "pdfplumber":
        return PdfPlumberParser(
            parser_version=config.parser_version,
            min_text_length=config.min_document_text_length,
        )
    if config.parser_name == "docling":
        return DoclingParser(
            parser_version=config.parser_version,
            min_text_length=config.min_document_text_length,
            enable_ocr=config.enable_ocr,
        )
    raise PdfIngestionError(f"未知 Parser：{config.parser_name}")


def _fallback_config(config: IngestionConfig) -> IngestionConfig | None:
    """返回仅替换 Parser 的兼容配置，保持分块参数与 OCR 策略不变。"""
    if config.fallback_parser_name is None or config.fallback_parser_version is None:
        return None
    return replace(
        config,
        parser_name=config.fallback_parser_name,
        parser_version=config.fallback_parser_version,
        fallback_parser_name=None,
        fallback_parser_version=None,
    )


def _build_parent_child_chunks(
    blocks: tuple[DocumentBlock, ...], config: IngestionConfig
) -> tuple[tuple[Chunk, ...], tuple[Chunk, ...]]:
    if config.chunker_version == "structure_aware_v1":
        return build_structure_aware_parent_child_chunks(
            blocks,
            parent_chunk_size=config.parent_chunk_size,
            parent_chunk_overlap=config.parent_chunk_overlap,
            child_chunk_size=config.child_chunk_size,
            child_chunk_overlap=config.child_chunk_overlap,
            chunker_version=config.chunker_version,
        )
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for block in blocks:
        for parent_index, parent_text in enumerate(
            _split_text(
                block.text,
                chunk_size=config.parent_chunk_size,
                overlap=config.parent_chunk_overlap,
            )
        ):
            parent = Chunk.create(
                paper_id=block.paper_id,
                parent_id=block.block_id,
                page_start=block.page,
                page_end=block.page,
                section_path=block.section_path,
                content=parent_text,
                block_types=(block.block_type,),
                chunker_version=config.chunker_version,
                occurrence_index=parent_index,
            )
            parents.append(parent)
            for child_index, child_text in enumerate(
                _split_text(
                    parent.content,
                    chunk_size=config.child_chunk_size,
                    overlap=config.child_chunk_overlap,
                )
            ):
                children.append(
                    Chunk.create(
                        paper_id=block.paper_id,
                        parent_id=parent.chunk_id,
                        page_start=block.page,
                        page_end=block.page,
                        section_path=block.section_path,
                        content=child_text,
                        block_types=(block.block_type,),
                        chunker_version=config.chunker_version,
                        occurrence_index=child_index,
                    )
                )
    return tuple(parents), tuple(children)


def ingest_pdf(
    pdf_source: Path | bytes | BinaryIO,
    *,
    source_name: str | None = None,
    config: IngestionConfig | None = None,
    parser: DocumentParser | None = None,
) -> IngestionResult:
    """解析一篇可选中文本的 PDF，并返回具有页码证据的父子 Chunk。

    纯扫描 PDF 会抛出 ``PdfIngestionError``，由 UI 决定如何展示用户提示。
    """
    resolved_config = config or IngestionConfig()
    source_bytes = _read_source_bytes(pdf_source)
    if not source_bytes:
        raise PdfIngestionError("PDF 文件为空。")
    resolved_source_name = _infer_source_name(pdf_source, source_name)
    source_hash = sha256_bytes(source_bytes)
    paper_id = f"paper_{source_hash[:16]}"

    requested_parser_name = resolved_config.parser_name
    requested_parser_version = resolved_config.parser_version
    actual_config = resolved_config
    fallback_used = False
    fallback_reason: str | None = None
    try:
        parsed = (parser or build_document_parser(actual_config)).parse(source_bytes, paper_id=paper_id)
    except DocumentParsingError as exc:
        fallback_config = _fallback_config(resolved_config) if parser is None else None
        if fallback_config is None:
            raise PdfIngestionError(str(exc)) from exc
        fallback_used = True
        fallback_reason = str(exc)
        actual_config = fallback_config
        try:
            parsed = build_document_parser(actual_config).parse(source_bytes, paper_id=paper_id)
        except DocumentParsingError as fallback_exc:
            raise PdfIngestionError(
                f"{requested_parser_name} 解析失败：{exc}；{actual_config.parser_name} 兼容解析也失败：{fallback_exc}"
            ) from fallback_exc

    full_text = parsed.full_text
    if len(full_text) < resolved_config.min_document_text_length:
        suffix = "；该文件可能是扫描版 PDF，请在后续 OCR 模式下重新解析。"
        raise PdfIngestionError("未提取到足够的可用文本" + suffix)

    blocks = parsed.blocks
    parents, children = _build_parent_child_chunks(blocks, resolved_config)
    if not parents or not children:
        raise PdfIngestionError("PDF 文本无法生成有效分块。")
    head_text = "\n\n".join(block.text for block in blocks if block.page <= resolved_config.head_page_limit).strip()
    paper = Paper(
        paper_id=paper_id,
        source_name=resolved_source_name,
        content_hash=source_hash,
        language=_detect_language(full_text),
        page_count=parsed.quality_report.page_count,
    )
    return IngestionResult(
        paper=paper,
        blocks=blocks,
        parent_chunks=parents,
        child_chunks=children,
        head_text=head_text,
        full_text=full_text,
        parse_quality=parsed.quality_report,
        requested_parser_name=requested_parser_name,
        requested_parser_version=requested_parser_version,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
