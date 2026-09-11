"""ResearchPilot RAG Core 的领域数据模型。

这些模型不依赖 Streamlit、Chroma 或大模型客户端。解析、索引、检索、生成与
评测模块都通过它们交换数据，避免再次把字符串字典散落在页面逻辑中。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any


class SchemaValidationError(ValueError):
    """领域对象不满足系统不变量时抛出。"""


class BlockType(str, Enum):
    """论文中可被解析器识别的基础块类型。"""

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FORMULA = "formula"
    FIGURE_CAPTION = "figure_caption"
    REFERENCE = "reference"
    OTHER = "other"


class QueryType(str, Enum):
    """后续 Query-aware Retrieval 使用的论文问题类型。"""

    METADATA = "metadata"
    DOCUMENT_SUMMARY = "document_summary"
    EXPERIMENTAL_SETUP = "experimental_setup"
    EXPERIMENTAL_RESULTS = "experimental_results"
    KEYWORD_EXACT = "keyword_exact"
    LOCAL_FACT = "local_fact"
    SEMANTIC_EXPLANATION = "semantic_explanation"
    TABLE_FIGURE = "table_figure"
    MULTI_SECTION = "multi_section"
    CROSS_DOCUMENT = "cross_document"
    UNANSWERABLE = "unanswerable"
    GENERAL_CHAT = "general_chat"


_P2_RUN_MODES = frozenset({"static", "oracle", "scoped", "end_to_end"})
_QUERY_TYPE_SOURCES = frozenset({"static_config", "oracle_eval", "rule_predicted", "model_predicted"})
_SCOPE_SOURCES = frozenset({"none", "oracle_eval", "user_selected", "session_default", "title_resolved"})


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} 不能为空。")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise SchemaValidationError(f"{field_name} 必须是 64 位 SHA-256 十六进制字符串。")


def sha256_text(value: str) -> str:
    """返回 UTF-8 文本的稳定内容哈希。"""
    _require_text(value, "待哈希文本")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """返回二进制文件内容的稳定 SHA-256 哈希。"""
    if not isinstance(value, bytes) or not value:
        raise SchemaValidationError("待哈希二进制内容不能为空。")
    return hashlib.sha256(value).hexdigest()


def _validate_page_range(page_start: int, page_end: int) -> None:
    if not isinstance(page_start, int) or not isinstance(page_end, int):
        raise SchemaValidationError("页码必须是整数。")
    if page_start < 1 or page_end < page_start:
        raise SchemaValidationError("页码范围必须满足 1 <= page_start <= page_end。")


def _validate_rank(rank: int | None, field_name: str) -> None:
    if rank is not None and (not isinstance(rank, int) or rank < 1):
        raise SchemaValidationError(f"{field_name} 必须是正整数或 None。")


def _validate_score(score: float | None, field_name: str) -> None:
    if score is not None and (not isinstance(score, int | float) or not math.isfinite(score)):
        raise SchemaValidationError(f"{field_name} 必须是有限数值或 None。")


@dataclass(frozen=True)
class Paper:
    """一篇已知来源、可重建的论文。"""

    paper_id: str
    source_name: str
    content_hash: str
    title: str | None = None
    language: str = "unknown"
    page_count: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.paper_id, "paper_id")
        _require_text(self.source_name, "source_name")
        _require_sha256(self.content_hash, "content_hash")
        if self.title is not None:
            _require_text(self.title, "title")
        _require_text(self.language, "language")
        if self.page_count is not None and (not isinstance(self.page_count, int) or self.page_count < 1):
            raise SchemaValidationError("page_count 必须是正整数或 None。")


@dataclass(frozen=True)
class DocumentBlock:
    """解析器从一页论文提取出的最小结构化文本块。"""

    paper_id: str
    page: int
    block_id: str
    block_type: BlockType
    text: str
    section_path: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    parser_name: str = "unknown"
    parser_version: str = "unknown"
    ocr_used: bool = False
    extraction_quality: str = "normal"

    def __post_init__(self) -> None:
        _require_text(self.paper_id, "paper_id")
        _require_text(self.block_id, "block_id")
        _require_text(self.text, "text")
        _require_text(self.parser_name, "parser_name")
        _require_text(self.parser_version, "parser_version")
        if not isinstance(self.ocr_used, bool):
            raise SchemaValidationError("ocr_used 必须是布尔值。")
        if self.extraction_quality not in {"normal", "low_text", "failed"}:
            raise SchemaValidationError("extraction_quality 必须是 normal、low_text 或 failed。")
        _validate_page_range(self.page, self.page)
        if not isinstance(self.block_type, BlockType):
            raise SchemaValidationError("block_type 必须是 BlockType。")
        if not isinstance(self.section_path, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.section_path
        ):
            raise SchemaValidationError("section_path 必须是由非空字符串组成的元组。")
        if self.bbox is not None:
            if len(self.bbox) != 4 or any(
                not isinstance(value, int | float) or not math.isfinite(value) for value in self.bbox
            ):
                raise SchemaValidationError("bbox 必须是四个有限数值组成的元组或 None。")


def derive_chunk_id(
    *,
    paper_id: str,
    parent_id: str | None,
    page_start: int,
    page_end: int,
    content_hash: str,
    chunker_version: str,
    occurrence_index: int = 0,
) -> str:
    """由文档位置、内容和分块器版本确定性生成 Chunk ID。

    同一父块中可以合法出现完全相同的文本（例如重复页眉、图注或短句）。
    ``occurrence_index`` 把该文本在当前分块序列中的位置纳入哈希，既避免
    Chroma 写入 ID 冲突，又保持相同输入下的 ID 可复现。
    """
    _require_text(paper_id, "paper_id")
    _validate_page_range(page_start, page_end)
    _require_sha256(content_hash, "content_hash")
    _require_text(chunker_version, "chunker_version")
    if not isinstance(occurrence_index, int) or isinstance(occurrence_index, bool) or occurrence_index < 0:
        raise SchemaValidationError("occurrence_index 必须是非负整数。")
    payload = {
        "paper_id": paper_id,
        "parent_id": parent_id,
        "page_start": page_start,
        "page_end": page_end,
        "content_hash": content_hash,
        "chunker_version": chunker_version,
        "occurrence_index": occurrence_index,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"chunk_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class Chunk:
    """统一的可索引文本单元；支持父子分块和页码级引用。"""

    chunk_id: str
    paper_id: str
    parent_id: str | None
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    content: str
    block_types: tuple[BlockType, ...]
    content_hash: str
    chunker_version: str

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.paper_id, "paper_id")
        _require_text(self.content, "content")
        _require_sha256(self.content_hash, "content_hash")
        _require_text(self.chunker_version, "chunker_version")
        _validate_page_range(self.page_start, self.page_end)
        if self.parent_id is not None:
            _require_text(self.parent_id, "parent_id")
        if self.content_hash != sha256_text(self.content):
            raise SchemaValidationError("content_hash 与 content 不一致。")
        if not isinstance(self.section_path, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.section_path
        ):
            raise SchemaValidationError("section_path 必须是由非空字符串组成的元组。")
        if not self.block_types or any(not isinstance(item, BlockType) for item in self.block_types):
            raise SchemaValidationError("block_types 至少应包含一个 BlockType。")

    @classmethod
    def create(
        cls,
        *,
        paper_id: str,
        parent_id: str | None,
        page_start: int,
        page_end: int,
        section_path: tuple[str, ...],
        content: str,
        block_types: tuple[BlockType, ...],
        chunker_version: str,
        occurrence_index: int = 0,
    ) -> Chunk:
        """创建包含内容哈希和确定性 ID 的 Chunk。"""
        content_hash = sha256_text(content)
        return cls(
            chunk_id=derive_chunk_id(
                paper_id=paper_id,
                parent_id=parent_id,
                page_start=page_start,
                page_end=page_end,
                content_hash=content_hash,
                chunker_version=chunker_version,
                occurrence_index=occurrence_index,
            ),
            paper_id=paper_id,
            parent_id=parent_id,
            page_start=page_start,
            page_end=page_end,
            section_path=section_path,
            content=content,
            block_types=block_types,
            content_hash=content_hash,
            chunker_version=chunker_version,
        )

    def index_metadata(self) -> dict[str, str | int]:
        """返回写入索引所需的、可 JSON 化的最小元数据。"""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "parent_id": self.parent_id or "",
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunker_version": self.chunker_version,
            "content_hash": self.content_hash,
            "section_path": " > ".join(self.section_path),
            "block_types": ",".join(block_type.value for block_type in self.block_types),
        }


@dataclass(frozen=True)
class Citation:
    """回答中的一条可定位证据。"""

    paper_id: str
    chunk_id: str
    page_start: int
    page_end: int
    evidence: str
    section_path: tuple[str, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.paper_id, "paper_id")
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.evidence, "evidence")
        _validate_page_range(self.page_start, self.page_end)
        _validate_score(self.confidence, "confidence")


@dataclass(frozen=True)
class AnswerClaim:
    """回答中的一个可审计事实断言及其证据绑定。"""

    claim_id: str
    text: str
    citation_ids: tuple[str, ...] = ()
    support_status: str = "unsupported"
    numeric_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.claim_id, "claim_id")
        _require_text(self.text, "text")
        if self.support_status not in {
            "supported",
            "unsupported",
            "numeric_unsupported",
            "not_applicable",
        }:
            raise SchemaValidationError("support_status 不是支持的 Claim 状态。")
        if len(set(self.citation_ids)) != len(self.citation_ids) or any(
            not isinstance(item, str) or not item.strip() for item in self.citation_ids
        ):
            raise SchemaValidationError("citation_ids 必须是不重复的非空字符串元组。")
        if any(not isinstance(item, str) or not item.strip() for item in self.numeric_values):
            raise SchemaValidationError("numeric_values 必须由非空字符串组成。")


@dataclass(frozen=True)
class ScoredChunk:
    """检索流水线中携带各阶段排名和分数的 Chunk。"""

    chunk: Chunk
    source_name: str | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, Chunk):
            raise SchemaValidationError("chunk 必须是 Chunk 对象。")
        if self.source_name is not None:
            _require_text(self.source_name, "source_name")
        _validate_rank(self.dense_rank, "dense_rank")
        _validate_rank(self.sparse_rank, "sparse_rank")
        _validate_score(self.dense_score, "dense_score")
        _validate_score(self.sparse_score, "sparse_score")
        _validate_score(self.fusion_score, "fusion_score")
        _validate_score(self.rerank_score, "rerank_score")


@dataclass(frozen=True)
class RetrievalTrace:
    """一次检索的可复现实验记录。"""

    trace_id: str
    query: str
    search_query: str
    query_type: QueryType | None
    candidate_count: int
    results: tuple[ScoredChunk, ...]
    elapsed_ms: float
    stages: tuple[str, ...] = ()
    expanded_queries: tuple[str, ...] = ()
    stage_elapsed_ms: tuple[tuple[str, float], ...] = ()
    index_version: str | None = None
    experiment_mode: str | None = None
    query_type_source: str | None = None
    scope_source: str | None = None
    target_paper_keys: tuple[str, ...] = ()
    scope_fallback_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.trace_id, "trace_id")
        _require_text(self.query, "query")
        _require_text(self.search_query, "search_query")
        if self.query_type is not None and not isinstance(self.query_type, QueryType):
            raise SchemaValidationError("query_type 必须是 QueryType 或 None。")
        if not isinstance(self.candidate_count, int) or self.candidate_count < len(self.results):
            raise SchemaValidationError("candidate_count 不能小于最终结果数量。")
        _validate_score(self.elapsed_ms, "elapsed_ms")
        if self.elapsed_ms < 0:
            raise SchemaValidationError("elapsed_ms 不能为负数。")
        if any(not isinstance(item, ScoredChunk) for item in self.results):
            raise SchemaValidationError("results 必须由 ScoredChunk 组成。")
        if not isinstance(self.stages, tuple) or any(not item.strip() for item in self.stages):
            raise SchemaValidationError("stages 必须是由非空字符串组成的元组。")
        if not isinstance(self.expanded_queries, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.expanded_queries
        ):
            raise SchemaValidationError("expanded_queries 必须是由非空字符串组成的元组。")
        if not isinstance(self.stage_elapsed_ms, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0].strip()
            or not isinstance(item[1], int | float)
            or not math.isfinite(item[1])
            or item[1] < 0
            for item in self.stage_elapsed_ms
        ):
            raise SchemaValidationError("stage_elapsed_ms 必须是非负阶段耗时元组。")
        if self.index_version is not None:
            _require_text(self.index_version, "index_version")
        provenance = (self.experiment_mode, self.query_type_source, self.scope_source)
        if any(value is not None for value in provenance):
            if not all(isinstance(value, str) and value.strip() for value in provenance):
                raise SchemaValidationError("P2 Trace 的模式与来源字段必须同时提供。")
            if self.experiment_mode not in _P2_RUN_MODES:
                raise SchemaValidationError("experiment_mode 不是支持的 P2 运行模式。")
            if self.query_type_source not in _QUERY_TYPE_SOURCES:
                raise SchemaValidationError("query_type_source 不是支持的来源。")
            if self.scope_source not in _SCOPE_SOURCES:
                raise SchemaValidationError("scope_source 不是支持的来源。")
            if self.experiment_mode == "oracle" and (
                self.query_type_source != "oracle_eval" or self.scope_source != "oracle_eval"
            ):
                raise SchemaValidationError("Oracle Trace 必须显式记录 oracle_eval 来源。")
            if self.experiment_mode == "end_to_end" and (
                self.query_type_source == "oracle_eval" or self.scope_source == "oracle_eval"
            ):
                raise SchemaValidationError("端到端 Trace 不能记录 Oracle 来源。")
        if (
            not isinstance(self.target_paper_keys, tuple)
            or any(not isinstance(key, str) or not key.strip() for key in self.target_paper_keys)
            or len(set(self.target_paper_keys)) != len(self.target_paper_keys)
        ):
            raise SchemaValidationError("target_paper_keys 必须是不重复的非空字符串元组。")
        if self.scope_fallback_reason is not None:
            _require_text(self.scope_fallback_reason, "scope_fallback_reason")


@dataclass(frozen=True)
class Answer:
    """生成服务返回的回答、证据与拒答状态。"""

    text: str
    citations: tuple[Citation, ...]
    refused: bool
    trace_id: str
    latency_ms: float
    refusal_reason: str | None = None
    claims: tuple[AnswerClaim, ...] = ()
    contract_version: str = "legacy"
    numeric_fidelity_passed: bool | None = None
    rewrite_model_calls: int = 0
    answer_model_calls: int = 0
    metadata_model_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit: bool = False
    estimated_cost: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.text, "text")
        _require_text(self.trace_id, "trace_id")
        _validate_score(self.latency_ms, "latency_ms")
        if self.latency_ms < 0:
            raise SchemaValidationError("latency_ms 不能为负数。")
        if any(not isinstance(item, Citation) for item in self.citations):
            raise SchemaValidationError("citations 必须由 Citation 组成。")
        if any(not isinstance(item, AnswerClaim) for item in self.claims):
            raise SchemaValidationError("claims 必须由 AnswerClaim 组成。")
        _require_text(self.contract_version, "contract_version")
        if self.numeric_fidelity_passed is not None and not isinstance(self.numeric_fidelity_passed, bool):
            raise SchemaValidationError("numeric_fidelity_passed 必须是布尔值或 None。")
        for field_name, value in (
            ("rewrite_model_calls", self.rewrite_model_calls),
            ("answer_model_calls", self.answer_model_calls),
            ("metadata_model_calls", self.metadata_model_calls),
        ):
            if not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"{field_name} 必须是非负整数。")
        for field_name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise SchemaValidationError(f"{field_name} 必须是非负整数或 None。")
        if not isinstance(self.cache_hit, bool):
            raise SchemaValidationError("cache_hit 必须是布尔值。")
        if self.estimated_cost is not None and (
            not isinstance(self.estimated_cost, int | float) or self.estimated_cost < 0
        ):
            raise SchemaValidationError("estimated_cost 必须是非负数或 None。")
        if self.refused:
            if self.citations:
                raise SchemaValidationError("拒答不能附带论文证据。")
            _require_text(self.refusal_reason or "", "refusal_reason")
        elif self.refusal_reason is not None:
            raise SchemaValidationError("非拒答结果不能包含 refusal_reason。")


def to_jsonable(value: Any) -> Any:
    """将领域对象递归转换为可直接写入 JSON/JSONL 的基础类型。"""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
