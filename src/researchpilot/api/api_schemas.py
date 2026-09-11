"""HTTP 边界的数据契约；不暴露 Chroma 或内部领域对象。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    trace_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["researchpilot-api"]
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded"]
    runtime_directories_ready: bool
    deepseek_key_configured: bool
    strict_crossdoc_enabled: bool


class ConfigResponse(BaseModel):
    api_version: str
    default_mode: Literal["default"]
    product_profile: str
    product_profile_hash: str
    strict_crossdoc_enabled: bool
    supported_upload_types: list[str]
    trace_storage: str


class CitationResponse(BaseModel):
    paper_id: str
    chunk_id: str
    page_start: int
    page_end: int
    section_path: list[str]
    confidence: float | None = None


class ClaimResponse(BaseModel):
    claim_id: str
    text: str
    citation_ids: list[str]
    support_status: str
    numeric_values: list[str]


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    paper_id: str
    source_name: str | None = None
    page_start: int
    page_end: int
    section_path: list[str]
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    paper_id: str | None = Field(default=None, min_length=1, max_length=128)
    recent_history: str = Field(default="", max_length=24_000)


class RetrieveResponse(BaseModel):
    trace_id: str
    search_query: str
    candidate_count: int
    elapsed_ms: float
    stages: list[str]
    results: list[RetrievedChunkResponse]


class ChatRequest(RetrieveRequest):
    mode: Literal["default", "strict_crossdoc"] = "default"


class ChatResponse(BaseModel):
    trace_id: str
    answer: str
    citations: list[CitationResponse]
    claims: list[ClaimResponse] = Field(default_factory=list)
    answer_contract_version: str = "legacy"
    numeric_fidelity_passed: bool | None = None
    refused: bool
    refusal_reason: str | None = None
    retrieval_elapsed_ms: float
    generation_elapsed_ms: float
    mode: str
    route: str
    rewrite_model_calls: int = 0
    answer_model_calls: int = 0
    metadata_model_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit: bool = False
    estimated_cost: float | None = None


class DocumentResponse(BaseModel):
    paper_id: str
    source_name: str
    page_count: int | None = None
    language: str
    parent_chunk_count: int
    child_chunk_count: int
    parser_name: str | None = None
    parser_version: str | None = None
    requested_parser_name: str | None = None
    chunker_version: str | None = None
    product_profile: str | None = None
    product_profile_hash: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    extraction_quality: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DeleteDocumentResponse(BaseModel):
    paper_id: str
    deleted_vector_records: int
