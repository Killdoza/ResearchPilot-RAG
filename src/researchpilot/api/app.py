"""FastAPI application factory for the existing ResearchPilot RAG Core."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..config import ConfigurationError, Settings, build_settings, ensure_runtime_directories, validate_storage_id
from ..generation import ModelServiceUnavailable
from ..ingestion import PdfIngestionError
from ..observability import LocalTraceStore, new_trace_id
from ..orchestration import RuntimeOrchestrator
from ..product_config import load_product_profile
from ..routing import RuntimePaper, RuntimeRequest
from .api_schemas import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
    ConfigResponse,
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    ReadyResponse,
    RetrievedChunkResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from .error_handlers import ApiError, api_error_handler, error_response, unexpected_error_handler, value_error_handler
from .runtime import DocumentRegistry, RuntimeServiceProvider, ServiceProvider

API_VERSION = "v1"


def _safe_filename(filename: str | None) -> str:
    if not filename:
        raise ApiError(422, "missing_filename", "上传文件必须包含文件名。")
    leaf = Path(filename).name
    if not leaf.lower().endswith(".pdf"):
        raise ApiError(415, "unsupported_media_type", "当前仅支持 .pdf；DOCX/OCR 是路线图后续独立能力。")
    if not leaf or leaf in {".", ".."}:
        raise ApiError(422, "invalid_filename", "文件名不合法。")
    return leaf


def _chunk_response(scored: Any) -> RetrievedChunkResponse:
    chunk = scored.chunk
    return RetrievedChunkResponse(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        source_name=scored.source_name,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=list(chunk.section_path),
        dense_score=scored.dense_score,
        sparse_score=scored.sparse_score,
        fusion_score=scored.fusion_score,
        rerank_score=scored.rerank_score,
    )


def _document_response(metadata: dict[str, object]) -> DocumentResponse:
    return DocumentResponse.model_validate(metadata)


def _validate_request_scope(*, user_id: str, session_id: str) -> None:
    try:
        validate_storage_id(user_id, "user_id")
        validate_storage_id(session_id, "session_id")
    except ConfigurationError as exc:
        raise ApiError(422, "invalid_scope", str(exc)) from exc


def _validate_optional_paper_id(paper_id: str | None) -> None:
    if paper_id is None:
        return
    try:
        validate_storage_id(paper_id, "paper_id")
    except ConfigurationError as exc:
        raise ApiError(422, "invalid_paper_id", str(exc)) from exc


def _runtime_papers(documents: list[dict[str, object]]) -> tuple[RuntimePaper, ...]:
    """Convert the scoped document registry into Gold-free router input."""
    papers: list[RuntimePaper] = []
    for document in documents:
        paper_id = document.get("paper_id")
        source_name = document.get("source_name")
        if not isinstance(paper_id, str) or not isinstance(source_name, str):
            continue
        page_count = document.get("page_count")
        parser_name = document.get("parser_name")
        chunker_version = document.get("chunker_version")
        extraction_quality = document.get("extraction_quality")
        papers.append(
            RuntimePaper(
                paper_id=paper_id,
                source_name=source_name,
                page_count=page_count if isinstance(page_count, int) else None,
                parser_name=parser_name if isinstance(parser_name, str) else None,
                chunker_version=chunker_version if isinstance(chunker_version, str) else None,
                extraction_quality=(extraction_quality if isinstance(extraction_quality, str) else None),
            )
        )
    return tuple(papers)


def _citation_response(citation: Any) -> CitationResponse:
    return CitationResponse(
        paper_id=citation.paper_id,
        chunk_id=citation.chunk_id,
        page_start=citation.page_start,
        page_end=citation.page_end,
        section_path=list(citation.section_path),
        confidence=citation.confidence,
    )


def create_app(
    *,
    settings: Settings | None = None,
    service_provider: ServiceProvider | None = None,
    trace_store: LocalTraceStore | None = None,
) -> FastAPI:
    """Create an app without loading embeddings, Chroma, CrossEncoder or DeepSeek."""
    resolved_settings = settings or build_settings()
    ensure_runtime_directories(resolved_settings)
    product_profile = load_product_profile(project_root=resolved_settings.project_root)
    provider = service_provider or RuntimeServiceProvider(resolved_settings)
    registry = DocumentRegistry(resolved_settings)
    traces = trace_store or LocalTraceStore(resolved_settings.traces_dir)

    app = FastAPI(title="ResearchPilot API", version="1.0.0", openapi_url="/openapi.json")
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)

    @app.middleware("http")
    async def attach_trace_id(request: Request, call_next: Any) -> Any:
        request.state.trace_id = new_trace_id()
        response = await call_next(request)
        response.headers["X-Trace-Id"] = request.state.trace_id
        return response

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="researchpilot-api", version=app.version)

    @app.get("/ready", response_model=ReadyResponse, tags=["system"])
    def ready() -> ReadyResponse:
        required = (resolved_settings.sessions_dir, resolved_settings.uploads_dir, resolved_settings.traces_dir)
        directories_ready = all(path.is_dir() for path in required)
        return ReadyResponse(
            status="ready" if directories_ready else "degraded",
            runtime_directories_ready=directories_ready,
            deepseek_key_configured=bool(resolved_settings.deepseek_api_key),
            strict_crossdoc_enabled=resolved_settings.api_strict_crossdoc_enabled,
        )

    @app.get("/v1/config", response_model=ConfigResponse, tags=["system"])
    def config() -> ConfigResponse:
        return ConfigResponse(
            api_version=API_VERSION,
            default_mode="default",
            product_profile=product_profile.name,
            product_profile_hash=product_profile.config_hash,
            strict_crossdoc_enabled=resolved_settings.api_strict_crossdoc_enabled,
            supported_upload_types=["application/pdf"],
            trace_storage="local_jsonl_open_telemetry_spans",
        )

    @app.post("/v1/documents", response_model=DocumentResponse, status_code=201, tags=["documents"])
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
        user_id: str = Form(...),
        session_id: str = Form(...),
    ) -> DocumentResponse:
        started_at = time.perf_counter()
        _validate_request_scope(user_id=user_id, session_id=session_id)
        source_name = _safe_filename(file.filename)
        if file.content_type not in {None, "application/pdf", "application/octet-stream"}:
            raise ApiError(415, "unsupported_media_type", "上传内容必须是 PDF。")
        content = await file.read()
        if not content:
            raise ApiError(422, "empty_file", "PDF 文件为空。")
        try:
            result = provider.get().ingest_pdf(content, source_name=source_name, user_id=user_id, session_id=session_id)
        except PdfIngestionError as exc:
            traces.record(
                trace_id=request.state.trace_id,
                operation="document.ingest",
                started_at=started_at,
                user_id=user_id,
                session_id=session_id,
                status="rejected",
                error_code="pdf_ingestion_failed",
            )
            raise ApiError(422, "pdf_ingestion_failed", str(exc)) from exc
        except ConfigurationError as exc:
            raise ApiError(503, "service_not_configured", str(exc)) from exc
        quality = result.parse_quality
        payload: dict[str, object] = {
            "paper_id": result.paper.paper_id,
            "source_name": result.paper.source_name,
            "page_count": result.paper.page_count,
            "language": result.paper.language,
            "parent_chunk_count": len(result.parent_chunks),
            "child_chunk_count": len(result.child_chunks),
            "parser_name": quality.parser_name if quality else None,
            "parser_version": getattr(quality, "parser_version", None) if quality else None,
            "requested_parser_name": getattr(result, "requested_parser_name", None),
            "chunker_version": product_profile.ingestion.chunker_version,
            "product_profile": product_profile.name,
            "product_profile_hash": product_profile.config_hash,
            "fallback_used": bool(getattr(result, "fallback_used", False)),
            "fallback_reason": getattr(result, "fallback_reason", None),
            "extraction_quality": quality.extraction_quality.value if quality else None,
        }
        try:
            registry.save(user_id=user_id, session_id=session_id, metadata=payload, pdf_bytes=content)
        except OSError as exc:
            provider.get().delete_paper(user_id=user_id, session_id=session_id, paper_id=result.paper.paper_id)
            raise ApiError(507, "upload_persistence_failed", "上传元数据保存失败，索引已回滚。") from exc
        traces.record(
            trace_id=request.state.trace_id,
            operation="document.ingest",
            started_at=started_at,
            user_id=user_id,
            session_id=session_id,
            status="ok",
        )
        return _document_response(payload)

    @app.get("/v1/documents", response_model=DocumentListResponse, tags=["documents"])
    def list_documents(user_id: str, session_id: str) -> DocumentListResponse:
        _validate_request_scope(user_id=user_id, session_id=session_id)
        documents = registry.list(user_id=user_id, session_id=session_id)
        return DocumentListResponse(documents=[_document_response(item) for item in documents])

    @app.delete("/v1/documents/{paper_id}", response_model=DeleteDocumentResponse, tags=["documents"])
    def delete_document(request: Request, paper_id: str, user_id: str, session_id: str) -> DeleteDocumentResponse:
        started_at = time.perf_counter()
        _validate_request_scope(user_id=user_id, session_id=session_id)
        if registry.load(user_id=user_id, session_id=session_id, paper_id=paper_id) is None:
            raise ApiError(404, "document_not_found", "当前用户和会话范围内不存在该论文。")
        try:
            deleted = provider.get().delete_paper(user_id=user_id, session_id=session_id, paper_id=paper_id)
            registry.delete(user_id=user_id, session_id=session_id, paper_id=paper_id)
        except ConfigurationError as exc:
            raise ApiError(503, "service_not_configured", str(exc)) from exc
        traces.record(
            trace_id=request.state.trace_id,
            operation="document.delete",
            started_at=started_at,
            user_id=user_id,
            session_id=session_id,
            status="ok",
        )
        return DeleteDocumentResponse(paper_id=paper_id, deleted_vector_records=deleted)

    @app.post("/v1/retrieve", response_model=RetrieveResponse, tags=["rag"])
    def retrieve(request: Request, payload: RetrieveRequest) -> RetrieveResponse:
        started_at = time.perf_counter()
        _validate_request_scope(user_id=payload.user_id, session_id=payload.session_id)
        _validate_optional_paper_id(payload.paper_id)
        try:
            result = provider.get().retrieve(
                question=payload.question,
                user_id=payload.user_id,
                session_id=payload.session_id,
                paper_id=payload.paper_id,
                recent_history=payload.recent_history,
            )
        except ConfigurationError as exc:
            raise ApiError(503, "service_not_configured", str(exc)) from exc
        trace = result.trace
        traces.record(
            trace_id=request.state.trace_id,
            operation="rag.retrieve",
            started_at=started_at,
            user_id=payload.user_id,
            session_id=payload.session_id,
            status="ok",
            retrieval_trace_id=trace.trace_id,
            candidate_count=trace.candidate_count,
        )
        return RetrieveResponse(
            trace_id=trace.trace_id,
            search_query=trace.search_query,
            candidate_count=trace.candidate_count,
            elapsed_ms=trace.elapsed_ms,
            stages=list(trace.stages),
            results=[_chunk_response(item) for item in result.results],
        )

    @app.post("/v1/chat", response_model=ChatResponse, tags=["rag"])
    def chat(request: Request, payload: ChatRequest) -> ChatResponse:
        started_at = time.perf_counter()
        _validate_request_scope(user_id=payload.user_id, session_id=payload.session_id)
        _validate_optional_paper_id(payload.paper_id)
        if payload.mode == "strict_crossdoc":
            # P2 proved that this must use explicit per-paper Scope and G0/G1/G2.
            # The generic upload session has no paper-selection catalog yet, so it is
            # deliberately feature-gated instead of silently degrading to global Top-K.
            traces.record(
                trace_id=request.state.trace_id,
                operation="rag.chat",
                started_at=started_at,
                user_id=payload.user_id,
                session_id=payload.session_id,
                status="rejected",
                mode=payload.mode,
                error_code="strict_crossdoc_not_available",
            )
            raise ApiError(
                409,
                "strict_crossdoc_not_available",
                "严格跨论文模式需要显式论文范围和 P2 证据控制适配器；当前 API 未启用该 Feature Flag。",
            )
        runtime_request = RuntimeRequest(
            question=payload.question,
            user_id=payload.user_id,
            session_id=payload.session_id,
            available_papers=_runtime_papers(registry.list(user_id=payload.user_id, session_id=payload.session_id)),
            selected_paper_ids=(payload.paper_id,) if payload.paper_id else (),
            recent_history=payload.recent_history,
            product_mode=product_profile.name,
        )
        try:
            get_runtime = getattr(provider, "get_runtime", None)
            runtime = get_runtime() if callable(get_runtime) else None
            execution = RuntimeOrchestrator(
                service=runtime.service if runtime is not None else provider.get(),
                strict_crossdoc_enabled=resolved_settings.api_strict_crossdoc_enabled,
                conversation_service=(runtime.conversation_service if runtime is not None else None),
                memory_service=runtime.memory_service if runtime is not None else None,
                annotation_service=(runtime.annotation_service if runtime is not None else None),
            ).handle(runtime_request)
        except ModelServiceUnavailable as exc:
            traces.record(
                trace_id=request.state.trace_id,
                operation="rag.chat",
                started_at=started_at,
                user_id=payload.user_id,
                session_id=payload.session_id,
                status="failed",
                mode=payload.mode,
                error_code="model_service_unavailable",
            )
            raise ApiError(
                503,
                "model_service_unavailable",
                "模型服务暂时不可用；论文索引不受影响，请稍后重试。",
            ) from exc
        except ConfigurationError as exc:
            raise ApiError(503, "service_not_configured", str(exc)) from exc
        answer, retrieval = execution.answer, execution.retrieval
        candidate_count = retrieval.trace.candidate_count if retrieval is not None else 0
        retrieval_elapsed_ms = retrieval.trace.elapsed_ms if retrieval is not None else 0.0
        traces.record(
            trace_id=request.state.trace_id,
            operation="rag.chat",
            started_at=started_at,
            user_id=payload.user_id,
            session_id=payload.session_id,
            status="ok",
            retrieval_trace_id=answer.trace_id,
            candidate_count=candidate_count,
            citation_count=len(answer.citations),
            refused=answer.refused,
            mode=payload.mode,
            route=execution.decision.route.value,
            prompt_tokens=getattr(answer, "prompt_tokens", None),
            completion_tokens=getattr(answer, "completion_tokens", None),
            rewrite_model_calls=getattr(answer, "rewrite_model_calls", 0),
            answer_model_calls=getattr(answer, "answer_model_calls", 0),
            metadata_model_calls=getattr(answer, "metadata_model_calls", 0),
            cache_hit=getattr(answer, "cache_hit", False),
            estimated_cost=getattr(answer, "estimated_cost", None),
        )
        return ChatResponse(
            trace_id=answer.trace_id,
            answer=answer.text,
            citations=[_citation_response(item) for item in answer.citations],
            claims=[
                {
                    "claim_id": item.claim_id,
                    "text": item.text,
                    "citation_ids": list(item.citation_ids),
                    "support_status": item.support_status,
                    "numeric_values": list(item.numeric_values),
                }
                for item in getattr(answer, "claims", ())
            ],
            answer_contract_version=getattr(answer, "contract_version", "legacy"),
            numeric_fidelity_passed=getattr(answer, "numeric_fidelity_passed", None),
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            retrieval_elapsed_ms=retrieval_elapsed_ms,
            generation_elapsed_ms=answer.latency_ms,
            mode=payload.mode,
            route=execution.decision.route.value,
            rewrite_model_calls=getattr(answer, "rewrite_model_calls", 0),
            answer_model_calls=getattr(answer, "answer_model_calls", 0),
            metadata_model_calls=getattr(answer, "metadata_model_calls", 0),
            prompt_tokens=getattr(answer, "prompt_tokens", None),
            completion_tokens=getattr(answer, "completion_tokens", None),
            cache_hit=getattr(answer, "cache_hit", False),
            estimated_cost=getattr(answer, "estimated_cost", None),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            request=request,
            status_code=422,
            code="request_validation_failed",
            message="请求字段缺失或格式不合法。",
        )

    return app
