"""ResearchPilot 后端服务入口：ingest → retrieve → answer。"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

from .absence_detection import infer_requested_field, scan_document_fields
from .annotation_service import AnnotationMode, AnnotationService
from .answer_contract import enforce_answer_contract
from .evidence_gate import cited_role_stages, full_document_scan_stages
from .generation import (
    METADATA_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
    AnswerGenerator,
    GenerationContext,
    MetadataIntent,
    PaperMetadataExtractor,
    format_metadata_field_answer,
    infer_metadata_intent,
    infer_summary_intent,
    summary_request_fingerprint,
)
from .index import ChunkKind, VectorIndexRepository
from .ingestion import IngestionConfig, IngestionResult, ingest_pdf
from .retrieval import DenseRetriever, RetrievalResult, RetrievalService
from .schemas import (
    Answer,
    AnswerClaim,
    BlockType,
    Chunk,
    Citation,
    QueryType,
    RetrievalTrace,
    ScoredChunk,
    sha256_text,
)
from .storage import CachedPaperSummary, PaperArtifactVersion, PaperMetadata, SQLiteMemoryRepository


@dataclass(frozen=True)
class AnswerExecution:
    """一次完整问答的结构化结果，供 CLI 与评测记录检索轨迹。"""

    answer: Answer
    retrieval: RetrievalResult


class ResearchPilotService:
    """页面、CLI 和未来 API 共用的唯一 RAG 核心入口。"""

    def __init__(
        self,
        *,
        index_repository: VectorIndexRepository,
        answer_generator: AnswerGenerator | None = None,
        ingestion_config: IngestionConfig | None = None,
        retrieval_service: RetrievalService | None = None,
        artifact_repository: SQLiteMemoryRepository | None = None,
        metadata_extractor: PaperMetadataExtractor | None = None,
        artifact_model_name: str = "unknown",
        annotation_service: AnnotationService | None = None,
    ) -> None:
        self._index_repository = index_repository
        self._answer_generator = answer_generator
        self._ingestion_config = ingestion_config
        self._retrieval_service = retrieval_service or RetrievalService(DenseRetriever(index_repository))
        self._artifact_repository = artifact_repository
        self._metadata_extractor = metadata_extractor
        self._artifact_model_name = artifact_model_name.strip() or "unknown"
        self._annotation_service = annotation_service

    def _artifact_version(self, prompt_version: str) -> PaperArtifactVersion:
        config = self._ingestion_config
        return PaperArtifactVersion(
            parser_name=config.parser_name if config is not None else "unknown",
            parser_version=config.parser_version if config is not None else "unknown",
            chunker_version=config.chunker_version if config is not None else "unknown",
            prompt_version=prompt_version,
            model_name=self._artifact_model_name,
        )

    @staticmethod
    def _cached_retrieval(
        *,
        question: str,
        query_type: QueryType,
        citations: tuple[Citation, ...],
        version: PaperArtifactVersion,
        source_name: str,
        cache_stage: str = "paper_artifact_cache_hit",
    ) -> RetrievalResult:
        """Recreate a minimal traceable context from persisted citations.

        The cache avoids retrieval and model calls, but callers still receive a
        normal ``AnswerExecution`` and Streamlit can render the original page
        evidence. Scores intentionally remain absent: they belong to the old
        retrieval run and must not be presented as a fresh ranking.
        """
        results = tuple(
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=citation.chunk_id,
                    paper_id=citation.paper_id,
                    parent_id=None,
                    page_start=citation.page_start,
                    page_end=citation.page_end,
                    section_path=citation.section_path,
                    content=citation.evidence,
                    block_types=(BlockType.OTHER,),
                    content_hash=sha256_text(citation.evidence),
                    chunker_version=version.chunker_version,
                ),
                source_name=source_name,
            )
            for citation in citations
        )
        trace = RetrievalTrace(
            trace_id=f"trace_cache_{uuid.uuid4().hex}",
            query=question.strip(),
            search_query=question.strip(),
            query_type=query_type,
            candidate_count=len(results),
            results=results,
            elapsed_ms=0.0,
            stages=(cache_stage,),
        )
        return RetrievalResult(results=results, trace=trace)

    @staticmethod
    def _cached_answer(
        *,
        text: str,
        citations: tuple[Citation, ...],
        retrieval: RetrievalResult,
        claims: tuple[AnswerClaim, ...] = (),
        contract_version: str = "legacy",
        numeric_fidelity_passed: bool | None = None,
        cache_hit: bool = True,
    ) -> Answer:
        artifact = Answer(
            text=text,
            citations=citations,
            refused=False,
            trace_id=retrieval.trace.trace_id,
            latency_ms=retrieval.trace.elapsed_ms,
            claims=claims,
            contract_version=contract_version,
            numeric_fidelity_passed=numeric_fidelity_passed,
            cache_hit=cache_hit,
        )
        if contract_version != "legacy":
            return artifact
        return enforce_answer_contract(artifact, default_to_all_citations=True)

    @staticmethod
    def _metadata_answer(
        *,
        metadata: PaperMetadata,
        intent: MetadataIntent,
        retrieval: RetrievalResult,
        cache_hit: bool = True,
    ) -> Answer:
        text, available = format_metadata_field_answer(metadata, intent)
        if not available:
            return Answer(
                text=text,
                citations=(),
                refused=True,
                refusal_reason="metadata_field_not_available",
                trace_id=retrieval.trace.trace_id,
                latency_ms=retrieval.trace.elapsed_ms,
            )
        return ResearchPilotService._cached_answer(
            text=text,
            citations=metadata.citations,
            retrieval=retrieval,
            cache_hit=cache_hit,
        )

    def _full_document_field_execution(
        self,
        *,
        question: str,
        user_id: str,
        session_id: str,
        paper_id: str,
        query_type: QueryType | None,
    ) -> AnswerExecution | None:
        field_type = infer_requested_field(question)
        if field_type is None:
            return None
        started_at = time.perf_counter()
        children = self._index_repository.list_children(
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
        )
        scan = scan_document_fields(children, field_type=field_type)
        results = tuple(
            ScoredChunk(chunk=item.hit.chunk, source_name=item.hit.source_name) for item in scan.matches[:5]
        )
        trace = RetrievalTrace(
            trace_id=f"trace_field_{uuid.uuid4().hex}",
            query=question.strip(),
            search_query=question.strip(),
            query_type=query_type or QueryType.LOCAL_FACT,
            candidate_count=len(children),
            results=results,
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            stages=full_document_scan_stages(scan),
        )
        retrieval = RetrievalResult(results=results, trace=trace)
        if not scan.matches:
            return AnswerExecution(
                answer=Answer(
                    text="已扫描当前论文的全部可提取文本，但没有找到所询问的字段。",
                    citations=(),
                    refused=True,
                    refusal_reason="full_document_field_not_found",
                    trace_id=trace.trace_id,
                    latency_ms=trace.elapsed_ms,
                ),
                retrieval=retrieval,
            )
        citations = tuple(
            Citation(
                paper_id=item.hit.chunk.paper_id,
                chunk_id=item.hit.chunk.chunk_id,
                page_start=item.hit.chunk.page_start,
                page_end=item.hit.chunk.page_end,
                section_path=item.hit.chunk.section_path,
                evidence=item.hit.chunk.content,
            )
            for item in scan.matches[:5]
        )
        lines = ["在论文全文中找到以下可核验字段："]
        lines.extend(
            f"- **{item.field_type.value}**：{item.value} [证据 {position}]"
            for position, item in enumerate(scan.matches[:5], start=1)
        )
        answer = enforce_answer_contract(
            Answer(
                text="\n".join(lines),
                citations=citations,
                refused=False,
                trace_id=trace.trace_id,
                latency_ms=trace.elapsed_ms,
            )
        )
        return AnswerExecution(answer=answer, retrieval=retrieval)

    def ingest_pdf(
        self,
        pdf_source: Path | bytes | BinaryIO,
        *,
        source_name: str | None,
        user_id: str,
        session_id: str,
    ) -> IngestionResult:
        """解析一篇 PDF，并以当前用户/会话范围写入父块和子块。"""
        result = ingest_pdf(
            pdf_source,
            source_name=source_name,
            config=self._ingestion_config,
        )
        if self._artifact_repository is not None:
            # ``paper_id`` is content-derived. A successful re-ingestion may
            # still use a new parser/chunker, so no cached generated artifact
            # may outlive the freshly written index.
            self._artifact_repository.delete_paper_artifacts(
                user_id=user_id,
                session_id=session_id,
                paper_id=result.paper.paper_id,
            )
        # 同一论文内容曾以旧 Parser / Chunker 写入当前 Scope 时，不能让新旧子块
        # 混在一次召回中。只有新解析完整成功后才删除旧记录，随后写入同一 paper_id
        # 的新版 Parent/Child，避免失败重建破坏仍可用的旧索引。
        self._index_repository.delete_paper(
            user_id=user_id,
            session_id=session_id,
            paper_id=result.paper.paper_id,
        )
        self._index_repository.upsert(
            chunks=result.parent_chunks,
            chunk_kind=ChunkKind.PARENT,
            source_name=result.paper.source_name,
            user_id=user_id,
            session_id=session_id,
        )
        self._index_repository.upsert(
            chunks=result.child_chunks,
            chunk_kind=ChunkKind.CHILD,
            source_name=result.paper.source_name,
            user_id=user_id,
            session_id=session_id,
        )
        # 文档内容变化后，当前 Scope 的持久化 BM25 缓存必须失效；下一次检索会
        # 用新的子 Chunk 指纹加载或重建索引，而不会继续使用旧语料。
        invalidate_scope = getattr(self._retrieval_service, "invalidate_scope", None)
        if callable(invalidate_scope):
            invalidate_scope(user_id=user_id, session_id=session_id)
        return result

    def retrieve(
        self,
        *,
        question: str,
        user_id: str,
        session_id: str,
        search_query: str | None = None,
        query_type: QueryType | None = None,
        recent_history: str = "",
        paper_id: str | None = None,
    ) -> RetrievalResult:
        """检索当前用户、当前会话中可引用的论文上下文。"""
        return self._retrieval_service.retrieve(
            query=question,
            search_query=search_query,
            query_type=query_type,
            user_id=user_id,
            session_id=session_id,
            recent_history=recent_history,
            paper_id=paper_id,
        )

    def delete_paper(self, *, user_id: str, session_id: str, paper_id: str) -> int:
        """Delete one paper's parent/child vectors and invalidate this scope's BM25 cache."""
        deleted = self._index_repository.delete_paper(user_id=user_id, session_id=session_id, paper_id=paper_id)
        invalidate_scope = getattr(self._retrieval_service, "invalidate_scope", None)
        if callable(invalidate_scope):
            invalidate_scope(user_id=user_id, session_id=session_id)
        if self._artifact_repository is not None:
            self._artifact_repository.delete_paper_scope_data(
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
            )
        return deleted

    def answer(
        self,
        *,
        question: str,
        user_id: str,
        session_id: str,
        search_query: str | None = None,
        query_type: QueryType | None = None,
        recent_history: str = "",
        context: GenerationContext | None = None,
        paper_id: str | None = None,
    ) -> Answer:
        """完成检索和回答；Answer 始终携带 trace_id、引用或明确拒答。"""
        return self.answer_with_retrieval(
            question=question,
            user_id=user_id,
            session_id=session_id,
            search_query=search_query,
            query_type=query_type,
            recent_history=recent_history,
            context=context,
            paper_id=paper_id,
        ).answer

    def answer_with_retrieval(
        self,
        *,
        question: str,
        user_id: str,
        session_id: str,
        search_query: str | None = None,
        query_type: QueryType | None = None,
        recent_history: str = "",
        context: GenerationContext | None = None,
        paper_id: str | None = None,
        on_text_delta: Callable[[str], None] | None = None,
        annotation_mode: AnnotationMode = "paper_only",
    ) -> AnswerExecution:
        """完成问答并保留 RetrievalResult，供评测/JSONL Runner 使用。"""
        if self._answer_generator is None:
            raise RuntimeError("当前服务未配置 AnswerGenerator，无法调用 answer。")
        metadata_intent = infer_metadata_intent(question)
        summary_intent = infer_summary_intent(question)
        summary_fingerprint = summary_request_fingerprint(question, summary_intent)
        if paper_id:
            field_execution = self._full_document_field_execution(
                question=question,
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
                query_type=query_type,
            )
            if field_execution is not None:
                if on_text_delta is not None:
                    on_text_delta(field_execution.answer.text)
                return field_execution
        if query_type is QueryType.METADATA and metadata_intent is MetadataIntent.OTHER:
            version = self._artifact_version(METADATA_PROMPT_VERSION)
            retrieval = self._cached_retrieval(
                question=question,
                query_type=QueryType.METADATA,
                citations=(),
                version=version,
                source_name=paper_id or "unknown",
                cache_stage="metadata_unsupported_field",
            )
            answer = Answer(
                text="论文元信息不包含所询问的个人信息，无法可靠回答。",
                citations=(),
                refused=True,
                refusal_reason="metadata_field_not_available",
                trace_id=retrieval.trace.trace_id,
                latency_ms=0.0,
            )
            if on_text_delta is not None:
                on_text_delta(answer.text)
            return AnswerExecution(answer=answer, retrieval=retrieval)
        if query_type is QueryType.METADATA and paper_id and self._artifact_repository is not None:
            version = self._artifact_version(METADATA_PROMPT_VERSION)
            cached = self._artifact_repository.load_paper_metadata(
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
                version=version,
            )
            if cached is not None:
                retrieval = self._cached_retrieval(
                    question=question,
                    query_type=QueryType.METADATA,
                    citations=cached.citations,
                    version=version,
                    source_name=cached.source_name,
                    cache_stage=f"paper_metadata_cache_hit:{metadata_intent.value}",
                )
                execution = AnswerExecution(
                    answer=self._metadata_answer(
                        metadata=cached,
                        intent=metadata_intent,
                        retrieval=retrieval,
                        cache_hit=True,
                    ),
                    retrieval=retrieval,
                )
                if on_text_delta is not None:
                    on_text_delta(execution.answer.text)
                return execution
        if query_type is QueryType.DOCUMENT_SUMMARY and paper_id and self._artifact_repository is not None:
            version = self._artifact_version(SUMMARY_PROMPT_VERSION)
            cached = self._artifact_repository.load_paper_summary(
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
                version=version,
                summary_intent=summary_intent.value,
                request_fingerprint=summary_fingerprint,
            )
            if cached is not None:
                retrieval = self._cached_retrieval(
                    question=question,
                    query_type=QueryType.DOCUMENT_SUMMARY,
                    citations=cached.citations,
                    version=version,
                    source_name=cached.source_name,
                    cache_stage=f"paper_summary_cache_hit:{summary_intent.value}",
                )
                execution = AnswerExecution(
                    answer=self._cached_answer(
                        text=cached.answer_text,
                        citations=cached.citations,
                        retrieval=retrieval,
                        claims=cached.claims,
                        contract_version=cached.contract_version,
                        numeric_fidelity_passed=cached.numeric_fidelity_passed,
                    ),
                    retrieval=retrieval,
                )
                if on_text_delta is not None:
                    on_text_delta(execution.answer.text)
                return execution
        retrieval = self.retrieve(
            question=question,
            user_id=user_id,
            session_id=session_id,
            search_query=search_query,
            query_type=query_type,
            recent_history=recent_history,
            paper_id=paper_id,
        )
        if (
            query_type is QueryType.METADATA
            and paper_id
            and self._artifact_repository is not None
            and self._metadata_extractor is not None
        ):
            extracted = self._metadata_extractor.extract(retrieval=retrieval)
            if extracted is not None:
                version = self._artifact_version(METADATA_PROMPT_VERSION)
                metadata = PaperMetadata(
                    user_id=user_id,
                    session_id=session_id,
                    paper_id=paper_id,
                    source_name=next(
                        (
                            item.source_name
                            for item in retrieval.results
                            if isinstance(item.source_name, str) and item.source_name.strip()
                        ),
                        paper_id,
                    ),
                    answer_text=extracted.answer_text,
                    citations=tuple(
                        Citation(
                            paper_id=item.chunk.paper_id,
                            chunk_id=item.chunk.chunk_id,
                            page_start=item.chunk.page_start,
                            page_end=item.chunk.page_end,
                            section_path=item.chunk.section_path,
                            evidence=item.chunk.content,
                            confidence=(item.rerank_score or item.fusion_score or item.dense_score),
                        )
                        for item in retrieval.results
                    ),
                    version=version,
                    title=extracted.title,
                    authors=extracted.authors,
                    affiliations=extracted.affiliations,
                    venue=extracted.venue,
                    year=extracted.year,
                    doi=extracted.doi,
                    field_confidence=extracted.field_confidence,
                )
                self._artifact_repository.save_paper_metadata(metadata)
                execution = AnswerExecution(
                    answer=self._metadata_answer(
                        metadata=metadata,
                        intent=metadata_intent,
                        retrieval=retrieval,
                        cache_hit=False,
                    ),
                    retrieval=retrieval,
                )
                if on_text_delta is not None:
                    on_text_delta(execution.answer.text)
                usage = self._metadata_extractor.last_usage
                return AnswerExecution(
                    answer=replace(
                        execution.answer,
                        metadata_model_calls=usage.model_calls,
                        prompt_tokens=(usage.prompt_tokens if usage.token_observations else None),
                        completion_tokens=(usage.completion_tokens if usage.token_observations else None),
                        estimated_cost=self._metadata_extractor.last_estimated_cost,
                    ),
                    retrieval=retrieval,
                )
        resolved_context = context or GenerationContext()
        if self._annotation_service is not None and paper_id:
            annotation_text = self._annotation_service.context_for_chunks(
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
                chunk_ids=tuple(item.chunk.chunk_id for item in retrieval.results),
                mode=annotation_mode,
                pages=tuple(item.chunk.page_start for item in retrieval.results),
                section_paths=tuple(item.chunk.section_path for item in retrieval.results),
                query=question,
            )
            if annotation_text:
                resolved_context = GenerationContext(
                    long_term_memory=resolved_context.long_term_memory,
                    working_memory=resolved_context.working_memory,
                    chat_history=resolved_context.chat_history,
                    user_annotations=annotation_text,
                )
        answer = self._answer_generator.generate(
            question=question,
            retrieval=retrieval,
            context=resolved_context,
            on_text_delta=on_text_delta,
        )
        if query_type is QueryType.MULTI_SECTION:
            role_stages = cited_role_stages(answer, retrieval.trace)
            if role_stages:
                retrieval = replace(
                    retrieval,
                    trace=replace(
                        retrieval.trace,
                        stages=tuple(dict.fromkeys((*retrieval.trace.stages, *role_stages))),
                    ),
                )
        if (
            query_type is QueryType.DOCUMENT_SUMMARY
            and paper_id
            and self._artifact_repository is not None
            and not answer.refused
        ):
            self._artifact_repository.save_paper_summary(
                CachedPaperSummary(
                    user_id=user_id,
                    session_id=session_id,
                    paper_id=paper_id,
                    source_name=next(
                        (
                            item.source_name
                            for item in retrieval.results
                            if isinstance(item.source_name, str) and item.source_name.strip()
                        ),
                        paper_id,
                    ),
                    answer_text=answer.text,
                    citations=answer.citations,
                    claims=answer.claims,
                    contract_version=answer.contract_version,
                    numeric_fidelity_passed=answer.numeric_fidelity_passed,
                    version=self._artifact_version(SUMMARY_PROMPT_VERSION),
                    summary_intent=summary_intent.value,
                    request_fingerprint=summary_fingerprint,
                )
            )
        return AnswerExecution(answer=answer, retrieval=retrieval)
