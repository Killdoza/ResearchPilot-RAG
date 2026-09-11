"""Regression tests for paper metadata and summary persistence caches."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.config import build_settings  # noqa: E402
from researchpilot.generation import (  # noqa: E402
    METADATA_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
    AnswerGenerator,
    PaperMetadataExtractor,
    SummaryIntent,
    summary_request_fingerprint,
)
from researchpilot.ingestion import IngestionConfig  # noqa: E402
from researchpilot.retrieval import RetrievalResult  # noqa: E402
from researchpilot.schemas import (  # noqa: E402
    AnswerClaim,
    BlockType,
    Chunk,
    QueryType,
    RetrievalTrace,
    ScoredChunk,
)
from researchpilot.services import ResearchPilotService  # noqa: E402
from researchpilot.storage import (  # noqa: E402
    CachedPaperSummary,
    PaperArtifactVersion,
    SQLiteMemoryRepository,
)


class _Index:
    def delete_paper(self, **kwargs: object) -> int:
        del kwargs
        return 0


class _FixedRetrieval:
    def __init__(self) -> None:
        parent = Chunk.create(
            paper_id="paper_demo",
            parent_id="block_demo",
            page_start=1,
            page_end=1,
            section_path=("Title",),
            content="Demo Paper\nAlice Example\nProceedings of DemoConf 2026",
            block_types=(BlockType.PARAGRAPH,),
            chunker_version="structure_aware_v1",
        )
        self.result = RetrievalResult(
            results=(ScoredChunk(chunk=parent, source_name="demo.pdf", dense_rank=1, dense_score=0.9),),
            trace=RetrievalTrace(
                trace_id="trace_live",
                query="question",
                search_query="question",
                query_type=QueryType.METADATA,
                candidate_count=1,
                results=(ScoredChunk(chunk=parent, source_name="demo.pdf", dense_rank=1, dense_score=0.9),),
                elapsed_ms=2.0,
                stages=("front_matter_selection",),
            ),
        )
        self.calls = 0

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        self.calls += 1
        query_type = kwargs.get("query_type")
        trace = RetrievalTrace(
            trace_id=f"trace_live_{self.calls}",
            query=str(kwargs["query"]),
            search_query=str(kwargs.get("search_query") or kwargs["query"]),
            query_type=query_type if isinstance(query_type, QueryType) else None,
            candidate_count=1,
            results=self.result.results,
            elapsed_ms=2.0,
            stages=("front_matter_selection",),
        )
        return RetrievalResult(results=self.result.results, trace=trace)


class _JsonClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, **kwargs: str) -> str:
        del kwargs
        self.calls += 1
        return self.response


def _version(prompt_version: str = "v1") -> PaperArtifactVersion:
    return PaperArtifactVersion(
        parser_name="docling",
        parser_version="2.120.2",
        chunker_version="structure_aware_v1",
        prompt_version=prompt_version,
        model_name="deepseek-chat",
    )


class TestPaperArtifactPersistence:
    def setup_method(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        settings = build_settings(
            project_root=Path(self.temporary.name),
            environ={"RESEARCHPILOT_MEMORY_DB_PATH": "runtime/memory.sqlite3"},
            load_dotenv_file=False,
        )
        self.repository = SQLiteMemoryRepository(settings)
        self.repository.initialize_schema()

    def teardown_method(self) -> None:
        self.temporary.cleanup()

    def test_summary_cache_is_versioned_and_isolated(self) -> None:
        retriever = _FixedRetrieval()
        citation = retriever.result.results[0].chunk
        from researchpilot.schemas import Citation

        summary = CachedPaperSummary(
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            source_name="demo.pdf",
            answer_text="Cached summary",
            citations=(
                Citation(
                    paper_id="paper_demo",
                    chunk_id=citation.chunk_id,
                    page_start=1,
                    page_end=1,
                    section_path=("Title",),
                    evidence=citation.content,
                    confidence=0.9,
                ),
            ),
            version=_version(),
            summary_intent=SummaryIntent.OVERVIEW.value,
            request_fingerprint=summary_request_fingerprint("总结论文", SummaryIntent.OVERVIEW),
            claims=(
                AnswerClaim(
                    claim_id="c1",
                    text="Cached summary",
                    citation_ids=("e1",),
                    support_status="supported",
                ),
            ),
            contract_version="claim_citation_v2",
            numeric_fidelity_passed=True,
        )
        self.repository.save_paper_summary(summary)

        loaded = self.repository.load_paper_summary(
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            version=_version(),
            summary_intent=summary.summary_intent,
            request_fingerprint=summary.request_fingerprint,
        )
        assert loaded is not None
        assert loaded.claims == summary.claims
        assert loaded.contract_version == "claim_citation_v2"
        assert loaded.numeric_fidelity_passed is True
        assert (
            self.repository.load_paper_summary(
                user_id="user_b",
                session_id="session_a",
                paper_id="paper_demo",
                version=_version(),
                summary_intent=summary.summary_intent,
                request_fingerprint=summary.request_fingerprint,
            )
            is None
        )
        assert (
            self.repository.load_paper_summary(
                user_id="user_a",
                session_id="session_a",
                paper_id="paper_demo",
                version=_version("v2"),
                summary_intent=summary.summary_intent,
                request_fingerprint=summary.request_fingerprint,
            )
            is None
        )

    def test_metadata_and_summary_cache_are_removed_with_paper_or_session(self) -> None:
        retriever = _FixedRetrieval()
        client = _JsonClient(
            '{"title":"Demo Paper","authors":["Alice Example"],"affiliations":[],"venue":"DemoConf","year":"2026"}'
        )
        generator_client = _JsonClient("Section-aware summary")
        service = ResearchPilotService(
            index_repository=_Index(),  # type: ignore[arg-type]
            answer_generator=AnswerGenerator(generator_client),
            retrieval_service=retriever,  # type: ignore[arg-type]
            ingestion_config=IngestionConfig(
                parser_name="docling",
                parser_version="2.120.2",
                chunker_version="structure_aware_v1",
            ),
            artifact_repository=self.repository,
            metadata_extractor=PaperMetadataExtractor(client),
            artifact_model_name="deepseek-chat",
        )
        first_metadata = service.answer_with_retrieval(
            question="作者是谁？",
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            query_type=QueryType.METADATA,
        )
        second_metadata = service.answer_with_retrieval(
            question="论文标题是什么？",
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            query_type=QueryType.METADATA,
        )
        assert "Alice Example" in first_metadata.answer.text
        assert "Demo Paper" in second_metadata.answer.text
        assert "Alice Example" not in second_metadata.answer.text
        assert second_metadata.retrieval.trace.stages == ("paper_metadata_cache_hit:title",)
        assert client.calls == 1

        unavailable_metadata = service.answer_with_retrieval(
            question="作者今年多大？",
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            query_type=QueryType.METADATA,
        )
        assert unavailable_metadata.answer.refused is True
        assert unavailable_metadata.answer.citations == ()
        assert unavailable_metadata.answer.refusal_reason == "metadata_field_not_available"
        assert client.calls == 1

        first_summary = service.answer_with_retrieval(
            question="总结论文",
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            query_type=QueryType.DOCUMENT_SUMMARY,
        )
        second_summary = service.answer_with_retrieval(
            question="总结论文",
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            query_type=QueryType.DOCUMENT_SUMMARY,
        )
        assert "当前检索证据不足" in first_summary.answer.text
        assert second_summary.retrieval.trace.stages == ("paper_summary_cache_hit:overview",)
        assert second_summary.answer.text == first_summary.answer.text
        assert second_summary.answer.citations == first_summary.answer.citations
        assert second_summary.answer.claims == first_summary.answer.claims
        assert second_summary.answer.contract_version == first_summary.answer.contract_version
        assert second_summary.answer.numeric_fidelity_passed == first_summary.answer.numeric_fidelity_passed
        assert generator_client.calls == 1

        third_summary = service.answer_with_retrieval(
            question="从动机、方法和实验结果总结论文",
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            query_type=QueryType.DOCUMENT_SUMMARY,
        )
        assert "当前检索证据不足" in third_summary.answer.text
        assert third_summary.retrieval.trace.stages != ("paper_summary_cache_hit:overview",)
        assert generator_client.calls == 2

        counts = service.delete_paper(user_id="user_a", session_id="session_a", paper_id="paper_demo")
        assert counts == 0
        assert (
            self.repository.load_paper_metadata(
                user_id="user_a",
                session_id="session_a",
                paper_id="paper_demo",
                version=_version(METADATA_PROMPT_VERSION),
            )
            is None
        )
        assert (
            self.repository.load_paper_summary(
                user_id="user_a",
                session_id="session_a",
                paper_id="paper_demo",
                version=_version(SUMMARY_PROMPT_VERSION),
                summary_intent=SummaryIntent.OVERVIEW.value,
                request_fingerprint=summary_request_fingerprint("总结论文", SummaryIntent.OVERVIEW),
            )
            is None
        )

    def test_unknown_metadata_field_refuses_without_retrieval_or_model_call(self) -> None:
        retriever = _FixedRetrieval()
        client = _JsonClient('{"title":"Demo Paper","authors":["Alice Example"],"affiliations":[]}')
        service = ResearchPilotService(
            index_repository=_Index(),  # type: ignore[arg-type]
            answer_generator=AnswerGenerator(_JsonClient("unused")),
            retrieval_service=retriever,  # type: ignore[arg-type]
            artifact_repository=self.repository,
            metadata_extractor=PaperMetadataExtractor(client),
        )

        execution = service.answer_with_retrieval(
            question="作者的年龄是多少？",
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            query_type=QueryType.METADATA,
        )

        assert execution.answer.refused is True
        assert execution.answer.citations == ()
        assert execution.retrieval.trace.stages == ("metadata_unsupported_field",)
        assert retriever.calls == 0
        assert client.calls == 0
