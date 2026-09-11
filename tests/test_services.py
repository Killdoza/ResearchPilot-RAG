"""ResearchPilotService 的编排测试，不启动 UI、不连接外部服务。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.generation import AnswerGenerator  # noqa: E402
from researchpilot.index import ChunkKind, IndexHit  # noqa: E402
from researchpilot.ingestion import IngestionResult  # noqa: E402
from researchpilot.retrieval import RetrievalResult  # noqa: E402
from researchpilot.schemas import (  # noqa: E402
    BlockType,
    Chunk,
    DocumentBlock,
    Paper,
    QueryType,
    RetrievalTrace,
    sha256_bytes,
)
from researchpilot.services import ResearchPilotService  # noqa: E402


class _FakeIndexRepository:
    def __init__(self) -> None:
        self.upserts: list[dict[str, object]] = []
        self.children: tuple[IndexHit, ...] = ()

    def upsert(self, **kwargs: object) -> None:
        self.upserts.append(kwargs)

    def query_children(self, **kwargs: object) -> tuple[object, ...]:
        return ()

    def get_parent_chunks(self, **kwargs: object) -> dict[str, object]:
        return {}

    def list_children(self, **kwargs: object) -> tuple[IndexHit, ...]:
        return self.children

    def delete_session(self, **kwargs: object) -> int:
        return 0

    def delete_paper(self, **kwargs: object) -> int:
        self.deleted_paper = kwargs
        return 2


class _FixedRetrievalService:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> RetrievalResult:
        self.calls.append(kwargs)
        return self.result


class _FixedClient:
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "这是有证据支持的回答 [证据 1]。"


def _fixture() -> tuple[IngestionResult, RetrievalResult]:
    parent = Chunk.create(
        paper_id="paper_demo",
        parent_id="block_paper_demo_p0001",
        page_start=1,
        page_end=1,
        section_path=(),
        content="完整论文正文，用于验证服务编排。",
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test_v1",
    )
    child = Chunk.create(
        paper_id="paper_demo",
        parent_id=parent.chunk_id,
        page_start=1,
        page_end=1,
        section_path=(),
        content="论文正文验证服务编排。",
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test_v1",
    )
    paper = Paper(
        paper_id="paper_demo",
        source_name="demo.pdf",
        content_hash=sha256_bytes(b"demo pdf"),
        language="zh",
        page_count=1,
    )
    ingested = IngestionResult(
        paper=paper,
        blocks=(
            DocumentBlock(
                paper_id=paper.paper_id,
                page=1,
                block_id="block_paper_demo_p0001",
                block_type=BlockType.PARAGRAPH,
                text=parent.content,
                parser_name="test",
                parser_version="1",
            ),
        ),
        parent_chunks=(parent,),
        child_chunks=(child,),
        head_text=parent.content,
        full_text=parent.content,
    )
    from researchpilot.schemas import ScoredChunk

    scored = ScoredChunk(chunk=parent, source_name="demo.pdf", dense_rank=1, dense_score=0.9)
    trace = RetrievalTrace(
        trace_id="trace_service",
        query="方法是什么？",
        search_query="方法是什么？",
        query_type=None,
        candidate_count=1,
        results=(scored,),
        elapsed_ms=1.0,
        stages=("dense_child_retrieval", "parent_hydration"),
    )
    return ingested, RetrievalResult(results=(scored,), trace=trace)


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ingested, self.retrieved = _fixture()
        self.index = _FakeIndexRepository()
        self.retrieval = _FixedRetrievalService(self.retrieved)
        self.service = ResearchPilotService(
            index_repository=self.index,
            answer_generator=AnswerGenerator(_FixedClient()),
            retrieval_service=self.retrieval,  # type: ignore[arg-type]
        )

    def test_ingest_pdf_writes_parent_and_child_in_same_scope(self) -> None:
        with patch("researchpilot.services.ingest_pdf", return_value=self.ingested) as mocked_ingest:
            result = self.service.ingest_pdf(
                b"pdf bytes",
                source_name="demo.pdf",
                user_id="user_a",
                session_id="session_a",
            )

        self.assertIs(result, self.ingested)
        mocked_ingest.assert_called_once()
        self.assertEqual(
            self.index.deleted_paper,
            {"user_id": "user_a", "session_id": "session_a", "paper_id": "paper_demo"},
        )
        self.assertEqual([item["chunk_kind"] for item in self.index.upserts], [ChunkKind.PARENT, ChunkKind.CHILD])
        self.assertEqual(self.index.upserts[0]["user_id"], "user_a")
        self.assertEqual(self.index.upserts[1]["session_id"], "session_a")

    def test_answer_uses_service_retrieval_and_returns_traceable_answer(self) -> None:
        answer = self.service.answer(
            question="方法是什么？",
            user_id="user_a",
            session_id="session_a",
            recent_history="用户：上一轮在讨论方法。",
        )

        self.assertFalse(answer.refused)
        self.assertEqual(answer.trace_id, "trace_service")
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(self.retrieval.calls[0]["user_id"], "user_a")
        self.assertEqual(self.retrieval.calls[0]["recent_history"], "用户：上一轮在讨论方法。")

    def test_answer_with_retrieval_keeps_scores_for_evaluation_runner(self) -> None:
        execution = self.service.answer_with_retrieval(
            question="方法是什么？", user_id="user_a", session_id="session_a"
        )

        self.assertEqual(execution.answer.trace_id, "trace_service")
        self.assertEqual(execution.retrieval.results[0].dense_score, 0.9)

    def test_answer_passes_explicit_paper_scope_to_retrieval(self) -> None:
        self.service.answer(
            question="方法是什么？",
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
        )

        self.assertEqual(self.retrieval.calls[0]["paper_id"], "paper_demo")

    def test_answer_without_generator_is_rejected(self) -> None:
        service = ResearchPilotService(
            index_repository=self.index,
            retrieval_service=self.retrieval,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(RuntimeError, "AnswerGenerator"):
            service.answer(question="方法是什么？", user_id="user_a", session_id="session_a")

    def test_missing_code_url_uses_full_document_gate_without_model(self) -> None:
        child = self.ingested.child_chunks[0]
        self.index.children = (IndexHit(chunk=child, source_name="demo.pdf", score=0.0),)
        execution = self.service.answer_with_retrieval(
            question="论文有没有公开代码地址？",
            user_id="user_a",
            session_id="session_a",
            paper_id="paper_demo",
            query_type=QueryType.LOCAL_FACT,
        )
        assert execution.answer.refused
        assert execution.answer.refusal_reason == "full_document_field_not_found"
        assert not execution.answer.citations
        assert "full_document_scan" in execution.retrieval.trace.stages
        assert not self.retrieval.calls

    def test_delete_paper_delegates_the_exact_scope(self) -> None:
        deleted = self.service.delete_paper(user_id="user_a", session_id="session_a", paper_id="paper_demo")

        self.assertEqual(deleted, 2)
        self.assertEqual(
            self.index.deleted_paper,
            {"user_id": "user_a", "session_id": "session_a", "paper_id": "paper_demo"},
        )


if __name__ == "__main__":
    unittest.main()
