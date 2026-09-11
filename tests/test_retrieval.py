"""Dense 检索的范围隔离、父子回溯与追踪记录测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.index import IndexHit  # noqa: E402
from researchpilot.retrieval import (  # noqa: E402
    DenseRetriever,
    HybridRetriever,
    LangChainQueryExpander,
    RetrievalService,
)
from researchpilot.schemas import BlockType, Chunk, QueryType  # noqa: E402


def _chunk(*, content: str, parent_id: str | None, page: int) -> Chunk:
    return Chunk.create(
        paper_id="paper_demo",
        parent_id=parent_id,
        page_start=page,
        page_end=page,
        section_path=("3 方法",),
        content=content,
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test_v1",
    )


class _FakeIndexRepository:
    """不连接 Chroma 的内存索引替身。"""

    def __init__(self, child_hits: tuple[IndexHit, ...], parents: dict[str, IndexHit]) -> None:
        self.child_hits = child_hits
        self.parents = parents
        self.query_calls: list[tuple[str, str, str, int]] = []
        self.parent_calls: list[tuple[tuple[str, ...], str, str]] = []

    def query_children(self, *, query: str, user_id: str, session_id: str, limit: int) -> tuple[IndexHit, ...]:
        self.query_calls.append((query, user_id, session_id, limit))
        return self.child_hits[:limit]

    def get_parent_chunks(
        self, *, parent_chunk_ids: tuple[str, ...], user_id: str, session_id: str
    ) -> dict[str, IndexHit]:
        self.parent_calls.append((parent_chunk_ids, user_id, session_id))
        return {chunk_id: self.parents[chunk_id] for chunk_id in parent_chunk_ids if chunk_id in self.parents}

    def upsert(self, **kwargs: object) -> None:
        raise AssertionError("此测试不应写入索引。")

    def delete_session(self, **kwargs: object) -> int:
        raise AssertionError("此测试不应删除索引。")


class _WhitespaceTokenizer:
    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(text.lower().split())


class _FixedQueryExpander:
    def expand(self, *, query: str, recent_history: str) -> tuple[str, ...]:
        return ("attention method", "evaluation metric")


class _FixedReranker:
    def score(self, *, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(9.0 if "evaluation" in document else 3.0 for document in documents)


class _HybridFakeIndex:
    def __init__(self) -> None:
        self.parent_method = _chunk(content="attention method parent context", parent_id="block_method", page=2)
        self.parent_evaluation = _chunk(
            content="evaluation metric parent context", parent_id="block_evaluation", page=4
        )
        child_method = _chunk(content="attention method details", parent_id=self.parent_method.chunk_id, page=2)
        child_evaluation = _chunk(
            content="evaluation metric MRR details", parent_id=self.parent_evaluation.chunk_id, page=4
        )
        self.children = (
            IndexHit(chunk=child_method, source_name="method.pdf", score=0.0),
            IndexHit(chunk=child_evaluation, source_name="evaluation.pdf", score=0.0),
        )
        self.parents = {
            self.parent_method.chunk_id: IndexHit(chunk=self.parent_method, source_name="method.pdf", score=0.0),
            self.parent_evaluation.chunk_id: IndexHit(
                chunk=self.parent_evaluation, source_name="evaluation.pdf", score=0.0
            ),
        }
        self.calls: list[tuple[str, str, str]] = []

    def query_children(self, *, query: str, user_id: str, session_id: str, limit: int) -> tuple[IndexHit, ...]:
        self.calls.append((query, user_id, session_id))
        if "attention" in query:
            return (IndexHit(chunk=self.children[0].chunk, source_name="method.pdf", score=0.96),)
        if "evaluation" in query:
            return (IndexHit(chunk=self.children[1].chunk, source_name="evaluation.pdf", score=0.88),)
        return ()

    def list_children(self, *, user_id: str, session_id: str) -> tuple[IndexHit, ...]:
        self.calls.append(("__list__", user_id, session_id))
        return self.children

    def get_parent_chunks(
        self, *, parent_chunk_ids: tuple[str, ...], user_id: str, session_id: str
    ) -> dict[str, IndexHit]:
        self.calls.append(("__parents__", user_id, session_id))
        return {chunk_id: self.parents[chunk_id] for chunk_id in parent_chunk_ids if chunk_id in self.parents}

    def upsert(self, **kwargs: object) -> None:
        raise AssertionError("此测试不应写入索引。")

    def delete_session(self, **kwargs: object) -> int:
        raise AssertionError("此测试不应删除索引。")


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = _chunk(
            content="父块包含完整的方法、数据集和实验设置。",
            parent_id="block_paper_demo_p0002",
            page=2,
        )
        first_child = _chunk(
            content="方法使用页码级证据。",
            parent_id=self.parent.chunk_id,
            page=2,
        )
        second_child = _chunk(
            content="实验报告 Recall 和 MRR。",
            parent_id=self.parent.chunk_id,
            page=2,
        )
        self.fake_index = _FakeIndexRepository(
            child_hits=(
                IndexHit(chunk=first_child, source_name="demo.pdf", score=0.93),
                IndexHit(chunk=second_child, source_name="demo.pdf", score=0.81),
            ),
            parents={
                self.parent.chunk_id: IndexHit(
                    chunk=self.parent,
                    source_name="demo.pdf",
                    score=0.0,
                )
            },
        )
        self.service = RetrievalService(DenseRetriever(self.fake_index, candidate_limit=3))

    def test_retrieval_uses_final_query_and_returns_parent_context_with_evidence(self) -> None:
        result = self.service.retrieve(
            query="它的指标是什么？",
            search_query="这篇论文的实验评测指标是什么？",
            user_id="user_a",
            session_id="session_a",
            query_type=QueryType.LOCAL_FACT,
        )

        self.assertEqual(
            self.fake_index.query_calls,
            [("这篇论文的实验评测指标是什么？", "user_a", "session_a", 3)],
        )
        self.assertEqual(self.fake_index.parent_calls, [((self.parent.chunk_id,), "user_a", "session_a")])
        self.assertEqual(len(result.results), 1)
        retrieved = result.results[0]
        self.assertEqual(retrieved.chunk.content, self.parent.content)
        self.assertEqual(retrieved.source_name, "demo.pdf")
        self.assertEqual(retrieved.chunk.page_start, 2)
        self.assertEqual(retrieved.dense_rank, 1)
        self.assertEqual(retrieved.dense_score, 0.93)
        self.assertEqual(result.trace.search_query, "这篇论文的实验评测指标是什么？")
        self.assertEqual(result.trace.query_type, QueryType.LOCAL_FACT)
        self.assertEqual(result.trace.candidate_count, 2)
        self.assertGreaterEqual(result.trace.elapsed_ms, 0)

    def test_missing_parent_is_not_returned_as_fabricated_context(self) -> None:
        self.fake_index.parents.clear()
        result = self.service.retrieve(query="核心方法是什么？", user_id="user_a", session_id="session_a")

        self.assertEqual(result.results, ())
        self.assertEqual(result.trace.candidate_count, 2)

    def test_empty_question_is_rejected_before_index_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "query"):
            self.service.retrieve(query="  ", user_id="user_a", session_id="session_a")
        self.assertEqual(self.fake_index.query_calls, [])

    def test_hybrid_retriever_preserves_expansion_dense_bm25_parent_and_rerank(self) -> None:
        index = _HybridFakeIndex()
        service = RetrievalService(
            HybridRetriever(
                index,
                query_expander=_FixedQueryExpander(),
                tokenizer=_WhitespaceTokenizer(),
                reranker=_FixedReranker(),
                dense_limit_per_query=3,
                bm25_limit_per_query=3,
                result_limit=3,
            )
        )

        result = service.retrieve(
            query="它的实验指标是什么？",
            search_query="这篇论文的方法和实验指标是什么？",
            recent_history="用户：请解释 attention 方法。",
            user_id="user_a",
            session_id="session_a",
        )

        self.assertEqual(
            result.trace.expanded_queries,
            ("attention method", "evaluation metric", "这篇论文的方法和实验指标是什么？"),
        )
        self.assertEqual(result.trace.candidate_count, 2)
        self.assertEqual(
            result.trace.stages,
            (
                "query_expansion",
                "dense_child_retrieval",
                "bm25_child_retrieval",
                "parent_hydration",
                "rerank",
            ),
        )
        self.assertEqual(result.results[0].source_name, "evaluation.pdf")
        self.assertEqual(result.results[0].chunk.page_start, 4)
        self.assertEqual(result.results[0].rerank_score, 9.0)
        self.assertIsNotNone(result.results[0].sparse_score)
        self.assertEqual(result.results[1].dense_score, 0.96)
        self.assertTrue(
            all(user_id == "user_a" and session_id == "session_a" for _, user_id, session_id in index.calls)
        )

    def test_langchain_query_expander_parses_plain_lines_without_explanations(self) -> None:
        class _Response:
            content = "1. first independent query\n- second independent query\n\n第三个问题"

        class _Model:
            def invoke(self, messages: object) -> _Response:
                return _Response()

        expanded = LangChainQueryExpander(_Model()).expand(query="它是什么？", recent_history="用户正在讨论论文方法。")
        self.assertEqual(expanded, ("first independent query", "second independent query", "第三个问题"))


if __name__ == "__main__":
    unittest.main()
