"""持久化 BM25 与 RRF 的单元测试，不连接真实 Chroma 或模型。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.index import IndexHit  # noqa: E402
from researchpilot.retrieval import PersistentHybridRetriever, RetrievalService  # noqa: E402
from researchpilot.schemas import BlockType, Chunk  # noqa: E402
from researchpilot.sparse import PersistentBM25Repository  # noqa: E402


def _chunk(*, content: str, parent_id: str | None, page: int) -> Chunk:
    return Chunk.create(
        paper_id="paper_demo",
        parent_id=parent_id,
        page_start=page,
        page_end=page,
        section_path=("Method",),
        content=content,
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test_v1",
    )


class _WhitespaceTokenizer:
    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(text.lower().split())


class _Index:
    def __init__(self) -> None:
        self.parent_dense = _chunk(content="dense parent", parent_id="block_a", page=1)
        self.parent_sparse = _chunk(content="sparse parent", parent_id="block_b", page=2)
        self.child_dense = _chunk(content="attention retrieval dense", parent_id=self.parent_dense.chunk_id, page=1)
        self.child_sparse = _chunk(content="keyword bm25 sparse", parent_id=self.parent_sparse.chunk_id, page=2)
        self.children = (
            IndexHit(self.child_dense, "dense.pdf", 0.0),
            IndexHit(self.child_sparse, "sparse.pdf", 0.0),
        )
        self.parents = {
            self.parent_dense.chunk_id: IndexHit(self.parent_dense, "dense.pdf", 0.0),
            self.parent_sparse.chunk_id: IndexHit(self.parent_sparse, "sparse.pdf", 0.0),
        }
        self.list_calls = 0

    def list_children(self, *, user_id: str, session_id: str) -> tuple[IndexHit, ...]:
        self.list_calls += 1
        return self.children

    def query_children(self, *, query: str, user_id: str, session_id: str, limit: int) -> tuple[IndexHit, ...]:
        return (IndexHit(self.child_dense, "dense.pdf", 0.9),)[:limit]

    def get_parent_chunks(
        self, *, parent_chunk_ids: tuple[str, ...], user_id: str, session_id: str
    ) -> dict[str, IndexHit]:
        return {item: self.parents[item] for item in parent_chunk_ids if item in self.parents}


class PersistentSparseTests(unittest.TestCase):
    def test_windows_permission_error_during_publish_is_retried(self) -> None:
        index = _Index()
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = PersistentBM25Repository(index, root_dir=Path(temp_dir), tokenizer=_WhitespaceTokenizer())
            original_replace = Path.replace
            attempts = 0

            def flaky_replace(source: Path, target: Path) -> Path:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("simulated short-lived Windows lock")
                return original_replace(source, target)

            with (
                patch.object(Path, "replace", autospec=True, side_effect=flaky_replace),
                patch("researchpilot.sparse.time.sleep"),
            ):
                manifest = repository.build(user_id="user_a", session_id="session_a")

            self.assertEqual(attempts, 2)
            self.assertEqual(manifest.chunk_count, 2)
            self.assertTrue((Path(temp_dir) / "user_a" / "session_a" / "manifest.json").is_file())

    def test_bm25_is_persisted_and_second_query_uses_memory_cache(self) -> None:
        index = _Index()
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = PersistentBM25Repository(index, root_dir=Path(temp_dir), tokenizer=_WhitespaceTokenizer())
            first = repository.query(query="bm25 keyword", user_id="user_a", session_id="session_a", limit=2)
            second = repository.query(query="bm25 keyword", user_id="user_a", session_id="session_a", limit=2)
            self.assertEqual(first[0].hit.chunk.chunk_id, index.child_sparse.chunk_id)
            self.assertEqual(second[0].hit.chunk.chunk_id, index.child_sparse.chunk_id)
            self.assertEqual(index.list_calls, 1)
            self.assertIsNotNone(repository.index_version(user_id="user_a", session_id="session_a"))
            self.assertTrue((Path(temp_dir) / "user_a" / "session_a" / "manifest.json").is_file())

            reloaded = PersistentBM25Repository(index, root_dir=Path(temp_dir), tokenizer=_WhitespaceTokenizer())
            loaded_hits = reloaded.query(query="bm25 keyword", user_id="user_a", session_id="session_a", limit=2)
            self.assertEqual(loaded_hits[0].hit.chunk.chunk_id, index.child_sparse.chunk_id)
            self.assertEqual(index.list_calls, 2)

    def test_rrf_fuses_child_ranks_before_parent_hydration(self) -> None:
        index = _Index()
        with tempfile.TemporaryDirectory() as temp_dir:
            sparse = PersistentBM25Repository(index, root_dir=Path(temp_dir), tokenizer=_WhitespaceTokenizer())
            service = RetrievalService(
                PersistentHybridRetriever(
                    index,
                    sparse,
                    fusion="rrf",
                    dense_limit_per_query=1,
                    bm25_limit_per_query=2,
                    result_limit=2,
                    rrf_k=60,
                )
            )
            result = service.retrieve(query="bm25 keyword", user_id="user_a", session_id="session_a")
            # dense 子块在 Dense 为第 1、BM25 为第 2，因此 RRF 累加后应超过
            # 仅由 BM25 第 1 命中的 sparse 子块。
            self.assertEqual(result.results[0].chunk.chunk_id, index.parent_dense.chunk_id)
            self.assertIsNotNone(result.results[0].fusion_score)
            self.assertIn("rrf_fusion", result.trace.stages)
            self.assertTrue(result.trace.stage_elapsed_ms)


if __name__ == "__main__":
    unittest.main()
