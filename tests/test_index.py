"""Chroma 索引封装测试；底层使用内存替身，不连接真实 Chroma。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_core.documents import Document  # noqa: E402

from researchpilot.index import (  # noqa: E402
    ChromaVectorIndexRepository,
    ChunkKind,
)
from researchpilot.schemas import BlockType, Chunk  # noqa: E402


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


class _FakeChroma:
    def __init__(self) -> None:
        self.added_documents: list[Document] = []
        self.added_ids: list[str] = []
        self.query_calls: list[tuple[str, int, dict[str, object]]] = []
        self.get_calls: list[dict[str, object]] = []
        self.deleted_ids: list[str] = []
        self.query_result: list[tuple[Document, float]] = []
        self.get_result: dict[str, object] = {"ids": []}

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self.added_documents = documents
        self.added_ids = ids

    def similarity_search_with_relevance_scores(
        self, query: str, *, k: int, filter: dict[str, object]
    ) -> list[tuple[Document, float]]:
        self.query_calls.append((query, k, filter))
        return self.query_result

    def get(self, **kwargs: object) -> dict[str, object]:
        self.get_calls.append(kwargs)
        return self.get_result

    def delete(self, *, ids: list[str]) -> None:
        self.deleted_ids.extend(ids)


class IndexRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_chroma = _FakeChroma()
        self.repository = ChromaVectorIndexRepository(self.fake_chroma)  # type: ignore[arg-type]
        self.parent = _chunk(
            content="完整父块内容，包含方法与实验细节。",
            parent_id="block_paper_demo_p0002",
            page=2,
        )
        self.child = _chunk(
            content="方法使用页码级证据。",
            parent_id=self.parent.chunk_id,
            page=2,
        )

    def test_upsert_uses_scope_and_stable_record_ids(self) -> None:
        self.repository.upsert(
            chunks=(self.child,),
            chunk_kind=ChunkKind.CHILD,
            source_name="demo.pdf",
            user_id="user_a",
            session_id="session_a",
        )

        self.assertEqual(len(self.fake_chroma.added_documents), 1)
        self.assertEqual(
            self.fake_chroma.added_ids,
            [f"rp:user_a:session_a:child_chunk:{self.child.chunk_id}"],
        )
        metadata = self.fake_chroma.added_documents[0].metadata
        self.assertEqual(metadata["user_id"], "user_a")
        self.assertEqual(metadata["session_id"], "session_a")
        self.assertEqual(metadata["chunk_kind"], "child_chunk")
        self.assertEqual(metadata["paper_id"], "paper_demo")
        self.assertEqual(metadata["page_start"], 2)

    def test_query_and_parent_hydration_keep_scope(self) -> None:
        child_metadata = self.child.index_metadata() | {
            "chunk_kind": "child_chunk",
            "source_name": "demo.pdf",
            "user_id": "user_a",
            "session_id": "session_a",
        }
        self.fake_chroma.query_result = [(Document(page_content=self.child.content, metadata=child_metadata), 0.88)]
        hits = self.repository.query_children(query="方法", user_id="user_a", session_id="session_a", limit=3)

        self.assertEqual(hits[0].chunk.chunk_id, self.child.chunk_id)
        self.assertEqual(hits[0].source_name, "demo.pdf")
        _, _, scope_filter = self.fake_chroma.query_calls[0]
        self.assertIn({"user_id": "user_a"}, scope_filter["$and"])
        self.assertIn({"session_id": "session_a"}, scope_filter["$and"])
        self.assertIn({"chunk_kind": "child_chunk"}, scope_filter["$and"])

        parent_metadata = self.parent.index_metadata() | {
            "chunk_kind": "parent_chunk",
            "source_name": "demo.pdf",
            "user_id": "user_a",
            "session_id": "session_a",
        }
        self.fake_chroma.get_result = {
            "documents": [self.parent.content],
            "metadatas": [parent_metadata],
        }
        parents = self.repository.get_parent_chunks(
            parent_chunk_ids=(self.parent.chunk_id,), user_id="user_a", session_id="session_a"
        )
        self.assertEqual(parents[self.parent.chunk_id].chunk.page_start, 2)
        self.assertEqual(parents[self.parent.chunk_id].source_name, "demo.pdf")

    def test_list_children_uses_the_same_user_and_session_scope(self) -> None:
        child_metadata = self.child.index_metadata() | {
            "chunk_kind": "child_chunk",
            "source_name": "demo.pdf",
            "user_id": "user_a",
            "session_id": "session_a",
        }
        self.fake_chroma.get_result = {
            "documents": [self.child.content],
            "metadatas": [child_metadata],
        }
        children = self.repository.list_children(user_id="user_a", session_id="session_a")

        self.assertEqual(children[0].chunk.chunk_id, self.child.chunk_id)
        where = self.fake_chroma.get_calls[0]["where"]
        self.assertIn({"chunk_kind": "child_chunk"}, where["$and"])

    def test_delete_session_deletes_only_ids_selected_by_scoped_get(self) -> None:
        self.fake_chroma.get_result = {"ids": ["rp:user_a:session_a:child_chunk:chunk_1"]}
        deleted_count = self.repository.delete_session(user_id="user_a", session_id="session_a")

        self.assertEqual(deleted_count, 1)
        self.assertEqual(self.fake_chroma.deleted_ids, ["rp:user_a:session_a:child_chunk:chunk_1"])
        where = self.fake_chroma.get_calls[0]["where"]
        self.assertEqual(
            where,
            {"$and": [{"user_id": "user_a"}, {"session_id": "session_a"}]},
        )

    def test_delete_paper_keeps_the_same_user_session_and_paper_boundary(self) -> None:
        self.fake_chroma.get_result = {
            "ids": [
                "rp:user_a:session_a:parent_chunk:chunk_parent",
                "rp:user_a:session_a:child_chunk:chunk_child",
            ]
        }

        deleted_count = self.repository.delete_paper(user_id="user_a", session_id="session_a", paper_id="paper_demo")

        self.assertEqual(deleted_count, 2)
        self.assertEqual(len(self.fake_chroma.deleted_ids), 2)
        where = self.fake_chroma.get_calls[0]["where"]
        self.assertIn({"user_id": "user_a"}, where["$and"])
        self.assertIn({"session_id": "session_a"}, where["$and"])
        self.assertIn({"paper_id": "paper_demo"}, where["$and"])

    def test_invalid_scope_is_rejected_before_chroma_call(self) -> None:
        with self.assertRaisesRegex(Exception, "user_id"):
            self.repository.query_children(query="方法", user_id="../user_b", session_id="session_a", limit=3)
        self.assertEqual(self.fake_chroma.query_calls, [])


if __name__ == "__main__":
    unittest.main()
