"""向量索引的边界接口与 Chroma 实现。

上层检索只依赖 ``VectorIndexRepository``，因此单元测试可以使用内存假实现，未来
迁移其他向量数据库时也不必触及检索和生成逻辑。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import validate_storage_id
from .schemas import BlockType, Chunk, SchemaValidationError


class ChunkKind(str, Enum):
    """索引中父块与子块的角色。"""

    # 与冻结版的 metadata type 保持一致，便于逐步替换旧页面逻辑。
    PARENT = "parent_chunk"
    CHILD = "child_chunk"


@dataclass(frozen=True)
class IndexHit:
    """向量索引返回的一条候选及其相似度。"""

    chunk: Chunk
    source_name: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, Chunk):
            raise SchemaValidationError("IndexHit.chunk 必须是 Chunk。")
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise SchemaValidationError("IndexHit.source_name 不能为空。")
        if not isinstance(self.score, int | float) or not math.isfinite(self.score):
            raise SchemaValidationError("IndexHit.score 必须是有限数值。")


class VectorIndexRepository(Protocol):
    """检索核心依赖的最小向量索引契约。"""

    def upsert(
        self,
        *,
        chunks: Sequence[Chunk],
        chunk_kind: ChunkKind,
        source_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        """写入或覆盖同一用户、会话、角色和 Chunk ID 的记录。"""

    def query_children(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str,
        limit: int,
        paper_id: str | None = None,
    ) -> tuple[IndexHit, ...]:
        """只查询当前用户、当前会话的子 Chunk。"""

    def get_parent_chunks(
        self,
        *,
        parent_chunk_ids: Sequence[str],
        user_id: str,
        session_id: str,
        paper_id: str | None = None,
    ) -> dict[str, IndexHit]:
        """按父 Chunk ID 回取当前用户、当前会话的父块。"""

    def list_children(self, *, user_id: str, session_id: str, paper_id: str | None = None) -> tuple[IndexHit, ...]:
        """列出当前用户、当前会话的全部子 Chunk，供 BM25 建立临时语料。"""

    def delete_session(self, *, user_id: str, session_id: str) -> int:
        """删除一个会话在该索引中的全部记录，并返回删除条数。"""

    def delete_paper(self, *, user_id: str, session_id: str, paper_id: str) -> int:
        """删除一个会话中某篇论文的父、子块，且不影响其他论文。"""


def _validate_scope(user_id: str, session_id: str) -> None:
    validate_storage_id(user_id, "user_id")
    validate_storage_id(session_id, "session_id")


def _validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空。")
    return query.strip()


def _vector_record_id(*, user_id: str, session_id: str, chunk_kind: ChunkKind, chunk_id: str) -> str:
    return f"rp:{user_id}:{session_id}:{chunk_kind.value}:{chunk_id}"


def _metadata_to_chunk(metadata: dict[str, object], content: str) -> Chunk:
    """从 Chroma 元数据恢复领域 Chunk；缺字段时拒绝返回不可信结果。"""
    try:
        section_path_raw = str(metadata.get("section_path", ""))
        block_types_raw = str(metadata["block_types"])
        block_types = tuple(BlockType(value) for value in block_types_raw.split(",") if value.strip())
        return Chunk(
            chunk_id=str(metadata["chunk_id"]),
            paper_id=str(metadata["paper_id"]),
            parent_id=str(metadata["parent_id"]) or None,
            page_start=int(metadata["page_start"]),
            page_end=int(metadata["page_end"]),
            section_path=tuple(item.strip() for item in section_path_raw.split(" > ") if item.strip()),
            content=content,
            block_types=block_types,
            content_hash=str(metadata["content_hash"]),
            chunker_version=str(metadata["chunker_version"]),
        )
    except (KeyError, TypeError, ValueError, SchemaValidationError) as exc:
        raise RuntimeError("索引记录元数据不完整或不合法，无法恢复 Chunk。") from exc


class ChromaVectorIndexRepository:
    """Chroma 的范围隔离封装，不向业务层暴露 Chroma API。"""

    def __init__(self, vectorstore: Chroma) -> None:
        self._vectorstore = vectorstore

    @staticmethod
    def _scope_filter(*, user_id: str, session_id: str) -> dict[str, object]:
        return {
            "$and": [
                {"user_id": user_id},
                {"session_id": session_id},
            ]
        }

    @staticmethod
    def _add_paper_filter(scope_filter: dict[str, object], paper_id: str | None) -> None:
        if paper_id is None:
            return
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError("paper_id 必须是非空字符串或 None。")
        clauses = scope_filter["$and"]
        assert isinstance(clauses, list)
        clauses.append({"paper_id": paper_id.strip()})

    def upsert(
        self,
        *,
        chunks: Sequence[Chunk],
        chunk_kind: ChunkKind,
        source_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        _validate_scope(user_id, session_id)
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError("source_name 不能为空。")
        if not chunks:
            return
        documents: list[Document] = []
        ids: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise TypeError("chunks 必须全部是 Chunk 对象。")
            metadata = chunk.index_metadata() | {
                "chunk_id": chunk.chunk_id,
                "chunk_kind": chunk_kind.value,
                "type": chunk_kind.value,
                "source_name": source_name.strip(),
                "source": source_name.strip(),
                "user_id": user_id,
                "session_id": session_id,
            }
            documents.append(Document(page_content=chunk.content, metadata=metadata))
            ids.append(
                _vector_record_id(
                    user_id=user_id,
                    session_id=session_id,
                    chunk_kind=chunk_kind,
                    chunk_id=chunk.chunk_id,
                )
            )
        self._vectorstore.add_documents(documents, ids=ids)

    def query_children(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str,
        limit: int,
        paper_id: str | None = None,
    ) -> tuple[IndexHit, ...]:
        normalized_query = _validate_query(query)
        _validate_scope(user_id, session_id)
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit 必须是正整数。")
        scope_filter = self._scope_filter(user_id=user_id, session_id=session_id)
        self._add_paper_filter(scope_filter, paper_id)
        clauses = scope_filter["$and"]
        assert isinstance(clauses, list)
        clauses.append({"chunk_kind": ChunkKind.CHILD.value})
        documents_and_scores = self._vectorstore.similarity_search_with_relevance_scores(
            normalized_query,
            k=limit,
            filter=scope_filter,
        )
        return tuple(
            IndexHit(
                chunk=_metadata_to_chunk(document.metadata, document.page_content),
                source_name=str(document.metadata["source_name"]),
                score=float(score),
            )
            for document, score in documents_and_scores
        )

    def get_parent_chunks(
        self,
        *,
        parent_chunk_ids: Sequence[str],
        user_id: str,
        session_id: str,
        paper_id: str | None = None,
    ) -> dict[str, IndexHit]:
        _validate_scope(user_id, session_id)
        if paper_id is not None and (not isinstance(paper_id, str) or not paper_id.strip()):
            raise ValueError("paper_id 必须是非空字符串或 None。")
        if not parent_chunk_ids:
            return {}
        unique_parent_ids = tuple(dict.fromkeys(parent_chunk_ids))
        vector_ids = [
            _vector_record_id(
                user_id=user_id,
                session_id=session_id,
                chunk_kind=ChunkKind.PARENT,
                chunk_id=parent_chunk_id,
            )
            for parent_chunk_id in unique_parent_ids
        ]
        records = self._vectorstore.get(ids=vector_ids, include=["documents", "metadatas"])
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        result: dict[str, IndexHit] = {}
        for content, metadata in zip(documents, metadatas, strict=True):
            if not isinstance(content, str) or not isinstance(metadata, dict):
                continue
            if metadata.get("user_id") != user_id or metadata.get("session_id") != session_id:
                raise RuntimeError("索引返回了越过用户或会话边界的记录。")
            if paper_id is not None and metadata.get("paper_id") != paper_id:
                raise RuntimeError("索引返回了越过论文范围边界的记录。")
            chunk = _metadata_to_chunk(metadata, content)
            result[chunk.chunk_id] = IndexHit(
                chunk=chunk,
                source_name=str(metadata["source_name"]),
                score=0.0,
            )
        return result

    def list_children(self, *, user_id: str, session_id: str, paper_id: str | None = None) -> tuple[IndexHit, ...]:
        _validate_scope(user_id, session_id)
        scope_filter = self._scope_filter(user_id=user_id, session_id=session_id)
        self._add_paper_filter(scope_filter, paper_id)
        clauses = scope_filter["$and"]
        assert isinstance(clauses, list)
        clauses.append({"chunk_kind": ChunkKind.CHILD.value})
        records = self._vectorstore.get(
            where=scope_filter,
            include=["documents", "metadatas"],
        )
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        result: list[IndexHit] = []
        for content, metadata in zip(documents, metadatas, strict=True):
            if not isinstance(content, str) or not isinstance(metadata, dict):
                continue
            if metadata.get("user_id") != user_id or metadata.get("session_id") != session_id:
                raise RuntimeError("索引返回了越过用户或会话边界的记录。")
            if paper_id is not None and metadata.get("paper_id") != paper_id:
                raise RuntimeError("索引返回了越过论文范围边界的记录。")
            result.append(
                IndexHit(
                    chunk=_metadata_to_chunk(metadata, content),
                    source_name=str(metadata["source_name"]),
                    score=0.0,
                )
            )
        return tuple(result)

    def delete_session(self, *, user_id: str, session_id: str) -> int:
        _validate_scope(user_id, session_id)
        records = self._vectorstore.get(
            where=self._scope_filter(user_id=user_id, session_id=session_id),
        )
        ids = [record_id for record_id in records.get("ids") or [] if isinstance(record_id, str)]
        if ids:
            self._vectorstore.delete(ids=ids)
        return len(ids)

    def delete_paper(self, *, user_id: str, session_id: str, paper_id: str) -> int:
        _validate_scope(user_id, session_id)
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError("paper_id must be a non-empty string.")
        scope_filter = self._scope_filter(user_id=user_id, session_id=session_id)
        self._add_paper_filter(scope_filter, paper_id)
        records = self._vectorstore.get(where=scope_filter)
        ids = [record_id for record_id in records.get("ids") or [] if isinstance(record_id, str)]
        if ids:
            self._vectorstore.delete(ids=ids)
        return len(ids)
