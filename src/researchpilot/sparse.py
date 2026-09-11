"""持久化 BM25 稀疏索引。

索引按 ``user_id + session_id`` 隔离，并以子 Chunk 的 ID、内容哈希和分词器版本
生成指纹。索引首次使用时构建或从磁盘加载；同一进程的后续查询命中内存缓存，不会
重新分词或重建 BM25。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import bm25s

from .config import validate_storage_id
from .index import IndexHit, VectorIndexRepository

SPARSE_INDEX_SCHEMA_VERSION = "researchpilot_bm25_v1"
_WINDOWS_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> tuple[str, ...]:
        """将文本转换为稳定的 token 序列。"""


@dataclass(frozen=True)
class SparseIndexManifest:
    schema_version: str
    user_id: str
    session_id: str
    corpus_fingerprint: str
    chunk_count: int
    tokenizer_name: str
    created_at: str

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> SparseIndexManifest:
        try:
            return cls(
                schema_version=str(raw["schema_version"]),
                user_id=str(raw["user_id"]),
                session_id=str(raw["session_id"]),
                corpus_fingerprint=str(raw["corpus_fingerprint"]),
                chunk_count=int(raw["chunk_count"]),
                tokenizer_name=str(raw["tokenizer_name"]),
                created_at=str(raw["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("BM25 Manifest 格式无效。") from error


@dataclass(frozen=True)
class SparseHit:
    """一条子 Chunk 的 BM25 命中。"""

    hit: IndexHit
    score: float


@dataclass(frozen=True)
class _LoadedScope:
    bm25: object
    children: tuple[IndexHit, ...]
    manifest: SparseIndexManifest


def _scope_key(user_id: str, session_id: str) -> tuple[str, str]:
    validate_storage_id(user_id, "user_id")
    validate_storage_id(session_id, "session_id")
    return user_id, session_id


def _fingerprint(children: Sequence[IndexHit], tokenizer_name: str) -> str:
    payload = {
        "tokenizer_name": tokenizer_name,
        "children": [
            {
                "chunk_id": item.chunk.chunk_id,
                "paper_id": item.chunk.paper_id,
                "parent_id": item.chunk.parent_id,
                "content_hash": item.chunk.content_hash,
            }
            for item in children
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class PersistentBM25Repository:
    """从向量库的子 Chunk 构建并持久化 BM25S 索引。"""

    def __init__(
        self,
        index: VectorIndexRepository,
        *,
        root_dir: Path,
        tokenizer: Tokenizer,
    ) -> None:
        self._index = index
        self._root_dir = root_dir.resolve()
        self._tokenizer = tokenizer
        self._tokenizer_name = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        self._cache: dict[tuple[str, str], _LoadedScope] = {}

    def invalidate_scope(self, *, user_id: str, session_id: str) -> None:
        """在上传、删除或重建文档后使当前进程缓存失效。"""
        self._cache.pop(_scope_key(user_id, session_id), None)

    def delete_scope(self, *, user_id: str, session_id: str) -> None:
        """删除指定 Scope 的磁盘索引和内存缓存。"""
        key = _scope_key(user_id, session_id)
        self._cache.pop(key, None)
        target = self._scope_dir(*key)
        if target.exists():
            target.relative_to(self._root_dir)
            shutil.rmtree(target)

    def build(self, *, user_id: str, session_id: str, force: bool = False) -> SparseIndexManifest:
        """显式构建或复用索引，供 CLI 与文档上传后的预热调用。"""
        loaded = self._load_or_build(user_id=user_id, session_id=session_id, force=force)
        return loaded.manifest

    def query(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str,
        limit: int,
        paper_id: str | None = None,
    ) -> tuple[SparseHit, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空。")
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit 必须是正整数。")
        if paper_id is not None and (not isinstance(paper_id, str) or not paper_id.strip()):
            raise ValueError("paper_id 必须是非空字符串或 None。")
        loaded = self._load_or_build(user_id=user_id, session_id=session_id, force=False)
        tokens = list(self._tokenizer.tokenize(query))
        if not tokens or not loaded.children:
            return ()
        documents, scores = loaded.bm25.retrieve(
            [tokens],
            corpus=list(range(len(loaded.children))),
            # BM25 以会话 Scope 持久化。需要论文级隔离时取完整排序后再
            # 过滤，避免全局 Top-K 被另一篇论文占满；不重建索引。
            k=min(len(loaded.children) if paper_id is not None else limit, len(loaded.children)),
            show_progress=False,
        )
        hits = tuple(
            SparseHit(hit=loaded.children[int(index)], score=float(score))
            for index, score in zip(documents[0], scores[0], strict=True)
        )
        if paper_id is not None:
            hits = tuple(item for item in hits if item.hit.chunk.paper_id == paper_id)
        return hits[:limit]

    def index_version(self, *, user_id: str, session_id: str) -> str | None:
        loaded = self._cache.get(_scope_key(user_id, session_id))
        return loaded.manifest.corpus_fingerprint[:16] if loaded else None

    def _scope_dir(self, user_id: str, session_id: str) -> Path:
        key = _scope_key(user_id, session_id)
        return self._root_dir / key[0] / key[1]

    def _load_or_build(self, *, user_id: str, session_id: str, force: bool) -> _LoadedScope:
        key = _scope_key(user_id, session_id)
        if not force and key in self._cache:
            return self._cache[key]
        children = tuple(
            sorted(
                self._index.list_children(user_id=user_id, session_id=session_id),
                key=lambda item: item.chunk.chunk_id,
            )
        )
        fingerprint = _fingerprint(children, self._tokenizer_name)
        target = self._scope_dir(*key)
        manifest = self._read_manifest(target)
        if (
            not force
            and manifest is not None
            and manifest.schema_version == SPARSE_INDEX_SCHEMA_VERSION
            and manifest.user_id == user_id
            and manifest.session_id == session_id
            and manifest.corpus_fingerprint == fingerprint
            and manifest.chunk_count == len(children)
            and manifest.tokenizer_name == self._tokenizer_name
        ):
            loaded = _LoadedScope(
                bm25=bm25s.BM25.load(target / "bm25", load_corpus=False),
                children=children,
                manifest=manifest,
            )
        else:
            loaded = self._build_scope(
                user_id=user_id,
                session_id=session_id,
                children=children,
                fingerprint=fingerprint,
                target=target,
            )
        self._cache[key] = loaded
        return loaded

    def _build_scope(
        self,
        *,
        user_id: str,
        session_id: str,
        children: tuple[IndexHit, ...],
        fingerprint: str,
        target: Path,
    ) -> _LoadedScope:
        corpus = [list(self._tokenizer.tokenize(item.chunk.content)) for item in children]
        bm25 = bm25s.BM25()
        bm25.index(corpus, show_progress=False)
        manifest = SparseIndexManifest(
            schema_version=SPARSE_INDEX_SCHEMA_VERSION,
            user_id=user_id,
            session_id=session_id,
            corpus_fingerprint=fingerprint,
            chunk_count=len(children),
            tokenizer_name=self._tokenizer_name,
            created_at=datetime.now(UTC).isoformat(),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
        try:
            (temporary / "bm25").mkdir(parents=True, exist_ok=False)
            # Chunk 映射由我们自己的 Manifest 与同序 children 维护；不把整数 ID
            # 交给 BM25S 的可选 corpus 序列化器，以避免其误判为无效文档。
            bm25.save(temporary / "bm25")
            (temporary / "manifest.json").write_text(
                json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if target.exists():
                target.relative_to(self._root_dir)
                shutil.rmtree(target)
            self._replace_directory_with_retry(temporary, target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise
        return _LoadedScope(bm25=bm25, children=children, manifest=manifest)

    @staticmethod
    def _replace_directory_with_retry(temporary: Path, target: Path) -> None:
        """Publish a completed index despite short-lived Windows file locks.

        Defender/indexing software can briefly hold a freshly-created BM25S
        file after ``save``.  A one-shot directory rename then raises
        ``PermissionError`` even though neither path is invalid.  Retrying the
        exact, already-validated scope preserves the atomic publication model
        without deleting any broader directory or rebuilding the index.
        """
        for delay in (*_WINDOWS_REPLACE_RETRY_DELAYS, None):
            try:
                temporary.replace(target)
                return
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)

    @staticmethod
    def _read_manifest(target: Path) -> SparseIndexManifest | None:
        path = target / "manifest.json"
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return SparseIndexManifest.from_dict(raw)
