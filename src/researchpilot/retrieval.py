"""论文检索流程：Dense、BM25、父块补全与可选 Rerank。

页面不应了解向量库、BM25 或 Cross-Encoder 的调用细节。本模块以可替换的协议封装这些
依赖，并为每一次检索保留查询扩展、各路分数、候选数与耗时。
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import jieba
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi

from .index import IndexHit, VectorIndexRepository
from .schemas import QueryType, RetrievalTrace, ScoredChunk
from .sparse import PersistentBM25Repository


@dataclass(frozen=True)
class RetrievalResult:
    """可直接提供给生成层和评测层的检索结果。"""

    results: tuple[ScoredChunk, ...]
    trace: RetrievalTrace


@dataclass(frozen=True)
class _RetrievalExecution:
    """检索器内部结果，供 RetrievalService 统一封装为 Trace。"""

    results: tuple[ScoredChunk, ...]
    candidate_count: int
    expanded_queries: tuple[str, ...]
    stages: tuple[str, ...]
    stage_elapsed_ms: tuple[tuple[str, float], ...] = ()
    index_version: str | None = None


class QueryExpander(Protocol):
    """将带指代的论文问题扩展为若干可检索子问题。"""

    def expand(self, *, query: str, recent_history: str) -> tuple[str, ...]:
        """返回不带编号的、独立可检索的问题。"""


class Tokenizer(Protocol):
    """BM25 使用的可替换分词器。"""

    def tokenize(self, text: str) -> tuple[str, ...]:
        """将文本转换为 token 序列。"""


class Reranker(Protocol):
    """Cross-Encoder 等精排模型的最小契约。"""

    def score(self, *, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        """返回与 documents 等长的相关性得分。"""


class _Retriever(Protocol):
    def retrieve(
        self,
        *,
        search_query: str,
        user_id: str,
        session_id: str,
        recent_history: str = "",
        paper_id: str | None = None,
    ) -> _RetrievalExecution:
        """执行一次范围隔离的检索。"""


class JiebaTokenizer:
    """默认中文分词器；通过协议可在测试或英文场景中替换。"""

    def tokenize(self, text: str) -> tuple[str, ...]:
        if not isinstance(text, str) or not text.strip():
            return ()
        return tuple(token.strip() for token in jieba.lcut(text) if token.strip())


class LangChainQueryExpander:
    """用已有 LangChain ChatModel 复现旧版的指代消解与三路查询扩展。"""

    def __init__(self, chat_model: object, *, expansion_count: int = 3) -> None:
        if not isinstance(expansion_count, int) or expansion_count < 1:
            raise ValueError("expansion_count 必须是正整数。")
        self._chat_model = chat_model
        self._expansion_count = expansion_count

    def expand(self, *, query: str, recent_history: str) -> tuple[str, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空。")
        invoke = getattr(self._chat_model, "invoke", None)
        if not callable(invoke):
            raise TypeError("chat_model 必须提供 invoke 方法。")
        prompt = f"""你是专业的学术文献检索专家。请完成两步：
1. 阅读近期对话，消解当前问题中的代词或省略实体；
2. 将消解后的问题扩展为 {self._expansion_count} 个不同角度、实体完整、可独立检索的子问题。

直接逐行输出子问题；不要编号、解释或 Markdown。

【近期对话上下文】
{recent_history.strip() or "无近期对话。"}

【当前问题】
{query.strip()}"""
        response = invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            raise RuntimeError("查询扩展模型未返回文本。")
        return tuple(
            line.strip().lstrip("-•0123456789.、 ")
            for line in content.splitlines()
            if line.strip().lstrip("-•0123456789.、 ")
        )


class SentenceTransformerReranker:
    """将 sentence-transformers CrossEncoder 适配为 Reranker。"""

    def __init__(self, model: object) -> None:
        self._model = model

    def score(self, *, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        if not documents:
            return ()
        predict = getattr(self._model, "predict", None)
        if not callable(predict):
            raise TypeError("reranker model 必须提供 predict 方法。")
        # 评测会连续处理多个问题；显式批处理避免 CrossEncoder 逐条推理造成
        # 不必要的 Python 调度开销。自定义测试替身只需实现 score 协议，不受影响。
        scores = predict(
            [[query, document] for document in documents],
            # CrossEncoder 的注意力显存/内存随 batch_size 和序列长度平方增长；在无 GPU
            # 的本地评测环境中 8 能避免分页抖动，同时仍保留批量推理。
            batch_size=8,
            show_progress_bar=False,
        )
        resolved = tuple(float(score) for score in scores)
        if len(resolved) != len(documents) or any(not math.isfinite(score) for score in resolved):
            raise RuntimeError("Reranker 返回了无效得分。")
        return resolved


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} 必须是正整数。")


def _normalise_queries(expanded: Sequence[str], anchor_query: str) -> tuple[str, ...]:
    """去空、去重，并保证原始检索问题一定参与召回。"""
    resolved: list[str] = []
    for item in (*expanded, anchor_query):
        if isinstance(item, str) and item.strip() and item.strip() not in resolved:
            resolved.append(item.strip())
    return tuple(resolved)


class DenseRetriever:
    """最小 Dense 基线：子块向量召回，并回溯到完整父 Chunk。"""

    def __init__(self, index: VectorIndexRepository, *, candidate_limit: int = 3) -> None:
        _validate_positive_int(candidate_limit, "candidate_limit")
        self._index = index
        self._candidate_limit = candidate_limit

    def retrieve(
        self,
        *,
        search_query: str,
        user_id: str,
        session_id: str,
        recent_history: str = "",
        paper_id: str | None = None,
    ) -> _RetrievalExecution:
        child_arguments: dict[str, object] = {
            "query": search_query,
            "user_id": user_id,
            "session_id": session_id,
            "limit": self._candidate_limit,
        }
        if paper_id is not None:
            child_arguments["paper_id"] = paper_id
        child_hits = self._index.query_children(**child_arguments)  # type: ignore[arg-type]
        best_child_by_parent: dict[str, tuple[int, IndexHit]] = {}
        for dense_rank, hit in enumerate(child_hits, start=1):
            parent_id = hit.chunk.parent_id
            if parent_id and parent_id not in best_child_by_parent:
                best_child_by_parent[parent_id] = (dense_rank, hit)
        parent_arguments: dict[str, object] = {
            "parent_chunk_ids": tuple(best_child_by_parent),
            "user_id": user_id,
            "session_id": session_id,
        }
        if paper_id is not None:
            parent_arguments["paper_id"] = paper_id
        parent_hits = self._index.get_parent_chunks(**parent_arguments)  # type: ignore[arg-type]
        results: list[ScoredChunk] = []
        for parent_id, (dense_rank, child_hit) in best_child_by_parent.items():
            parent_hit = parent_hits.get(parent_id)
            if parent_hit is None:
                continue
            results.append(
                ScoredChunk(
                    chunk=parent_hit.chunk,
                    source_name=parent_hit.source_name,
                    dense_rank=dense_rank,
                    dense_score=child_hit.score,
                )
            )
        return _RetrievalExecution(
            results=tuple(results),
            # Dense 基线的候选数沿用“向量召回到的子块数”，便于与既有评测对比。
            candidate_count=len(child_hits),
            expanded_queries=(search_query,),
            stages=("dense_child_retrieval", "parent_hydration"),
        )


@dataclass
class _CandidateScores:
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None

    def observe_dense(self, *, rank: int, score: float) -> None:
        if self.dense_rank is None or rank < self.dense_rank:
            self.dense_rank = rank
            self.dense_score = score

    def observe_sparse(self, *, rank: int, score: float) -> None:
        if self.sparse_rank is None or rank < self.sparse_rank:
            self.sparse_rank = rank
            self.sparse_score = score


class HybridRetriever:
    """复现旧版的 Query Expansion + Dense + BM25 + Parent + Rerank 流程。"""

    def __init__(
        self,
        index: VectorIndexRepository,
        *,
        query_expander: QueryExpander | None = None,
        tokenizer: Tokenizer | None = None,
        reranker: Reranker | None = None,
        dense_limit_per_query: int = 3,
        bm25_limit_per_query: int = 3,
        result_limit: int = 3,
    ) -> None:
        for value, name in (
            (dense_limit_per_query, "dense_limit_per_query"),
            (bm25_limit_per_query, "bm25_limit_per_query"),
            (result_limit, "result_limit"),
        ):
            _validate_positive_int(value, name)
        self._index = index
        self._query_expander = query_expander
        self._tokenizer = tokenizer or JiebaTokenizer()
        self._reranker = reranker
        self._dense_limit_per_query = dense_limit_per_query
        self._bm25_limit_per_query = bm25_limit_per_query
        self._result_limit = result_limit

    def retrieve(
        self,
        *,
        search_query: str,
        user_id: str,
        session_id: str,
        recent_history: str = "",
        paper_id: str | None = None,
    ) -> _RetrievalExecution:
        if not isinstance(search_query, str) or not search_query.strip():
            raise ValueError("search_query 不能为空。")
        expanded = (
            self._query_expander.expand(query=search_query, recent_history=recent_history)
            if self._query_expander
            else ()
        )
        queries = _normalise_queries(expanded, search_query)
        all_children = self._index.list_children(user_id=user_id, session_id=session_id)
        if paper_id is not None:
            all_children = tuple(hit for hit in all_children if hit.chunk.paper_id == paper_id)
        bm25 = self._build_bm25(all_children)
        candidates: dict[str, _CandidateScores] = {}

        for query in queries:
            dense_arguments: dict[str, object] = {
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "limit": self._dense_limit_per_query,
            }
            if paper_id is not None:
                dense_arguments["paper_id"] = paper_id
            dense_hits = self._index.query_children(**dense_arguments)  # type: ignore[arg-type]
            for rank, hit in enumerate(dense_hits, start=1):
                if hit.chunk.parent_id:
                    candidates.setdefault(hit.chunk.parent_id, _CandidateScores()).observe_dense(
                        rank=rank, score=hit.score
                    )
            if bm25 is not None:
                for rank, (child_hit, score) in enumerate(self._top_bm25_hits(bm25, all_children, query), start=1):
                    if child_hit.chunk.parent_id:
                        candidates.setdefault(child_hit.chunk.parent_id, _CandidateScores()).observe_sparse(
                            rank=rank, score=score
                        )

        parent_arguments: dict[str, object] = {
            "parent_chunk_ids": tuple(candidates),
            "user_id": user_id,
            "session_id": session_id,
        }
        if paper_id is not None:
            parent_arguments["paper_id"] = paper_id
        parent_hits = self._index.get_parent_chunks(**parent_arguments)  # type: ignore[arg-type]
        materialised: list[tuple[IndexHit, _CandidateScores]] = [
            (parent_hits[parent_id], scores) for parent_id, scores in candidates.items() if parent_id in parent_hits
        ]
        rerank_scores: tuple[float | None, ...]
        if self._reranker and materialised:
            resolved_scores = self._reranker.score(
                query=search_query,
                documents=tuple(hit.chunk.content for hit, _ in materialised),
            )
            rerank_scores = tuple(resolved_scores)
        else:
            rerank_scores = tuple(None for _ in materialised)

        scored_results = [
            ScoredChunk(
                chunk=parent_hit.chunk,
                source_name=parent_hit.source_name,
                dense_rank=scores.dense_rank,
                dense_score=scores.dense_score,
                sparse_rank=scores.sparse_rank,
                sparse_score=scores.sparse_score,
                rerank_score=rerank_score,
            )
            for (parent_hit, scores), rerank_score in zip(materialised, rerank_scores, strict=True)
        ]
        scored_results.sort(key=self._result_sort_key)
        stages = ["query_expansion", "dense_child_retrieval", "bm25_child_retrieval", "parent_hydration"]
        if self._reranker:
            stages.append("rerank")
        return _RetrievalExecution(
            results=tuple(scored_results[: self._result_limit]),
            candidate_count=len(candidates),
            expanded_queries=queries,
            stages=tuple(stages),
        )

    def _build_bm25(self, child_hits: Sequence[IndexHit]) -> BM25Okapi | None:
        corpus = [list(self._tokenizer.tokenize(hit.chunk.content)) for hit in child_hits]
        return BM25Okapi(corpus) if corpus else None

    def _top_bm25_hits(
        self,
        bm25: BM25Okapi,
        child_hits: Sequence[IndexHit],
        query: str,
    ) -> tuple[tuple[IndexHit, float], ...]:
        if not child_hits:
            return ()
        scores = bm25.get_scores(list(self._tokenizer.tokenize(query)))
        ranked_indices = sorted(range(len(child_hits)), key=lambda index: float(scores[index]), reverse=True)
        return tuple(
            (child_hits[index], float(scores[index])) for index in ranked_indices[: self._bm25_limit_per_query]
        )

    @staticmethod
    def _result_sort_key(result: ScoredChunk) -> tuple[float, float, float, str]:
        # 有 Rerank 时按其结果排序；否则按最优通道排名保持稳定的降级行为。
        rerank_key = -(result.rerank_score if result.rerank_score is not None else float("-inf"))
        dense_key = float(result.dense_rank if result.dense_rank is not None else 10**9)
        sparse_key = float(result.sparse_rank if result.sparse_rank is not None else 10**9)
        return rerank_key, dense_key, sparse_key, result.chunk.chunk_id


class BM25Retriever:
    """持久化 BM25 的 Sparse-only 基线。"""

    def __init__(
        self,
        index: VectorIndexRepository,
        sparse_index: PersistentBM25Repository,
        *,
        candidate_limit: int = 10,
    ) -> None:
        _validate_positive_int(candidate_limit, "candidate_limit")
        self._index = index
        self._sparse_index = sparse_index
        self._candidate_limit = candidate_limit

    def invalidate_scope(self, *, user_id: str, session_id: str) -> None:
        self._sparse_index.invalidate_scope(user_id=user_id, session_id=session_id)

    def retrieve(
        self,
        *,
        search_query: str,
        user_id: str,
        session_id: str,
        recent_history: str = "",
        paper_id: str | None = None,
    ) -> _RetrievalExecution:
        started_at = time.perf_counter()
        child_hits = self._sparse_index.query(
            query=search_query,
            user_id=user_id,
            session_id=session_id,
            limit=self._candidate_limit,
            paper_id=paper_id,
        )
        sparse_elapsed = (time.perf_counter() - started_at) * 1000
        best_child_by_parent: dict[str, tuple[int, IndexHit, float]] = {}
        for sparse_rank, sparse_hit in enumerate(child_hits, start=1):
            parent_id = sparse_hit.hit.chunk.parent_id
            if parent_id and parent_id not in best_child_by_parent:
                best_child_by_parent[parent_id] = (sparse_rank, sparse_hit.hit, sparse_hit.score)
        hydration_started = time.perf_counter()
        parent_hits = self._index.get_parent_chunks(
            parent_chunk_ids=tuple(best_child_by_parent),
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
        )
        hydration_elapsed = (time.perf_counter() - hydration_started) * 1000
        results = tuple(
            ScoredChunk(
                chunk=parent_hits[parent_id].chunk,
                source_name=parent_hits[parent_id].source_name,
                sparse_rank=sparse_rank,
                sparse_score=sparse_score,
            )
            for parent_id, (sparse_rank, _, sparse_score) in best_child_by_parent.items()
            if parent_id in parent_hits
        )
        return _RetrievalExecution(
            results=results,
            candidate_count=len(child_hits),
            expanded_queries=(search_query,),
            stages=("bm25_child_retrieval", "parent_hydration"),
            stage_elapsed_ms=(
                ("bm25_child_retrieval", sparse_elapsed),
                ("parent_hydration", hydration_elapsed),
            ),
            index_version=self._sparse_index.index_version(user_id=user_id, session_id=session_id),
        )


@dataclass
class _ChildCandidate:
    hit: IndexHit
    scores: _CandidateScores


class PersistentHybridRetriever:
    """用于 E2--E4 和产品链路的持久化 Dense + BM25 混合检索器。"""

    def __init__(
        self,
        index: VectorIndexRepository,
        sparse_index: PersistentBM25Repository,
        *,
        fusion: str,
        query_expander: QueryExpander | None = None,
        reranker: Reranker | None = None,
        dense_limit_per_query: int = 10,
        bm25_limit_per_query: int = 10,
        result_limit: int = 10,
        rrf_k: int = 60,
    ) -> None:
        if fusion not in {"set_union", "rrf"}:
            raise ValueError("fusion 必须是 set_union 或 rrf。")
        for value, name in (
            (dense_limit_per_query, "dense_limit_per_query"),
            (bm25_limit_per_query, "bm25_limit_per_query"),
            (result_limit, "result_limit"),
            (rrf_k, "rrf_k"),
        ):
            _validate_positive_int(value, name)
        self._index = index
        self._sparse_index = sparse_index
        self._fusion = fusion
        self._query_expander = query_expander
        self._reranker = reranker
        self._dense_limit_per_query = dense_limit_per_query
        self._bm25_limit_per_query = bm25_limit_per_query
        self._result_limit = result_limit
        self._rrf_k = rrf_k

    def invalidate_scope(self, *, user_id: str, session_id: str) -> None:
        self._sparse_index.invalidate_scope(user_id=user_id, session_id=session_id)

    @staticmethod
    def _rrf_score(scores: _CandidateScores, rrf_k: int) -> float:
        return sum(1.0 / (rrf_k + rank) for rank in (scores.dense_rank, scores.sparse_rank) if rank is not None)

    @staticmethod
    def _union_key(candidate: _ChildCandidate) -> tuple[int, int, int, str]:
        dense_rank = candidate.scores.dense_rank or 10**9
        sparse_rank = candidate.scores.sparse_rank or 10**9
        return min(dense_rank, sparse_rank), dense_rank, sparse_rank, candidate.hit.chunk.chunk_id

    def retrieve(
        self,
        *,
        search_query: str,
        user_id: str,
        session_id: str,
        recent_history: str = "",
        paper_id: str | None = None,
        dense_limit_per_query: int | None = None,
        bm25_limit_per_query: int | None = None,
        result_limit: int | None = None,
        rerank_enabled: bool | None = None,
    ) -> _RetrievalExecution:
        if not isinstance(search_query, str) or not search_query.strip():
            raise ValueError("search_query 不能为空。")
        if paper_id is not None and (not isinstance(paper_id, str) or not paper_id.strip()):
            raise ValueError("paper_id 必须是非空字符串或 None。")
        dense_limit = dense_limit_per_query or self._dense_limit_per_query
        sparse_limit = bm25_limit_per_query or self._bm25_limit_per_query
        output_limit = result_limit or self._result_limit
        for value, name in (
            (dense_limit, "dense_limit_per_query"),
            (sparse_limit, "bm25_limit_per_query"),
            (output_limit, "result_limit"),
        ):
            _validate_positive_int(value, name)
        use_reranker = self._reranker is not None and (rerank_enabled is not False)
        timings: list[tuple[str, float]] = []
        expansion_started = time.perf_counter()
        expanded = (
            self._query_expander.expand(query=search_query, recent_history=recent_history)
            if self._query_expander
            else ()
        )
        if self._query_expander:
            timings.append(("query_expansion", (time.perf_counter() - expansion_started) * 1000))
        queries = _normalise_queries(expanded, search_query)
        candidates: dict[str, _ChildCandidate] = {}
        dense_elapsed = 0.0
        sparse_elapsed = 0.0
        for query in queries:
            dense_started = time.perf_counter()
            dense_arguments: dict[str, object] = {
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "limit": dense_limit,
            }
            if paper_id is not None:
                dense_arguments["paper_id"] = paper_id
            dense_hits = self._index.query_children(**dense_arguments)  # type: ignore[arg-type]
            dense_elapsed += (time.perf_counter() - dense_started) * 1000
            for rank, hit in enumerate(dense_hits, start=1):
                if not hit.chunk.parent_id:
                    continue
                candidate = candidates.setdefault(
                    hit.chunk.chunk_id, _ChildCandidate(hit=hit, scores=_CandidateScores())
                )
                candidate.scores.observe_dense(rank=rank, score=hit.score)
            sparse_started = time.perf_counter()
            sparse_hits = self._sparse_index.query(
                query=query,
                user_id=user_id,
                session_id=session_id,
                limit=sparse_limit,
                paper_id=paper_id,
            )
            sparse_elapsed += (time.perf_counter() - sparse_started) * 1000
            for rank, sparse_hit in enumerate(sparse_hits, start=1):
                hit = sparse_hit.hit
                if not hit.chunk.parent_id:
                    continue
                candidate = candidates.setdefault(
                    hit.chunk.chunk_id, _ChildCandidate(hit=hit, scores=_CandidateScores())
                )
                candidate.scores.observe_sparse(rank=rank, score=sparse_hit.score)
        timings.extend([("dense_child_retrieval", dense_elapsed), ("bm25_child_retrieval", sparse_elapsed)])

        candidates_by_parent: dict[str, _ChildCandidate] = {}
        for candidate in candidates.values():
            parent_id = candidate.hit.chunk.parent_id
            assert parent_id is not None
            current = candidates_by_parent.get(parent_id)
            if current is None:
                candidates_by_parent[parent_id] = candidate
                continue
            if self._fusion == "rrf":
                if self._rrf_score(candidate.scores, self._rrf_k) > self._rrf_score(current.scores, self._rrf_k):
                    candidates_by_parent[parent_id] = candidate
            elif self._union_key(candidate) < self._union_key(current):
                candidates_by_parent[parent_id] = candidate

        hydration_started = time.perf_counter()
        parent_arguments: dict[str, object] = {
            "parent_chunk_ids": tuple(candidates_by_parent),
            "user_id": user_id,
            "session_id": session_id,
        }
        if paper_id is not None:
            parent_arguments["paper_id"] = paper_id
        parent_hits = self._index.get_parent_chunks(**parent_arguments)  # type: ignore[arg-type]
        timings.append(("parent_hydration", (time.perf_counter() - hydration_started) * 1000))
        materialised: list[tuple[IndexHit, _ChildCandidate, float | None]] = []
        for parent_id, candidate in candidates_by_parent.items():
            parent_hit = parent_hits.get(parent_id)
            if parent_hit is None:
                continue
            fusion_score = self._rrf_score(candidate.scores, self._rrf_k) if self._fusion == "rrf" else None
            materialised.append((parent_hit, candidate, fusion_score))

        rerank_scores: tuple[float | None, ...]
        if use_reranker and materialised:
            # Reranker 只需重新排序 RRF/Union 已筛出的最终候选池。这样 E4 的输出仍为
            # Top-10，但不会把所有最多 20 个父块都送入 CPU CrossEncoder。
            materialised.sort(
                key=lambda item: (
                    -(item[2] if item[2] is not None else float("-inf")),
                    item[1].scores.dense_rank if item[1].scores.dense_rank is not None else 10**9,
                    item[1].scores.sparse_rank if item[1].scores.sparse_rank is not None else 10**9,
                    item[0].chunk.chunk_id,
                )
            )
            materialised = materialised[:output_limit]
            rerank_started = time.perf_counter()
            rerank_scores = self._reranker.score(
                query=search_query,
                documents=tuple(parent_hit.chunk.content for parent_hit, _, _ in materialised),
            )
            timings.append(("rerank", (time.perf_counter() - rerank_started) * 1000))
        else:
            rerank_scores = tuple(None for _ in materialised)

        results = [
            ScoredChunk(
                chunk=parent_hit.chunk,
                source_name=parent_hit.source_name,
                dense_rank=candidate.scores.dense_rank,
                dense_score=candidate.scores.dense_score,
                sparse_rank=candidate.scores.sparse_rank,
                sparse_score=candidate.scores.sparse_score,
                fusion_score=fusion_score,
                rerank_score=rerank_score,
            )
            for (parent_hit, candidate, fusion_score), rerank_score in zip(materialised, rerank_scores, strict=True)
        ]
        results.sort(key=self._result_sort_key)
        stages = ["dense_child_retrieval", "bm25_child_retrieval", "parent_hydration"]
        if self._query_expander:
            stages.insert(0, "query_expansion")
        if self._fusion == "rrf":
            stages.insert(-1, "rrf_fusion")
        if use_reranker:
            stages.append("rerank")
        return _RetrievalExecution(
            results=tuple(results[:output_limit]),
            candidate_count=len(candidates),
            expanded_queries=queries,
            stages=tuple(stages),
            stage_elapsed_ms=tuple(timings),
            index_version=self._sparse_index.index_version(user_id=user_id, session_id=session_id),
        )

    @staticmethod
    def _result_sort_key(result: ScoredChunk) -> tuple[float, float, int, int, str]:
        rerank_key = -(result.rerank_score if result.rerank_score is not None else float("-inf"))
        fusion_key = -(result.fusion_score if result.fusion_score is not None else float("-inf"))
        dense_key = result.dense_rank if result.dense_rank is not None else 10**9
        sparse_key = result.sparse_rank if result.sparse_rank is not None else 10**9
        return rerank_key, fusion_key, dense_key, sparse_key, result.chunk.chunk_id


class RetrievalService:
    """检索服务唯一入口，统一记录查询、结果、范围与耗时。"""

    def __init__(self, retriever: _Retriever) -> None:
        self._retriever = retriever

    def invalidate_scope(self, *, user_id: str, session_id: str) -> None:
        invalidate = getattr(self._retriever, "invalidate_scope", None)
        if callable(invalidate):
            invalidate(user_id=user_id, session_id=session_id)

    def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str,
        search_query: str | None = None,
        query_type: QueryType | None = None,
        recent_history: str = "",
        paper_id: str | None = None,
    ) -> RetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空。")
        final_search_query = (search_query or query).strip()
        if not final_search_query:
            raise ValueError("search_query 不能为空。")
        started_at = time.perf_counter()
        execution = self._retriever.retrieve(
            search_query=final_search_query,
            user_id=user_id,
            session_id=session_id,
            recent_history=recent_history,
            paper_id=paper_id,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        trace = RetrievalTrace(
            trace_id=f"trace_{uuid.uuid4().hex}",
            query=query.strip(),
            search_query=final_search_query,
            query_type=query_type,
            candidate_count=execution.candidate_count,
            results=execution.results,
            elapsed_ms=elapsed_ms,
            stages=execution.stages,
            expanded_queries=execution.expanded_queries,
            stage_elapsed_ms=execution.stage_elapsed_ms,
            index_version=execution.index_version,
        )
        return RetrievalResult(results=execution.results, trace=trace)
