"""将 P2 QueryPlan 接入已冻结的持久化 Hybrid Retriever。

该适配器只做论文级范围过滤和计划参数下传；均衡候选、Gate、重试与拒答
由 :mod:`researchpilot.evidence_control` 负责，因此可独立替换底层索引。
"""

from __future__ import annotations

from collections.abc import Mapping

from .evidence_control import EvidenceCandidate
from .query_plan import QueryPlan
from .retrieval import PersistentHybridRetriever
from .schemas import SchemaValidationError


class PersistentHybridPerPaperProvider:
    """每次仅在一个 ``paper_id`` 上运行 Dense + BM25 + RRF + 可选 Rerank。"""

    def __init__(
        self,
        *,
        retriever: PersistentHybridRetriever,
        paper_id_by_key: Mapping[str, str],
        user_id: str,
        session_id: str,
        extraction_quality_by_key: Mapping[str, str] | None = None,
        retry_limit_multiplier: int = 2,
    ) -> None:
        if not paper_id_by_key:
            raise SchemaValidationError("paper_id_by_key 不能为空。")
        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(session_id, str)
            or not session_id.strip()
        ):
            raise SchemaValidationError("P2 检索 Provider 必须指定 user_id 和 session_id。")
        if not isinstance(retry_limit_multiplier, int) or retry_limit_multiplier < 1:
            raise SchemaValidationError("retry_limit_multiplier 必须为正整数。")
        self._retriever = retriever
        self._paper_id_by_key = dict(paper_id_by_key)
        self._user_id = user_id
        self._session_id = session_id
        self._quality_by_key = dict(extraction_quality_by_key or {})
        self._retry_limit_multiplier = retry_limit_multiplier

    def retrieve_for_paper(
        self, *, query: str, paper_key: str, plan: QueryPlan, expanded: bool = False
    ) -> tuple[EvidenceCandidate, ...]:
        paper_id = self._paper_id_by_key.get(paper_key)
        if paper_id is None:
            raise SchemaValidationError(f"未配置运行时 paper_key 到 paper_id 映射：{paper_key}")
        multiplier = self._retry_limit_multiplier if expanded else 1
        # Retriever 在单论文 Scope 内执行；输出上限至少覆盖 Dense / BM25 的每路候选。
        output_limit = max(plan.dense_limit, plan.sparse_limit) * multiplier
        execution = self._retriever.retrieve(
            search_query=query,
            user_id=self._user_id,
            session_id=self._session_id,
            paper_id=paper_id,
            dense_limit_per_query=plan.dense_limit * multiplier,
            bm25_limit_per_query=plan.sparse_limit * multiplier,
            result_limit=output_limit,
            rerank_enabled=plan.rerank_enabled,
        )
        quality = self._quality_by_key.get(paper_key, "normal")
        return tuple(
            EvidenceCandidate(paper_key=paper_key, result=result, extraction_quality=quality)
            for result in execution.results
        )
