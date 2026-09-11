"""P2-B2：按论文均衡候选与 G0/G1 Evidence Coverage Gate。

本模块不读取评测 Gold，也不调用 LLM。它只对运行时召回器给出的候选
执行论文范围过滤、配额选择和可用证据检查；B3 会在此基础上添加一次
扩检与结构化拒答状态机。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .query_plan import QueryPlan
from .schemas import SchemaValidationError, ScoredChunk

_EXTRACTION_QUALITIES = frozenset({"normal", "low_text", "failed"})


def _score(item: EvidenceCandidate) -> float:
    """使用最终可用分数排序，分数缺失时退回到召回排名。"""
    result = item.result
    for value in (result.rerank_score, result.fusion_score, result.dense_score, result.sparse_score):
        if value is not None:
            return float(value)
    ranks = (rank for rank in (result.dense_rank, result.sparse_rank) if rank is not None)
    return -float(min(ranks, default=10**9))


@dataclass(frozen=True)
class EvidenceCandidate:
    """已绑定运行时论文范围的候选；``paper_key`` 不能来自 Gold。"""

    paper_key: str
    result: ScoredChunk
    extraction_quality: str = "normal"

    def __post_init__(self) -> None:
        if not isinstance(self.paper_key, str) or not self.paper_key.strip():
            raise SchemaValidationError("EvidenceCandidate.paper_key 不能为空。")
        if not isinstance(self.result, ScoredChunk):
            raise SchemaValidationError("EvidenceCandidate.result 必须是 ScoredChunk。")
        if self.extraction_quality not in _EXTRACTION_QUALITIES:
            raise SchemaValidationError("extraction_quality 必须是 normal、low_text 或 failed。")

    @property
    def score(self) -> float:
        return _score(self)

    def is_usable(self, *, minimum_score: float) -> bool:
        return (
            self.extraction_quality == "normal"
            and bool(self.result.chunk.content.strip())
            and self.score >= minimum_score
        )


@dataclass(frozen=True)
class CoverageGateDecision:
    """G0 文档覆盖和 G1 可用相关证据覆盖的确定性决策。"""

    g0_document_coverage: bool
    g1_relevant_evidence_coverage: bool
    passed: bool
    missing_document_paper_keys: tuple[str, ...]
    missing_relevant_evidence_paper_keys: tuple[str, ...]
    per_paper_candidate_counts: tuple[tuple[str, int], ...]
    per_paper_usable_counts: tuple[tuple[str, int], ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.passed != (self.g0_document_coverage and self.g1_relevant_evidence_coverage):
            raise SchemaValidationError("Coverage Gate 的 passed 必须等于 G0 与 G1 的合取。")
        if self.passed and self.reason is not None:
            raise SchemaValidationError("Coverage Gate 通过时不应设置失败原因。")
        if not self.passed and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise SchemaValidationError("Coverage Gate 失败时必须记录原因。")


@dataclass(frozen=True)
class QueryAwareRetrievalAttempt:
    """一次（尚未扩检的）按论文召回、均衡候选和 Gate 结果。"""

    candidates: tuple[EvidenceCandidate, ...]
    selected: tuple[EvidenceCandidate, ...]
    gate: CoverageGateDecision
    per_paper_request_count: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RetryThenRefuseDecision:
    """P2-B3 的最终控制决策；生成层只可消费 ``selected`` 或拒答。"""

    selected: tuple[EvidenceCandidate, ...]
    gate: CoverageGateDecision
    retry_count: int
    refused: bool
    refusal_reason: str | None
    per_paper_request_count: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.retry_count, int) or self.retry_count not in {0, 1}:
            raise SchemaValidationError("retry_count 当前只能为 0 或 1。")
        if self.refused:
            if self.gate.passed:
                raise SchemaValidationError("Coverage Gate 通过时不能拒答。")
            if not isinstance(self.refusal_reason, str) or not self.refusal_reason.strip():
                raise SchemaValidationError("拒答必须记录 refusal_reason。")
        elif not self.gate.passed or self.refusal_reason is not None:
            raise SchemaValidationError("非拒答决策必须来自通过的 Coverage Gate。")

    def trace_fields(self) -> dict[str, object]:
        """供后续 RetrievalTrace / JSONL 运行记录写入的原生类型字段。"""
        return {
            "g0_document_coverage": self.gate.g0_document_coverage,
            "g1_relevant_evidence_coverage": self.gate.g1_relevant_evidence_coverage,
            "coverage_gate_reason": self.gate.reason,
            "retry_count": self.retry_count,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "per_paper_candidate_counts": list(self.gate.per_paper_candidate_counts),
        }


class PerPaperCandidateProvider(Protocol):
    """P2 适配器协议：每次调用只允许返回指定论文的运行时候选。"""

    def retrieve_for_paper(
        self, *, query: str, paper_key: str, plan: QueryPlan, expanded: bool = False
    ) -> Sequence[EvidenceCandidate]: ...


def _deduplicate(candidates: Sequence[EvidenceCandidate]) -> tuple[EvidenceCandidate, ...]:
    """同一论文的同一 Chunk 只保留最终分数最高的一条。"""
    best: dict[tuple[str, str], EvidenceCandidate] = {}
    for candidate in candidates:
        key = (candidate.paper_key, candidate.result.chunk.chunk_id)
        current = best.get(key)
        if current is None or candidate.score > current.score:
            best[key] = candidate
    return tuple(best.values())


def balance_candidates(
    *,
    candidates: Sequence[EvidenceCandidate],
    required_paper_keys: Sequence[str],
    per_paper_quota: int,
    final_limit: int,
) -> tuple[EvidenceCandidate, ...]:
    """先按论文保留配额，再按全局分数补足余量，禁止一篇论文垄断 Top-K。"""
    required = tuple(dict.fromkeys(required_paper_keys))
    if not required:
        raise SchemaValidationError("均衡召回至少需要一篇目标论文。")
    if len(required) != len(required_paper_keys):
        raise SchemaValidationError("required_paper_keys 不能重复。")
    if not isinstance(per_paper_quota, int) or per_paper_quota < 1:
        raise SchemaValidationError("per_paper_quota 必须为正整数。")
    if not isinstance(final_limit, int) or final_limit < len(required) * per_paper_quota:
        raise SchemaValidationError("final_limit 必须能容纳每篇论文的候选配额。")
    unknown = {candidate.paper_key for candidate in candidates} - set(required)
    if unknown:
        raise SchemaValidationError(f"候选包含不属于运行时目标 Scope 的论文：{sorted(unknown)}")

    unique = _deduplicate(candidates)
    grouped = {
        paper_key: sorted(
            (candidate for candidate in unique if candidate.paper_key == paper_key),
            key=lambda item: (-item.score, item.result.chunk.chunk_id),
        )
        for paper_key in required
    }
    selected: list[EvidenceCandidate] = []
    selected_ids: set[tuple[str, str]] = set()
    for paper_key in required:
        for candidate in grouped[paper_key][:per_paper_quota]:
            selected.append(candidate)
            selected_ids.add((candidate.paper_key, candidate.result.chunk.chunk_id))
    remaining = sorted(
        (
            candidate
            for candidate in unique
            if (candidate.paper_key, candidate.result.chunk.chunk_id) not in selected_ids
        ),
        key=lambda item: (-item.score, item.result.chunk.chunk_id),
    )
    selected.extend(remaining[: max(0, final_limit - len(selected))])
    return tuple(sorted(selected, key=lambda item: (-item.score, item.result.chunk.chunk_id)))


def evaluate_coverage_gate(
    *,
    all_candidates: Sequence[EvidenceCandidate],
    selected_candidates: Sequence[EvidenceCandidate],
    required_paper_keys: Sequence[str],
    minimum_evidence_per_paper: int,
    minimum_score: float = 0.0,
) -> CoverageGateDecision:
    """G0 看是否至少召回一条，G1 看最终候选是否存在可引用的证据。"""
    required = tuple(dict.fromkeys(required_paper_keys))
    if not required:
        raise SchemaValidationError("Coverage Gate 至少需要一篇目标论文。")
    if not isinstance(minimum_evidence_per_paper, int) or minimum_evidence_per_paper < 1:
        raise SchemaValidationError("minimum_evidence_per_paper 必须为正整数。")
    if not isinstance(minimum_score, int | float):
        raise SchemaValidationError("minimum_score 必须为数值。")
    unknown = {item.paper_key for item in (*all_candidates, *selected_candidates)} - set(required)
    if unknown:
        raise SchemaValidationError(f"Coverage Gate 收到 Scope 外候选：{sorted(unknown)}")

    candidate_counts = tuple(
        (paper_key, sum(item.paper_key == paper_key for item in all_candidates)) for paper_key in required
    )
    usable_counts = tuple(
        (
            paper_key,
            sum(
                item.paper_key == paper_key and item.is_usable(minimum_score=float(minimum_score))
                for item in selected_candidates
            ),
        )
        for paper_key in required
    )
    missing_documents = tuple(paper_key for paper_key, count in candidate_counts if count < 1)
    missing_evidence = tuple(paper_key for paper_key, count in usable_counts if count < minimum_evidence_per_paper)
    g0 = not missing_documents
    g1 = not missing_evidence
    if not g0:
        reason = "g0_missing_document_coverage"
    elif not g1:
        reason = "g1_insufficient_relevant_evidence"
    else:
        reason = None
    return CoverageGateDecision(
        g0_document_coverage=g0,
        g1_relevant_evidence_coverage=g1,
        passed=g0 and g1,
        missing_document_paper_keys=missing_documents,
        missing_relevant_evidence_paper_keys=missing_evidence,
        per_paper_candidate_counts=candidate_counts,
        per_paper_usable_counts=usable_counts,
        reason=reason,
    )


class QueryAwareCandidateController:
    """用 QueryPlan 驱动每篇论文独立召回；不执行扩检或生成。"""

    def __init__(
        self, *, provider: PerPaperCandidateProvider, minimum_score: float = 0.0, per_paper_quota: int = 1
    ) -> None:
        if not isinstance(minimum_score, int | float):
            raise SchemaValidationError("minimum_score 必须为数值。")
        if not isinstance(per_paper_quota, int) or per_paper_quota < 1:
            raise SchemaValidationError("per_paper_quota 必须为正整数。")
        self._provider = provider
        self._minimum_score = float(minimum_score)
        self._per_paper_quota = per_paper_quota

    def retrieve_initial(self, *, query: str, plan: QueryPlan, final_limit: int) -> QueryAwareRetrievalAttempt:
        if not isinstance(query, str) or not query.strip():
            raise SchemaValidationError("query 不能为空。")
        if not plan.needs_retrieval:
            raise SchemaValidationError("不检索的 QueryPlan 不能进入 Candidate Controller。")
        if not plan.coverage_gate_required:
            raise SchemaValidationError("P2 Candidate Controller 只接受启用 Coverage Gate 的计划。")
        if not isinstance(final_limit, int) or final_limit < 1:
            raise SchemaValidationError("final_limit 必须为正整数。")

        candidates, requested = self._collect(query=query.strip(), plan=plan, expanded=False)
        return self._finalize(
            candidates=candidates,
            requested=requested,
            plan=plan,
            final_limit=final_limit,
        )

    def retrieve_with_retry(self, *, query: str, plan: QueryPlan, final_limit: int) -> RetryThenRefuseDecision:
        """初检失败时只允许一次按论文扩检，仍失败则拒答而非补无关引用。"""
        initial = self.retrieve_initial(query=query, plan=plan, final_limit=final_limit)
        if initial.gate.passed:
            return RetryThenRefuseDecision(
                selected=initial.selected,
                gate=initial.gate,
                retry_count=0,
                refused=False,
                refusal_reason=None,
                per_paper_request_count=initial.per_paper_request_count,
            )
        if not plan.allow_one_retry:
            return RetryThenRefuseDecision(
                selected=initial.selected,
                gate=initial.gate,
                retry_count=0,
                refused=True,
                refusal_reason=f"insufficient_evidence:{initial.gate.reason}",
                per_paper_request_count=initial.per_paper_request_count,
            )

        expanded_candidates, expanded_requests = self._collect(query=query.strip(), plan=plan, expanded=True)
        merged = self._finalize(
            candidates=(*initial.candidates, *expanded_candidates),
            requested=tuple(
                (paper_key, initial_count + expanded_count)
                for (paper_key, initial_count), (_, expanded_count) in zip(
                    initial.per_paper_request_count, expanded_requests, strict=True
                )
            ),
            plan=plan,
            final_limit=final_limit,
        )
        return RetryThenRefuseDecision(
            selected=merged.selected,
            gate=merged.gate,
            retry_count=1,
            refused=not merged.gate.passed,
            refusal_reason=(None if merged.gate.passed else f"insufficient_evidence:{merged.gate.reason}"),
            per_paper_request_count=merged.per_paper_request_count,
        )

    def _collect(
        self, *, query: str, plan: QueryPlan, expanded: bool
    ) -> tuple[tuple[EvidenceCandidate, ...], tuple[tuple[str, int], ...]]:
        requested: list[tuple[str, int]] = []
        candidates: list[EvidenceCandidate] = []
        for paper_key in plan.target_paper_keys:
            returned = tuple(
                self._provider.retrieve_for_paper(query=query, paper_key=paper_key, plan=plan, expanded=expanded)
            )
            if any(item.paper_key != paper_key for item in returned):
                raise SchemaValidationError("PerPaperCandidateProvider 不能返回其他论文的候选。")
            requested.append((paper_key, 1))
            candidates.extend(returned)
        return tuple(_deduplicate(candidates)), tuple(requested)

    def _finalize(
        self,
        *,
        candidates: Sequence[EvidenceCandidate],
        requested: tuple[tuple[str, int], ...],
        plan: QueryPlan,
        final_limit: int,
    ) -> QueryAwareRetrievalAttempt:
        selected = balance_candidates(
            candidates=candidates,
            required_paper_keys=plan.target_paper_keys,
            per_paper_quota=self._per_paper_quota,
            final_limit=max(final_limit, len(plan.target_paper_keys) * self._per_paper_quota),
        )
        gate = evaluate_coverage_gate(
            all_candidates=candidates,
            selected_candidates=selected,
            required_paper_keys=plan.target_paper_keys,
            minimum_evidence_per_paper=plan.minimum_evidence_per_paper,
            minimum_score=self._minimum_score,
        )
        return QueryAwareRetrievalAttempt(
            candidates=tuple(_deduplicate(candidates)),
            selected=selected,
            gate=gate,
            per_paper_request_count=requested,
        )
