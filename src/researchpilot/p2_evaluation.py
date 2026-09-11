"""P2 Query-aware 检索阶段的页级评测与聚合。

该模块只消费运行时返回的候选与评测集 Gold，用于*评分*；Gold 不会传入
Planner、Retriever 或 Coverage Gate。这样可以同时记录 Oracle 上界与真实
Scoped / End-to-End 输入边界下的表现，而不产生标注泄漏。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import fmean

from .evidence_control import EvidenceCandidate, RetryThenRefuseDecision
from .p2_dataset import P2Case


def _overlaps(*, left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


@dataclass(frozen=True)
class P2RetrievalCaseEvaluation:
    """一题的页级证据覆盖结果；``None`` 表示不可回答题不参与该指标。"""

    case_id: str
    split: str
    mode: str
    selected_count: int
    retrieved_paper_keys: tuple[str, ...]
    per_paper_recall: float | None
    evidence_coverage: bool | None
    coverage_gate_passed: bool | None
    retry_count: int
    refused_before_generation: bool
    refusal_reason: str | None
    retrieval_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["retrieved_paper_keys"] = list(self.retrieved_paper_keys)
        return result


@dataclass(frozen=True)
class P2RetrievalSummary:
    mode: str
    evaluated_cases: int
    answerable_cases: int
    unanswerable_cases: int
    per_paper_recall: float | None
    evidence_coverage_rate: float | None
    coverage_gate_pass_rate: float | None
    retry_rate: float | None
    pre_generation_refusal_rate: float | None
    mean_retrieval_latency_ms: float
    p50_retrieval_latency_ms: float
    p95_retrieval_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_p2_retrieval_case(
    *,
    case: P2Case,
    mode: str,
    selected: Sequence[EvidenceCandidate],
    retrieval_latency_ms: float,
    decision: RetryThenRefuseDecision | None = None,
) -> P2RetrievalCaseEvaluation:
    """按“论文键 + 页码区间重叠”评分，刻意不绑定不稳定的 Chunk ID。"""
    selected_tuple = tuple(selected)
    if not isinstance(retrieval_latency_ms, int | float) or not math.isfinite(retrieval_latency_ms):
        raise ValueError("retrieval_latency_ms 必须为有限数值")

    if case.answerable:
        found_by_paper: list[bool] = []
        for paper_key in case.required_paper_keys:
            paper_gold = tuple(item for item in case.gold_evidence if item.paper_key == paper_key)
            found_by_paper.append(
                any(
                    candidate.paper_key == paper_key
                    and any(
                        _overlaps(
                            left_start=candidate.result.chunk.page_start,
                            left_end=candidate.result.chunk.page_end,
                            right_start=evidence.page_start,
                            right_end=evidence.page_end,
                        )
                        for evidence in paper_gold
                    )
                    for candidate in selected_tuple
                )
            )
        per_paper_recall: float | None = fmean(1.0 if found else 0.0 for found in found_by_paper)
        evidence_coverage: bool | None = all(found_by_paper)
    else:
        per_paper_recall = None
        evidence_coverage = None

    return P2RetrievalCaseEvaluation(
        case_id=case.case_id,
        split=case.split,
        mode=mode,
        selected_count=len(selected_tuple),
        retrieved_paper_keys=tuple(sorted({item.paper_key for item in selected_tuple})),
        per_paper_recall=per_paper_recall,
        evidence_coverage=evidence_coverage,
        coverage_gate_passed=(decision.gate.passed if decision is not None else None),
        retry_count=(decision.retry_count if decision is not None else 0),
        refused_before_generation=(decision.refused if decision is not None else False),
        refusal_reason=(decision.refusal_reason if decision is not None else None),
        retrieval_latency_ms=float(retrieval_latency_ms),
    )


def _percentile(values: Sequence[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * ratio) - 1))
    return ordered[index]


def summarize_p2_retrieval(
    evaluations: Sequence[P2RetrievalCaseEvaluation],
) -> P2RetrievalSummary:
    if not evaluations:
        raise ValueError("至少需要一条 P2 检索评测结果")
    modes = {item.mode for item in evaluations}
    if len(modes) != 1:
        raise ValueError("一次聚合只能包含一个实验模式")
    answerable = tuple(item for item in evaluations if item.per_paper_recall is not None)
    gated = tuple(item for item in evaluations if item.coverage_gate_passed is not None)
    latencies = tuple(item.retrieval_latency_ms for item in evaluations)
    return P2RetrievalSummary(
        mode=next(iter(modes)),
        evaluated_cases=len(evaluations),
        answerable_cases=len(answerable),
        unanswerable_cases=len(evaluations) - len(answerable),
        per_paper_recall=(fmean(item.per_paper_recall for item in answerable) if answerable else None),
        evidence_coverage_rate=(
            fmean(1.0 if item.evidence_coverage else 0.0 for item in answerable) if answerable else None
        ),
        coverage_gate_pass_rate=(fmean(1.0 if item.coverage_gate_passed else 0.0 for item in gated) if gated else None),
        retry_rate=(fmean(1.0 if item.retry_count else 0.0 for item in gated) if gated else None),
        pre_generation_refusal_rate=(
            fmean(1.0 if item.refused_before_generation else 0.0 for item in gated) if gated else None
        ),
        mean_retrieval_latency_ms=fmean(latencies),
        p50_retrieval_latency_ms=_percentile(latencies, 0.50),
        p95_retrieval_latency_ms=_percentile(latencies, 0.95),
    )
