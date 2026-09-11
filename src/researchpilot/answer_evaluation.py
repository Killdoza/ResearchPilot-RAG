"""P1-E 回答与引用的确定性评测，不使用 LLM-as-a-Judge。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Protocol

from .crossdoc import CrossDocumentCase, CrossDocumentEvidence
from .evaluation import EvalCase, GoldEvidence


class _AnswerCase(Protocol):
    case_id: str
    split: str
    answerable: bool
    gold_evidence: tuple[GoldEvidence | CrossDocumentEvidence, ...]


@dataclass(frozen=True)
class AnswerCitation:
    """模型声称使用的一条论文页级引用。"""

    paper_key: str
    page_start: int
    page_end: int
    context_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_key, str) or not self.paper_key.strip():
            raise ValueError("citation.paper_key 不能为空。")
        if (
            not isinstance(self.page_start, int)
            or not isinstance(self.page_end, int)
            or self.page_start < 1
            or self.page_end < self.page_start
        ):
            raise ValueError("citation 页码范围非法。")
        if self.context_id is not None and (not isinstance(self.context_id, str) or not self.context_id.strip()):
            raise ValueError("citation.context_id 必须为非空字符串或 None。")


@dataclass(frozen=True)
class GeneratedAnswer:
    """四种模式均写入同一 JSONL 结构的模型输出。"""

    answer: str
    citations: tuple[AnswerCitation, ...]
    refused: bool
    refusal_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: float | None
    retrieval_latency_ms: float
    generation_latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer 不能为空。")
        if any(not isinstance(citation, AnswerCitation) for citation in self.citations):
            raise ValueError("citations 必须由 AnswerCitation 组成。")
        if not isinstance(self.refused, bool):
            raise ValueError("refused 必须为布尔值。")
        if self.refused and (not isinstance(self.refusal_reason, str) or not self.refusal_reason.strip()):
            raise ValueError("拒答必须填写 refusal_reason。")
        if not self.refused and self.refusal_reason is not None:
            raise ValueError("非拒答不能填写 refusal_reason。")
        for value, name in (
            (self.prompt_tokens, "prompt_tokens"),
            (self.completion_tokens, "completion_tokens"),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} 必须为非负整数或 None。")
        if self.cost is not None and (not isinstance(self.cost, int | float) or self.cost < 0):
            raise ValueError("cost 必须为非负数或 None。")
        for value, name in (
            (self.retrieval_latency_ms, "retrieval_latency_ms"),
            (self.generation_latency_ms, "generation_latency_ms"),
        ):
            if not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{name} 必须为非负数。")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerCaseEvaluation:
    case_id: str
    split: str
    answerable: bool
    citation_precision: float | None
    citation_recall: float | None
    citation_f1: float | None
    correct_refusal: bool | None
    answer_correctness: bool | None
    cited_count: int
    gold_evidence_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerEvaluationSummary:
    evaluated_cases: int
    answerable_cases: int
    unanswerable_cases: int
    citation_precision: float | None
    citation_recall: float | None
    citation_f1: float | None
    correct_refusal_rate: float | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
    mean_cost: float | None
    mean_retrieval_latency_ms: float
    mean_generation_latency_ms: float
    p50_end_to_end_latency_ms: float
    p95_end_to_end_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _overlaps(*, cited: AnswerCitation, gold: GoldEvidence | CrossDocumentEvidence) -> bool:
    return cited.paper_key == gold.paper_key and cited.page_start <= gold.page_end and cited.page_end >= gold.page_start


def _deduplicate(citations: Sequence[AnswerCitation]) -> tuple[AnswerCitation, ...]:
    unique: list[AnswerCitation] = []
    seen: set[tuple[str, int, int]] = set()
    for citation in citations:
        key = (citation.paper_key, citation.page_start, citation.page_end)
        if key not in seen:
            unique.append(citation)
            seen.add(key)
    return tuple(unique)


def evaluate_generated_answer(*, case: EvalCase | CrossDocumentCase, answer: GeneratedAnswer) -> AnswerCaseEvaluation:
    """按论文键与页码重叠计算 Citation P/R/F1；语义正确性留给人工复核。"""
    citations = _deduplicate(answer.citations)
    if not case.answerable:
        return AnswerCaseEvaluation(
            case_id=case.case_id,
            split=case.split,
            answerable=False,
            citation_precision=None,
            citation_recall=None,
            citation_f1=None,
            correct_refusal=answer.refused and not citations,
            answer_correctness=None,
            cited_count=len(citations),
            gold_evidence_count=0,
        )
    gold = case.gold_evidence
    cited_hits = sum(any(_overlaps(cited=cited, gold=item) for item in gold) for cited in citations)
    gold_hits = sum(any(_overlaps(cited=cited, gold=item) for cited in citations) for item in gold)
    precision = cited_hits / len(citations) if citations else 0.0
    recall = gold_hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return AnswerCaseEvaluation(
        case_id=case.case_id,
        split=case.split,
        answerable=True,
        citation_precision=precision,
        citation_recall=recall,
        citation_f1=f1,
        correct_refusal=None,
        answer_correctness=None,
        cited_count=len(citations),
        gold_evidence_count=len(gold),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_answer_evaluations(
    *, outcomes: Sequence[AnswerCaseEvaluation], answers: Sequence[GeneratedAnswer]
) -> AnswerEvaluationSummary:
    if not outcomes or len(outcomes) != len(answers):
        raise ValueError("必须提供数量相同且非空的回答结果与评测结果。")
    answerable = [outcome for outcome in outcomes if outcome.answerable]
    unanswerable = [outcome for outcome in outcomes if not outcome.answerable]

    def _mean_optional(values: Sequence[int | float | None]) -> float | None:
        populated = [float(value) for value in values if value is not None]
        return fmean(populated) if populated else None

    total_latency = [answer.retrieval_latency_ms + answer.generation_latency_ms for answer in answers]
    return AnswerEvaluationSummary(
        evaluated_cases=len(outcomes),
        answerable_cases=len(answerable),
        unanswerable_cases=len(unanswerable),
        citation_precision=(
            fmean(item.citation_precision for item in answerable if item.citation_precision is not None)
            if answerable
            else None
        ),
        citation_recall=(
            fmean(item.citation_recall for item in answerable if item.citation_recall is not None)
            if answerable
            else None
        ),
        citation_f1=(
            fmean(item.citation_f1 for item in answerable if item.citation_f1 is not None) if answerable else None
        ),
        correct_refusal_rate=(
            fmean(float(bool(item.correct_refusal)) for item in unanswerable) if unanswerable else None
        ),
        mean_prompt_tokens=_mean_optional([answer.prompt_tokens for answer in answers]),
        mean_completion_tokens=_mean_optional([answer.completion_tokens for answer in answers]),
        mean_cost=_mean_optional([answer.cost for answer in answers]),
        mean_retrieval_latency_ms=fmean(answer.retrieval_latency_ms for answer in answers),
        mean_generation_latency_ms=fmean(answer.generation_latency_ms for answer in answers),
        p50_end_to_end_latency_ms=_percentile(total_latency, 0.50),
        p95_end_to_end_latency_ms=_percentile(total_latency, 0.95),
    )


def render_answer_summary_markdown(*, experiment_name: str, split: str, summary: AnswerEvaluationSummary) -> str:
    def value(number: float | None) -> str:
        return "未配置 / 未产生" if number is None else f"{number:.4f}"

    rows = [
        ("评测问题数", summary.evaluated_cases),
        ("可回答问题数", summary.answerable_cases),
        ("不可回答问题数", summary.unanswerable_cases),
        ("Citation Precision", value(summary.citation_precision)),
        ("Citation Recall", value(summary.citation_recall)),
        ("Citation F1", value(summary.citation_f1)),
        ("不可回答题正确拒答率", value(summary.correct_refusal_rate)),
        ("平均 Prompt Token", value(summary.mean_prompt_tokens)),
        ("平均 Completion Token", value(summary.mean_completion_tokens)),
        ("平均调用成本", value(summary.mean_cost)),
        ("平均检索耗时（ms）", f"{summary.mean_retrieval_latency_ms:.2f}"),
        ("平均生成耗时（ms）", f"{summary.mean_generation_latency_ms:.2f}"),
        ("端到端 P50（ms）", f"{summary.p50_end_to_end_latency_ms:.2f}"),
        ("端到端 P95（ms）", f"{summary.p95_end_to_end_latency_ms:.2f}"),
    ]
    table = "\n".join(f"| {label} | {value_} |" for label, value_ in rows)
    return (
        f"# {experiment_name} · {split} 回答与引用评测\n\n"
        "| 指标 | 数值 |\n|---|---:|\n"
        f"{table}\n\n"
        "说明：Citation 指标按论文键和 Gold 页码重叠确定；答案语义正确性必须通过人工复核填写，"
        "本报告不会用另一个模型伪造正确性分数。\n"
    )
