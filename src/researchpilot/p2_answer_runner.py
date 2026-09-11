"""P2 的受证据约束回答生成。

这里的职责仅是把已经选出的运行时 Context 交给模型，并验证模型不会伪造
论文、页码或 Context ID。它不读取 Gold 标注，也不决定检索 Scope。
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .answer_runner import ModelCompletion, StructuredAnswerClient, _extract_json
from .evaluation import EvaluationDataError
from .evidence_control import EvidenceCandidate, RetryThenRefuseDecision
from .p2_answer_protocol import P2Citation, P2Claim, P2GeneratedAnswer, validate_claim_citations
from .p2_runner import P2ExecutionConfig


@dataclass(frozen=True)
class P2AnswerExecution:
    answer: P2GeneratedAnswer
    prompt_tokens: int | None
    completion_tokens: int | None
    generation_latency_ms: float
    cost: float | None
    validation_violations: tuple[str, ...] = ()


def render_evidence_context(candidates: Sequence[EvidenceCandidate]) -> str:
    """显式暴露可引用 ID；模型不得引用该列表以外的任何内容。"""
    sections: list[str] = []
    for rank, candidate in enumerate(candidates, start=1):
        chunk = candidate.result.chunk
        sections.append(
            "\n".join(
                (
                    f"[证据 {rank}] context_id={chunk.chunk_id}",
                    f"paper_key={candidate.paper_key}; pages={chunk.page_start}-{chunk.page_end}; "
                    f"section={' > '.join(chunk.section_path) or 'unknown'}",
                    chunk.content.strip(),
                )
            )
        )
    return "\n\n".join(sections)


def estimate_p2_prompt_tokens(*, question: str, candidates: Sequence[EvidenceCandidate]) -> int:
    return math.ceil((len(question) + len(render_evidence_context(candidates)) + 1500) / 3.5)


def _prompts(*, question: str, candidates: Sequence[EvidenceCandidate]) -> tuple[str, str]:
    system = """你是 ResearchPilot 的跨论文证据问答助手。只能依据提供的【可用证据】作答，不能使用外部知识。
不要声称“谁更好”、数值可直接比较或因果优劣，除非每篇相关论文都提供了能支持该比较的证据；证据不足时必须拒答。
你必须只输出一个 JSON 对象，不要 Markdown 或解释：
{
  "answer": "中文回答",
  "citations": [{"paper_key": "...", "page_start": 1, "page_end": 1, "context_id": "..."}],
  "claims": [{"claim_id": "C1", "text": "可核验的事实", "citation_context_ids": ["..."], "comparative": false}],
  "refused": false,
  "refusal_reason": null
}
每个 claim 都必须至少引用一个 citation；citation 的 paper_key、页码和 context_id 必须与证据标签逐字对应。
若拒答，citations 和 claims 必须均为 []，并填写 refusal_reason。"""
    user = f"【问题】\n{question.strip()}\n\n【可用证据】\n{render_evidence_context(candidates)}"
    return system, user


def _parse_answer(raw: Mapping[str, Any]) -> P2GeneratedAnswer:
    refused = raw.get("refused")
    answer = raw.get("answer")
    if not isinstance(refused, bool) or not isinstance(answer, str) or not answer.strip():
        raise EvaluationDataError("P2 模型 JSON 缺少合法 answer 或 refused")
    reason = raw.get("refusal_reason")
    if reason is not None and not isinstance(reason, str):
        raise EvaluationDataError("P2 模型 JSON 的 refusal_reason 必须为字符串或 null")
    citations_raw = raw.get("citations")
    claims_raw = raw.get("claims")
    if not isinstance(citations_raw, list) or not isinstance(claims_raw, list):
        raise EvaluationDataError("P2 模型 JSON 的 citations/claims 必须为数组")
    citations = tuple(
        P2Citation(
            paper_key=item.get("paper_key", ""),
            page_start=item.get("page_start"),
            page_end=item.get("page_end", item.get("page_start")),
            context_id=item.get("context_id", ""),
        )
        for item in citations_raw
        if isinstance(item, Mapping)
    )
    if len(citations) != len(citations_raw):
        raise EvaluationDataError("P2 citations[] 必须都是对象")
    claims = tuple(
        P2Claim(
            claim_id=item.get("claim_id", ""),
            text=item.get("text", ""),
            citation_context_ids=tuple(item.get("citation_context_ids", ())),
            comparative=item.get("comparative", False),
        )
        for item in claims_raw
        if isinstance(item, Mapping)
    )
    if len(claims) != len(claims_raw):
        raise EvaluationDataError("P2 claims[] 必须都是对象")
    return P2GeneratedAnswer(
        answer=answer.strip(),
        citations=citations,
        claims=claims,
        refused=refused,
        refusal_reason=(reason.strip() if isinstance(reason, str) else None),
    )


def _cost(*, prompt_tokens: int | None, completion_tokens: int | None, config: P2ExecutionConfig) -> float | None:
    if (
        prompt_tokens is None
        or completion_tokens is None
        or config.input_cost_per_million is None
        or config.output_cost_per_million is None
    ):
        return None
    return (
        prompt_tokens * float(config.input_cost_per_million) + completion_tokens * float(config.output_cost_per_million)
    ) / 1_000_000


def _local_refusal(reason: str) -> P2AnswerExecution:
    return P2AnswerExecution(
        answer=P2GeneratedAnswer(
            answer="当前检索到的论文证据不足，无法可靠完成该跨论文回答。",
            citations=(),
            claims=(),
            refused=True,
            refusal_reason=reason,
        ),
        prompt_tokens=0,
        completion_tokens=0,
        generation_latency_ms=0.0,
        cost=0.0,
    )


def generate_p2_answer(
    *,
    question: str,
    selected: Sequence[EvidenceCandidate],
    required_paper_keys: Sequence[str],
    decision: RetryThenRefuseDecision | None,
    config: P2ExecutionConfig,
    client: StructuredAnswerClient,
) -> P2AnswerExecution:
    """一次受限调用；Gate 拒绝或上下文超预算时本地拒答且不计模型调用。"""
    if decision is not None and decision.refused:
        return _local_refusal(decision.refusal_reason or "insufficient_evidence")
    if not selected:
        return _local_refusal("scope_insufficient_or_no_context")
    estimated = estimate_p2_prompt_tokens(question=question, candidates=selected)
    if estimated > config.max_input_tokens:
        return _local_refusal("capacity_failure:prompt_token_limit")
    system_prompt, user_prompt = _prompts(question=question, candidates=selected)
    started = time.perf_counter()
    completion: ModelCompletion = client.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=config.max_output_tokens,
        temperature=float(config.temperature),
    )
    latency = (time.perf_counter() - started) * 1000
    prompt_tokens = completion.prompt_tokens if completion.prompt_tokens is not None else estimated
    try:
        answer = _parse_answer(_extract_json(completion.text))
    except Exception as error:
        result = _local_refusal(f"model_output_invalid:{type(error).__name__}")
        return P2AnswerExecution(
            answer=result.answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion.completion_tokens,
            generation_latency_ms=latency,
            cost=_cost(prompt_tokens=prompt_tokens, completion_tokens=completion.completion_tokens, config=config),
            validation_violations=(f"model_output_invalid:{type(error).__name__}",),
        )
    if decision is not None:
        validation = validate_claim_citations(
            answer=answer,
            decision=decision,
            required_paper_keys=required_paper_keys,
        )
        if not validation.valid:
            result = _local_refusal("claim_citation_protocol_violation")
            return P2AnswerExecution(
                answer=result.answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion.completion_tokens,
                generation_latency_ms=latency,
                cost=_cost(prompt_tokens=prompt_tokens, completion_tokens=completion.completion_tokens, config=config),
                validation_violations=validation.violations,
            )
    return P2AnswerExecution(
        answer=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion.completion_tokens,
        generation_latency_ms=latency,
        cost=_cost(prompt_tokens=prompt_tokens, completion_tokens=completion.completion_tokens, config=config),
    )
