"""P2-B3 的 Claim—Citation—Context 协议校验。

模型输出的页码、论文名和 Context ID 都被视为不可信输入；只有它们能映射到
本次 Coverage Gate 允许的候选，且跨论文比较 Claim 覆盖全部目标论文时，才可
交给后续回答运行器记录为有效答案。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .evidence_control import EvidenceCandidate, RetryThenRefuseDecision
from .schemas import SchemaValidationError


@dataclass(frozen=True)
class P2Citation:
    paper_key: str
    page_start: int
    page_end: int
    context_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.paper_key, str) or not self.paper_key.strip():
            raise SchemaValidationError("P2Citation.paper_key 不能为空。")
        if not isinstance(self.context_id, str) or not self.context_id.strip():
            raise SchemaValidationError("P2Citation.context_id 不能为空。")
        if (
            not isinstance(self.page_start, int)
            or not isinstance(self.page_end, int)
            or self.page_start < 1
            or self.page_end < self.page_start
        ):
            raise SchemaValidationError("P2Citation 页码范围非法。")


@dataclass(frozen=True)
class P2Claim:
    claim_id: str
    text: str
    citation_context_ids: tuple[str, ...]
    comparative: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id.strip():
            raise SchemaValidationError("P2Claim.claim_id 不能为空。")
        if not isinstance(self.text, str) or not self.text.strip():
            raise SchemaValidationError("P2Claim.text 不能为空。")
        if not self.citation_context_ids or any(
            not isinstance(item, str) or not item.strip() for item in self.citation_context_ids
        ):
            raise SchemaValidationError("每个 P2 Claim 必须引用至少一个 Context ID。")
        if len(set(self.citation_context_ids)) != len(self.citation_context_ids):
            raise SchemaValidationError("同一个 Claim 的 Context ID 不能重复。")
        if not isinstance(self.comparative, bool):
            raise SchemaValidationError("comparative 必须为布尔值。")


@dataclass(frozen=True)
class P2GeneratedAnswer:
    answer: str
    citations: tuple[P2Citation, ...]
    claims: tuple[P2Claim, ...]
    refused: bool
    refusal_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise SchemaValidationError("P2GeneratedAnswer.answer 不能为空。")
        if any(not isinstance(item, P2Citation) for item in self.citations):
            raise SchemaValidationError("citations 必须由 P2Citation 组成。")
        if any(not isinstance(item, P2Claim) for item in self.claims):
            raise SchemaValidationError("claims 必须由 P2Claim 组成。")
        if self.refused:
            if not isinstance(self.refusal_reason, str) or not self.refusal_reason.strip():
                raise SchemaValidationError("拒答必须提供 refusal_reason。")
            if self.citations or self.claims:
                raise SchemaValidationError("拒答不得附加 citation 或 Claim。")
        elif self.refusal_reason is not None:
            raise SchemaValidationError("非拒答不能提供 refusal_reason。")


@dataclass(frozen=True)
class ClaimCitationValidation:
    valid: bool
    violations: tuple[str, ...]


def validate_claim_citations(
    *,
    answer: P2GeneratedAnswer,
    decision: RetryThenRefuseDecision,
    required_paper_keys: Sequence[str],
) -> ClaimCitationValidation:
    """验证 G2：Claim、Citation 与通过 Gate 的 Context 之间的运行时一致性。"""
    required = tuple(dict.fromkeys(required_paper_keys))
    if not required:
        raise SchemaValidationError("G2 校验至少需要一篇目标论文。")
    allowed: dict[str, EvidenceCandidate] = {item.result.chunk.chunk_id: item for item in decision.selected}
    violations: list[str] = []
    if decision.refused:
        if not answer.refused:
            violations.append("gate_refused_but_answer_not_refused")
        return ClaimCitationValidation(not violations, tuple(violations))
    if answer.refused:
        violations.append("gate_passed_but_answer_refused")
        return ClaimCitationValidation(False, tuple(violations))
    if not answer.citations:
        violations.append("answer_missing_citations")
    if not answer.claims:
        violations.append("answer_missing_claims")

    citation_by_context: dict[str, P2Citation] = {}
    for citation in answer.citations:
        candidate = allowed.get(citation.context_id)
        if candidate is None:
            violations.append(f"unknown_context_id:{citation.context_id}")
            continue
        chunk = candidate.result.chunk
        if citation.paper_key != candidate.paper_key:
            violations.append(f"citation_paper_mismatch:{citation.context_id}")
        if citation.page_start < chunk.page_start or citation.page_end > chunk.page_end:
            violations.append(f"citation_page_out_of_context:{citation.context_id}")
        if citation.context_id in citation_by_context:
            violations.append(f"duplicate_citation_context:{citation.context_id}")
        citation_by_context[citation.context_id] = citation

    for claim in answer.claims:
        missing_contexts = [
            context_id for context_id in claim.citation_context_ids if context_id not in citation_by_context
        ]
        if missing_contexts:
            violations.append(f"claim_missing_citation:{claim.claim_id}")
            continue
        if claim.comparative:
            claim_papers = {citation_by_context[context_id].paper_key for context_id in claim.citation_context_ids}
            missing_papers = set(required) - claim_papers
            if missing_papers:
                violations.append(
                    f"comparative_claim_missing_papers:{claim.claim_id}:{','.join(sorted(missing_papers))}"
                )
    return ClaimCitationValidation(not violations, tuple(violations))
