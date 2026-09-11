"""ResearchPilot 的 Claim-Citation-Context 回答契约。"""

from __future__ import annotations

from dataclasses import replace

from .evidence_alignment import align_claims, retain_used_citations
from .schemas import Answer

ANSWER_CONTRACT_VERSION = "claim_citation_v2"
_NUMERIC_WARNING = (
    "\n\n> 数字核验提示：部分具体数值未能在其对应证据中逐值匹配，已从本轮回答中移除；请以下方原文证据为准。"
)


def _remove_orphan_introductions(lines: list[str]) -> list[str]:
    """Remove a dangling ``标题：`` left after its only numeric claim vanished."""
    cleaned: list[str] = []
    for position, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not stripped.rstrip("*").endswith(("：", ":")):
            cleaned.append(line)
            continue
        next_line = next(
            (candidate.strip() for candidate in lines[position + 1 :] if candidate.strip()),
            "",
        )
        if not next_line or next_line.startswith(("#", ">")):
            continue
        cleaned.append(line)
    return cleaned


def _remove_numeric_unsupported_lines(text: str, unsupported_texts: set[str]) -> str:
    kept = [
        line
        for line in text.splitlines()
        if not any(
            line.strip() == unsupported or line.strip().endswith(unsupported) for unsupported in unsupported_texts
        )
    ]
    kept = _remove_orphan_introductions(kept)
    return "\n".join(kept).strip() or "当前证据不足以可靠报告具体实验数值。"


def enforce_answer_contract(
    answer: Answer,
    *,
    default_to_all_citations: bool = False,
) -> Answer:
    """绑定 Claim/Citation，并移除未被对应证据支持的定量断言。"""
    # Persisted final artifacts have already crossed this boundary.  Returning
    # them byte-for-byte makes cache hits deterministic and the contract
    # explicitly idempotent.
    if answer.contract_version == ANSWER_CONTRACT_VERSION:
        return answer
    if answer.refused:
        return replace(
            answer,
            claims=(),
            contract_version=ANSWER_CONTRACT_VERSION,
            numeric_fidelity_passed=True,
        )

    base_text = answer.text.replace(_NUMERIC_WARNING, "").rstrip()
    text, citations = retain_used_citations(
        base_text,
        answer.citations,
        default_to_all_citations=default_to_all_citations,
    )
    claims = align_claims(text, citations, default_to_all_citations=default_to_all_citations)
    numeric_failures = {claim.text for claim in claims if claim.support_status == "numeric_unsupported"}
    if numeric_failures:
        text = _remove_numeric_unsupported_lines(text, numeric_failures)
        text, citations = retain_used_citations(
            text,
            citations,
            default_to_all_citations=default_to_all_citations,
        )
        claims = align_claims(
            text,
            citations,
            default_to_all_citations=default_to_all_citations,
        )
    warnings = []
    if numeric_failures:
        warnings.append(_NUMERIC_WARNING)
    return replace(
        answer,
        text=text + "".join(warnings),
        citations=citations,
        claims=claims,
        contract_version=ANSWER_CONTRACT_VERSION,
        numeric_fidelity_passed=not numeric_failures,
    )
