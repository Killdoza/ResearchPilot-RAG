"""将回答文本中的证据编号对齐到本轮 Citation。"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .claim_verification import extract_numeric_values, unsupported_numeric_values
from .schemas import AnswerClaim, Citation

_CANONICAL_REFERENCE_RE = re.compile(r"\[证据\s*([0-9\s,，、]+)\]", re.IGNORECASE)
_EVIDENCE_BRACKET_RE = re.compile(
    r"[\[【［]\s*证据\s*([0-9\s,，、]+)\s*[\]】］]",
    re.IGNORECASE,
)
_EVIDENCE_PAREN_RE = re.compile(
    r"[（(]\s*证据\s*([0-9\s,，、]+)\s*[）)]",
    re.IGNORECASE,
)
_EVIDENCE_WITH_PAGE_RE = re.compile(
    r"[\[【［]\s*证据\s*(\d+)\s*[|｜·,，;；]\s*第?\s*\d+(?:\s*[-—至]\s*\d+)?\s*页?\s*[\]】］]",
    re.IGNORECASE,
)
_EVIDENCE_COMPOUND_WITH_PAGE_RE = re.compile(
    r"[\[【［]\s*证据\s*\d+[^\]】］\r\n]*?第?\s*\d+(?:\s*[-—至]\s*\d+)?\s*页"
    r"[^\]】］\r\n]*?[\]】］]",
    re.IGNORECASE,
)
_EVIDENCE_COMPOUND_PAREN_RE = re.compile(
    r"[（(]\s*证据\s*\d+[^）)\r\n]*?[）)]",
    re.IGNORECASE,
)
_EVIDENCE_BARE_RE = re.compile(
    r"(?<![\[【［])证据\s*(\d+(?:\s*[,，、和及]\s*(?:证据\s*)?\d+)*)"
    r"(?!\s*[-—至]\s*\d)",
    re.IGNORECASE,
)
_SUP_REFERENCE_RE = re.compile(
    r"<sup>\s*((?:\[\s*\d+(?:\s*[,，、]\s*\d+)*\s*\]\s*)+)\s*</sup>",
    re.IGNORECASE,
)
# Plain ``[1]`` is accepted only at a natural claim boundary.  This keeps an
# inline bibliography marker such as ``Smith [1] 提出`` untouched while still
# accepting the formats observed from DeepSeek: ``句子。[1][5]``.
_PLAIN_REFERENCE_SEQUENCE_RE = re.compile(
    r"((?:\[\s*\d+(?:\s*[,，、]\s*\d+)*\s*\]\s*)+)"
    r"(?=(?:[。！？；，,.!?;:]|\s*$))",
    re.MULTILINE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.、)])\s*")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_BOLD_LABEL_RE = re.compile(r"^\s*\*\*[^*]+\*\*\s*[：:]?\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
_CITATION_ONLY_RE = re.compile(r"^\s*(?:\[证据\s*[0-9\s,，、]+\]\s*)+[。.]?\s*$")
_NON_FACTUAL_META_RE = re.compile(
    r"^(?:根据用户要求|根据(?:所提供的)?论文证据[，,]\s*(?:我可以|以下)|"
    r"以下(?:分别)?(?:陈述|说明|列出)|回答如下)[，,:：\s]?",
)
_EPISTEMIC_BOUNDARY_RE = re.compile(
    r"(?:当前|现有|所提供的|本轮检索)?证据(?:中|范围内)?[^。；\n]*"
    r"(?:不足|无法(?:支持|确认|回答)|未给出|没有(?:提供|包含)|未包含|不充分)",
)


def _claim_lines(text: str) -> tuple[str, ...]:
    claims: list[str] = []
    for raw_line in text.splitlines():
        if _HEADING_RE.match(raw_line) or raw_line.lstrip().startswith(">"):
            continue
        if _BOLD_LABEL_RE.match(raw_line) or _TABLE_SEPARATOR_RE.match(raw_line):
            continue
        line = _BULLET_RE.sub("", _HEADING_RE.sub("", raw_line)).strip()
        if _CITATION_ONLY_RE.match(line) or _NON_FACTUAL_META_RE.match(line):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
            if cells and all(not re.search(r"\d", cell) for cell in cells):
                continue
        if not line or line.endswith("：") or line.endswith(":"):
            continue
        if len(line) < 4:
            continue
        claims.append(line)
    if not claims and text.strip():
        claims.append(text.strip())
    return tuple(claims)


def _valid_indexes(raw: str, citation_count: int) -> tuple[int, ...]:
    values: list[int] = []
    for value in re.findall(r"\d+", raw):
        index = int(value)
        if 1 <= index <= citation_count and index not in values:
            values.append(index)
    return tuple(values)


def _canonical_marker(indexes: Sequence[int]) -> str:
    return f"[证据 {'、'.join(str(index) for index in indexes)}]"


def normalize_citation_markers(text: str, citation_count: int) -> str:
    """Normalize supported model citation styles to ``[证据 1、2]``.

    Only indexes that exist in the current answer are accepted.  Four-digit
    bibliography years and plain numeric brackets embedded inside prose are
    therefore not silently turned into product citations.
    """
    if citation_count < 1 or not text:
        return text

    def replace_evidence(match: re.Match[str]) -> str:
        indexes = _valid_indexes(match.group(1), citation_count)
        return _canonical_marker(indexes) if indexes else match.group(0)

    def replace_compound_evidence(match: re.Match[str]) -> str:
        indexes = _valid_indexes(match.group(0), citation_count)
        return _canonical_marker(indexes) if indexes else match.group(0)

    def replace_plain(match: re.Match[str]) -> str:
        indexes = _valid_indexes(match.group(1), citation_count)
        return _canonical_marker(indexes) if indexes else match.group(0)

    normalized = _EVIDENCE_COMPOUND_WITH_PAGE_RE.sub(replace_compound_evidence, text)
    normalized = _EVIDENCE_COMPOUND_PAREN_RE.sub(replace_compound_evidence, normalized)
    normalized = _EVIDENCE_WITH_PAGE_RE.sub(replace_evidence, normalized)
    normalized = _EVIDENCE_BRACKET_RE.sub(replace_evidence, normalized)
    normalized = _EVIDENCE_PAREN_RE.sub(replace_evidence, normalized)
    normalized = _EVIDENCE_BARE_RE.sub(replace_evidence, normalized)
    normalized = _SUP_REFERENCE_RE.sub(replace_plain, normalized)
    return _PLAIN_REFERENCE_SEQUENCE_RE.sub(replace_plain, normalized)


def ensure_annotation_citation_markers(text: str, citation_count: int) -> str:
    """Make the paper side of a user-note comparison explicitly traceable.

    The note itself remains visibly user-authored and receives no paper
    citation.  Statements under ``论文原文`` or ``二者关系`` receive the current
    evidence set only when the model omitted a marker.  This is a structural
    attribution fallback; normal numeric verification still runs afterwards.
    """
    if citation_count < 1 or not text:
        return text
    normalized = normalize_citation_markers(text, citation_count)
    observed_indexes: list[int] = []
    for line in normalized.splitlines():
        for index in referenced_citation_indexes(line, citation_count):
            if index not in observed_indexes:
                observed_indexes.append(index)
    fallback_indexes = tuple(observed_indexes[:3]) or tuple(range(1, min(citation_count, 3) + 1))
    marker = _canonical_marker(fallback_indexes)
    section: str | None = None
    rendered: list[str] = []
    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if "论文原文" in stripped and (stripped.startswith(("#", "**")) or stripped.endswith(("：", ":"))):
            section = "paper"
        elif "用户批注" in stripped and (stripped.startswith(("#", "**")) or stripped.endswith(("：", ":"))):
            section = "note"
        elif "二者关系" in stripped and (stripped.startswith(("#", "**")) or stripped.endswith(("：", ":"))):
            section = "relationship"
        should_cite = (
            section in {"paper", "relationship"}
            and bool(stripped)
            and not stripped.startswith(("#", ">", "---"))
            and not _BOLD_LABEL_RE.match(stripped)
            and not stripped.endswith(("：", ":"))
            and referenced_citation_indexes(stripped, citation_count) == ()
        )
        is_user_note_line = (
            section == "note"
            and bool(stripped)
            and not stripped.startswith(("#", ">", "---"))
            and not _BOLD_LABEL_RE.match(stripped)
            and "【用户批注】" not in stripped
        )
        if should_cite:
            rendered.append(f"{raw_line.rstrip()} {marker}")
        elif is_user_note_line:
            rendered.append(f"{raw_line.rstrip()} 【用户批注】")
        else:
            rendered.append(raw_line)
    return "\n".join(rendered)


def ensure_factual_citation_markers(text: str, citation_count: int) -> str:
    """Close structurally uncited paper claims before contract validation.

    This function is used only for evidence-backed paper routes.  It does not
    invent new sources: the fallback marker is the union of evidence indexes
    that the model already cited elsewhere in the same answer.  Epistemic
    boundary messages remain uncited and are classified as product metadata.
    """
    if citation_count < 1 or not text:
        return text
    normalized = normalize_citation_markers(text, citation_count)
    observed: list[int] = []
    for line in normalized.splitlines():
        for index in referenced_citation_indexes(line, citation_count):
            if index not in observed:
                observed.append(index)
    if not observed:
        return normalized
    marker = _canonical_marker(tuple(observed))
    rendered: list[str] = []
    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        candidate = _BULLET_RE.sub("", stripped).strip()
        should_cite = (
            bool(candidate)
            and not stripped.startswith(("#", ">", "---"))
            and not _BOLD_LABEL_RE.match(stripped)
            and not _TABLE_SEPARATOR_RE.match(stripped)
            and not candidate.endswith(("：", ":"))
            and not _NON_FACTUAL_META_RE.match(candidate)
            and not _EPISTEMIC_BOUNDARY_RE.search(candidate)
            and referenced_citation_indexes(candidate, citation_count) == ()
        )
        rendered.append(f"{raw_line.rstrip()} {marker}" if should_cite else raw_line)
    return "\n".join(rendered)


def referenced_citation_indexes(line: str, citation_count: int) -> tuple[int, ...]:
    """Return unique, 1-based citation indexes in first-appearance order."""
    normalized = normalize_citation_markers(line, citation_count)
    values: list[int] = []
    for group in _CANONICAL_REFERENCE_RE.findall(normalized):
        for index in _valid_indexes(group, citation_count):
            if index not in values:
                values.append(index)
    return tuple(values)


def retain_used_citations(
    text: str,
    citations: Sequence[Citation],
    *,
    default_to_all_citations: bool = False,
) -> tuple[str, tuple[Citation, ...]]:
    """Prune unused display citations and compact every marker deterministically."""
    normalized = normalize_citation_markers(text, len(citations))
    used = referenced_citation_indexes(normalized, len(citations))
    if not used and default_to_all_citations and citations:
        return normalized, tuple(citations)
    if not used:
        return normalized, ()

    old_to_new = {old: new for new, old in enumerate(used, start=1)}

    def remap(match: re.Match[str]) -> str:
        indexes = _valid_indexes(match.group(1), len(citations))
        mapped = tuple(old_to_new[index] for index in indexes if index in old_to_new)
        return _canonical_marker(mapped) if mapped else ""

    compacted_text = _CANONICAL_REFERENCE_RE.sub(remap, normalized)
    compacted_citations = tuple(citations[index - 1] for index in used)
    return compacted_text, compacted_citations


def format_citation_markers_with_pages(
    text: str,
    citations: Sequence[Citation],
) -> str:
    """Enrich display markers with page numbers without changing cached text."""
    if not citations:
        return text

    normalized = normalize_citation_markers(text, len(citations))

    def replace_marker(match: re.Match[str]) -> str:
        indexes = _valid_indexes(match.group(1), len(citations))
        if not indexes:
            return match.group(0)
        labels: list[str] = []
        for index in indexes:
            citation = citations[index - 1]
            page = (
                f"第 {citation.page_start} 页"
                if citation.page_start == citation.page_end
                else f"第 {citation.page_start}-{citation.page_end} 页"
            )
            labels.append(f"证据 {index} · {page}")
        return "；".join(labels)

    return _CANONICAL_REFERENCE_RE.sub(replace_marker, normalized)


def align_claims(
    text: str,
    citations: Sequence[Citation],
    *,
    default_to_all_citations: bool = False,
) -> tuple[AnswerClaim, ...]:
    """解析回答 Claim，并验证引用范围和数字是否由对应证据支持。"""
    aligned: list[AnswerClaim] = []
    for position, line in enumerate(_claim_lines(text), start=1):
        referenced_indexes = referenced_citation_indexes(line, len(citations))
        if not referenced_indexes and default_to_all_citations and citations:
            referenced_indexes = tuple(range(1, len(citations) + 1))
        citation_ids = tuple(f"e{index}" for index in referenced_indexes)
        evidence = tuple(citations[index - 1].evidence for index in referenced_indexes)
        numeric_values = extract_numeric_values(line)
        if "【用户批注】" in line or _EPISTEMIC_BOUNDARY_RE.search(line):
            status = "not_applicable"
        elif not citation_ids:
            status = "unsupported"
        elif unsupported_numeric_values(line, evidence):
            status = "numeric_unsupported"
        else:
            status = "supported"
        aligned.append(
            AnswerClaim(
                claim_id=f"c{position}",
                text=line,
                citation_ids=citation_ids,
                support_status=status,
                numeric_values=numeric_values,
            )
        )
    return tuple(aligned)
