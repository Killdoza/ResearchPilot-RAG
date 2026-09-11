"""Deterministic structural validation for full-paper summaries.

The model is allowed to draft a summary once.  This module then enforces the
visible contract locally: required sections must exist, empty Markdown tables
and orphan headings are removed, and factual sections without a citation
marker are replaced by an explicit evidence-gap notice.  No second model call
is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION_RE = re.compile(r"(?:\[证据\s*\d+(?:\s*[,，、]\s*\d+)*\]|证据\s*\d+)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
_BOLD_LABEL_RE = re.compile(r"^\s*\*\*[^*]+\*\*\s*[：:]?\s*$")
_CITATION_ONLY_RE = re.compile(r"^\s*(?:\[证据\s*[0-9\s,，、]+\]\s*)+[。.]?\s*$")
_EMPTY_NOTICE = "> 当前检索证据不足，无法形成可核验内容。"


_REQUIRED_SECTIONS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "overview": (
        ("研究问题", ("研究问题", "问题", "research question")),
        ("核心方法", ("核心方法", "方法", "method")),
        ("实验设置", ("实验设置", "实验", "experimental setup")),
        ("主要结果", ("主要结果", "实验结果", "results")),
        ("结论与局限", ("结论与局限", "结论", "局限", "limitations")),
    ),
    "motivation_to_results": (
        ("研究动机", ("研究动机", "动机", "motivation")),
        ("方法演进", ("方法演进", "方法设计", "核心方法", "方法", "method")),
        ("实验设置", ("实验设置", "实验", "experimental setup")),
        ("实验结果", ("实验结果", "实验与结果", "主要结果", "主要定量结果", "定量结果", "results")),
        ("结论", ("结论", "conclusion")),
    ),
    "contributions_and_limitations": (
        ("核心贡献", ("核心贡献", "主要贡献", "贡献", "contributions")),
        ("证据", ("证据依据", "支持证据", "证据", "evidence")),
        ("局限", ("局限", "限制", "不足", "limitations")),
        (
            "未明确说明的边界",
            ("未明确说明的边界", "未报告边界", "未报告", "证据边界", "boundaries"),
        ),
    ),
}

_REQUIRED_SECTIONS["motivation_to_results"] = (
    ("研究动机", ("研究动机", "研究问题", "动机", "motivation")),
    ("方法演进", ("方法演进", "方法设计", "核心方法", "方法", "method")),
    ("实验设置", ("实验设置", "实验与结果", "实验", "experimental setup")),
    (
        "实验结果",
        ("实验结果", "实验与结果", "主要结果", "主要定量结果", "定量结果", "results"),
    ),
    ("结论", ("结论", "conclusion")),
)


@dataclass(frozen=True)
class SummaryValidationResult:
    text: str
    valid: bool
    repaired: bool
    issues: tuple[str, ...]


def _normalize_heading(value: str) -> str:
    return re.sub(r"[\s：:、，,。.!！?？*_`]+", "", value).casefold()


def _section_blocks(text: str) -> tuple[str, list[tuple[str, list[str]]]]:
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in text.strip().splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current_title is None:
                preamble.extend(current_lines)
            else:
                sections.append((current_title, current_lines))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line.rstrip())
    if current_title is None:
        preamble.extend(current_lines)
    else:
        sections.append((current_title, current_lines))
    return "\n".join(preamble).strip(), sections


def _remove_empty_tables(lines: list[str]) -> tuple[list[str], int]:
    """Remove every header-only table, not only the first table in a section."""
    remove_indexes: set[int] = set()
    empty_count = 0
    for index, line in enumerate(lines):
        if not _TABLE_SEPARATOR_RE.match(line) or index == 0:
            continue
        following = index + 1
        while following < len(lines) and not lines[following].strip():
            following += 1
        has_data_row = following < len(lines) and lines[following].lstrip().startswith("|")
        if has_data_row:
            continue
        empty_count += 1
        remove_indexes.add(index)
        if lines[index - 1].lstrip().startswith("|"):
            remove_indexes.add(index - 1)
    return [line for index, line in enumerate(lines) if index not in remove_indexes], empty_count


def _substantive(lines: list[str]) -> bool:
    return any(
        line.strip() and not line.lstrip().startswith(">") and not _TABLE_SEPARATOR_RE.match(line) for line in lines
    )


def _marker_from_lines(lines: list[str]) -> str | None:
    for line in lines:
        matches = _CITATION_RE.findall(line)
        if matches:
            return matches[-1]
    return None


def _propagate_table_citations(lines: list[str]) -> list[str]:
    """Attach the section's explicit marker to factual table rows.

    Models frequently cite the sentence immediately before or after a table
    instead of each row.  The prompt already supplies numbered evidence; this
    local normalization merely makes the row-to-evidence relationship visible.
    Numeric fidelity is still checked later against the cited evidence, so a
    wrong value is removed rather than legitimized by propagation.
    """
    marker = _marker_from_lines(lines)
    if marker is None:
        return lines
    rendered: list[str] = []
    separator_seen = False
    for line in lines:
        if _TABLE_SEPARATOR_RE.match(line):
            separator_seen = True
            rendered.append(line)
            continue
        is_table_row = separator_seen and line.strip().startswith("|") and line.strip().endswith("|")
        if is_table_row and _CITATION_RE.search(line) is None:
            rendered.append(f"{line.rstrip()} {marker}")
        else:
            rendered.append(line)
        if separator_seen and line.strip() and not line.strip().startswith("|"):
            separator_seen = False
    return rendered


def _line_requires_citation(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith((">", "---")):
        return False
    if _TABLE_SEPARATOR_RE.match(stripped) or _BOLD_LABEL_RE.match(stripped):
        return False
    if _CITATION_ONLY_RE.match(stripped) or stripped.endswith(("：", ":")):
        return False
    if stripped.startswith("|") and stripped.endswith("|"):
        cells = tuple(cell.strip() for cell in stripped.strip("|").split("|"))
        return any(re.search(r"\d", cell) for cell in cells)
    return len(stripped) >= 4


def _remove_uncited_factual_lines(title: str, lines: list[str], issues: list[str]) -> list[str]:
    """Drop untraceable generated facts while preserving layout labels.

    A section-level citation does not support an unrelated uncited bullet.
    Keeping such a line would make the product look complete while silently
    breaking the claim contract.  The caller inserts an explicit evidence-gap
    notice when no cited fact remains.
    """
    rendered: list[str] = []
    for index, line in enumerate(lines, start=1):
        if _line_requires_citation(line) and _CITATION_RE.search(line) is None:
            issues.append(f"uncited_line:{title}:{index}")
            continue
        rendered.append(line)
    return rendered


def validate_and_repair_summary(text: str, *, intent: str) -> SummaryValidationResult:
    """Return a locally repaired summary and a complete issue list."""
    required = _REQUIRED_SECTIONS.get(intent, _REQUIRED_SECTIONS["overview"])
    preamble, sections = _section_blocks(text)
    issues: list[str] = []
    rendered: list[str] = []
    seen: set[str] = set()

    for title, lines in sections:
        lines = _propagate_table_citations(lines)
        lines, empty_table_count = _remove_empty_tables(lines)
        issues.extend(f"empty_table:{title}" for _ in range(empty_table_count))
        lines = _remove_uncited_factual_lines(title, lines, issues)
        normalized = _normalize_heading(title)
        matches = {label for label, aliases in required for alias in aliases if _normalize_heading(alias) in normalized}
        seen.update(matches)
        content = "\n".join(lines).strip()
        if not content or (_substantive(lines) and _CITATION_RE.search(content) is None):
            issues.append(f"empty_section:{title}")
            content = _EMPTY_NOTICE
        rendered.extend((f"## {title}", "", content, ""))

    for label, _aliases in required:
        if label not in seen:
            issues.append(f"missing_section:{label}")
            rendered.extend((f"## {label}", "", _EMPTY_NOTICE, ""))

    # A prose-only model answer has no stable section boundary.  Keep any
    # cited preamble under an explicit heading; otherwise discard it rather
    # than preserving an unauditable orphan paragraph.
    if preamble and _CITATION_RE.search(preamble):
        rendered = ["## 补充说明", "", preamble, "", *rendered]
    elif preamble:
        issues.append("uncited_preamble")

    repaired_text = "\n".join(rendered).strip()
    return SummaryValidationResult(
        text=repaired_text,
        valid=not issues,
        repaired=repaired_text != text.strip(),
        issues=tuple(issues),
    )
