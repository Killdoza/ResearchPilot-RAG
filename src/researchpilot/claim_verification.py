"""无需额外模型调用的 Claim 数字忠实度校验。"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_CITATION_MARKER_RE = re.compile(r"[\[【（(]\s*证据\s*\d+(?:\s*[,，、]\s*\d+)*\s*[\]】）)]")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<number>[+-]?\d[\d, ]*(?:\.\d+)?)\s*"
    r"(?P<scale>[kKmM](?![A-Za-z]))?\s*"
    r"(?P<unit>%|％|tokens?|ms|毫秒|秒)?",
    re.IGNORECASE,
)


def _canonical_decimal(number: str, scale: str, unit: str) -> str | None:
    compact = number.replace(",", "").replace(" ", "")
    try:
        value = Decimal(compact)
    except InvalidOperation:
        return None
    lowered_scale = scale.lower()
    if lowered_scale == "k":
        value *= 1_000
    elif lowered_scale == "m":
        value *= 1_000_000
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    lowered_unit = unit.lower()
    suffix = {
        "%": "%",
        "％": "%",
        "token": "tokens",
        "tokens": "tokens",
        "ms": "ms",
        "毫秒": "ms",
        "秒": "s",
    }.get(lowered_unit, "")
    return f"{normalized}{suffix}"


def extract_numeric_values(text: str) -> tuple[str, ...]:
    """提取并标准化值得核验的数字，忽略列表序号和证据编号。"""
    cleaned = _CITATION_MARKER_RE.sub("", text)
    values: list[str] = []
    for match in _NUMBER_RE.finditer(cleaned):
        raw_number = match.group("number").strip()
        scale = (match.group("scale") or "").strip()
        unit = (match.group("unit") or "").strip()
        # 单个无单位的一位整数通常只是列表序号；不把它当实验数字。
        digits = raw_number.replace(",", "").replace(" ", "").replace(".", "")
        if not scale and not unit and "." not in raw_number and len(digits) < 2:
            continue
        canonical = _canonical_decimal(raw_number, scale, unit)
        if canonical and canonical not in values:
            values.append(canonical)
    return tuple(values)


def unsupported_numeric_values(claim_text: str, evidence_texts: tuple[str, ...]) -> tuple[str, ...]:
    """返回 Claim 中未能在其绑定证据中找到的数字。"""
    claim_values = extract_numeric_values(claim_text)
    evidence_values = {value for evidence in evidence_texts for value in extract_numeric_values(evidence)}
    return tuple(value for value in claim_values if value not in evidence_values)
