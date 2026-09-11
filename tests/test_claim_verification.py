"""Deterministic numeric-fidelity normalization tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.claim_verification import (  # noqa: E402
    extract_numeric_values,
    unsupported_numeric_values,
)


def test_percentage_spacing_and_thousands_are_normalized() -> None:
    assert extract_numeric_values("胜率 67.6 %") == ("67.6%",)
    assert extract_numeric_values("共有 610,000 tokens") == ("610000tokens",)


def test_scaled_token_values_are_equivalent() -> None:
    assert extract_numeric_values("共有 1.2M tokens") == ("1200000tokens",)
    assert extract_numeric_values("共有 1,200,000 tokens") == ("1200000tokens",)


def test_units_and_signs_prevent_false_matches() -> None:
    assert unsupported_numeric_values("延迟为 100 ms", ("上下文长度为 100 tokens",)) == ("100ms",)
    assert unsupported_numeric_values("变化为 +3.2%", ("变化为 -3.2%",)) == ("3.2%",)


def test_evidence_numbers_and_short_list_markers_are_ignored() -> None:
    assert extract_numeric_values("1. 方法说明。[证据 12]") == ()
