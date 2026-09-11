"""Keep the manual product smoke set small, stable and machine-readable."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_product_regression.py"

spec = importlib.util.spec_from_file_location("product_regression_validator", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


def test_fixed_product_regression_dataset_is_valid() -> None:
    cases = validator.load_cases(PROJECT_ROOT / "eval" / "datasets" / "product_regression_v0.1.jsonl")

    validator.validate_cases(cases)
    assert len(cases) == 12


def test_invalid_regression_route_is_rejected() -> None:
    invalid = [
        {
            "id": f"RP{index:03d}",
            "phase": "fresh",
            "question": "question",
            "fixture": "fixture",
            "expected_route": "not_a_route" if index == 1 else "metadata",
            "expected_outcome": "answer",
            "required_pages": [],
            "notes": "note",
        }
        for index in range(1, 11)
    ]

    try:
        validator.validate_cases(invalid)
    except ValueError as error:
        assert "expected_route" in str(error)
    else:
        raise AssertionError("invalid route must be rejected")
