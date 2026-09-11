from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "validate_product_trust_regression.py"
spec = importlib.util.spec_from_file_location("trust_validator", SCRIPT)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


def test_trust_dataset_distribution_and_routes_are_frozen() -> None:
    cases = validator.load_cases(PROJECT_ROOT / "eval" / "datasets" / "product_trust_regression_v0.3.jsonl")
    validator.validate_cases(cases)
    rows = validator.dry_run(cases)
    assert len(rows) == 18
    assert all(row["route_matches"] for row in rows)
    indexed = {case["id"]: case for case in cases}
    assert indexed["TR009"]["expected_route"] == "multi_section"
    assert indexed["TR011"]["required_evidence_roles"] == ["problem", "method"]
    assert indexed["TR012"]["failure_turn_policy"]["eligible_for_memory"] is False
    assert indexed["TR017"]["expected_refused"] is True


def test_quantitative_semantics_reject_swapped_values_and_concepts() -> None:
    cases = validator.load_cases(PROJECT_ROOT / "eval" / "datasets" / "product_trust_regression_v0.3.jsonl")
    indexed = {case["id"]: case for case in cases}
    correct = {
        "comparisons": indexed["TR003"]["expected_comparisons"],
        "concept_mappings": indexed["TR003"]["concept_mappings"],
        "expressions": [],
    }
    assert validator.validate_structured_result(indexed["TR003"], correct) == ()

    swapped = dict(correct)
    swapped_comparisons = [dict(item) for item in correct["comparisons"]]
    swapped_comparisons[1]["value_a"], swapped_comparisons[1]["value_b"] = (
        swapped_comparisons[1]["value_b"],
        swapped_comparisons[1]["value_a"],
    )
    swapped["comparisons"] = swapped_comparisons
    swapped["concept_mappings"] = {"-High": "remove_low_level_retrieval"}

    errors = validator.validate_structured_result(indexed["TR003"], swapped)
    assert any(error.startswith("missing_or_misattributed_comparison") for error in errors)
    assert "concept_mapping_mismatch:-High" in errors
