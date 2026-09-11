"""Validate and dry-run the P2.8.4 product trust regression set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.routing import RuntimePaper, RuntimeRequest, RuntimeRouter  # noqa: E402, I001
from researchpilot.quantitative_answer import build_quantitative_evidence  # noqa: E402
from researchpilot.schemas import BlockType, Chunk, ScoredChunk  # noqa: E402


REQUIRED_CATEGORIES = {
    "quantitative_result": 5,
    "document_summary": 3,
    "multi_page_claim": 3,
    "memory_isolation": 3,
    "annotation_conflict": 2,
    "unanswerable": 2,
}
SCHEMA_VERSION = "product_trust_regression_v0.3"
ALLOWED_STEP_ROUTES = {
    "metadata",
    "document_summary",
    "experimental_setup",
    "experimental_results",
    "document_detail",
    "multi_section",
    "general_chat",
}
COMPARISON_FIELDS = {
    "dataset",
    "metric",
    "method_a",
    "value_a",
    "method_b",
    "value_b",
    "relation",
}
ALLOWED_RELATIONS = {"method_a_higher", "method_b_higher", "equal"}


def load_cases(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {line_number}: invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number}: case must be an object")
        cases.append(payload)
    return cases


def validate_cases(cases: list[dict[str, object]]) -> None:
    if len(cases) != 18:
        raise ValueError("product trust dataset must contain exactly 18 cases")
    identifiers = [case.get("id") for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case IDs must be unique")
    categories = Counter(str(case.get("category")) for case in cases)
    if categories != Counter(REQUIRED_CATEGORIES):
        raise ValueError(f"unexpected category distribution: {dict(categories)}")
    for case in cases:
        if case.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{case.get('id')}: schema_version must be {SCHEMA_VERSION}")
        for key in (
            "id",
            "category",
            "paper_id",
            "question",
            "expected_route",
            "expected_pages",
            "required_numbers",
            "forbidden_claims",
            "execution",
        ):
            if key not in case:
                raise ValueError(f"{case.get('id')}: missing {key}")
        pages = case["expected_pages"]
        if not isinstance(pages, list) or any(not isinstance(page, int) or page < 1 for page in pages):
            raise ValueError(f"{case['id']}: invalid expected_pages")
        steps = case.get("steps")
        if steps is not None:
            if not isinstance(steps, list) or len(steps) < 2:
                raise ValueError(f"{case['id']}: steps must contain at least two actions")
            for step in steps:
                if not isinstance(step, dict):
                    raise ValueError(f"{case['id']}: each step must be an object")
                for field in ("paper_id", "question", "expected_route", "memory_scope"):
                    if not isinstance(step.get(field), str) or not str(step[field]).strip():
                        raise ValueError(f"{case['id']}: step missing {field}")
                if step["expected_route"] not in ALLOWED_STEP_ROUTES:
                    raise ValueError(f"{case['id']}: unsupported step route")
        failure_policy = case.get("failure_turn_policy")
        if failure_policy is not None and (
            not isinstance(failure_policy, dict)
            or failure_policy.get("eligible_for_memory") is not False
            or failure_policy.get("retry_button_required") is not True
        ):
            raise ValueError(f"{case['id']}: invalid failure_turn_policy")
        comparisons = case.get("expected_comparisons", [])
        if not isinstance(comparisons, list):
            raise ValueError(f"{case['id']}: expected_comparisons must be a list")
        for comparison in comparisons:
            if not isinstance(comparison, dict) or set(comparison) != COMPARISON_FIELDS:
                raise ValueError(f"{case['id']}: invalid comparison fields")
            if any(
                not isinstance(comparison[field], str) or not comparison[field].strip() for field in COMPARISON_FIELDS
            ):
                raise ValueError(f"{case['id']}: comparison values must be non-empty strings")
            if comparison["relation"] not in ALLOWED_RELATIONS:
                raise ValueError(f"{case['id']}: invalid comparison relation")
            if not str(comparison["value_a"]).endswith("%") or not str(comparison["value_b"]).endswith("%"):
                raise ValueError(f"{case['id']}: comparison values must preserve percent units")
        mappings = case.get("concept_mappings", {})
        if not isinstance(mappings, dict) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
            for key, value in mappings.items()
        ):
            raise ValueError(f"{case['id']}: invalid concept_mappings")
        expressions = case.get("expected_expressions", [])
        if not isinstance(expressions, list) or any(
            not isinstance(value, str) or not value.strip() for value in expressions
        ):
            raise ValueError(f"{case['id']}: invalid expected_expressions")
        if str(case["id"]) in {f"TR00{index}" for index in range(1, 6)} and not (
            comparisons or mappings or expressions
        ):
            raise ValueError(f"{case['id']}: quantitative trust case lacks structured assertions")


def validate_structured_result(
    case: dict[str, object],
    result: dict[str, object],
) -> tuple[str, ...]:
    """Compare deterministic quantitative output against frozen semantic gold."""
    errors: list[str] = []
    actual_comparisons = result.get("comparisons", [])
    if not isinstance(actual_comparisons, list):
        return ("result.comparisons must be a list",)

    def signature(value: object) -> tuple[str, ...] | None:
        if not isinstance(value, dict) or any(field not in value for field in COMPARISON_FIELDS):
            return None
        return tuple(str(value[field]) for field in sorted(COMPARISON_FIELDS))

    actual_signatures = {item for value in actual_comparisons if (item := signature(value))}
    for expected in case.get("expected_comparisons", []):
        expected_signature = signature(expected)
        if expected_signature not in actual_signatures:
            errors.append(f"missing_or_misattributed_comparison:{expected_signature}")

    actual_mappings = result.get("concept_mappings", {})
    if not isinstance(actual_mappings, dict):
        errors.append("result.concept_mappings must be an object")
    else:
        for term, meaning in dict(case.get("concept_mappings", {})).items():
            if actual_mappings.get(term) != meaning:
                errors.append(f"concept_mapping_mismatch:{term}")

    actual_expressions = result.get("expressions", [])
    if not isinstance(actual_expressions, list):
        errors.append("result.expressions must be a list")
    else:
        for expected in case.get("expected_expressions", []):
            if expected not in actual_expressions:
                errors.append(f"missing_expression:{expected}")
    return tuple(errors)


def dry_run(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    router = RuntimeRouter()
    rows: list[dict[str, object]] = []
    for case in cases:
        paper_id = str(case["paper_id"])
        papers = () if paper_id == "general" else (RuntimePaper(paper_id, f"{paper_id}.pdf"),)
        decision = router.route(
            RuntimeRequest(
                question=str(case["question"]),
                user_id="trust_dry_run",
                session_id="trust_dry_run",
                available_papers=papers,
                selected_paper_ids=(paper_id,) if papers else (),
            )
        )
        rows.append(
            {
                "id": case["id"],
                "expected_route": case["expected_route"],
                "actual_route": decision.route.value,
                "route_matches": decision.route.value == case["expected_route"],
                "expected_pages": case["expected_pages"],
            }
        )
    return rows


def _recorded_scored_chunks(record: dict[str, object]) -> tuple[ScoredChunk, ...]:
    actual = record.get("actual")
    if not isinstance(actual, dict):
        return ()
    citations = actual.get("citations")
    if not isinstance(citations, list):
        return ()
    results: list[ScoredChunk] = []
    for position, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict):
            continue
        try:
            section_path = citation.get("section_path", [])
            chunk = Chunk.create(
                paper_id=str(citation["paper_id"]),
                parent_id=None,
                page_start=int(citation["page_start"]),
                page_end=int(citation["page_end"]),
                section_path=tuple(str(item) for item in section_path),
                content=str(citation["evidence"]),
                block_types=(BlockType.OTHER,),
                chunker_version="recorded_acceptance",
                occurrence_index=position,
            )
        except (KeyError, TypeError, ValueError):
            continue
        results.append(ScoredChunk(chunk=chunk, source_name=str(citation["paper_id"])))
    return tuple(results)


def validate_recorded_quantitative_results(
    cases: list[dict[str, object]],
    result_path: Path,
) -> list[dict[str, object]]:
    """Replay structured extraction on recorded citations without any model call."""
    recorded = {
        str(item["id"]): item
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in (json.loads(line),)
        if isinstance(item, dict) and "id" in item
    }
    rows: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["id"])
        if case_id not in {f"TR00{index}" for index in range(1, 6)}:
            continue
        results = _recorded_scored_chunks(recorded.get(case_id, {}))
        structured = build_quantitative_evidence(str(case["question"]), results)
        errors = validate_structured_result(case, structured.as_regression_record())
        rows.append(
            {
                "id": case_id,
                "citation_count": len(results),
                "structured_assertions_passed": not errors,
                "errors": list(errors),
                "structured": structured.as_regression_record(),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "eval" / "datasets" / "product_trust_regression_v0.3.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recorded-results", type=Path)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    validate_cases(cases)
    print(f"validated_cases={len(cases)}")
    if args.recorded_results is not None:
        rows = validate_recorded_quantitative_results(cases, args.recorded_results)
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        failures = [row for row in rows if not row["structured_assertions_passed"]]
        print(f"structured_failures={len(failures)} model_calls=0")
        return 1 if failures else 0
    if args.dry_run:
        rows = dry_run(cases)
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        mismatches = [row for row in rows if not row["route_matches"]]
        print(f"route_mismatches={len(mismatches)} model_calls=0")
        return 1 if mismatches else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
