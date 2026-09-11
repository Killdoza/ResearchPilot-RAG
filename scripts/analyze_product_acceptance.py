"""Analyze and optionally merge product-rc2 real acceptance records."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.generation import infer_summary_intent  # noqa: E402
from researchpilot.quantitative_answer import build_quantitative_evidence  # noqa: E402
from researchpilot.summary_validation import validate_and_repair_summary  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "eval" / "datasets" / "product_trust_regression_v0.3.jsonl"
TRUST_VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_product_trust_regression.py"
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)\]}>]+|\b[a-z0-9.-]+\.(?:ai|com|io|org)/[^\s)\]}>]+", re.I)
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")


@dataclass(frozen=True)
class AcceptanceSummary:
    case_count: int
    gate_failures: tuple[str, ...]
    expected_page_hits: int
    expected_page_total: int
    prompt_tokens: int
    completion_tokens: int
    model_calls: int
    cache_hits: int
    estimated_cost: float | None

    @property
    def passed(self) -> bool:
        return not self.gate_failures


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_records(base: Path, overrides: tuple[Path, ...]) -> tuple[dict[str, object], ...]:
    merged = {str(record["id"]): record for record in _load_jsonl(base)}
    for path in overrides:
        for record in _load_jsonl(path):
            merged[str(record["id"])] = record
    return tuple(merged[key] for key in sorted(merged))


def _load_trust_validator() -> object:
    spec = importlib.util.spec_from_file_location("product_trust_validator", TRUST_VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 product trust validator。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _empty_table(text: str) -> bool:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not _TABLE_SEPARATOR_RE.match(line) or index == 0:
            continue
        data_rows = []
        for following in lines[index + 1 :]:
            if not following.strip():
                continue
            if not following.strip().startswith("|"):
                break
            data_rows.append(following)
        if not data_rows:
            return True
    return False


def analyze_records(
    records: tuple[dict[str, object], ...],
    dataset: tuple[dict[str, object], ...],
) -> AcceptanceSummary:
    failures: list[str] = []
    indexed_records = {str(record["id"]): record for record in records}
    validator = _load_trust_validator()
    expected_page_hits = 0
    expected_page_total = 0
    prompt_tokens = 0
    completion_tokens = 0
    model_calls = 0
    cache_hits = 0
    costs: list[float] = []

    if len(indexed_records) != len(dataset):
        failures.append(f"case_count:{len(indexed_records)}!={len(dataset)}")

    for case in dataset:
        case_id = str(case["id"])
        record = indexed_records.get(case_id)
        if record is None:
            failures.append(f"{case_id}:missing_record")
            continue
        if record.get("runner_error") is not None:
            failures.append(f"{case_id}:runner_error")
            continue
        actual = record.get("actual")
        if not isinstance(actual, dict):
            failures.append(f"{case_id}:missing_actual")
            continue
        if actual.get("route") != case.get("expected_route"):
            failures.append(f"{case_id}:route_mismatch")
        answer = str(actual.get("answer", ""))
        for forbidden in case.get("forbidden_claims", []):
            if str(forbidden) and str(forbidden) in answer:
                failures.append(f"{case_id}:forbidden_claim:{forbidden}")

        citations = actual.get("citations", [])
        citations = citations if isinstance(citations, list) else []
        target_ids = set(actual.get("target_paper_ids", []))
        if target_ids and any(citation.get("paper_id") not in target_ids for citation in citations):
            failures.append(f"{case_id}:wrong_paper_citation")
        if actual.get("route") == "general_chat" and citations:
            failures.append(f"{case_id}:general_chat_has_citation")
        if actual.get("refused") and citations:
            failures.append(f"{case_id}:refusal_has_citation")
        if case.get("expected_refused") is True and actual.get("refused") is not True:
            failures.append(f"{case_id}:expected_refusal_missing")

        claims = actual.get("claims", [])
        claims = claims if isinstance(claims, list) else []
        for claim in claims:
            status = claim.get("support_status")
            if status in {"unsupported", "numeric_unsupported"}:
                failures.append(f"{case_id}:claim_{claim.get('claim_id')}:{status}")

        retrieval = actual.get("retrieval")
        result_pages = retrieval.get("result_pages", []) if isinstance(retrieval, dict) else []
        for expected_page in case.get("expected_pages", []):
            expected_page_total += 1
            if any(start <= expected_page <= end for start, end in result_pages):
                expected_page_hits += 1

        if case.get("category") == "quantitative_result":
            chunks = validator._recorded_scored_chunks(record)
            structured = build_quantitative_evidence(str(case["question"]), chunks)
            errors = validator.validate_structured_result(case, structured.as_regression_record())
            failures.extend(f"{case_id}:quantitative:{error}" for error in errors)

        if case.get("category") == "document_summary":
            intent = infer_summary_intent(str(case["question"]))
            validation = validate_and_repair_summary(answer, intent=intent.value)
            if validation.issues:
                failures.extend(f"{case_id}:summary:{issue}" for issue in validation.issues)
            if _empty_table(answer):
                failures.append(f"{case_id}:empty_table")

        required_roles = case.get("required_evidence_roles", [])
        stages = retrieval.get("stages", []) if isinstance(retrieval, dict) else []
        for role in required_roles:
            if f"role_cited:{role}" not in stages:
                failures.append(f"{case_id}:role_not_cited:{role}")

        cited_evidence = "\n".join(str(citation.get("evidence", "")) for citation in citations)
        for url in _URL_RE.findall(answer):
            if url not in cited_evidence:
                failures.append(f"{case_id}:url_not_in_cited_evidence:{url}")

        usage = actual.get("usage")
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            model_calls += int(usage.get("rewrite_model_calls") or 0)
            model_calls += int(usage.get("answer_model_calls") or 0)
            model_calls += int(usage.get("metadata_model_calls") or 0)
            cache_hits += int(bool(usage.get("cache_hit")))
            if isinstance(usage.get("estimated_cost"), int | float):
                costs.append(float(usage["estimated_cost"]))

        if case_id == "TR006":
            repeat = record.get("cache_repeat")
            if not isinstance(repeat, dict):
                failures.append("TR006:cache_repeat_missing")
            else:
                for field in ("answer", "citations", "claims", "numeric_fidelity_passed"):
                    if repeat.get(field) != actual.get(field):
                        failures.append(f"TR006:cache_repeat_changed:{field}")
                repeat_usage = repeat.get("usage")
                if not isinstance(repeat_usage, dict) or repeat_usage.get("cache_hit") is not True:
                    failures.append("TR006:cache_hit_not_reported")
                else:
                    prompt_tokens += int(repeat_usage.get("prompt_tokens") or 0)
                    completion_tokens += int(repeat_usage.get("completion_tokens") or 0)
                    model_calls += int(repeat_usage.get("rewrite_model_calls") or 0)
                    model_calls += int(repeat_usage.get("answer_model_calls") or 0)
                    model_calls += int(repeat_usage.get("metadata_model_calls") or 0)
                    cache_hits += 1
                    if isinstance(repeat_usage.get("estimated_cost"), int | float):
                        costs.append(float(repeat_usage["estimated_cost"]))

    return AcceptanceSummary(
        case_count=len(indexed_records),
        gate_failures=tuple(dict.fromkeys(failures)),
        expected_page_hits=expected_page_hits,
        expected_page_total=expected_page_total,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model_calls=model_calls,
        cache_hits=cache_hits,
        estimated_cost=sum(costs) if costs else None,
    )


def _render_markdown(summary: AcceptanceSummary) -> str:
    coverage = summary.expected_page_hits / summary.expected_page_total if summary.expected_page_total else 0.0
    rows = [
        "# product-rc2 自动验收门禁",
        "",
        f"- 最终结论：{'通过' if summary.passed else '不通过'}",
        f"- 用例数：{summary.case_count}",
        f"- 预期页覆盖：{summary.expected_page_hits}/{summary.expected_page_total}（{coverage:.1%}）",
        f"- 模型调用：{summary.model_calls}",
        f"- Prompt / Completion Token：{summary.prompt_tokens} / {summary.completion_tokens}",
        f"- 缓存命中：{summary.cache_hits}",
        f"- 估算费用：{summary.estimated_cost if summary.estimated_cost is not None else '未配置单价，无法估算'}",
        "",
        "## 门禁失败",
        "",
    ]
    rows.extend(f"- `{failure}`" for failure in summary.gate_failures)
    if not summary.gate_failures:
        rows.append("- 无")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, action="append", default=[])
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    records = merge_records(args.base, tuple(args.override))
    dataset = tuple(_load_jsonl(DATASET_PATH))
    summary = analyze_records(records, dataset)
    if args.merged_output is not None:
        args.merged_output.parent.mkdir(parents=True, exist_ok=True)
        args.merged_output.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    report = _render_markdown(summary)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
