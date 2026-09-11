"""Validate the small, human-run product regression checklist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROUTES = {
    "metadata",
    "document_summary",
    "experimental_setup",
    "experimental_results",
    "document_detail",
    "general_chat",
    "scope_insufficient",
    "cross_document",
    "legacy_read_only",
}


def load_cases(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是合法 JSON：{exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"第 {line_number} 行必须是对象")
        cases.append(item)
    return cases


def validate_cases(cases: list[dict[str, object]]) -> None:
    if not 10 <= len(cases) <= 15:
        raise ValueError("产品回归集必须包含 10 至 15 条固定用例")
    seen: set[str] = set()
    for item in cases:
        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("每条用例必须有唯一、非空的 id")
        seen.add(case_id)
        for field in ("phase", "question", "fixture", "expected_route", "expected_outcome", "notes"):
            if not isinstance(item.get(field), str) or not str(item[field]).strip():
                raise ValueError(f"{case_id} 缺少非空字段 {field}")
        if item["expected_route"] not in ALLOWED_ROUTES:
            raise ValueError(f"{case_id} 的 expected_route 不受支持")
        pages = item.get("required_pages")
        if not isinstance(pages, list) or any(
            not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in pages
        ):
            raise ValueError(f"{case_id} 的 required_pages 必须是正整数数组")


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 ResearchPilot 产品回归用例")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "eval" / "datasets" / "product_regression_v0.1.jsonl",
    )
    args = parser.parse_args()
    try:
        cases = load_cases(args.dataset)
        validate_cases(cases)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"产品回归集校验失败：{exc}")
        return 2
    print(f"产品回归集校验通过：{len(cases)} 条用例")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
