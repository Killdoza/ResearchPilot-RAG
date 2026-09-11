from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "analyze_product_acceptance.py"
spec = importlib.util.spec_from_file_location("acceptance_analyzer", SCRIPT)
assert spec is not None and spec.loader is not None
analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


def _case(case_id: str = "TR014") -> dict[str, object]:
    return {
        "id": case_id,
        "question": "你好",
        "expected_route": "general_chat",
        "category": "memory_isolation",
        "expected_pages": [],
        "forbidden_claims": [],
    }


def _record(*, citation: bool = False) -> dict[str, object]:
    return {
        "id": "TR014",
        "runner_error": None,
        "actual": {
            "route": "general_chat",
            "answer": "你好",
            "refused": False,
            "target_paper_ids": [],
            "citations": (
                [{"paper_id": "paper_a", "page_start": 1, "page_end": 1, "evidence": "x"}] if citation else []
            ),
            "claims": [],
            "retrieval": None,
            "usage": {},
        },
    }


def test_acceptance_analyzer_passes_clean_general_chat() -> None:
    summary = analyzer.analyze_records((_record(),), (_case(),))

    assert summary.passed


def test_acceptance_analyzer_rejects_general_chat_citation() -> None:
    summary = analyzer.analyze_records((_record(citation=True),), (_case(),))

    assert not summary.passed
    assert "TR014:general_chat_has_citation" in summary.gate_failures
