"""Router / Scope 指标的纯函数测试，不接触 Planner 或 Gold 运行时路径。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.router_evaluation import (  # noqa: E402
    RouterGold,
    RouterPrediction,
    evaluate_router_predictions,
)
from researchpilot.schemas import QueryType, SchemaValidationError  # noqa: E402


class RouterEvaluationTests(unittest.TestCase):
    def test_metrics_include_missing_type_and_scope_fallback(self) -> None:
        summary = evaluate_router_predictions(
            gold_cases=(
                RouterGold("q1", QueryType.METADATA, ("paper_a",)),
                RouterGold("q2", QueryType.CROSS_DOCUMENT, ("paper_a", "paper_b")),
            ),
            predictions=(
                RouterPrediction("q1", QueryType.METADATA, ("paper_a",), False),
                RouterPrediction("q2", None, (), True),
            ),
        )
        self.assertEqual(summary.evaluated_cases, 2)
        self.assertEqual(summary.query_type_accuracy, 0.5)
        self.assertEqual(summary.query_type_macro_f1, 0.5)
        self.assertEqual(summary.target_doc_precision, 1.0)
        self.assertEqual(summary.target_doc_recall, 1 / 3)
        self.assertEqual(summary.scope_fallback_rate, 0.5)

    def test_case_id_mismatch_is_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            evaluate_router_predictions(
                gold_cases=(RouterGold("q1", QueryType.METADATA, ("paper_a",)),),
                predictions=(RouterPrediction("q2", QueryType.METADATA, ("paper_a",), False),),
            )


if __name__ == "__main__":
    unittest.main()
