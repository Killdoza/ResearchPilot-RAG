"""Router / Scope 的确定性评测指标。

Gold 标注只在本模块出现，Planner 与 Query Analyzer 均不导入本模块，
从结构上避免评测答案流回运行时路径。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .schemas import QueryType, SchemaValidationError


@dataclass(frozen=True)
class RouterGold:
    case_id: str
    query_type: QueryType
    required_paper_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.required_paper_keys:
            raise SchemaValidationError("RouterGold 必须包含 case_id 和 required_paper_keys。")
        if not isinstance(self.query_type, QueryType):
            raise SchemaValidationError("RouterGold.query_type 必须是 QueryType。")


@dataclass(frozen=True)
class RouterPrediction:
    case_id: str
    query_type: QueryType | None
    target_paper_keys: tuple[str, ...]
    scope_fallback: bool

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise SchemaValidationError("RouterPrediction.case_id 不能为空。")
        if self.query_type is not None and not isinstance(self.query_type, QueryType):
            raise SchemaValidationError("RouterPrediction.query_type 必须是 QueryType 或 None。")


@dataclass(frozen=True)
class RouterEvaluationSummary:
    evaluated_cases: int
    query_type_accuracy: float
    query_type_macro_f1: float
    target_doc_precision: float
    target_doc_recall: float
    scope_fallback_rate: float


def evaluate_router_predictions(
    *, gold_cases: Iterable[RouterGold], predictions: Iterable[RouterPrediction]
) -> RouterEvaluationSummary:
    """按已覆盖的 Gold 类型计算 Macro-F1；未预测类型计为错误。"""
    gold_items = tuple(gold_cases)
    prediction_items = tuple(predictions)
    gold_by_id = {item.case_id: item for item in gold_items}
    predicted_by_id = {item.case_id: item for item in prediction_items}
    if not gold_by_id:
        raise SchemaValidationError("Router 评测至少需要一条 Gold 标注。")
    if len(gold_by_id) != len(gold_items) or len(predicted_by_id) != len(prediction_items):
        raise SchemaValidationError("Router 评测 case_id 不能重复。")
    if set(gold_by_id) != set(predicted_by_id):
        raise SchemaValidationError("Router Gold 与 Prediction 的 case_id 必须完全一致。")

    correct_type = 0
    true_positive_docs = false_positive_docs = false_negative_docs = 0
    fallback_count = 0
    labels = {item.query_type for item in gold_by_id.values()}
    per_label: dict[QueryType, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for case_id, gold in gold_by_id.items():
        prediction = predicted_by_id[case_id]
        if prediction.query_type == gold.query_type:
            correct_type += 1
            per_label[gold.query_type]["tp"] += 1
        else:
            per_label[gold.query_type]["fn"] += 1
            if prediction.query_type in labels:
                per_label[prediction.query_type]["fp"] += 1
        gold_docs = set(gold.required_paper_keys)
        predicted_docs = set(prediction.target_paper_keys)
        true_positive_docs += len(gold_docs & predicted_docs)
        false_positive_docs += len(predicted_docs - gold_docs)
        false_negative_docs += len(gold_docs - predicted_docs)
        fallback_count += int(prediction.scope_fallback)

    f1_values: list[float] = []
    for label in labels:
        counts = per_label[label]
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        f1_values.append(2 * counts["tp"] / denominator if denominator else 0.0)
    precision_denominator = true_positive_docs + false_positive_docs
    recall_denominator = true_positive_docs + false_negative_docs
    total = len(gold_by_id)
    return RouterEvaluationSummary(
        evaluated_cases=total,
        query_type_accuracy=correct_type / total,
        query_type_macro_f1=sum(f1_values) / len(f1_values),
        target_doc_precision=(true_positive_docs / precision_denominator if precision_denominator else 0.0),
        target_doc_recall=(true_positive_docs / recall_denominator if recall_denominator else 0.0),
        scope_fallback_rate=fallback_count / total,
    )
