"""Build deterministic quantitative answers from structured table evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .schemas import ScoredChunk
from .table_evidence import (
    ConceptMapping,
    CostFact,
    TableComparison,
    extract_concept_mappings,
    extract_cost_facts,
    extract_structured_tables,
)


@dataclass(frozen=True)
class QuantitativeEvidence:
    comparisons: tuple[TableComparison, ...] = ()
    concept_mappings: tuple[ConceptMapping, ...] = ()
    cost_facts: tuple[CostFact, ...] = ()

    def as_regression_record(self) -> dict[str, object]:
        return {
            "comparisons": [item.as_record() for item in self.comparisons],
            "concept_mappings": {item.term: item.normalized_meaning for item in self.concept_mappings},
            "expressions": [item.expression for item in self.cost_facts],
        }


def _margin(comparison: TableComparison) -> float:
    return float(comparison.value_b.rstrip("%")) - float(comparison.value_a.rstrip("%"))


def _representative_three(
    comparisons: Sequence[TableComparison],
) -> tuple[TableComparison, ...]:
    """Pick a wide win, closest contest, and reversal when the table contains one."""
    if len(comparisons) <= 3:
        return tuple(comparisons)
    positive = [item for item in comparisons if _margin(item) > 0]
    negative = [item for item in comparisons if _margin(item) < 0]
    selected: list[TableComparison] = []
    if positive:
        selected.append(max(positive, key=_margin))
        closest = min(positive, key=_margin)
        if closest not in selected:
            selected.append(closest)
    if negative:
        reversal = max(negative, key=_margin)
        if reversal not in selected:
            selected.append(reversal)
    for item in comparisons:
        if item not in selected:
            selected.append(item)
        if len(selected) == 3:
            break
    return tuple(selected[:3])


def build_quantitative_evidence(
    question: str,
    results: Sequence[ScoredChunk],
) -> QuantitativeEvidence:
    lowered = question.casefold()
    if re.search(r"成本|效率|token|api\s*call|cost|efficien", lowered):
        return QuantitativeEvidence(cost_facts=extract_cost_facts(results))

    tables = extract_structured_tables(results)
    if not tables:
        return QuantitativeEvidence()
    wants_ablation = bool(re.search(r"消融|ablation|-high|-low|-origin|双层", lowered))
    selected_tables = (
        tuple(table for table in tables if ("ablat" in table.caption.casefold()) == wants_ablation) or tables
    )
    comparisons = [item for table in selected_tables for item in table.comparisons]

    datasets = tuple(dict.fromkeys(item.dataset for item in comparisons))
    mentioned_datasets = [name for name in datasets if name.casefold() in lowered]
    if mentioned_datasets:
        comparisons = [item for item in comparisons if item.dataset in mentioned_datasets]

    methods = tuple(dict.fromkeys(method for item in comparisons for method in (item.method_a, item.method_b)))
    # LightRAG is the shared reference in this corpus; mentioning it alone must
    # not remove every baseline.  A specifically named non-shared method does.
    mentioned_methods = [
        method for method in methods if method.casefold() in lowered and method.casefold() != "lightrag"
    ]
    if mentioned_methods:
        comparisons = [
            item for item in comparisons if item.method_a in mentioned_methods or item.method_b in mentioned_methods
        ]

    mentioned_metrics = [
        metric
        for metric in ("Comprehensiveness", "Diversity", "Empowerment", "Overall")
        if metric.casefold() in lowered
    ]
    if mentioned_metrics:
        comparisons = [item for item in comparisons if item.metric in mentioned_metrics]
    elif any(item.metric == "Overall" for item in comparisons):
        comparisons = [item for item in comparisons if item.metric == "Overall"]

    if wants_ablation and comparisons and not mentioned_datasets:
        by_dataset: dict[str, list[TableComparison]] = {}
        for item in comparisons:
            by_dataset.setdefault(item.dataset, []).append(item)
        representative_dataset = max(
            by_dataset,
            key=lambda dataset: sum(abs(_margin(item)) for item in by_dataset[dataset]) / len(by_dataset[dataset]),
        )
        comparisons = by_dataset[representative_dataset]

    # Stable de-duplication is important when overlapping Parent chunks contain
    # the same table row with different citation indexes.
    unique: dict[tuple[str, str, str, str], TableComparison] = {}
    for item in comparisons:
        unique.setdefault((item.dataset, item.metric, item.method_a, item.method_b), item)
    chosen = tuple(unique.values())
    if re.search(r"三个|3\s*(?:个|项)|three", lowered):
        chosen = _representative_three(chosen)
    else:
        chosen = chosen[:8]
    mappings = extract_concept_mappings(results) if wants_ablation else ()
    return QuantitativeEvidence(comparisons=chosen, concept_mappings=mappings)


def _mapping_explanation(mapping: ConceptMapping) -> str:
    return {
        "remove_high_level_retrieval": "移除高层检索，因此该变体只保留低层检索",
        "remove_low_level_retrieval": "移除低层检索，因此该变体只保留高层检索",
        "remove_original_text": "移除原始文本，而不是移除图索引",
    }.get(mapping.normalized_meaning, mapping.normalized_meaning)


def render_quantitative_answer(evidence: QuantitativeEvidence) -> str | None:
    if not (evidence.comparisons or evidence.concept_mappings or evidence.cost_facts):
        return None
    lines: list[str] = []
    if evidence.comparisons:
        lines.extend(("", "### 可核验的表格比较"))
        for item in evidence.comparisons:
            relation = {
                "method_a_higher": f"{item.method_a} 更高",
                "method_b_higher": f"{item.method_b} 更高",
                "equal": "两者相同",
            }[item.relation]
            indexes = tuple(dict.fromkeys((item.citation_index, *item.supporting_citation_indexes)))
            marker = f"[证据 {'、'.join(str(index) for index in indexes)}]"
            lines.append(
                f"- **{item.dataset} · {item.metric}**：{item.method_a} 为 {item.value_a}，"
                f"{item.method_b} 为 {item.value_b}；{relation}。{marker}"
            )
    if evidence.concept_mappings:
        lines.extend(("", "### 消融名称与实际含义"))
        for item in evidence.concept_mappings:
            lines.append(f"- **{item.term}**：{_mapping_explanation(item)}。[证据 {item.citation_index}]")
    if evidence.cost_facts:
        lines.extend(("", "### 成本与效率"))
        for item in evidence.cost_facts:
            explanation = {
                "610,000 tokens": "GraphRAG 检索阶段的 token 消耗（610 个社区 × 每个 1,000 tokens）",
                "(610 × 1,000) / Cmax API calls": ("GraphRAG 检索阶段的 API 调用公式；该式不能解释为 610,000 次调用"),
                "< 100 tokens": "LightRAG 生成关键词并完成检索所用的 token 少于 100",
                "1 API call": "LightRAG 的整个检索阶段只需一次 API 调用",
            }.get(item.expression, item.explanation)
            lines.append(f"- **{item.expression}**：{explanation}。[证据 {item.citation_index}]")
        lines.extend(
            (
                "",
                "> 注意：API 调用数中的 Cmax 是分母；`610,000` 表示 token 数，不能改写成 610,000 次 API 调用。",
            )
        )
    return "\n".join(lines)
