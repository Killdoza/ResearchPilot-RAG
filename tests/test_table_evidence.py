from __future__ import annotations

from researchpilot.schemas import BlockType, Chunk, ScoredChunk
from researchpilot.table_evidence import (
    extract_concept_mappings,
    extract_cost_facts,
    extract_structured_tables,
)


def _scored(text: str, *, page: int, section: str, index: int) -> ScoredChunk:
    chunk = Chunk.create(
        paper_id="paper_demo",
        parent_id=f"parent_{index}",
        page_start=page,
        page_end=page,
        section_path=(section,),
        content=text,
        block_types=(BlockType.TABLE,),
        chunker_version="test",
        occurrence_index=index,
    )
    return ScoredChunk(chunk=chunk, source_name="demo.pdf")


def test_split_ablation_table_preserves_method_value_direction() -> None:
    first = _scored(
        """Table 2: Performance of ablated versions, using NaiveRAG as reference.
| Agriculture CS Legal Mix | Agriculture CS Legal Mix | | |
|---|---|---|---|
| NaiveRAG LightRAG | NaiveRAG LightRAG | NaiveRAG LightRAG | NaiveRAG LightRAG |
| Overall 32.4% 67.6% | 38.8% 61.2% | 15.2% 84.8% | 40.0% 60.0% |
| NaiveRAG -High | NaiveRAG -High | NaiveRAG -High | NaiveRAG -High |
| Overall 35.2% 64.8% | 44.0% 56.0% | 22.0% 78.0% | 42.4% 57.6% |""",
        page=8,
        section="4.3 ABLATION STUDIES",
        index=1,
    )
    continuation = _scored(
        """| NaiveRAG -Low | NaiveRAG -Low | NaiveRAG -Low | NaiveRAG -Low |
| Overall 34.8% 65.2% | 43.6% 56.4% | 18.8% 81.2% | 35.2% 64.8% |
| NaiveRAG -Origin | NaiveRAG -Origin | NaiveRAG -Origin | NaiveRAG -Origin |
| Overall 25.6% 74.4% | 39.2% 60.8% | 15.6% 84.4% | 44.4% 55.6% |""",
        page=8,
        section="4.3 ABLATION STUDIES",
        index=2,
    )

    tables = extract_structured_tables((first, continuation))
    records = [item.as_record() for item in tables[0].comparisons]

    assert {
        "dataset": "Legal",
        "metric": "Overall",
        "method_a": "NaiveRAG",
        "value_a": "22.0%",
        "method_b": "-High",
        "value_b": "78.0%",
        "relation": "method_b_higher",
    } in records
    assert any(item["method_b"] == "-Low" and item["value_b"] == "81.2%" for item in records)


def test_split_table_recovers_dataset_header_when_continuation_is_retrieved_first() -> None:
    continuation = _scored(
        """| NaiveRAG -Low | NaiveRAG -Low | NaiveRAG -Low | NaiveRAG -Low |
| Overall 34.8% 65.2% | 43.6% 56.4% | 18.8% 81.2% | 35.2% 64.8% |""",
        page=8,
        section="4.3 ABLATION STUDIES",
        index=1,
    )
    header = _scored(
        """Table 2: Performance of ablated versions.
| Agriculture CS Legal Mix | Agriculture CS Legal Mix | | |
|---|---|---|---|
| NaiveRAG -High | NaiveRAG -High | NaiveRAG -High | NaiveRAG -High |
| Overall 35.2% 64.8% | 44.0% 56.0% | 22.0% 78.0% | 42.4% 57.6% |""",
        page=8,
        section="4.3 ABLATION STUDIES",
        index=2,
    )

    records = [
        item.as_record() for table in extract_structured_tables((continuation, header)) for item in table.comparisons
    ]

    assert any(
        item["dataset"] == "Legal" and item["method_b"] == "-High" and item["value_b"] == "78.0%" for item in records
    )
    assert all(item["dataset"] != "NaiveRAG" for item in records)

    comparison = next(
        item
        for table in extract_structured_tables((continuation, header))
        for item in table.comparisons
        if item.dataset == "Legal" and item.method_b == "-Low"
    )
    assert comparison.citation_index == 1
    assert comparison.supporting_citation_indexes == (2,)


def test_ablation_meanings_are_derived_from_neighboring_prose() -> None:
    results = (
        _scored(
            "Low-level-only Retrieval: The -High variant removes high-order retrieval.",
            page=8,
            section="Ablation",
            index=1,
        ),
        _scored(
            "High-level-only Retrieval: The -Low variant prioritizes broader content.",
            page=8,
            section="Ablation",
            index=2,
        ),
        _scored(
            "We eliminated the use of original text. The resulting variant, -Origin, remains strong.",
            page=8,
            section="Ablation",
            index=3,
        ),
    )

    mappings = {item.term: item.normalized_meaning for item in extract_concept_mappings(results)}

    assert mappings == {
        "-High": "remove_high_level_retrieval",
        "-Low": "remove_low_level_retrieval",
        "-Origin": "remove_original_text",
    }


def test_cost_formula_keeps_cmax_as_denominator() -> None:
    results = (
        _scored(
            "| API Calls | 610×1,000 Cmax | 1 |\nCmax denotes the maximum tokens per API call.",
            page=9,
            section="Cost",
            index=1,
        ),
        _scored(
            "total token consumption of 610,000 tokens. LightRAG uses fewer than 100 tokens "
            "and requires only a single API call.",
            page=9,
            section="Cost",
            index=2,
        ),
    )

    expressions = [item.expression for item in extract_cost_facts(results)]

    assert expressions == [
        "610,000 tokens",
        "(610 × 1,000) / Cmax API calls",
        "< 100 tokens",
        "1 API call",
    ]
