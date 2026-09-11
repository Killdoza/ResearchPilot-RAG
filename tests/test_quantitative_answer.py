from __future__ import annotations

from researchpilot.generation import AnswerGenerator
from researchpilot.quantitative_answer import (
    build_quantitative_evidence,
    render_quantitative_answer,
)
from researchpilot.retrieval import RetrievalResult
from researchpilot.schemas import BlockType, Chunk, QueryType, RetrievalTrace, ScoredChunk

TABLE_1 = """Table 1: Win rates (%) of baselines v.s. LightRAG.
| Agriculture CS Legal Mix | Agriculture CS Legal Mix | | |
|---|---|---|---|
| NaiveRAG LightRAG | NaiveRAG LightRAG | NaiveRAG LightRAG | NaiveRAG LightRAG |
| Overall 32.4% 67.6% | 38.8% 61.2% | 15.2% 84.8% | 40.0% 60.0% |
| RQ-RAG LightRAG | RQ-RAG LightRAG | RQ-RAG LightRAG | RQ-RAG LightRAG |
| Overall 32.4% 67.6% | 38.0% 62.0% | 14.4% 85.6% | 40.0% 60.0% |
| HyDE LightRAG | HyDE LightRAG | HyDE LightRAG | HyDE LightRAG |
| Overall 24.8% 75.2% | 41.6% 58.4% | 26.4% 73.6% | 42.4% 57.6% |
| GraphRAG LightRAG | GraphRAG LightRAG | GraphRAG LightRAG | GraphRAG LightRAG |
| Overall 45.2% 54.8% | 48.0% 52.0% | 47.2% 52.8% | 50.4% 49.6% |"""

TABLE_2 = """Table 2: Performance of ablated versions of LightRAG, using NaiveRAG as reference.
| Agriculture CS Legal Mix | Agriculture CS Legal Mix | | |
|---|---|---|---|
| NaiveRAG LightRAG | NaiveRAG LightRAG | NaiveRAG LightRAG | NaiveRAG LightRAG |
| Overall 32.4% 67.6% | 38.8% 61.2% | 15.2% 84.8% | 40.0% 60.0% |
| NaiveRAG -High | NaiveRAG -High | NaiveRAG -High | NaiveRAG -High |
| Overall 35.2% 64.8% | 44.0% 56.0% | 22.0% 78.0% | 42.4% 57.6% |
| NaiveRAG -Low | NaiveRAG -Low | NaiveRAG -Low | NaiveRAG -Low |
| Overall 34.8% 65.2% | 43.6% 56.4% | 18.8% 81.2% | 35.2% 64.8% |
| NaiveRAG -Origin | NaiveRAG -Origin | NaiveRAG -Origin | NaiveRAG -Origin |
| Overall 25.6% 74.4% | 39.2% 60.8% | 15.6% 84.4% | 44.4% 55.6% |"""


def _result(text: str, *, page: int, section: str, occurrence: int) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk.create(
            paper_id="paper_demo",
            parent_id=f"p{occurrence}",
            page_start=page,
            page_end=page,
            section_path=(section,),
            content=text,
            block_types=(BlockType.TABLE,),
            chunker_version="test",
            occurrence_index=occurrence,
        ),
        source_name="demo.pdf",
    )


def _tables() -> tuple[ScoredChunk, ...]:
    return (
        _result(TABLE_1, page=7, section="4.2 RESULTS", occurrence=1),
        _result(TABLE_2, page=8, section="4.3 ABLATION STUDIES", occurrence=2),
        _result(
            "Low-level-only Retrieval: The -High variant removes high-order retrieval.\n"
            "High-level-only Retrieval: The -Low variant prioritizes broader content.\n"
            "We eliminated the use of original text. The resulting variant, -Origin, remains strong.",
            page=8,
            section="4.3 ABLATION STUDIES",
            occurrence=3,
        ),
    )


def test_graph_comparison_keeps_mix_reversal() -> None:
    evidence = build_quantitative_evidence("LightRAG 与 GraphRAG 的主要胜率比较是什么？", _tables())
    records = evidence.as_regression_record()["comparisons"]
    assert {
        "dataset": "Mix",
        "metric": "Overall",
        "method_a": "GraphRAG",
        "value_a": "50.4%",
        "method_b": "LightRAG",
        "value_b": "49.6%",
        "relation": "method_a_higher",
    } in records


def test_ablation_answer_explains_high_and_low_without_reversing_them() -> None:
    evidence = build_quantitative_evidence("消融实验说明双层检索有什么作用？", _tables())
    text = render_quantitative_answer(evidence)
    assert text is not None
    assert "-High**：移除高层检索，因此该变体只保留低层检索" in text
    assert "-Low**：移除低层检索，因此该变体只保留高层检索" in text
    assert "-Origin**：移除原始文本，而不是移除图索引" in text


class _NeverCalledClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: str) -> str:
        del kwargs
        self.calls += 1
        raise AssertionError("structured quantitative answer must not call the LLM")


def test_experimental_results_use_deterministic_table_path_and_prune_citations() -> None:
    results = _tables()
    trace = RetrievalTrace(
        trace_id="trace_table",
        query="LightRAG 与 GraphRAG 的主要胜率比较是什么？",
        search_query="LightRAG GraphRAG win rate",
        query_type=QueryType.EXPERIMENTAL_RESULTS,
        candidate_count=len(results),
        results=results,
        elapsed_ms=1.0,
    )
    client = _NeverCalledClient()

    answer = AnswerGenerator(client).generate(
        question=trace.query,
        retrieval=RetrievalResult(results=results, trace=trace),
    )

    assert client.calls == 0
    assert "Mix · Overall" in answer.text
    assert "GraphRAG 更高" in answer.text
    assert answer.citations
    assert len(answer.citations) < len(results)
    assert all(claim.support_status == "supported" for claim in answer.claims)


def test_split_table_answer_retains_header_and_body_citations() -> None:
    header = _result(
        "Table 1: Win rates.\n| Agriculture CS Legal Mix | Agriculture CS Legal Mix | | |",
        page=7,
        section="4.2 RESULTS",
        occurrence=1,
    )
    body = _result(
        "| GraphRAG LightRAG | GraphRAG LightRAG | GraphRAG LightRAG | GraphRAG LightRAG |\n"
        "| Overall 45.2% 54.8% | 48.0% 52.0% | 47.2% 52.8% | 50.4% 49.6% |",
        page=7,
        section="4.2 RESULTS",
        occurrence=2,
    )
    evidence = build_quantitative_evidence("LightRAG 与 GraphRAG 的主要胜率比较是什么？", (body, header))
    rendered = render_quantitative_answer(evidence)

    assert rendered is not None
    assert "[证据 1、2]" in rendered
