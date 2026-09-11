"""产品任务感知检索测试；不连接 Chroma、Embedding、Reranker 或 LLM。"""

from __future__ import annotations

from types import SimpleNamespace

from researchpilot.index import IndexHit
from researchpilot.product_retrieval import ProductTaskRetrievalService, table_quality_score
from researchpilot.schemas import BlockType, Chunk, QueryType, ScoredChunk


def _parent(name: str, *, page: int, section: str, content: str) -> Chunk:
    return Chunk.create(
        paper_id="paper_demo",
        parent_id=f"block_{name}",
        page_start=page,
        page_end=page,
        section_path=(section,),
        content=content,
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="structure_aware_v1",
    )


def _child(parent: Chunk, name: str) -> Chunk:
    return Chunk.create(
        paper_id=parent.paper_id,
        parent_id=parent.chunk_id,
        page_start=parent.page_start,
        page_end=parent.page_end,
        section_path=parent.section_path,
        content=parent.content,
        block_types=parent.block_types,
        chunker_version=parent.chunker_version,
        occurrence_index=len(name),
    )


class _Index:
    def __init__(self, parents: tuple[Chunk, ...]) -> None:
        self.parents = {parent.chunk_id: IndexHit(parent, "demo.pdf", 0.0) for parent in parents}
        self.children = tuple(
            IndexHit(_child(parent, str(index)), "demo.pdf", 0.0) for index, parent in enumerate(parents)
        )

    def list_children(self, **kwargs: object) -> tuple[IndexHit, ...]:
        return self.children

    def get_parent_chunks(self, *, parent_chunk_ids: tuple[str, ...], **kwargs: object):
        return {item: self.parents[item] for item in parent_chunk_ids if item in self.parents}


class _Retriever:
    def __init__(self, normal: tuple[ScoredChunk, ...]) -> None:
        self.normal = normal
        self.calls: list[str] = []

    def retrieve(self, *, search_query: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append(search_query)
        return SimpleNamespace(
            results=self.normal,
            candidate_count=len(self.normal),
            expanded_queries=(search_query,),
            stages=("dense_child_retrieval", "bm25_child_retrieval", "rrf_fusion"),
            stage_elapsed_ms=(),
            index_version="index-v1",
        )


def _service():
    parents = (
        _parent("front", page=1, section="Title", content="Demo Paper\nAlice and Bob"),
        _parent("intro", page=2, section="Introduction", content="The problem and motivation."),
        _parent("method", page=4, section="Method", content="The proposed architecture."),
        _parent(
            "experiment",
            page=7,
            section="Experimental Setup",
            content="Datasets, baselines, evaluation metrics and implementation details.",
        ),
        _parent(
            "results",
            page=8,
            section="Evaluation Results and Ablation Study",
            content="Table 1 reports a 68.4% win rate over the baseline.",
        ),
        _parent(
            "ablation",
            page=9,
            section="Ablation Study",
            content="Table 2 reports a 57.2% win rate for the full model.",
        ),
        _parent(
            "cost",
            page=10,
            section="Model Cost and Efficiency",
            content="The retrieval phase uses fewer than 100 prompt tokens.",
        ),
        _parent("conclusion", page=12, section="Conclusion", content="Findings and limitations."),
    )
    index = _Index(parents)
    normal = (
        ScoredChunk(
            chunk=parents[2],
            source_name="demo.pdf",
            dense_rank=1,
            dense_score=0.8,
            fusion_score=0.02,
        ),
        ScoredChunk(
            chunk=parents[3],
            source_name="demo.pdf",
            dense_rank=2,
            dense_score=0.7,
            fusion_score=0.01,
        ),
    )
    retriever = _Retriever(normal)
    return ProductTaskRetrievalService(retriever, index=index, result_limit=10), retriever


def _retrieve(service: ProductTaskRetrievalService, query_type: QueryType):
    return service.retrieve(
        query="问题",
        user_id="user_a",
        session_id="session_a",
        paper_id="paper_demo",
        query_type=query_type,
    )


def test_metadata_forces_front_matter_and_keeps_page_citation_source() -> None:
    service, retriever = _service()

    result = _retrieve(service, QueryType.METADATA)

    assert retriever.calls == []
    assert result.results
    assert {item.chunk.page_start for item in result.results}.issubset({1, 2})
    assert result.trace.stages == ("front_matter_selection", "parent_hydration")


def test_summary_balances_key_sections_before_hybrid_supplement() -> None:
    service, _ = _service()

    result = _retrieve(service, QueryType.DOCUMENT_SUMMARY)

    sections = {item.chunk.section_path[0] for item in result.results}
    assert {
        "Introduction",
        "Method",
        "Experimental Setup",
        "Evaluation Results and Ablation Study",
        "Conclusion",
    }.issubset(sections)
    assert "section_balanced_selection" in result.trace.stages


def test_experimental_setup_uses_bilingual_expansion_and_prioritises_experiment_section() -> None:
    service, retriever = _service()

    result = _retrieve(service, QueryType.EXPERIMENTAL_SETUP)

    assert len(retriever.calls) == 2
    assert "experimental setup" in retriever.calls[1]
    assert result.results[0].chunk.section_path == ("Experimental Setup",)
    assert result.trace.query_type is QueryType.EXPERIMENTAL_SETUP
    assert "experimental_structure_selection" in result.trace.stages


def test_experimental_results_prioritise_results_and_ablation_sections() -> None:
    service, retriever = _service()

    result = _retrieve(service, QueryType.EXPERIMENTAL_RESULTS)

    assert len(retriever.calls) == 2
    assert "experimental results" in retriever.calls[1]
    result_pages = {item.chunk.page_start for item in result.results}
    assert result.results[0].chunk.page_start == 8
    assert {8, 9, 10}.issubset(result_pages)
    assert result.trace.query_type is QueryType.EXPERIMENTAL_RESULTS
    assert "experimental_results_structure_selection" in result.trace.stages


def test_table_quality_prefers_readable_numeric_table_over_corrupted_ocr() -> None:
    readable = """Table 2: Ablation results
| Model | Score |
|---|---|
| Full | 67.6% |
| -High | 57.2% |
| -Low | 52.4% |
| -Origin | 60.0% |"""
    corrupted = """Figure 2\n| Model | Tokens |\n| Ours | 610 �� 1 , |\n| API Calls | 610��1 , 000 C max |"""

    assert table_quality_score(readable) >= 0.45
    assert table_quality_score(corrupted) < table_quality_score(readable)


def test_result_selection_uses_best_child_from_same_parent() -> None:
    heading = _parent(
        "shared_heading",
        page=8,
        section="Ablation Study",
        content="Ablation Study",
    )
    table_parent = Chunk.create(
        paper_id="paper_demo",
        parent_id="block_shared_table",
        page_start=8,
        page_end=8,
        section_path=("Ablation Study",),
        content="Table 2\n| Model | Score |\n|---|---|\n| Full | 67.6% |\n| -High | 57.2% |",
        block_types=(BlockType.TABLE,),
        chunker_version="structure_aware_v1",
    )
    index = _Index((heading, table_parent))
    retriever = _Retriever(())
    service = ProductTaskRetrievalService(retriever, index=index, result_limit=4)

    result = service.retrieve(
        query="消融实验结果是什么？",
        user_id="user_a",
        session_id="session_a",
        paper_id="paper_demo",
        query_type=QueryType.EXPERIMENTAL_RESULTS,
    )

    assert result.results[0].chunk.block_types == (BlockType.TABLE,)


def test_ablation_result_reserves_neighboring_variant_explanation() -> None:
    table = _parent(
        "ablation_table",
        page=8,
        section="Ablation Study",
        content="Table 2 reports -High and -Low results: 78.0% and 81.2%.",
    )
    explanation = _parent(
        "ablation_meaning",
        page=8,
        section="Ablation Study",
        content="Low-level-only Retrieval: The -High variant removes high-order retrieval.",
    )
    index = _Index((table, explanation))
    service = ProductTaskRetrievalService(_Retriever(()), index=index, result_limit=10)

    result = service.retrieve(
        query="消融实验说明双层检索有什么作用？",
        user_id="user_a",
        session_id="session_a",
        paper_id="paper_demo",
        query_type=QueryType.EXPERIMENTAL_RESULTS,
    )

    assert any("removes high-order retrieval" in item.chunk.content for item in result.results)


def test_cost_result_reserves_neighboring_prose_with_units_and_api_count() -> None:
    table = _parent(
        "cost_table",
        page=9,
        section="Model Cost and Efficiency",
        content="| API Calls | 610×1,000 Cmax | 1 |",
    )
    overview = _parent(
        "cost_overview",
        page=9,
        section="Model Cost and Efficiency",
        content="We compare model cost from tokens and API calls.",
    )
    details = _parent(
        "cost_details",
        page=9,
        section="Model Cost and Efficiency",
        content=(
            "The total token consumption is 610,000 tokens. LightRAG uses fewer than "
            "100 tokens and requires only a single API call."
        ),
    )
    index = _Index((table, overview, details))
    service = ProductTaskRetrievalService(_Retriever(()), index=index, result_limit=10)

    result = service.retrieve(
        query="论文报告了哪些成本或效率结果？",
        user_id="user_a",
        session_id="session_a",
        paper_id="paper_demo",
        query_type=QueryType.EXPERIMENTAL_RESULTS,
    )

    assert any("610,000 tokens" in item.chunk.content for item in result.results)
    assert any("single API call" in item.chunk.content for item in result.results)


def test_default_detail_keeps_frozen_hybrid_path() -> None:
    service, retriever = _service()

    result = _retrieve(service, QueryType.SEMANTIC_EXPLANATION)

    assert retriever.calls == ["问题"]
    assert result.trace.stages[0] == "dense_child_retrieval"


def test_multi_section_reserves_problem_method_and_result_evidence() -> None:
    service, retriever = _service()

    result = service.retrieve(
        query="研究问题、方法设计与实验结果之间有什么联系？",
        user_id="user_a",
        session_id="session_a",
        paper_id="paper_demo",
        query_type=QueryType.MULTI_SECTION,
    )

    sections = {item.chunk.section_path[0] for item in result.results}
    assert "Introduction" in sections
    assert "Method" in sections
    assert "Evaluation Results and Ablation Study" in sections
    assert len(retriever.calls) == 3
    assert "evidence_role_covered:problem" in result.trace.stages
    assert "evidence_role_covered:method" in result.trace.stages
    assert "evidence_role_covered:results" in result.trace.stages
    assert result.trace.query_type is QueryType.MULTI_SECTION
