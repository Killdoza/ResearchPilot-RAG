from researchpilot.absence_detection import DocumentField, FullDocumentScan
from researchpilot.evidence_gate import cited_role_stages, full_document_scan_stages
from researchpilot.schemas import Answer, Citation, QueryType, RetrievalTrace


def test_full_document_scan_trace_is_explicit():
    stages = full_document_scan_stages(
        FullDocumentScan(
            field_type=DocumentField.CODE_URL,
            scanned_block_count=42,
            matches=(),
        )
    )
    assert stages == (
        "full_document_scan",
        "field_type:code_url",
        "scanned_block_count:42",
        "matched_block_count:0",
    )


def test_role_cited_is_distinct_from_retrieved_and_selected():
    citation = Citation(
        paper_id="paper_demo",
        chunk_id="chunk_results",
        page_start=8,
        page_end=8,
        section_path=("4 Experiments", "4.2 Results"),
        evidence="The experimental results improve performance.",
    )
    answer = Answer(
        text="结果得到提升 [证据 1]。",
        citations=(citation,),
        refused=False,
        trace_id="trace_roles",
        latency_ms=1.0,
    )
    trace = RetrievalTrace(
        trace_id="trace_roles",
        query="方法如何影响结果？",
        search_query="方法如何影响结果？",
        query_type=QueryType.MULTI_SECTION,
        candidate_count=0,
        results=(),
        elapsed_ms=1.0,
        stages=("role_retrieved:results", "role_selected:results"),
    )
    assert cited_role_stages(answer, trace) == ("role_cited:results",)
