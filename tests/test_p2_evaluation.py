from researchpilot.evidence_control import EvidenceCandidate
from researchpilot.p2_dataset import P2Case
from researchpilot.p2_evaluation import evaluate_p2_retrieval_case, summarize_p2_retrieval
from researchpilot.schemas import BlockType, Chunk, ScoredChunk


def _candidate(paper_key: str, page: int) -> EvidenceCandidate:
    chunk = Chunk.create(
        paper_id=f"paper_{paper_key}",
        parent_id=None,
        page_start=page,
        page_end=page,
        section_path=("Method",),
        content=f"evidence {paper_key} {page}",
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test",
    )
    return EvidenceCandidate(paper_key=paper_key, result=ScoredChunk(chunk=chunk, dense_rank=1, dense_score=0.9))


def _case() -> P2Case:
    return P2Case.from_dict(
        {
            "schema_version": "ai_cs_p2_eval_v0.2",
            "id": "p2_case",
            "source_id": "s",
            "split": "dev",
            "task_type": "cross_document",
            "answerable": True,
            "paper_keys": ["paper_a", "paper_b"],
            "required_paper_keys": ["paper_a", "paper_b"],
            "comparison_dimension": "method",
            "comparison_allowed": False,
            "refusal_reason_target": None,
            "question": "compare",
            "expected_answer": "answer",
            "expected_claims": [
                {"claim_id": "C1", "text": "a"},
                {"claim_id": "C2", "text": "b"},
                {"claim_id": "C3", "text": "c"},
            ],
            "gold_evidence": [
                {
                    "evidence_id": "e1",
                    "claim_ids": ["C1", "C3"],
                    "paper_key": "paper_a",
                    "page_start": 2,
                    "page_end": 2,
                    "section_hint": "M",
                    "supporting_text_summary": "a",
                },
                {
                    "evidence_id": "e2",
                    "claim_ids": ["C2", "C3"],
                    "paper_key": "paper_b",
                    "page_start": 3,
                    "page_end": 3,
                    "section_hint": "M",
                    "supporting_text_summary": "b",
                },
            ],
            "annotation_notes": [],
        }
    )


def test_p2_retrieval_scores_each_required_paper_by_page_overlap():
    outcome = evaluate_p2_retrieval_case(
        case=_case(),
        mode="scoped",
        selected=(_candidate("paper_a", 2), _candidate("paper_b", 1)),
        retrieval_latency_ms=12.0,
    )
    assert outcome.per_paper_recall == 0.5
    assert outcome.evidence_coverage is False
    summary = summarize_p2_retrieval((outcome,))
    assert summary.evidence_coverage_rate == 0.0
