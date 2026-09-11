"""P2-B3 G2 Claim—Citation—Context 协议测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.evidence_control import (  # noqa: E402
    CoverageGateDecision,
    EvidenceCandidate,
    RetryThenRefuseDecision,
)
from researchpilot.p2_answer_protocol import (  # noqa: E402
    P2Citation,
    P2Claim,
    P2GeneratedAnswer,
    validate_claim_citations,
)
from researchpilot.schemas import BlockType, Chunk, ScoredChunk  # noqa: E402


def _candidate(paper_key: str, page: int) -> EvidenceCandidate:
    chunk = Chunk.create(
        paper_id=f"runtime_{paper_key}",
        parent_id=None,
        page_start=page,
        page_end=page,
        section_path=("Method",),
        content=f"{paper_key} evidence",
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test",
    )
    return EvidenceCandidate(
        paper_key=paper_key,
        result=ScoredChunk(chunk=chunk, source_name=f"{paper_key}.pdf", rerank_score=0.9),
    )


class P2AnswerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = _candidate("paper_a", 2)
        self.b = _candidate("paper_b", 4)
        gate = CoverageGateDecision(
            g0_document_coverage=True,
            g1_relevant_evidence_coverage=True,
            passed=True,
            missing_document_paper_keys=(),
            missing_relevant_evidence_paper_keys=(),
            per_paper_candidate_counts=(("paper_a", 1), ("paper_b", 1)),
            per_paper_usable_counts=(("paper_a", 1), ("paper_b", 1)),
            reason=None,
        )
        self.decision = RetryThenRefuseDecision(
            selected=(self.a, self.b),
            gate=gate,
            retry_count=0,
            refused=False,
            refusal_reason=None,
            per_paper_request_count=(("paper_a", 1), ("paper_b", 1)),
        )

    def test_comparative_claim_requires_context_from_every_target_paper(self) -> None:
        a_id = self.a.result.chunk.chunk_id
        b_id = self.b.result.chunk.chunk_id
        answer = P2GeneratedAnswer(
            answer="两篇论文的机制不同。",
            citations=(P2Citation("paper_a", 2, 2, a_id), P2Citation("paper_b", 4, 4, b_id)),
            claims=(P2Claim("C1", "比较结论", (a_id, b_id), comparative=True),),
            refused=False,
            refusal_reason=None,
        )
        self.assertTrue(
            validate_claim_citations(
                answer=answer, decision=self.decision, required_paper_keys=("paper_a", "paper_b")
            ).valid
        )

    def test_fabricated_context_or_one_sided_comparison_is_rejected(self) -> None:
        a_id = self.a.result.chunk.chunk_id
        answer = P2GeneratedAnswer(
            answer="A 更好。",
            citations=(P2Citation("paper_a", 2, 2, a_id),),
            claims=(P2Claim("C1", "比较结论", (a_id,), comparative=True),),
            refused=False,
            refusal_reason=None,
        )
        validation = validate_claim_citations(
            answer=answer, decision=self.decision, required_paper_keys=("paper_a", "paper_b")
        )
        self.assertFalse(validation.valid)
        self.assertIn("comparative_claim_missing_papers:C1:paper_b", validation.violations)

    def test_gate_refusal_requires_a_refused_answer(self) -> None:
        rejected_gate = CoverageGateDecision(
            g0_document_coverage=False,
            g1_relevant_evidence_coverage=False,
            passed=False,
            missing_document_paper_keys=("paper_b",),
            missing_relevant_evidence_paper_keys=("paper_b",),
            per_paper_candidate_counts=(("paper_a", 1), ("paper_b", 0)),
            per_paper_usable_counts=(("paper_a", 1), ("paper_b", 0)),
            reason="g0_missing_document_coverage",
        )
        decision = RetryThenRefuseDecision(
            selected=(self.a,),
            gate=rejected_gate,
            retry_count=1,
            refused=True,
            refusal_reason="insufficient_evidence:g0_missing_document_coverage",
            per_paper_request_count=(("paper_a", 2), ("paper_b", 2)),
        )
        answer = P2GeneratedAnswer(
            answer="证据不足，不能比较。",
            citations=(),
            claims=(),
            refused=True,
            refusal_reason="insufficient_evidence:g0_missing_document_coverage",
        )
        self.assertTrue(
            validate_claim_citations(answer=answer, decision=decision, required_paper_keys=("paper_a", "paper_b")).valid
        )


if __name__ == "__main__":
    unittest.main()
