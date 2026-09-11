"""P2-B2：均衡召回和 G0/G1 Gate 的离线测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.evidence_control import (  # noqa: E402
    EvidenceCandidate,
    QueryAwareCandidateController,
    balance_candidates,
    evaluate_coverage_gate,
)
from researchpilot.query_plan import P2RulesRouter  # noqa: E402
from researchpilot.schemas import BlockType, Chunk, QueryType, SchemaValidationError, ScoredChunk  # noqa: E402


def _candidate(paper_key: str, number: int, score: float, quality: str = "normal") -> EvidenceCandidate:
    chunk = Chunk.create(
        paper_id=f"runtime_{paper_key}",
        parent_id=None,
        page_start=number,
        page_end=number,
        section_path=("Method",),
        content=f"{paper_key} evidence {number}",
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test",
    )
    return EvidenceCandidate(
        paper_key=paper_key,
        result=ScoredChunk(chunk=chunk, source_name=f"{paper_key}.pdf", rerank_score=score),
        extraction_quality=quality,
    )


class _Provider:
    def __init__(self, records: dict[str, tuple[EvidenceCandidate, ...]]) -> None:
        self.records = records
        self.calls: list[tuple[str, str, bool]] = []

    def retrieve_for_paper(self, *, query: str, paper_key: str, plan: object, expanded: bool = False):  # type: ignore[no-untyped-def]
        self.calls.append((query, paper_key, expanded))
        return self.records.get(paper_key, ())


class _RetryProvider(_Provider):
    def __init__(
        self, initial: dict[str, tuple[EvidenceCandidate, ...]], expanded: dict[str, tuple[EvidenceCandidate, ...]]
    ) -> None:
        super().__init__(initial)
        self.expanded_records = expanded

    def retrieve_for_paper(self, *, query: str, paper_key: str, plan: object, expanded: bool = False):  # type: ignore[no-untyped-def]
        self.calls.append((query, paper_key, expanded))
        return (self.expanded_records if expanded else self.records).get(paper_key, ())


class EvidenceControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = P2RulesRouter().plan(query_type=QueryType.CROSS_DOCUMENT, target_paper_keys=("paper_a", "paper_b"))

    def test_balance_reserves_one_candidate_per_paper_before_global_fill(self) -> None:
        selected = balance_candidates(
            candidates=(
                _candidate("paper_a", 1, 0.99),
                _candidate("paper_a", 2, 0.98),
                _candidate("paper_a", 3, 0.97),
                _candidate("paper_b", 1, 0.10),
            ),
            required_paper_keys=("paper_a", "paper_b"),
            per_paper_quota=1,
            final_limit=3,
        )
        self.assertEqual({candidate.paper_key for candidate in selected}, {"paper_a", "paper_b"})
        self.assertEqual(len(selected), 3)

    def test_gate_distinguishes_g0_missing_document_from_g1_low_quality(self) -> None:
        g0 = evaluate_coverage_gate(
            all_candidates=(_candidate("paper_a", 1, 0.8),),
            selected_candidates=(_candidate("paper_a", 1, 0.8),),
            required_paper_keys=("paper_a", "paper_b"),
            minimum_evidence_per_paper=1,
        )
        self.assertFalse(g0.g0_document_coverage)
        self.assertEqual(g0.reason, "g0_missing_document_coverage")
        g1 = evaluate_coverage_gate(
            all_candidates=(_candidate("paper_a", 1, 0.8), _candidate("paper_b", 1, 0.8, "low_text")),
            selected_candidates=(_candidate("paper_a", 1, 0.8), _candidate("paper_b", 1, 0.8, "low_text")),
            required_paper_keys=("paper_a", "paper_b"),
            minimum_evidence_per_paper=1,
        )
        self.assertTrue(g1.g0_document_coverage)
        self.assertFalse(g1.g1_relevant_evidence_coverage)
        self.assertEqual(g1.missing_relevant_evidence_paper_keys, ("paper_b",))

    def test_controller_calls_each_target_paper_and_never_accepts_cross_scope_results(self) -> None:
        provider = _Provider(
            {
                "paper_a": (_candidate("paper_a", 1, 0.9),),
                "paper_b": (_candidate("paper_b", 1, 0.8),),
            }
        )
        outcome = QueryAwareCandidateController(provider=provider).retrieve_initial(
            query="比较两篇论文", plan=self.plan, final_limit=3
        )
        self.assertEqual([call[1] for call in provider.calls], ["paper_a", "paper_b"])
        self.assertTrue(outcome.gate.passed)
        self.assertEqual({candidate.paper_key for candidate in outcome.selected}, {"paper_a", "paper_b"})

        quota_outcome = QueryAwareCandidateController(provider=provider, per_paper_quota=2).retrieve_initial(
            query="比较两篇论文", plan=self.plan, final_limit=2
        )
        self.assertEqual(len(quota_outcome.selected), 2)

        bad_provider = _Provider({"paper_a": (_candidate("paper_b", 1, 0.9),)})
        with self.assertRaises(SchemaValidationError):
            QueryAwareCandidateController(provider=bad_provider).retrieve_initial(
                query="比较两篇论文", plan=self.plan, final_limit=3
            )

    def test_retry_recovers_missing_paper_once(self) -> None:
        provider = _RetryProvider(
            initial={"paper_a": (_candidate("paper_a", 1, 0.9),)},
            expanded={"paper_b": (_candidate("paper_b", 2, 0.8),)},
        )
        decision = QueryAwareCandidateController(provider=provider).retrieve_with_retry(
            query="比较两篇论文", plan=self.plan, final_limit=3
        )
        self.assertFalse(decision.refused)
        self.assertEqual(decision.retry_count, 1)
        self.assertTrue(decision.gate.passed)
        self.assertEqual([call[2] for call in provider.calls], [False, False, True, True])

    def test_retry_then_refuse_does_not_add_unrelated_citation(self) -> None:
        provider = _RetryProvider(initial={"paper_a": (_candidate("paper_a", 1, 0.9),)}, expanded={})
        decision = QueryAwareCandidateController(provider=provider).retrieve_with_retry(
            query="比较两篇论文", plan=self.plan, final_limit=3
        )
        self.assertTrue(decision.refused)
        self.assertEqual(decision.retry_count, 1)
        self.assertEqual(decision.refusal_reason, "insufficient_evidence:g0_missing_document_coverage")
        self.assertEqual({item.paper_key for item in decision.selected}, {"paper_a"})


if __name__ == "__main__":
    unittest.main()
