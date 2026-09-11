"""P2 到持久化 Hybrid Retriever 的论文范围适配测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.p2_retrieval import PersistentHybridPerPaperProvider  # noqa: E402
from researchpilot.query_plan import P2RulesRouter  # noqa: E402
from researchpilot.schemas import BlockType, Chunk, QueryType, ScoredChunk  # noqa: E402


def _scored() -> ScoredChunk:
    chunk = Chunk.create(
        paper_id="runtime_a",
        parent_id=None,
        page_start=1,
        page_end=1,
        section_path=("Method",),
        content="method evidence",
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test",
    )
    return ScoredChunk(chunk=chunk, source_name="a.pdf", rerank_score=0.9)


class _Retriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(results=(_scored(),))


class P2RetrievalTests(unittest.TestCase):
    def test_provider_forwards_plan_budget_and_paper_scope(self) -> None:
        retriever = _Retriever()
        provider = PersistentHybridPerPaperProvider(
            retriever=retriever,  # type: ignore[arg-type]
            paper_id_by_key={"paper_a": "runtime_a"},
            user_id="eval_user",
            session_id="eval_session",
        )
        plan = P2RulesRouter().plan(query_type=QueryType.CROSS_DOCUMENT, target_paper_keys=("paper_a", "paper_b"))
        candidates = provider.retrieve_for_paper(query="question", paper_key="paper_a", plan=plan, expanded=True)
        self.assertEqual(candidates[0].paper_key, "paper_a")
        self.assertEqual(retriever.calls[0]["paper_id"], "runtime_a")
        self.assertEqual(retriever.calls[0]["dense_limit_per_query"], 24)
        self.assertEqual(retriever.calls[0]["bm25_limit_per_query"], 24)
        self.assertEqual(retriever.calls[0]["result_limit"], 24)
        self.assertTrue(retriever.calls[0]["rerank_enabled"])


if __name__ == "__main__":
    unittest.main()
