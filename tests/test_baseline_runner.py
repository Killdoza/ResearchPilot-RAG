"""Baseline Runner 的记录格式与本地 JSONL 写入测试；不加载模型、不调用 API。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.config import ConfigurationError, build_settings  # noqa: E402
from researchpilot.retrieval import RetrievalResult  # noqa: E402
from researchpilot.schemas import (  # noqa: E402
    Answer,
    BlockType,
    Chunk,
    Citation,
    RetrievalTrace,
    ScoredChunk,
)
from researchpilot.services import AnswerExecution  # noqa: E402


def _load_runner_module() -> object:
    path = PROJECT_ROOT / "scripts" / "run_baseline.py"
    spec = importlib.util.spec_from_file_location("researchpilot_baseline_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def _execution() -> AnswerExecution:
    chunk = Chunk.create(
        paper_id="paper_demo",
        parent_id="block_paper_demo_p0002",
        page_start=2,
        page_end=2,
        section_path=("3 方法",),
        content="论文使用 Hybrid Retrieval 和页码级证据。",
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test_v1",
    )
    scored = ScoredChunk(
        chunk=chunk,
        source_name="demo.pdf",
        dense_rank=1,
        dense_score=0.9,
        sparse_rank=2,
        sparse_score=1.3,
        rerank_score=4.2,
    )
    trace = RetrievalTrace(
        trace_id="trace_runner",
        query="方法是什么？",
        search_query="论文的核心方法是什么？",
        query_type=None,
        candidate_count=4,
        results=(scored,),
        elapsed_ms=123.4,
        stages=("query_expansion", "dense_child_retrieval", "bm25_child_retrieval", "rerank"),
        expanded_queries=("核心方法", "论文的核心方法是什么？"),
    )
    answer = Answer(
        text="论文使用 Hybrid Retrieval [证据 1]。",
        citations=(
            Citation(
                paper_id="paper_demo",
                chunk_id=chunk.chunk_id,
                page_start=2,
                page_end=2,
                evidence=chunk.content,
            ),
        ),
        refused=False,
        trace_id="trace_runner",
        latency_ms=321.0,
    )
    return AnswerExecution(answer=answer, retrieval=RetrievalResult(results=(scored,), trace=trace))


class BaselineRunnerTests(unittest.TestCase):
    def test_record_contains_queries_scores_answer_and_citations(self) -> None:
        record = RUNNER.build_run_record(
            execution=_execution(),
            question="方法是什么？",
            search_query="论文的核心方法是什么？",
            user_id="user_a",
            session_id="session_a",
        )

        self.assertEqual(record["schema_version"], "baseline_run_v1")
        self.assertEqual(record["retrieval"]["expanded_queries"][0], "核心方法")
        self.assertEqual(record["retrieval"]["results"][0]["rerank_score"], 4.2)
        self.assertEqual(record["answer"]["citations"][0]["page_start"], 2)

    def test_write_jsonl_appends_one_valid_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = RUNNER.write_jsonl_record(
                runs_dir=Path(temp_dir),
                record={"run_id": "run_test", "question": "测试"},
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0])["run_id"], "run_test")

    def test_user_id_is_explicit_for_anonymous_browser_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = build_settings(project_root=Path(temp_dir), environ={}, load_dotenv_file=False)
            with self.assertRaises(ConfigurationError):
                RUNNER.resolve_user_id(requested_user_id=None, settings=settings)
            self.assertEqual(RUNNER.resolve_user_id(requested_user_id="user_a", settings=settings), "user_a")


if __name__ == "__main__":
    unittest.main()
