"""ResearchPilot 核心领域模型的确定性单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.schemas import (  # noqa: E402
    Answer,
    BlockType,
    Chunk,
    Citation,
    DocumentBlock,
    Paper,
    QueryType,
    RetrievalTrace,
    SchemaValidationError,
    ScoredChunk,
    sha256_bytes,
    sha256_text,
    to_jsonable,
)


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.content = "本文提出了一种可复现的论文检索方法。"
        self.chunk = Chunk.create(
            paper_id="paper_abc",
            parent_id="parent_01",
            page_start=2,
            page_end=2,
            section_path=("3 方法",),
            content=self.content,
            block_types=(BlockType.PARAGRAPH,),
            chunker_version="fixed_parent_child_v1",
        )

    def test_paper_accepts_content_sha256(self) -> None:
        paper = Paper(
            paper_id="paper_abc",
            source_name="example.pdf",
            content_hash=sha256_text("PDF 二进制内容的测试替身"),
            title="示例论文",
            language="zh",
            page_count=12,
        )
        self.assertEqual(paper.page_count, 12)

    def test_invalid_paper_hash_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "SHA-256"):
            Paper(paper_id="paper_abc", source_name="example.pdf", content_hash="not-a-hash")

    def test_binary_hash_is_stable(self) -> None:
        self.assertEqual(
            sha256_bytes(b"pdf bytes"),
            "d1cb546b102fab8362de413fdacc187b05be10df72b72db3b3e50b4953f6a555",
        )

    def test_document_block_requires_page_and_structured_type(self) -> None:
        block = DocumentBlock(
            paper_id="paper_abc",
            page=1,
            block_id="block_001",
            block_type=BlockType.HEADING,
            text="1 引言",
            section_path=("1 引言",),
            parser_name="pdfplumber",
            parser_version="0.11.9",
        )
        self.assertEqual(block.block_type, BlockType.HEADING)
        with self.assertRaisesRegex(SchemaValidationError, "页码"):
            DocumentBlock(
                paper_id="paper_abc",
                page=0,
                block_id="block_bad",
                block_type=BlockType.PARAGRAPH,
                text="无效页码",
            )

    def test_chunk_id_and_hash_are_deterministic(self) -> None:
        same_chunk = Chunk.create(
            paper_id="paper_abc",
            parent_id="parent_01",
            page_start=2,
            page_end=2,
            section_path=("3 方法",),
            content=self.content,
            block_types=(BlockType.PARAGRAPH,),
            chunker_version="fixed_parent_child_v1",
        )
        changed_chunk = Chunk.create(
            paper_id="paper_abc",
            parent_id="parent_01",
            page_start=2,
            page_end=2,
            section_path=("3 方法",),
            content=self.content + " 新内容。",
            block_types=(BlockType.PARAGRAPH,),
            chunker_version="fixed_parent_child_v1",
        )
        self.assertEqual(self.chunk.chunk_id, same_chunk.chunk_id)
        self.assertEqual(self.chunk.content_hash, sha256_text(self.content))
        self.assertEqual(self.chunk.index_metadata()["paper_id"], "paper_abc")
        self.assertNotEqual(self.chunk.chunk_id, changed_chunk.chunk_id)

    def test_chunk_occurrence_index_disambiguates_repeated_content(self) -> None:
        first = Chunk.create(
            paper_id="paper_demo",
            parent_id="block_demo_p0001",
            page_start=1,
            page_end=1,
            section_path=(),
            content="重复的文本",
            block_types=(BlockType.PARAGRAPH,),
            chunker_version="test_v1",
            occurrence_index=0,
        )
        second = Chunk.create(
            paper_id="paper_demo",
            parent_id="block_demo_p0001",
            page_start=1,
            page_end=1,
            section_path=(),
            content="重复的文本",
            block_types=(BlockType.PARAGRAPH,),
            chunker_version="test_v1",
            occurrence_index=1,
        )

        self.assertNotEqual(first.chunk_id, second.chunk_id)

    def test_chunk_rejects_hash_that_does_not_match_content(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "不一致"):
            Chunk(
                chunk_id="chunk_bad",
                paper_id="paper_abc",
                parent_id=None,
                page_start=1,
                page_end=1,
                section_path=(),
                content="真实文本",
                block_types=(BlockType.PARAGRAPH,),
                content_hash=sha256_text("其他文本"),
                chunker_version="v1",
            )

    def test_retrieval_trace_retains_all_stage_scores(self) -> None:
        scored = ScoredChunk(
            chunk=self.chunk,
            source_name="example.pdf",
            dense_rank=2,
            dense_score=0.72,
            sparse_rank=1,
            sparse_score=8.5,
            fusion_score=0.031,
            rerank_score=4.2,
        )
        trace = RetrievalTrace(
            trace_id="trace_001",
            query="该方法解决什么问题？",
            search_query="论文提出的方法解决什么问题？",
            query_type=QueryType.SEMANTIC_EXPLANATION,
            candidate_count=5,
            results=(scored,),
            elapsed_ms=31.4,
            stages=("query_rewrite", "dense", "bm25", "rerank"),
            experiment_mode="end_to_end",
            query_type_source="rule_predicted",
            scope_source="title_resolved",
            target_paper_keys=("paper_abc",),
        )
        serialised = to_jsonable(trace)
        self.assertEqual(serialised["query_type"], "semantic_explanation")
        self.assertEqual(serialised["results"][0]["chunk"]["page_start"], 2)
        self.assertEqual(serialised["results"][0]["source_name"], "example.pdf")
        self.assertEqual(serialised["results"][0]["rerank_score"], 4.2)
        self.assertEqual(serialised["scope_source"], "title_resolved")

    def test_answer_requires_reason_when_refused(self) -> None:
        citation = Citation(
            paper_id="paper_abc",
            chunk_id=self.chunk.chunk_id,
            page_start=2,
            page_end=2,
            evidence=self.content,
        )
        answer = Answer(
            text="该方法通过可复现实验提高检索可靠性。",
            citations=(citation,),
            refused=False,
            trace_id="trace_001",
            latency_ms=250.0,
        )
        self.assertFalse(answer.refused)
        with self.assertRaisesRegex(SchemaValidationError, "refusal_reason"):
            Answer(
                text="论文中没有这一信息。",
                citations=(),
                refused=True,
                trace_id="trace_002",
                latency_ms=10.0,
            )


if __name__ == "__main__":
    unittest.main()
