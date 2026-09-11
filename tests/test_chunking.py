"""结构感知 Parent-Child Chunker 的单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.chunking import build_structure_aware_parent_child_chunks  # noqa: E402
from researchpilot.schemas import BlockType, DocumentBlock  # noqa: E402


def _block(*, page: int, block_id: str, block_type: BlockType, text: str, section: tuple[str, ...]) -> DocumentBlock:
    return DocumentBlock(
        paper_id="paper_demo",
        page=page,
        block_id=block_id,
        block_type=block_type,
        text=text,
        section_path=section,
        parser_name="docling",
        parser_version="test",
    )


class StructureAwareChunkingTests(unittest.TestCase):
    def test_keeps_heading_context_and_does_not_cross_page_boundary(self) -> None:
        blocks = (
            _block(page=1, block_id="h1", block_type=BlockType.HEADING, text="3 Method", section=("3 Method",)),
            _block(
                page=1, block_id="p1", block_type=BlockType.PARAGRAPH, text="第一段解释方法。", section=("3 Method",)
            ),
            _block(
                page=1, block_id="p2", block_type=BlockType.PARAGRAPH, text="第二段补充方法。", section=("3 Method",)
            ),
            _block(
                page=2, block_id="p3", block_type=BlockType.PARAGRAPH, text="下一页仍属于方法。", section=("3 Method",)
            ),
        )
        parents, children = build_structure_aware_parent_child_chunks(
            blocks,
            parent_chunk_size=100,
            parent_chunk_overlap=0,
            child_chunk_size=30,
            child_chunk_overlap=0,
        )

        self.assertEqual(len(parents), 2)
        self.assertTrue(all("[章节] 3 Method" in parent.content for parent in parents))
        self.assertEqual([(parent.page_start, parent.page_end) for parent in parents], [(1, 1), (2, 2)])
        self.assertTrue(all(child.parent_id in {parent.chunk_id for parent in parents} for child in children))

    def test_table_is_an_atomic_parent_with_section_context(self) -> None:
        blocks = (
            _block(
                page=3,
                block_id="t",
                block_type=BlockType.TABLE,
                text="| Model | MRR |\n| A | 0.8 |",
                section=("5 Results",),
            ),
            _block(page=3, block_id="p", block_type=BlockType.PARAGRAPH, text="表格后的解释。", section=("5 Results",)),
        )
        parents, _ = build_structure_aware_parent_child_chunks(
            blocks,
            parent_chunk_size=200,
            parent_chunk_overlap=0,
            child_chunk_size=100,
            child_chunk_overlap=0,
        )

        table_parent = next(parent for parent in parents if BlockType.TABLE in parent.block_types)
        self.assertIn("[章节] 5 Results", table_parent.content)
        self.assertIn("| Model | MRR |", table_parent.content)
        self.assertNotIn("表格后的解释", table_parent.content)

    def test_formula_is_an_atomic_parent_with_section_context(self) -> None:
        blocks = (
            _block(
                page=4,
                block_id="f",
                block_type=BlockType.FORMULA,
                text="PE(pos, 2i) = sin(pos / 10000)",
                section=("3.5 Positional Encoding",),
            ),
            _block(
                page=4,
                block_id="p",
                block_type=BlockType.PARAGRAPH,
                text="公式后的解释。",
                section=("3.5 Positional Encoding",),
            ),
        )
        parents, _ = build_structure_aware_parent_child_chunks(
            blocks,
            parent_chunk_size=200,
            parent_chunk_overlap=0,
            child_chunk_size=100,
            child_chunk_overlap=0,
        )

        formula_parent = next(parent for parent in parents if BlockType.FORMULA in parent.block_types)
        self.assertIn("[章节] 3.5 Positional Encoding", formula_parent.content)
        self.assertIn("PE(pos, 2i)", formula_parent.content)
        self.assertNotIn("公式后的解释", formula_parent.content)


if __name__ == "__main__":
    unittest.main()
