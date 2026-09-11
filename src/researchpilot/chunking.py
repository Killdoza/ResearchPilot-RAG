"""可替换的 Chunker：把结构化 DocumentBlock 转换为 Parent-Child Chunk。"""

from __future__ import annotations

from dataclasses import dataclass

from .parsing import clean_extracted_text
from .schemas import BlockType, Chunk, DocumentBlock

_SENTENCE_BOUNDARIES = ("\n\n", "\n", "。", "！", "？", ". ", "; ", ", ", " ")
_ATOMIC_BLOCK_TYPES = {BlockType.TABLE, BlockType.FORMULA, BlockType.FIGURE_CAPTION}


def split_text(text: str, *, chunk_size: int, overlap: int) -> tuple[str, ...]:
    """确定性长度切分，仅在一个结构 Parent 过长时使用。"""
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return ()
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        target_end = min(start + chunk_size, len(cleaned))
        end = target_end
        if target_end < len(cleaned):
            boundary = max(
                (cleaned.rfind(token, start + 1, target_end + 1) for token in _SENTENCE_BOUNDARIES),
                default=-1,
            )
            if boundary > start:
                end = boundary + 1
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return tuple(chunks)


@dataclass(frozen=True)
class _StructuralGroup:
    group_id: str
    paper_id: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    block_types: tuple[BlockType, ...]
    content: str


def _render_content(blocks: list[DocumentBlock], section_path: tuple[str, ...]) -> str:
    parts: list[str] = []
    if section_path:
        parts.append("[章节] " + " > ".join(section_path))
    parts.extend(block.text for block in blocks)
    return "\n\n".join(parts).strip()


def _group_blocks(blocks: tuple[DocumentBlock, ...]) -> tuple[_StructuralGroup, ...]:
    """按页码和章节合并相邻段落；表格/公式/图注保持原子块。"""
    groups: list[_StructuralGroup] = []
    buffered: list[DocumentBlock] = []
    current_page: int | None = None
    current_section: tuple[str, ...] = ()
    group_index = 0

    def flush() -> None:
        nonlocal buffered, current_page, current_section, group_index
        if not buffered:
            return
        content = _render_content(buffered, current_section)
        if content:
            groups.append(
                _StructuralGroup(
                    group_id=f"struct_{buffered[0].block_id}_{group_index:04d}",
                    paper_id=buffered[0].paper_id,
                    page_start=min(block.page for block in buffered),
                    page_end=max(block.page for block in buffered),
                    section_path=current_section,
                    block_types=tuple(dict.fromkeys(block.block_type for block in buffered)),
                    content=content,
                )
            )
            group_index += 1
        buffered = []
        current_page = None
        current_section = ()

    for block in blocks:
        if block.block_type is BlockType.HEADING:
            flush()
            continue
        if block.block_type in _ATOMIC_BLOCK_TYPES:
            flush()
            content = _render_content([block], block.section_path)
            if content:
                groups.append(
                    _StructuralGroup(
                        group_id=f"struct_{block.block_id}_{group_index:04d}",
                        paper_id=block.paper_id,
                        page_start=block.page,
                        page_end=block.page,
                        section_path=block.section_path,
                        block_types=(block.block_type,),
                        content=content,
                    )
                )
                group_index += 1
            continue
        if buffered and (block.page != current_page or block.section_path != current_section):
            flush()
        if not buffered:
            current_page = block.page
            current_section = block.section_path
        buffered.append(block)
    flush()
    return tuple(groups)


def build_structure_aware_parent_child_chunks(
    blocks: tuple[DocumentBlock, ...],
    *,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
    chunker_version: str = "structure_aware_v1",
) -> tuple[tuple[Chunk, ...], tuple[Chunk, ...]]:
    """构建标题/段落/表格感知的 Parent-Child Chunk。

    普通段落只在“同页 + 同章节”内合并，避免仅因扩大页码范围而放大页级评测命中。
    """
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for group in _group_blocks(blocks):
        for parent_index, parent_text in enumerate(
            split_text(
                group.content,
                chunk_size=parent_chunk_size,
                overlap=parent_chunk_overlap,
            )
        ):
            parent = Chunk.create(
                paper_id=group.paper_id,
                parent_id=group.group_id,
                page_start=group.page_start,
                page_end=group.page_end,
                section_path=group.section_path,
                content=parent_text,
                block_types=group.block_types,
                chunker_version=chunker_version,
                occurrence_index=parent_index,
            )
            parents.append(parent)
            for child_index, child_text in enumerate(
                split_text(
                    parent.content,
                    chunk_size=child_chunk_size,
                    overlap=child_chunk_overlap,
                )
            ):
                children.append(
                    Chunk.create(
                        paper_id=group.paper_id,
                        parent_id=parent.chunk_id,
                        page_start=group.page_start,
                        page_end=group.page_end,
                        section_path=group.section_path,
                        content=child_text,
                        block_types=group.block_types,
                        chunker_version=chunker_version,
                        occurrence_index=child_index,
                    )
                )
    return tuple(parents), tuple(children)
