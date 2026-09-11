"""P1-E 的上下文选择器：全文 Long-context 与 E5 检索上下文共用同一 Block 来源。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .evaluation import CorpusManifest, EvaluationDataError, paper_ids_from_manifest
from .retrieval import RetrievalResult


@dataclass(frozen=True)
class ContextReference:
    """模型可引用的一段上下文及其可审计的论文页码。"""

    reference_id: str
    paper_key: str
    paper_title: str
    page_start: int
    page_end: int
    block_id: str
    section_path: tuple[str, ...]
    text: str
    source_kind: str  # full_block / retrieved_parent

    def __post_init__(self) -> None:
        if not self.reference_id or not self.paper_key or not self.block_id or not self.text.strip():
            raise EvaluationDataError("上下文引用必须具有非空 ID、论文键、Block ID 和文本。")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise EvaluationDataError("上下文引用的页码范围非法。")
        if self.source_kind not in {"full_block", "retrieved_parent"}:
            raise EvaluationDataError("未知上下文来源类型。")


@dataclass(frozen=True)
class ContextBundle:
    """一次回答实际可见的上下文。它不包含 Prompt 或模型响应。"""

    mode: str
    references: tuple[ContextReference, ...]
    selection_note: str
    retrieval_latency_ms: float = 0.0
    retrieval_trace_id: str | None = None

    @property
    def character_count(self) -> int:
        return len(self.render())

    def render(self) -> str:
        parts: list[str] = []
        for reference in self.references:
            pages = (
                str(reference.page_start)
                if reference.page_start == reference.page_end
                else f"{reference.page_start}-{reference.page_end}"
            )
            section = " > ".join(reference.section_path) or "未标注章节"
            parts.append(
                f"[CTX:{reference.reference_id} | paper_key={reference.paper_key} | "
                f"page={pages} | section={section} | kind={reference.source_kind}]\n"
                f"{reference.text.strip()}"
            )
        return "\n\n".join(parts)


class E5BlockRepository:
    """读取 P1-D 已冻结的 Docling Block 工件；不重新解析 PDF。"""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        corpus: CorpusManifest,
        expected_parser_name: str = "docling",
        expected_parser_version: str = "2.120.2",
    ) -> None:
        self._artifact_dir = artifact_dir
        self._papers = {paper.paper_key: paper for paper in corpus.papers}
        self._expected_parser_name = expected_parser_name
        self._expected_parser_version = expected_parser_version
        self._cache: dict[str, tuple[ContextReference, ...]] = {}

    def all_blocks(self, *, paper_keys: Sequence[str]) -> tuple[ContextReference, ...]:
        references: list[ContextReference] = []
        for paper_key in paper_keys:
            references.extend(self._load_paper(paper_key))
        if not references:
            raise EvaluationDataError("Long-context 模式没有得到任何 E5 Block；拒绝调用模型。")
        return tuple(references)

    def _load_paper(self, paper_key: str) -> tuple[ContextReference, ...]:
        if paper_key in self._cache:
            return self._cache[paper_key]
        paper = self._papers.get(paper_key)
        if paper is None or not paper.sha256:
            raise EvaluationDataError(f"E5 Block 找不到已冻结论文：{paper_key}")
        artifact_path = self._artifact_dir / f"{paper.sha256}.json"
        if not artifact_path.is_file():
            raise EvaluationDataError(
                f"缺少 E5 Block 工件：{artifact_path}。请先用 e5_structure_aware 配置重新导入评测语料。"
            )
        try:
            raw = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise EvaluationDataError(f"E5 Block 工件不是合法 JSON：{artifact_path}") from error
        if not isinstance(raw, Mapping) or raw.get("schema_version") != "parsed_blocks_v1":
            raise EvaluationDataError(f"E5 Block 工件 Schema 不正确：{artifact_path}")
        paper_info = raw.get("paper")
        blocks = raw.get("blocks")
        if not isinstance(paper_info, Mapping) or paper_info.get("content_hash") != paper.sha256:
            raise EvaluationDataError(f"E5 Block 工件的论文哈希不匹配：{artifact_path}")
        if not isinstance(blocks, list) or not blocks:
            raise EvaluationDataError(f"E5 Block 工件为空：{artifact_path}")
        references: list[ContextReference] = []
        for position, block in enumerate(blocks, start=1):
            if not isinstance(block, Mapping):
                raise EvaluationDataError(f"{artifact_path} 第 {position} 个 Block 不是对象。")
            if (
                block.get("parser_name") != self._expected_parser_name
                or block.get("parser_version") != self._expected_parser_version
            ):
                raise EvaluationDataError(f"{artifact_path} 不是冻结的 E5 Docling Block。")
            text = block.get("text")
            page = block.get("page")
            block_id = block.get("block_id")
            section_path = block.get("section_path", [])
            if (
                not isinstance(text, str)
                or not text.strip()
                or not isinstance(page, int)
                or page < 1
                or not isinstance(block_id, str)
                or not block_id
                or not isinstance(section_path, list)
                or any(not isinstance(item, str) or not item.strip() for item in section_path)
            ):
                raise EvaluationDataError(f"{artifact_path} 第 {position} 个 Block 字段非法。")
            references.append(
                ContextReference(
                    reference_id=f"B{position:04d}",
                    paper_key=paper_key,
                    paper_title=paper.title,
                    page_start=page,
                    page_end=page,
                    block_id=block_id,
                    section_path=tuple(section_path),
                    text=text,
                    source_kind="full_block",
                )
            )
        self._cache[paper_key] = tuple(references)
        return self._cache[paper_key]


def full_context(*, mode: str, block_repository: E5BlockRepository, paper_keys: Sequence[str]) -> ContextBundle:
    """构建 LC1 或 LC2：不触发任何向量/BM25 检索。"""
    if mode not in {"lc1", "lc2"}:
        raise EvaluationDataError("全文上下文只支持 lc1 或 lc2。")
    if mode == "lc1" and len(paper_keys) != 1:
        raise EvaluationDataError("LC1 每次只能输入一篇论文；跨论文题必须使用 LC2。")
    references = tuple(
        replace(reference, reference_id=f"B{position:04d}")
        for position, reference in enumerate(block_repository.all_blocks(paper_keys=paper_keys), start=1)
    )
    return ContextBundle(
        mode=mode,
        references=references,
        selection_note=("单论文全部 E5 Docling Block" if mode == "lc1" else "题目涉及论文的全部 E5 Docling Block"),
    )


def retrieval_context(*, mode: str, retrieval: RetrievalResult, corpus: CorpusManifest) -> ContextBundle:
    """将 N0/A0 的检索父 Chunk 转为与全文模式相同的可引用格式。"""
    if mode not in {"n0", "a0"}:
        raise EvaluationDataError("检索上下文只支持 n0 或 a0。")
    paper_ids = paper_ids_from_manifest(corpus)
    paper_keys_by_id = {paper_id: paper_key for paper_key, paper_id in paper_ids.items()}
    papers_by_key = {paper.paper_key: paper for paper in corpus.papers}
    references: list[ContextReference] = []
    for position, scored in enumerate(retrieval.results, start=1):
        chunk = scored.chunk
        paper_key = paper_keys_by_id.get(chunk.paper_id)
        if paper_key is None:
            raise EvaluationDataError(f"检索结果引用了评测语料外 paper_id：{chunk.paper_id}")
        references.append(
            ContextReference(
                reference_id=f"R{position:02d}",
                paper_key=paper_key,
                paper_title=papers_by_key[paper_key].title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                block_id=chunk.chunk_id,
                section_path=chunk.section_path,
                text=chunk.content,
                source_kind="retrieved_parent",
            )
        )
    return ContextBundle(
        mode=mode,
        references=tuple(references),
        selection_note=("Dense Top-K 父 Chunk" if mode == "n0" else "Dense + BM25 + RRF + Rerank 父 Chunk"),
        retrieval_latency_ms=retrieval.trace.elapsed_ms,
        retrieval_trace_id=retrieval.trace.trace_id,
    )
