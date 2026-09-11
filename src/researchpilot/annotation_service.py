"""User annotation capture and paper-fact conflict presentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .storage import PaperAnnotation, SQLiteMemoryRepository

AnnotationMode = Literal["paper_only", "with_user_notes"]


def infer_annotation_relation(text: str) -> str | None:
    """Conservative local classifier; ordinary questions are not auto-saved."""
    lowered = text.strip().lower()
    if any(token in lowered for token in ("这里错", "不正确", "不同意", "有问题", "contradict", "incorrect")):
        return "contradicts"
    if any(token in lowered for token in ("我的批注", "记一下", "标注", "补充说明", "note that")):
        return "comments_on"
    if any(token in lowered for token in ("我认为这里", "这个结论值得", "需要核验")):
        return "questions"
    return None


@dataclass
class AnnotationService:
    repository: SQLiteMemoryRepository

    def capture(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        chunk_id: str,
        content: str,
        relation: str = "comments_on",
        note_type: str = "comment",
        anchor_page: int | None = None,
        section_path: tuple[str, ...] = (),
    ) -> int:
        return self.repository.add_annotation(
            PaperAnnotation(
                annotation_id=None,
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
                chunk_id=chunk_id,
                content=content,
                relation=relation,
                note_type=note_type,
                anchor_page=anchor_page,
                section_path=section_path,
            )
        )

    def context_for_chunks(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        chunk_ids: tuple[str, ...],
        mode: AnnotationMode,
        pages: tuple[int, ...] = (),
        section_paths: tuple[tuple[str, ...], ...] = (),
        query: str = "",
    ) -> str:
        if mode == "paper_only":
            return ""
        annotations = self.repository.search_annotations(
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
            chunk_ids=chunk_ids,
            pages=pages,
            section_paths=section_paths,
            query=query,
            limit=3,
        )
        if not annotations:
            return ""
        labels = {
            "comments_on": "补充说明",
            "contradicts": "质疑/冲突",
            "supports": "支持",
            "questions": "待核验问题",
        }
        rows = ["【用户批注】以下内容不是论文原文，也不能覆盖论文事实："]
        rows.extend(
            f"- 页码 {item.anchor_page or '未知'} · "
            f"{' > '.join(item.section_path) or '章节未知'} · "
            f"{labels.get(item.relation, item.relation)}：{item.content}"
            for item in annotations
        )
        rows.append("【关系规则】回答必须先陈述论文原文，再单独陈述用户批注；冲突时明确标记。")
        return "\n".join(rows)
