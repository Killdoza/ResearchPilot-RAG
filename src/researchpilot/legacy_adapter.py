"""Read-only compatibility boundary for pre-ResearchPilot sessions.

The original Streamlit demo stored only filenames and page fragments.  Those
records have no stable paper ID, parser/chunker provenance, or page-level
citation contract.  They may be displayed for continuity, but are never
silently fed into the current product retrieval pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegacySessionStatus:
    has_legacy_documents: bool
    has_structured_documents: bool
    source_names: tuple[str, ...]

    @property
    def requires_reupload(self) -> bool:
        """A session containing only legacy documents cannot answer safely."""
        return self.has_legacy_documents and not self.has_structured_documents

    @property
    def banner_text(self) -> str:
        names = "、".join(self.source_names[:3]) or "该历史论文"
        if self.requires_reupload:
            return (
                f"检测到旧版会话论文（{names}）。历史消息仍可查看，但旧索引缺少 "
                "paper_id、解析版本和页码证据边界，已进入只读兼容模式；请新建会话后重新上传论文。"
            )
        return f"当前会话还包含旧版论文记录（{names}）。它们不会混入新版可追溯检索；请重新上传这些论文以获得页码引用。"

    @property
    def answer_text(self) -> str:
        return (
            "该历史会话中的论文来自旧 Demo 索引，缺少可验证的 paper_id、解析版本和页码级证据。"
            "为避免把旧检索结果伪装成新版引用回答，本轮不执行兼容检索。请新建会话并重新上传该 PDF。"
        )


class LegacySessionAdapter:
    """Classify legacy session metadata without loading a vector store or LLM."""

    @staticmethod
    def inspect(active_documents: Sequence[dict[str, Any]]) -> LegacySessionStatus:
        legacy_names: list[str] = []
        structured = False
        for document in active_documents:
            source_name = document.get("source_name")
            paper_id = document.get("paper_id")
            if isinstance(source_name, str) and source_name.strip() and isinstance(paper_id, str) and paper_id.strip():
                structured = True
            elif isinstance(source_name, str) and source_name.strip():
                legacy_names.append(source_name)
        return LegacySessionStatus(
            has_legacy_documents=bool(legacy_names),
            has_structured_documents=structured,
            source_names=tuple(dict.fromkeys(legacy_names)),
        )
