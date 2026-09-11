"""Small, deterministic helpers for evidence-state traces."""

from __future__ import annotations

import re

from .absence_detection import FullDocumentScan
from .schemas import Answer, RetrievalTrace

_ROLE_PATTERNS = {
    "problem": re.compile(r"研究问题|问题|动机|挑战|problem|motivation|challenge", re.I),
    "method": re.compile(r"方法|模型|架构|检索|索引|method|model|architecture|retrieval", re.I),
    "setup": re.compile(r"实验设置|数据集|基线|指标|setup|dataset|baseline|metric", re.I),
    "results": re.compile(r"实验结果|结果|性能|消融|result|performance|ablation", re.I),
    "conclusion": re.compile(r"结论|局限|限制|conclusion|limitation", re.I),
}


def full_document_scan_stages(scan: FullDocumentScan) -> tuple[str, ...]:
    return (
        "full_document_scan",
        f"field_type:{scan.field_type.value}",
        f"scanned_block_count:{scan.scanned_block_count}",
        f"matched_block_count:{scan.matched_block_count}",
    )


def cited_role_stages(answer: Answer, trace: RetrievalTrace) -> tuple[str, ...]:
    if not answer.citations:
        return ()
    evidence = "\n".join(f"{' > '.join(item.section_path)}\n{item.evidence[:500]}" for item in answer.citations)
    roles = tuple(role for role, pattern in _ROLE_PATTERNS.items() if pattern.search(evidence))
    selected = {stage.split(":", 1)[1] for stage in trace.stages if stage.startswith("role_selected:")}
    return tuple(f"role_cited:{role}" for role in roles if not selected or role in selected)
