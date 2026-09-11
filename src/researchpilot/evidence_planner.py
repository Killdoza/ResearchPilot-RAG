"""Deterministic evidence-role planning for multi-section paper questions.

This module does not read evaluation Gold and does not call an LLM.  It turns
the wording of a runtime question into auditable evidence roles so retrieval
can reserve context for every part of a synthesis instead of relying on one
global Top-K list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class EvidenceRole(str, Enum):
    PROBLEM = "problem"
    METHOD = "method"
    RESULTS = "results"
    CONCLUSION = "conclusion"


@dataclass(frozen=True)
class EvidenceRequirement:
    role: EvidenceRole
    search_query: str
    minimum_parents: int = 1

    def __post_init__(self) -> None:
        if not self.search_query.strip():
            raise ValueError("search_query 不能为空。")
        if self.minimum_parents < 1:
            raise ValueError("minimum_parents 必须大于等于 1。")


@dataclass(frozen=True)
class MultiSectionEvidencePlan:
    requirements: tuple[EvidenceRequirement, ...]
    planner_version: str = "multi_section_rules_v1"

    def __post_init__(self) -> None:
        if len(self.requirements) < 2:
            raise ValueError("跨章节证据计划至少需要两个证据角色。")
        roles = tuple(item.role for item in self.requirements)
        if len(set(roles)) != len(roles):
            raise ValueError("跨章节证据角色不能重复。")


_ROLE_QUERY = {
    EvidenceRole.PROBLEM: "research problem motivation challenge introduction 研究问题 动机 挑战",
    EvidenceRole.METHOD: "method architecture framework indexing retrieval 方法 架构 索引 检索",
    EvidenceRole.RESULTS: "experimental results evaluation ablation performance 实验结果 评估 消融 性能",
    EvidenceRole.CONCLUSION: "conclusion discussion limitations 结论 讨论 局限",
}


def plan_multi_section_evidence(question: str) -> MultiSectionEvidencePlan:
    """Create an explainable role plan using only the runtime question."""
    text = question.casefold()
    requested: list[EvidenceRole] = []

    def add(role: EvidenceRole) -> None:
        if role not in requested:
            requested.append(role)

    if re.search(r"研究问题|动机|挑战|为什么|problem|motivation|challenge", text):
        add(EvidenceRole.PROBLEM)
    if re.search(r"方法|设计|模型|架构|图索引|检索|method|design|architecture|retrieval", text):
        add(EvidenceRole.METHOD)
    if re.search(r"实验|结果|优势|效果|性能|消融|result|experiment|performance|ablation", text):
        add(EvidenceRole.RESULTS)
    if re.search(r"结论|局限|限制|讨论|conclusion|limitation|discussion", text):
        add(EvidenceRole.CONCLUSION)

    # A vague relation/correspondence question still needs two independently
    # grounded sides.  The default is problem -> method, not arbitrary Top-K.
    if len(requested) < 2:
        add(EvidenceRole.PROBLEM)
        add(EvidenceRole.METHOD)

    requirements = tuple(
        EvidenceRequirement(
            role=role,
            search_query=f"{question.strip()} {_ROLE_QUERY[role]}",
        )
        for role in requested
    )
    return MultiSectionEvidencePlan(requirements=requirements)
