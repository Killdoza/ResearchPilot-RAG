"""Unit tests for deterministic multi-section evidence planning."""

from researchpilot.evidence_planner import EvidenceRole, plan_multi_section_evidence


def test_method_result_relation_reserves_both_roles() -> None:
    plan = plan_multi_section_evidence("图索引、双层检索与实验优势之间有什么联系？")

    assert tuple(item.role for item in plan.requirements) == (
        EvidenceRole.METHOD,
        EvidenceRole.RESULTS,
    )
    assert plan.planner_version == "multi_section_rules_v1"


def test_problem_method_correspondence_reserves_both_roles() -> None:
    plan = plan_multi_section_evidence("方法设计如何对应论文提出的研究问题？")

    assert tuple(item.role for item in plan.requirements) == (
        EvidenceRole.PROBLEM,
        EvidenceRole.METHOD,
    )


def test_vague_relation_uses_safe_two_role_default() -> None:
    plan = plan_multi_section_evidence("这两部分是什么关系？")

    assert {item.role for item in plan.requirements} == {
        EvidenceRole.PROBLEM,
        EvidenceRole.METHOD,
    }
