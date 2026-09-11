"""P2 v0.2 跨论文评测集的数据契约与冻结校验。

P1-E 的 ``crossdoc.py`` 记录了跨论文答案与页级引用。P2 在此基础上额外
记录目标论文覆盖、可比较性和拒答目标，以便后续实现 Coverage Gate 与
``retry_then_refuse``。本模块保持纯数据逻辑：不访问 Chroma、LLM 或 UI。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .crossdoc import CrossDocumentEvidence, ExpectedClaim
from .evaluation import CorpusManifest, EvaluationDataError

P2_SCHEMA_VERSION = "ai_cs_p2_eval_v0.2"
P2_MANIFEST_SCHEMA_VERSION = "ai_cs_p2_manifest_v0.2"
P2_SPLITS = frozenset({"dev", "test"})
P2_REFUSAL_REASONS = frozenset(
    {"comparison_not_allowed", "incompatible_setup", "not_reported", "insufficient_evidence"}
)

P2_PAPER_SPLITS: dict[str, str] = {
    "self_rag_2023": "dev",
    "r2ag_2024": "dev",
    "rag_foundry_2024": "dev",
    "llms_meet_nlp_survey_2024": "dev",
    "agile_requirements_2024": "dev",
    "vhelm_2024": "dev",
    "efficient_rag_2024": "test",
    "lightrag_2024": "test",
    "seed_bench_2024": "test",
    "mmt_bench_2024": "test",
}


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDataError(f"{field_name} 必须是非空字符串。")
    return value.strip()


def _identifier(value: object, field_name: str) -> str:
    result = _text(value, field_name)
    if len(result) > 128 or not all(char.isalnum() or char in "_.-" for char in result):
        raise EvaluationDataError(f"{field_name} 只能包含字母、数字、点、下划线或连字符。")
    return result


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationDataError(f"{field_name} 必须是对象。")
    return value


@dataclass(frozen=True)
class P2Case:
    """一条用于 Query-aware Evidence-Constrained Retrieval 的跨论文题。"""

    case_id: str
    source_id: str
    split: str
    task_type: str
    answerable: bool
    paper_keys: tuple[str, ...]
    required_paper_keys: tuple[str, ...]
    comparison_dimension: str
    comparison_allowed: bool
    refusal_reason_target: str | None
    question: str
    expected_answer: str
    expected_claims: tuple[ExpectedClaim, ...]
    gold_evidence: tuple[CrossDocumentEvidence, ...]
    annotation_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.case_id, "id")
        _identifier(self.source_id, "source_id")
        if self.split not in P2_SPLITS:
            raise EvaluationDataError("P2 split 必须是 dev 或 test。")
        if self.task_type not in {"cross_document", "unanswerable"}:
            raise EvaluationDataError("P2 task_type 必须是 cross_document 或 unanswerable。")
        if not isinstance(self.answerable, bool):
            raise EvaluationDataError("P2 answerable 必须是布尔值。")
        if len(self.paper_keys) < 2 or len(set(self.paper_keys)) != len(self.paper_keys):
            raise EvaluationDataError("P2 跨论文题必须包含至少两篇不重复论文。")
        if len(self.required_paper_keys) < 2 or len(set(self.required_paper_keys)) != len(self.required_paper_keys):
            raise EvaluationDataError("required_paper_keys 必须包含至少两篇不重复目标论文。")
        for key in (*self.paper_keys, *self.required_paper_keys):
            _identifier(key, "paper_key")
        if not set(self.required_paper_keys).issubset(self.paper_keys):
            raise EvaluationDataError("required_paper_keys 必须是 paper_keys 的子集。")
        _text(self.comparison_dimension, "comparison_dimension")
        if not isinstance(self.comparison_allowed, bool):
            raise EvaluationDataError("comparison_allowed 必须是布尔值。")
        _text(self.question, "question")
        _text(self.expected_answer, "expected_answer")
        if any(not isinstance(claim, ExpectedClaim) for claim in self.expected_claims):
            raise EvaluationDataError("expected_claims 必须由 ExpectedClaim 组成。")
        if any(not isinstance(item, CrossDocumentEvidence) for item in self.gold_evidence):
            raise EvaluationDataError("gold_evidence 必须由 CrossDocumentEvidence 组成。")
        if any(not isinstance(note, str) or not note.strip() for note in self.annotation_notes):
            raise EvaluationDataError("annotation_notes 必须由非空字符串组成。")

        if self.answerable:
            if self.task_type != "cross_document":
                raise EvaluationDataError("可回答 P2 题必须使用 task_type=cross_document。")
            if self.refusal_reason_target is not None:
                raise EvaluationDataError("可回答 P2 题不能设置 refusal_reason_target。")
            claim_ids = {claim.claim_id for claim in self.expected_claims}
            if claim_ids != {"C1", "C2", "C3"} or len(self.expected_claims) != 3:
                raise EvaluationDataError("可回答 P2 题必须恰好提供 C1、C2、C3 三个 Claim。")
            if not self.gold_evidence:
                raise EvaluationDataError("可回答 P2 题至少需要一条 Gold Evidence。")
            evidence_ids = {item.evidence_id for item in self.gold_evidence}
            if len(evidence_ids) != len(self.gold_evidence):
                raise EvaluationDataError("gold_evidence.evidence_id 不能重复。")
            covered_claims = {claim_id for evidence in self.gold_evidence for claim_id in evidence.claim_ids}
            if covered_claims != claim_ids:
                raise EvaluationDataError("P2 的每个 Claim 必须被至少一条 Gold Evidence 覆盖。")
            evidence_papers = {item.paper_key for item in self.gold_evidence}
            if not set(self.required_paper_keys).issubset(evidence_papers):
                raise EvaluationDataError("每篇 required_paper_key 必须至少有一条 Gold Evidence。")
            c3_papers = {item.paper_key for item in self.gold_evidence if "C3" in item.claim_ids}
            if not set(self.required_paper_keys).issubset(c3_papers):
                raise EvaluationDataError("C3 比较边界必须由每篇 required_paper_key 的证据共同支持。")
        else:
            if self.task_type != "unanswerable":
                raise EvaluationDataError("不可回答 P2 题必须使用 task_type=unanswerable。")
            if self.comparison_allowed:
                raise EvaluationDataError("不可回答 P2 题不能设置 comparison_allowed=true。")
            if self.expected_claims or self.gold_evidence:
                raise EvaluationDataError("不可回答 P2 题不能伪造 Claim 或 Gold Evidence。")
            if self.refusal_reason_target not in P2_REFUSAL_REASONS:
                raise EvaluationDataError(f"refusal_reason_target 必须是 {sorted(P2_REFUSAL_REASONS)} 之一。")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> P2Case:
        paper_keys = raw.get("paper_keys")
        required_paper_keys = raw.get("required_paper_keys")
        expected_claims = raw.get("expected_claims")
        gold_evidence = raw.get("gold_evidence")
        annotation_notes = raw.get("annotation_notes", [])
        for value, name in (
            (paper_keys, "paper_keys"),
            (required_paper_keys, "required_paper_keys"),
            (expected_claims, "expected_claims"),
            (gold_evidence, "gold_evidence"),
            (annotation_notes, "annotation_notes"),
        ):
            if not isinstance(value, list):
                raise EvaluationDataError(f"{name} 必须是数组。")
        return cls(
            case_id=_identifier(raw.get("id"), "id"),
            source_id=_identifier(raw.get("source_id"), "source_id"),
            split=_text(raw.get("split"), "split"),
            task_type=_text(raw.get("task_type"), "task_type"),
            answerable=raw.get("answerable"),
            paper_keys=tuple(_identifier(item, "paper_keys[]") for item in paper_keys),
            required_paper_keys=tuple(_identifier(item, "required_paper_keys[]") for item in required_paper_keys),
            comparison_dimension=_text(raw.get("comparison_dimension"), "comparison_dimension"),
            comparison_allowed=raw.get("comparison_allowed"),
            refusal_reason_target=(
                _text(raw["refusal_reason_target"], "refusal_reason_target")
                if raw.get("refusal_reason_target") is not None
                else None
            ),
            question=_text(raw.get("question"), "question"),
            expected_answer=_text(raw.get("expected_answer"), "expected_answer"),
            expected_claims=tuple(
                ExpectedClaim.from_dict(_mapping(item, "expected_claims[]")) for item in expected_claims
            ),
            gold_evidence=tuple(
                CrossDocumentEvidence.from_dict(_mapping(item, "gold_evidence[]")) for item in gold_evidence
            ),
            annotation_notes=tuple(_text(item, "annotation_notes[]") for item in annotation_notes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": P2_SCHEMA_VERSION,
            "id": self.case_id,
            "source_id": self.source_id,
            "split": self.split,
            "task_type": self.task_type,
            "answerable": self.answerable,
            "paper_keys": list(self.paper_keys),
            "required_paper_keys": list(self.required_paper_keys),
            "comparison_dimension": self.comparison_dimension,
            "comparison_allowed": self.comparison_allowed,
            "refusal_reason_target": self.refusal_reason_target,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_claims": [{"claim_id": item.claim_id, "text": item.text} for item in self.expected_claims],
            "gold_evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "claim_ids": list(item.claim_ids),
                    "paper_key": item.paper_key,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "section_hint": item.section_hint,
                    "supporting_text_summary": item.supporting_text_summary,
                }
                for item in self.gold_evidence
            ],
            "annotation_notes": list(self.annotation_notes),
        }


def p2_dataset_sha256(path: Path) -> str:
    if not path.is_file():
        raise EvaluationDataError(f"P2 数据集不存在：{path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_p2_cases(path: Path) -> tuple[P2Case, ...]:
    if not path.is_file():
        raise EvaluationDataError(f"P2 数据集不存在：{path}")
    cases: list[P2Case] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationDataError(f"P2 数据集第 {line_number} 行不是合法 JSON：{error}") from error
        if not isinstance(raw, Mapping):
            raise EvaluationDataError(f"P2 数据集第 {line_number} 行必须是对象。")
        if raw.get("schema_version") != P2_SCHEMA_VERSION:
            raise EvaluationDataError(f"P2 数据集第 {line_number} 行 schema_version 必须为 {P2_SCHEMA_VERSION}。")
        try:
            cases.append(P2Case.from_dict(raw))
        except EvaluationDataError as error:
            raise EvaluationDataError(f"P2 数据集第 {line_number} 行无效：{error}") from error
    if not cases:
        raise EvaluationDataError("P2 数据集不能为空。")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise EvaluationDataError("P2 数据集存在重复 id。")
    return tuple(cases)


def validate_p2_cases(
    *,
    cases: Sequence[P2Case],
    corpus: CorpusManifest,
    paper_splits: Mapping[str, str],
    page_counts: Mapping[str, int],
) -> None:
    """验证 Split 隔离、目标论文证据覆盖与页码上界。"""
    if not cases:
        raise EvaluationDataError("P2 数据集不能为空。")
    papers = {paper.paper_key for paper in corpus.papers}
    if set(paper_splits) != papers:
        raise EvaluationDataError("paper_splits 必须覆盖 P2 Corpus 的全部论文。")
    if set(page_counts) != papers:
        raise EvaluationDataError("page_counts 必须覆盖 P2 Corpus 的全部论文。")
    if set(paper_splits.values()) - P2_SPLITS:
        raise EvaluationDataError("paper_splits 只能为 dev 或 test。")
    for case in cases:
        unknown = set(case.paper_keys) - papers
        if unknown:
            raise EvaluationDataError(f"{case.case_id} 引用了 Manifest 外论文：{sorted(unknown)}")
        wrong_split = [key for key in case.paper_keys if paper_splits[key] != case.split]
        if wrong_split:
            raise EvaluationDataError(f"{case.case_id} 将论文混入错误 split：{sorted(wrong_split)}")
        for evidence in case.gold_evidence:
            if evidence.paper_key not in papers:
                raise EvaluationDataError(f"{case.case_id} Evidence 引用了 Manifest 外论文。")
            if evidence.page_end > page_counts[evidence.paper_key]:
                raise EvaluationDataError(
                    f"{case.case_id}/{evidence.evidence_id} 页码超出 PDF 实际页数："
                    f"{evidence.page_end} > {page_counts[evidence.paper_key]}。"
                )


def build_p2_manifest(
    *,
    dataset_path: Path,
    source_markdown_path: Path,
    corpus_path: Path,
    paper_splits: Mapping[str, str],
    page_counts: Mapping[str, int],
    cases: Sequence[P2Case],
) -> dict[str, Any]:
    """构造提交到 Git 的 P2 冻结 Manifest，不包含机器绝对路径。"""
    if not source_markdown_path.is_file() or not corpus_path.is_file():
        raise EvaluationDataError("构造 P2 Manifest 前必须存在 Markdown 原稿和 Corpus Manifest。")
    return {
        "schema_version": P2_MANIFEST_SCHEMA_VERSION,
        "dataset": dataset_path.name,
        "dataset_sha256": p2_dataset_sha256(dataset_path),
        "source_markdown": source_markdown_path.name,
        "source_markdown_sha256": hashlib.sha256(source_markdown_path.read_bytes()).hexdigest(),
        "corpus_manifest": corpus_path.name,
        "corpus_manifest_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "paper_splits": dict(sorted(paper_splits.items())),
        "pdf_page_counts": dict(sorted(page_counts.items())),
        "case_counts": {
            "total": len(cases),
            "dev": sum(case.split == "dev" for case in cases),
            "test": sum(case.split == "test" for case in cases),
            "answerable": sum(case.answerable for case in cases),
            "unanswerable": sum(not case.answerable for case in cases),
            "comparison_allowed": sum(case.comparison_allowed for case in cases),
        },
    }


def validate_frozen_p2_manifest(*, dataset_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise EvaluationDataError(f"P2 Manifest 不存在：{manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationDataError(f"P2 Manifest 不是合法 JSON：{error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema_version") != P2_MANIFEST_SCHEMA_VERSION:
        raise EvaluationDataError("P2 Manifest schema_version 不正确。")
    if raw.get("dataset_sha256") != p2_dataset_sha256(dataset_path):
        raise EvaluationDataError("P2 数据集哈希与冻结 Manifest 不一致；拒绝运行。")
    return dict(raw)
