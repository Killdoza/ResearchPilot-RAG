"""跨论文回答评测集的数据契约与冻结校验。

单论文检索评测使用 :mod:`researchpilot.evaluation` 的 ``EvalCase``；跨论文
比较还需要保存“回答中的原子结论 C1/C2/C3 分别由哪条页级证据支持”，因此采用
独立、向后兼容的 Schema。本模块不连接 Chroma、LLM 或 Streamlit。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import CorpusManifest, EvaluationDataError

CROSSDOC_SCHEMA_VERSION = "ai_cs_crossdoc_eval_v0.1"
CROSSDOC_MANIFEST_SCHEMA_VERSION = "ai_cs_crossdoc_manifest_v0.1"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SPLITS = frozenset({"dev", "test"})

# 与 ai_cs_eval_v0.1 的论文级 Dev/Test 隔离保持完全一致。跨论文题不允许
# 同时引用两侧论文，否则后续根据 Dev 调整 Prompt 会污染 Test。
DEFAULT_PAPER_SPLITS: dict[str, str] = {
    "lora_2021": "dev",
    "mapreduce_2004": "dev",
    "ppo_2017": "dev",
    "rag_2020": "dev",
    "resnet_2015": "dev",
    "transformer_2017": "dev",
    "ddpm_2020": "test",
    "sam_2023": "test",
    "simclr_2020": "test",
    "spectre_2018": "test",
}


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDataError(f"{field_name} 必须是非空字符串。")
    return value.strip()


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationDataError(f"{field_name} 必须是对象。")
    return value


def _identifier(value: object, field_name: str) -> str:
    identifier = _text(value, field_name)
    if not _SAFE_ID.fullmatch(identifier):
        raise EvaluationDataError(f"{field_name} 只能使用字母、数字、点、下划线或连字符。")
    return identifier


def _page(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise EvaluationDataError(f"{field_name} 必须是大于等于 1 的整数。")
    return value


@dataclass(frozen=True)
class CrossDocumentEvidence:
    """支撑一个或多个原子结论的页级证据。"""

    evidence_id: str
    claim_ids: tuple[str, ...]
    paper_key: str
    page_start: int
    page_end: int
    section_hint: str
    supporting_text_summary: str

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence_id")
        if not self.claim_ids:
            raise EvaluationDataError("evidence.claim_ids 至少需要一个结论 ID。")
        if len(set(self.claim_ids)) != len(self.claim_ids):
            raise EvaluationDataError("evidence.claim_ids 不能重复。")
        for claim_id in self.claim_ids:
            _identifier(claim_id, "evidence.claim_ids[]")
        _identifier(self.paper_key, "evidence.paper_key")
        _page(self.page_start, "evidence.page_start")
        _page(self.page_end, "evidence.page_end")
        if self.page_end < self.page_start:
            raise EvaluationDataError("evidence.page_end 不能小于 page_start。")
        _text(self.section_hint, "evidence.section_hint")
        _text(self.supporting_text_summary, "evidence.supporting_text_summary")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CrossDocumentEvidence:
        claim_ids = raw.get("claim_ids")
        if not isinstance(claim_ids, list):
            raise EvaluationDataError("evidence.claim_ids 必须是数组。")
        return cls(
            evidence_id=_identifier(raw.get("evidence_id"), "evidence_id"),
            claim_ids=tuple(_identifier(item, "evidence.claim_ids[]") for item in claim_ids),
            paper_key=_identifier(raw.get("paper_key"), "evidence.paper_key"),
            page_start=_page(raw.get("page_start"), "evidence.page_start"),
            page_end=_page(raw.get("page_end"), "evidence.page_end"),
            section_hint=_text(raw.get("section_hint"), "evidence.section_hint"),
            supporting_text_summary=_text(raw.get("supporting_text_summary"), "evidence.supporting_text_summary"),
        )


@dataclass(frozen=True)
class ExpectedClaim:
    """人工核验的最小答案结论。"""

    claim_id: str
    text: str

    def __post_init__(self) -> None:
        _identifier(self.claim_id, "claim_id")
        _text(self.text, "claim.text")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExpectedClaim:
        return cls(
            claim_id=_identifier(raw.get("claim_id"), "claim_id"),
            text=_text(raw.get("text"), "claim.text"),
        )


@dataclass(frozen=True)
class CrossDocumentCase:
    """一条跨论文比较或跨论文拒答问题。"""

    case_id: str
    source_id: str
    split: str
    task_type: str
    answerable: bool
    paper_keys: tuple[str, ...]
    comparison_dimension: str | None
    question: str
    expected_answer: str
    expected_claims: tuple[ExpectedClaim, ...]
    gold_evidence: tuple[CrossDocumentEvidence, ...]
    annotation_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.case_id, "id")
        _identifier(self.source_id, "source_id")
        if self.split not in _SPLITS:
            raise EvaluationDataError("crossdoc split 必须是 dev 或 test。")
        if self.task_type not in {"cross_document", "unanswerable"}:
            raise EvaluationDataError("crossdoc task_type 必须是 cross_document 或 unanswerable。")
        if not isinstance(self.answerable, bool):
            raise EvaluationDataError("crossdoc answerable 必须是布尔值。")
        if len(self.paper_keys) < 2 or len(set(self.paper_keys)) != len(self.paper_keys):
            raise EvaluationDataError("跨论文问题必须恰当地包含至少两篇不重复论文。")
        for paper_key in self.paper_keys:
            _identifier(paper_key, "paper_keys[]")
        if self.comparison_dimension is not None:
            _text(self.comparison_dimension, "comparison_dimension")
        _text(self.question, "question")
        _text(self.expected_answer, "expected_answer")
        if any(not isinstance(item, ExpectedClaim) for item in self.expected_claims):
            raise EvaluationDataError("expected_claims 必须由 ExpectedClaim 组成。")
        if any(not isinstance(item, CrossDocumentEvidence) for item in self.gold_evidence):
            raise EvaluationDataError("gold_evidence 必须由 CrossDocumentEvidence 组成。")
        if any(not isinstance(note, str) or not note.strip() for note in self.annotation_notes):
            raise EvaluationDataError("annotation_notes 必须由非空字符串组成。")
        if self.answerable:
            if self.task_type != "cross_document":
                raise EvaluationDataError("可回答跨论文题必须使用 task_type=cross_document。")
            if not self.expected_claims or not self.gold_evidence:
                raise EvaluationDataError("可回答跨论文题必须同时提供结论和页级证据。")
            claim_ids = {claim.claim_id for claim in self.expected_claims}
            if len(claim_ids) != len(self.expected_claims):
                raise EvaluationDataError("expected_claims.claim_id 不能重复。")
            evidence_ids = {evidence.evidence_id for evidence in self.gold_evidence}
            if len(evidence_ids) != len(self.gold_evidence):
                raise EvaluationDataError("gold_evidence.evidence_id 不能重复。")
            evidence_papers = {evidence.paper_key for evidence in self.gold_evidence}
            if evidence_papers != set(self.paper_keys):
                raise EvaluationDataError("可回答跨论文题的每篇 paper_key 都必须有页级证据。")
            covered_claims = {claim_id for evidence in self.gold_evidence for claim_id in evidence.claim_ids}
            if covered_claims != claim_ids:
                raise EvaluationDataError("每个 ExpectedClaim 必须且只能由至少一条 Gold Evidence 覆盖。")
            if any(evidence.paper_key not in self.paper_keys for evidence in self.gold_evidence):
                raise EvaluationDataError("Gold Evidence 的论文必须包含在 paper_keys 中。")
        elif self.task_type != "unanswerable" or self.expected_claims or self.gold_evidence:
            raise EvaluationDataError("不可回答跨论文题必须使用 task_type=unanswerable，且不能伪造结论或证据。")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CrossDocumentCase:
        paper_keys = raw.get("paper_keys")
        claims = raw.get("expected_claims")
        evidence = raw.get("gold_evidence")
        notes = raw.get("annotation_notes", [])
        if not isinstance(paper_keys, list) or not isinstance(claims, list) or not isinstance(evidence, list):
            raise EvaluationDataError("paper_keys、expected_claims、gold_evidence 必须是数组。")
        if not isinstance(notes, list):
            raise EvaluationDataError("annotation_notes 必须是数组。")
        return cls(
            case_id=_identifier(raw.get("id"), "id"),
            source_id=_identifier(raw.get("source_id"), "source_id"),
            split=_text(raw.get("split"), "split"),
            task_type=_text(raw.get("task_type"), "task_type"),
            answerable=raw.get("answerable"),
            paper_keys=tuple(_identifier(item, "paper_keys[]") for item in paper_keys),
            comparison_dimension=(
                _text(raw["comparison_dimension"], "comparison_dimension")
                if raw.get("comparison_dimension") is not None
                else None
            ),
            question=_text(raw.get("question"), "question"),
            expected_answer=_text(raw.get("expected_answer"), "expected_answer"),
            expected_claims=tuple(
                ExpectedClaim.from_dict(_require_mapping(item, "expected_claims[]")) for item in claims
            ),
            gold_evidence=tuple(
                CrossDocumentEvidence.from_dict(_require_mapping(item, "gold_evidence[]")) for item in evidence
            ),
            annotation_notes=tuple(_text(note, "annotation_notes[]") for note in notes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CROSSDOC_SCHEMA_VERSION,
            "id": self.case_id,
            "source_id": self.source_id,
            "split": self.split,
            "task_type": self.task_type,
            "answerable": self.answerable,
            "paper_keys": list(self.paper_keys),
            "comparison_dimension": self.comparison_dimension,
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


def canonical_json_sha256(value: object) -> str:
    """计算 JSON 兼容对象的稳定 SHA-256，用于冻结可提交的数据集。"""
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def dataset_sha256(path: Path) -> str:
    if not path.is_file():
        raise EvaluationDataError(f"跨论文数据集不存在：{path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_crossdoc_cases(path: Path) -> tuple[CrossDocumentCase, ...]:
    """读取 JSONL，并在错误中给出准确行号。"""
    if not path.is_file():
        raise EvaluationDataError(f"跨论文数据集不存在：{path}")
    cases: list[CrossDocumentCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationDataError(f"跨论文数据集第 {line_number} 行不是合法 JSON：{error}") from error
        if not isinstance(raw, Mapping):
            raise EvaluationDataError(f"跨论文数据集第 {line_number} 行必须是对象。")
        if raw.get("schema_version") != CROSSDOC_SCHEMA_VERSION:
            raise EvaluationDataError(
                f"跨论文数据集第 {line_number} 行 schema_version 必须为 {CROSSDOC_SCHEMA_VERSION}。"
            )
        try:
            cases.append(CrossDocumentCase.from_dict(raw))
        except EvaluationDataError as error:
            raise EvaluationDataError(f"跨论文数据集第 {line_number} 行无效：{error}") from error
    if not cases:
        raise EvaluationDataError("跨论文数据集不能为空。")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise EvaluationDataError("跨论文数据集存在重复 id。")
    return tuple(cases)


def validate_crossdoc_cases(
    *,
    cases: Sequence[CrossDocumentCase],
    corpus: CorpusManifest,
    paper_splits: Mapping[str, str],
    page_counts: Mapping[str, int] | None = None,
) -> None:
    """验证语料覆盖、论文级 Split 隔离和每个 PDF 的页码上界。"""
    if not cases:
        raise EvaluationDataError("跨论文数据集不能为空。")
    papers = {paper.paper_key: paper for paper in corpus.papers}
    if set(paper_splits) != set(papers):
        raise EvaluationDataError("paper_splits 必须为基础 Corpus 的每篇论文提供 split。")
    if set(paper_splits.values()) - _SPLITS:
        raise EvaluationDataError("paper_splits 只能包含 dev 或 test。")
    if page_counts is not None:
        missing_pages = set(papers) - set(page_counts)
        if missing_pages:
            raise EvaluationDataError(f"page_counts 缺少论文：{sorted(missing_pages)}")
        for paper_key, page_count in page_counts.items():
            if paper_key not in papers:
                raise EvaluationDataError(f"page_counts 包含 Manifest 外论文：{paper_key}")
            if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count < 1:
                raise EvaluationDataError(f"{paper_key} 的 PDF 页数必须是正整数。")
    for case in cases:
        wrong_split = [paper for paper in case.paper_keys if paper_splits.get(paper) != case.split]
        if wrong_split:
            raise EvaluationDataError(f"{case.case_id} 将论文混入错误 split：{', '.join(sorted(wrong_split))}")
        for evidence in case.gold_evidence:
            paper = papers.get(evidence.paper_key)
            if paper is None:
                raise EvaluationDataError(f"{case.case_id} 引用了 Manifest 中不存在的论文：{evidence.paper_key}")
            if page_counts is not None and evidence.page_end > page_counts[evidence.paper_key]:
                raise EvaluationDataError(
                    f"{case.case_id} 的 {evidence.paper_key} 页码超出 PDF 实际页数："
                    f"{evidence.page_end} > {page_counts[evidence.paper_key]}。"
                )


def build_crossdoc_manifest(
    *,
    dataset_path: Path,
    source_markdown_path: Path,
    base_corpus_path: Path,
    paper_splits: Mapping[str, str],
    cases: Sequence[CrossDocumentCase],
) -> dict[str, Any]:
    """构造含数据集/源草稿哈希的冻结 Manifest。"""
    if not source_markdown_path.is_file() or not base_corpus_path.is_file():
        raise EvaluationDataError("构造跨论文 Manifest 前必须存在源草稿和基础 Corpus Manifest。")
    return {
        "schema_version": CROSSDOC_MANIFEST_SCHEMA_VERSION,
        "dataset": dataset_path.name,
        "dataset_sha256": dataset_sha256(dataset_path),
        "source_markdown": source_markdown_path.name,
        "source_markdown_sha256": hashlib.sha256(source_markdown_path.read_bytes()).hexdigest(),
        # 只记录文件名，不能把开发者电脑上的绝对路径写入可提交 Manifest。
        "base_corpus_manifest": base_corpus_path.name,
        "base_corpus_manifest_sha256": hashlib.sha256(base_corpus_path.read_bytes()).hexdigest(),
        "paper_splits": dict(sorted(paper_splits.items())),
        "case_counts": {
            "total": len(cases),
            "dev": sum(case.split == "dev" for case in cases),
            "test": sum(case.split == "test" for case in cases),
            "answerable": sum(case.answerable for case in cases),
            "unanswerable": sum(not case.answerable for case in cases),
        },
    }


def validate_frozen_crossdoc_manifest(*, dataset_path: Path, manifest_path: Path) -> dict[str, Any]:
    """确认被运行的数据集仍与提交的冻结 Manifest 一致。"""
    if not manifest_path.is_file():
        raise EvaluationDataError(f"跨论文 Manifest 不存在：{manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationDataError(f"跨论文 Manifest 不是合法 JSON：{error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema_version") != CROSSDOC_MANIFEST_SCHEMA_VERSION:
        raise EvaluationDataError("跨论文 Manifest schema_version 不正确。")
    actual = dataset_sha256(dataset_path)
    if raw.get("dataset_sha256") != actual:
        raise EvaluationDataError("跨论文数据集哈希与冻结 Manifest 不一致；拒绝运行，避免 Test 被悄悄改写。")
    return dict(raw)
