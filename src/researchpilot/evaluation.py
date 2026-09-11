"""AI/CS 论文评测集契约、确定性检索指标与实验配置读取。

本模块不连接 Chroma、LLM 或 Streamlit。它只定义可版本化的数据格式，并将一次检索
结果转换为可复核的页级命中与聚合指标，供脚本层执行真实实验。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from .schemas import ScoredChunk

EVAL_SCHEMA_VERSION = "ai_cs_eval_v0.1"
CORPUS_SCHEMA_VERSION = "ai_cs_corpus_v0.1"
EXPERIMENT_SCHEMA_VERSION = "researchpilot_retrieval_experiment_v1"
EVALUATION_TASK_TYPES = frozenset(
    {
        "metadata",
        "keyword_exact",
        "local_fact",
        "semantic_explanation",
        "experimental_setup",
        "multi_section",
        "cross_lingual",
        "unanswerable",
    }
)
EVALUATION_SPLITS = frozenset({"dev", "test"})
_SAFE_ID_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")


class EvaluationDataError(ValueError):
    """评测数据、论文 Manifest 或实验配置不符合契约时抛出。"""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDataError(f"{field_name} 必须是非空字符串。")
    return value.strip()


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationDataError(f"{field_name} 必须是对象。")
    return value


def _require_positive_page(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise EvaluationDataError(f"{field_name} 必须是大于等于 1 的整数。")
    return value


def _validate_identifier(value: object, field_name: str) -> str:
    identifier = _require_text(value, field_name)
    if len(identifier) > 128 or any(character not in _SAFE_ID_CHARACTERS for character in identifier):
        raise EvaluationDataError(f"{field_name} 只能包含字母、数字、点、下划线和连字符，长度不超过 128。")
    return identifier


def _validate_sha256(value: object, field_name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    digest = _require_text(value, field_name).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise EvaluationDataError(f"{field_name} 必须是 64 位十六进制 SHA-256。")
    return digest


@dataclass(frozen=True)
class GoldEvidence:
    """人工确认的一条页级证据定位。"""

    paper_key: str
    page_start: int
    page_end: int
    section_hint: str = ""

    def __post_init__(self) -> None:
        _validate_identifier(self.paper_key, "gold_evidence.paper_key")
        _require_positive_page(self.page_start, "gold_evidence.page_start")
        _require_positive_page(self.page_end, "gold_evidence.page_end")
        if self.page_end < self.page_start:
            raise EvaluationDataError("gold_evidence.page_end 不能小于 page_start。")
        if not isinstance(self.section_hint, str):
            raise EvaluationDataError("gold_evidence.section_hint 必须是字符串。")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GoldEvidence:
        return cls(
            paper_key=_validate_identifier(raw.get("paper_key"), "gold_evidence.paper_key"),
            page_start=_require_positive_page(raw.get("page_start"), "gold_evidence.page_start"),
            page_end=_require_positive_page(raw.get("page_end"), "gold_evidence.page_end"),
            section_hint=str(raw.get("section_hint") or "").strip(),
        )


@dataclass(frozen=True)
class EvalCase:
    """一条人工核验的论文检索问题。"""

    case_id: str
    split: str
    paper_keys: tuple[str, ...]
    question: str
    question_language: str
    task_type: str
    answerable: bool
    gold_evidence: tuple[GoldEvidence, ...]
    expected_answer: str

    def __post_init__(self) -> None:
        _validate_identifier(self.case_id, "id")
        if self.split not in EVALUATION_SPLITS:
            raise EvaluationDataError(f"split 必须是 {sorted(EVALUATION_SPLITS)} 之一。")
        if not self.paper_keys:
            raise EvaluationDataError("paper_keys 至少包含一篇论文。")
        if len(set(self.paper_keys)) != len(self.paper_keys):
            raise EvaluationDataError("paper_keys 不能重复。")
        for paper_key in self.paper_keys:
            _validate_identifier(paper_key, "paper_keys[]")
        _require_text(self.question, "question")
        if self.question_language not in {"zh", "en"}:
            raise EvaluationDataError("question_language 目前只支持 zh 或 en。")
        if self.task_type not in EVALUATION_TASK_TYPES:
            raise EvaluationDataError(f"task_type 不在允许集合中：{self.task_type}。")
        if not isinstance(self.answerable, bool):
            raise EvaluationDataError("answerable 必须是布尔值。")
        _require_text(self.expected_answer, "expected_answer")
        if any(not isinstance(item, GoldEvidence) for item in self.gold_evidence):
            raise EvaluationDataError("gold_evidence 必须全部是 GoldEvidence。")
        evidence_locations = {(item.paper_key, item.page_start, item.page_end) for item in self.gold_evidence}
        if len(evidence_locations) != len(self.gold_evidence):
            raise EvaluationDataError("同一论文的相同页码范围不能重复标注为 gold_evidence。")
        if self.answerable:
            if self.task_type == "unanswerable":
                raise EvaluationDataError("unanswerable 类型不能标记为可回答。")
            if not self.gold_evidence:
                raise EvaluationDataError("可回答问题至少需要一条 gold_evidence。")
        elif self.task_type != "unanswerable" or self.gold_evidence:
            raise EvaluationDataError("不可回答问题必须使用 task_type=unanswerable 且 gold_evidence 为空。")
        unknown_evidence_papers = {
            evidence.paper_key for evidence in self.gold_evidence if evidence.paper_key not in self.paper_keys
        }
        if unknown_evidence_papers:
            raise EvaluationDataError(f"gold_evidence 的论文不在 paper_keys 中：{sorted(unknown_evidence_papers)}。")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvalCase:
        papers_raw = raw.get("paper_keys")
        if not isinstance(papers_raw, list):
            raise EvaluationDataError("paper_keys 必须是数组。")
        evidence_raw = raw.get("gold_evidence")
        if not isinstance(evidence_raw, list):
            raise EvaluationDataError("gold_evidence 必须是数组。")
        answerable = raw.get("answerable")
        if not isinstance(answerable, bool):
            raise EvaluationDataError("answerable 必须是布尔值。")
        return cls(
            case_id=_validate_identifier(raw.get("id"), "id"),
            split=_require_text(raw.get("split"), "split"),
            paper_keys=tuple(_validate_identifier(item, "paper_keys[]") for item in papers_raw),
            question=_require_text(raw.get("question"), "question"),
            question_language=_require_text(raw.get("question_language", "zh"), "question_language"),
            task_type=_require_text(raw.get("task_type"), "task_type"),
            answerable=answerable,
            gold_evidence=tuple(
                GoldEvidence.from_dict(_require_mapping(item, "gold_evidence[]")) for item in evidence_raw
            ),
            expected_answer=_require_text(raw.get("expected_answer"), "expected_answer"),
        )


@dataclass(frozen=True)
class CorpusPaper:
    """一篇可复现导入的开放获取论文。"""

    paper_key: str
    title: str
    area: str
    year: int
    source_url: str
    local_filename: str
    sha256: str | None

    def __post_init__(self) -> None:
        _validate_identifier(self.paper_key, "paper_key")
        _require_text(self.title, "title")
        _require_text(self.area, "area")
        if not isinstance(self.year, int) or isinstance(self.year, bool) or not 1900 <= self.year <= 2100:
            raise EvaluationDataError("year 必须是 1900 到 2100 之间的整数。")
        source_url = _require_text(self.source_url, "source_url")
        if not source_url.startswith(("https://", "http://")):
            raise EvaluationDataError("source_url 必须以 http:// 或 https:// 开头。")
        filename = _require_text(self.local_filename, "local_filename")
        if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
            raise EvaluationDataError("local_filename 必须是单个 PDF 文件名，不能包含路径。")
        _validate_sha256(self.sha256, "sha256", allow_none=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CorpusPaper:
        return cls(
            paper_key=_validate_identifier(raw.get("paper_key"), "paper_key"),
            title=_require_text(raw.get("title"), "title"),
            area=_require_text(raw.get("area"), "area"),
            year=raw.get("year"),
            source_url=_require_text(raw.get("source_url"), "source_url"),
            local_filename=_require_text(raw.get("local_filename"), "local_filename"),
            sha256=_validate_sha256(raw.get("sha256"), "sha256", allow_none=True),
        )


@dataclass(frozen=True)
class CorpusManifest:
    """评测语料的版本化论文清单。"""

    schema_version: str
    description: str
    papers: tuple[CorpusPaper, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_SCHEMA_VERSION:
            raise EvaluationDataError(f"corpus schema_version 必须为 {CORPUS_SCHEMA_VERSION}。")
        _require_text(self.description, "description")
        if len({paper.paper_key for paper in self.papers}) != len(self.papers):
            raise EvaluationDataError("Manifest 中 paper_key 不能重复。")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CorpusManifest:
        papers_raw = raw.get("papers")
        if not isinstance(papers_raw, list):
            raise EvaluationDataError("Manifest 的 papers 必须是数组。")
        return cls(
            schema_version=_require_text(raw.get("schema_version"), "schema_version"),
            description=_require_text(raw.get("description"), "description"),
            papers=tuple(CorpusPaper.from_dict(_require_mapping(item, "papers[]")) for item in papers_raw),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """一次检索实验所需的全部可复现参数。"""

    name: str
    description: str
    parser_name: str
    parser_version: str
    enable_ocr: bool
    chunker_version: str
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int
    embedding_model: str
    reranker_model: str | None
    generation_model: str | None
    strategy: str
    query_expansion: bool
    query_expansion_count: int
    dense_limit_per_query: int
    bm25_limit_per_query: int
    fusion: str
    reranker: bool
    result_limit: int
    rrf_k: int

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "experiment.name")
        _require_text(self.description, "experiment.description")
        for value, field_name in (
            (self.parser_name, "parser_name"),
            (self.parser_version, "parser_version"),
            (self.chunker_version, "chunker_version"),
            (self.embedding_model, "embedding_model"),
        ):
            _require_text(value, field_name)
        if not isinstance(self.enable_ocr, bool):
            raise EvaluationDataError("ingestion.enable_ocr 必须是布尔值。")
        for size, overlap, label in (
            (self.parent_chunk_size, self.parent_chunk_overlap, "parent"),
            (self.child_chunk_size, self.child_chunk_overlap, "child"),
        ):
            if not isinstance(size, int) or not isinstance(overlap, int) or size < 1 or overlap < 0 or overlap >= size:
                raise EvaluationDataError(f"{label} 分块参数必须满足 size > overlap >= 0。")
        if self.strategy not in {"dense", "bm25", "hybrid_union", "hybrid_rrf"}:
            raise EvaluationDataError("strategy 只支持 dense、bm25、hybrid_union 或 hybrid_rrf。")
        if not isinstance(self.dense_limit_per_query, int) or self.dense_limit_per_query < 0:
            raise EvaluationDataError("dense_limit_per_query 必须是非负整数。")
        if not isinstance(self.result_limit, int) or self.result_limit < 1:
            raise EvaluationDataError("result_limit 必须是正整数。")
        for value, field_name in ((self.result_limit, "result_limit"),):
            if not isinstance(value, int) or value < 1:
                raise EvaluationDataError(f"{field_name} 必须是正整数。")
        if not isinstance(self.bm25_limit_per_query, int) or self.bm25_limit_per_query < 0:
            raise EvaluationDataError("bm25_limit_per_query 必须是非负整数。")
        if not isinstance(self.query_expansion, bool) or not isinstance(self.reranker, bool):
            raise EvaluationDataError("query_expansion 和 reranker 必须是布尔值。")
        if not isinstance(self.query_expansion_count, int) or self.query_expansion_count < 0:
            raise EvaluationDataError("query_expansion_count 必须是非负整数。")
        if self.query_expansion and self.query_expansion_count < 1:
            raise EvaluationDataError("开启 query_expansion 时 query_expansion_count 至少为 1。")
        if not self.query_expansion and self.query_expansion_count != 0:
            raise EvaluationDataError("关闭 query_expansion 时 query_expansion_count 必须为 0。")
        if not isinstance(self.rrf_k, int) or self.rrf_k < 0:
            raise EvaluationDataError("rrf_k 必须是非负整数。")
        if self.strategy == "dense":
            if self.dense_limit_per_query < 1:
                raise EvaluationDataError("dense 基线的 dense_limit_per_query 必须为正整数。")
            if self.query_expansion or self.bm25_limit_per_query != 0 or self.fusion != "none" or self.reranker:
                raise EvaluationDataError("dense 基线不能启用 Query Expansion、BM25、融合或 Reranker。")
            if self.reranker_model is not None or self.generation_model is not None:
                raise EvaluationDataError("dense 基线的 reranker_model 和 generation_model 必须为 null。")
            if self.rrf_k != 0:
                raise EvaluationDataError("dense 基线的 rrf_k 必须为 0。")
        elif self.strategy == "bm25":
            if self.dense_limit_per_query != 0 or self.bm25_limit_per_query < 1:
                raise EvaluationDataError("bm25 基线必须关闭 Dense 并启用 BM25。")
            if self.query_expansion or self.fusion != "none" or self.reranker or self.rrf_k != 0:
                raise EvaluationDataError("bm25 基线不能启用 Query Expansion、融合、Reranker 或 RRF。")
            if self.reranker_model is not None or self.generation_model is not None:
                raise EvaluationDataError("bm25 基线的 reranker_model 和 generation_model 必须为 null。")
        elif self.strategy == "hybrid_union":
            if self.dense_limit_per_query < 1 or self.bm25_limit_per_query < 1:
                raise EvaluationDataError("hybrid_union 必须同时启用 Dense 与 BM25。")
            if self.fusion not in {"set_union", "parent_id_set_union"} or self.rrf_k != 0:
                raise EvaluationDataError("hybrid_union 必须使用 Set Union 且 rrf_k 为 0。")
            if self.reranker and not self.reranker_model:
                raise EvaluationDataError("启用 Reranker 时必须指定 reranker_model。")
        else:
            if self.dense_limit_per_query < 1 or self.bm25_limit_per_query < 1:
                raise EvaluationDataError("hybrid_rrf 必须同时启用 Dense 与 BM25。")
            if self.fusion != "rrf" or self.rrf_k < 1:
                raise EvaluationDataError("hybrid_rrf 必须使用 rrf 融合且 rrf_k 为正整数。")
            if self.reranker and not self.reranker_model:
                raise EvaluationDataError("启用 Reranker 时必须指定 reranker_model。")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        if raw.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
            raise EvaluationDataError(f"实验 schema_version 必须为 {EXPERIMENT_SCHEMA_VERSION}。")
        ingestion = _require_mapping(raw.get("ingestion"), "ingestion")
        models = _require_mapping(raw.get("models"), "models")
        retrieval = _require_mapping(raw.get("retrieval"), "retrieval")
        return cls(
            name=_validate_identifier(raw.get("name"), "name"),
            description=_require_text(raw.get("description"), "description"),
            parser_name=_require_text(ingestion.get("parser_name"), "ingestion.parser_name"),
            parser_version=_require_text(ingestion.get("parser_version"), "ingestion.parser_version"),
            enable_ocr=ingestion.get("enable_ocr", False),
            chunker_version=_require_text(ingestion.get("chunker_version"), "ingestion.chunker_version"),
            parent_chunk_size=ingestion.get("parent_chunk_size"),
            parent_chunk_overlap=ingestion.get("parent_chunk_overlap"),
            child_chunk_size=ingestion.get("child_chunk_size"),
            child_chunk_overlap=ingestion.get("child_chunk_overlap"),
            embedding_model=_require_text(models.get("embedding_model"), "models.embedding_model"),
            reranker_model=models.get("reranker_model"),
            generation_model=models.get("generation_model"),
            strategy=_require_text(retrieval.get("strategy"), "retrieval.strategy"),
            query_expansion=retrieval.get("query_expansion"),
            query_expansion_count=retrieval.get("query_expansion_count"),
            dense_limit_per_query=retrieval.get("dense_limit_per_query"),
            bm25_limit_per_query=retrieval.get("bm25_limit_per_query"),
            fusion=_require_text(retrieval.get("fusion"), "retrieval.fusion"),
            reranker=retrieval.get("reranker"),
            result_limit=retrieval.get("result_limit"),
            rrf_k=retrieval.get("rrf_k", 0),
        )


def load_eval_cases(path: Path, *, allow_empty: bool = False) -> tuple[EvalCase, ...]:
    """读取 JSONL，并在报错时保留行号以便人工修订标注。"""
    if not path.is_file():
        raise EvaluationDataError(f"评测集文件不存在：{path}")
    cases: list[EvalCase] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = EvalCase.from_dict(_require_mapping(raw, f"第 {line_number} 行"))
        except (json.JSONDecodeError, EvaluationDataError) as exc:
            raise EvaluationDataError(f"评测集第 {line_number} 行无效：{exc}") from exc
        if case.case_id in identifiers:
            raise EvaluationDataError(f"评测集存在重复 id：{case.case_id}。")
        identifiers.add(case.case_id)
        cases.append(case)
    if not cases and not allow_empty:
        raise EvaluationDataError("评测集为空；请先完成至少一条人工核验标注。")
    return tuple(cases)


def load_corpus_manifest(path: Path) -> CorpusManifest:
    if not path.is_file():
        raise EvaluationDataError(f"论文 Manifest 不存在：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(f"论文 Manifest 不是合法 JSON：{exc}") from exc
    return CorpusManifest.from_dict(_require_mapping(raw, "论文 Manifest"))


def load_experiment_config(path: Path) -> ExperimentConfig:
    if not path.is_file():
        raise EvaluationDataError(f"实验配置不存在：{path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvaluationDataError(f"实验配置不是合法 YAML：{exc}") from exc
    return ExperimentConfig.from_dict(_require_mapping(raw, "实验配置"))


def validate_dataset_manifest_coverage(cases: Sequence[EvalCase], manifest: CorpusManifest) -> None:
    """确保问题和证据引用的每一篇论文都出现在 Manifest。"""
    available = {paper.paper_key for paper in manifest.papers}
    referenced = {paper_key for case in cases for paper_key in case.paper_keys}
    missing = sorted(referenced - available)
    if missing:
        raise EvaluationDataError(f"评测集引用了 Manifest 中不存在的 paper_key：{missing}。")


def paper_ids_from_manifest(manifest: CorpusManifest, *, require_hashes: bool = True) -> dict[str, str]:
    """从冻结的 PDF SHA-256 推导与 ingestion.py 一致的 paper_id。"""
    result: dict[str, str] = {}
    missing_hashes: list[str] = []
    for paper in manifest.papers:
        if not paper.sha256:
            missing_hashes.append(paper.paper_key)
            continue
        result[paper.paper_key] = f"paper_{paper.sha256[:16]}"
    if require_hashes and missing_hashes:
        raise EvaluationDataError(f"以下论文尚未冻结 SHA-256，不能运行可复现实验：{missing_hashes}。")
    return result


def _pages_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


def _result_matches_evidence(result: ScoredChunk, evidence: GoldEvidence, paper_ids: Mapping[str, str]) -> bool:
    expected_paper_id = paper_ids.get(evidence.paper_key)
    if expected_paper_id is None:
        raise EvaluationDataError(f"缺少 paper_key={evidence.paper_key} 的运行时 paper_id 映射。")
    chunk = result.chunk
    return chunk.paper_id == expected_paper_id and _pages_overlap(
        chunk.page_start, chunk.page_end, evidence.page_start, evidence.page_end
    )


@dataclass(frozen=True)
class CaseEvaluation:
    """一条问题的确定性检索判定，不包含模型主观打分。"""

    case_id: str
    split: str
    task_type: str
    answerable: bool
    result_count: int
    relevant_ranks: tuple[int, ...]
    evidence_first_ranks: tuple[int | None, ...]
    retrieval_latency_ms: float

    def __post_init__(self) -> None:
        _validate_identifier(self.case_id, "case_id")
        if self.split not in EVALUATION_SPLITS or self.task_type not in EVALUATION_TASK_TYPES:
            raise EvaluationDataError("CaseEvaluation 的 split 或 task_type 无效。")
        if not isinstance(self.result_count, int) or self.result_count < 0:
            raise EvaluationDataError("result_count 必须是非负整数。")
        if any(not isinstance(rank, int) or rank < 1 for rank in self.relevant_ranks):
            raise EvaluationDataError("relevant_ranks 必须是正整数。")
        if any(rank is not None and (not isinstance(rank, int) or rank < 1) for rank in self.evidence_first_ranks):
            raise EvaluationDataError("evidence_first_ranks 必须是正整数或 None。")
        if not isinstance(self.retrieval_latency_ms, int | float) or not math.isfinite(self.retrieval_latency_ms):
            raise EvaluationDataError("retrieval_latency_ms 必须是有限数值。")


def evaluate_case(
    case: EvalCase,
    results: Sequence[ScoredChunk],
    *,
    paper_ids: Mapping[str, str],
    retrieval_latency_ms: float,
) -> CaseEvaluation:
    """按“论文一致且页码重叠”计算一条问题的证据命中。"""
    if not isinstance(retrieval_latency_ms, int | float) or not math.isfinite(retrieval_latency_ms):
        raise EvaluationDataError("retrieval_latency_ms 必须是有限数值。")
    if not case.answerable:
        return CaseEvaluation(
            case_id=case.case_id,
            split=case.split,
            task_type=case.task_type,
            answerable=False,
            result_count=len(results),
            relevant_ranks=(),
            evidence_first_ranks=(),
            retrieval_latency_ms=float(retrieval_latency_ms),
        )

    evidence_first_ranks: list[int | None] = []
    for evidence in case.gold_evidence:
        first_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if _result_matches_evidence(result, evidence, paper_ids)
            ),
            None,
        )
        evidence_first_ranks.append(first_rank)
    relevant_ranks = tuple(
        rank
        for rank, result in enumerate(results, start=1)
        if any(_result_matches_evidence(result, evidence, paper_ids) for evidence in case.gold_evidence)
    )
    return CaseEvaluation(
        case_id=case.case_id,
        split=case.split,
        task_type=case.task_type,
        answerable=True,
        result_count=len(results),
        relevant_ranks=relevant_ranks,
        evidence_first_ranks=tuple(evidence_first_ranks),
        retrieval_latency_ms=float(retrieval_latency_ms),
    )


@dataclass(frozen=True)
class EvaluationSummary:
    """聚合后的检索指标；所有指标均能从逐题 CaseEvaluation 再算得到。"""

    evaluated_cases: int
    answerable_cases: int
    unanswerable_cases: int
    question_recall_at: dict[int, float]
    evidence_recall_at: dict[int, float]
    mrr: float
    ndcg_at: dict[int, float]
    mean_retrieval_latency_ms: float
    unanswerable_top1_retrieval_rate: float | None
    p50_retrieval_latency_ms: float = 0.0
    p95_retrieval_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """使用线性插值计算稳定的延迟分位数，避免引入额外数值依赖。"""
    if not values:
        raise EvaluationDataError("无法计算空延迟序列的分位数。")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1 - ratio) + ordered[upper] * ratio


def summarize_evaluations(
    outcomes: Sequence[CaseEvaluation], *, cutoffs: Sequence[int] = (1, 5, 10)
) -> EvaluationSummary:
    """汇总 Question Recall、Evidence Recall、MRR、nDCG 与延迟。"""
    normalized_cutoffs = tuple(dict.fromkeys(cutoffs))
    if not normalized_cutoffs or any(not isinstance(cutoff, int) or cutoff < 1 for cutoff in normalized_cutoffs):
        raise EvaluationDataError("cutoffs 必须是非空的正整数序列。")
    if not outcomes:
        raise EvaluationDataError("无法汇总空评测结果。")
    answerable = [outcome for outcome in outcomes if outcome.answerable]
    unanswerable = [outcome for outcome in outcomes if not outcome.answerable]
    if not answerable:
        raise EvaluationDataError("评测集至少需要一条可回答问题，才能计算检索指标。")

    question_recall_at: dict[int, float] = {}
    evidence_recall_at: dict[int, float] = {}
    ndcg_at: dict[int, float] = {}
    for cutoff in normalized_cutoffs:
        question_recall_at[cutoff] = sum(
            bool(outcome.relevant_ranks and outcome.relevant_ranks[0] <= cutoff) for outcome in answerable
        ) / len(answerable)
        evidence_total = sum(len(outcome.evidence_first_ranks) for outcome in answerable)
        evidence_hits = sum(
            sum(rank is not None and rank <= cutoff for rank in outcome.evidence_first_ranks) for outcome in answerable
        )
        evidence_recall_at[cutoff] = evidence_hits / evidence_total
        ndcg_scores: list[float] = []
        for outcome in answerable:
            # 同一页的多个重叠 Chunk 不能因为重复召回而重复加分；nDCG 只采用每条
            # Gold Evidence 第一次被命中的排名。
            unique_evidence_ranks = sorted(
                {rank for rank in outcome.evidence_first_ranks if rank is not None and rank <= cutoff}
            )
            dcg = sum(_discount(rank) for rank in unique_evidence_ranks)
            ideal_relevant_count = min(cutoff, len(outcome.evidence_first_ranks))
            ideal_dcg = sum(_discount(rank) for rank in range(1, ideal_relevant_count + 1))
            ndcg_scores.append(dcg / ideal_dcg if ideal_dcg else 0.0)
        ndcg_at[cutoff] = fmean(ndcg_scores)

    reciprocal_ranks = [1.0 / outcome.relevant_ranks[0] if outcome.relevant_ranks else 0.0 for outcome in answerable]
    unanswerable_rate = (
        sum(outcome.result_count > 0 for outcome in unanswerable) / len(unanswerable) if unanswerable else None
    )
    latencies = [outcome.retrieval_latency_ms for outcome in outcomes]
    return EvaluationSummary(
        evaluated_cases=len(outcomes),
        answerable_cases=len(answerable),
        unanswerable_cases=len(unanswerable),
        question_recall_at=question_recall_at,
        evidence_recall_at=evidence_recall_at,
        mrr=fmean(reciprocal_ranks),
        ndcg_at=ndcg_at,
        mean_retrieval_latency_ms=fmean(latencies),
        unanswerable_top1_retrieval_rate=unanswerable_rate,
        p50_retrieval_latency_ms=_percentile(latencies, 0.50),
        p95_retrieval_latency_ms=_percentile(latencies, 0.95),
    )


def case_evaluation_to_dict(outcome: CaseEvaluation) -> dict[str, Any]:
    """为 JSONL 运行记录提供只包含基础类型的判定结果。"""
    return asdict(outcome)


def render_summary_markdown(*, experiment: ExperimentConfig, split: str, summary: EvaluationSummary) -> str:
    """生成可提交到 eval/results 的聚合报告，不包含论文全文或用户数据。"""
    rows = [
        ("评测问题数", str(summary.evaluated_cases)),
        ("可回答问题数", str(summary.answerable_cases)),
        ("不可回答问题数", str(summary.unanswerable_cases)),
        ("MRR", f"{summary.mrr:.4f}"),
        ("平均检索耗时（ms）", f"{summary.mean_retrieval_latency_ms:.2f}"),
        ("P50 检索耗时（ms）", f"{summary.p50_retrieval_latency_ms:.2f}"),
        ("P95 检索耗时（ms）", f"{summary.p95_retrieval_latency_ms:.2f}"),
    ]
    for cutoff in sorted(summary.question_recall_at):
        rows.extend(
            [
                (f"Question Recall@{cutoff}", f"{summary.question_recall_at[cutoff]:.4f}"),
                (f"Evidence Recall@{cutoff}", f"{summary.evidence_recall_at[cutoff]:.4f}"),
                (f"nDCG@{cutoff}", f"{summary.ndcg_at[cutoff]:.4f}"),
            ]
        )
    if summary.unanswerable_top1_retrieval_rate is not None:
        rows.append(("不可回答问题 Top-1 仍有召回的比例", f"{summary.unanswerable_top1_retrieval_rate:.4f}"))
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return (
        f"# {experiment.name} · {split} 检索评测\n\n"
        f"> {experiment.description}\n\n"
        "| 指标 | 数值 |\n|---|---:|\n"
        f"{table}\n\n"
        "说明：本报告仅评估确定性页级检索命中。不可回答问题当前只记录误召回，"
        "不将其解释为已校准的拒答准确率。\n"
    )
