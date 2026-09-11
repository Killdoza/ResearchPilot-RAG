"""基于可追溯检索证据的回答生成。"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from .answer_contract import enforce_answer_contract
from .evidence_alignment import ensure_annotation_citation_markers, ensure_factual_citation_markers
from .quantitative_answer import build_quantitative_evidence, render_quantitative_answer
from .retrieval import RetrievalResult
from .schemas import Answer, Citation, QueryType, ScoredChunk
from .summary_validation import validate_and_repair_summary


class ModelServiceUnavailable(RuntimeError):
    """模型请求未成功完成；调用方应提示重试，而不是破坏已有论文索引。"""


class TextGenerationClient(Protocol):
    """生成模块依赖的极小 LLM 契约，测试可使用固定字符串替身。"""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """根据系统指令和用户输入返回完整文本。"""


@dataclass(frozen=True)
class ModelUsageSnapshot:
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_observations: int = 0


def usage_delta(before: ModelUsageSnapshot, after: ModelUsageSnapshot) -> ModelUsageSnapshot:
    return ModelUsageSnapshot(
        model_calls=max(0, after.model_calls - before.model_calls),
        prompt_tokens=max(0, after.prompt_tokens - before.prompt_tokens),
        completion_tokens=max(0, after.completion_tokens - before.completion_tokens),
        token_observations=max(0, after.token_observations - before.token_observations),
    )


class LangChainTextGenerationClient:
    """将 LangChain ChatModel 适配为核心模块需要的生成契约。"""

    def __init__(
        self,
        chat_model: object,
        *,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._input_cost_per_million = input_cost_per_million
        self._output_cost_per_million = output_cost_per_million
        self._usage = ModelUsageSnapshot()
        self._usage_lock = threading.Lock()

    def usage_snapshot(self) -> ModelUsageSnapshot:
        with self._usage_lock:
            return self._usage

    def estimate_cost(self, usage: ModelUsageSnapshot) -> float | None:
        if (
            not usage.token_observations
            or self._input_cost_per_million is None
            or self._output_cost_per_million is None
        ):
            return None
        return (
            usage.prompt_tokens * self._input_cost_per_million + usage.completion_tokens * self._output_cost_per_million
        ) / 1_000_000

    @staticmethod
    def _response_tokens(response: object) -> tuple[int | None, int | None]:
        usage = getattr(response, "usage_metadata", None)
        if not isinstance(usage, dict):
            metadata = getattr(response, "response_metadata", None)
            usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
        if not isinstance(usage, dict):
            return None, None
        prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
        completion = usage.get("output_tokens", usage.get("completion_tokens"))
        return (
            int(prompt) if isinstance(prompt, int | float) and prompt >= 0 else None,
            int(completion) if isinstance(completion, int | float) and completion >= 0 else None,
        )

    def _record_usage(self, *, response: object | None = None, model_call: bool = False) -> None:
        prompt, completion = self._response_tokens(response) if response is not None else (None, None)
        with self._usage_lock:
            self._usage = ModelUsageSnapshot(
                model_calls=self._usage.model_calls + int(model_call),
                prompt_tokens=self._usage.prompt_tokens + (prompt or 0),
                completion_tokens=self._usage.completion_tokens + (completion or 0),
                token_observations=self._usage.token_observations + int(prompt is not None or completion is not None),
            )

    @staticmethod
    def _retryable_connection_error(error: Exception) -> bool:
        retryable_names = {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectTimeout",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
        }
        return isinstance(error, ConnectionError | TimeoutError) or type(error).__name__ in retryable_names

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        invoke = getattr(self._chat_model, "invoke", None)
        if not callable(invoke):
            raise TypeError("chat_model 必须提供 invoke 方法。")
        response = None
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                self._record_usage(model_call=True)
                response = invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0 and self._retryable_connection_error(exc):
                    time.sleep(0.25)
                    continue
                raise ModelServiceUnavailable("模型服务暂时不可用；请检查网络、API 配置或服务状态后重试。") from exc
        if response is None:
            raise ModelServiceUnavailable("模型服务暂时不可用；请检查网络、API 配置或服务状态后重试。") from last_error
        self._record_usage(response=response)
        content = getattr(response, "content", response)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("模型未返回可用文本。")
        return content.strip()

    def generate_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        on_text_delta: Callable[[str], None],
    ) -> str:
        """消费 ChatModel 的真实增量响应，同时返回最终完整文本。"""
        stream = getattr(self._chat_model, "stream", None)
        if not callable(stream):
            text = self.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            on_text_delta(text)
            return text

        fragments: list[str] = []
        for attempt in range(2):
            try:
                self._record_usage(model_call=True)
                responses = stream([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
                for response in responses:
                    content = getattr(response, "content", response)
                    if isinstance(content, str) and content:
                        fragments.append(content)
                        on_text_delta(content)
                    self._record_usage(response=response)
                break
            except Exception as exc:
                # A retry is safe only before the first visible token. Once a
                # delta was shown, repeating the call could duplicate content.
                if attempt == 0 and not fragments and self._retryable_connection_error(exc):
                    time.sleep(0.25)
                    continue
                raise ModelServiceUnavailable("模型服务暂时不可用；请检查网络、API 配置或服务状态后重试。") from exc

        text = "".join(fragments).strip()
        if not text:
            raise RuntimeError("模型未返回可用文本。")
        return text


@dataclass(frozen=True)
class GenerationContext:
    """回答时允许注入的非论文上下文；论文证据始终来自检索结果。"""

    long_term_memory: str = ""
    working_memory: str = ""
    chat_history: str = ""
    user_annotations: str = ""


METADATA_PROMPT_VERSION = "front_matter_json_v2"
SUMMARY_PROMPT_VERSION = "section_summary_v5_validated"


class MetadataIntent(str, Enum):
    TITLE = "title"
    AUTHORS = "authors"
    AFFILIATIONS = "affiliations"
    VENUE = "venue"
    YEAR = "year"
    DOI = "doi"
    OTHER = "other"


class SummaryIntent(str, Enum):
    OVERVIEW = "overview"
    MOTIVATION_TO_RESULTS = "motivation_to_results"
    CONTRIBUTIONS_AND_LIMITATIONS = "contributions_and_limitations"


def infer_metadata_intent(question: str) -> MetadataIntent:
    """Resolve the requested metadata field without asking an LLM."""
    lowered = question.casefold()
    if re.search(r"年龄|年纪|多大|几岁|age|出生|生日|个人", lowered):
        return MetadataIntent.OTHER
    if re.search(r"\bdoi\b", lowered):
        return MetadataIntent.DOI
    if re.search(r"机构|单位|affiliation|institution", lowered):
        return MetadataIntent.AFFILIATIONS
    if re.search(r"作者|author", lowered):
        return MetadataIntent.AUTHORS
    if re.search(r"标题|题目|title", lowered):
        return MetadataIntent.TITLE
    if re.search(r"会议|期刊|发表|venue|conference|journal", lowered):
        return MetadataIntent.VENUE
    if re.search(r"年份|哪年|何时|year|when", lowered):
        return MetadataIntent.YEAR
    return MetadataIntent.OTHER


def infer_summary_intent(question: str) -> SummaryIntent:
    lowered = question.casefold()
    explicit_dimensions = re.search(
        r"动机|研究问题|实验(?:设置|结果)?|贡献|局限|限制|不足|"
        r"motivation|experiment|result|conclusion|limitation|contribution",
        lowered,
    )
    if re.search(r"总结|概述|介绍|overview|summar", lowered) and explicit_dimensions is None:
        return SummaryIntent.OVERVIEW
    if re.search(r"贡献|局限|限制|不足|limitation|contribution", lowered):
        return SummaryIntent.CONTRIBUTIONS_AND_LIMITATIONS
    if re.search(r"动机|研究问题|实验|结果|结论|motivation|result|conclusion", lowered):
        return SummaryIntent.MOTIVATION_TO_RESULTS
    return SummaryIntent.OVERVIEW


def summary_request_fingerprint(question: str, intent: SummaryIntent) -> str:
    normalized = re.sub(r"\s+", " ", question.strip().casefold())
    return sha256(f"{intent.value}\n{normalized}".encode()).hexdigest()


@dataclass(frozen=True)
class ExtractedPaperMetadata:
    """Strict, evidence-derived fields returned by the metadata extractor."""

    title: str | None
    authors: tuple[str, ...]
    affiliations: tuple[str, ...]
    venue: str | None
    year: str | None
    doi: str | None
    answer_text: str
    field_confidence: dict[str, float]


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"unknown", "null", "none", "n/a", "未提及", "未知"}:
        return None
    return stripped


def _text_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        text = _optional_text(item)
        if text is not None and text not in result:
            result.append(text)
    return tuple(result)


def _extract_json_object(text: str) -> dict[str, object] | None:
    """Accept plain JSON or a fenced JSON response without guessing fields."""
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced is not None:
        candidate = fenced.group(1)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def format_metadata_answer(metadata: ExtractedPaperMetadata) -> str:
    """Render cached fields consistently without a second model invocation."""
    rows: list[tuple[str, str | None]] = [
        ("标题", metadata.title),
        ("作者", "、".join(metadata.authors) if metadata.authors else None),
        ("机构", "；".join(metadata.affiliations) if metadata.affiliations else None),
        ("发表信息", metadata.venue),
        ("年份", metadata.year),
        ("DOI", metadata.doi),
    ]
    present = [f"- **{label}**：{value}" for label, value in rows if value]
    if not present:
        return "论文首页证据中未能稳定抽取到结构化元信息；请查看下方页码证据并人工核验。"
    return "根据论文首页的可追溯证据，已抽取到以下元信息：\n" + "\n".join(present)


def format_metadata_field_answer(
    metadata: object,
    intent: MetadataIntent,
) -> tuple[str, bool]:
    """Render only the field the user asked for; bool reports availability."""
    values: dict[MetadataIntent, tuple[str, object]] = {
        MetadataIntent.TITLE: ("标题", getattr(metadata, "title", None)),
        MetadataIntent.AUTHORS: ("作者", getattr(metadata, "authors", ())),
        MetadataIntent.AFFILIATIONS: ("机构", getattr(metadata, "affiliations", ())),
        MetadataIntent.VENUE: ("发表信息", getattr(metadata, "venue", None)),
        MetadataIntent.YEAR: ("年份", getattr(metadata, "year", None)),
        MetadataIntent.DOI: ("DOI", getattr(metadata, "doi", None)),
    }
    if intent is MetadataIntent.OTHER:
        return "论文首页和已抽取的结构化元信息中没有提供所询问的个人信息，无法可靠回答。", False
    label, raw = values[intent]
    if isinstance(raw, tuple):
        rendered = "、".join(str(item) for item in raw if str(item).strip())
    else:
        rendered = str(raw).strip() if raw is not None else ""
    if not rendered:
        return f"论文首页和已抽取的结构化元信息中没有提供{label}，无法可靠回答。", False
    return f"根据论文首页证据，论文的{label}是：{rendered}。", True


class PaperMetadataExtractor:
    """Extract a reusable metadata record from front-matter evidence once."""

    def __init__(self, client: TextGenerationClient) -> None:
        self._client = client
        self.last_usage = ModelUsageSnapshot()
        self.last_estimated_cost: float | None = None

    def extract(self, *, retrieval: RetrievalResult) -> ExtractedPaperMetadata | None:
        if not retrieval.results:
            return None
        system_prompt = """你是 ResearchPilot 的论文元信息抽取器。只能依据给出的论文首页证据，
不要根据文件名、常识或外部知识猜测。请只返回一个合法 JSON 对象，不能使用 Markdown 或解释文字：
{
  "title": "字符串或 null",
  "authors": ["作者 1"],
  "affiliations": ["机构 1"],
  "venue": "会议、期刊、预印本或 null",
  "year": "年份或 null",
  "doi": "DOI 字符串或 null"
}
匿名作者必须如实保留为 Anonymous Authors；证据没有明确写出的字段必须为 null 或空数组。"""
        snapshot = getattr(self._client, "usage_snapshot", None)
        before = snapshot() if callable(snapshot) else ModelUsageSnapshot()
        raw = self._client.generate(
            system_prompt=system_prompt,
            user_prompt=f"【论文首页证据】\n{_format_evidence(retrieval.results)}",
        )
        after = snapshot() if callable(snapshot) else before
        self.last_usage = usage_delta(before, after)
        estimate = getattr(self._client, "estimate_cost", None)
        self.last_estimated_cost = estimate(self.last_usage) if callable(estimate) else None
        payload = _extract_json_object(raw)
        if payload is None:
            return None
        title = _optional_text(payload.get("title"))
        authors = _text_list(payload.get("authors"))
        affiliations = _text_list(payload.get("affiliations"))
        venue = _optional_text(payload.get("venue"))
        year = _optional_text(payload.get("year"))
        doi = _optional_text(payload.get("doi"))
        if not any((title, authors, affiliations, venue, year, doi)):
            return None
        # This is an evidence-coverage signal, not an assertion that the model
        # has mathematically verified every field. It is kept conservative and
        # documented as such in the persistence layer.
        confidence = 0.7 if retrieval.results else 0.0
        fields = {
            name: confidence
            for name, value in (
                ("title", title),
                ("authors", authors),
                ("affiliations", affiliations),
                ("venue", venue),
                ("year", year),
                ("doi", doi),
            )
            if value
        }
        provisional = ExtractedPaperMetadata(
            title=title,
            authors=authors,
            affiliations=affiliations,
            venue=venue,
            year=year,
            doi=doi,
            answer_text="",
            field_confidence=fields,
        )
        return ExtractedPaperMetadata(
            title=title,
            authors=authors,
            affiliations=affiliations,
            venue=venue,
            year=year,
            doi=doi,
            answer_text=format_metadata_answer(provisional),
            field_confidence=fields,
        )


def _format_evidence(results: Sequence[ScoredChunk]) -> str:
    evidence_parts: list[str] = []
    for position, scored_chunk in enumerate(results, start=1):
        chunk = scored_chunk.chunk
        source = scored_chunk.source_name or chunk.paper_id
        pages = (
            f"第 {chunk.page_start} 页"
            if chunk.page_start == chunk.page_end
            else f"第 {chunk.page_start}-{chunk.page_end} 页"
        )
        evidence_parts.append(
            f"[证据 {position} | 来源：{source} | {pages} | Chunk：{chunk.chunk_id}]\n{chunk.content}"
        )
    return "\n\n".join(evidence_parts)


def _build_citations(results: Sequence[ScoredChunk]) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    for scored_chunk in results:
        chunk = scored_chunk.chunk
        citations.append(
            Citation(
                paper_id=chunk.paper_id,
                chunk_id=chunk.chunk_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                evidence=chunk.content,
                confidence=scored_chunk.rerank_score or scored_chunk.fusion_score or scored_chunk.dense_score,
            )
        )
    return tuple(citations)


class AnswerGenerator:
    """只在有论文证据时调用模型，并输出结构化 Answer。"""

    def __init__(self, client: TextGenerationClient) -> None:
        self._client = client

    def generate(
        self,
        *,
        question: str,
        retrieval: RetrievalResult,
        context: GenerationContext | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> Answer:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 不能为空。")
        started_at = time.perf_counter()
        if not retrieval.results:
            answer = Answer(
                text="我没有在当前上传论文中检索到足以支持该问题的证据，因此不能可靠作答。",
                citations=(),
                refused=True,
                refusal_reason="未检索到论文证据。",
                trace_id=retrieval.trace.trace_id,
                latency_ms=(time.perf_counter() - started_at) * 1000,
            )
            if on_text_delta is not None:
                on_text_delta(answer.text)
            return answer

        resolved_context = context or GenerationContext()
        if retrieval.trace.query_type is not None and retrieval.trace.query_type.value == "experimental_results":
            quantitative = build_quantitative_evidence(question, retrieval.results)
            deterministic_text = render_quantitative_answer(quantitative)
            if deterministic_text is not None:
                if on_text_delta is not None:
                    on_text_delta(deterministic_text)
                return enforce_answer_contract(
                    Answer(
                        text=deterministic_text,
                        citations=_build_citations(retrieval.results),
                        refused=False,
                        trace_id=retrieval.trace.trace_id,
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                    )
                )
        task_instruction = {
            "metadata": (
                "这是论文元信息问题。只转写证据中明确出现的标题、作者、机构、年份、会议或 DOI；"
                "不得根据文件名或常识猜测。作者匿名时必须如实说明。"
            ),
            "document_summary": (
                "这是论文整体总结任务。按研究问题、核心方法、实验与结果、结论与局限组织回答；"
                "某一部分没有证据时明确省略或说明不足，不要把方法描述冒充实验结果。"
                "如果证据中含结果表格、胜率、准确率、延迟或成本等定量信息，必须报告有代表性的"
                "具体数值，不得声称论文没有定量结果。"
            ),
            "experimental_setup": (
                "这是实验设置任务。优先提取数据集、基线、评价指标、训练/实现配置和硬件；"
                "只回答证据明确覆盖的项目，不得把示例或相关工作误当成本文实验。"
            ),
            "experimental_results": (
                "这是实验结果任务。优先提取主要结果、与 Baseline 的定量比较、消融实验、"
                "效率与成本结果；如果证据中出现表格或百分比，必须报告代表性数值；"
                "没有召回具体数值时保守说明，但不得否认论文实验章节存在。"
            ),
            "multi_section": (
                "这是跨章节综合问题。必须分别陈述研究问题/动机、方法设计、实验结果或结论中"
                "实际被问题要求的部分，再说明它们之间的联系；每一部分都必须有自己的证据编号，"
                "不得用工作记忆补齐缺失章节。"
            ),
        }.get(retrieval.trace.query_type.value if retrieval.trace.query_type else "", "")
        if retrieval.trace.query_type is not None and retrieval.trace.query_type.value == "document_summary":
            summary_intent = infer_summary_intent(question)
            summary_contract = {
                SummaryIntent.OVERVIEW: "研究问题、核心方法、实验设置、主要结果、结论与局限",
                SummaryIntent.MOTIVATION_TO_RESULTS: "研究动机、方法演进、实验设置、实验结果、结论",
                SummaryIntent.CONTRIBUTIONS_AND_LIMITATIONS: "核心贡献、证据、局限、未明确说明的边界",
            }[summary_intent]
            task_instruction += (
                f" 必须使用二级标题依次覆盖：{summary_contract}。"
                "不要添加空表格；表格每个数据行都必须在行末给出证据编号。"
                "若某个必需部分的证据不足，只写‘当前检索证据不足’，不得补造内容。"
            )
        if resolved_context.user_annotations:
            task_instruction += (
                " 当前回答包含用户批注，必须使用“论文原文”“用户批注”“二者关系”三个明确标签；"
                "论文事实与用户观点不得合并陈述。"
            )
        system_prompt = f"""你是 ResearchPilot，一个面向人工智能与计算机论文阅读的严谨助手。
只能依据下方给出的论文证据回答；不要补造实验数据、方法细节、论文结论或来源。
若证据不足，请明确说明不确定性。回答应使用中文、简洁清晰；每个事实段落、项目符号或
表格行都必须在同一行末尾标明对应的“证据编号”。系统记忆和聊天历史只能帮助理解问题，
不能替代论文证据。
用户批注明确属于个人笔记：不得把批注伪装成论文原文，也不得让它静默覆盖论文事实；
若批注与原文冲突，必须分别陈述“论文原文”和“用户笔记”。
{task_instruction}"""
        user_prompt = f"""【用户问题】
{question.strip()}

【论文证据】
{_format_evidence(retrieval.results)}

【长期记忆（仅供理解偏好）】
{resolved_context.long_term_memory or "无"}

【工作记忆（仅供理解当前对话）】
{resolved_context.working_memory or "无"}

【近期聊天记录（仅供处理指代）】
{resolved_context.chat_history or "无"}

【用户批注（不是论文事实）】
{resolved_context.user_annotations or "无"}
"""
        snapshot = getattr(self._client, "usage_snapshot", None)
        before = snapshot() if callable(snapshot) else ModelUsageSnapshot()
        stream_generate = getattr(self._client, "generate_stream", None)
        if on_text_delta is not None and callable(stream_generate):
            text = stream_generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                on_text_delta=on_text_delta,
            )
        else:
            text = self._client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            if on_text_delta is not None:
                on_text_delta(text)
        after = snapshot() if callable(snapshot) else before
        model_usage = usage_delta(before, after)
        estimate = getattr(self._client, "estimate_cost", None)
        estimated_cost = estimate(model_usage) if callable(estimate) else None
        if resolved_context.user_annotations:
            text = ensure_annotation_citation_markers(text, len(retrieval.results))
        elif retrieval.trace.query_type not in {
            None,
            QueryType.DOCUMENT_SUMMARY,
            QueryType.METADATA,
            QueryType.GENERAL_CHAT,
        }:
            text = ensure_factual_citation_markers(text, len(retrieval.results))
        answer = enforce_answer_contract(
            Answer(
                text=text,
                citations=_build_citations(retrieval.results),
                refused=False,
                trace_id=retrieval.trace.trace_id,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                answer_model_calls=model_usage.model_calls,
                prompt_tokens=(model_usage.prompt_tokens if model_usage.token_observations else None),
                completion_tokens=(model_usage.completion_tokens if model_usage.token_observations else None),
                estimated_cost=estimated_cost,
            )
        )
        if retrieval.trace.query_type is not None and retrieval.trace.query_type.value == "document_summary":
            summary_intent = infer_summary_intent(question)
            validation = validate_and_repair_summary(answer.text, intent=summary_intent.value)
            if validation.repaired:
                answer = enforce_answer_contract(
                    replace(
                        answer,
                        text=validation.text,
                        claims=(),
                        contract_version="legacy",
                        numeric_fidelity_passed=None,
                    ),
                    default_to_all_citations=False,
                )
        return answer
