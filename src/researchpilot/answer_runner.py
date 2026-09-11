"""P1-E 回答运行核心：统一四种上下文模式的 Prompt、JSON 输出和使用量记录。"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from .answer_evaluation import AnswerCitation, GeneratedAnswer
from .context_modes import ContextBundle, E5BlockRepository, full_context, retrieval_context
from .crossdoc import CrossDocumentCase
from .evaluation import CorpusManifest, EvalCase, EvaluationDataError, ExperimentConfig, load_experiment_config
from .retrieval import RetrievalService

ANSWER_EXPERIMENT_SECTION = "answer_evaluation"
_MODES = frozenset({"lc1", "lc2", "n0", "a0"})


@dataclass(frozen=True)
class AnswerExperimentConfig:
    """在已有检索 YAML 之上增加 P1-E 专属的生成协议。"""

    retrieval_experiment: ExperimentConfig
    context_mode: str
    artifact_dir: Path
    parser_name: str
    parser_version: str
    max_input_tokens: int
    max_output_tokens: int
    temperature: float
    input_cost_per_million: float | None
    output_cost_per_million: float | None
    evaluation_user_id: str
    evaluation_session_id: str

    def __post_init__(self) -> None:
        if self.context_mode not in _MODES:
            raise EvaluationDataError("context_mode 只能为 lc1、lc2、n0 或 a0。")
        if self.retrieval_experiment.parser_name != self.parser_name:
            raise EvaluationDataError("P1-E 配置 Parser 必须与 E5 Block Parser 一致。")
        if self.retrieval_experiment.parser_version != self.parser_version:
            raise EvaluationDataError("P1-E 配置 Parser 版本必须与 E5 Block 一致。")
        if self.retrieval_experiment.chunker_version != "structure_aware_v1":
            raise EvaluationDataError("P1-E 必须复用 E5 structure_aware_v1 Block。")
        for value, name in ((self.max_input_tokens, "max_input_tokens"), (self.max_output_tokens, "max_output_tokens")):
            if not isinstance(value, int) or value < 1:
                raise EvaluationDataError(f"{name} 必须是正整数。")
        if not isinstance(self.temperature, float | int) or not math.isfinite(float(self.temperature)):
            raise EvaluationDataError("temperature 必须是有限数值。")
        for value, name in (
            (self.input_cost_per_million, "input_cost_per_million"),
            (self.output_cost_per_million, "output_cost_per_million"),
        ):
            if value is not None and (not isinstance(value, float | int) or value < 0):
                raise EvaluationDataError(f"{name} 必须是非负数或 null。")
        if not self.evaluation_user_id.strip() or not self.evaluation_session_id.strip():
            raise EvaluationDataError("P1-E evaluation scope 的 user_id、session_id 不能为空。")


@dataclass(frozen=True)
class ModelCompletion:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class StructuredAnswerClient(Protocol):
    def complete(
        self, *, system_prompt: str, user_prompt: str, max_output_tokens: int, temperature: float
    ) -> ModelCompletion:
        """返回模型文本和（若供应商返回）准确 Token 用量。"""


class LangChainStructuredAnswerClient:
    """将 ChatOpenAI 适配为可记录 usage metadata 的回答客户端。"""

    def __init__(self, chat_model: object) -> None:
        self._chat_model = chat_model

    def complete(
        self, *, system_prompt: str, user_prompt: str, max_output_tokens: int, temperature: float
    ) -> ModelCompletion:
        from langchain_core.messages import HumanMessage, SystemMessage

        invoke = getattr(self._chat_model, "invoke", None)
        if not callable(invoke):
            raise TypeError("chat_model 必须提供 invoke 方法。")
        response = invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
        content = getattr(response, "content", response)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("模型未返回可用文本。")
        usage = getattr(response, "usage_metadata", None)
        if not isinstance(usage, Mapping):
            metadata = getattr(response, "response_metadata", {})
            usage = metadata.get("token_usage", metadata.get("usage")) if isinstance(metadata, Mapping) else None
        prompt_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
        completion_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
        return ModelCompletion(text=content.strip(), prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def _usage_int(usage: object, *keys: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def load_answer_experiment_config(path: Path, *, project_root: Path) -> AnswerExperimentConfig:
    """加载兼容 ExperimentConfig 的 YAML，并读取额外 answer_evaluation 节。"""
    experiment = load_experiment_config(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise EvaluationDataError(f"P1-E 配置不是合法 YAML：{error}") from error
    if not isinstance(raw, Mapping) or not isinstance(raw.get(ANSWER_EXPERIMENT_SECTION), Mapping):
        raise EvaluationDataError(f"P1-E 配置缺少 {ANSWER_EXPERIMENT_SECTION} 节。")
    section = raw[ANSWER_EXPERIMENT_SECTION]
    mode = section.get("context_mode")
    artifact_value = section.get("e5_artifact_dir")
    artifact_dir = Path(artifact_value) if isinstance(artifact_value, str) else None
    if artifact_dir is None:
        raise EvaluationDataError("answer_evaluation.e5_artifact_dir 必须为路径字符串。")
    if not artifact_dir.is_absolute():
        artifact_dir = project_root / artifact_dir
    pricing = section.get("pricing", {})
    if not isinstance(pricing, Mapping):
        raise EvaluationDataError("answer_evaluation.pricing 必须为对象。")
    generation = section.get("generation", {})
    if not isinstance(generation, Mapping):
        raise EvaluationDataError("answer_evaluation.generation 必须为对象。")
    scope = section.get("evaluation_scope")
    if not isinstance(scope, Mapping):
        raise EvaluationDataError("answer_evaluation.evaluation_scope 必须为对象。")
    return AnswerExperimentConfig(
        retrieval_experiment=experiment,
        context_mode=mode if isinstance(mode, str) else "",
        artifact_dir=artifact_dir.resolve(),
        parser_name=section.get("parser_name", ""),
        parser_version=section.get("parser_version", ""),
        max_input_tokens=section.get("max_input_tokens"),
        max_output_tokens=generation.get("max_output_tokens"),
        temperature=generation.get("temperature", 0.0),
        input_cost_per_million=pricing.get("input_cost_per_million"),
        output_cost_per_million=pricing.get("output_cost_per_million"),
        evaluation_user_id=scope.get("user_id", ""),
        evaluation_session_id=scope.get("session_id", ""),
    )


def estimate_prompt_tokens(*, question: str, context: ContextBundle) -> int:
    """Dry-run 的保守估算。真实运行优先写入供应商返回的 usage metadata。"""
    # 英文约 4 字符/Token、中文通常更密；取 3.5 使提示不要低估大上下文风险。
    return math.ceil((len(question) + len(context.render()) + 900) / 3.5)


def build_context_for_case(
    *,
    case: EvalCase | CrossDocumentCase,
    config: AnswerExperimentConfig,
    corpus: CorpusManifest,
    block_repository: E5BlockRepository,
    retrieval_service: RetrievalService | None = None,
    user_id: str = "eval_ai_cs_v01",
    session_id: str = "corpus_v01_e5",
) -> ContextBundle:
    """只在 N0/A0 调用检索，LC1/LC2 明确禁止隐式检索。"""
    if config.context_mode == "lc1":
        return full_context(mode="lc1", block_repository=block_repository, paper_keys=case.paper_keys)
    if config.context_mode == "lc2":
        return full_context(mode="lc2", block_repository=block_repository, paper_keys=case.paper_keys)
    if retrieval_service is None:
        raise EvaluationDataError(f"{config.context_mode} 需要显式提供 RetrievalService。")
    retrieval = retrieval_service.retrieve(
        query=case.question,
        search_query=case.question,
        user_id=user_id,
        session_id=session_id,
        recent_history="",
    )
    return retrieval_context(mode=config.context_mode, retrieval=retrieval, corpus=corpus)


def build_prompts(*, question: str, context: ContextBundle) -> tuple[str, str]:
    system_prompt = """你是 ResearchPilot 的论文证据问答评测助手。只能依据【论文上下文】回答，
不得使用外部常识、训练数据记忆或题目暗示。对于不同论文不存在同任务、同数据集、同指标直接比较的情况，
必须拒绝给出“谁更强”等结论。请用中文回答，并且只输出一个合法 JSON 对象，不要 Markdown 或额外解释：
{
  "answer": "回答正文",
  "citations": [{"paper_key": "论文键", "page_start": 1, "page_end": 1, "context_id": "CTX ID"}],
  "refused": false,
  "refusal_reason": null
}
若证据不足或问题不可回答，设置 refused=true、citations=[]，并给出 refusal_reason。每条 citation
必须来自实际提供的上下文标记，不能编造论文键或页码。"""
    user_prompt = f"""【问题】
{question.strip()}

【上下文选择方式】
{context.mode}：{context.selection_note}

【论文上下文】
{context.render()}
"""
    return system_prompt, user_prompt


def _extract_json(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE).strip()
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise EvaluationDataError("模型没有返回可解析的 JSON。")
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise EvaluationDataError("模型 JSON 格式非法。") from error
    if not isinstance(raw, Mapping):
        raise EvaluationDataError("模型 JSON 顶层必须是对象。")
    return raw


def _parse_citations(raw: object) -> tuple[AnswerCitation, ...]:
    if not isinstance(raw, list):
        raise EvaluationDataError("模型 JSON 的 citations 必须是数组。")
    citations: list[AnswerCitation] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise EvaluationDataError("模型 JSON 的 citations[] 必须是对象。")
        page = item.get("page")
        page_start = item.get("page_start", page)
        page_end = item.get("page_end", page_start)
        citations.append(
            AnswerCitation(
                paper_key=item.get("paper_key", ""),
                page_start=page_start,
                page_end=page_end,
                context_id=item.get("context_id"),
            )
        )
    return tuple(citations)


def _cost(*, prompt_tokens: int | None, completion_tokens: int | None, config: AnswerExperimentConfig) -> float | None:
    if (
        prompt_tokens is None
        or completion_tokens is None
        or config.input_cost_per_million is None
        or config.output_cost_per_million is None
    ):
        return None
    return (
        prompt_tokens * float(config.input_cost_per_million) + completion_tokens * float(config.output_cost_per_million)
    ) / 1_000_000


def generate_structured_answer(
    *,
    question: str,
    context: ContextBundle,
    config: AnswerExperimentConfig,
    client: StructuredAnswerClient,
) -> GeneratedAnswer:
    """调用一次模型并把 JSON 文本转为统一回答结构。"""
    system_prompt, user_prompt = build_prompts(question=question, context=context)
    estimated = estimate_prompt_tokens(question=question, context=context)
    if estimated > config.max_input_tokens:
        raise EvaluationDataError(
            f"上下文预计约 {estimated} Token，超过 {config.max_input_tokens} Token 上限；"
            "请在 Dev 阶段调整上限或选择支持更长上下文的模型，不会自动截断论文。"
        )
    started = time.perf_counter()
    completion = client.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=config.max_output_tokens,
        temperature=float(config.temperature),
    )
    generation_latency = (time.perf_counter() - started) * 1000
    raw = _extract_json(completion.text)
    refused = raw.get("refused")
    if not isinstance(refused, bool):
        raise EvaluationDataError("模型 JSON 的 refused 必须是布尔值。")
    answer = raw.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise EvaluationDataError("模型 JSON 的 answer 必须是非空字符串。")
    refusal_reason = raw.get("refusal_reason")
    if refusal_reason is not None and not isinstance(refusal_reason, str):
        raise EvaluationDataError("模型 JSON 的 refusal_reason 必须是字符串或 null。")
    citations = _parse_citations(raw.get("citations"))
    # 有些模型会违反 Prompt，在拒答时仍输出 citations。保留原输出，才能在评测中将
    # “拒答但仍引用”计为错误拒答，而不是让整轮实验因单个样本中断。
    if refused and not isinstance(refusal_reason, str):
        # 上游模型偶尔会把拒答理由遗漏为 null。将其规范成可追踪的占位说明，
        # 保留该样本供评测，而不是因一次格式瑕疵浪费整轮已完成的调用。
        refusal_reason = "模型未提供拒答理由。"
    if not refused:
        refusal_reason = None
    return GeneratedAnswer(
        answer=answer.strip(),
        citations=citations,
        refused=refused,
        refusal_reason=refusal_reason.strip() if isinstance(refusal_reason, str) else None,
        prompt_tokens=completion.prompt_tokens if completion.prompt_tokens is not None else estimated,
        completion_tokens=completion.completion_tokens,
        cost=_cost(
            prompt_tokens=completion.prompt_tokens if completion.prompt_tokens is not None else estimated,
            completion_tokens=completion.completion_tokens,
            config=config,
        ),
        retrieval_latency_ms=context.retrieval_latency_ms,
        generation_latency_ms=generation_latency,
    )
