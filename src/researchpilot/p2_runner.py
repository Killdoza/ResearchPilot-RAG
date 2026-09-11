"""P2-C 的统一运行前预算预检。

本模块刻意只规划运行时输入与估算最大预算：不构建索引、不调用 LLM，且不会
让 Gold Evidence 进入 Scoped / End-to-End Planner。真实检索与生成运行器只能
在 Dry-run 已冻结后复用其配置和记录格式。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .evaluation import EvaluationDataError, ExperimentConfig, load_experiment_config
from .p2_dataset import P2Case
from .p2_runtime import GoldAnnotation, P2RuntimePlanner, RuntimeRequest
from .query_plan import P2RulesRouter, P2RunMode, QueryPlan, QueryTypeSource, ScopeSource
from .schemas import QueryType

_CONFIG_MODES = {
    "static_baseline": P2RunMode.STATIC,
    "oracle": P2RunMode.ORACLE,
    "scoped": P2RunMode.SCOPED,
    "end_to_end": P2RunMode.END_TO_END,
}


@dataclass(frozen=True)
class P2ExecutionConfig:
    retrieval_experiment: ExperimentConfig
    mode: P2RunMode
    per_paper_quota: int
    max_input_tokens: int
    max_output_tokens: int
    temperature: float
    input_cost_per_million: float | None
    output_cost_per_million: float | None
    evaluation_user_id: str
    evaluation_session_id: str
    coverage_gate_minimum_score: float = 0.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.per_paper_quota, "per_paper_quota"),
            (self.max_input_tokens, "max_input_tokens"),
            (self.max_output_tokens, "max_output_tokens"),
        ):
            if not isinstance(value, int) or value < 1:
                raise EvaluationDataError(f"{name} 必须为正整数。")
        if not isinstance(self.temperature, int | float) or not math.isfinite(float(self.temperature)):
            raise EvaluationDataError("temperature 必须为有限数值。")
        for value, name in (
            (self.input_cost_per_million, "input_cost_per_million"),
            (self.output_cost_per_million, "output_cost_per_million"),
        ):
            if value is not None and (not isinstance(value, int | float) or value < 0):
                raise EvaluationDataError(f"{name} 必须为非负数或 null。")
        if not self.evaluation_user_id.strip() or not self.evaluation_session_id.strip():
            raise EvaluationDataError("P2 evaluation scope 不能为空。")
        if not isinstance(self.coverage_gate_minimum_score, int | float) or not math.isfinite(
            float(self.coverage_gate_minimum_score)
        ):
            raise EvaluationDataError("coverage_gate_minimum_score 必须为有限数值。")


def load_p2_execution_config(path: Path) -> P2ExecutionConfig:
    """读取检索 YAML 附带的 P2 运行和预算契约。"""
    experiment = load_experiment_config(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise EvaluationDataError(f"P2 配置不是合法 YAML：{error}") from error
    if not isinstance(raw, Mapping):
        raise EvaluationDataError("P2 配置根节点必须是对象。")
    p2 = raw.get("p2")
    answer = raw.get("answer_evaluation")
    if not isinstance(p2, Mapping) or not isinstance(answer, Mapping):
        raise EvaluationDataError("P2 配置必须包含 p2 和 answer_evaluation 节。")
    mode_name = p2.get("mode")
    mode = _CONFIG_MODES.get(mode_name) if isinstance(mode_name, str) else None
    if mode is None:
        raise EvaluationDataError(f"P2 p2.mode 不受支持：{mode_name}")
    generation = answer.get("generation")
    pricing = answer.get("pricing", {})
    scope = answer.get("evaluation_scope")
    if not isinstance(generation, Mapping) or not isinstance(pricing, Mapping) or not isinstance(scope, Mapping):
        raise EvaluationDataError("P2 answer_evaluation 的 generation/pricing/evaluation_scope 必须为对象。")
    # Static baseline deliberately records quota=0 because it does not perform
    # per-paper reservation.  The common execution contract still requires a
    # positive placeholder, which is never used by the static branch.
    quota = p2.get("per_paper_quota", 1)
    if mode == P2RunMode.STATIC and quota == 0:
        quota = 1
    return P2ExecutionConfig(
        retrieval_experiment=experiment,
        mode=mode,
        per_paper_quota=quota,
        max_input_tokens=answer.get("max_input_tokens"),
        max_output_tokens=generation.get("max_output_tokens"),
        temperature=generation.get("temperature", 0.0),
        input_cost_per_million=pricing.get("input_cost_per_million"),
        output_cost_per_million=pricing.get("output_cost_per_million"),
        evaluation_user_id=scope.get("user_id", ""),
        evaluation_session_id=scope.get("session_id", ""),
        coverage_gate_minimum_score=p2.get("coverage_gate_minimum_score", 0.0),
    )


@dataclass(frozen=True)
class P2DryRunRecord:
    case_id: str
    split: str
    mode: str
    query_type: str | None
    query_type_source: str | None
    scope_source: str | None
    target_paper_keys: tuple[str, ...]
    fallback_reason: str | None
    retrieval_calls_upper_bound: int
    model_calls_upper_bound: int
    estimated_prompt_tokens_upper_bound: int
    estimated_completion_tokens_upper_bound: int
    estimated_cost_upper_bound: float | None
    within_context_limit: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["target_paper_keys"] = list(self.target_paper_keys)
        return result


@dataclass(frozen=True)
class P2DryRunSummary:
    config_name: str
    mode: str
    evaluated_cases: int
    retrieval_calls_upper_bound: int
    model_calls_upper_bound: int
    prompt_tokens_upper_bound: int
    completion_tokens_upper_bound: int
    cost_upper_bound: float | None
    capacity_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["capacity_failures"] = list(self.capacity_failures)
        return result


def _static_plan(case: P2Case) -> QueryPlan:
    """Static 对照使用固定 Hybrid 预算，不伪装为 Query-aware 路由。"""
    return P2RulesRouter().plan(
        query_type=QueryType.CROSS_DOCUMENT,
        target_paper_keys=case.paper_keys,
        experiment_mode=P2RunMode.STATIC,
        query_type_source=QueryTypeSource.STATIC_CONFIG,
        scope_source=ScopeSource.SESSION_DEFAULT,
    )


def _runtime_outcome(*, case: P2Case, config: P2ExecutionConfig, planner: P2RuntimePlanner):
    # `paper_keys` is the predeclared user-visible test scenario; only
    # `required_paper_keys` and evidence are Gold.  The latter is passed solely
    # to the explicit Oracle branch.
    if config.mode == P2RunMode.STATIC:
        return _static_plan(case), None
    request = RuntimeRequest(
        question=case.question,
        session_available_paper_keys=case.paper_keys,
        user_selected_paper_keys=(case.paper_keys if config.mode == P2RunMode.SCOPED else ()),
        experiment_mode=(P2RunMode.SCOPED if config.mode == P2RunMode.SCOPED else P2RunMode.END_TO_END),
    )
    if config.mode == P2RunMode.ORACLE:
        outcome = planner.plan_oracle(request=request, gold=GoldAnnotation.from_case(case))
    elif config.mode == P2RunMode.SCOPED:
        outcome = planner.plan_scoped(request=request)
    else:
        outcome = planner.plan_end_to_end(request=request)
    return outcome.plan, outcome.fallback_reason


def build_p2_dry_run(
    *, cases: Sequence[P2Case], config: P2ExecutionConfig, planner: P2RuntimePlanner
) -> tuple[tuple[P2DryRunRecord, ...], P2DryRunSummary]:
    """计算保守上界：E6 最多初检 + 一次逐论文扩检 + 一次回答调用。"""
    records: list[P2DryRunRecord] = []
    for case in cases:
        plan, fallback = _runtime_outcome(case=case, config=config, planner=planner)
        if plan is None:
            retrieval_calls = model_calls = prompt_tokens = completion_tokens = 0
            target = ()
            query_type = query_type_source = scope_source = None
        else:
            target = plan.target_paper_keys
            query_type = plan.query_type.value
            query_type_source = plan.query_type_source.value
            scope_source = plan.scope_source.value
            # Static performs one global retrieval. Query-aware paths run every
            # selected paper; retry is included only when the plan allows it.
            retrieval_calls = 1 if config.mode == P2RunMode.STATIC else len(target) * (2 if plan.allow_one_retry else 1)
            model_calls = 1
            context_chars = config.retrieval_experiment.result_limit * 1200
            prompt_tokens = math.ceil((len(case.question) + context_chars + 1200) / 3.5)
            completion_tokens = config.max_output_tokens
        cost = None
        if config.input_cost_per_million is not None and config.output_cost_per_million is not None:
            cost = (
                prompt_tokens * float(config.input_cost_per_million)
                + completion_tokens * float(config.output_cost_per_million)
            ) / 1_000_000
        records.append(
            P2DryRunRecord(
                case_id=case.case_id,
                split=case.split,
                mode=config.mode.value,
                query_type=query_type,
                query_type_source=query_type_source,
                scope_source=scope_source,
                target_paper_keys=target,
                fallback_reason=fallback,
                retrieval_calls_upper_bound=retrieval_calls,
                model_calls_upper_bound=model_calls,
                estimated_prompt_tokens_upper_bound=prompt_tokens,
                estimated_completion_tokens_upper_bound=completion_tokens,
                estimated_cost_upper_bound=cost,
                within_context_limit=prompt_tokens <= config.max_input_tokens,
            )
        )
    costs = [item.estimated_cost_upper_bound for item in records]
    return tuple(records), P2DryRunSummary(
        config_name=config.retrieval_experiment.name,
        mode=config.mode.value,
        evaluated_cases=len(records),
        retrieval_calls_upper_bound=sum(item.retrieval_calls_upper_bound for item in records),
        model_calls_upper_bound=sum(item.model_calls_upper_bound for item in records),
        prompt_tokens_upper_bound=sum(item.estimated_prompt_tokens_upper_bound for item in records),
        completion_tokens_upper_bound=sum(item.estimated_completion_tokens_upper_bound for item in records),
        cost_upper_bound=(
            sum(item for item in costs if item is not None) if all(item is not None for item in costs) else None
        ),
        capacity_failures=tuple(item.case_id for item in records if not item.within_context_limit),
    )
