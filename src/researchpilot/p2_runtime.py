"""P2 运行时输入边界：Gold 只供 Evaluator，Planner 只接收用户可见信息。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .crossdoc import CrossDocumentEvidence
from .p2_dataset import P2Case
from .query_analysis import QueryAnalysis, RuleQueryAnalyzer
from .query_plan import (
    P2RulesRouter,
    P2RunMode,
    QueryPlan,
    QueryTypeSource,
    ScopeSource,
)
from .schemas import QueryType, SchemaValidationError
from .scope_resolution import PaperScopeCatalog, ScopeResolution


def _paper_keys(values: Iterable[str], label: str) -> tuple[str, ...]:
    result = tuple(value.strip() for value in values if isinstance(value, str) and value.strip())
    if len(set(result)) != len(result):
        raise SchemaValidationError(f"{label} 不能重复。")
    if any(not value.replace("_", "a").replace("-", "a").replace(".", "a").isalnum() for value in result):
        raise SchemaValidationError(f"{label} 必须由合法 paper_key 组成。")
    return result


@dataclass(frozen=True)
class RuntimeRequest:
    """Planner 唯一可见的请求，不含 Gold 标签、答案或证据。"""

    question: str
    session_available_paper_keys: tuple[str, ...]
    user_selected_paper_keys: tuple[str, ...] = ()
    experiment_mode: P2RunMode = P2RunMode.END_TO_END

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise SchemaValidationError("question 不能为空。")
        available = _paper_keys(self.session_available_paper_keys, "session_available_paper_keys")
        selected = _paper_keys(self.user_selected_paper_keys, "user_selected_paper_keys")
        if not available:
            raise SchemaValidationError("session_available_paper_keys 不能为空。")
        if not set(selected).issubset(available):
            raise SchemaValidationError("user_selected_paper_keys 必须属于 session_available_paper_keys。")
        if not isinstance(self.experiment_mode, P2RunMode):
            raise SchemaValidationError("experiment_mode 必须是 P2RunMode。")
        if self.experiment_mode == P2RunMode.ORACLE:
            raise SchemaValidationError("Oracle 请求必须通过 GoldAnnotation 显式构造。")


@dataclass(frozen=True)
class GoldAnnotation:
    """仅 Evaluator / Oracle 上界使用的人工标注；禁止进入端到端接口。"""

    case_id: str
    gold_query_type: QueryType
    required_paper_keys: tuple[str, ...]
    answerable: bool
    gold_evidence: tuple[CrossDocumentEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise SchemaValidationError("case_id 不能为空。")
        if not isinstance(self.gold_query_type, QueryType):
            raise SchemaValidationError("gold_query_type 必须是 QueryType。")
        if len(_paper_keys(self.required_paper_keys, "required_paper_keys")) < 2:
            raise SchemaValidationError("required_paper_keys 至少需要两篇论文。")
        if not isinstance(self.answerable, bool):
            raise SchemaValidationError("answerable 必须是布尔值。")
        if any(not isinstance(item, CrossDocumentEvidence) for item in self.gold_evidence):
            raise SchemaValidationError("gold_evidence 必须由 CrossDocumentEvidence 组成。")

    @classmethod
    def from_case(cls, case: P2Case) -> GoldAnnotation:
        return cls(
            case_id=case.case_id,
            gold_query_type=(QueryType.UNANSWERABLE if not case.answerable else QueryType.CROSS_DOCUMENT),
            required_paper_keys=case.required_paper_keys,
            answerable=case.answerable,
            gold_evidence=case.gold_evidence,
        )


@dataclass(frozen=True)
class PlanningOutcome:
    """Planner 的成功计划或范围不足回退；两者都可写入 Trace。"""

    mode: P2RunMode
    scope: ScopeResolution
    plan: QueryPlan | None
    fallback_reason: str | None = None
    analysis: QueryAnalysis | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, P2RunMode):
            raise SchemaValidationError("mode 必须是 P2RunMode。")
        if not isinstance(self.scope, ScopeResolution):
            raise SchemaValidationError("scope 必须是 ScopeResolution。")
        if self.plan is None:
            if not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip():
                raise SchemaValidationError("无计划时必须给出 fallback_reason。")
        elif self.fallback_reason is not None:
            raise SchemaValidationError("有 QueryPlan 时不能同时给出 fallback_reason。")
        if self.analysis is not None and not isinstance(self.analysis, QueryAnalysis):
            raise SchemaValidationError("analysis 必须是 QueryAnalysis 或 None。")

    @property
    def is_scope_insufficient(self) -> bool:
        return self.plan is None and self.fallback_reason == "scope_insufficient"


class P2RuntimePlanner:
    """分模式构造计划；端到端入口没有 GoldAnnotation 参数。"""

    def __init__(
        self,
        *,
        catalog: PaperScopeCatalog,
        router: P2RulesRouter | None = None,
        analyzer: RuleQueryAnalyzer | None = None,
    ) -> None:
        self._catalog = catalog
        self._router = router or P2RulesRouter()
        self._analyzer = analyzer or RuleQueryAnalyzer(catalog=catalog)

    def plan_oracle(self, *, request: RuntimeRequest, gold: GoldAnnotation) -> PlanningOutcome:
        if request.experiment_mode != P2RunMode.END_TO_END:
            raise SchemaValidationError("Oracle 上界只能从独立的 End-to-End RuntimeRequest 派生。")
        if not set(gold.required_paper_keys).issubset(request.session_available_paper_keys):
            raise SchemaValidationError("Gold 论文不在本次可访问 Scope 中，不能构造 Oracle 上界。")
        scope = ScopeResolution(gold.required_paper_keys, ScopeSource.ORACLE_EVAL)
        plan = self._router.plan(
            query_type=gold.gold_query_type,
            target_paper_keys=scope.paper_keys,
            experiment_mode=P2RunMode.ORACLE,
            query_type_source=QueryTypeSource.ORACLE_EVAL,
            scope_source=ScopeSource.ORACLE_EVAL,
        )
        return PlanningOutcome(P2RunMode.ORACLE, scope, plan)

    def plan_scoped(self, *, request: RuntimeRequest) -> PlanningOutcome:
        if request.experiment_mode != P2RunMode.SCOPED:
            raise SchemaValidationError("E6-Scoped 必须使用 experiment_mode=scoped。")
        analysis = self._analyzer.analyze(
            question=request.question,
            session_available_paper_keys=request.session_available_paper_keys,
            user_selected_paper_keys=request.user_selected_paper_keys,
        )
        scope = analysis.scope
        if scope.source != ScopeSource.USER_SELECTED:
            return PlanningOutcome(P2RunMode.SCOPED, scope, None, "scope_insufficient", analysis)
        if analysis.query_type is None:
            return PlanningOutcome(P2RunMode.SCOPED, scope, None, analysis.fallback_reason, analysis)
        return PlanningOutcome(
            P2RunMode.SCOPED,
            scope,
            self._router.plan(
                query_type=analysis.query_type,
                target_paper_keys=scope.paper_keys,
                experiment_mode=P2RunMode.SCOPED,
                query_type_source=QueryTypeSource.RULE_PREDICTED,
                scope_source=scope.source,
            ),
            analysis=analysis,
        )

    def plan_end_to_end(self, *, request: RuntimeRequest) -> PlanningOutcome:
        """端到端入口仅接受 RuntimeRequest，不能传入 Gold 或外部类型标签。"""
        if request.experiment_mode != P2RunMode.END_TO_END:
            raise SchemaValidationError("E6-End-to-End 必须使用 experiment_mode=end_to_end。")
        analysis = self._analyzer.analyze(
            question=request.question,
            session_available_paper_keys=request.session_available_paper_keys,
            user_selected_paper_keys=request.user_selected_paper_keys,
        )
        scope = analysis.scope
        if analysis.query_type is None:
            return PlanningOutcome(P2RunMode.END_TO_END, scope, None, analysis.fallback_reason, analysis)
        return PlanningOutcome(
            P2RunMode.END_TO_END,
            scope,
            self._router.plan(
                query_type=analysis.query_type,
                target_paper_keys=scope.paper_keys,
                experiment_mode=P2RunMode.END_TO_END,
                query_type_source=QueryTypeSource.RULE_PREDICTED,
                scope_source=scope.source,
            ),
            analysis=analysis,
        )
