"""P2 的可解释 QueryPlan 与不调用 LLM 的 ``p2_rules_v1`` 路由器。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum

from .schemas import QueryType, SchemaValidationError

ROUTER_VERSION = "p2_rules_v1"
_BLOCK_TYPES = frozenset({"metadata", "heading", "paragraph", "table", "caption", "formula"})


class P2RunMode(str, Enum):
    """P2 报告中必须分开的运行模式。"""

    STATIC = "static"
    ORACLE = "oracle"
    SCOPED = "scoped"
    END_TO_END = "end_to_end"


class QueryTypeSource(str, Enum):
    """Query Type 的来源；Oracle 不能被误写为模型预测。"""

    STATIC_CONFIG = "static_config"
    ORACLE_EVAL = "oracle_eval"
    RULE_PREDICTED = "rule_predicted"
    MODEL_PREDICTED = "model_predicted"


class ScopeSource(str, Enum):
    """论文范围的来源，直接决定结果可否称为端到端。"""

    NONE = "none"
    ORACLE_EVAL = "oracle_eval"
    USER_SELECTED = "user_selected"
    SESSION_DEFAULT = "session_default"
    TITLE_RESOLVED = "title_resolved"


@dataclass(frozen=True)
class QueryPlan:
    """一次检索允许使用的预算与证据约束，可直接写入 Trace JSON。"""

    query_type: QueryType
    target_paper_keys: tuple[str, ...]
    retrieval_policy: str
    dense_limit: int
    sparse_limit: int
    rerank_enabled: bool
    block_types: tuple[str, ...]
    minimum_evidence_per_paper: int
    allow_one_retry: bool
    coverage_gate_required: bool
    needs_retrieval: bool
    router_version: str = ROUTER_VERSION
    experiment_mode: P2RunMode = P2RunMode.STATIC
    query_type_source: QueryTypeSource = QueryTypeSource.STATIC_CONFIG
    scope_source: ScopeSource = ScopeSource.SESSION_DEFAULT

    def __post_init__(self) -> None:
        if not isinstance(self.query_type, QueryType):
            raise SchemaValidationError("query_type 必须是 QueryType。")
        if self.needs_retrieval and not self.target_paper_keys:
            raise SchemaValidationError("论文检索 QueryPlan 必须指定 target_paper_keys。")
        if len(set(self.target_paper_keys)) != len(self.target_paper_keys):
            raise SchemaValidationError("target_paper_keys 不能重复。")
        if any(not isinstance(key, str) or not key.strip() for key in self.target_paper_keys):
            raise SchemaValidationError("target_paper_keys 必须由非空字符串组成。")
        if not isinstance(self.retrieval_policy, str) or not self.retrieval_policy.strip():
            raise SchemaValidationError("retrieval_policy 不能为空。")
        for value, label in ((self.dense_limit, "dense_limit"), (self.sparse_limit, "sparse_limit")):
            if not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"{label} 必须是非负整数。")
        if self.needs_retrieval and self.dense_limit + self.sparse_limit < 1:
            raise SchemaValidationError("启用检索时 dense_limit 与 sparse_limit 至少一个为正数。")
        if not isinstance(self.rerank_enabled, bool) or not isinstance(self.allow_one_retry, bool):
            raise SchemaValidationError("rerank_enabled 和 allow_one_retry 必须是布尔值。")
        if not isinstance(self.minimum_evidence_per_paper, int) or self.minimum_evidence_per_paper < 0:
            raise SchemaValidationError("minimum_evidence_per_paper 必须是非负整数。")
        if self.coverage_gate_required and self.minimum_evidence_per_paper < 1:
            raise SchemaValidationError("Coverage Gate 至少需要每篇一条证据。")
        if not self.block_types or set(self.block_types) - _BLOCK_TYPES:
            raise SchemaValidationError(f"block_types 必须非空且属于 {sorted(_BLOCK_TYPES)}。")
        if not isinstance(self.router_version, str) or not self.router_version.strip():
            raise SchemaValidationError("router_version 不能为空。")
        if not isinstance(self.experiment_mode, P2RunMode):
            raise SchemaValidationError("experiment_mode 必须是 P2RunMode。")
        if not isinstance(self.query_type_source, QueryTypeSource):
            raise SchemaValidationError("query_type_source 必须是 QueryTypeSource。")
        if not isinstance(self.scope_source, ScopeSource):
            raise SchemaValidationError("scope_source 必须是 ScopeSource。")
        if self.experiment_mode == P2RunMode.ORACLE and (
            self.query_type_source != QueryTypeSource.ORACLE_EVAL or self.scope_source != ScopeSource.ORACLE_EVAL
        ):
            raise SchemaValidationError("Oracle 计划必须显式记录 oracle_eval 的类型和 Scope 来源。")
        if self.experiment_mode == P2RunMode.END_TO_END and (
            self.query_type_source == QueryTypeSource.ORACLE_EVAL or self.scope_source == ScopeSource.ORACLE_EVAL
        ):
            raise SchemaValidationError("端到端计划不能使用 Oracle 类型或论文 Scope。")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["query_type"] = self.query_type.value
        result["target_paper_keys"] = list(self.target_paper_keys)
        result["block_types"] = list(self.block_types)
        result["experiment_mode"] = self.experiment_mode.value
        result["query_type_source"] = self.query_type_source.value
        result["scope_source"] = self.scope_source.value
        return result


class P2RulesRouter:
    """第一版规则路由器；规则可审计，后续模型路由只能作为独立实验变量。"""

    def plan(
        self,
        *,
        query_type: QueryType,
        target_paper_keys: Iterable[str],
        experiment_mode: P2RunMode = P2RunMode.STATIC,
        query_type_source: QueryTypeSource = QueryTypeSource.STATIC_CONFIG,
        scope_source: ScopeSource = ScopeSource.SESSION_DEFAULT,
    ) -> QueryPlan:
        papers = tuple(dict.fromkeys(key.strip() for key in target_paper_keys if key.strip()))
        if query_type == QueryType.GENERAL_CHAT:
            return QueryPlan(
                query_type=query_type,
                target_paper_keys=(),
                retrieval_policy="skip_paper_retrieval",
                dense_limit=0,
                sparse_limit=0,
                rerank_enabled=False,
                block_types=("paragraph",),
                minimum_evidence_per_paper=0,
                allow_one_retry=False,
                coverage_gate_required=False,
                needs_retrieval=False,
                experiment_mode=experiment_mode,
                query_type_source=query_type_source,
                scope_source=scope_source,
            )

        policies: dict[QueryType, tuple[str, int, int, bool, tuple[str, ...], bool]] = {
            QueryType.METADATA: ("metadata_first", 4, 3, False, ("metadata", "heading", "paragraph"), False),
            QueryType.KEYWORD_EXACT: ("lexical_priority", 6, 12, True, ("heading", "paragraph", "table"), True),
            QueryType.LOCAL_FACT: ("local_evidence", 10, 10, True, ("paragraph", "table", "caption"), True),
            QueryType.SEMANTIC_EXPLANATION: (
                "parent_child_context",
                12,
                8,
                True,
                ("heading", "paragraph", "formula"),
                True,
            ),
            QueryType.TABLE_FIGURE: ("table_caption_first", 8, 10, True, ("table", "caption", "paragraph"), True),
            QueryType.MULTI_SECTION: ("decompose_then_merge", 14, 12, True, ("heading", "paragraph", "table"), True),
            QueryType.CROSS_DOCUMENT: (
                "balanced_cross_document",
                12,
                12,
                True,
                ("heading", "paragraph", "table", "caption"),
                True,
            ),
            QueryType.UNANSWERABLE: (
                "evidence_constrained_refusal",
                12,
                12,
                True,
                ("heading", "paragraph", "table", "caption"),
                True,
            ),
        }
        policy, dense, sparse, rerank, blocks, retry = policies[query_type]
        cross_document = query_type in {QueryType.CROSS_DOCUMENT, QueryType.UNANSWERABLE}
        return QueryPlan(
            query_type=query_type,
            target_paper_keys=papers,
            retrieval_policy=policy,
            dense_limit=dense,
            sparse_limit=sparse,
            rerank_enabled=rerank,
            block_types=blocks,
            minimum_evidence_per_paper=1 if cross_document else 0,
            allow_one_retry=retry,
            coverage_gate_required=cross_document,
            needs_retrieval=True,
            experiment_mode=experiment_mode,
            query_type_source=query_type_source,
            scope_source=scope_source,
        )
