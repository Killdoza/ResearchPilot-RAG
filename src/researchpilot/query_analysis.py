"""P2-B1 的确定性 Query Analyzer。

本模块只读取用户问题、会话可见论文目录和用户显式选择；它不知道
评测集的 Gold 类型、标准答案或证据页码。规则故意保持保守：无法确定
论文范围时返回 ``scope_insufficient``，而不是猜测某一篇论文。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .query_plan import QueryTypeSource
from .schemas import QueryType, SchemaValidationError
from .scope_resolution import PaperScopeCatalog, ScopeResolution

_COMPARISON_RE = re.compile(
    r"比较|对比|区别|差异|不同|谁更好|哪个更好|分别|优缺点|vs\.?|versus|compare|comparison|difference",
    re.IGNORECASE,
)
_METADATA_RE = re.compile(
    r"作者|作者列表|年份|发表.*(会议|期刊)|会议|期刊|arxiv|doi|author|venue|year",
    re.IGNORECASE,
)
_TABLE_FIGURE_RE = re.compile(r"表\s*\d|图\s*\d|表格|图注|figure|table|caption", re.IGNORECASE)
_KEYWORD_EXACT_RE = re.compile(
    r"全称|缩写|叫什么|名称|模型名|术语|公式|定义|acronym|abbreviation|called|name of",
    re.IGNORECASE,
)
_LOCAL_FACT_RE = re.compile(
    r"学习率|批大小|batch\s*size|batch|数据集|样本量|训练步数|超参数|指标|baseline|基线|"
    r"实验设置|设置为多少|多少个|多少次|learning rate|dataset|metric|hyperparameter",
    re.IGNORECASE,
)
_MULTI_SECTION_RE = re.compile(
    r"(?:方法|模型|架构).*(?:实验|结果)|(?:实验|结果).*(?:方法|模型|架构)|为什么.*(?:如何|怎么)|如何.*(?:影响|导致)",
    re.IGNORECASE,
)
_UNANSWERABLE_RISK_RE = re.compile(
    r"有没有提到|是否提到|是否报告|有没有报告|未提及|没有给出|没有报告|not mention|not report",
    re.IGNORECASE,
)
_GENERAL_CHAT_RE = re.compile(
    r"^\s*(你好|嗨|你是谁|你能做什么|hello|hi|what can you do)[？?！!。.]?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryAnalysis:
    """不含 Gold 的规则分析结果，可直接审计并写入后续 Trace。"""

    query_type: QueryType | None
    scope: ScopeResolution
    needs_retrieval: bool
    unanswerable_risk: bool
    signals: tuple[str, ...]
    fallback_reason: str | None = None
    query_type_source: QueryTypeSource = QueryTypeSource.RULE_PREDICTED

    def __post_init__(self) -> None:
        if self.query_type is not None and not isinstance(self.query_type, QueryType):
            raise SchemaValidationError("query_type 必须是 QueryType 或 None。")
        if not isinstance(self.scope, ScopeResolution):
            raise SchemaValidationError("scope 必须是 ScopeResolution。")
        if not isinstance(self.needs_retrieval, bool) or not isinstance(self.unanswerable_risk, bool):
            raise SchemaValidationError("needs_retrieval 和 unanswerable_risk 必须为布尔值。")
        if not isinstance(self.query_type_source, QueryTypeSource):
            raise SchemaValidationError("query_type_source 必须是 QueryTypeSource。")
        if self.query_type is None:
            if self.needs_retrieval:
                raise SchemaValidationError("没有 query_type 时不能请求论文检索。")
            if not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip():
                raise SchemaValidationError("没有 query_type 时必须记录 fallback_reason。")
        elif self.fallback_reason is not None:
            raise SchemaValidationError("已有 query_type 时不能同时记录 fallback_reason。")
        if self.query_type == QueryType.GENERAL_CHAT and self.needs_retrieval:
            raise SchemaValidationError("general_chat 不应触发论文检索。")


class RuleQueryAnalyzer:
    """可解释的 ``p2_rules_v1`` 查询分析器，不调用 LLM。"""

    def __init__(self, *, catalog: PaperScopeCatalog) -> None:
        self._catalog = catalog

    def analyze(
        self,
        *,
        question: str,
        session_available_paper_keys: Iterable[str],
        user_selected_paper_keys: Iterable[str] = (),
    ) -> QueryAnalysis:
        if not isinstance(question, str) or not question.strip():
            raise SchemaValidationError("question 不能为空。")
        if _GENERAL_CHAT_RE.fullmatch(question):
            # General chat deliberately skips scope resolution: a conversation
            # greeting must not be reinterpreted as a paper answer.
            from .query_plan import ScopeSource
            from .scope_resolution import ScopeResolution

            return QueryAnalysis(
                query_type=QueryType.GENERAL_CHAT,
                scope=ScopeResolution((), ScopeSource.NONE, "general_chat:no_paper_scope"),
                needs_retrieval=False,
                unanswerable_risk=False,
                signals=("general_chat",),
            )

        scope = self._catalog.resolve(
            question=question,
            available_paper_keys=session_available_paper_keys,
            user_selected_paper_keys=user_selected_paper_keys,
        )
        if not scope.paper_keys:
            return QueryAnalysis(
                query_type=None,
                scope=scope,
                needs_retrieval=False,
                unanswerable_risk=bool(_UNANSWERABLE_RISK_RE.search(question)),
                signals=("scope_insufficient",),
                fallback_reason="scope_insufficient",
            )

        # A comparison question with only one identified paper is unsafe.  It
        # must not silently compare that paper against whatever happens to rank
        # next in a global index.  Retain the resolved single-paper scope in the
        # trace, but require the caller to select or name the second paper.
        if _COMPARISON_RE.search(question) and len(scope.paper_keys) < 2:
            return QueryAnalysis(
                query_type=None,
                scope=scope,
                needs_retrieval=False,
                unanswerable_risk=False,
                signals=("comparison_signal", "scope_insufficient"),
                fallback_reason="scope_insufficient:comparison_requires_two_papers",
            )

        query_type, signals = self._classify(question=question, resolved_paper_count=len(scope.paper_keys))
        return QueryAnalysis(
            query_type=query_type,
            scope=scope,
            needs_retrieval=True,
            unanswerable_risk=bool(_UNANSWERABLE_RISK_RE.search(question)),
            signals=signals,
        )

    @staticmethod
    def _classify(*, question: str, resolved_paper_count: int) -> tuple[QueryType, tuple[str, ...]]:
        # Scope and explicit comparison language have priority: an otherwise
        # metadata-like question can still demand evidence from two papers.
        if resolved_paper_count >= 2 or _COMPARISON_RE.search(question):
            return QueryType.CROSS_DOCUMENT, ("cross_document",)
        if _METADATA_RE.search(question):
            return QueryType.METADATA, ("metadata",)
        if _TABLE_FIGURE_RE.search(question):
            return QueryType.TABLE_FIGURE, ("table_figure",)
        if _KEYWORD_EXACT_RE.search(question):
            return QueryType.KEYWORD_EXACT, ("keyword_exact",)
        if _LOCAL_FACT_RE.search(question):
            return QueryType.LOCAL_FACT, ("local_fact",)
        if _MULTI_SECTION_RE.search(question):
            return QueryType.MULTI_SECTION, ("multi_section",)
        return QueryType.SEMANTIC_EXPLANATION, ("semantic_explanation",)
