"""产品运行时的可解释路由器。

该模块只读取用户问题、当前会话可见论文与用户主动选择的范围；它不读取
评测 Gold、标准答案或证据页码，也不调用 LLM。P2 的 ``QueryPlan`` 保持用于
离线实验；这里的路由器负责产品入口共用的任务级分流。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .schemas import QueryType, SchemaValidationError


class RuntimeRoute(str, Enum):
    """产品入口可展示、可审计的高层任务类型。"""

    METADATA = "metadata"
    DOCUMENT_SUMMARY = "document_summary"
    EXPERIMENTAL_SETUP = "experimental_setup"
    EXPERIMENTAL_RESULTS = "experimental_results"
    DOCUMENT_DETAIL = "document_detail"
    MULTI_SECTION = "multi_section"
    CROSS_DOCUMENT = "cross_document"
    CHAT_SUMMARY = "chat_summary"
    GENERAL_CHAT = "general_chat"
    SCOPE_INSUFFICIENT = "scope_insufficient"


_GENERAL_CHAT_RE = re.compile(
    r"^\s*(你好|嗨|你是谁|你能做什么|hello|hi|what can you do)[？?！!。.]?\s*$",
    re.IGNORECASE,
)
_CHAT_SUMMARY_RE = re.compile(r"总结.*(我们|对话|讨论)|回顾.*(我们|对话|讨论)|聊了什么", re.IGNORECASE)
_METADATA_RE = re.compile(
    r"作者|作者列表|年份|发表.*(会议|期刊)|会议|期刊|arxiv|doi|author|venue|year",
    re.IGNORECASE,
)
_DOCUMENT_SUMMARY_RE = re.compile(
    r"(整篇|全文|这篇|本文|论文).*(总结|概括|归纳|摘要|讲了什么|介绍一下|贡献|主要内容)|"
    r"(总结|概述|概括|归纳|介绍).*(论文|文章|这篇|本文)|核心贡献|主要贡献",
    re.IGNORECASE,
)
_EXPERIMENTAL_SETUP_RE = re.compile(
    r"实验设置|实验配置|数据集|基线|评估指标|评价指标|训练配置|实现细节|超参数|"
    r"学习率|批大小|样本量|experimental\s+setup|experiment\s+settings?|dataset|"
    r"baseline|evaluation\s+metrics?|implementation\s+details?|hyperparameters?|"
    r"learning\s+rate|batch\s+size",
    re.IGNORECASE,
)
_EXPERIMENTAL_RESULTS_RE = re.compile(
    r"实验结果|主要结果|评估结果|性能结果|消融(?:实验|结果)?|结果如何|"
    r"experimental\s+results?|evaluation\s+results?|main\s+results?|"
    r"performance\s+(?:results?|comparison)|ablation(?:\s+stud(?:y|ies))?|"
    r"\u5b9a\u91cf\u7ed3\u679c|\u7ed3\u679c\u8868|\u6210\u672c.*\u7ed3\u679c|"
    r"\u6548\u7387.*\u7ed3\u679c|\u80dc\u7387",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"比较|对比|区别|差异|不同|谁更好|哪个更好|优缺点|vs\.?|versus|compare|comparison|difference",
    re.IGNORECASE,
)
_EXPLICIT_CROSS_PAPER_RE = re.compile(
    r"\u4e24\u7bc7(?:\u8bba\u6587|\u6587\u7ae0)|\u8fd9\u4e24\u7bc7|\u591a\u7bc7(?:\u8bba\u6587|\u6587\u7ae0)|"
    r"\u8fd9\u4e9b\u8bba\u6587|\u8de8\u8bba\u6587|cross[-\s]?document",
    re.IGNORECASE,
)
_MULTI_SECTION_RE = re.compile(
    r"(?:方法|设计|模型|架构|索引|检索).*(?:实验|结果|优势|研究问题|动机|挑战).*(?:联系|对应|关系)|"
    r"(?:实验|结果|优势|研究问题|动机|挑战).*(?:方法|设计|模型|架构|索引|检索).*(?:联系|对应|关系)|"
    r"(?:方法|设计|模型|架构|索引|检索).*(?:联系|对应|关系).*(?:实验|结果|优势|研究问题|动机|挑战)|"
    r"为什么.*(?:如何|怎么)|如何.*(?:影响|导致|对应)|"
    r"(?:联系|对应|关系).*(?:方法|设计|研究问题|实验|结果)",
    re.IGNORECASE,
)


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} 不能为空。")


@dataclass(frozen=True)
class RuntimePaper:
    """产品运行时能看到的论文目录项，不包含评测标注。"""

    paper_id: str
    source_name: str
    page_count: int | None = None
    parser_name: str | None = None
    chunker_version: str | None = None
    extraction_quality: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.paper_id, "paper_id")
        _require_text(self.source_name, "source_name")
        if self.page_count is not None and (not isinstance(self.page_count, int) or self.page_count < 1):
            raise SchemaValidationError("page_count 必须是正整数或 None。")


@dataclass(frozen=True)
class RuntimeEvidenceAnchor:
    """The last visible citation anchor; it contains no evaluator Gold."""

    chunk_id: str
    page: int | None = None
    section_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "chunk_id")
        if self.page is not None and self.page < 1:
            raise SchemaValidationError("page 必须是正整数或 None。")
        if any(not isinstance(item, str) or not item.strip() for item in self.section_path):
            raise SchemaValidationError("section_path 必须由非空字符串组成。")


@dataclass(frozen=True)
class RuntimeRequest:
    """产品 Planner 的唯一输入边界；禁止携带任何评测 Gold。"""

    question: str
    user_id: str
    session_id: str
    available_papers: tuple[RuntimePaper, ...] = ()
    selected_paper_ids: tuple[str, ...] = ()
    recent_history: str = ""
    product_mode: str = "high_precision"
    last_evidence_chunk_ids: tuple[str, ...] = ()
    last_evidence_anchors: tuple[RuntimeEvidenceAnchor, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.question, "question")
        _require_text(self.user_id, "user_id")
        _require_text(self.session_id, "session_id")
        _require_text(self.product_mode, "product_mode")
        if any(not isinstance(paper, RuntimePaper) for paper in self.available_papers):
            raise SchemaValidationError("available_papers 必须由 RuntimePaper 组成。")
        paper_ids = tuple(paper.paper_id for paper in self.available_papers)
        if len(set(paper_ids)) != len(paper_ids):
            raise SchemaValidationError("available_papers 不能包含重复 paper_id。")
        if len(set(self.selected_paper_ids)) != len(self.selected_paper_ids):
            raise SchemaValidationError("selected_paper_ids 不能重复。")
        if any(not isinstance(item, str) or not item.strip() for item in self.selected_paper_ids):
            raise SchemaValidationError("selected_paper_ids 必须由非空字符串组成。")
        if not set(self.selected_paper_ids).issubset(paper_ids):
            raise SchemaValidationError("selected_paper_ids 必须属于 available_papers。")
        if any(not isinstance(item, str) or not item.strip() for item in self.last_evidence_chunk_ids):
            raise SchemaValidationError("last_evidence_chunk_ids 必须由非空字符串组成。")
        if any(not isinstance(item, RuntimeEvidenceAnchor) for item in self.last_evidence_anchors):
            raise SchemaValidationError("last_evidence_anchors 必须由 RuntimeEvidenceAnchor 组成。")


@dataclass(frozen=True)
class RuntimeRouteDecision:
    """路由结果与其真实 Scope；可直接写入 Trace 或 UI 技术详情。"""

    route: RuntimeRoute
    target_paper_ids: tuple[str, ...]
    query_type: QueryType | None
    needs_retrieval: bool
    reason: str | None = None
    router_version: str = "product_rules_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.route, RuntimeRoute):
            raise SchemaValidationError("route 必须是 RuntimeRoute。")
        if len(set(self.target_paper_ids)) != len(self.target_paper_ids):
            raise SchemaValidationError("target_paper_ids 不能重复。")
        if any(not isinstance(item, str) or not item.strip() for item in self.target_paper_ids):
            raise SchemaValidationError("target_paper_ids 必须由非空字符串组成。")
        if self.query_type is not None and not isinstance(self.query_type, QueryType):
            raise SchemaValidationError("query_type 必须是 QueryType 或 None。")
        if not isinstance(self.needs_retrieval, bool):
            raise SchemaValidationError("needs_retrieval 必须是布尔值。")
        if self.route == RuntimeRoute.SCOPE_INSUFFICIENT:
            if self.needs_retrieval or not isinstance(self.reason, str) or not self.reason.strip():
                raise SchemaValidationError("scope_insufficient 必须包含拒绝原因且不能检索。")
        elif self.reason is not None:
            raise SchemaValidationError("非 scope_insufficient 路由不应写入 reason。")


class RuntimeRouter:
    """不调用模型的产品规则路由器，供 Streamlit 与 API 共用。"""

    def route(self, request: RuntimeRequest) -> RuntimeRouteDecision:
        question = request.question.strip()
        if _GENERAL_CHAT_RE.fullmatch(question):
            return RuntimeRouteDecision(RuntimeRoute.GENERAL_CHAT, (), None, False)
        if _CHAT_SUMMARY_RE.search(question):
            return RuntimeRouteDecision(RuntimeRoute.CHAT_SUMMARY, (), None, False)

        available_ids = tuple(paper.paper_id for paper in request.available_papers)
        target_ids = request.selected_paper_ids or available_ids
        if not target_ids:
            return RuntimeRouteDecision(
                RuntimeRoute.SCOPE_INSUFFICIENT,
                (),
                None,
                False,
                "scope_insufficient:no_active_paper",
            )
        comparison_requested = bool(_COMPARISON_RE.search(question))
        explicit_cross_paper = bool(_EXPLICIT_CROSS_PAPER_RE.search(question))
        if comparison_requested and (len(target_ids) >= 2 or explicit_cross_paper):
            if len(target_ids) < 2:
                return RuntimeRouteDecision(
                    RuntimeRoute.SCOPE_INSUFFICIENT,
                    target_ids,
                    None,
                    False,
                    "scope_insufficient:comparison_requires_two_papers",
                )
            return RuntimeRouteDecision(
                RuntimeRoute.CROSS_DOCUMENT,
                target_ids,
                QueryType.CROSS_DOCUMENT,
                False,
            )
        if _METADATA_RE.search(question):
            return RuntimeRouteDecision(RuntimeRoute.METADATA, target_ids, QueryType.METADATA, True)
        if _DOCUMENT_SUMMARY_RE.search(question):
            return RuntimeRouteDecision(
                RuntimeRoute.DOCUMENT_SUMMARY,
                target_ids,
                QueryType.DOCUMENT_SUMMARY,
                True,
            )
        if _MULTI_SECTION_RE.search(question):
            return RuntimeRouteDecision(
                RuntimeRoute.MULTI_SECTION,
                target_ids,
                QueryType.MULTI_SECTION,
                True,
            )
        if _EXPERIMENTAL_RESULTS_RE.search(question):
            return RuntimeRouteDecision(
                RuntimeRoute.EXPERIMENTAL_RESULTS,
                target_ids,
                QueryType.EXPERIMENTAL_RESULTS,
                True,
            )
        if _EXPERIMENTAL_SETUP_RE.search(question):
            return RuntimeRouteDecision(
                RuntimeRoute.EXPERIMENTAL_SETUP,
                target_ids,
                QueryType.EXPERIMENTAL_SETUP,
                True,
            )
        return RuntimeRouteDecision(
            RuntimeRoute.DOCUMENT_DETAIL,
            target_ids,
            QueryType.SEMANTIC_EXPLANATION,
            True,
        )
