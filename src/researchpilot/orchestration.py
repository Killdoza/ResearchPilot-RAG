"""产品运行时的统一编排入口。

它将纯规则路由与现有 ``ResearchPilotService`` 连接起来。首轮只开放经过
产品验证的单论文/会话级问答；跨论文严格比较仍显式拒绝，绝不回退到无范围的
全局 Top-K。该模块不接触 Streamlit 状态、FastAPI 对象或评测 Gold。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from .annotation_service import AnnotationMode, AnnotationService, infer_annotation_relation
from .conversation_service import ConversationService
from .generation import GenerationContext
from .memory_service import ScopedMemoryService
from .retrieval import RetrievalResult
from .routing import RuntimeRequest, RuntimeRoute, RuntimeRouteDecision, RuntimeRouter
from .schemas import Answer
from .services import AnswerExecution


class AnsweringService(Protocol):
    """避免编排层绑定具体向量库或模型实现。"""

    def answer_with_retrieval(
        self,
        *,
        question: str,
        user_id: str,
        session_id: str,
        query_type: object | None = None,
        recent_history: str = "",
        paper_id: str | None = None,
        search_query: str | None = None,
        context: GenerationContext | None = None,
        on_text_delta: Callable[[str], None] | None = None,
        annotation_mode: AnnotationMode = "paper_only",
    ) -> AnswerExecution:
        """返回带检索轨迹的可引用回答。"""


@dataclass(frozen=True)
class RuntimeResult:
    """Streamlit/API 共享的编排输出。"""

    decision: RuntimeRouteDecision
    answer: Answer
    retrieval: RetrievalResult | None = None
    annotation_saved: bool = False


def _local_answer(*, text: str, refused: bool, refusal_reason: str | None = None) -> Answer:
    return Answer(
        text=text,
        citations=(),
        refused=refused,
        refusal_reason=refusal_reason,
        trace_id=f"runtime_{uuid.uuid4().hex}",
        latency_ms=0.0,
    )


class RuntimeOrchestrator:
    """产品默认编排器：路由、范围保护和高精度论文问答。"""

    def __init__(
        self,
        *,
        service: AnsweringService,
        router: RuntimeRouter | None = None,
        strict_crossdoc_enabled: bool = False,
        conversation_service: ConversationService | None = None,
        memory_service: ScopedMemoryService | None = None,
        annotation_service: AnnotationService | None = None,
        annotation_mode: AnnotationMode = "paper_only",
    ) -> None:
        self._service = service
        self._router = router or RuntimeRouter()
        self._strict_crossdoc_enabled = strict_crossdoc_enabled
        self._conversation_service = conversation_service
        self._memory_service = memory_service
        self._annotation_service = annotation_service
        self._annotation_mode = annotation_mode

    def handle(
        self,
        request: RuntimeRequest,
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> RuntimeResult:
        decision = self._router.route(request)
        if decision.route == RuntimeRoute.SCOPE_INSUFFICIENT:
            return RuntimeResult(
                decision,
                _local_answer(
                    text="当前问题需要论文证据，但没有足够的论文范围。请先上传论文，或在“回答范围”中选择目标论文。",
                    refused=True,
                    refusal_reason=decision.reason,
                ),
            )
        if decision.route == RuntimeRoute.GENERAL_CHAT:
            if self._conversation_service is not None:
                text = self._conversation_service.general_chat(
                    question=request.question,
                    recent_history=request.recent_history,
                    on_text_delta=on_text_delta,
                )
            else:
                text = (
                    "我是 ResearchPilot，专注于 AI/CS 论文的可追溯阅读与问答。"
                    "上传或选择论文后，我可以基于页码证据回答方法、实验和结论问题。"
                )
            return RuntimeResult(
                decision,
                _local_answer(
                    text=text,
                    refused=False,
                ),
            )
        if decision.route == RuntimeRoute.CHAT_SUMMARY:
            if not request.recent_history.strip():
                return RuntimeResult(
                    decision,
                    _local_answer(
                        text="当前没有可供总结的近期对话记录。",
                        refused=True,
                        refusal_reason="chat_history_unavailable",
                    ),
                )
            text = (
                self._conversation_service.summarize_chat(recent_history=request.recent_history)
                if self._conversation_service is not None
                else "API 的会话总结需要由调用方传入完整近期对话；Streamlit 页面会使用当前会话历史生成总结。"
            )
            return RuntimeResult(decision, _local_answer(text=text, refused=False))
        if decision.route == RuntimeRoute.CROSS_DOCUMENT:
            reason = (
                "strict_crossdoc_not_available"
                if not self._strict_crossdoc_enabled
                else "strict_crossdoc_adapter_not_configured"
            )
            return RuntimeResult(
                decision,
                _local_answer(
                    text="严格跨论文比较尚未在产品链路开放；系统不会把多篇论文退化为全局 Top-K 比较。",
                    refused=True,
                    refusal_reason=reason,
                ),
            )

        annotation_saved = False
        relation = infer_annotation_relation(request.question)
        if (
            relation is not None
            and self._annotation_service is not None
            and len(decision.target_paper_ids) == 1
            and request.last_evidence_chunk_ids
        ):
            anchor = request.last_evidence_anchors[0] if request.last_evidence_anchors else None
            self._annotation_service.capture(
                user_id=request.user_id,
                session_id=request.session_id,
                paper_id=decision.target_paper_ids[0],
                chunk_id=request.last_evidence_chunk_ids[0],
                content=request.question,
                relation=relation,
                note_type="correction" if relation == "contradicts" else "comment",
                anchor_page=anchor.page if anchor is not None else None,
                section_path=anchor.section_path if anchor is not None else (),
            )
            annotation_saved = True

        search_query = request.question
        if self._conversation_service is not None:
            search_query = self._conversation_service.rewrite_query(
                question=request.question,
                recent_history=request.recent_history,
            )
        # Recent history/WM/LTM may resolve pronouns before retrieval, but they
        # are not paper evidence and must not enter factual answer generation.
        # ResearchPilotService may still inject explicitly labelled user notes.
        context = GenerationContext()
        execution = self._service.answer_with_retrieval(
            question=request.question,
            user_id=request.user_id,
            session_id=request.session_id,
            query_type=decision.query_type,
            recent_history=request.recent_history,
            search_query=search_query,
            context=context,
            paper_id=(decision.target_paper_ids[0] if len(decision.target_paper_ids) == 1 else None),
            on_text_delta=on_text_delta,
            annotation_mode=self._annotation_mode,
        )
        answer = execution.answer
        if self._conversation_service is not None:
            usage = self._conversation_service.last_rewrite_usage
            observed = usage.token_observations > 0
            answer = replace(
                answer,
                rewrite_model_calls=usage.model_calls,
                prompt_tokens=((answer.prompt_tokens or 0) + usage.prompt_tokens if observed else answer.prompt_tokens),
                completion_tokens=(
                    (answer.completion_tokens or 0) + usage.completion_tokens if observed else answer.completion_tokens
                ),
                estimated_cost=(
                    (answer.estimated_cost or 0.0) + self._conversation_service.last_rewrite_cost
                    if self._conversation_service.last_rewrite_cost is not None
                    else answer.estimated_cost
                ),
            )
        return RuntimeResult(
            decision,
            answer,
            execution.retrieval,
            annotation_saved=annotation_saved,
        )
