"""统一产品编排器测试，不调用真实模型或向量库。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.orchestration import RuntimeOrchestrator  # noqa: E402
from researchpilot.routing import RuntimePaper, RuntimeRequest, RuntimeRoute  # noqa: E402
from researchpilot.schemas import Answer, QueryType  # noqa: E402


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def answer_with_retrieval(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        answer = Answer(
            text="有证据的正文回答",
            citations=(),
            refused=False,
            trace_id="trace_detail",
            latency_ms=10.0,
        )
        retrieval = SimpleNamespace(trace=SimpleNamespace(candidate_count=1))
        return SimpleNamespace(answer=answer, retrieval=retrieval)


class _FakeAnnotationService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def capture(self, **kwargs: object) -> int:
        self.calls.append(kwargs)
        return 1


class _ForbiddenMemoryService:
    def generation_context(self, **kwargs: object) -> tuple[str, str]:
        del kwargs
        raise AssertionError("factual generation must not load memory as evidence")


class RuntimeOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _FakeService()
        self.orchestrator = RuntimeOrchestrator(service=self.service)
        self.paper = RuntimePaper("paper_a", "UICompass.pdf", page_count=12, parser_name="docling")

    def _request(self, question: str, *, selected: tuple[str, ...] = ()) -> RuntimeRequest:
        return RuntimeRequest(
            question=question,
            user_id="user_a",
            session_id="session_a",
            available_papers=(self.paper,),
            selected_paper_ids=selected,
        )

    def test_detail_delegates_to_service_with_single_paper_scope(self) -> None:
        result = self.orchestrator.handle(self._request("核心方法是什么？", selected=("paper_a",)))

        self.assertEqual(result.decision.route, RuntimeRoute.DOCUMENT_DETAIL)
        self.assertFalse(result.answer.refused)
        self.assertEqual(self.service.calls[0]["paper_id"], "paper_a")
        self.assertEqual(self.service.calls[0]["query_type"], QueryType.SEMANTIC_EXPLANATION)

    def test_metadata_uses_traceable_front_matter_retrieval(self) -> None:
        result = self.orchestrator.handle(self._request("作者和发表会议是什么？", selected=("paper_a",)))

        self.assertEqual(result.decision.route, RuntimeRoute.METADATA)
        self.assertFalse(result.answer.refused)
        self.assertEqual(self.service.calls[0]["paper_id"], "paper_a")
        self.assertEqual(self.service.calls[0]["query_type"], QueryType.METADATA)

    def test_summary_and_experimental_setup_keep_their_task_types(self) -> None:
        summary = self.orchestrator.handle(self._request("请总结这篇论文", selected=("paper_a",)))
        experiment = self.orchestrator.handle(self._request("论文的实验设置是什么？", selected=("paper_a",)))

        self.assertEqual(summary.decision.route, RuntimeRoute.DOCUMENT_SUMMARY)
        self.assertEqual(self.service.calls[0]["query_type"], QueryType.DOCUMENT_SUMMARY)
        self.assertEqual(experiment.decision.route, RuntimeRoute.EXPERIMENTAL_SETUP)
        self.assertEqual(self.service.calls[1]["query_type"], QueryType.EXPERIMENTAL_SETUP)

    def test_scope_insufficient_is_structured_refusal(self) -> None:
        request = RuntimeRequest(question="这篇论文的方法是什么？", user_id="user_a", session_id="session_a")
        result = self.orchestrator.handle(request)

        self.assertEqual(result.decision.route, RuntimeRoute.SCOPE_INSUFFICIENT)
        self.assertTrue(result.answer.refused)
        self.assertEqual(self.service.calls, [])

    def test_cross_document_never_silently_degrades(self) -> None:
        second_paper = RuntimePaper("paper_b", "RAG.pdf")
        request = RuntimeRequest(
            question="比较两篇论文的结果",
            user_id="user_a",
            session_id="session_a",
            available_papers=(self.paper, second_paper),
        )
        result = self.orchestrator.handle(request)

        self.assertEqual(result.decision.route, RuntimeRoute.CROSS_DOCUMENT)
        self.assertTrue(result.answer.refused)
        self.assertEqual(result.answer.refusal_reason, "strict_crossdoc_not_available")
        self.assertEqual(self.service.calls, [])

    def test_explicit_user_annotation_is_captured_by_orchestrator(self) -> None:
        annotations = _FakeAnnotationService()
        orchestrator = RuntimeOrchestrator(
            service=self.service,
            annotation_service=annotations,
        )
        request = RuntimeRequest(
            question="我的批注：这里的比较可能有问题。",
            user_id="user_a",
            session_id="session_a",
            available_papers=(self.paper,),
            selected_paper_ids=("paper_a",),
            last_evidence_chunk_ids=("chunk_previous",),
        )

        result = orchestrator.handle(request)

        self.assertTrue(result.annotation_saved)
        self.assertEqual(annotations.calls[0]["paper_id"], "paper_a")
        self.assertEqual(annotations.calls[0]["chunk_id"], "chunk_previous")

    def test_factual_generation_context_excludes_wm_ltm_and_chat_history(self) -> None:
        orchestrator = RuntimeOrchestrator(
            service=self.service,
            memory_service=_ForbiddenMemoryService(),  # type: ignore[arg-type]
        )
        request = RuntimeRequest(
            question="核心方法是什么？",
            user_id="user_a",
            session_id="session_a",
            available_papers=(self.paper,),
            selected_paper_ids=("paper_a",),
            recent_history="AI: 一个未经本轮证据验证的旧结论",
        )

        orchestrator.handle(request)

        context = self.service.calls[0]["context"]
        self.assertEqual(context.long_term_memory, "")
        self.assertEqual(context.working_memory, "")
        self.assertEqual(context.chat_history, "")


if __name__ == "__main__":
    unittest.main()
