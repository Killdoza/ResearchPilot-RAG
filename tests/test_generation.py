"""引用约束的回答生成测试，不调用真实大模型。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.generation import (  # noqa: E402
    AnswerGenerator,
    GenerationContext,
    LangChainTextGenerationClient,
    ModelServiceUnavailable,
)
from researchpilot.retrieval import RetrievalResult  # noqa: E402
from researchpilot.schemas import (  # noqa: E402
    BlockType,
    Chunk,
    QueryType,
    RetrievalTrace,
    ScoredChunk,
)


class _FakeGenerationClient:
    def __init__(self, response: str = "该方法使用页码级证据 [证据 1]。") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


class _FailingChatModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: object) -> object:
        del messages
        self.calls += 1
        raise ConnectionError("temporary network error")


class _RecoveringChatModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: object) -> object:
        del messages
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary network error")
        return type(
            "Response",
            (),
            {
                "content": "恢复成功",
                "usage_metadata": {"input_tokens": 12, "output_tokens": 3},
            },
        )()


class _RecoveringStreamingChatModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, messages: object):
        del messages
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary network error")
        yield type("Chunk", (), {"content": "恢复"})()
        yield type("Chunk", (), {"content": "成功"})()


class _PartialThenFailStreamingChatModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, messages: object):
        del messages
        self.calls += 1
        yield type("Chunk", (), {"content": "部分内容"})()
        raise ConnectionError("connection lost after first token")


class _StreamingChatModel:
    def stream(self, messages: object):
        del messages
        for content in ("第一段", "，第二段"):
            yield type("Chunk", (), {"content": content})()


def _retrieval_result(*, with_evidence: bool, query_type: QueryType | None = None) -> RetrievalResult:
    if not with_evidence:
        trace = RetrievalTrace(
            trace_id="trace_empty",
            query="方法是什么？",
            search_query="方法是什么？",
            query_type=None,
            candidate_count=0,
            results=(),
            elapsed_ms=1.0,
            stages=("dense_child_retrieval",),
        )
        return RetrievalResult(results=(), trace=trace)
    chunk = Chunk.create(
        paper_id="paper_demo",
        parent_id="block_paper_demo_p0002",
        page_start=2,
        page_end=2,
        section_path=("3 方法",),
        content="论文的方法使用父子分块，并保留页码级证据。",
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test_v1",
    )
    scored = ScoredChunk(chunk=chunk, source_name="demo.pdf", dense_rank=1, dense_score=0.91)
    trace = RetrievalTrace(
        trace_id="trace_with_evidence",
        query="方法是什么？",
        search_query="论文的核心方法是什么？",
        query_type=query_type,
        candidate_count=1,
        results=(scored,),
        elapsed_ms=1.0,
        stages=("dense_child_retrieval", "parent_hydration"),
    )
    return RetrievalResult(results=(scored,), trace=trace)


class GenerationTests(unittest.TestCase):
    def test_langchain_adapter_records_attempts_and_provider_tokens(self) -> None:
        client = LangChainTextGenerationClient(_RecoveringChatModel())
        assert client.generate(system_prompt="s", user_prompt="u") == "恢复成功"
        usage = client.usage_snapshot()
        assert usage.model_calls == 2
        assert usage.prompt_tokens == 12
        assert usage.completion_tokens == 3

    def test_answer_contains_citations_and_evidence_bound_prompt(self) -> None:
        client = _FakeGenerationClient()
        answer = AnswerGenerator(client).generate(
            question="这篇论文的方法是什么？",
            retrieval=_retrieval_result(with_evidence=True),
            context=GenerationContext(working_memory="用户正在阅读方法章节。"),
        )

        self.assertFalse(answer.refused)
        self.assertEqual(answer.trace_id, "trace_with_evidence")
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].page_start, 2)
        self.assertEqual(answer.citations[0].evidence, "论文的方法使用父子分块，并保留页码级证据。")
        system_prompt, user_prompt = client.calls[0]
        self.assertIn("只能依据", system_prompt)
        self.assertIn("来源：demo.pdf", user_prompt)
        self.assertIn("第 2 页", user_prompt)

    def test_empty_retrieval_refuses_without_calling_model(self) -> None:
        client = _FakeGenerationClient()
        answer = AnswerGenerator(client).generate(
            question="论文作者是谁？", retrieval=_retrieval_result(with_evidence=False)
        )

        self.assertTrue(answer.refused)
        self.assertEqual(answer.citations, ())
        self.assertIn("未检索", answer.refusal_reason or "")
        self.assertEqual(client.calls, [])

    def test_langchain_adapter_wraps_provider_failure_as_retryable_error(self) -> None:
        model = _FailingChatModel()
        client = LangChainTextGenerationClient(model)

        with self.assertRaises(ModelServiceUnavailable) as raised:
            client.generate(system_prompt="system", user_prompt="question")

        self.assertIn("模型服务暂时不可用", str(raised.exception))
        self.assertEqual(model.calls, 2)

    def test_langchain_adapter_retries_once_before_first_token(self) -> None:
        model = _RecoveringChatModel()
        client = LangChainTextGenerationClient(model)

        text = client.generate(system_prompt="system", user_prompt="question")

        self.assertEqual(text, "恢复成功")
        self.assertEqual(model.calls, 2)

    def test_stream_retry_is_allowed_before_first_delta(self) -> None:
        model = _RecoveringStreamingChatModel()
        client = LangChainTextGenerationClient(model)
        deltas: list[str] = []

        text = client.generate_stream(system_prompt="system", user_prompt="question", on_text_delta=deltas.append)

        self.assertEqual(text, "恢复成功")
        self.assertEqual(deltas, ["恢复", "成功"])
        self.assertEqual(model.calls, 2)

    def test_stream_does_not_retry_after_first_visible_delta(self) -> None:
        model = _PartialThenFailStreamingChatModel()
        client = LangChainTextGenerationClient(model)
        deltas: list[str] = []

        with self.assertRaises(ModelServiceUnavailable):
            client.generate_stream(system_prompt="system", user_prompt="question", on_text_delta=deltas.append)

        self.assertEqual(deltas, ["部分内容"])
        self.assertEqual(model.calls, 1)

    def test_task_specific_generation_instructions_are_added_to_prompt(self) -> None:
        expectations = {
            QueryType.METADATA: "作者匿名时必须如实说明",
            QueryType.DOCUMENT_SUMMARY: "研究问题、核心方法、实验与结果",
            QueryType.EXPERIMENTAL_SETUP: "数据集、基线、评价指标",
            QueryType.EXPERIMENTAL_RESULTS: "定量比较、消融实验",
            QueryType.MULTI_SECTION: "跨章节综合问题",
        }

        for query_type, expected in expectations.items():
            with self.subTest(query_type=query_type):
                client = _FakeGenerationClient()
                AnswerGenerator(client).generate(
                    question="请回答论文问题",
                    retrieval=_retrieval_result(
                        with_evidence=True,
                        query_type=query_type,
                    ),
                )
                self.assertIn(expected, client.calls[0][0])

    def test_langchain_adapter_forwards_real_stream_deltas(self) -> None:
        client = LangChainTextGenerationClient(_StreamingChatModel())
        deltas: list[str] = []

        text = client.generate_stream(
            system_prompt="system",
            user_prompt="question",
            on_text_delta=deltas.append,
        )

        self.assertEqual(deltas, ["第一段", "，第二段"])
        self.assertEqual(text, "第一段，第二段")

    def test_answer_generator_streams_and_still_returns_structured_answer(self) -> None:
        client = LangChainTextGenerationClient(_StreamingChatModel())
        deltas: list[str] = []

        answer = AnswerGenerator(client).generate(
            question="这篇论文的方法是什么？",
            retrieval=_retrieval_result(with_evidence=True),
            on_text_delta=deltas.append,
        )

        self.assertEqual(deltas, ["第一段", "，第二段"])
        self.assertEqual(answer.text, "第一段，第二段")
        # The stream is preserved, but evidence that the model never cited is
        # no longer presented as if it supported the answer.
        self.assertEqual(len(answer.citations), 0)
        self.assertEqual(answer.claims[0].support_status, "unsupported")


if __name__ == "__main__":
    unittest.main()
