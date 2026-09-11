"""HTTP 契约与范围边界测试，只使用假服务，不加载真实模型。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.api.app import create_app  # noqa: E402
from researchpilot.config import build_settings  # noqa: E402
from researchpilot.generation import ModelServiceUnavailable  # noqa: E402


class _FakeService:
    def __init__(self) -> None:
        self.deleted: list[dict[str, str]] = []
        self.retrieve_calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> SimpleNamespace:
        self.retrieve_calls.append(kwargs)
        chunk = SimpleNamespace(
            chunk_id="chunk_demo",
            paper_id="paper_demo",
            page_start=2,
            page_end=2,
            section_path=("Method",),
        )
        scored = SimpleNamespace(
            chunk=chunk,
            source_name="demo.pdf",
            dense_score=0.8,
            sparse_score=3.2,
            fusion_score=0.1,
            rerank_score=1.3,
        )
        trace = SimpleNamespace(
            trace_id="trace_demo",
            search_query=str(kwargs["question"]),
            candidate_count=1,
            elapsed_ms=12.0,
            stages=("dense", "rerank"),
        )
        return SimpleNamespace(trace=trace, results=(scored,))

    def ingest_pdf(self, content: bytes, **kwargs: object) -> SimpleNamespace:
        self.ingested = {"content": content, **kwargs}
        paper = SimpleNamespace(
            paper_id="paper_api_demo",
            source_name=str(kwargs["source_name"]),
            page_count=2,
            language="en",
        )
        quality = SimpleNamespace(
            parser_name="pdfplumber",
            extraction_quality=SimpleNamespace(value="normal"),
        )
        return SimpleNamespace(
            paper=paper,
            parent_chunks=(object(),),
            child_chunks=(object(), object()),
            parse_quality=quality,
        )

    def answer_with_retrieval(self, **kwargs: object) -> SimpleNamespace:
        retrieved = self.retrieve(**kwargs)
        citation = SimpleNamespace(
            paper_id="paper_demo",
            chunk_id="chunk_demo",
            page_start=2,
            page_end=2,
            section_path=("Method",),
            confidence=0.9,
        )
        answer = SimpleNamespace(
            trace_id="trace_demo",
            text="带证据的回答",
            citations=(citation,),
            refused=False,
            refusal_reason=None,
            latency_ms=20.0,
        )
        return SimpleNamespace(answer=answer, retrieval=retrieved)

    def delete_paper(self, **kwargs: str) -> int:
        self.deleted.append(kwargs)
        return 2


class _FakeProvider:
    def __init__(self) -> None:
        self.service = _FakeService()

    def get(self) -> _FakeService:
        return self.service


class _UnavailableModelService(_FakeService):
    def answer_with_retrieval(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        raise ModelServiceUnavailable("network unavailable")


class _UnavailableModelProvider:
    def __init__(self) -> None:
        self.service = _UnavailableModelService()

    def get(self) -> _UnavailableModelService:
        return self.service


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        settings = build_settings(project_root=Path(self.tempdir.name), environ={}, load_dotenv_file=False)
        self.provider = _FakeProvider()
        self.client = TestClient(create_app(settings=settings, service_provider=self.provider))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_retrieve_keeps_page_and_scores_but_not_chunk_text(self) -> None:
        response = self.client.post(
            "/v1/retrieve", json={"question": "核心方法是什么？", "user_id": "user_a", "session_id": "session_a"}
        )
        self.assertEqual(response.status_code, 200)
        item = response.json()["results"][0]
        self.assertEqual(item["page_start"], 2)
        self.assertEqual(item["dense_score"], 0.8)
        self.assertNotIn("content", item)

    def test_explicit_paper_scope_is_forwarded_without_exposing_other_scope(self) -> None:
        response = self.client.post(
            "/v1/retrieve",
            json={
                "question": "核心方法是什么？",
                "user_id": "user_a",
                "session_id": "session_a",
                "paper_id": "paper_demo",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.provider.service.retrieve_calls[-1]["paper_id"], "paper_demo")

    def test_default_chat_has_stable_contract(self) -> None:
        upload = self.client.post(
            "/v1/documents",
            files={"file": ("demo.pdf", b"%PDF-test", "application/pdf")},
            data={"user_id": "user_a", "session_id": "session_a"},
        )
        self.assertEqual(upload.status_code, 201)
        response = self.client.post(
            "/v1/chat",
            json={
                "question": "核心方法是什么？",
                "user_id": "user_a",
                "session_id": "session_a",
                "paper_id": "paper_api_demo",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"][0]["paper_id"], "paper_demo")
        self.assertEqual(response.json()["route"], "document_detail")

    def test_general_chat_and_missing_paper_scope_do_not_call_retrieval(self) -> None:
        general_chat = self.client.post(
            "/v1/chat", json={"question": "你好", "user_id": "user_a", "session_id": "session_a"}
        )
        self.assertEqual(general_chat.status_code, 200)
        self.assertEqual(general_chat.json()["route"], "general_chat")
        self.assertEqual(general_chat.json()["citations"], [])

        missing_scope = self.client.post(
            "/v1/chat", json={"question": "核心方法是什么？", "user_id": "user_a", "session_id": "session_a"}
        )
        self.assertEqual(missing_scope.status_code, 200)
        self.assertEqual(missing_scope.json()["route"], "scope_insufficient")
        self.assertTrue(missing_scope.json()["refused"])
        self.assertEqual(self.provider.service.retrieve_calls, [])

    def test_strict_crossdoc_is_explicitly_gated_not_silently_downgraded(self) -> None:
        response = self.client.post(
            "/v1/chat",
            json={
                "question": "比较两篇论文",
                "user_id": "user_a",
                "session_id": "session_a",
                "mode": "strict_crossdoc",
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "strict_crossdoc_not_available")

    def test_document_registry_is_scoped_and_delete_removes_vectors(self) -> None:
        upload = self.client.post(
            "/v1/documents",
            files={"file": ("demo.pdf", b"%PDF-test", "application/pdf")},
            data={"user_id": "user_a", "session_id": "session_a"},
        )
        self.assertEqual(upload.status_code, 201)
        self.assertEqual(upload.json()["paper_id"], "paper_api_demo")

        listed = self.client.get("/v1/documents?user_id=user_a&session_id=session_a")
        self.assertEqual([item["paper_id"] for item in listed.json()["documents"]], ["paper_api_demo"])
        other_scope = self.client.get("/v1/documents?user_id=user_b&session_id=session_a")
        self.assertEqual(other_scope.json()["documents"], [])

        deleted = self.client.delete("/v1/documents/paper_api_demo?user_id=user_a&session_id=session_a")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json()["deleted_vector_records"], 2)
        self.assertEqual(self.provider.service.deleted[0]["paper_id"], "paper_api_demo")

    def test_invalid_scope_is_an_explicit_client_error(self) -> None:
        response = self.client.post(
            "/v1/retrieve", json={"question": "方法", "user_id": "../other", "session_id": "session_a"}
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_scope")

    def test_model_connection_failure_returns_retryable_503_without_stack_trace(self) -> None:
        provider = _UnavailableModelProvider()
        client = TestClient(
            create_app(
                settings=build_settings(
                    project_root=Path(self.tempdir.name),
                    environ={},
                    load_dotenv_file=False,
                ),
                service_provider=provider,
            )
        )
        upload = client.post(
            "/v1/documents",
            files={"file": ("demo.pdf", b"%PDF-test", "application/pdf")},
            data={"user_id": "user_a", "session_id": "session_a"},
        )
        self.assertEqual(upload.status_code, 201)

        response = client.post(
            "/v1/chat",
            json={
                "question": "实验设置是什么？",
                "user_id": "user_a",
                "session_id": "session_a",
                "paper_id": "paper_api_demo",
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "model_service_unavailable")
        self.assertNotIn("network unavailable", response.text)


if __name__ == "__main__":
    unittest.main()
