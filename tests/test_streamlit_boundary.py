"""Static guardrails for the thin Streamlit product boundary."""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")


class StreamlitBoundaryTests(unittest.TestCase):
    def test_page_does_not_open_sqlite_or_execute_sql_directly(self) -> None:
        self.assertNotIn("import sqlite3", APP_SOURCE)
        self.assertNotIn("cursor.execute", APP_SOURCE)
        self.assertNotIn("SQLiteMemoryRepository", APP_SOURCE)
        self.assertIn("product_runtime.data_maintenance", APP_SOURCE)

    def test_pdf_ingestion_and_retrieval_are_delegated_to_core(self) -> None:
        self.assertIn("active_rag_service().ingest_pdf(", APP_SOURCE)
        self.assertIn("RuntimeOrchestrator(", APP_SOURCE)
        self.assertIn("orchestrator.handle(request", APP_SOURCE)
        self.assertNotIn("import pdfplumber", APP_SOURCE)
        self.assertNotIn("BM25Okapi", APP_SOURCE)
        self.assertNotIn("PersistentBM25Repository", APP_SOURCE)
        self.assertNotIn("ChromaVectorIndexRepository", APP_SOURCE)
        self.assertNotIn("CrossEncoder", APP_SOURCE)
        self.assertNotIn("reranker.predict", APP_SOURCE)
        self.assertNotIn("llm.invoke(", APP_SOURCE)
        self.assertNotIn("global_vdb.", APP_SOURCE)
        self.assertNotIn("session_vdb.", APP_SOURCE)

    def test_routing_uses_original_question_and_single_orchestrator(self) -> None:
        self.assertIn("question=user_prompt", APP_SOURCE)
        self.assertIn("conversation_service=conversation_service", APP_SOURCE)
        self.assertEqual(APP_SOURCE.count("RuntimeRouter().route("), 0)
        self.assertEqual(APP_SOURCE.count("orchestrator.handle("), 1)

    def test_scoped_working_memory_and_scoped_messages_are_used(self) -> None:
        self.assertIn("product_runtime.memory_service", APP_SOURCE)
        self.assertIn("paper_scope_key", APP_SOURCE)
        self.assertIn("format_recent_history(", APP_SOURCE)
        self.assertIn("switch_working_memory_scope", APP_SOURCE)
        self.assertIn("maybe_update_working_memory", APP_SOURCE)
        self.assertIn("maybe_extract_incremental_conclusions", APP_SOURCE)

    def test_legacy_is_read_only_without_old_hybrid_query(self) -> None:
        self.assertIn("LegacySessionAdapter.inspect", APP_SOURCE)
        self.assertIn("legacy_status.answer_text", APP_SOURCE)
        self.assertNotIn("Legacy 意图路由判定结果", APP_SOURCE)
        self.assertNotIn("裂变后的查询矩阵", APP_SOURCE)

    def test_claim_contract_and_page_view_are_rendered(self) -> None:
        self.assertIn('"claims": [', APP_SOURCE)
        self.assertIn("Claim-Citation 契约", APP_SOURCE)
        self.assertIn('"runtime_route": getattr(route, "value", None)', APP_SOURCE)
        self.assertIn("任务路由：{runtime_route}", APP_SOURCE)
        self.assertIn("render_pdf_page_png", APP_SOURCE)
        self.assertIn("查看第 {page_start} 页", APP_SOURCE)

    def test_research_notes_export_and_session_title_are_product_features(self) -> None:
        self.assertIn("保存为研究笔记", APP_SOURCE)
        self.assertIn("export_markdown", APP_SOURCE)
        self.assertIn("derive_session_title", APP_SOURCE)
        self.assertIn("session_title", APP_SOURCE)

    def test_upload_is_model_independent_and_persists_local_pdf_copy(self) -> None:
        self.assertIn("元信息与全文摘要将按需生成并附页码证据", APP_SOURCE)
        self.assertIn("persist_pdf_copy", APP_SOURCE)
        self.assertNotIn("正在生成元信息与论文摘要（会调用模型）", APP_SOURCE)

    def test_product_profile_branding_and_streaming_remain_wired(self) -> None:
        self.assertIn('DEFAULT_PRODUCT_PROFILE_NAME = "high_precision"', APP_SOURCE)
        self.assertIn('page_title="ResearchPilot-RAG"', APP_SOURCE)
        self.assertIn('full_logo_path = settings.resources_dir / "full_logo.png"', APP_SOURCE)
        self.assertIn("on_text_delta=render_delta", APP_SOURCE)

    def test_browser_scope_is_stable_across_refreshes(self) -> None:
        self.assertIn('st.query_params.get("rp_user")', APP_SOURCE)
        self.assertIn('st.query_params["rp_user"] = generated_user_id', APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
