"""P2.8 本地 Trace 的隐私与字段边界测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.observability import LocalTraceStore  # noqa: E402


class ObservabilityTests(unittest.TestCase):
    def test_trace_is_append_only_and_does_not_accept_question_or_answer_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalTraceStore(Path(directory))
            started_at = time.perf_counter()
            store.record(
                trace_id="api_demo",
                operation="rag.retrieve",
                started_at=started_at,
                user_id="user_a",
                session_id="session_a",
                status="ok",
                candidate_count=3,
                route="document_detail",
                prompt_tokens=120,
                completion_tokens=30,
                rewrite_model_calls=1,
                answer_model_calls=1,
                cache_hit=False,
            )
            line = store.path.read_text(encoding="utf-8").strip()
            payload = json.loads(line)

        self.assertEqual(payload["candidate_count"], 3)
        self.assertEqual(payload["route"], "document_detail")
        self.assertEqual(payload["prompt_tokens"], 120)
        self.assertEqual(payload["answer_model_calls"], 1)
        self.assertNotIn("question", payload)
        self.assertNotIn("answer", payload)
        self.assertNotEqual(payload["user_scope_hash"], "user_a")
        self.assertNotEqual(payload["session_scope_hash"], "session_a")


if __name__ == "__main__":
    unittest.main()
