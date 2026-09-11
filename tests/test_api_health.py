"""P2.8 API 健康检查：不得因模型、Chroma 或 API Key 而失败。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.api.app import create_app  # noqa: E402
from researchpilot.config import build_settings  # noqa: E402


class ApiHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.settings = build_settings(
            project_root=root,
            environ={"DEEPSEEK_API_KEY": ""},
            load_dotenv_file=False,
        )
        self.client = TestClient(create_app(settings=self.settings))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_health_is_model_free_and_has_trace_id(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.headers["x-trace-id"].startswith("api_"))

    def test_ready_reports_configuration_without_exposing_key(self) -> None:
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["runtime_directories_ready"])
        self.assertFalse(response.json()["deepseek_key_configured"])
        self.assertNotIn("DEEPSEEK_API_KEY", response.text)


if __name__ == "__main__":
    unittest.main()
