"""Unit tests for deterministic, non-secret application configuration."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.config import (  # noqa: E402
    ConfigurationError,
    build_settings,
    ensure_runtime_directories,
    require_ascii_chroma_path,
    require_deepseek_api_key,
    validate_storage_id,
)
from researchpilot.session import generate_local_user_id, generate_session_id  # noqa: E402


class SettingsTests(unittest.TestCase):
    def test_defaults_use_repository_relative_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = build_settings(
                project_root=root,
                environ={"DEEPSEEK_API_KEY": "test-key"},
                load_dotenv_file=False,
            )

            self.assertEqual(settings.global_chroma_dir, root / "data/runtime/chroma_global")
            self.assertEqual(settings.session_chroma_dir, root / "data/runtime/chroma_session")
            self.assertEqual(settings.evaluation_chroma_dir, root / "data/runtime/chroma_evaluation")
            self.assertEqual(settings.sessions_dir, root / "data/runtime/sessions")
            self.assertEqual(settings.memory_db_path, root / "data/runtime/researchpilot.sqlite3")
            self.assertEqual(settings.runs_dir, root / "data/runtime/runs")
            self.assertEqual(settings.global_collection_name, "global_memory")
            self.assertEqual(settings.session_collection_name, "session_memory")
            self.assertEqual(settings.evaluation_collection_name, "evaluation_papers_v01")
            self.assertEqual(require_deepseek_api_key(settings), "test-key")

    def test_environment_overrides_are_resolved_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = build_settings(
                project_root=root,
                environ={
                    "DEEPSEEK_API_KEY": "test-key",
                    "RESEARCHPILOT_SESSIONS_DIR": "runtime/sessions",
                    "RESEARCHPILOT_MEMORY_DB_PATH": "runtime/memory.sqlite3",
                    "RESEARCHPILOT_SESSION_COLLECTION": "isolated_session_chunks",
                    "RESEARCHPILOT_EVALUATION_CHROMA_DIR": "runtime/evaluation_chroma",
                    "RESEARCHPILOT_EVALUATION_COLLECTION": "isolated_evaluation_chunks",
                },
                load_dotenv_file=False,
            )

            self.assertEqual(settings.sessions_dir, root / "runtime" / "sessions")
            self.assertEqual(settings.memory_db_path, root / "runtime" / "memory.sqlite3")
            self.assertEqual(settings.session_collection_name, "isolated_session_chunks")
            self.assertEqual(settings.evaluation_chroma_dir, root / "runtime" / "evaluation_chroma")
            self.assertEqual(settings.evaluation_collection_name, "isolated_evaluation_chunks")

    def test_optional_local_user_id_must_be_storage_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = build_settings(
                project_root=Path(temp_dir),
                environ={"RESEARCHPILOT_LOCAL_USER_ID": "local_killdoza"},
                load_dotenv_file=False,
            )
            self.assertEqual(settings.local_user_id, "local_killdoza")
            with self.assertRaises(ConfigurationError):
                build_settings(
                    project_root=Path(temp_dir),
                    environ={"RESEARCHPILOT_LOCAL_USER_ID": "../unsafe"},
                    load_dotenv_file=False,
                )

    def test_session_paths_reject_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = build_settings(
                project_root=Path(temp_dir),
                environ={},
                load_dotenv_file=False,
            )

            self.assertEqual(
                settings.session_file_path("local_user", "2026-08-13_demo"),
                Path(temp_dir) / "data/runtime/sessions" / "local_user" / "2026-08-13_demo.json",
            )
            with self.assertRaises(ConfigurationError):
                settings.session_file_path("local_user", "../another-session")
            with self.assertRaises(ConfigurationError):
                settings.session_file_path("../other-user", "2026-08-13_demo")

    def test_shared_storage_id_validation_rejects_path_traversal(self) -> None:
        validate_storage_id("user_a-01", "user_id")
        with self.assertRaises(ConfigurationError):
            validate_storage_id("../user_b", "user_id")

    def test_evaluation_chroma_path_requires_ascii(self) -> None:
        require_ascii_chroma_path(Path("E:/ResearchPilotRuntime/chroma_evaluation"), "test")
        with self.assertRaisesRegex(ConfigurationError, "非 ASCII"):
            require_ascii_chroma_path(Path("E:/python_projects/RAG项目/data/runtime"), "test")

    def test_missing_key_fails_with_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = build_settings(project_root=Path(temp_dir), environ={}, load_dotenv_file=False)
            with self.assertRaisesRegex(ConfigurationError, "DEEPSEEK_API_KEY"):
                require_deepseek_api_key(settings)

    def test_runtime_directories_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = build_settings(project_root=Path(temp_dir), environ={}, load_dotenv_file=False)
            ensure_runtime_directories(settings)
            self.assertTrue(settings.global_chroma_dir.is_dir())
            self.assertTrue(settings.session_chroma_dir.is_dir())
            self.assertTrue(settings.evaluation_chroma_dir.is_dir())
            self.assertTrue(settings.sessions_dir.is_dir())
            self.assertTrue(settings.runs_dir.is_dir())

    def test_session_id_is_sortable_and_collision_resistant(self) -> None:
        now = __import__("datetime").datetime(2026, 8, 13, 12, 30, 45)
        first = generate_session_id(now)
        second = generate_session_id(now)

        self.assertRegex(first, r"^2026-08-13_12-30-45_[0-9a-f]{12}$")
        self.assertNotEqual(first, second)

    def test_local_user_id_is_storage_safe_and_unique(self) -> None:
        first = generate_local_user_id()
        second = generate_local_user_id()

        self.assertRegex(first, r"^local_[0-9a-f]{32}$")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
