"""Regression test for safe copying of legacy SQLite and session data."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.config import build_settings  # noqa: E402
from researchpilot.storage import SQLiteMemoryRepository  # noqa: E402


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "researchpilot_legacy_migration", PROJECT_ROOT / "scripts" / "migrate_legacy_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


migration = _load_migration_module()


class LegacyMigrationTests(unittest.TestCase):
    def test_apply_helpers_copy_without_retagging_legacy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = build_settings(project_root=root, environ={}, load_dotenv_file=False)
            session_id = "2026-03-31_20-25-13"

            legacy_database = root / "memory_manager.db"
            connection = sqlite3.connect(legacy_database)
            try:
                connection.executescript(
                    """
                    CREATE TABLE working_memory (
                        session_id TEXT PRIMARY KEY, current_topic TEXT, core_demands TEXT,
                        confirmed_conclusions TEXT, unresolved_questions TEXT,
                        context_summary TEXT, last_updated DATETIME
                    );
                    CREATE TABLE long_term_memory (
                        id INTEGER PRIMARY KEY, user_id TEXT, memory_type TEXT, content TEXT,
                        source_session TEXT, source_doc TEXT, priority INTEGER,
                        is_active INTEGER, last_updated DATETIME
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO working_memory VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, "legacy topic", "[]", "[]", "[]", "summary", "2026-03-31"),
                )
                connection.execute(
                    "INSERT INTO long_term_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        1,
                        "default_user",
                        "session_conclusion",
                        "legacy conclusion",
                        session_id,
                        None,
                        4,
                        1,
                        "2026-03-31",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            legacy_sessions = root / "sessions"
            legacy_sessions.mkdir(parents=True)
            (legacy_sessions / f"{session_id}.json").write_text(
                json.dumps({"messages": [], "active_pdf_names": []}), encoding="utf-8"
            )

            self.assertEqual(migration.apply_session_copies(settings, root, "local_alice"), 1)
            result = migration.import_legacy_sqlite(settings, root, "local_alice")
            self.assertEqual(result, {"working_memory_copied": 1, "long_term_memory_copied": 1})

            repository = SQLiteMemoryRepository(settings)
            self.assertEqual(
                repository.load_working_memory("local_alice", session_id)["current_topic"],
                "legacy topic",
            )
            self.assertEqual(
                [item.content for item in repository.list_active_long_term_memories("local_alice", session_id)],
                ["legacy conclusion"],
            )

            source = sqlite3.connect(legacy_database)
            try:
                legacy_user = source.execute("SELECT user_id FROM long_term_memory WHERE id=1").fetchone()[0]
            finally:
                source.close()
            self.assertEqual(legacy_user, "default_user")

            self.assertEqual(migration.apply_session_copies(settings, root, "local_alice"), 0)
            self.assertEqual(
                migration.import_legacy_sqlite(settings, root, "local_alice"),
                {"working_memory_copied": 0, "long_term_memory_copied": 0},
            )


if __name__ == "__main__":
    unittest.main()
