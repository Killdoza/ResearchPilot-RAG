"""Tests for user-scoped JSON sessions and transactional SQLite memory."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.config import ConfigurationError, build_settings  # noqa: E402
from researchpilot.storage import (  # noqa: E402
    PaperAnnotation,
    SessionDataCorrupted,
    SessionFileRepository,
    SQLiteMemoryRepository,
)


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.settings = build_settings(
            project_root=self.root,
            environ={"RESEARCHPILOT_MEMORY_DB_PATH": "runtime/memory.sqlite3"},
            load_dotenv_file=False,
        )
        self.memory = SQLiteMemoryRepository(self.settings)
        self.memory.initialize_schema()
        self.sessions = SessionFileRepository(self.settings)
        self.user_a = "local_alice"
        self.user_b = "local_bob"
        self.session_a = "2026-08-15_alice"
        self.session_b = "2026-08-15_bob"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_user_scopes_are_isolated_for_files_and_memory(self) -> None:
        self.sessions.save(self.user_a, self.session_a, [{"role": "user", "content": "A"}], [])
        self.memory.save_working_memory(self.user_a, self.session_a, {"current_topic": "Alice paper"})
        self.memory.add_long_term_memory(self.user_a, self.session_a, "session_conclusion", "Alice conclusion")

        self.assertIsNone(self.sessions.load(self.user_b, self.session_a))
        self.assertEqual(self.memory.load_working_memory(self.user_b, self.session_a)["current_topic"], "")
        self.assertEqual(self.memory.list_active_long_term_memories(self.user_b, self.session_b), [])

    def test_delete_one_session_does_not_touch_another_user(self) -> None:
        for user_id, session_id, text in (
            (self.user_a, self.session_a, "Alice"),
            (self.user_b, self.session_b, "Bob"),
        ):
            self.sessions.save(user_id, session_id, [], [])
            self.memory.save_working_memory(user_id, session_id, {"current_topic": text})
            self.memory.add_long_term_memory(user_id, session_id, "document_insight", text)

        self.assertTrue(self.sessions.delete(self.user_a, self.session_a))
        counts = self.memory.delete_session(self.user_a, self.session_a)

        self.assertEqual(counts["working_memory_deleted"], 1)
        self.assertIsNone(self.sessions.load(self.user_a, self.session_a))
        self.assertEqual(self.memory.list_active_long_term_memories(self.user_a, self.session_a), [])
        self.assertIsNotNone(self.sessions.load(self.user_b, self.session_b))
        self.assertEqual(self.memory.load_working_memory(self.user_b, self.session_b)["current_topic"], "Bob")
        self.assertEqual(
            [item.content for item in self.memory.list_active_long_term_memories(self.user_b, self.session_b)],
            ["Bob"],
        )

    def test_invalid_session_id_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.sessions.save(self.user_a, "../escape", [], [])
        with self.assertRaises(ConfigurationError):
            self.memory.save_working_memory("../escape", self.session_a, {})

    def test_failed_transaction_rolls_back_every_write(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with self.memory.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO rp_long_term_memory
                        (user_id, memory_type, content, source_session, priority, is_active, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (self.user_a, "test", "will rollback", self.session_a, 1, 1, "2026-08-15"),
                )
                connection.execute(
                    """
                    INSERT INTO rp_long_term_memory
                        (user_id, memory_type, content, source_session, priority, is_active, last_updated)
                    VALUES (?, ?, NULL, ?, ?, ?, ?)
                    """,
                    (self.user_a, "test", self.session_a, 1, 1, "2026-08-15"),
                )
        self.assertEqual(self.memory.list_active_long_term_memories(self.user_a, self.session_a), [])

    def test_empty_duplicate_and_corrupted_sessions_are_handled(self) -> None:
        self.sessions.save(self.user_a, self.session_a, [], [])
        self.sessions.save(
            self.user_a,
            self.session_a,
            [{"role": "user", "content": "replacement"}],
            ["paper.pdf"],
            [{"source_name": "paper.pdf", "paper_id": "paper_demo"}],
            "paper_demo",
        )
        loaded = self.sessions.load(self.user_a, self.session_a)
        self.assertEqual(loaded.messages[0]["content"], "replacement")
        self.assertEqual(loaded.active_pdf_names, ["paper.pdf"])
        self.assertEqual(loaded.active_documents[0]["paper_id"], "paper_demo")
        self.assertEqual(loaded.selected_paper_id, "paper_demo")
        self.assertEqual(self.sessions.list(self.user_a, self.session_a), [self.session_a])

        corrupt_id = "2026-08-15_corrupt"
        corrupt_path = self.settings.session_file_path(self.user_a, corrupt_id)
        corrupt_path.parent.mkdir(parents=True, exist_ok=True)
        corrupt_path.write_text("{not-json", encoding="utf-8")
        with self.assertRaises(SessionDataCorrupted):
            self.sessions.load(self.user_a, corrupt_id)

    def test_v1_session_without_product_metadata_remains_readable(self) -> None:
        legacy_path = self.settings.session_file_path(self.user_a, "legacy_session")
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "current_session": "legacy_session",
                    "messages": [],
                    "active_pdf_names": ["legacy.pdf"],
                }
            ),
            encoding="utf-8",
        )

        loaded = self.sessions.load(self.user_a, "legacy_session")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.active_documents, [])
        self.assertIsNone(loaded.selected_paper_id)

    def test_relevant_long_term_memory_respects_session_and_paper_scope(self) -> None:
        other_session = "2026-08-16_alice"
        self.memory.add_long_term_memory(
            self.user_a,
            other_session,
            "user_preference",
            "用户偏好中文回答",
            priority=5,
        )
        self.memory.add_long_term_memory(
            self.user_a,
            self.session_a,
            "session_conclusion",
            "当前会话结论",
            source_doc="paper_a",
            priority=4,
        )
        self.memory.add_long_term_memory(
            self.user_a,
            other_session,
            "session_conclusion",
            "同论文历史结论",
            source_doc="paper_a",
            priority=4,
        )
        self.memory.add_long_term_memory(
            self.user_a,
            other_session,
            "session_conclusion",
            "其他论文历史结论",
            source_doc="paper_b",
            priority=4,
        )
        self.memory.add_long_term_memory(
            self.user_a,
            self.session_a,
            "session_conclusion",
            "同一会话中的论文 B 结论",
            source_doc="paper_b",
            priority=4,
        )
        self.memory.add_long_term_memory(
            self.user_a,
            other_session,
            "session_conclusion",
            "未绑定论文的旧结论",
            priority=4,
        )
        self.memory.add_long_term_memory(
            self.user_a,
            other_session,
            "document_insight",
            "只能由具体 Chunk 注入的批注",
            source_doc="paper_a",
            priority=3,
        )

        contents = {
            item.content
            for item in self.memory.list_relevant_long_term_memories(
                self.user_a,
                self.session_a,
                source_docs=("paper_a",),
            )
        }

        self.assertEqual(
            contents,
            {"用户偏好中文回答", "当前会话结论", "同论文历史结论"},
        )

    def test_annotations_can_be_recalled_across_chunks_within_same_paper(self) -> None:
        self.memory.add_annotation(
            PaperAnnotation(
                annotation_id=None,
                user_id=self.user_a,
                session_id=self.session_a,
                paper_id="paper_a",
                chunk_id="chunk_old",
                content="这里的胜率方向需要核验",
                anchor_page=6,
                section_path=("4 Experiments", "4.2 Results"),
            )
        )
        recalled = self.memory.search_annotations(
            user_id=self.user_a,
            session_id=self.session_a,
            paper_id="paper_a",
            chunk_ids=("chunk_new",),
            pages=(7,),
            section_paths=(("4 Experiments", "4.2 Results"),),
            query="请核验实验胜率",
        )
        assert len(recalled) == 1
        assert recalled[0].chunk_id == "chunk_old"
        assert recalled[0].anchor_page == 6
        assert recalled[0].section_path[-1] == "4.2 Results"

        assert not self.memory.search_annotations(
            user_id=self.user_a,
            session_id=self.session_a,
            paper_id="paper_b",
            query="胜率",
        )

    def test_relevant_long_term_memory_without_paper_excludes_cross_session_research(self) -> None:
        other_session = "2026-08-16_alice"
        self.memory.add_long_term_memory(
            self.user_a,
            other_session,
            "user_preference",
            "偏好简洁回答",
        )
        self.memory.add_long_term_memory(
            self.user_a,
            other_session,
            "unresolved_question",
            "另一篇论文的问题",
            source_doc="paper_b",
        )

        contents = [
            item.content
            for item in self.memory.list_relevant_long_term_memories(
                self.user_a,
                self.session_a,
            )
        ]

        self.assertEqual(contents, ["偏好简洁回答"])

    def test_settling_one_paper_does_not_deactivate_another_paper(self) -> None:
        other_session = "2026-08-16_alice"
        self.memory.add_long_term_memory(
            self.user_a,
            other_session,
            "session_conclusion",
            "论文 B 的既有结论",
            source_doc="paper_b",
        )

        self.memory.replace_memory_types(
            self.user_a,
            self.session_a,
            ("session_conclusion", "unresolved_question"),
            (("session_conclusion", "论文 A 的新结论", 4),),
            source_doc="paper_a",
        )

        active_contents = {
            item.content
            for item in self.memory.list_active_long_term_memories(
                self.user_a,
                self.session_a,
            )
        }
        self.assertEqual(active_contents, {"论文 A 的新结论", "论文 B 的既有结论"})

    def test_user_preferences_remain_global_when_scoped_research_is_settled(self) -> None:
        other_session = "2026-08-16_alice"
        self.memory.add_long_term_memory(
            self.user_a,
            other_session,
            "user_preference",
            "旧偏好",
        )

        self.memory.replace_memory_types(
            self.user_a,
            self.session_a,
            ("user_preference", "session_conclusion"),
            (
                ("user_preference", "新偏好", 5),
                ("session_conclusion", "论文 A 的结论", 4),
            ),
            source_doc="paper_a",
        )

        memories = self.memory.list_active_long_term_memories(
            self.user_a,
            self.session_a,
        )
        preference_rows = [item for item in memories if item.memory_type == "user_preference"]
        conclusion_rows = [item for item in memories if item.memory_type == "session_conclusion"]
        self.assertEqual([item.content for item in preference_rows], ["新偏好"])
        self.assertIsNone(preference_rows[0].source_doc)
        self.assertEqual([item.source_doc for item in conclusion_rows], ["paper_a"])

    def test_annotations_are_separate_from_long_term_paper_facts(self) -> None:
        annotation_id = self.memory.add_annotation(
            PaperAnnotation(
                None,
                self.user_a,
                self.session_a,
                "paper_a",
                "chunk_a",
                "该比较没有控制模型规模",
                relation="contradicts",
            )
        )
        annotations = self.memory.list_annotations(
            user_id=self.user_a,
            session_id=self.session_a,
            paper_id="paper_a",
            chunk_ids=("chunk_a",),
        )
        self.assertGreater(annotation_id, 0)
        self.assertEqual(annotations[0].verification_status, "user_note")
        self.assertEqual(self.memory.list_document_insights(self.user_a, self.session_a, "chunk_a"), [])


if __name__ == "__main__":
    unittest.main()
