from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.config import build_settings  # noqa: E402
from researchpilot.memory_service import (  # noqa: E402
    ScopedMemoryService,
    format_recent_history,
    scope_key_for_papers,
)
from researchpilot.storage import SQLiteMemoryRepository  # noqa: E402


class FakeMemoryConversation:
    def update_working_memory(self, *, current_memory, recent_history):
        return {
            "current_topic": "A 的实验结果",
            "core_demands": ["核对数字"],
            "confirmed_conclusions": [],
            "unresolved_questions": [],
            "context_summary": recent_history,
        }

    def consolidate_long_term_memory(self, *, working_memory, existing_preferences, existing_conclusions):
        return {
            "user_preferences": [*existing_preferences, "使用中文"],
            "session_conclusions": [*existing_conclusions, "A 的结论"],
            "unresolved_questions": ["继续核对 A"],
        }

    def extract_incremental_conclusions(self, *, recent_history):
        return {"new_conclusions": ["A 的新增结论"], "unresolved_issues": []}


class ScopedMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        settings = build_settings(project_root=Path(self.temp.name), environ={}, load_dotenv_file=False)
        self.repo = SQLiteMemoryRepository(settings)
        self.repo.initialize_schema()
        self.service = ScopedMemoryService(self.repo)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_paper_working_memories_do_not_leak(self) -> None:
        self.service.save(
            user_id="user_a",
            session_id="session_a",
            scope_key="paper:paper_a",
            memory={"current_topic": "A 方法"},
        )
        paper_a_memory = self.service.load(user_id="user_a", session_id="session_a", scope_key="paper:paper_a")
        paper_b_memory = self.service.load(user_id="user_a", session_id="session_a", scope_key="paper:paper_b")
        self.assertEqual(paper_a_memory["current_topic"], "A 方法")
        self.assertEqual(paper_b_memory["current_topic"], "")

    def test_crossdoc_scope_is_order_independent(self) -> None:
        self.assertEqual(scope_key_for_papers(("b", "a")), scope_key_for_papers(("a", "b")))
        self.assertTrue(scope_key_for_papers(("a", "b")).startswith("crossdoc:"))

    def test_recent_history_only_uses_same_scope(self) -> None:
        messages = [
            {"role": "user", "content": "A 的方法", "paper_scope_key": "paper:a"},
            {"role": "assistant", "content": "B 的结果", "paper_scope_key": "paper:b"},
        ]
        text = format_recent_history(messages, "paper:a")
        self.assertIn("A 的方法", text)
        self.assertNotIn("B 的结果", text)

    def test_failed_turn_is_excluded_from_history_and_memory_update(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "失败问题",
                "paper_scope_key": "paper:paper_a",
                "status": "pending",
                "eligible_for_memory": False,
            },
            {
                "role": "assistant",
                "content": "网络错误",
                "paper_scope_key": "paper:paper_a",
                "status": "failed",
                "eligible_for_memory": False,
            },
            {
                "role": "user",
                "content": "成功问题",
                "paper_scope_key": "paper:paper_a",
                "status": "completed",
                "eligible_for_memory": True,
            },
        ]

        text = format_recent_history(messages, "paper:paper_a")

        self.assertNotIn("失败问题", text)
        self.assertNotIn("网络错误", text)
        self.assertIn("成功问题", text)

    def test_memory_updates_and_long_term_records_stay_in_paper_scope(self) -> None:
        messages = [
            {"role": "user", "content": f"A 问题 {index}", "paper_scope_key": "paper:paper_a"} for index in range(5)
        ]
        conversation = FakeMemoryConversation()
        update = self.service.maybe_update_working_memory(
            user_id="user_a",
            session_id="session_a",
            scope_key="paper:paper_a",
            messages=messages,
            current_memory={},
            last_update_count=0,
            conversation=conversation,
        )
        self.assertTrue(update.updated)
        self.assertEqual(update.memory["current_topic"], "A 的实验结果")

        self.service.settle_long_term_memory(
            user_id="user_a",
            session_id="session_a",
            paper_ids=("paper_a",),
            working_memory=update.memory,
            conversation=conversation,
        )
        last_turn = self.service.maybe_extract_incremental_conclusions(
            user_id="user_a",
            session_id="session_a",
            paper_ids=("paper_a",),
            scope_key="paper:paper_a",
            messages=messages,
            last_turn=0,
            conversation=conversation,
        )
        self.assertEqual(last_turn, 5)
        paper_a = self.repo.list_relevant_long_term_memories("user_a", "session_a", source_docs=("paper_a",))
        paper_b = self.repo.list_relevant_long_term_memories("user_a", "session_a", source_docs=("paper_b",))
        self.assertIn("A 的新增结论", {item.content for item in paper_a})
        self.assertNotIn("A 的新增结论", {item.content for item in paper_b})


if __name__ == "__main__":
    unittest.main()
