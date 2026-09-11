"""Conversation and memory-prompt adapters without external model calls."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.conversation_service import ConversationService  # noqa: E402


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


def test_working_memory_output_is_normalized_to_the_fixed_schema() -> None:
    client = FakeClient(
        '```json\n{"current_topic":"实验", "core_demands":["核对结果"], '
        '"confirmed_conclusions":[], "unresolved_questions":["表格在哪"], '
        '"context_summary":"只讨论论文 A"}\n```'
    )
    result = ConversationService(client).update_working_memory(
        current_memory={}, recent_history="用户：论文 A 的结果是什么？"
    )
    assert result["current_topic"] == "实验"
    assert result["core_demands"] == ["核对结果"]
    assert set(result) == {
        "current_topic",
        "core_demands",
        "confirmed_conclusions",
        "unresolved_questions",
        "context_summary",
    }


def test_invalid_memory_json_is_rejected_instead_of_being_silently_persisted() -> None:
    service = ConversationService(FakeClient("not-json"))
    with pytest.raises(ValueError, match="JSON"):
        service.extract_incremental_conclusions(recent_history="用户：继续")
