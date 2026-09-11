"""Paper-scoped working-memory and recent-history policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .storage import SQLiteMemoryRepository, empty_working_memory


class MemoryConversationClient(Protocol):
    def update_working_memory(self, *, current_memory: dict[str, Any], recent_history: str) -> dict[str, Any]: ...

    def consolidate_long_term_memory(
        self,
        *,
        working_memory: dict[str, Any],
        existing_preferences: list[str],
        existing_conclusions: list[str],
    ) -> dict[str, list[str]]: ...

    def extract_incremental_conclusions(self, *, recent_history: str) -> dict[str, list[str]]: ...


@dataclass(frozen=True)
class WorkingMemoryUpdate:
    memory: dict[str, Any]
    user_message_count: int
    updated: bool


def scope_key_for_papers(paper_ids: Iterable[str]) -> str:
    """Return a stable scope key without leaking one paper into another."""
    normalized = tuple(sorted({item.strip() for item in paper_ids if isinstance(item, str) and item.strip()}))
    if not normalized:
        return "general"
    if len(normalized) == 1:
        return f"paper:{normalized[0]}"
    digest = hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:16]
    return f"crossdoc:{digest}"


def messages_in_scope(messages: Sequence[dict[str, Any]], scope_key: str) -> tuple[dict[str, Any], ...]:
    """Select only messages explicitly belonging to the active scope.

    Old messages without a scope are readable in the general scope only. This
    conservative rule avoids assigning historical paper claims to a new paper.
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        message_scope = message.get("paper_scope_key")
        if message_scope == scope_key or (scope_key == "general" and not message_scope):
            result.append(message)
    return tuple(result)


def memory_eligible_messages(
    messages: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Exclude failed or incomplete turns from rewrite and memory updates."""
    return tuple(
        message
        for message in messages
        if message.get("eligible_for_memory", True) is not False and message.get("status", "completed") != "failed"
    )


def format_recent_history(
    messages: Sequence[dict[str, Any]], scope_key: str, *, limit: int = 6, max_chars: int = 300
) -> str:
    selected = memory_eligible_messages(messages_in_scope(messages, scope_key))[-limit:]
    if not selected:
        return "无近期对话。"
    rows = []
    for message in selected:
        role = "用户" if message.get("role") == "user" else "AI"
        rows.append(f"{role}: {str(message.get('content') or '')[:max_chars]}")
    return "\n".join(rows)


@dataclass
class ScopedMemoryService:
    repository: SQLiteMemoryRepository

    def load(self, *, user_id: str, session_id: str, scope_key: str) -> dict[str, Any]:
        return self.repository.load_scoped_working_memory(user_id, session_id, scope_key)

    def save(
        self,
        *,
        user_id: str,
        session_id: str,
        scope_key: str,
        memory: dict[str, Any] | None,
    ) -> None:
        self.repository.save_scoped_working_memory(user_id, session_id, scope_key, memory or empty_working_memory())

    def generation_context(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_ids: tuple[str, ...],
        recent_history: str,
    ) -> tuple[str, str]:
        scope_key = scope_key_for_papers(paper_ids)
        working = self.load(user_id=user_id, session_id=session_id, scope_key=scope_key)
        memories = self.repository.list_relevant_long_term_memories(
            user_id,
            session_id,
            source_docs=paper_ids,
            limit=30,
        )
        long_term = "\n".join(f"- {item.memory_type}: {item.content}" for item in memories) or "无相关历史记忆。"
        return long_term, json.dumps(working, ensure_ascii=False)

    def maybe_update_working_memory(
        self,
        *,
        user_id: str,
        session_id: str,
        scope_key: str,
        messages: Sequence[dict[str, Any]],
        current_memory: dict[str, Any],
        last_update_count: int,
        conversation: MemoryConversationClient,
    ) -> WorkingMemoryUpdate:
        scoped = memory_eligible_messages(messages_in_scope(messages, scope_key))
        user_count = sum(1 for message in scoped if message.get("role") == "user")
        total_chars = sum(len(str(message.get("content") or "")) for message in scoped)
        should_update = user_count - last_update_count >= 3 or (total_chars > 2000 and user_count > last_update_count)
        if not should_update:
            return WorkingMemoryUpdate(current_memory, user_count, False)
        history = format_recent_history(scoped, scope_key, limit=8, max_chars=1000)
        updated = conversation.update_working_memory(
            current_memory=current_memory,
            recent_history=history,
        )
        self.save(
            user_id=user_id,
            session_id=session_id,
            scope_key=scope_key,
            memory=updated,
        )
        return WorkingMemoryUpdate(updated, user_count, True)

    def settle_long_term_memory(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_ids: tuple[str, ...],
        working_memory: dict[str, Any],
        conversation: MemoryConversationClient,
    ) -> None:
        memories = self.repository.list_relevant_long_term_memories(
            user_id,
            session_id,
            source_docs=paper_ids,
            limit=100,
        )
        consolidated = conversation.consolidate_long_term_memory(
            working_memory=working_memory,
            existing_preferences=[item.content for item in memories if item.memory_type == "user_preference"],
            existing_conclusions=[item.content for item in memories if item.memory_type == "session_conclusion"],
        )
        records = [("user_preference", item, 5) for item in consolidated["user_preferences"]]
        records.extend(("session_conclusion", item, 4) for item in consolidated["session_conclusions"])
        records.extend(("unresolved_question", item, 3) for item in consolidated["unresolved_questions"])
        self.repository.replace_memory_types(
            user_id,
            session_id,
            ["user_preference", "session_conclusion", "unresolved_question"],
            records,
            source_doc=paper_ids[0] if len(paper_ids) == 1 else None,
        )

    def maybe_extract_incremental_conclusions(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_ids: tuple[str, ...],
        scope_key: str,
        messages: Sequence[dict[str, Any]],
        last_turn: int,
        conversation: MemoryConversationClient,
    ) -> int:
        scoped = memory_eligible_messages(messages_in_scope(messages, scope_key))
        user_count = sum(1 for message in scoped if message.get("role") == "user")
        if user_count - last_turn < 5:
            return last_turn
        history = format_recent_history(scoped, scope_key, limit=10, max_chars=1000)
        extracted = conversation.extract_incremental_conclusions(recent_history=history)
        source_doc = paper_ids[0] if len(paper_ids) == 1 else None
        for conclusion in extracted["new_conclusions"]:
            self.repository.add_long_term_memory(
                user_id,
                session_id,
                "session_conclusion",
                conclusion,
                source_doc=source_doc,
                priority=4,
            )
        for issue in extracted["unresolved_issues"]:
            self.repository.add_long_term_memory(
                user_id,
                session_id,
                "unresolved_question",
                issue,
                source_doc=source_doc,
                priority=3,
            )
        return user_count
