"""Query rewrite and non-paper conversation outside the UI layer."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .generation import ModelUsageSnapshot, TextGenerationClient, usage_delta


@dataclass
class ConversationService:
    client: TextGenerationClient
    last_rewrite_usage: ModelUsageSnapshot = field(default_factory=ModelUsageSnapshot, init=False)
    last_rewrite_cost: float | None = field(default=None, init=False)

    def rewrite_query(self, *, question: str, recent_history: str) -> str:
        """Resolve references for retrieval; routing must still use the original question."""
        if not recent_history.strip() or recent_history.strip() == "无近期对话。":
            self.last_rewrite_usage = ModelUsageSnapshot()
            self.last_rewrite_cost = None
            return question.strip()
        system_prompt = (
            "你是学术检索问题重写器。根据同一论文作用域内的近期对话，"
            "把最新问题改写成独立、完整的检索问题。只输出改写后的问题；"
            "不要回答，不要引入历史中没有的论文或事实。"
        )
        try:
            snapshot = getattr(self.client, "usage_snapshot", None)
            before = snapshot() if callable(snapshot) else ModelUsageSnapshot()
            rewritten = self.client.generate(
                system_prompt=system_prompt,
                user_prompt=f"【同一作用域近期对话】\n{recent_history}\n\n【最新问题】\n{question}",
            ).strip()
            after = snapshot() if callable(snapshot) else before
            self.last_rewrite_usage = usage_delta(before, after)
            estimate = getattr(self.client, "estimate_cost", None)
            self.last_rewrite_cost = estimate(self.last_rewrite_usage) if callable(estimate) else None
        except Exception:
            self.last_rewrite_usage = ModelUsageSnapshot()
            self.last_rewrite_cost = None
            return question.strip()
        return rewritten or question.strip()

    def general_chat(
        self,
        *,
        question: str,
        recent_history: str,
        preference_memory: str = "",
        on_text_delta: Callable[[str], None] | None = None,
    ) -> str:
        system_prompt = (
            "你是 ResearchPilot。当前问题不需要论文检索，请自然简洁地回答。"
            "不得伪造论文 Citation；长期记忆只可用于语言和格式偏好。"
        )
        user_prompt = (
            f"【偏好】\n{preference_memory or '无'}\n\n【近期对话】\n{recent_history or '无'}\n\n【问题】\n{question}"
        )
        stream_generate = getattr(self.client, "generate_stream", None)
        if on_text_delta is not None and callable(stream_generate):
            return stream_generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                on_text_delta=on_text_delta,
            )
        return self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    def summarize_chat(self, *, recent_history: str) -> str:
        if not recent_history.strip() or recent_history.strip() == "无近期对话。":
            return "当前没有可供总结的近期对话记录。"
        return self.client.generate(
            system_prompt="仅总结给出的对话脉络、已确认结论和待解决问题，不引入论文外事实。",
            user_prompt=recent_history,
        )

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型没有返回 JSON 对象。")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("模型返回的记忆结果不是 JSON 对象。")
        return payload

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    def update_working_memory(
        self,
        *,
        current_memory: dict[str, Any],
        recent_history: str,
    ) -> dict[str, Any]:
        """Summarize one paper scope without exposing prompt logic to the UI."""
        result = self.client.generate(
            system_prompt=(
                "你是 ResearchPilot 的作用域工作记忆整理器。只整理给出的同一论文作用域对话，"
                "不得引入其他论文或外部事实。只返回合法 JSON。"
            ),
            user_prompt=(
                "请合并当前工作记忆与近期对话，严格返回以下字段：current_topic（字符串）、"
                "core_demands（字符串数组）、confirmed_conclusions（字符串数组）、"
                "unresolved_questions（字符串数组）、context_summary（字符串）。\n\n"
                f"【当前工作记忆】\n{json.dumps(current_memory, ensure_ascii=False)}\n\n"
                f"【同一作用域近期对话】\n{recent_history}"
            ),
        )
        payload = self._json_object(result)
        return {
            "current_topic": str(payload.get("current_topic") or "")[:200],
            "core_demands": self._string_list(payload.get("core_demands")),
            "confirmed_conclusions": self._string_list(payload.get("confirmed_conclusions")),
            "unresolved_questions": self._string_list(payload.get("unresolved_questions")),
            "context_summary": str(payload.get("context_summary") or "")[:1000],
        }

    def consolidate_long_term_memory(
        self,
        *,
        working_memory: dict[str, Any],
        existing_preferences: list[str],
        existing_conclusions: list[str],
    ) -> dict[str, list[str]]:
        """Consolidate memory records while preserving their explicit source scope."""
        result = self.client.generate(
            system_prompt=(
                "你是 ResearchPilot 的长期记忆整理器。只合并用户偏好、已确认结论和遗留问题；"
                "不要把用户观点改写成论文事实。只返回合法 JSON。"
            ),
            user_prompt=(
                "严格返回 user_preferences、session_conclusions、unresolved_questions 三个字符串数组。\n\n"
                f"【已有用户偏好】\n{json.dumps(existing_preferences, ensure_ascii=False)}\n"
                f"【同论文作用域既有结论】\n{json.dumps(existing_conclusions, ensure_ascii=False)}\n"
                f"【当前工作记忆】\n{json.dumps(working_memory, ensure_ascii=False)}"
            ),
        )
        payload = self._json_object(result)
        return {
            "user_preferences": self._string_list(payload.get("user_preferences")),
            "session_conclusions": self._string_list(payload.get("session_conclusions")),
            "unresolved_questions": self._string_list(payload.get("unresolved_questions")),
        }

    def extract_incremental_conclusions(self, *, recent_history: str) -> dict[str, list[str]]:
        """Extract only confirmed outcomes from the latest same-scope turns."""
        result = self.client.generate(
            system_prompt=(
                "你是 ResearchPilot 的研究纪要整理器。只提取对话中已确认的结论和仍未解决的问题；"
                "探讨中的猜测不得写成结论。只返回合法 JSON。"
            ),
            user_prompt=(
                "严格返回 new_conclusions 和 unresolved_issues 两个字符串数组。\n\n"
                f"【同一作用域增量对话】\n{recent_history}"
            ),
        )
        payload = self._json_object(result)
        return {
            "new_conclusions": self._string_list(payload.get("new_conclusions")),
            "unresolved_issues": self._string_list(payload.get("unresolved_issues")),
        }
