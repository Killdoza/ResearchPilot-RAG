"""不依赖 Gold 标签的论文 Scope 解析基础设施。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .evaluation import CorpusManifest
from .query_plan import ScopeSource
from .schemas import SchemaValidationError


def _normalize(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _identifiers(values: Iterable[str], label: str) -> tuple[str, ...]:
    result = tuple(value.strip() for value in values if isinstance(value, str) and value.strip())
    if len(set(result)) != len(result):
        raise SchemaValidationError(f"{label} 不能重复。")
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", value) for value in result):
        raise SchemaValidationError(f"{label} 包含非法 paper_key。")
    return result


@dataclass(frozen=True)
class PaperScopeEntry:
    """运行时可访问论文的最小目录信息，不含评测答案或 Evidence。"""

    paper_key: str
    title: str
    source_name: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifiers((self.paper_key,), "paper_key")
        if not isinstance(self.title, str) or not self.title.strip():
            raise SchemaValidationError("title 不能为空。")
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise SchemaValidationError("source_name 不能为空。")
        if any(not isinstance(alias, str) or not alias.strip() for alias in self.aliases):
            raise SchemaValidationError("aliases 必须由非空字符串组成。")

    @property
    def normalized_aliases(self) -> tuple[str, ...]:
        values = {
            _normalize(self.paper_key.replace("_", " ")),
            _normalize(self.title),
            _normalize(self.source_name.rsplit(".", maxsplit=1)[0]),
        }
        short_title = self.title.split(":", maxsplit=1)[0]
        values.add(_normalize(short_title))
        values.update(_normalize(alias) for alias in self.aliases)
        return tuple(sorted(value for value in values if len(value) >= 4))


@dataclass(frozen=True)
class ScopeResolution:
    paper_keys: tuple[str, ...]
    source: ScopeSource
    fallback_reason: str | None = None
    matched_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifiers(self.paper_keys, "paper_keys")
        if not isinstance(self.source, ScopeSource):
            raise SchemaValidationError("source 必须是 ScopeSource。")
        if self.source == ScopeSource.NONE:
            if self.paper_keys:
                raise SchemaValidationError("无 Scope 来源时不能返回论文。")
            if not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip():
                raise SchemaValidationError("Scope 不足必须提供 fallback_reason。")
        elif not self.paper_keys:
            raise SchemaValidationError("已解析的 Scope 必须至少包含一篇论文。")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip()
        ):
            raise SchemaValidationError("fallback_reason 必须为非空字符串或 None。")


class PaperScopeCatalog:
    """从用户可访问论文范围中解析标题、文件名和显式别名。"""

    def __init__(self, entries: Iterable[PaperScopeEntry]) -> None:
        self._entries = {entry.paper_key: entry for entry in entries}
        if not self._entries:
            raise SchemaValidationError("PaperScopeCatalog 至少需要一篇论文。")

    @classmethod
    def from_corpus(cls, corpus: CorpusManifest) -> PaperScopeCatalog:
        return cls(
            PaperScopeEntry(
                paper_key=paper.paper_key,
                title=paper.title,
                source_name=paper.local_filename,
            )
            for paper in corpus.papers
        )

    @property
    def paper_keys(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def resolve(
        self,
        *,
        question: str,
        available_paper_keys: Iterable[str],
        user_selected_paper_keys: Iterable[str] = (),
    ) -> ScopeResolution:
        if not isinstance(question, str) or not question.strip():
            raise SchemaValidationError("question 不能为空。")
        available = _identifiers(available_paper_keys, "available_paper_keys")
        selected = _identifiers(user_selected_paper_keys, "user_selected_paper_keys")
        unknown_available = set(available) - self._entries.keys()
        if unknown_available:
            raise SchemaValidationError(f"available_paper_keys 不在 Catalog 中：{sorted(unknown_available)}")
        if not available:
            return ScopeResolution((), ScopeSource.NONE, "scope_insufficient:no_available_papers")
        unknown_selected = set(selected) - set(available)
        if unknown_selected:
            raise SchemaValidationError(f"user_selected_paper_keys 不在当前会话 Scope 中：{sorted(unknown_selected)}")
        if selected:
            return ScopeResolution(selected, ScopeSource.USER_SELECTED)

        normalized_question = _normalize(question)
        matched_keys: list[str] = []
        matched_aliases: list[str] = []
        alias_owners: dict[str, set[str]] = {}
        for paper_key in available:
            entry = self._entries[paper_key]
            aliases = [alias for alias in entry.normalized_aliases if alias in normalized_question]
            if aliases:
                matched_keys.append(paper_key)
                matched_aliases.append(aliases[0])
                for alias in aliases:
                    alias_owners.setdefault(alias, set()).add(paper_key)
        if not matched_keys:
            return ScopeResolution((), ScopeSource.NONE, "scope_insufficient:no_title_or_alias_match")
        # A single ambiguous title/alias is not evidence that the user wants a
        # cross-document comparison.  Returning both papers here would let a
        # later Coverage Gate treat an unresolved title collision as a valid
        # comparison scope, so stop explicitly and ask for a selection instead.
        ambiguous_aliases = sorted(alias for alias, owners in alias_owners.items() if len(owners) > 1)
        if ambiguous_aliases:
            return ScopeResolution(
                (),
                ScopeSource.NONE,
                "scope_insufficient:ambiguous_title_or_alias",
                matched_aliases=tuple(ambiguous_aliases),
            )
        return ScopeResolution(tuple(matched_keys), ScopeSource.TITLE_RESOLVED, matched_aliases=tuple(matched_aliases))
