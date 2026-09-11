"""Isolated, transactional persistence for ResearchPilot sessions and memory.

The legacy demo wrote JSON and SQL directly from the Streamlit page, sharing a
global SQLite cursor across users.  This module is the single persistence
boundary for the new application.  It deliberately uses new ``rp_*`` tables so
the legacy tables remain an untouched migration source.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .schemas import AnswerClaim, Citation


class SessionDataCorrupted(ValueError):
    """Raised when a persisted session JSON file cannot be safely consumed."""


@dataclass(frozen=True)
class SessionSnapshot:
    """The UI state persisted for a single conversation."""

    session_id: str
    messages: list[dict[str, Any]]
    active_pdf_names: list[str]
    active_documents: list[dict[str, Any]]
    selected_paper_id: str | None
    session_title: str | None = None


@dataclass(frozen=True)
class LongTermMemory:
    """A user-scoped, active or inactive long-term memory record."""

    id: int
    user_id: str
    memory_type: str
    content: str
    source_session: str
    source_doc: str | None
    priority: int
    is_active: bool
    last_updated: str


@dataclass(frozen=True)
class PaperAnnotation:
    """A user note that remains explicitly separate from paper facts."""

    annotation_id: int | None
    user_id: str
    session_id: str
    paper_id: str
    chunk_id: str
    content: str
    note_type: str = "comment"
    relation: str = "comments_on"
    verification_status: str = "user_note"
    created_at: str | None = None
    anchor_page: int | None = None
    section_path: tuple[str, ...] = ()

    @property
    def anchor_chunk_id(self) -> str:
        """Explicit alias retained while the SQLite column stays backward compatible."""
        return self.chunk_id


@dataclass(frozen=True)
class ResearchNote:
    """A saved answer with its question and immutable citation snapshot."""

    note_id: str
    user_id: str
    session_id: str
    paper_id: str
    question: str
    answer_text: str
    citations: tuple[Citation, ...]
    created_at: str | None = None


@dataclass(frozen=True)
class PaperArtifactVersion:
    """Version boundary for cached paper-level generated artifacts.

    A paper ID is content-derived, but the same paper can be parsed or chunked
    differently over the lifetime of the product.  Persisting these dimensions
    makes a cache hit explicit rather than silently reusing an incompatible
    summary or metadata extraction.
    """

    parser_name: str
    parser_version: str
    chunker_version: str
    prompt_version: str
    model_name: str

    def __post_init__(self) -> None:
        for name, value in (
            ("parser_name", self.parser_name),
            ("parser_version", self.parser_version),
            ("chunker_version", self.chunker_version),
            ("prompt_version", self.prompt_version),
            ("model_name", self.model_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class PaperMetadata:
    """Evidence-backed, paper-scoped bibliographic metadata cache."""

    user_id: str
    session_id: str
    paper_id: str
    source_name: str
    answer_text: str
    citations: tuple[Citation, ...]
    version: PaperArtifactVersion
    title: str | None = None
    authors: tuple[str, ...] = ()
    affiliations: tuple[str, ...] = ()
    venue: str | None = None
    year: str | None = None
    doi: str | None = None
    extraction_method: str = "front_matter_json_v1"
    field_confidence: dict[str, float] | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("user_id", self.user_id),
            ("session_id", self.session_id),
            ("paper_id", self.paper_id),
            ("source_name", self.source_name),
            ("answer_text", self.answer_text),
            ("extraction_method", self.extraction_method),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.title is not None and not self.title.strip():
            raise ValueError("title must be non-empty when present")
        if any(not isinstance(item, str) or not item.strip() for item in self.authors):
            raise ValueError("authors must contain non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.affiliations):
            raise ValueError("affiliations must contain non-empty strings")
        if self.venue is not None and not self.venue.strip():
            raise ValueError("venue must be non-empty when present")
        if self.year is not None and not self.year.strip():
            raise ValueError("year must be non-empty when present")
        if self.doi is not None and not self.doi.strip():
            raise ValueError("doi must be non-empty when present")
        if any(not isinstance(item, Citation) for item in self.citations):
            raise ValueError("citations must contain Citation values")
        if self.field_confidence is not None:
            for field_name, score in self.field_confidence.items():
                if not isinstance(field_name, str) or not field_name.strip():
                    raise ValueError("field confidence names must be non-empty")
                if not isinstance(score, int | float) or not 0 <= float(score) <= 1:
                    raise ValueError("field confidence must be between 0 and 1")


@dataclass(frozen=True)
class CachedPaperSummary:
    """Versioned, final answer artifact for one full-paper summary request."""

    user_id: str
    session_id: str
    paper_id: str
    source_name: str
    answer_text: str
    citations: tuple[Citation, ...]
    version: PaperArtifactVersion
    summary_intent: str
    request_fingerprint: str
    claims: tuple[AnswerClaim, ...] = ()
    contract_version: str = "legacy"
    numeric_fidelity_passed: bool | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("user_id", self.user_id),
            ("session_id", self.session_id),
            ("paper_id", self.paper_id),
            ("source_name", self.source_name),
            ("answer_text", self.answer_text),
            ("summary_intent", self.summary_intent),
            ("request_fingerprint", self.request_fingerprint),
            ("contract_version", self.contract_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if any(not isinstance(item, Citation) for item in self.citations):
            raise ValueError("citations must contain Citation values")
        if any(not isinstance(item, AnswerClaim) for item in self.claims):
            raise ValueError("claims must contain AnswerClaim values")
        if self.numeric_fidelity_passed is not None and not isinstance(self.numeric_fidelity_passed, bool):
            raise ValueError("numeric_fidelity_passed must be bool or None")


def empty_working_memory() -> dict[str, Any]:
    """Return the canonical empty working-memory shape used by the UI."""
    return {
        "current_topic": "",
        "core_demands": [],
        "confirmed_conclusions": [],
        "unresolved_questions": [],
        "context_summary": "",
    }


class SQLiteMemoryRepository:
    """SQLite persistence with explicit user and session boundaries.

    A short-lived connection is used for every operation.  It prevents a
    Streamlit rerun or one failed transaction from leaving a global cursor in a
    broken state, while WAL keeps local read/write concurrency practical.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _validate_scope(self, user_id: str, session_id: str) -> None:
        self.settings.session_file_path(user_id, session_id)

    def _connect(self) -> sqlite3.Connection:
        self.settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.settings.memory_db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize_schema(self) -> None:
        """Create only ResearchPilot-owned tables; legacy tables are untouched."""
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rp_working_memory (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        current_topic TEXT NOT NULL DEFAULT '',
                        core_demands TEXT NOT NULL DEFAULT '[]',
                        confirmed_conclusions TEXT NOT NULL DEFAULT '[]',
                        unresolved_questions TEXT NOT NULL DEFAULT '[]',
                        context_summary TEXT NOT NULL DEFAULT '',
                        last_updated TEXT NOT NULL,
                        PRIMARY KEY (user_id, session_id)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rp_scoped_working_memory (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        scope_key TEXT NOT NULL,
                        current_topic TEXT NOT NULL DEFAULT '',
                        core_demands TEXT NOT NULL DEFAULT '[]',
                        confirmed_conclusions TEXT NOT NULL DEFAULT '[]',
                        unresolved_questions TEXT NOT NULL DEFAULT '[]',
                        context_summary TEXT NOT NULL DEFAULT '',
                        last_updated TEXT NOT NULL,
                        PRIMARY KEY (user_id, session_id, scope_key)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rp_long_term_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_session TEXT NOT NULL,
                        source_doc TEXT,
                        priority INTEGER NOT NULL DEFAULT 1,
                        is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                        last_updated TEXT NOT NULL,
                        legacy_source_id TEXT UNIQUE
                    )
                    """
                )
                columns = {row["name"] for row in connection.execute("PRAGMA table_info(rp_long_term_memory)")}
                if "legacy_source_id" not in columns:
                    connection.execute("ALTER TABLE rp_long_term_memory ADD COLUMN legacy_source_id TEXT")
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rp_ltm_user_active_updated
                    ON rp_long_term_memory (user_id, is_active, last_updated DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rp_ltm_user_document
                    ON rp_long_term_memory (user_id, source_doc, is_active, last_updated DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_rp_ltm_legacy_source
                    ON rp_long_term_memory (legacy_source_id)
                    WHERE legacy_source_id IS NOT NULL
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rp_paper_metadata (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        paper_id TEXT NOT NULL,
                        source_name TEXT NOT NULL DEFAULT '',
                        parser_name TEXT NOT NULL,
                        parser_version TEXT NOT NULL,
                        chunker_version TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        title TEXT,
                        authors_json TEXT NOT NULL DEFAULT '[]',
                        affiliations_json TEXT NOT NULL DEFAULT '[]',
                        venue TEXT,
                        year TEXT,
                        doi TEXT,
                        extraction_method TEXT NOT NULL,
                        field_confidence_json TEXT NOT NULL DEFAULT '{}',
                        citations_json TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (
                            user_id, session_id, paper_id, parser_name,
                            parser_version, chunker_version, prompt_version, model_name
                        )
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rp_paper_summary_cache (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        paper_id TEXT NOT NULL,
                        source_name TEXT NOT NULL DEFAULT '',
                        parser_name TEXT NOT NULL,
                        parser_version TEXT NOT NULL,
                        chunker_version TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        citations_json TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (
                            user_id, session_id, paper_id, parser_name,
                            parser_version, chunker_version, prompt_version, model_name
                        )
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rp_paper_summary_cache_v2 (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        paper_id TEXT NOT NULL,
                        source_name TEXT NOT NULL DEFAULT '',
                        parser_name TEXT NOT NULL,
                        parser_version TEXT NOT NULL,
                        chunker_version TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        summary_intent TEXT NOT NULL,
                        request_fingerprint TEXT NOT NULL,
                        citations_json TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        claims_json TEXT NOT NULL DEFAULT '[]',
                        contract_version TEXT NOT NULL DEFAULT 'legacy',
                        numeric_fidelity_passed INTEGER,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (
                            user_id, session_id, paper_id, parser_name,
                            parser_version, chunker_version, prompt_version, model_name,
                            summary_intent, request_fingerprint
                        )
                    )
                    """
                )
                summary_v2_columns = {
                    row["name"] for row in connection.execute("PRAGMA table_info(rp_paper_summary_cache_v2)")
                }
                for column, definition in (
                    ("claims_json", "TEXT NOT NULL DEFAULT '[]'"),
                    ("contract_version", "TEXT NOT NULL DEFAULT 'legacy'"),
                    ("numeric_fidelity_passed", "INTEGER"),
                ):
                    if column not in summary_v2_columns:
                        connection.execute(f"ALTER TABLE rp_paper_summary_cache_v2 ADD COLUMN {column} {definition}")
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rp_paper_metadata_scope
                    ON rp_paper_metadata (user_id, session_id, paper_id, updated_at DESC)
                    """
                )
                for table_name in ("rp_paper_metadata", "rp_paper_summary_cache"):
                    table_columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})")}
                    if "source_name" not in table_columns:
                        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN source_name TEXT NOT NULL DEFAULT ''")
                    if table_name == "rp_paper_metadata" and "doi" not in table_columns:
                        connection.execute("ALTER TABLE rp_paper_metadata ADD COLUMN doi TEXT")
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rp_paper_summary_scope
                    ON rp_paper_summary_cache (user_id, session_id, paper_id, updated_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rp_paper_summary_v2_scope
                    ON rp_paper_summary_cache_v2
                        (user_id, session_id, paper_id, summary_intent, updated_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rp_annotations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        paper_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        note_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        verification_status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        anchor_page INTEGER,
                        section_path_json TEXT NOT NULL DEFAULT '[]'
                    )
                    """
                )
                annotation_columns = {row["name"] for row in connection.execute("PRAGMA table_info(rp_annotations)")}
                if "anchor_page" not in annotation_columns:
                    connection.execute("ALTER TABLE rp_annotations ADD COLUMN anchor_page INTEGER")
                if "section_path_json" not in annotation_columns:
                    connection.execute(
                        "ALTER TABLE rp_annotations ADD COLUMN section_path_json TEXT NOT NULL DEFAULT '[]'"
                    )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rp_annotations_scope
                    ON rp_annotations (user_id, session_id, paper_id, chunk_id, created_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rp_research_notes (
                        note_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        paper_id TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        citations_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rp_research_notes_scope
                    ON rp_research_notes (user_id, session_id, paper_id, created_at DESC)
                    """
                )
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield an explicit transaction and rollback every exception."""
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _json_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _decode_list(value: str | None) -> list[Any]:
        if not value:
            return []
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
        return decoded if isinstance(decoded, list) else []

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _validate_paper_scope(self, user_id: str, session_id: str, paper_id: str) -> None:
        self._validate_scope(user_id, session_id)
        from .config import validate_storage_id

        validate_storage_id(paper_id, "paper_id")

    @staticmethod
    def _citation_payload(citations: Sequence[Citation]) -> str:
        return json.dumps(
            [
                {
                    "paper_id": item.paper_id,
                    "chunk_id": item.chunk_id,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "section_path": list(item.section_path),
                    "evidence": item.evidence,
                    "confidence": item.confidence,
                }
                for item in citations
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _claim_payload(claims: Sequence[AnswerClaim]) -> str:
        return json.dumps(
            [
                {
                    "claim_id": item.claim_id,
                    "text": item.text,
                    "citation_ids": list(item.citation_ids),
                    "support_status": item.support_status,
                    "numeric_values": list(item.numeric_values),
                }
                for item in claims
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_citations(value: str | None) -> tuple[Citation, ...] | None:
        try:
            decoded = json.loads(value or "[]")
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, list):
            return None
        citations: list[Citation] = []
        try:
            for item in decoded:
                if not isinstance(item, dict):
                    return None
                section_path = item.get("section_path", [])
                if not isinstance(section_path, list) or not all(
                    isinstance(part, str) and part.strip() for part in section_path
                ):
                    return None
                citations.append(
                    Citation(
                        paper_id=str(item["paper_id"]),
                        chunk_id=str(item["chunk_id"]),
                        page_start=int(item["page_start"]),
                        page_end=int(item["page_end"]),
                        section_path=tuple(section_path),
                        evidence=str(item["evidence"]),
                        confidence=(float(item["confidence"]) if item.get("confidence") is not None else None),
                    )
                )
        except (KeyError, TypeError, ValueError):
            return None
        return tuple(citations)

    @staticmethod
    def _decode_claims(value: str | None) -> tuple[AnswerClaim, ...] | None:
        try:
            decoded = json.loads(value or "[]")
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, list):
            return None
        claims: list[AnswerClaim] = []
        try:
            for item in decoded:
                if not isinstance(item, dict):
                    return None
                citation_ids = item.get("citation_ids", [])
                numeric_values = item.get("numeric_values", [])
                if not isinstance(citation_ids, list) or not isinstance(numeric_values, list):
                    return None
                claims.append(
                    AnswerClaim(
                        claim_id=str(item["claim_id"]),
                        text=str(item["text"]),
                        citation_ids=tuple(str(value) for value in citation_ids),
                        support_status=str(item["support_status"]),
                        numeric_values=tuple(str(value) for value in numeric_values),
                    )
                )
        except (KeyError, TypeError, ValueError):
            return None
        return tuple(claims)

    @staticmethod
    def _decode_mapping(value: str | None) -> dict[str, Any] | None:
        try:
            decoded = json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError):
            return None
        return decoded if isinstance(decoded, dict) else None

    @staticmethod
    def _decode_string_tuple(value: str | None) -> tuple[str, ...] | None:
        decoded = SQLiteMemoryRepository._decode_list(value)
        if not all(isinstance(item, str) and item.strip() for item in decoded):
            return None
        return tuple(item.strip() for item in decoded)

    def save_working_memory(self, user_id: str, session_id: str, memory: dict[str, Any]) -> None:
        self._validate_scope(user_id, session_id)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO rp_working_memory
                    (user_id, session_id, current_topic, core_demands,
                     confirmed_conclusions, unresolved_questions,
                     context_summary, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                    current_topic=excluded.current_topic,
                    core_demands=excluded.core_demands,
                    confirmed_conclusions=excluded.confirmed_conclusions,
                    unresolved_questions=excluded.unresolved_questions,
                    context_summary=excluded.context_summary,
                    last_updated=excluded.last_updated
                """,
                (
                    user_id,
                    session_id,
                    str(memory.get("current_topic") or ""),
                    json.dumps(self._json_list(memory.get("core_demands")), ensure_ascii=False),
                    json.dumps(self._json_list(memory.get("confirmed_conclusions")), ensure_ascii=False),
                    json.dumps(self._json_list(memory.get("unresolved_questions")), ensure_ascii=False),
                    str(memory.get("context_summary") or ""),
                    self._now(),
                ),
            )

    def save_scoped_working_memory(
        self,
        user_id: str,
        session_id: str,
        scope_key: str,
        memory: dict[str, Any],
    ) -> None:
        """Save WM under an explicit paper/cross-document/general scope."""
        self._validate_scope(user_id, session_id)
        if not isinstance(scope_key, str) or not scope_key.strip():
            raise ValueError("scope_key must not be empty")
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO rp_scoped_working_memory
                    (user_id, session_id, scope_key, current_topic, core_demands,
                     confirmed_conclusions, unresolved_questions, context_summary, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id, scope_key) DO UPDATE SET
                    current_topic=excluded.current_topic,
                    core_demands=excluded.core_demands,
                    confirmed_conclusions=excluded.confirmed_conclusions,
                    unresolved_questions=excluded.unresolved_questions,
                    context_summary=excluded.context_summary,
                    last_updated=excluded.last_updated
                """,
                (
                    user_id,
                    session_id,
                    scope_key.strip(),
                    str(memory.get("current_topic") or ""),
                    json.dumps(self._json_list(memory.get("core_demands")), ensure_ascii=False),
                    json.dumps(self._json_list(memory.get("confirmed_conclusions")), ensure_ascii=False),
                    json.dumps(self._json_list(memory.get("unresolved_questions")), ensure_ascii=False),
                    str(memory.get("context_summary") or ""),
                    self._now(),
                ),
            )

    def load_scoped_working_memory(self, user_id: str, session_id: str, scope_key: str) -> dict[str, Any]:
        """Load only the requested WM scope; never borrow another paper's WM."""
        self._validate_scope(user_id, session_id)
        if not isinstance(scope_key, str) or not scope_key.strip():
            raise ValueError("scope_key must not be empty")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT current_topic, core_demands, confirmed_conclusions,
                       unresolved_questions, context_summary
                FROM rp_scoped_working_memory
                WHERE user_id=? AND session_id=? AND scope_key=?
                """,
                (user_id, session_id, scope_key.strip()),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            # Only the general scope may inherit the pre-P2.8.4 unscoped WM.
            return self.load_working_memory(user_id, session_id) if scope_key == "general" else empty_working_memory()
        return {
            "current_topic": str(row["current_topic"]),
            "core_demands": self._decode_list(row["core_demands"]),
            "confirmed_conclusions": self._decode_list(row["confirmed_conclusions"]),
            "unresolved_questions": self._decode_list(row["unresolved_questions"]),
            "context_summary": str(row["context_summary"]),
        }

    def load_working_memory(self, user_id: str, session_id: str) -> dict[str, Any]:
        self._validate_scope(user_id, session_id)
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT current_topic, core_demands, confirmed_conclusions,
                       unresolved_questions, context_summary
                FROM rp_working_memory
                WHERE user_id=? AND session_id=?
                """,
                (user_id, session_id),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return empty_working_memory()
        return {
            "current_topic": row["current_topic"] or "",
            "core_demands": self._decode_list(row["core_demands"]),
            "confirmed_conclusions": self._decode_list(row["confirmed_conclusions"]),
            "unresolved_questions": self._decode_list(row["unresolved_questions"]),
            "context_summary": row["context_summary"] or "",
        }

    def add_long_term_memory(
        self,
        user_id: str,
        session_id: str,
        memory_type: str,
        content: str,
        *,
        source_doc: str | None = None,
        priority: int = 1,
    ) -> int:
        self._validate_scope(user_id, session_id)
        if not memory_type.strip() or not content.strip():
            raise ValueError("memory_type and content must not be empty")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO rp_long_term_memory
                    (user_id, memory_type, content, source_session, source_doc,
                     priority, is_active, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (user_id, memory_type, content, session_id, source_doc, priority, self._now()),
            )
            return int(cursor.lastrowid)

    def replace_memory_types(
        self,
        user_id: str,
        session_id: str,
        memory_types: Sequence[str],
        records: Sequence[tuple[str, str, int]],
        *,
        source_doc: str | None = None,
    ) -> None:
        """Atomically supersede named types inside their intended memory scope.

        ``user_preference`` is a user-global fact and therefore supersedes prior
        preferences across sessions. Research conclusions are session scoped by
        default, or paper scoped when ``source_doc`` carries the active paper ID.
        This prevents settling one paper from deactivating conclusions belonging
        to another paper.
        """
        self._validate_scope(user_id, session_id)
        types = tuple(item for item in memory_types if item)
        with self.transaction() as connection:
            if "user_preference" in types:
                connection.execute(
                    """
                    UPDATE rp_long_term_memory SET is_active=0
                    WHERE user_id=? AND memory_type='user_preference' AND is_active=1
                    """,
                    (user_id,),
                )
            research_types = tuple(item for item in types if item != "user_preference")
            if research_types:
                placeholders = ", ".join("?" for _ in research_types)
                if source_doc:
                    connection.execute(
                        f"""
                        UPDATE rp_long_term_memory SET is_active=0
                        WHERE user_id=? AND memory_type IN ({placeholders})
                          AND source_doc=? AND is_active=1
                        """,
                        (user_id, *research_types, source_doc),
                    )
                else:
                    connection.execute(
                        f"""
                        UPDATE rp_long_term_memory SET is_active=0
                        WHERE user_id=? AND memory_type IN ({placeholders})
                          AND source_session=? AND source_doc IS NULL AND is_active=1
                        """,
                        (user_id, *research_types, session_id),
                    )
            for memory_type, content, priority in records:
                if not content.strip():
                    continue
                connection.execute(
                    """
                    INSERT INTO rp_long_term_memory
                        (user_id, memory_type, content, source_session, source_doc, priority,
                         is_active, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        user_id,
                        memory_type,
                        content,
                        session_id,
                        None if memory_type == "user_preference" else source_doc,
                        priority,
                        self._now(),
                    ),
                )

    def get_active_memory_contents(self, user_id: str, session_id: str, memory_types: Sequence[str]) -> list[str]:
        self._validate_scope(user_id, session_id)
        types = tuple(item for item in memory_types if item)
        if not types:
            return []
        placeholders = ", ".join("?" for _ in types)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT content FROM rp_long_term_memory
                WHERE user_id=? AND is_active=1 AND memory_type IN ({placeholders})
                ORDER BY last_updated DESC
                """,
                (user_id, *types),
            ).fetchall()
        finally:
            connection.close()
        return [str(row["content"]) for row in rows]

    def list_active_long_term_memories(self, user_id: str, session_id: str, *, limit: int = 30) -> list[LongTermMemory]:
        self._validate_scope(user_id, session_id)
        if limit < 1:
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT id, user_id, memory_type, content, source_session, source_doc,
                       priority, is_active, last_updated
                FROM rp_long_term_memory
                WHERE user_id=? AND is_active=1
                ORDER BY last_updated DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        finally:
            connection.close()
        return [self._to_long_term_memory(row) for row in rows]

    def list_relevant_long_term_memories(
        self,
        user_id: str,
        session_id: str,
        *,
        source_docs: Sequence[str] = (),
        limit: int = 30,
    ) -> list[LongTermMemory]:
        """读取可安全注入当前论文问答的记忆。

        用户偏好允许跨会话复用；研究结论默认仅限当前会话。只有显式绑定到
        当前 ``paper_id`` 的结论才能跨会话复用。``document_insight`` 不在这里
        全局注入，它由命中的具体 Chunk 单独加载，避免另一篇论文的批注串入。
        """
        self._validate_scope(user_id, session_id)
        if limit < 1:
            return []
        scoped_docs = tuple(
            dict.fromkeys(item.strip() for item in source_docs if isinstance(item, str) and item.strip())
        )
        conditions = ["memory_type='user_preference'"]
        parameters: list[object] = [user_id]
        if scoped_docs:
            placeholders = ", ".join("?" for _ in scoped_docs)
            conditions.append(
                f"(source_doc IN ({placeholders}) AND memory_type IN ('session_conclusion', 'unresolved_question'))"
            )
            parameters.extend(scoped_docs)
        else:
            conditions.append(
                "(source_session=? AND source_doc IS NULL "
                "AND memory_type IN ('session_conclusion', 'unresolved_question'))"
            )
            parameters.append(session_id)
        parameters.append(limit)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT id, user_id, memory_type, content, source_session, source_doc,
                       priority, is_active, last_updated
                FROM rp_long_term_memory
                WHERE user_id=? AND is_active=1
                  AND ({" OR ".join(conditions)})
                ORDER BY priority DESC, last_updated DESC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        finally:
            connection.close()
        return [self._to_long_term_memory(row) for row in rows]

    def list_document_insights(self, user_id: str, session_id: str, source_doc: str) -> list[LongTermMemory]:
        self._validate_scope(user_id, session_id)
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT id, user_id, memory_type, content, source_session, source_doc,
                       priority, is_active, last_updated
                FROM rp_long_term_memory
                WHERE user_id=? AND memory_type='document_insight'
                  AND source_doc=? AND is_active=1
                ORDER BY last_updated DESC
                """,
                (user_id, source_doc),
            ).fetchall()
        finally:
            connection.close()
        return [self._to_long_term_memory(row) for row in rows]

    def add_annotation(self, annotation: PaperAnnotation) -> int:
        """Persist a user note without treating it as a paper-grounded fact."""
        self._validate_paper_scope(annotation.user_id, annotation.session_id, annotation.paper_id)
        if not annotation.chunk_id.strip() or not annotation.content.strip():
            raise ValueError("annotation chunk_id and content must not be empty")
        if annotation.relation not in {"comments_on", "contradicts", "supports", "questions"}:
            raise ValueError("unsupported annotation relation")
        if annotation.anchor_page is not None and annotation.anchor_page < 1:
            raise ValueError("annotation anchor_page must be positive")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO rp_annotations
                    (user_id, session_id, paper_id, chunk_id, note_type, content,
                     relation, verification_status, created_at, anchor_page, section_path_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation.user_id,
                    annotation.session_id,
                    annotation.paper_id,
                    annotation.chunk_id,
                    annotation.note_type,
                    annotation.content,
                    annotation.relation,
                    annotation.verification_status,
                    annotation.created_at or self._now(),
                    annotation.anchor_page,
                    json.dumps(list(annotation.section_path), ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list_annotations(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        chunk_ids: Sequence[str] = (),
    ) -> list[PaperAnnotation]:
        self._validate_paper_scope(user_id, session_id, paper_id)
        conditions = ["user_id=?", "session_id=?", "paper_id=?"]
        parameters: list[object] = [user_id, session_id, paper_id]
        normalized = tuple(dict.fromkeys(item for item in chunk_ids if item))
        if normalized:
            placeholders = ", ".join("?" for _ in normalized)
            conditions.append(f"chunk_id IN ({placeholders})")
            parameters.extend(normalized)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT * FROM rp_annotations WHERE {' AND '.join(conditions)} ORDER BY created_at DESC",
                tuple(parameters),
            ).fetchall()
        finally:
            connection.close()
        return [
            PaperAnnotation(
                annotation_id=int(row["id"]),
                user_id=str(row["user_id"]),
                session_id=str(row["session_id"]),
                paper_id=str(row["paper_id"]),
                chunk_id=str(row["chunk_id"]),
                note_type=str(row["note_type"]),
                content=str(row["content"]),
                relation=str(row["relation"]),
                verification_status=str(row["verification_status"]),
                created_at=str(row["created_at"]),
                anchor_page=(
                    int(row["anchor_page"]) if "anchor_page" in row.keys() and row["anchor_page"] is not None else None
                ),
                section_path=(
                    self._decode_string_tuple(row["section_path_json"]) or ()
                    if "section_path_json" in row.keys()
                    else ()
                ),
            )
            for row in rows
        ]

    def search_annotations(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        chunk_ids: Sequence[str] = (),
        pages: Sequence[int] = (),
        section_paths: Sequence[Sequence[str]] = (),
        query: str = "",
        limit: int = 3,
    ) -> list[PaperAnnotation]:
        """Rank paper-scoped notes without requiring an exact Chunk hit."""
        if limit < 1:
            return []
        notes = self.list_annotations(
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
        )
        chunk_set = {item for item in chunk_ids if item}
        page_set = {int(item) for item in pages if isinstance(item, int) and item > 0}
        section_set = {
            " > ".join(part.strip() for part in path if isinstance(part, str) and part.strip()).casefold()
            for path in section_paths
        }
        section_set.discard("")
        query_terms = set(re.findall(r"[a-z0-9_-]+|[\u4e00-\u9fff]", query.casefold()))

        def score(note: PaperAnnotation) -> tuple[float, str]:
            value = 0.0
            if note.chunk_id in chunk_set:
                value += 100.0
            note_section = " > ".join(note.section_path).casefold()
            if note_section and note_section in section_set:
                value += 35.0
            if note.anchor_page is not None and page_set:
                distance = min(abs(note.anchor_page - page) for page in page_set)
                if distance == 0:
                    value += 25.0
                elif distance == 1:
                    value += 12.0
            note_terms = set(re.findall(r"[a-z0-9_-]+|[\u4e00-\u9fff]", note.content.casefold()))
            if query_terms:
                value += 20.0 * len(query_terms & note_terms) / len(query_terms)
            return value, note.created_at or ""

        return sorted(notes, key=score, reverse=True)[:limit]

    def save_research_note(self, note: ResearchNote) -> None:
        self._validate_paper_scope(note.user_id, note.session_id, note.paper_id)
        if not note.note_id.strip() or not note.question.strip() or not note.answer_text.strip():
            raise ValueError("research note fields must not be empty")
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO rp_research_notes
                    (note_id, user_id, session_id, paper_id, question, answer_text,
                     citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    question=excluded.question,
                    answer_text=excluded.answer_text,
                    citations_json=excluded.citations_json
                """,
                (
                    note.note_id,
                    note.user_id,
                    note.session_id,
                    note.paper_id,
                    note.question,
                    note.answer_text,
                    self._citation_payload(note.citations),
                    note.created_at or self._now(),
                ),
            )

    def list_research_notes(self, *, user_id: str, session_id: str, paper_id: str) -> list[ResearchNote]:
        self._validate_paper_scope(user_id, session_id, paper_id)
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM rp_research_notes
                WHERE user_id=? AND session_id=? AND paper_id=?
                ORDER BY created_at ASC
                """,
                (user_id, session_id, paper_id),
            ).fetchall()
        finally:
            connection.close()
        result: list[ResearchNote] = []
        for row in rows:
            citations = self._decode_citations(row["citations_json"])
            if citations is None:
                continue
            result.append(
                ResearchNote(
                    note_id=str(row["note_id"]),
                    user_id=str(row["user_id"]),
                    session_id=str(row["session_id"]),
                    paper_id=str(row["paper_id"]),
                    question=str(row["question"]),
                    answer_text=str(row["answer_text"]),
                    citations=citations,
                    created_at=str(row["created_at"]),
                )
            )
        return result

    def delete_research_note(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        note_id: str,
    ) -> bool:
        """Delete exactly one note inside the caller's paper scope."""
        self._validate_paper_scope(user_id, session_id, paper_id)
        if not isinstance(note_id, str) or not note_id.strip():
            raise ValueError("note_id must not be empty")
        with self.transaction() as connection:
            deleted = connection.execute(
                """
                DELETE FROM rp_research_notes
                WHERE note_id=? AND user_id=? AND session_id=? AND paper_id=?
                """,
                (note_id, user_id, session_id, paper_id),
            ).rowcount
        return bool(deleted)

    def save_paper_metadata(self, metadata: PaperMetadata) -> None:
        """Atomically save one evidence-backed metadata extraction.

        The primary key includes every version dimension.  A changed parser,
        chunker, prompt, or model therefore creates a new cache entry instead
        of overwriting a result that was produced from different evidence.
        """
        self._validate_paper_scope(metadata.user_id, metadata.session_id, metadata.paper_id)
        timestamp = self._now()
        version = metadata.version
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO rp_paper_metadata (
                    user_id, session_id, paper_id, source_name, parser_name, parser_version,
                    chunker_version, prompt_version, model_name, title, authors_json,
                    affiliations_json, venue, year, doi, extraction_method,
                    field_confidence_json, citations_json, answer_text, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    user_id, session_id, paper_id, parser_name, parser_version,
                    chunker_version, prompt_version, model_name
                ) DO UPDATE SET
                    source_name=excluded.source_name,
                    title=excluded.title,
                    authors_json=excluded.authors_json,
                    affiliations_json=excluded.affiliations_json,
                    venue=excluded.venue,
                    year=excluded.year,
                    doi=excluded.doi,
                    extraction_method=excluded.extraction_method,
                    field_confidence_json=excluded.field_confidence_json,
                    citations_json=excluded.citations_json,
                    answer_text=excluded.answer_text,
                    updated_at=excluded.updated_at
                """,
                (
                    metadata.user_id,
                    metadata.session_id,
                    metadata.paper_id,
                    metadata.source_name,
                    version.parser_name,
                    version.parser_version,
                    version.chunker_version,
                    version.prompt_version,
                    version.model_name,
                    metadata.title,
                    json.dumps(list(metadata.authors), ensure_ascii=False),
                    json.dumps(list(metadata.affiliations), ensure_ascii=False),
                    metadata.venue,
                    metadata.year,
                    metadata.doi,
                    metadata.extraction_method,
                    json.dumps(metadata.field_confidence or {}, ensure_ascii=False, sort_keys=True),
                    self._citation_payload(metadata.citations),
                    metadata.answer_text,
                    metadata.created_at or timestamp,
                    timestamp,
                ),
            )

    def load_paper_metadata(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        version: PaperArtifactVersion,
    ) -> PaperMetadata | None:
        self._validate_paper_scope(user_id, session_id, paper_id)
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM rp_paper_metadata
                WHERE user_id=? AND session_id=? AND paper_id=?
                  AND parser_name=? AND parser_version=? AND chunker_version=?
                  AND prompt_version=? AND model_name=?
                """,
                (
                    user_id,
                    session_id,
                    paper_id,
                    version.parser_name,
                    version.parser_version,
                    version.chunker_version,
                    version.prompt_version,
                    version.model_name,
                ),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        authors = self._decode_string_tuple(row["authors_json"])
        affiliations = self._decode_string_tuple(row["affiliations_json"])
        confidences = self._decode_mapping(row["field_confidence_json"])
        citations = self._decode_citations(row["citations_json"])
        if authors is None or affiliations is None or confidences is None or citations is None:
            return None
        try:
            return PaperMetadata(
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
                source_name=str(row["source_name"]),
                answer_text=str(row["answer_text"]),
                citations=citations,
                version=version,
                title=str(row["title"]) if row["title"] is not None else None,
                authors=authors,
                affiliations=affiliations,
                venue=str(row["venue"]) if row["venue"] is not None else None,
                year=str(row["year"]) if row["year"] is not None else None,
                doi=str(row["doi"]) if row["doi"] is not None else None,
                extraction_method=str(row["extraction_method"]),
                field_confidence={str(name): float(score) for name, score in confidences.items()},
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
        except (TypeError, ValueError):
            return None

    def save_paper_summary(self, summary: CachedPaperSummary) -> None:
        """Save a full-paper summary and its citations under an explicit version key."""
        self._validate_paper_scope(summary.user_id, summary.session_id, summary.paper_id)
        timestamp = self._now()
        version = summary.version
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO rp_paper_summary_cache_v2 (
                    user_id, session_id, paper_id, source_name, parser_name, parser_version,
                    chunker_version, prompt_version, model_name, summary_intent,
                    request_fingerprint, citations_json, answer_text, claims_json,
                    contract_version, numeric_fidelity_passed, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    user_id, session_id, paper_id, parser_name, parser_version,
                    chunker_version, prompt_version, model_name, summary_intent,
                    request_fingerprint
                ) DO UPDATE SET
                    source_name=excluded.source_name,
                    citations_json=excluded.citations_json,
                    answer_text=excluded.answer_text,
                    claims_json=excluded.claims_json,
                    contract_version=excluded.contract_version,
                    numeric_fidelity_passed=excluded.numeric_fidelity_passed,
                    updated_at=excluded.updated_at
                """,
                (
                    summary.user_id,
                    summary.session_id,
                    summary.paper_id,
                    summary.source_name,
                    version.parser_name,
                    version.parser_version,
                    version.chunker_version,
                    version.prompt_version,
                    version.model_name,
                    summary.summary_intent,
                    summary.request_fingerprint,
                    self._citation_payload(summary.citations),
                    summary.answer_text,
                    self._claim_payload(summary.claims),
                    summary.contract_version,
                    (int(summary.numeric_fidelity_passed) if summary.numeric_fidelity_passed is not None else None),
                    summary.created_at or timestamp,
                    timestamp,
                ),
            )

    def load_paper_summary(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        version: PaperArtifactVersion,
        summary_intent: str,
        request_fingerprint: str,
    ) -> CachedPaperSummary | None:
        self._validate_paper_scope(user_id, session_id, paper_id)
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM rp_paper_summary_cache_v2
                WHERE user_id=? AND session_id=? AND paper_id=?
                  AND parser_name=? AND parser_version=? AND chunker_version=?
                  AND prompt_version=? AND model_name=?
                  AND summary_intent=? AND request_fingerprint=?
                """,
                (
                    user_id,
                    session_id,
                    paper_id,
                    version.parser_name,
                    version.parser_version,
                    version.chunker_version,
                    version.prompt_version,
                    version.model_name,
                    summary_intent,
                    request_fingerprint,
                ),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        citations = self._decode_citations(row["citations_json"])
        claims = self._decode_claims(row["claims_json"])
        if citations is None or claims is None:
            return None
        try:
            return CachedPaperSummary(
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
                source_name=str(row["source_name"]),
                answer_text=str(row["answer_text"]),
                citations=citations,
                version=version,
                summary_intent=str(row["summary_intent"]),
                request_fingerprint=str(row["request_fingerprint"]),
                claims=claims,
                contract_version=str(row["contract_version"]),
                numeric_fidelity_passed=(
                    bool(row["numeric_fidelity_passed"]) if row["numeric_fidelity_passed"] is not None else None
                ),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
        except ValueError:
            return None

    def delete_paper_artifacts(self, *, user_id: str, session_id: str, paper_id: str) -> dict[str, int]:
        """Delete cache entries with the same scope as a deleted/re-ingested paper."""
        self._validate_paper_scope(user_id, session_id, paper_id)
        with self.transaction() as connection:
            metadata_deleted = connection.execute(
                "DELETE FROM rp_paper_metadata WHERE user_id=? AND session_id=? AND paper_id=?",
                (user_id, session_id, paper_id),
            ).rowcount
            summary_deleted = connection.execute(
                "DELETE FROM rp_paper_summary_cache WHERE user_id=? AND session_id=? AND paper_id=?",
                (user_id, session_id, paper_id),
            ).rowcount
            summary_deleted += connection.execute(
                "DELETE FROM rp_paper_summary_cache_v2 WHERE user_id=? AND session_id=? AND paper_id=?",
                (user_id, session_id, paper_id),
            ).rowcount
        return {"metadata_deleted": metadata_deleted, "summary_deleted": summary_deleted}

    def delete_paper_scope_data(self, *, user_id: str, session_id: str, paper_id: str) -> dict[str, int]:
        """Delete caches plus user-created data only after an explicit paper removal."""
        counts = self.delete_paper_artifacts(user_id=user_id, session_id=session_id, paper_id=paper_id)
        with self.transaction() as connection:
            counts["annotations_deleted"] = connection.execute(
                "DELETE FROM rp_annotations WHERE user_id=? AND session_id=? AND paper_id=?",
                (user_id, session_id, paper_id),
            ).rowcount
            counts["research_notes_deleted"] = connection.execute(
                "DELETE FROM rp_research_notes WHERE user_id=? AND session_id=? AND paper_id=?",
                (user_id, session_id, paper_id),
            ).rowcount
            counts["scoped_working_memory_deleted"] = connection.execute(
                "DELETE FROM rp_scoped_working_memory WHERE user_id=? AND session_id=? AND scope_key=?",
                (user_id, session_id, f"paper:{paper_id}"),
            ).rowcount
            counts["long_term_memory_deactivated"] = connection.execute(
                """
                UPDATE rp_long_term_memory SET is_active=0
                WHERE user_id=? AND source_session=? AND source_doc=? AND is_active=1
                """,
                (user_id, session_id, paper_id),
            ).rowcount
        return counts

    def paper_related_counts(self, *, user_id: str, session_id: str, paper_id: str) -> dict[str, int]:
        """Return deletion-impact counts for UI confirmation."""
        self._validate_paper_scope(user_id, session_id, paper_id)
        connection = self._connect()
        try:
            queries = {
                "annotations": "SELECT COUNT(*) FROM rp_annotations WHERE user_id=? AND session_id=? AND paper_id=?",
                "research_notes": (
                    "SELECT COUNT(*) FROM rp_research_notes WHERE user_id=? AND session_id=? AND paper_id=?"
                ),
                "long_term_memories": (
                    "SELECT COUNT(*) FROM rp_long_term_memory "
                    "WHERE user_id=? AND source_session=? AND source_doc=? AND is_active=1"
                ),
            }
            return {
                name: int(connection.execute(sql, (user_id, session_id, paper_id)).fetchone()[0])
                for name, sql in queries.items()
            }
        finally:
            connection.close()

    def delete_session(self, user_id: str, session_id: str) -> dict[str, int]:
        """Remove working memory and deactivate related LTM in one transaction."""
        self._validate_scope(user_id, session_id)
        with self.transaction() as connection:
            deactivated = connection.execute(
                """
                UPDATE rp_long_term_memory SET is_active=0
                WHERE user_id=? AND source_session=? AND is_active=1
                """,
                (user_id, session_id),
            ).rowcount
            deleted = connection.execute(
                "DELETE FROM rp_working_memory WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).rowcount
            scoped_deleted = connection.execute(
                "DELETE FROM rp_scoped_working_memory WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).rowcount
            metadata_deleted = connection.execute(
                "DELETE FROM rp_paper_metadata WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).rowcount
            summary_deleted = connection.execute(
                "DELETE FROM rp_paper_summary_cache WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).rowcount
            summary_deleted += connection.execute(
                "DELETE FROM rp_paper_summary_cache_v2 WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).rowcount
            annotations_deleted = connection.execute(
                "DELETE FROM rp_annotations WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).rowcount
            notes_deleted = connection.execute(
                "DELETE FROM rp_research_notes WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).rowcount
        return {
            "working_memory_deleted": deleted,
            "scoped_working_memory_deleted": scoped_deleted,
            "long_term_memory_deactivated": deactivated,
            "paper_metadata_deleted": metadata_deleted,
            "paper_summary_deleted": summary_deleted,
            "annotations_deleted": annotations_deleted,
            "research_notes_deleted": notes_deleted,
        }

    @staticmethod
    def _to_long_term_memory(row: sqlite3.Row) -> LongTermMemory:
        return LongTermMemory(
            id=int(row["id"]),
            user_id=str(row["user_id"]),
            memory_type=str(row["memory_type"]),
            content=str(row["content"]),
            source_session=str(row["source_session"]),
            source_doc=row["source_doc"],
            priority=int(row["priority"]),
            is_active=bool(row["is_active"]),
            last_updated=str(row["last_updated"]),
        )


class SessionFileRepository:
    """User-isolated session JSON storage with atomic replace on every save."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def save(
        self,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        active_pdf_names: list[str],
        active_documents: list[dict[str, Any]] | None = None,
        selected_paper_id: str | None = None,
        session_title: str | None = None,
    ) -> None:
        path = self.settings.session_file_path(user_id, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 4,
            "current_session": session_id,
            "messages": messages,
            "active_pdf_names": active_pdf_names,
            "active_documents": active_documents or [],
            "selected_paper_id": selected_paper_id,
            "session_title": session_title,
        }
        descriptor, temporary_path = tempfile.mkstemp(
            dir=path.parent, prefix=f".{session_id}.", suffix=".tmp", text=True
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            Path(temporary_path).replace(path)
        except Exception:
            Path(temporary_path).unlink(missing_ok=True)
            raise

    def load(self, user_id: str, session_id: str) -> SessionSnapshot | None:
        path = self.settings.session_file_path(user_id, session_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SessionDataCorrupted(f"Session JSON is unreadable: {path.name}") from error
        if not isinstance(payload, dict):
            raise SessionDataCorrupted(f"Session JSON is not an object: {path.name}")
        messages = payload.get("messages", [])
        active_pdf_names = payload.get("active_pdf_names", [])
        active_documents = payload.get("active_documents", [])
        selected_paper_id = payload.get("selected_paper_id")
        session_title = payload.get("session_title")
        if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
            raise SessionDataCorrupted(f"Session messages have an invalid shape: {path.name}")
        if not isinstance(active_pdf_names, list) or not all(isinstance(item, str) for item in active_pdf_names):
            raise SessionDataCorrupted(f"Session PDF list has an invalid shape: {path.name}")
        if not isinstance(active_documents, list) or not all(
            isinstance(item, dict) and isinstance(item.get("source_name"), str) for item in active_documents
        ):
            raise SessionDataCorrupted(f"Session document metadata has an invalid shape: {path.name}")
        if selected_paper_id is not None and not isinstance(selected_paper_id, str):
            raise SessionDataCorrupted(f"Session paper scope has an invalid shape: {path.name}")
        if session_title is not None and not isinstance(session_title, str):
            raise SessionDataCorrupted(f"Session title has an invalid shape: {path.name}")
        return SessionSnapshot(
            session_id=session_id,
            messages=messages,
            active_pdf_names=active_pdf_names,
            active_documents=active_documents,
            selected_paper_id=selected_paper_id,
            session_title=session_title,
        )

    def list(self, user_id: str, session_id: str) -> list[str]:
        """List a user's sessions; ``session_id`` validates the caller's scope."""
        self.settings.session_file_path(user_id, session_id)
        directory = self.settings.user_sessions_dir(user_id)
        if not directory.is_dir():
            return []
        return sorted((path.stem for path in directory.glob("*.json")), reverse=True)

    def delete(self, user_id: str, session_id: str) -> bool:
        path = self.settings.session_file_path(user_id, session_id)
        if not path.exists():
            return False
        path.unlink()
        return True
