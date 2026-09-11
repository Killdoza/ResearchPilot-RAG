# ruff: noqa: E501  # Migration manifest reasons remain complete audit sentences.
"""Safe, explicit migration for legacy sessions and user-scoping metadata.

Default mode is a dry run. ``--apply`` requires a user ID and creates a full
backup plus a JSON manifest before any source data is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.config import ConfigurationError, build_settings  # noqa: E402
from researchpilot.storage import SQLiteMemoryRepository  # noqa: E402

BACKUP_TARGETS = (
    "memory_manager.db",
    "memory_manager.db-wal",
    "memory_manager.db-shm",
    "chroma_db",
    "chroma_global",
    "chroma_session",
    "sessions",
)
LEGACY_MEMORY_DATABASE = "memory_manager.db"
LEGACY_SESSIONS_DIRECTORY = "sessions"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_session_files(sessions_dir: Path) -> list[Path]:
    return sorted(sessions_dir.glob("*.json")) if sessions_dir.is_dir() else []


def sqlite_unscoped_count(path: Path) -> int:
    if not path.exists():
        return 0
    connection = sqlite3.connect(path)
    try:
        existing = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='long_term_memory'"
        ).fetchone()
        if not existing:
            return 0
        return connection.execute(
            "SELECT COUNT(*) FROM long_term_memory WHERE user_id IS NULL OR user_id = '' OR user_id = 'default_user'"
        ).fetchone()[0]
    finally:
        connection.close()


def chroma_missing_user_ids(path: Path) -> dict[str, Any]:
    """Never open a legacy Chroma directory during migration planning.

    Some Chroma versions update internal SQLite metadata while opening a
    PersistentClient. The audit script inspects a disposable copy instead.
    """
    return {
        "exists": path.is_dir(),
        "status": "not_opened",
        "reason": "Run audit_legacy_data.py; it inspects a disposable copy rather than this legacy store.",
    }


def build_plan(project_root: Path, user_id: str | None) -> dict[str, Any]:
    settings = build_settings(project_root=project_root)
    legacy_sessions_dir = project_root / LEGACY_SESSIONS_DIRECTORY
    legacy_database_path = project_root / LEGACY_MEMORY_DATABASE
    session_files = legacy_session_files(legacy_sessions_dir)
    chroma_paths = {
        "chroma_db": project_root / "chroma_db",
        "chroma_global": project_root / "chroma_global",
        "chroma_session": project_root / "chroma_session",
    }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "project_root": str(project_root),
        "requested_user_id": user_id,
        "legacy_root_session_files": [str(path) for path in session_files],
        "session_copy_target": str(settings.user_sessions_dir(user_id)) if user_id else None,
        "sqlite_unscoped_long_term_memory_count": sqlite_unscoped_count(legacy_database_path),
        "new_runtime_targets": {
            "memory_database": str(settings.memory_db_path),
            "sessions_directory": str(settings.sessions_dir),
            "global_chroma_directory": str(settings.global_chroma_dir),
            "session_chroma_directory": str(settings.session_chroma_dir),
        },
        "chroma_missing_user_id_counts": {name: chroma_missing_user_ids(path) for name, path in chroma_paths.items()},
        "backup_targets": [str(project_root / target) for target in BACKUP_TARGETS if (project_root / target).exists()],
    }


def create_backup(project_root: Path, backup_root: Path) -> list[dict[str, str]]:
    backup_root.mkdir(parents=True, exist_ok=False)
    copied: list[dict[str, str]] = []
    for relative_path in BACKUP_TARGETS:
        source = project_root / relative_path
        if not source.exists():
            continue
        target = backup_root / relative_path
        if source.is_dir():
            shutil.copytree(source, target)
            copied.append({"source": str(source), "backup": str(target), "kind": "directory"})
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(
                {
                    "source": str(source),
                    "backup": str(target),
                    "kind": "file",
                    "sha256": sha256(target),
                }
            )
    return copied


def import_legacy_sqlite(settings: Any, project_root: Path, user_id: str) -> dict[str, int]:
    """Copy unscoped legacy rows into the new tables without altering old rows."""
    legacy_database_path = project_root / LEGACY_MEMORY_DATABASE
    if not legacy_database_path.exists():
        return {"working_memory_copied": 0, "long_term_memory_copied": 0}
    if settings.memory_db_path.resolve() == legacy_database_path.resolve():
        raise RuntimeError("Migration target database must not be the frozen legacy database.")
    SQLiteMemoryRepository(settings).initialize_schema()
    source_connection = sqlite3.connect(f"file:{legacy_database_path.as_posix()}?mode=ro", uri=True)
    source_connection.row_factory = sqlite3.Row
    target_connection = sqlite3.connect(settings.memory_db_path)
    copied_working_memory = 0
    copied_long_term_memory = 0
    try:
        tables = {row[0] for row in source_connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        with target_connection:
            if "working_memory" in tables:
                rows = source_connection.execute(
                    """
                    SELECT session_id, current_topic, core_demands, confirmed_conclusions,
                           unresolved_questions, context_summary, last_updated
                    FROM working_memory
                    """
                ).fetchall()
                for row in rows:
                    cursor = target_connection.execute(
                        """
                        INSERT INTO rp_working_memory
                            (user_id, session_id, current_topic, core_demands,
                             confirmed_conclusions, unresolved_questions,
                             context_summary, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, session_id) DO NOTHING
                        """,
                        (
                            user_id,
                            row["session_id"],
                            row["current_topic"] or "",
                            row["core_demands"] or "[]",
                            row["confirmed_conclusions"] or "[]",
                            row["unresolved_questions"] or "[]",
                            row["context_summary"] or "",
                            row["last_updated"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
                    copied_working_memory += cursor.rowcount
            if "long_term_memory" in tables:
                rows = source_connection.execute(
                    """
                    SELECT id, memory_type, content, source_session, source_doc,
                           priority, is_active, last_updated
                    FROM long_term_memory
                    WHERE user_id IS NULL OR user_id = '' OR user_id = 'default_user'
                    """
                ).fetchall()
                for row in rows:
                    cursor = target_connection.execute(
                        """
                        INSERT INTO rp_long_term_memory
                            (user_id, memory_type, content, source_session, source_doc,
                             priority, is_active, last_updated, legacy_source_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(legacy_source_id) DO NOTHING
                        """,
                        (
                            user_id,
                            row["memory_type"] or "legacy_memory",
                            row["content"] or "",
                            row["source_session"] or "legacy_unassigned",
                            row["source_doc"],
                            row["priority"] if row["priority"] is not None else 1,
                            row["is_active"] if row["is_active"] in (0, 1) else 1,
                            row["last_updated"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            f"legacy:long_term_memory:{row['id']}",
                        ),
                    )
                    copied_long_term_memory += cursor.rowcount
        return {
            "working_memory_copied": copied_working_memory,
            "long_term_memory_copied": copied_long_term_memory,
        }
    finally:
        source_connection.close()
        target_connection.close()


def apply_session_copies(settings: Any, project_root: Path, user_id: str) -> int:
    copied = 0
    target_dir = settings.user_sessions_dir(user_id)
    legacy_sessions_dir = project_root / LEGACY_SESSIONS_DIRECTORY
    if target_dir.resolve().is_relative_to(legacy_sessions_dir.resolve()):
        raise RuntimeError("Migration target sessions directory must not be the frozen legacy directory.")
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in legacy_session_files(legacy_sessions_dir):
        target = target_dir / source.name
        if target.exists():
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy RAG data with an explicit backup")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--user-id", help="Target user ID; required with --apply")
    parser.add_argument("--apply", action="store_true", help="Create backup then perform the migration")
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="Create a byte-for-byte local backup and manifest without changing source data",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if args.apply and args.backup_only:
        parser.error("Choose either --apply or --backup-only, not both.")
    if args.apply and not args.user_id:
        parser.error("--apply requires --user-id so legacy records cannot be assigned implicitly.")

    if args.user_id:
        try:
            build_settings(project_root=project_root).user_sessions_dir(args.user_id)
        except ConfigurationError as error:
            parser.error(str(error))

    plan = build_plan(project_root, args.user_id)
    if args.backup_only:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = project_root / "data" / "runtime" / "backups" / f"legacy_snapshot_{timestamp}"
        manifest = {
            "mode": "backup_only",
            "plan": plan,
            "backup_root": str(backup_root),
            "backup_manifest": create_backup(project_root, backup_root),
        }
        manifest_path = backup_root / "backup_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Backup complete. Manifest: {manifest_path}")
        return
    if not args.apply:
        print(json.dumps({"mode": "dry_run", "plan": plan}, ensure_ascii=False, indent=2))
        return

    settings = build_settings(project_root=project_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = project_root / "data" / "runtime" / "backups" / f"legacy_migration_{timestamp}"
    backup_manifest = create_backup(project_root, backup_root)
    result = {
        "mode": "applied",
        "plan": plan,
        "backup_root": str(backup_root),
        "backup_manifest": backup_manifest,
        "copied_legacy_sessions": apply_session_copies(settings, project_root, args.user_id),
        "sqlite_records_copied": import_legacy_sqlite(settings, project_root, args.user_id),
        "chroma_migration": {
            "status": "deferred",
            "reason": "Legacy Chroma stores are preserved read-only. The current stores must be repaired and a separate clone migration reviewed before vector data is imported.",
        },
    }
    manifest_path = backup_root / "migration_manifest.json"
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Migration complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
