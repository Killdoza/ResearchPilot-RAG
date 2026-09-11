"""Central, non-secret runtime configuration for ResearchPilot.

All paths are resolved from the repository root rather than the process working
directory. This makes local development, tests and future API deployment use
the same storage layout.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when the local runtime configuration is incomplete or unsafe."""


_SAFE_STORAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


@dataclass(frozen=True)
class Settings:
    """Resolved settings for one local ResearchPilot deployment."""

    project_root: Path
    legacy_app_path: Path
    resources_dir: Path
    logo_path: Path
    global_chroma_dir: Path
    session_chroma_dir: Path
    evaluation_chroma_dir: Path
    sparse_index_dir: Path
    sessions_dir: Path
    memory_db_path: Path
    runs_dir: Path
    uploads_dir: Path
    traces_dir: Path
    global_collection_name: str
    session_collection_name: str
    evaluation_collection_name: str
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    deepseek_input_cost_per_million: float | None
    deepseek_output_cost_per_million: float | None
    local_user_id: str | None
    api_strict_crossdoc_enabled: bool

    @staticmethod
    def _validate_storage_id(value: str, field_name: str) -> None:
        if not _SAFE_STORAGE_ID.fullmatch(value) or value in {".", ".."}:
            raise ConfigurationError(
                f"{field_name} must contain only letters, numbers, '.', '_' or '-' and must not be '.' or '..'."
            )

    def user_sessions_dir(self, user_id: str) -> Path:
        """Return the isolated session directory for one local user scope."""
        self._validate_storage_id(user_id, "user_id")
        return self.sessions_dir / user_id

    def session_file_path(self, user_id: str, session_id: str) -> Path:
        """Return an isolated JSON path after rejecting unsafe path components."""
        self._validate_storage_id(session_id, "session_id")
        return self.user_sessions_dir(user_id) / f"{session_id}.json"


def validate_storage_id(value: str, field_name: str) -> None:
    """校验会进入文件路径或索引 ID 的用户/会话标识。"""
    Settings._validate_storage_id(value, field_name)


def require_ascii_chroma_path(path: Path, setting_name: str) -> None:
    """拒绝已知会在 Windows 上损坏大型 HNSW 索引的非 ASCII 持久化路径。"""
    rendered_path = str(path)
    if any(ord(character) > 127 for character in rendered_path):
        raise ConfigurationError(
            f"{setting_name} 包含非 ASCII 字符，当前 Windows + Chroma 组合无法可靠持久化"
            "大型 HNSW 索引。请在 .env 中将它设为 ASCII 路径，例如："
            "E:/python_projects/ResearchPilot_RAG/data/runtime/chroma_evaluation。"
        )


def _project_root() -> Path:
    # config.py -> researchpilot -> src -> repository root
    return Path(__file__).resolve().parents[2]


def _resolve_path(project_root: Path, value: str | None, default: str) -> Path:
    path = Path(value or default).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _value(environ: Mapping[str, str], name: str, default: str) -> str:
    return environ.get(name, default).strip() or default


def _optional_nonnegative_float(environ: Mapping[str, str], name: str) -> float | None:
    raw = environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a non-negative number") from exc
    if value < 0:
        raise ConfigurationError(f"{name} must be a non-negative number")
    return value


def build_settings(
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    load_dotenv_file: bool = True,
) -> Settings:
    """Build settings without performing writes.

    When ``environ`` is provided, it is used as-is. This makes tests isolated
    from the developer's shell and local ``.env`` file.
    """
    root = (project_root or _project_root()).resolve()
    if environ is None:
        if load_dotenv_file:
            load_dotenv(root / ".env", override=False)
        environ = os.environ

    configured_local_user_id = environ.get("RESEARCHPILOT_LOCAL_USER_ID") or None
    if configured_local_user_id:
        Settings._validate_storage_id(configured_local_user_id, "RESEARCHPILOT_LOCAL_USER_ID")

    return Settings(
        project_root=root,
        legacy_app_path=root / "legacy" / "RAG_4.2.py",
        resources_dir=_resolve_path(root, environ.get("RESEARCHPILOT_RESOURCES_DIR"), "resources"),
        logo_path=_resolve_path(root, environ.get("RESEARCHPILOT_LOGO_PATH"), "resources/logo.png"),
        # New runtime stores are separated from the frozen legacy data at the
        # repository root. The migration script is the only code allowed to
        # read legacy storage.
        global_chroma_dir=_resolve_path(
            root, environ.get("RESEARCHPILOT_GLOBAL_CHROMA_DIR"), "data/runtime/chroma_global"
        ),
        session_chroma_dir=_resolve_path(
            root, environ.get("RESEARCHPILOT_SESSION_CHROMA_DIR"), "data/runtime/chroma_session"
        ),
        # Evaluation data must never share a physical Chroma collection with
        # browser sessions.  A separate store also lets a broken experiment be
        # recreated without risking product-session documents.
        evaluation_chroma_dir=_resolve_path(
            root, environ.get("RESEARCHPILOT_EVALUATION_CHROMA_DIR"), "data/runtime/chroma_evaluation"
        ),
        sparse_index_dir=_resolve_path(root, environ.get("RESEARCHPILOT_SPARSE_INDEX_DIR"), "data/runtime/bm25"),
        sessions_dir=_resolve_path(root, environ.get("RESEARCHPILOT_SESSIONS_DIR"), "data/runtime/sessions"),
        memory_db_path=_resolve_path(
            root, environ.get("RESEARCHPILOT_MEMORY_DB_PATH"), "data/runtime/researchpilot.sqlite3"
        ),
        runs_dir=_resolve_path(root, environ.get("RESEARCHPILOT_RUNS_DIR"), "data/runtime/runs"),
        uploads_dir=_resolve_path(root, environ.get("RESEARCHPILOT_UPLOADS_DIR"), "data/runtime/uploads"),
        traces_dir=_resolve_path(root, environ.get("RESEARCHPILOT_TRACES_DIR"), "data/runtime/traces"),
        global_collection_name=_value(environ, "RESEARCHPILOT_GLOBAL_COLLECTION", "global_memory"),
        session_collection_name=_value(environ, "RESEARCHPILOT_SESSION_COLLECTION", "session_memory"),
        evaluation_collection_name=_value(environ, "RESEARCHPILOT_EVALUATION_COLLECTION", "evaluation_papers_v01"),
        deepseek_api_key=environ.get("DEEPSEEK_API_KEY") or None,
        deepseek_base_url=_value(environ, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=_value(environ, "DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_input_cost_per_million=_optional_nonnegative_float(environ, "DEEPSEEK_INPUT_COST_PER_MILLION"),
        deepseek_output_cost_per_million=_optional_nonnegative_float(environ, "DEEPSEEK_OUTPUT_COST_PER_MILLION"),
        local_user_id=configured_local_user_id,
        api_strict_crossdoc_enabled=(
            environ.get("RESEARCHPILOT_API_STRICT_CROSSDOC_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
        ),
    )


def ensure_runtime_directories(settings: Settings) -> None:
    """Create only safe, ignored runtime directories needed by the app."""
    for directory in (
        settings.global_chroma_dir,
        settings.session_chroma_dir,
        settings.evaluation_chroma_dir,
        settings.sparse_index_dir,
        settings.sessions_dir,
        settings.memory_db_path.parent,
        settings.runs_dir,
        settings.uploads_dir,
        settings.traces_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def require_deepseek_api_key(settings: Settings) -> str:
    """Return the API key or raise a clear, actionable startup error."""
    if not settings.deepseek_api_key:
        raise ConfigurationError(
            "DEEPSEEK_API_KEY is not configured. Copy .env.example to .env and "
            "set your own key, or export DEEPSEEK_API_KEY in the current shell."
        )
    return settings.deepseek_api_key
