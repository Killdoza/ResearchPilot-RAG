"""延迟装配现有 RAG Core，确保 /health 从不加载模型或连接模型 API。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

from ..config import Settings
from ..runtime_factory import ProductRuntime, build_product_runtime
from ..services import ResearchPilotService


class ServiceProvider(Protocol):
    def get(self) -> ResearchPilotService:
        """Return the lazily initialized, process-local RAG service."""


class RuntimeServiceProvider:
    """按产品 Profile 延迟创建与 Streamlit 相同的 RAG Service。"""

    def __init__(self, settings: Settings, *, profile_name: str = "high_precision") -> None:
        self._settings = settings
        self._profile_name = profile_name
        self._runtime: ProductRuntime | None = None
        self._lock = threading.Lock()

    def get(self) -> ResearchPilotService:
        return self.get_runtime().service

    def get_runtime(self) -> ProductRuntime:
        if self._runtime is not None:
            return self._runtime
        with self._lock:
            if self._runtime is None:
                self._runtime = self._build()
        return self._runtime

    def _build(self) -> ProductRuntime:
        # Factory 延迟导入模型，因此 /health 和 /ready 仍不会加载模型或调用 API。
        return build_product_runtime(
            settings=self._settings,
            profile_name=self._profile_name,
        )


class DocumentRegistry:
    """Store upload metadata separately from vectors; every path is scope-validated."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _scope_dir(self, *, user_id: str, session_id: str) -> Path:
        from ..config import validate_storage_id

        validate_storage_id(user_id, "user_id")
        validate_storage_id(session_id, "session_id")
        return self._settings.uploads_dir / user_id / session_id

    def save(self, *, user_id: str, session_id: str, metadata: dict[str, object], pdf_bytes: bytes) -> None:
        paper_id = str(metadata["paper_id"])
        from ..config import validate_storage_id

        validate_storage_id(paper_id, "paper_id")
        directory = self._scope_dir(user_id=user_id, session_id=session_id)
        directory.mkdir(parents=True, exist_ok=True)
        pdf_target = directory / f"{paper_id}.pdf"
        metadata_target = directory / f"{paper_id}.json"
        pdf_temp = directory / f".{paper_id}.pdf.tmp"
        json_temp = directory / f".{paper_id}.json.tmp"
        pdf_temp.write_bytes(pdf_bytes)
        import json

        json_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        pdf_temp.replace(pdf_target)
        json_temp.replace(metadata_target)

    def list(self, *, user_id: str, session_id: str) -> list[dict[str, object]]:
        directory = self._scope_dir(user_id=user_id, session_id=session_id)
        if not directory.exists():
            return []
        import json

        result: list[dict[str, object]] = []
        for path in sorted(directory.glob("paper_*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(item, dict):
                result.append(item)
        return result

    def load(self, *, user_id: str, session_id: str, paper_id: str) -> dict[str, object] | None:
        import json

        from ..config import validate_storage_id

        validate_storage_id(paper_id, "paper_id")
        path = self._scope_dir(user_id=user_id, session_id=session_id) / f"{paper_id}.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    def delete(self, *, user_id: str, session_id: str, paper_id: str) -> None:
        from ..config import validate_storage_id

        validate_storage_id(paper_id, "paper_id")
        directory = self._scope_dir(user_id=user_id, session_id=session_id)
        for suffix in (".pdf", ".json"):
            path = directory / f"{paper_id}{suffix}"
            if path.exists():
                path.unlink()
