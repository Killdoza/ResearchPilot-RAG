"""The old Demo session boundary must never silently enter new RAG."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.legacy_adapter import LegacySessionAdapter  # noqa: E402


def test_legacy_only_session_requires_reupload() -> None:
    status = LegacySessionAdapter.inspect([{"source_name": "old_demo.pdf", "legacy": True}])

    assert status.requires_reupload is True
    assert "重新上传" in status.banner_text
    assert "不执行兼容检索" in status.answer_text


def test_mixed_session_never_marks_structured_document_as_legacy_only() -> None:
    status = LegacySessionAdapter.inspect(
        [
            {"source_name": "old_demo.pdf", "legacy": True},
            {"source_name": "new.pdf", "paper_id": "paper_new"},
        ]
    )

    assert status.has_legacy_documents is True
    assert status.has_structured_documents is True
    assert status.requires_reupload is False
