"""Session identifiers and filesystem-safe session helpers."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4


def generate_session_id(now: datetime | None = None) -> str:
    """Generate a sortable, collision-resistant ID for one browser session.

    The legacy timestamp-only IDs could collide when multiple users created a
    session in the same second. A random suffix preserves readable ordering
    while making the storage namespace independent for every session.
    """
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{uuid4().hex[:12]}"


def generate_local_user_id() -> str:
    """Create an anonymous browser-local user scope for the local demo.

    This is not authentication. It prevents one active browser session from
    reading another session's persisted chats, archives or long-term memories
    until a real identity provider is introduced.
    """
    return f"local_{uuid4().hex}"
