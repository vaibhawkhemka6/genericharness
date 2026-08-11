"""In-memory conversation store, keyed by session_id.

Phase 1 only - not persistent across process restarts. DB-backed session
storage is a later phase, matching Agno's layered approach where the core
agent loop doesn't own storage directly.
"""

from __future__ import annotations

from typing import Any


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Returns the mutable history list for a session (creating it if new)."""
        return self._sessions.setdefault(session_id, [])

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())
