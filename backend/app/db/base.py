"""`Db` - the storage-backend contract, mirroring Agno's `agno/db/base.py`
(`BaseDb`) trimmed to the session-CRUD subset this project's agent loop
actually calls: `get_session`/`upsert_session`/`delete_session`/
`get_sessions`.

One `ABC`, one concrete implementation this stage (`db/sqlite/sqlite.py`) -
the same "prove the pattern with one provider" reasoning as
`models/base.py:Model` having exactly two concrete subclasses
(`OpenAIChat`/`Claude`). Real Agno's `BaseDb` is ~2000 lines because it also
covers memory, evals, knowledge, culture, scheduler, and approval-request
storage - none of those subsystems exist in this project, so none of their
methods are declared here. Also dropped from the real session-CRUD surface
itself: `upsert_sessions` (bulk variant), `rename_session` (cosmetic),
`get_session_metrics`/summary lookups (session/summary.py's feature, not
built) - `get_session`/`upsert_session`/`delete_session`/`get_sessions` are
the whole contract `agent/_storage.py` and `agent/_session.py` need.

No `SessionType` discriminator either (real Agno's `BaseDb` supports
agent/team/workflow sessions with one enum picking between them) - this
project has no team/workflow layer, so every session a `Db` handles is
implicitly an `AgentSession`; adding a type tag for a distinction that can't
exist yet would just be dead parameter-plumbing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.session.agent import AgentSession


class Db(ABC):
    """Storage-backend contract for `AgentSession`s. A concrete `Db`
    (`db/sqlite/sqlite.py`) owns wherever the data actually lives -
    `agent/_storage.py`/`agent/_session.py` only ever talk to this
    interface, never to a specific backend directly.
    """

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Look up one session by id. `None` if it doesn't exist yet -
        `read_or_create_session()` (`agent/_storage.py`) is what turns that
        into a fresh, in-memory `AgentSession` for the caller."""
        raise NotImplementedError

    @abstractmethod
    def get_sessions(
        self,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[AgentSession]:
        """All sessions matching the given filters (either/both/neither) -
        not used by the agent loop itself this stage, but the natural
        counterpart to `get_session()` for anything listing a user's or an
        agent's conversation history."""
        raise NotImplementedError

    @abstractmethod
    def upsert_session(self, session: AgentSession) -> AgentSession:
        """Insert `session` if its `session_id` is new, otherwise overwrite
        the existing row in place - the one write path both a fresh session
        and an updated one go through, keyed by `session_id`."""
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Remove one session's row entirely. No-op if it doesn't exist."""
        raise NotImplementedError
