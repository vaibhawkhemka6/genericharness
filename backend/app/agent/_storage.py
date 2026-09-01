"""Stage 5 - the load-bookend: `read_or_create_session()`, mirroring real
Agno's `agent/_storage.py:read_or_create_session()` trimmed to the one
thing this project's loop needs from it.

Real Agno's `agent/_storage.py` is ~1100 lines, but almost all of it
(`to_dict`/`from_dict`/`save`/`load`/`delete`) is a *different* concern this
project doesn't build: persisting the `Agent`'s own *config* (model id,
tools, instructions - "save this agent definition so it can be reconstructed
later"), not its conversation history. Easy to conflate the two because
they live in the same real-Agno file; this project keeps only the
session/history half, and gives it a different name (`AgentSession`, not
`Agent`) precisely so the distinction can't get lost the way it can there.

No `agent._cached_session` in-memory-caching optimization either - every
call here is a real `agent.db.get_session(...)` round trip. Fine at this
project's scale; the caching layer is a perf detail this project's `Agent`
(`agent/agent.py`) doesn't have a field for.
"""

from __future__ import annotations

from typing import Optional

from app.agent.agent import Agent
from app.session.agent import AgentSession


def read_or_create_session(
    agent: Agent,
    session_id: str,
    user_id: Optional[str] = None,
) -> Optional[AgentSession]:
    """Look up `session_id` in `agent.db`, or build a fresh (not-yet-saved)
    `AgentSession` if nothing's there yet.

    Returns `None` when `agent.db` isn't configured at all - a `None`
    session is the signal `_messages.py:get_session_history_messages()` and
    `_run.py`'s terminal paths use to fall back to the pre-Stage-5
    `agent.session_history` stub / skip saving entirely, so an `Agent`
    built without a `db` keeps behaving exactly as it did before this
    stage, unchanged.
    """
    if agent.db is None:
        return None

    session = agent.db.get_session(session_id)
    if session is not None:
        return session

    return AgentSession(
        session_id=session_id,
        agent_id=agent.id,
        user_id=user_id,
    )
