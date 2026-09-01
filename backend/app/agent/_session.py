"""Stage 5 - the save-bookend: `save_session()`, mirroring real Agno's
`agent/_session.py:save_session()` trimmed to the one call `_run.py`'s
terminal paths need.

Real Agno's `agent/_session.py` also has `rename_session()`,
`generate_session_name()` (an LLM call that titles a session),
`get_session_metrics()`, and session-summary lookups - all cosmetic or
feature-adjacent (naming/summarization/analytics), none of them are "does
this run's data actually get persisted," so none of them are built here.
`asave_session()` (the async twin) is Stage 6+ territory, same as every
other `a`-prefixed method skipped this stage.
"""

from __future__ import annotations

from typing import Optional

from app.agent.agent import Agent
from app.session.agent import AgentSession


def save_session(agent: Agent, session: Optional[AgentSession]) -> Optional[AgentSession]:
    """Write `session` to `agent.db`, if there is one. No-op (returns
    `session` unchanged) when either `agent.db` or `session` itself is
    `None` - the same "Stage 5 is opt-in, unconfigured agents are
    unaffected" guarantee `read_or_create_session()` (`agent/_storage.py`)
    makes on the read side.
    """
    if agent.db is None or session is None:
        return session

    return agent.db.upsert_session(session)
