"""`AgentSession`, mirroring Agno's `agno/session/agent.py` trimmed to the
persistence surface this project needs: a run history container that
round-trips to/from a DB row as one JSON blob, plus the one read-side method
`_messages.py:get_session_history_messages()` calls into.

Stage 5 - the real replacement for the `agent.session_history` stub
(`agent/agent.py`) `_messages.py` has used as a stand-in since Stage 3 Half
B. Where that stub was just a plain `List[Message]]` the caller handed in by
hand, an `AgentSession` is what actually gets read from and written back to
a `Db` (`db/base.py`) - one row per `(agent_id, session_id)`, `runs` holding
every `RunOutput` (`run/agent.py`) that session has produced so far.

Trimmed against real Agno's `AgentSession`: no `session_data["session_name"]`/
`session_state` merge helpers, no team/workflow session variants, no
`SessionSummary` field (session/summary.py - a separate LLM-recap feature,
not raw persistence), no chat-history-token-trimming. `session_data` is kept
as a free-form `Optional[Dict[str, Any]]` bucket for parity/future use, but
nothing in this stage writes anything into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from time import time
from typing import Any, Dict, List, Optional

from app.models.message import Message
from app.run.agent import RunOutput
from app.run.base import RunStatus


@dataclass
class AgentSession:
    """One agent's conversation history, as persisted to (and loaded from) a
    `Db` - the object `to_dict()`s into the JSON blob a DB row stores, and
    `from_dict()`s back out of it."""

    session_id: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None

    # Free-form bucket for session-level data (kept for parity/future use -
    # nothing in this stage writes to it).
    session_data: Optional[Dict[str, Any]] = None

    # Every RunOutput this session has produced so far, oldest first.
    runs: List[RunOutput] = field(default_factory=list)

    created_at: int = field(default_factory=lambda: int(time()))
    updated_at: int = field(default_factory=lambda: int(time()))

    def upsert_run(self, run: RunOutput) -> None:
        """Dedupe-by-`run_id`-or-append: a run that already exists in
        `self.runs` (same `run_id` - this happens when a caller re-saves a
        run that was already persisted once, e.g. a cancelled/errored run
        that gets updated in place rather than appended twice) is replaced
        in position; a new one is appended. Also bumps `updated_at`, since
        this is the one method that actually changes what's persisted.
        """
        for i, existing in enumerate(self.runs):
            if existing.run_id == run.run_id:
                self.runs[i] = run
                self.updated_at = int(time())
                return
        self.runs.append(run)
        self.updated_at = int(time())

    def get_messages(
        self,
        skip_statuses: Optional[List[RunStatus]] = None,
        skip_roles: Optional[List[str]] = None,
        last_n_runs: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """History-assembly: flatten every kept run's `.messages` into one
        list, in run order - what `get_session_history_messages()`
        (`agent/_messages.py`) calls to get the "history" piece of a new
        run's `RunMessages`.

        `skip_statuses` defaults to `[error, cancelled]` - this is the
        "persist everything, filter bad runs out of future context"
        mechanism: a run that failed or was cancelled still gets saved
        (`_run.py`'s terminal paths save regardless of outcome, partial
        transcript and all), but its messages don't get replayed as context
        for the *next* run - a model shouldn't have to make sense of a
        conversation that includes its own failed attempt.

        Every returned `Message` is a copy with `from_history=True` set -
        never a mutation of what's stored on `self.runs` (a `Message`
        returned here today could be re-persisted, unchanged, in a future
        run's `RunOutput.messages`; mutating the stored copies in place
        would leak the "true when replayed" flag backwards into history
        that hadn't been replayed yet).
        """
        skip_statuses = skip_statuses if skip_statuses is not None else [RunStatus.error, RunStatus.cancelled]
        runs = self.runs[-last_n_runs:] if last_n_runs else self.runs

        messages: List[Message] = []
        for run in runs:
            if run.status in skip_statuses:
                continue
            for message in run.messages or []:
                if skip_roles and message.role in skip_roles:
                    continue
                messages.append(message.model_copy(update={"from_history": True}))

        if limit is not None:
            messages = messages[-limit:]
        return messages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_data": self.session_data,
            "runs": [run.to_dict() for run in self.runs],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSession":
        data = dict(data)

        runs_data = data.pop("runs", None) or []
        runs = [RunOutput.from_dict(r) if isinstance(r, dict) else r for r in runs_data]

        known = {f.name for f in fields(cls)} - {"runs"}
        filtered = {k: v for k, v in data.items() if k in known}

        return cls(runs=runs, **filtered)
