"""`Agent` - the minimal config container Stage 3 Half B's orchestration
(`_run.py`/`_messages.py`/`_response.py`) operates on.

Not in the original Half B file list, but structurally required by it: every
function in that list takes an `agent` parameter (`get_system_message(agent)`,
`get_run_messages(agent, ...)`, `_run.run(agent, ...)`) and reads
`agent.description`/`agent.instructions`/`agent.model`/`agent.tools`/etc. off
of it - there has to be *something* holding those fields before any of Half
B can run. Kept intentionally tiny: no session/memory/knowledge wiring
(Stage 5+), no hooks, no HITL, no reasoning/output-schema config - just the
handful of fields Half B's message assembly and orchestration actually read
this stage.

`session_history`/`num_history_messages` are the "stub standing in for
session/db" the build order calls for: a plain list the caller already has
in memory (or sets here once, for convenience), not something `Agent.run()`
fetches from a real store - `_messages.py:get_session_history_messages()` is
what reads them.

`Agent.run()` itself is a two-line dispatch to `_run.run()` - the free
function is where the actual orchestration logic lives (same "module-level
function, not a fat method" pattern as `models/*/chat.py`'s translation
functions), so a future stage script can call `run(agent, ...)` directly,
by hand, without going through the class if useful. Imported lazily inside
the method (not at module level) to avoid a real import cycle: `_run.py`
imports `Agent` (for type hints) at its own module level, so `agent.py`
can't import `_run.py` back at module level too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Union
from uuid import uuid4

from app.db.base import Db
from app.models.base import Model
from app.models.message import Message
from app.tools.function import Function
from app.tools.toolkit import Toolkit

if TYPE_CHECKING:
    from app.run.agent import RunOutput, RunOutputEvent


@dataclass
class Agent:
    """Provider-agnostic agent config: a model, its tools, and the system
    prompt pieces (`description`/`instructions`) `_messages.py` assembles
    into a system message."""

    model: Model

    id: str = field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = None

    # Concatenated (description, then instructions) by get_system_message()
    # into the run's system Message. instructions may be one string or a
    # list of lines, rendered as "- " bullets.
    description: Optional[str] = None
    instructions: Optional[Union[str, List[str]]] = None

    # Whatever Toolkit.register() already accepts: plain callables,
    # @tool-decorated Functions, or whole Toolkits. Resolved into a flat
    # Dict[str, Function] by _run.py:resolve_tools(), not here - Agent
    # itself doesn't need the resolved shape, only what the caller gave it.
    tools: Optional[List[Union[Callable, Function, Toolkit]]] = None
    tool_call_limit: Optional[int] = None

    # Stage 5 addition: the real session/db backend. None (default) keeps
    # every earlier stage's behavior exactly as it was - get_run_messages()
    # falls back to the session_history stub below whenever this is unset,
    # rather than requiring a db to run at all.
    db: Optional[Db] = None

    # Pre-Stage-5 stub standing in for session/db: a plain list the caller
    # hands in (or sets here) rather than something Agent.run() fetches
    # from a real store. get_run_messages() reads this only when `db` is
    # None (or the caller passes an explicit session_history override) -
    # once `db` is set, real session history (loaded via
    # agent/_storage.py:read_or_create_session()) takes over instead.
    session_history: Optional[List[Message]] = None
    num_history_messages: Optional[int] = None

    def run(
        self,
        input,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_history: Optional[List[Message]] = None,
        stream: bool = False,
    ) -> Union["RunOutput", Iterator["RunOutputEvent"]]:
        """Two-line dispatch, same as the module docstring describes - the
        only new piece this stage adds is `stream`: `False` (default) still
        goes to `_run.run()` and blocks for one `RunOutput`; `True` goes to
        `_run.run_stream()` instead and returns a generator of
        `RunOutputEvent`s the caller iterates as the model streams. Which
        one actually runs is decided here, once, rather than `_run.py`
        having two public entry points a caller has to choose between
        directly.
        """
        if stream:
            from app.agent._run import run_stream as _run_stream

            return _run_stream(self, input, session_id=session_id, user_id=user_id, session_history=session_history)

        from app.agent._run import run as _run

        return _run(self, input, session_id=session_id, user_id=user_id, session_history=session_history)
