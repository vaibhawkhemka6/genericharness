"""`AgentRegistry` - the process-local `agent_id -> Agent` lookup table every
other piece of the OS layer reads from, mirroring the lookup half of real
Agno's `agno/os/app.py:AgentOS` (which bundles registry + FastAPI app
construction into one class) - split apart here so `create_app()`
(`os/app.py`) stays a thin assembly function rather than a class that's also
doing agent bookkeeping.

Per-process, not shared: each `create_app()` call builds its own
`AgentRegistry` from whatever `Agent` instances the caller passed in -
nothing here talks to a database. That's deliberate (see `os/app.py`'s
module docstring's block-diagram note): the *registry* just has to be
rebuilt identically at each process's startup from the same source code; the
*session/run data* those agents produce is what has to be shared across
processes, and that's `Db`'s job (`db/base.py`), not this module's.

`Agent.id` (`agent/agent.py`) defaults to a random `uuid4()` when not set
explicitly - fine for a single in-process run, but useless as a stable HTTP
route segment (`POST /agents/{agent_id}/runs`) or as the `agent_id` column
a persisted `AgentSession` gets tagged with, since a fresh random id on
every restart would orphan every session that referenced the old one. So:
every `Agent` handed to `AgentRegistry.register()` is expected to have been
constructed with an explicit, stable `id=` (e.g. `id="weather_agent"`) -
`register()` doesn't enforce this (there's no way to tell a "the caller
picked this on purpose" uuid4 apart from an accidental default one), but
it's the one thing a caller building agents for the OS layer has to get
right by hand.
"""

from __future__ import annotations

from typing import Dict, List

from wind.agent.agent import Agent


class AgentNotFoundError(Exception):
    """Raised by `AgentRegistry.resolve()` when `agent_id` isn't registered.
    Plain-data exception (just the id that was looked up) - `os/app.py`
    registers a FastAPI exception handler that turns this into a 404, so
    routers (`os/routers/agents.py`) can call `resolve()` directly without
    each one hand-rolling its own "not found" HTTP response.
    """

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"No agent registered with id={agent_id!r}")
        self.agent_id = agent_id


class AgentRegistry:
    """Holds every `Agent` a given OS process knows about, keyed by
    `agent.id`. Three operations, nothing else - no persistence, no
    hot-reloading a registered agent's config, no un-registering (a
    process's agent set is fixed for its lifetime; restarting the process is
    how you change it).
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        """Add `agent` under `agent.id`. Raises `ValueError` on a duplicate
        id rather than silently overwriting - two different `Agent` objects
        sharing an id is almost certainly a copy-paste mistake in how the
        caller built its agent list (`run_os.py`), not an intentional
        replace-in-place.
        """
        if agent.id in self._agents:
            raise ValueError(f"Agent id={agent.id!r} is already registered")
        self._agents[agent.id] = agent

    def resolve(self, agent_id: str) -> Agent:
        """Look up one agent by id, or raise `AgentNotFoundError`."""
        agent = self._agents.get(agent_id)
        if agent is None:
            raise AgentNotFoundError(agent_id)
        return agent

    def list_agents(self) -> List[Agent]:
        """Every registered agent, in registration order (plain dict
        insertion order - Python 3.7+ guarantees this, nothing special done
        here to preserve it)."""
        return list(self._agents.values())
