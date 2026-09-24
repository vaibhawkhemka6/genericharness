"""`create_app()`/`serve()` - the assembly point for the whole OS layer,
mirroring real Agno's `agno/os/app.py:AgentOS.get_app()`/`.serve()` but kept
as two free functions operating on a plain `FastAPI` instance rather than a
class wrapping one - same "module-level function, not a fat method"
convention this project has used since `models/*/chat.py`'s translation
functions, so a future stage (a CLI, a test) can call `create_app(...)`
directly without instantiating anything else first.

Block-diagram summary (the picture this file wires into existence):

    HTTP client
        |
        v
    FastAPI app (this file)
      - CORSMiddleware, if OSSettings.cors_origins is non-empty
      - routers/health.py    - GET /health                     (no auth)
      - routers/agents.py    - GET /agents, POST /agents/{id}/runs  (auth)
      - routers/sessions.py  - GET/DELETE /sessions[/{id}]      (auth)
        |                                   |
        v                                   v
    AgentRegistry                          Db (shared)
    (per-process, in-memory,               (e.g. SqliteDb - the same
     rebuilt identically at                 instance every agent's
     every startup from the                 `.db` points at, and what
     same `agents=[...]` list)              routers/sessions.py reads
        |                                   directly)
        v
    Agent.run() / Agent.run(stream=True)   <- unchanged since Stage 3b/4/5
        |
        v
    Model.response() / .response_stream()  <- unchanged since Stage 2/4

Everything below "Agent.run()" is untouched by this stage - the OS layer is
purely a new front door onto code that already worked before this file
existed, proven again by `stage5_os.py` calling the exact same
`agent.run(...)` shapes directly (no HTTP) right alongside the HTTP-routed
versions and checking they agree.

Explicitly deferred (real Agno has these, this stage doesn't build them):
background runs + `GET /runs/{run_id}` polling (needs a `db.get_run()`-style
single-run lookup this project's lean `Db` doesn't have - `get_session()`
is the whole read surface; revisit once single-run lookup earns its keep),
JWT auth + `os/scopes.py` RBAC (needs a real user-identity system this
project doesn't have; static-key auth, `os/auth.py`, is the whole auth
story this stage builds), and the rest of real Agno's ~17-router surface
(evals, memory, knowledge, metrics, workflows - none of those subsystems
exist in this project).
"""

from __future__ import annotations

from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from wind.agent.agent import Agent
from wind.db.base import Db
from wind.os.auth import require_auth
from wind.os.registry import AgentNotFoundError, AgentRegistry
from wind.os.routers import agents as agents_router
from wind.os.routers import health as health_router
from wind.os.routers import sessions as sessions_router
from wind.os.settings import OSSettings


def create_app(
    agents: List[Agent],
    db: Optional[Db] = None,
    settings: Optional[OSSettings] = None,
) -> FastAPI:
    """Builds one ready-to-serve `FastAPI` app: registers every agent in
    `agents` into a fresh `AgentRegistry`, stashes `db`/`settings` on
    `app.state` for the routers/dependencies above to read at request time,
    adds CORS if configured, and wires up the three routers - `agents`/
    `sessions` behind `require_auth`, `health` in front of it.

    `db` is separate from each `Agent`'s own `agent.db` on purpose (see this
    module's docstring's block diagram) - an `Agent` still needs its own
    `db` set for `Agent.run()` itself to persist anything (unchanged from
    Stage 5), this parameter is what `routers/sessions.py` reads to serve
    `GET /sessions`. In the common case they're the same `SqliteDb`
    instance, passed to both - `run_os.py` shows that pattern.
    """
    settings = settings or OSSettings.from_env()

    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)

    app = FastAPI(title="Agent OS")
    app.state.agent_registry = registry
    app.state.db = db
    app.state.os_settings = settings

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(AgentNotFoundError)
    def _agent_not_found_handler(request: Request, exc: AgentNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    app.include_router(health_router.router)
    app.include_router(agents_router.router, dependencies=[Depends(require_auth)])
    app.include_router(sessions_router.router, dependencies=[Depends(require_auth)])

    return app


def serve(app: FastAPI, host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Blocking `uvicorn.run(...)` call - `host`/`port` default to whatever
    `OSSettings` this `app` was built with (`app.state.os_settings`), but
    can be overridden per-call without touching the app's own config.
    """
    settings: OSSettings = app.state.os_settings
    uvicorn.run(app, host=host or settings.host, port=port or settings.port)
