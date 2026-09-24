"""Pydantic request/response shapes for the OS layer's HTTP surface -
mirroring real Agno's `agno/os/schema.py`, trimmed to the three shapes this
stage's routers actually need.

`RunOutput`/`AgentSession` (both already Stage 3b/5 types with their own
hand-written `to_dict()`) are deliberately NOT re-modeled as Pydantic
schemas here - `os/routers/agents.py`/`sessions.py` return their `.to_dict()`
output directly as a plain `dict`, which FastAPI serializes fine on its own.
Building a parallel Pydantic mirror of those two dataclasses would be two
schemas to keep in sync with every future field added there for zero
validation benefit (they're *outputs*, never parsed back in from a request
body) - the two schemas below (`AgentSummary`, `SessionSummary`) exist only
for shapes that don't already have a `to_dict()` of their own.

Why `RunRequest` uses a JSON body (`BaseModel`) rather than `Form(...)`
(what real Agno's actual run-creation endpoint uses, since it also accepts
file uploads over multipart): this project has no media/file-upload
support, and `requirements.txt` doesn't pin `python-multipart` (the package
FastAPI needs to parse multipart forms at all) - a typed JSON body needs
neither, and is the more typed choice for `/docs` besides.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class RunRequest(BaseModel):
    """Body for `POST /agents/{agent_id}/runs`."""

    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    stream: bool = False


class AgentSummary(BaseModel):
    """One row of `GET /agents` - just enough to let a caller discover what
    `agent_id` values exist and what each one is running on, without
    leaking the full `Agent` config (tools, instructions, `db` handle)."""

    id: str
    name: Optional[str] = None
    model: Optional[str] = None
    model_provider: Optional[str] = None


class SessionSummary(BaseModel):
    """One row of `GET /sessions` - the listing view. `GET /sessions/{id}`
    (singular) returns the full `AgentSession.to_dict()` instead, runs and
    all; this is deliberately the lighter shape so listing many sessions
    doesn't ship every run's full message history over the wire just to
    render a list."""

    session_id: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    run_count: int
    created_at: int
    updated_at: int
