"""`GET /agents` (discover what's registered) and
`POST /agents/{agent_id}/runs` (the actual point of this whole stage) -
mirroring the run-creation half of real Agno's `agno/os/routers/agent.py`,
trimmed to non-streaming + streaming (no background runs - see
`os/app.py`'s module docstring for why that's explicitly deferred, not
missed).

`get_registry()` below is this router's only dependency on `create_app()`
having run first - it reads `request.app.state.agent_registry` at request
time, same "read `app.state`, don't close over anything at import time"
shape `os/auth.py:require_auth()` uses, and for the same reason: this
module has to stay importable/registrable without a live app already
existing (`stage5_os.py` builds more than one `create_app()` in-process).
"""

from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.os.registry import AgentRegistry
from app.os.schemas import AgentSummary, RunRequest
from app.os.streaming import agent_event_stream

router = APIRouter(prefix="/agents", tags=["agents"])


def get_registry(request: Request) -> AgentRegistry:
    return request.app.state.agent_registry


@router.get("", response_model=List[AgentSummary])
def list_agents(request: Request) -> List[AgentSummary]:
    registry = get_registry(request)
    return [
        AgentSummary(
            id=agent.id,
            name=agent.name,
            model=agent.model.id,
            model_provider=agent.model.get_provider(),
        )
        for agent in registry.list_agents()
    ]


@router.post("/{agent_id}/runs")
def create_agent_run(agent_id: str, run_request: RunRequest, request: Request) -> Any:
    """Non-streaming branch returns `RunOutput.to_dict()` (Stage 3b/5's own
    serialization - not re-modeled as a Pydantic schema, see `schemas.py`'s
    docstring) with HTTP 200 regardless of `RunOutput.status` - a run that
    completed with `status=error` (e.g. the model provider's API call
    failed) is still a *successful* HTTP request that successfully reported
    a failed run; that distinction is `RunOutput.status`'s job to carry, not
    the HTTP status code's (see `agent/_run.py`'s own "report what
    happened, don't raise" design, unchanged since Stage 3b).

    Streaming branch returns `EventSourceResponse` wrapping
    `agent_event_stream()` (`os/streaming.py`) - same underlying
    `agent.run(stream=True)` call, framed for SSE.

    `agent_id` not found raises `AgentNotFoundError` (`os/registry.py`),
    left to propagate to the exception handler `create_app()` registers
    (`os/app.py`) rather than caught here - keeps this handler from needing
    to know its own error-to-status-code mapping.
    """
    if not run_request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    registry = get_registry(request)
    agent = registry.resolve(agent_id)

    if run_request.stream:
        return EventSourceResponse(
            agent_event_stream(
                agent,
                run_request.message,
                session_id=run_request.session_id,
                user_id=run_request.user_id,
            )
        )

    run_output = agent.run(
        run_request.message,
        session_id=run_request.session_id,
        user_id=run_request.user_id,
    )
    return run_output.to_dict()
