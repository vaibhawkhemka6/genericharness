"""`GET /sessions`, `GET /sessions/{session_id}`, `DELETE /sessions/{session_id}`
- thin wrappers exposing the `Db` (`db/base.py`) this project already built
in Stage 5 over HTTP, mirroring the session-listing half of real Agno's
`agno/os/routers/session.py`.

Reads the shared `Db` off `request.app.state.db` (set by `create_app()`,
`os/app.py`) - deliberately NOT `agent.db` off whichever agent the caller
happens to mention, since the block-diagram decision this stage is built
around is "the registry is per-process, the DB is shared": every agent in a
given OS process is expected to point at the same `Db` instance, and
sessions are addressed by `session_id` alone here, with no `agent_id` in
the URL - so there's no single agent to read `.db` off of in the first
place. `get_db()` raises a 500 if `create_app()` was never given a `db` at
all, rather than letting `request.app.state.db` be `None` and failing later
with a confusing `AttributeError` on `None.get_session(...)`.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request

from app.db.base import Db
from app.os.schemas import SessionSummary

router = APIRouter(prefix="/sessions", tags=["sessions"])


def get_db(request: Request) -> Db:
    db = request.app.state.db
    if db is None:
        raise HTTPException(status_code=500, detail="No db configured for this OS process")
    return db


@router.get("", response_model=List[SessionSummary])
def list_sessions(
    request: Request,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[SessionSummary]:
    db = get_db(request)
    sessions = db.get_sessions(agent_id=agent_id, user_id=user_id)
    return [
        SessionSummary(
            session_id=session.session_id,
            agent_id=session.agent_id,
            user_id=session.user_id,
            run_count=len(session.runs),
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
        for session in sessions
    ]


@router.get("/{session_id}")
def get_session(session_id: str, request: Request):
    """Full detail view - `AgentSession.to_dict()` as-is, runs and all
    (unlike the listing route's lighter `SessionSummary`)."""
    db = get_db(request)
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session with id={session_id!r}")
    return session.to_dict()


@router.delete("/{session_id}")
def delete_session(session_id: str, request: Request):
    db = get_db(request)
    db.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
