"""`GET /health` - the trivial route every other OS-layer piece gets built
and tested against first, per this stage's own build order: confirms
`create_app()` (`os/app.py`) produces something that actually boots and
serves a request before any registry/auth/streaming logic is layered on
top. No `dependencies=` here at the router level (auth is applied, if at
all, by `create_app()`'s `include_router()` call) and `create_app()`
deliberately does NOT put this router behind `require_auth` - a health
check that itself requires a valid API key isn't useful to whatever's
polling it (load balancers, container orchestrators) before traffic is
supposed to flow.
"""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
