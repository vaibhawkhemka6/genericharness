"""FastAPI `APIRouter`s for the OS layer - one module per resource
(`health.py`, `agents.py`, `sessions.py`), assembled by `os/app.py`'s
`create_app()`. Kept auth-agnostic on purpose: none of these routers import
`os/auth.py` or attach `dependencies=` to themselves - `create_app()` is the
one place that decides which routers require auth, via
`include_router(..., dependencies=[Depends(require_auth)])`, so a router
module can be imported and tested (see `stage5_os.py`) without needing a
live `OSSettings` wired up first.
"""
