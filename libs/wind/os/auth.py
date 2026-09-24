"""`require_auth()` - the one FastAPI dependency the OS layer's auth story
is built on, mirroring the static-key half of real Agno's
`agno/os/auth.py` (which also supports JWT - deliberately not built here,
see below).

Reads `OSSettings` off `request.app.state.os_settings` at *request* time,
not at import/router-definition time - `os/routers/*.py` stay auth-agnostic
(no `OSSettings` import, no `Depends(require_auth)` baked into the router
objects themselves; see `os/routers/__init__.py`'s docstring), and
`create_app()` (`os/app.py`) is the one place that decides which routers get
`dependencies=[Depends(require_auth)]` at `include_router()` time. Reading
`app.state` per-request (rather than closing over a specific `OSSettings`
instance built once) also means the same `require_auth` function object
works unmodified no matter which `FastAPI` app it's attached to - useful for
`stage5_os.py`, which builds more than one app in-process against different
`OSSettings`.

Dev-mode escape hatch: `settings.os_security_key` unset (the default -
`OSSettings.from_env()` with no `OS_SECURITY_KEY` in the environment) makes
`require_auth()` a no-op for every request. This project's `.env` has no
`OS_SECURITY_KEY` line, so a fresh checkout runs open by default - matches
this project's own `ANTHROPIC_BASE_URL` comment's spirit of "explicit is
safer than a silent default," applied the other way here: *not* setting a
key is the explicit, visible choice a `.env` file makes, not an accident.

Trimmed against real Agno's `agno/os/auth.py`: no JWT verification, no
per-route scope/permission checking (`os/scopes.py` in real Agno) - those
depend on a real user-identity system this project doesn't have yet. Static
bearer-token comparison via `hmac.compare_digest` (constant-time, avoids a
timing side-channel on the comparison itself) is the whole auth model this
stage builds.
"""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Header, HTTPException, Request

from wind.os.settings import OSSettings


def require_auth(request: Request, authorization: Optional[str] = Header(None)) -> None:
    """FastAPI dependency: raises `HTTPException(401)` if
    `request.app.state.os_settings.os_security_key` is set and the
    `Authorization` header doesn't present it as `Bearer <key>`. Returns
    `None` (the dependency's only job is to raise-or-not) on success or
    when auth isn't configured at all.
    """
    settings: OSSettings = request.app.state.os_settings

    if not settings.os_security_key:
        return

    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header required")

    expected = f"Bearer {settings.os_security_key}"
    if not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid authorization credentials")
