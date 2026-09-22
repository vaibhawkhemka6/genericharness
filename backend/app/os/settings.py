"""`OSSettings` - the handful of env-driven knobs the OS layer needs, read
once at `create_app()` time via `OSSettings.from_env()`.

Plain dataclass + a `from_env()` classmethod reading `os.environ` directly,
not Pydantic `BaseSettings` - this project's `requirements.txt` doesn't pin
`pydantic-settings` (the package that split out of Pydantic v2 for exactly
this), and pulling in a new dependency for four `os.getenv()` calls isn't
worth it. Every other env-driven value in this project (`ANTHROPIC_API_KEY`
et al.) is already read the same plain way, by each `Model` subclass's own
`__init__` - this just applies the identical pattern one layer up.

Reuses `FRONTEND_ORIGIN` (already in `.env`, already read by nothing yet in
this rebuild - the original Phase 1 prototype's `app/config.py` used it for
CORS before the Stage 0 rewrite) rather than inventing a new env var name
for the same concept.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OSSettings:
    """Config for one OS process. `os_security_key` unset (the default) is
    what puts the whole API in dev mode - see `os/auth.py`'s docstring for
    exactly what that means for `require_auth()`.
    """

    os_security_key: Optional[str] = None
    cors_origins: List[str] = field(default_factory=list)
    host: str = "0.0.0.0"
    port: int = 7777

    @classmethod
    def from_env(cls) -> "OSSettings":
        """Reads `OS_SECURITY_KEY`, `FRONTEND_ORIGIN` (comma-separated ->
        list, blank/unset -> empty list, meaning "no CORS middleware added"
        - see `os/app.py`), `OS_HOST`, `OS_PORT`. Every var is optional;
        missing envs fall back to the dataclass field defaults above, same
        "opt-in, sane defaults" shape as `Agent`'s own optional fields.
        """
        raw_origins = os.getenv("FRONTEND_ORIGIN", "")
        cors_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

        raw_port = os.getenv("OS_PORT")

        return cls(
            os_security_key=os.getenv("OS_SECURITY_KEY") or None,
            cors_origins=cors_origins,
            host=os.getenv("OS_HOST", "0.0.0.0"),
            port=int(raw_port) if raw_port else 7777,
        )
