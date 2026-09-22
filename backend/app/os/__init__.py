"""The OS layer - a FastAPI front door onto the `Agent`s this project already
knows how to build, mirroring Agno's `agno/os/` package trimmed to what a
single-process deployment actually needs: an agent registry, one HTTP API
surface (run an agent, list/inspect/delete sessions), and static-key auth.

Everything below this package is untouched by this stage - `os/` doesn't
change how `Agent.run()`/`Agent.run(stream=True)` work, it just gives HTTP
callers a way to reach them. See `app/os/app.py`'s module docstring for the
full picture of what's built here vs. explicitly deferred (background runs,
JWT/RBAC, multi-router real-Agno feature surface).
"""
