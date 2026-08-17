"""`RunContext` and `RunStatus`, mirroring Agno's `agno/run/base.py` trimmed
to the fields this project needs (no workflow/team/knowledge-filter/output-
schema plumbing, since there's no workflow or team layer yet).

This is the module the rest of `run/` and the agent loop build on: every
model call and tool call during `Agent.run()` happens with one `RunContext`
in scope, and every run - completed, paused, or errored - ends in one of the
`RunStatus` states.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from app.models.message import Message


@dataclass
class RunContext:
    """Per-run state threaded through the agent loop and into tool calls."""

    run_id: str
    session_id: str
    user_id: Optional[str] = None

    dependencies: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    session_state: Optional[Dict[str, Any]] = None

    # Live reference to the current run's message list. Available in tool
    # hooks via run_context.messages. Individual Message objects are shared
    # references - do not mutate them; the list itself is the same object
    # the agent loop is appending to, not a copy.
    messages: Optional[List[Message]] = None


class RunStatus(str, Enum):
    """State of a run's response."""

    pending = "PENDING"
    running = "RUNNING"
    completed = "COMPLETED"
    paused = "PAUSED"
    cancelled = "CANCELLED"
    error = "ERROR"
