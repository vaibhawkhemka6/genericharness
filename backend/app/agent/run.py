"""Streamed event types emitted by Agent.run().

These are what the FastAPI layer serializes over SSE, and what the frontend
renders (text deltas vs. distinct tool-call blocks, similar to Claude's UI).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

# type values: run_started | content_delta | tool_call_started | tool_call_result | run_completed | error


@dataclass
class RunEvent:
    type: str
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict[str, Any]] = None
    tool_result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
