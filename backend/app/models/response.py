"""Model-adapter-facing response types, mirroring Agno's
`agno/models/response.py` trimmed to the token/tool-execution subset this
project needs (no media, citations, or HITL confirmation fields).

Plain `@dataclass`, not Pydantic - unlike `Message` in `message.py`, these
never cross an API boundary directly. They're internal plumbing a provider
adapter builds and the agent loop consumes on the way to producing a
`Message`/`RunOutput`, so there's nothing here that needs request-body
validation.

`ToolExecution` records one tool call's outcome (what ran, what it
returned, whether it errored); `ModelResponse` is what a provider adapter
hands back to the agent loop for a single model call - the raw tool_calls
plus already-executed ToolExecutions accumulate as the loop works through
one iteration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import time
from typing import Any, Dict, List, Optional

from app.metrics import MessageMetrics


@dataclass
class ToolExecution:
    """The execution of a single tool call - request and result together."""

    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_error: Optional[bool] = None
    result: Optional[str] = None
    metrics: Optional[MessageMetrics] = None

    # If True, the agent stops executing after this tool call.
    stop_after_tool_call: bool = False

    created_at: int = field(default_factory=lambda: int(time()))

    def to_dict(self) -> Dict[str, Any]:
        _dict = asdict(self)
        if self.metrics is not None:
            _dict["metrics"] = self.metrics.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolExecution":
        data = dict(data)
        metrics = data.pop("metrics", None)
        return cls(
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
            tool_call_error=data.get("tool_call_error"),
            result=data.get("result"),
            stop_after_tool_call=data.get("stop_after_tool_call", False),
            metrics=MessageMetrics.from_dict(metrics) if metrics else None,
            **({"created_at": data["created_at"]} if "created_at" in data else {}),
        )


@dataclass
class ModelResponse:
    """One model call's response, as handed back to the agent loop by a
    provider adapter."""

    role: Optional[str] = None

    content: Optional[Any] = None
    parsed: Optional[Any] = None

    # Raw tool calls as returned by the provider (arguments still a JSON
    # string) - same shape as Message.tool_calls.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    # Tool calls that have actually been executed this iteration.
    tool_executions: Optional[List[ToolExecution]] = field(default_factory=list)

    response_usage: Optional[MessageMetrics] = None

    created_at: int = field(default_factory=lambda: int(time()))

    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        _dict = asdict(self)

        if self.tool_executions is not None:
            _dict["tool_executions"] = [te.to_dict() for te in self.tool_executions]

        response_usage = _dict.pop("response_usage", None)
        if response_usage is not None:
            _dict["response_usage"] = response_usage

        return _dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelResponse":
        data = dict(data)

        if data.get("tool_executions"):
            data["tool_executions"] = [ToolExecution.from_dict(te) for te in data["tool_executions"]]

        if data.get("response_usage") and isinstance(data["response_usage"], dict):
            data["response_usage"] = MessageMetrics.from_dict(data["response_usage"])

        return cls(**data)
