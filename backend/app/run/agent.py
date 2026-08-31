"""`RunInput` and `RunOutput`, mirroring Agno's `agno/run/agent.py` trimmed to
the fields this project needs (no media, reasoning, citations, HITL
requirements, forking/checkpointing, or events - none of that exists yet).

`RunInput` captures the raw input to `Agent.run()` exactly as the caller
passed it in, kept separate from the processed `RunMessages` that actually go
to the model - useful for logging/replay/debugging without reconstructing
what the user originally sent. `RunOutput` is the terminal artifact of a run:
everything a caller needs after `Agent.run()` returns (or after re-hydrating
a persisted run from the DB).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from app.metrics import RunMetrics
from app.models.message import Message
from app.models.response import ToolExecution
from app.run.base import RunStatus


@dataclass
class RunInput:
    """Container for the raw input data passed to `Agent.run()`.

    Captures the original input exactly as provided by the caller, separate
    from the processed messages that go to the model.
    """

    input_content: Union[str, List, Dict, Message, BaseModel, List[Message]]

    def input_content_string(self) -> str:
        if isinstance(self.input_content, str):
            return self.input_content
        elif isinstance(self.input_content, BaseModel):
            return self.input_content.model_dump_json(exclude_none=True)
        elif isinstance(self.input_content, Message):
            return json.dumps(self.input_content.to_dict(), ensure_ascii=False)
        elif isinstance(self.input_content, list):
            try:
                return json.dumps(self.to_dict().get("input_content"), ensure_ascii=False)
            except Exception:
                return str(self.input_content)
        else:
            return str(self.input_content)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        if self.input_content is not None:
            if isinstance(self.input_content, str):
                result["input_content"] = self.input_content
            elif isinstance(self.input_content, BaseModel):
                result["input_content"] = self.input_content.model_dump(exclude_none=True)
            elif isinstance(self.input_content, Message):
                result["input_content"] = self.input_content.to_dict()
            elif isinstance(self.input_content, list):
                serialized_items: List[Any] = []
                for item in self.input_content:
                    if isinstance(item, Message):
                        serialized_items.append(item.to_dict())
                    elif isinstance(item, BaseModel):
                        serialized_items.append(item.model_dump(exclude_none=True))
                    else:
                        serialized_items.append(item)
                result["input_content"] = serialized_items
            else:
                result["input_content"] = self.input_content

        return result


@dataclass
class RunOutput:
    """Response returned by `Agent.run()`."""

    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Input passed to Agent.run()
    input: Optional[RunInput] = None

    content: Optional[Any] = None
    content_type: str = "str"

    model: Optional[str] = None
    model_provider: Optional[str] = None
    messages: Optional[List[Message]] = None
    metrics: Optional[RunMetrics] = None

    tools: Optional[List[ToolExecution]] = None

    metadata: Optional[Dict[str, Any]] = None
    session_state: Optional[Dict[str, Any]] = None

    created_at: int = field(default_factory=lambda: int(time()))

    status: RunStatus = RunStatus.running

    def get_content_as_string(self, **kwargs) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, BaseModel):
            return self.content.model_dump_json(exclude_none=True, **kwargs)
        else:
            kwargs.setdefault("ensure_ascii", False)
            return json.dumps(self.content, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        _dict = {
            k: v
            for k, v in asdict(self).items()
            if v is not None and k not in ["messages", "metrics", "tools", "metadata", "input"]
        }

        if self.metrics is not None:
            _dict["metrics"] = self.metrics.to_dict() if isinstance(self.metrics, RunMetrics) else self.metrics

        if self.status is not None:
            _dict["status"] = self.status.value if isinstance(self.status, RunStatus) else self.status

        if self.messages is not None:
            _dict["messages"] = [m.to_dict() for m in self.messages]

        if self.metadata is not None:
            _dict["metadata"] = self.metadata

        if self.tools is not None:
            _dict["tools"] = [t.to_dict() if isinstance(t, ToolExecution) else t for t in self.tools]

        if self.input is not None:
            _dict["input"] = self.input.to_dict()

        return _dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunOutput":
        data = dict(data)

        messages = data.pop("messages", None)
        messages = [Message.from_dict(m) for m in messages] if messages else None

        tools = data.pop("tools", None)
        tools = [ToolExecution.from_dict(t) for t in tools] if tools else None

        input_data = data.pop("input", None)
        input_obj = RunInput(input_content=input_data.get("input_content")) if input_data else None

        metrics = data.pop("metrics", None)
        if metrics:
            metrics = RunMetrics.from_dict(metrics)

        status = data.pop("status", None)
        if status is not None and not isinstance(status, RunStatus):
            status = RunStatus(status)

        from dataclasses import fields

        supported_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in supported_fields}

        return cls(
            messages=messages,
            metrics=metrics,
            tools=tools,
            input=input_obj,
            **({"status": status} if status is not None else {}),
            **filtered_data,
        )


# --- Stage 4 addition: the CONSUMER-facing event vocabulary -----------------
#
# `agent/_response.py:handle_model_response_stream()` is the one place that
# builds these - it watches `Model.response_stream()`'s `ModelResponse`
# deltas (`models/base.py`) and translates each into exactly one of the
# dataclasses below, which is what `Agent.run(stream=True)` actually yields
# to a caller. Same envelope-plus-marker trick as `ModelResponse.event`
# (`models/response.py`) one layer down: every event dataclass shares
# `BaseAgentRunEvent`'s envelope fields (who/when) plus its own `event`
# discriminator string, so a caller iterating the stream can dispatch on
# `.event` alone without isinstance-checking each subclass.
#
# Trimmed against real Agno's `agno/run/agent.py` `RunEvent`/event-class
# set: this project builds only the events `response_stream()` actually has
# a reason to emit - a content delta, the run's final result, a model
# call's start/end, and one tool call's start/end. Left out entirely (per
# this stage's explicit skip-for-lean list): run_paused/run_continued
# (HITL - not built), pre_hook_*/post_hook_* (hooks - not built),
# reasoning_* (reasoning - not built), memory_update_*/session_summary_*
# (memory/session-summary - not built), parser_model_response_*/
# output_model_response_* (parser/output model - not built), compression_*
# (CompressionManager - not built), followups_* (followups - not built),
# tool_call_error (no HITL/error surfacing built for tool calls yet),
# custom_event (no custom-event escape hatch built).


class RunEvent(str, Enum):
    """Event-type vocabulary for `Agent.run(stream=True)`'s output."""

    run_started = "RunStarted"
    run_content = "RunContent"
    run_completed = "RunCompleted"
    run_error = "RunError"
    run_cancelled = "RunCancelled"

    model_request_started = "ModelRequestStarted"
    model_request_completed = "ModelRequestCompleted"

    tool_call_started = "ToolCallStarted"
    tool_call_completed = "ToolCallCompleted"


@dataclass
class BaseAgentRunEvent:
    """Envelope fields every event below carries - who (`agent_id`/
    `agent_name`/`run_id`/`session_id`) and when (`created_at`), plus
    `event` itself, the field a consumer dispatches on."""

    event: str = ""
    created_at: int = field(default_factory=lambda: int(time()))
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    run_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class RunStartedEvent(BaseAgentRunEvent):
    """Sent once, before the first model call of a run."""

    event: str = RunEvent.run_started.value
    model: Optional[str] = None
    model_provider: Optional[str] = None


@dataclass
class RunContentEvent(BaseAgentRunEvent):
    """One content delta - the agent-layer sibling of a plain-text
    `ModelResponse` chunk (`event=None` at the model layer, matching
    `RunEvent.run_content` up here)."""

    event: str = RunEvent.run_content.value
    content: Optional[Any] = None
    content_type: str = "str"


@dataclass
class RunCompletedEvent(BaseAgentRunEvent):
    """Sent once, after the run's loop has fully exited - the streaming
    sibling of a non-streaming `Agent.run()`'s returned `RunOutput`."""

    event: str = RunEvent.run_completed.value
    content: Optional[Any] = None
    content_type: str = "str"
    metrics: Optional[RunMetrics] = None


@dataclass
class RunErrorEvent(BaseAgentRunEvent):
    """Sent instead of `RunCompletedEvent` if the model call raised
    `ModelProviderError` - the streaming sibling of `_run.py:run()`'s
    `except ModelProviderError` branch."""

    event: str = RunEvent.run_error.value
    content: Optional[str] = None


@dataclass
class RunCancelledEvent(BaseAgentRunEvent):
    """Sent instead of `RunCompletedEvent` if a tool raised
    `RunCancelledException` mid-stream - the streaming sibling of
    `_run.py:run()`'s `except RunCancelledException` branch."""

    event: str = RunEvent.run_cancelled.value
    content: Optional[str] = None


@dataclass
class ModelRequestStartedEvent(BaseAgentRunEvent):
    """One per `response_stream()` iteration, sent before that iteration's
    `invoke_stream()` sub-loop starts - the agent-layer translation of the
    model layer's `ModelResponseEvent.model_request_started`."""

    event: str = RunEvent.model_request_started.value
    model: Optional[str] = None
    model_provider: Optional[str] = None


@dataclass
class ModelRequestCompletedEvent(BaseAgentRunEvent):
    """One per `response_stream()` iteration, sent after that iteration's
    assistant message is fully accumulated - the agent-layer translation of
    `ModelResponseEvent.model_request_completed`, carrying that iteration's
    token counts/TTFT straight across."""

    event: str = RunEvent.model_request_completed.value
    model: Optional[str] = None
    model_provider: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    time_to_first_token: Optional[float] = None


@dataclass
class ToolCallStartedEvent(BaseAgentRunEvent):
    """Sent right before a tool call executes - `tool` is a partial
    `ToolExecution` (id/name/args set, `result`/`metrics` still None), the
    agent-layer translation of `ModelResponseEvent.tool_call_started`."""

    event: str = RunEvent.tool_call_started.value
    tool: Optional[ToolExecution] = None


@dataclass
class ToolCallCompletedEvent(BaseAgentRunEvent):
    """Sent right after a tool call executes - `tool` is the same
    `ToolExecution` object, now finished (`result`/`metrics` set), the
    agent-layer translation of `ModelResponseEvent.tool_call_completed`."""

    event: str = RunEvent.tool_call_completed.value
    tool: Optional[ToolExecution] = None


# What Agent.run(stream=True) yields, one instance at a time.
RunOutputEvent = Union[
    RunStartedEvent,
    RunContentEvent,
    RunCompletedEvent,
    RunErrorEvent,
    RunCancelledEvent,
    ModelRequestStartedEvent,
    ModelRequestCompletedEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
]
