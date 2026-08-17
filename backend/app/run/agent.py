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
