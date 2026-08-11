"""Provider-agnostic model interface.

Every adapter (Claude, OpenAI, ...) implements `stream()` yielding a common
set of events, and `build_tool_result_messages()` to shape tool results into
whatever the provider's message format expects. The Agent loop only talks to
this interface - it never knows which provider it's calling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union

from app.tools.base import Tool


@dataclass
class TextDelta:
    """A chunk of assistant text, as it streams in."""

    text: str


@dataclass
class ToolCall:
    """A single fully-parsed tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class StreamEnd:
    """Terminal event for one model turn.

    `assistant_message` is the provider-native message shape (already
    matching whatever `messages` list this provider expects) - the Agent
    appends it directly to conversation history.
    """

    stop_reason: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: Any = None


ModelEvent = Union[TextDelta, StreamEnd]


class Model(ABC):
    """Abstract base for all model provider adapters."""

    id: str

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[Tool]] = None,
        system: Optional[str] = None,
    ) -> Iterator[ModelEvent]:
        """Stream one model turn. Yields TextDelta chunks, ends with StreamEnd."""
        raise NotImplementedError

    @abstractmethod
    def build_tool_result_messages(
        self, results: list[tuple[ToolCall, Any]]
    ) -> list[dict[str, Any]]:
        """Turn (ToolCall, result) pairs into provider-native history message(s)."""
        raise NotImplementedError
