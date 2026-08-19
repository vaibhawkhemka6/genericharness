"""`Model` - the abstract base every provider adapter (`OpenAIChat`, `Claude`)
implements, mirroring Agno's `agno/models/base.py` trimmed to a single
synchronous round-trip: build the request, call the SDK, parse the response,
write it onto the assistant `Message`.

This is deliberately not real Agno's `Model` (~3100 lines, `models/base.py`
on GitHub): no streaming (sync or async), no `ainvoke`, no response caching
to disk, no retry/backoff, no structured-output (`response_format`)
plumbing, and no tool-execution loop - running the tools a model asked for
is the agent loop's job (a later stage, once one exists in this rebuild),
not the model adapter's. What's left is the one thing every provider
adapter must do the same way: take `List[Message]` + optional tool
declarations in, get a `ModelResponse` out, and copy that response onto the
`Message` the caller will append to history.

`OpenAIChat` (`models/openai/chat.py`) and `Claude` (`models/anthropic/claude.py`)
are the two concrete subclasses. Each still exports its translation
functions (`format_message`/`format_messages`, `parse_provider_response`,
`get_metrics`) as plain module-level functions rather than methods - the
Stage 0/0.5 test scripts (`stage0_openai.py`, `stage0_anthropic.py`) call
those directly, by hand, to walk through request/response translation step
by step, so they can't be folded into the class without breaking those
scripts. `invoke()` is the new piece each class adds: it wires client
construction + those existing functions + this base class's shared
response-populating logic together into one call, which is what "provider-
agnostic" means from a caller's point of view - construct either subclass,
call `.invoke(messages, assistant_message, tools)`, get a `ModelResponse`
back, regardless of which provider is underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.models.message import Message
from app.models.response import ModelResponse


@dataclass
class Model(ABC):
    """Provider-agnostic interface: one model call in, one `ModelResponse` out."""

    # ID of the model to use, e.g. "gpt-4o-mini" or "claude-sonnet-4-5-20250929".
    id: str
    # Human-readable name for this model. Not sent to the provider API.
    name: Optional[str] = None
    # Provider label, e.g. "OpenAI" or "Anthropic". Not sent to the provider API.
    provider: Optional[str] = None
    # API key override. If None, each adapter's get_client() falls back to
    # the provider SDK's normal env-var lookup (OPENAI_API_KEY / ANTHROPIC_API_KEY).
    api_key: Optional[str] = None

    def get_provider(self) -> str:
        return self.provider or self.__class__.__name__

    @abstractmethod
    def invoke(
        self,
        messages: List[Message],
        assistant_message: Message,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Send `messages` to the provider and return the parsed `ModelResponse`.

        Implementations are expected to: format `messages` into the
        provider's wire shape, call the SDK, time the call onto
        `assistant_message.metrics.duration`, parse the raw response via
        that module's `parse_provider_response()`, and finish by returning
        `self._populate_assistant_message(assistant_message, model_response)`
        so every provider adapter writes the response onto the message the
        same way.
        """
        raise NotImplementedError

    def _populate_assistant_message(self, assistant_message: Message, model_response: ModelResponse) -> Message:
        """Copy a parsed `ModelResponse` onto the `Message` the caller will
        append to history. Shared by every `invoke()` implementation so the
        response -> message mapping only lives in one place, not once per
        provider.

        Does *not* touch `assistant_message.metrics.duration` - each
        `invoke()` sets that itself from its own request timer, before the
        response even exists to parse from.
        """
        if model_response.role is not None:
            assistant_message.role = model_response.role
        if model_response.content is not None:
            assistant_message.content = model_response.content
        if model_response.tool_calls:
            assistant_message.tool_calls = model_response.tool_calls
        if model_response.response_usage is not None:
            assistant_message.metrics.input_tokens = model_response.response_usage.input_tokens
            assistant_message.metrics.output_tokens = model_response.response_usage.output_tokens
            assistant_message.metrics.total_tokens = model_response.response_usage.total_tokens
        return assistant_message

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "provider": self.get_provider()}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id!r}>"
