"""OpenAI Chat Completions wire adapter - the three pieces that translate
between our internal types (`Message`, `ModelResponse`, `MessageMetrics`) and
the OpenAI SDK's types, mirroring `agno/models/openai/chat.py`.

Three module-level functions, matching the three jobs a provider adapter has
to do once per model call:

    format_message()         Message -> OpenAI's wire dict (the request side)
    parse_provider_response() OpenAI's ChatCompletion -> our ModelResponse (the response side)
    get_metrics()             OpenAI's usage object -> our MessageMetrics (the accounting side)

Trimmed against real Agno: no media (images/audio/video/files), no
citations, no reasoning-content extraction, no provider_data passthrough -
none of that exists on our `Message`/`ModelResponse` yet. `get_metrics` only
maps the three token counts our `MessageMetrics` actually has; OpenAI's
audio/cache/reasoning token breakdown is dropped rather than silently
mismapped, since `MessageMetrics` (metrics.py) has nowhere to put it.

`OpenAIChat` (below the three functions) is the Stage 2 addition: a
`Model` (`models/base.py`) subclass that owns client construction and
`invoke()` - build the request from `format_message()`, call the SDK, time
it, parse via `parse_provider_response()`, populate the assistant message
via `Model._populate_assistant_message()`. The three free functions stay
exactly as they were rather than becoming methods, since `stage0_openai.py`
calls them directly by hand to demonstrate the translation step by step -
folding them into the class would break that script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from app.exceptions import ModelProviderError
from app.metrics import MessageMetrics, Timer
from app.models.base import Model
from app.models.message import Message
from app.models.response import ModelResponse

# Our role names map straight onto OpenAI's, except "system" -> "developer"
# (OpenAI's newer models replaced the system role with a developer role;
# Agno keeps mapping "system" to it for backwards-compatible call sites).
DEFAULT_ROLE_MAP = {
    "system": "developer",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


def format_message(message: Message) -> Dict[str, Any]:
    """Format one `Message` into the dict shape OpenAI's Chat Completions
    API expects for a single entry in `messages=[...]`."""
    message_dict: Dict[str, Any] = {
        "role": DEFAULT_ROLE_MAP[message.role],
        "content": message.content,
        "name": message.name,
        "tool_call_id": message.tool_call_id,
        "tool_calls": message.tool_calls,
    }
    message_dict = {k: v for k, v in message_dict.items() if v is not None}

    # OpenAI expects tool_calls to be omitted/None if empty, not an empty list.
    if message.tool_calls is not None and len(message.tool_calls) == 0:
        message_dict["tool_calls"] = None

    return message_dict


def parse_provider_response(
    response: ChatCompletion,
    model_name: str | None = None,
    model_id: str | None = None,
) -> ModelResponse:
    """Parse an OpenAI `ChatCompletion` into our `ModelResponse`."""
    model_response = ModelResponse()

    if hasattr(response, "error") and response.error:  # type: ignore[attr-defined]
        raise ModelProviderError(
            message=response.error.get("message", "Unknown model error"),  # type: ignore[union-attr]
            model_name=model_name,
            model_id=model_id,
        )

    response_message = response.choices[0].message

    if response_message.role is not None:
        model_response.role = response_message.role

    if response_message.content is not None:
        model_response.content = response_message.content

    if response_message.tool_calls is not None and len(response_message.tool_calls) > 0:
        model_response.tool_calls = [t.model_dump() for t in response_message.tool_calls]

    if response.usage is not None:
        model_response.response_usage = get_metrics(response.usage)

    return model_response


def get_metrics(response_usage: CompletionUsage) -> MessageMetrics:
    """Map OpenAI's usage object onto our `MessageMetrics`."""
    metrics = MessageMetrics()

    metrics.input_tokens = response_usage.prompt_tokens or 0
    metrics.output_tokens = response_usage.completion_tokens or 0
    metrics.total_tokens = response_usage.total_tokens or 0

    return metrics


@dataclass
class OpenAIChat(Model):
    """`Model` subclass for OpenAI's Chat Completions API."""

    id: str = "gpt-4o-mini"
    provider: str = "OpenAI"

    def get_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

    def invoke(
        self,
        messages: List[Message],
        assistant_message: Message,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """One synchronous round-trip: format `messages`, call the OpenAI
        SDK, time it onto `assistant_message.metrics.duration`, parse the
        response, and write it onto `assistant_message`."""
        request_kwargs: Dict[str, Any] = {}
        if tools:
            request_kwargs["tools"] = tools

        timer = Timer()
        try:
            timer.start()
            response = self.get_client().chat.completions.create(
                model=self.id,
                messages=[format_message(m) for m in messages],
                **request_kwargs,
            )
            timer.stop()
        except Exception as e:
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e

        assistant_message.metrics.duration = timer.elapsed

        model_response = parse_provider_response(response, model_name=self.name, model_id=self.id)
        return self._populate_assistant_message(assistant_message, model_response)
