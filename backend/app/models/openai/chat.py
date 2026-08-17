"""OpenAI Chat Completions wire adapter - the three pieces that translate
between our internal types (`Message`, `ModelResponse`, `MessageMetrics`) and
the OpenAI SDK's types, mirroring `agno/models/openai/chat.py`.

This is deliberately just the translation layer, not the full `OpenAIChat`
model (no client construction, request params, or streaming/error-retry
plumbing yet - that's the `models/base.py` + provider-config stage). Three
functions, matching the three jobs a provider adapter has to do once per
model call:

    format_message()         Message -> OpenAI's wire dict (the request side)
    parse_provider_response() OpenAI's ChatCompletion -> our ModelResponse (the response side)
    get_metrics()             OpenAI's usage object -> our MessageMetrics (the accounting side)

Trimmed against real Agno: no media (images/audio/video/files), no
citations, no reasoning-content extraction, no provider_data passthrough -
none of that exists on our `Message`/`ModelResponse` yet. `get_metrics` only
maps the three token counts our `MessageMetrics` actually has; OpenAI's
audio/cache/reasoning token breakdown is dropped rather than silently
mismapped, since `MessageMetrics` (metrics.py) has nowhere to put it.
"""

from __future__ import annotations

from typing import Any, Dict

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from app.exceptions import ModelProviderError
from app.metrics import MessageMetrics
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
