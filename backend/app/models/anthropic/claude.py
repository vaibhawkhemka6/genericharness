"""Anthropic Messages API adapter - the pieces that translate between our
internal types and the `anthropic` SDK, mirroring
`agno/models/anthropic/claude.py`.

Same shape as `models/openai/chat.py` (a translation layer, not the full
provider `Model` dataclass - no client construction, streaming, or
retry/error-classification plumbing yet), plus `invoke()`: Anthropic's
request needs the `(chat_messages, system)` tuple `format_messages()`
produces threaded through to `client.messages.create()`, so building the
request and calling the API belongs here rather than being left to a caller.

    invoke()                  build the request, call the API, parse the result
    parse_provider_response() Anthropic's Message -> our ModelResponse (text blocks only)
    get_metrics()              Anthropic's Usage -> our MessageMetrics

Trimmed against real Agno: no beta features (skills, MCP connectors,
extended context management), no citations, no thinking/redacted-thinking
blocks, no structured-output (response_format) parsing, no
container/file-id bookkeeping. `get_metrics` only maps input/output/total
tokens - cache read/write token accounting is dropped for the same reason
as the OpenAI adapter: `MessageMetrics` (metrics.py) has nowhere to put it
yet.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage

from app.exceptions import ModelProviderError
from app.metrics import MessageMetrics, Timer
from app.models.message import Message
from app.models.response import ModelResponse
from app.utils.models.claude import format_messages


def invoke(
    client: Anthropic,
    model_id: str,
    messages: List[Message],
    assistant_message: Message,
    max_tokens: int = 4096,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> ModelResponse:
    """Send `messages` to the Anthropic API and return the parsed response.

    Times the call with `metrics.py`'s `Timer` and writes the result onto
    `assistant_message.metrics.duration`. Note this is a plain `Timer`, not
    `RunMetrics.start_timer()`/`stop_timer()` - `Message.metrics` is typed
    as `MessageMetrics`, which (per this project's metrics.py) doesn't carry
    its own timer; only `RunMetrics` does, for the whole-run total.
    """
    chat_messages, system_message = format_messages(messages)

    request_kwargs: Dict[str, Any] = {"max_tokens": max_tokens}
    if system_message:
        request_kwargs["system"] = system_message
    if tools:
        request_kwargs["tools"] = tools

    timer = Timer()
    try:
        timer.start()
        provider_response = client.messages.create(
            model=model_id,
            messages=chat_messages,  # type: ignore[arg-type]
            **request_kwargs,
        )
        timer.stop()
    except Exception as e:
        raise ModelProviderError(message=str(e), model_name=model_id, model_id=model_id) from e

    assistant_message.metrics.duration = timer.elapsed

    return parse_provider_response(provider_response)


def parse_provider_response(response: AnthropicMessage) -> ModelResponse:
    """Parse an Anthropic `Message` into our `ModelResponse` - text content
    and tool calls only (no thinking/citations/server-tool blocks)."""
    model_response = ModelResponse()
    model_response.role = response.role or "assistant"

    if response.content:
        for block in response.content:
            if block.type == "text":
                if model_response.content is None:
                    model_response.content = block.text
                else:
                    model_response.content += block.text

    if response.stop_reason == "tool_use":
        for block in response.content:
            if block.type == "tool_use":
                function_def: Dict[str, Any] = {"name": block.name}
                if block.input:
                    function_def["arguments"] = json.dumps(block.input)
                model_response.tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": function_def,
                    }
                )

    if response.usage is not None:
        model_response.response_usage = get_metrics(response.usage)

    return model_response


def get_metrics(response_usage: Usage) -> MessageMetrics:
    """Map Anthropic's usage object onto our `MessageMetrics`."""
    metrics = MessageMetrics()

    metrics.input_tokens = response_usage.input_tokens or 0
    metrics.output_tokens = response_usage.output_tokens or 0
    metrics.total_tokens = metrics.input_tokens + metrics.output_tokens

    return metrics
