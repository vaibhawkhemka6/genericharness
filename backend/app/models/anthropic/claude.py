"""Anthropic Messages API adapter - the pieces that translate between our
internal types and the `anthropic` SDK, mirroring
`agno/models/anthropic/claude.py`.

Same shape as `models/openai/chat.py`: module-level functions doing the
translation, plus a free `invoke()` that threads the `(chat_messages,
system)` tuple `format_messages()` produces through to
`client.messages.create()` given an explicit `client`/`model_id`.

    invoke()                  build the request, call the API, parse the result (explicit client)
    parse_provider_response() Anthropic's Message -> our ModelResponse (text blocks only)
    get_metrics()              Anthropic's Usage -> our MessageMetrics

Trimmed against real Agno: no beta features (skills, MCP connectors,
extended context management), no citations, no thinking/redacted-thinking
blocks, no structured-output (response_format) parsing, no
container/file-id bookkeeping. `get_metrics` only maps input/output/total
tokens - cache read/write token accounting is dropped for the same reason
as the OpenAI adapter: `MessageMetrics` (metrics.py) has nowhere to put it
yet.

`Claude` (below the free functions) is the Stage 2 addition: a `Model`
(`models/base.py`) subclass that owns its own client instead of taking one
as a parameter - `.invoke(messages, assistant_message, tools)` builds the
client from `self.api_key`, calls `format_messages()`/`parse_provider_response()`
itself, and populates the assistant message via
`Model._populate_assistant_message()`. It doesn't replace the free
`invoke()` function above (kept as-is - unused internally, but harmless to
keep as a lower-level building block) or any of `format_messages()`/
`parse_provider_response()`/`get_metrics()`, which `stage0_anthropic.py`
still calls directly by hand.

Stage 3: `Model.response()` (`models/base.py`) always builds and passes
tool declarations in the OpenAI-shaped *canonical* form
(`utils/models/openai.py:format_tools_for_model`) - it doesn't know or care
which provider it's talking to. `Claude.invoke()` is where that gets
converted to Anthropic's `input_schema` shape, via this module's own
`format_tools_for_model()`, right before the request is built - keeping
that conversion inside the provider adapter, not `response()`, is what lets
`response()` stay provider-agnostic.

Stage 4 addition - `invoke_stream()` and `parse_provider_response_delta()`,
the streaming TRANSPORT layer for this provider. Notably simpler than
OpenAI's on the tool-call side, for one specific reason: this uses the
`anthropic` SDK's *high-level* `client.messages.stream(...)` context
manager (not the raw `stream=True` event union) - it accumulates each
content block for you internally, so the event this module reads for a
`tool_use` block, `content_block_stop`, already carries the fully-formed
block (`.content_block.input` is a complete parsed dict, not a partial JSON
string fragment). OpenAI's chunks carry raw argument-string *fragments*
addressed by index, needing `OpenAIChat.parse_tool_calls()`'s explicit
by-index merge (Phase B); here, each `content_block_stop` for a `tool_use`
block already IS one complete tool call - `parse_provider_response_delta()`
below yields it as a finished dict in one shot, and `Model.parse_tool_calls()`'s
no-op default (`models/base.py`) is all `Claude` needs - there is nothing
left for Phase B to reassemble.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from anthropic import Anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage

from app.exceptions import ModelProviderError
from app.metrics import MessageMetrics, Timer
from app.models.base import Model
from app.models.message import Message
from app.models.response import ModelResponse
from app.utils.models.claude import format_messages
from app.utils.models.claude import format_tools_for_model as format_tools_for_claude


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


def parse_provider_response_delta(event: Any) -> ModelResponse:
    """Parse one event from `client.messages.stream(...)`'s high-level
    iterator into a `ModelResponse` delta - the streaming sibling of
    `parse_provider_response()` above.

    Only three of the SDK's accumulated event types carry anything this
    project's trimmed `ModelResponse`/`MessageMetrics` have a field for
    (duck-typed on `.type` rather than imported event classes - the SDK's
    accumulated types live under a private `anthropic.lib.streaming`
    submodule):

      - `content_block_delta` with `delta.type == "text_delta"` - a plain
        text fragment (`delta.text`).
      - `content_block_stop` where the now-complete `content_block.type ==
        "tool_use"` - see this module's docstring for why this is already
        one whole, mergeable-with-nothing tool call, unlike OpenAI's
        fragments.
      - `message_stop` - carries the final accumulated `message.usage` for
        the whole turn (Anthropic reports usage once, at the end, not per
        chunk the way OpenAI's trailing chunk does).

    Every other event type (`message_start`, `content_block_start`, the
    `input_json_delta` deltas that fire *while* a tool_use block is still
    being accumulated, `message_delta`) produces an empty `ModelResponse` -
    nothing on it is set, so `_populate_stream_data()`
    (`models/base.py`, Phase A) sees no reason to yield it upward.
    """
    model_response = ModelResponse()
    event_type = getattr(event, "type", None)

    if event_type == "content_block_delta":
        delta = event.delta
        if getattr(delta, "type", None) == "text_delta":
            model_response.content = delta.text

    elif event_type == "content_block_stop":
        block = event.content_block
        if getattr(block, "type", None) == "tool_use":
            function_def: Dict[str, Any] = {"name": block.name}
            if block.input:
                function_def["arguments"] = json.dumps(block.input)
            model_response.tool_calls = [
                {
                    "id": block.id,
                    "type": "function",
                    "function": function_def,
                }
            ]

    elif event_type == "message_stop":
        message = getattr(event, "message", None)
        if message is not None and message.usage is not None:
            model_response.response_usage = get_metrics(message.usage)

    return model_response


@dataclass
class Claude(Model):
    """`Model` subclass for Anthropic's Messages API."""

    id: str = "claude-sonnet-4-5-20250929"
    provider: str = "Anthropic"
    max_tokens: int = 4096

    def get_client(self) -> Anthropic:
        return Anthropic(api_key=self.api_key) if self.api_key else Anthropic()

    def invoke(
        self,
        messages: List[Message],
        assistant_message: Message,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """One synchronous round-trip: format `messages` into Anthropic's
        `(chat_messages, system)` shape, convert `tools` from the OpenAI-
        shaped canonical form into Anthropic's `input_schema` shape, call
        the SDK, time it onto `assistant_message.metrics.duration`, parse
        the response, and write it onto `assistant_message`.

        `tools` arrives OpenAI-shaped (`{"type": "function", "function":
        {...}}`) regardless of caller - `Model.response()` builds it once,
        the same way, for every provider (see `models/base.py`'s
        docstring) - so it's converted here, not by the caller.
        """
        chat_messages, system_message = format_messages(messages)
        anthropic_tools = format_tools_for_claude(tools) if tools else None

        request_kwargs: Dict[str, Any] = {"max_tokens": self.max_tokens}
        if system_message:
            request_kwargs["system"] = system_message
        if anthropic_tools:
            request_kwargs["tools"] = anthropic_tools

        timer = Timer()
        try:
            timer.start()
            response = self.get_client().messages.create(
                model=self.id,
                messages=chat_messages,  # type: ignore[arg-type]
                **request_kwargs,
            )
            timer.stop()
        except Exception as e:
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e

        assistant_message.metrics.duration = timer.elapsed

        model_response = parse_provider_response(response)
        self._populate_assistant_message(assistant_message, model_response)
        return model_response

    def invoke_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[ModelResponse]:
        """The TRANSPORT layer for streaming: open a
        `client.messages.stream(...)` context manager and `yield` one
        `ModelResponse` delta per accumulated SDK event, via
        `parse_provider_response_delta()`.

        Same request-building as `invoke()` above (format messages/tools,
        pass `max_tokens`/`system`/`tools`) - only the call itself differs,
        swapping `messages.create()` for the high-level `messages.stream()`
        context manager (see this module's docstring for why that specific
        SDK entry point matters: it's what makes tool_use blocks arrive
        already-complete at `content_block_stop`, sparing this provider the
        by-index fragment merge `OpenAIChat.parse_tool_calls()` needs).

        Does not call `_populate_assistant_message()` itself - each yielded
        `ModelResponse` is one fragment of the eventual response, not a
        finished one; `_populate_stream_data()` (`models/base.py`, Phase A)
        is what accumulates these, and `_populate_assistant_message_from_stream_data()`
        (Phase B) is what eventually writes the accumulated result onto
        `assistant_message`.
        """
        chat_messages, system_message = format_messages(messages)
        anthropic_tools = format_tools_for_claude(tools) if tools else None

        request_kwargs: Dict[str, Any] = {"max_tokens": self.max_tokens}
        if system_message:
            request_kwargs["system"] = system_message
        if anthropic_tools:
            request_kwargs["tools"] = anthropic_tools

        try:
            with self.get_client().messages.stream(
                model=self.id,
                messages=chat_messages,  # type: ignore[arg-type]
                **request_kwargs,
            ) as stream:
                for event in stream:
                    yield parse_provider_response_delta(event)
        except Exception as e:
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
