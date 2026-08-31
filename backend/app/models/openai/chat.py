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

Stage 4 addition - this is "the one concrete provider" the streaming
TRANSPORT layer lives in: `invoke_stream()` (the actual
`client.chat.completions.create(stream=True)` call),
`parse_provider_response_delta()` (raw `ChatCompletionChunk` ->
`ModelResponse` delta, the streaming sibling of `parse_provider_response()`
above), and `OpenAIChat.parse_tool_calls()` (overriding `Model`'s no-op
default). That override exists because of how OpenAI streams tool calls:
each chunk carries a *fragment* of one tool call, addressed by `index` (not
`id` - a fresh tool call's `id`/`name` only appear on its first fragment,
every fragment after that repeats only `index` and a piece of
`arguments`). `parse_provider_response_delta()` turns each fragment into a
`{"index": ..., "id": ..., "type": ..., "function": {"name": ...,
"arguments": ...}}` dict (arguments still just that one fragment's raw
text); `_populate_stream_data()` (`models/base.py`, Phase A) blindly
`.extend()`s these fragment-dicts onto `stream_data.response_tool_calls` -
after N chunks for one tool call, that list holds N separate fragment
dicts, all sharing the same `index`. `parse_tool_calls()` below is Phase B:
it walks that list once, groups by `index`, and string-concatenates each
group's `arguments` fragments into one complete JSON string per tool call -
only after that concatenation is the result the same shape a non-streaming
`ModelResponse.tool_calls` entry has (one dict, `arguments` a single valid
JSON string ready for `get_function_call()` to `json.loads()`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk

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


def parse_provider_response_delta(chunk: ChatCompletionChunk) -> ModelResponse:
    """Parse one OpenAI streaming `ChatCompletionChunk` into a `ModelResponse`
    delta - the streaming sibling of `parse_provider_response()` above.

    Two things a chunk can carry, handled independently (a chunk with
    `stream_options={"include_usage": True}` sends one trailing chunk with
    `usage` set and `choices=[]` - no `choice_delta` to read at all, so the
    content/tool_calls branch below is skipped for it, but the usage branch
    still needs to run):

      - `choice_delta.content` - a plain text fragment, copied straight
        across as `model_response.content`.
      - `choice_delta.tool_calls` - a list of `ChoiceDeltaToolCall` SDK
        objects, one per tool call *touched* by this chunk (usually just
        one). Converted here into this project's dict shape immediately
        (`{"index", "id", "type", "function": {"name", "arguments"}}`) -
        `Message.tool_calls` is typed as raw provider dicts everywhere else
        in this project, so the streaming path shouldn't be the one place
        that leaks SDK objects into it. `index` is what
        `OpenAIChat.parse_tool_calls()` (Phase B) groups fragments by -
        it's carried in the dict specifically so that merge step doesn't
        need the original SDK object back.
    """
    model_response = ModelResponse()

    if chunk.choices:
        choice_delta = chunk.choices[0].delta
        if choice_delta is not None:
            if choice_delta.content is not None:
                model_response.content = choice_delta.content

            if choice_delta.tool_calls:
                model_response.tool_calls = [
                    {
                        "index": tc.index,
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name if tc.function else None,
                            "arguments": tc.function.arguments if tc.function else None,
                        },
                    }
                    for tc in choice_delta.tool_calls
                ]

    if chunk.usage is not None:
        model_response.response_usage = get_metrics(chunk.usage)

    return model_response


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
        self._populate_assistant_message(assistant_message, model_response)
        return model_response

    def invoke_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[ModelResponse]:
        """The TRANSPORT layer for streaming: open a
        `chat.completions.create(stream=True)` request and `yield` one
        `ModelResponse` delta per raw `ChatCompletionChunk`, via
        `parse_provider_response_delta()`.

        `stream_options={"include_usage": True}` is what makes OpenAI send
        that one trailing usage-only chunk (`choices=[]`, `usage` set) -
        without it, streaming responses report no token usage at all.
        """
        request_kwargs: Dict[str, Any] = {}
        if tools:
            request_kwargs["tools"] = tools

        try:
            for chunk in self.get_client().chat.completions.create(
                model=self.id,
                messages=[format_message(m) for m in messages],
                stream=True,
                stream_options={"include_usage": True},
                **request_kwargs,
            ):
                yield parse_provider_response_delta(chunk)
        except Exception as e:
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e

    @staticmethod
    def parse_tool_calls(tool_calls_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase B: merge the by-`index` tool-call fragments
        `_populate_stream_data()` (Phase A, `models/base.py`) collected into
        one complete dict per tool call - `id`/`type`/`name` taken from
        whichever fragment first set them, `arguments` built by
        concatenating every fragment's `arguments` piece in arrival order.

        Overrides `Model.parse_tool_calls()`'s no-op default (see this
        module's docstring for why OpenAI specifically needs this and
        Claude doesn't).
        """
        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []

        for fragment in tool_calls_data:
            index = fragment.get("index", 0)
            if index not in tool_calls_by_index:
                tool_calls_by_index[index] = {"id": None, "type": None, "function": {"name": "", "arguments": ""}}
                order.append(index)

            entry = tool_calls_by_index[index]
            if fragment.get("id"):
                entry["id"] = fragment["id"]
            if fragment.get("type"):
                entry["type"] = fragment["type"]

            function_fragment = fragment.get("function") or {}
            if function_fragment.get("name"):
                entry["function"]["name"] = function_fragment["name"]
            if function_fragment.get("arguments"):
                entry["function"]["arguments"] += function_fragment["arguments"]

        return [tool_calls_by_index[index] for index in order]
