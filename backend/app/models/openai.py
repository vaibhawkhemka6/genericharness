"""OpenAI adapter (Chat Completions API).

Validates that the Model abstraction isn't accidentally Claude-shaped:
OpenAI's tool schema, streaming delta format, and tool-result message shape
are all meaningfully different from Anthropic's, and this adapter handles
that translation entirely on its own.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from openai import OpenAI

from app import config
from app.models.base import Model, ModelEvent, StreamEnd, TextDelta, ToolCall
from app.tools.base import Tool


def _tool_to_openai(t: Tool) -> dict[str, Any]:
    schema = t.to_json_schema()
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        },
    }


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIModel(Model):
    def __init__(
        self,
        id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        self.id = id
        self.max_tokens = max_tokens
        # See app/models/anthropic.py for why this is pinned explicitly
        # rather than letting the SDK auto-read OPENAI_BASE_URL from the
        # ambient environment.
        self.client = OpenAI(
            api_key=api_key or config.OPENAI_API_KEY,
            base_url=base_url or config.OPENAI_BASE_URL or DEFAULT_OPENAI_BASE_URL,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[Tool]] = None,
        system: Optional[str] = None,
    ) -> Iterator[ModelEvent]:
        openai_messages = list(messages)
        if system:
            openai_messages = [{"role": "system", "content": system}] + openai_messages

        kwargs: dict[str, Any] = {
            "model": self.id,
            "messages": openai_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [_tool_to_openai(t) for t in tools]

        response = self.client.chat.completions.create(**kwargs)

        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        stop_reason: Optional[str] = None

        for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                stop_reason = choice.finish_reason

            if delta.content:
                text_parts.append(delta.content)
                yield TextDelta(text=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    acc = tool_calls_acc.setdefault(
                        tc_delta.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        acc["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        acc["arguments"] += tc_delta.function.arguments

        tool_calls: list[ToolCall] = []
        raw_tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_calls_acc):
            acc = tool_calls_acc[idx]
            try:
                args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=acc["id"], name=acc["name"], arguments=args))
            raw_tool_calls.append(
                {
                    "id": acc["id"],
                    "type": "function",
                    "function": {"name": acc["name"], "arguments": acc["arguments"]},
                }
            )

        text = "".join(text_parts)
        assistant_message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if raw_tool_calls:
            assistant_message["tool_calls"] = raw_tool_calls

        yield StreamEnd(
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
        )

    def build_tool_result_messages(
        self, results: list[tuple[ToolCall, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            for tc, result in results
        ]
