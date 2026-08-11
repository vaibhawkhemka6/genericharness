"""Anthropic Claude adapter."""

from __future__ import annotations

from typing import Any, Iterator, Optional

import anthropic

from app import config
from app.models.base import Model, ModelEvent, StreamEnd, TextDelta, ToolCall
from app.tools.base import Tool


DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"


class Claude(Model):
    def __init__(
        self,
        id: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        self.id = id
        self.max_tokens = max_tokens
        # Explicitly pin base_url rather than letting the SDK auto-read
        # ANTHROPIC_BASE_URL from the ambient environment: some sandboxes/dev
        # environments pre-set that var (sometimes already including a "/v1"
        # suffix), which makes the SDK double it up into ".../v1/v1/messages"
        # and every request 404s. Only use a custom base_url if the caller
        # (or our own .env, via config.py) explicitly asked for one.
        self.client = anthropic.Anthropic(
            api_key=api_key or config.ANTHROPIC_API_KEY,
            base_url=base_url or config.ANTHROPIC_BASE_URL or DEFAULT_ANTHROPIC_BASE_URL,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[Tool]] = None,
        system: Optional[str] = None,
    ) -> Iterator[ModelEvent]:
        kwargs: dict[str, Any] = {
            "model": self.id,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [t.to_json_schema() for t in tools]

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield TextDelta(text=text)
            final = stream.get_final_message()

        tool_calls = [
            ToolCall(id=block.id, name=block.name, arguments=block.input)
            for block in final.content
            if block.type == "tool_use"
        ]
        assistant_message = {
            "role": "assistant",
            "content": [block.model_dump() for block in final.content],
        }
        yield StreamEnd(
            stop_reason=final.stop_reason,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
        )

    def build_tool_result_messages(
        self, results: list[tuple[ToolCall, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc.id, "content": str(result)}
                    for tc, result in results
                ],
            }
        ]
