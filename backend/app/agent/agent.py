"""Core agent loop: build context -> call model -> run tools -> respond.

Mirrors Agno's Agent: give it a model, a list of tools, and instructions,
then call `.run()` with conversation history and a new user message. The
loop keeps calling the model and executing tools until the model returns a
turn with no further tool calls (or a safety iteration cap is hit).
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from app.agent.run import RunEvent
from app.models.base import Model, StreamEnd, TextDelta, ToolCall
from app.tools.base import Tool, ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_ITERATIONS = 8


class Agent:
    def __init__(
        self,
        model: Model,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[str] = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ):
        self.model = model
        self.registry = ToolRegistry(tools or [])
        self.instructions = instructions
        self.max_tool_iterations = max_tool_iterations

    def run(self, history: list[dict[str, Any]], user_message: str) -> Iterator[RunEvent]:
        """Run one turn. `history` is mutated in place with new messages so the
        caller can persist it (session store) for the next turn."""
        messages = history
        messages.append({"role": "user", "content": user_message})

        yield RunEvent(type="run_started")

        tools = self.registry.list()

        for _ in range(self.max_tool_iterations):
            stream_end: Optional[StreamEnd] = None
            try:
                for event in self.model.stream(messages, tools=tools, system=self.instructions):
                    if isinstance(event, TextDelta):
                        yield RunEvent(type="content_delta", content=event.text)
                    elif isinstance(event, StreamEnd):
                        stream_end = event
            except Exception as exc:  # noqa: BLE001 - surface any provider error to the client
                logger.exception("Model call failed")
                yield RunEvent(type="error", error=str(exc))
                return

            if stream_end is None:
                yield RunEvent(type="error", error="Model stream ended without a result.")
                return

            messages.append(stream_end.assistant_message)

            if not stream_end.tool_calls:
                yield RunEvent(type="run_completed")
                return

            results: list[tuple[ToolCall, Any]] = []
            for tc in stream_end.tool_calls:
                yield RunEvent(type="tool_call_started", tool_name=tc.name, tool_arguments=tc.arguments)
                try:
                    result = self.registry.execute(tc.name, tc.arguments)
                except Exception as exc:  # noqa: BLE001 - tool failures shouldn't crash the run
                    result = f"Error executing tool {tc.name!r}: {exc}"
                yield RunEvent(type="tool_call_result", tool_name=tc.name, tool_result=str(result))
                results.append((tc, result))

            messages.extend(self.model.build_tool_result_messages(results))

        yield RunEvent(
            type="error",
            error=f"Stopped after {self.max_tool_iterations} tool-call iterations without a final answer.",
        )
