"""Offline test of the Agent loop mechanics using a fake Model (no network calls).

Validates: multi-turn tool-call loop, message history construction, and that
the loop terminates cleanly once the model stops requesting tools.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.agent import Agent
from app.models.base import Model, StreamEnd, TextDelta, ToolCall
from app.tools.base import tool


@tool
def add(a: int, b: int) -> str:
    """Add two numbers.

    Args:
        a: first number
        b: second number
    """
    return str(a + b)


class FakeModel(Model):
    """Turn 1: request a tool call. Turn 2: return final text."""

    id = "fake-model-v1"

    def __init__(self):
        self.calls = 0

    def stream(self, messages, tools=None, system=None):
        self.calls += 1
        if self.calls == 1:
            yield TextDelta(text="Let me add those numbers. ")
            yield StreamEnd(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3})],
                assistant_message={"role": "assistant", "content": "[requested tool: add]"},
            )
        else:
            yield TextDelta(text="The result is 5.")
            yield StreamEnd(
                stop_reason="end_turn",
                tool_calls=[],
                assistant_message={"role": "assistant", "content": "The result is 5."},
            )

    def build_tool_result_messages(self, results):
        return [{"role": "tool_result", "content": str(result)} for _, result in results]


def main():
    agent = Agent(model=FakeModel(), tools=[add], instructions="test")
    history = []
    events = list(agent.run(history, "what is 2 + 3?"))

    event_types = [e.type for e in events]
    print("event sequence:", event_types)
    assert event_types == [
        "run_started",
        "content_delta",
        "tool_call_started",
        "tool_call_result",
        "content_delta",
        "run_completed",
    ], f"unexpected event sequence: {event_types}"

    tool_started = next(e for e in events if e.type == "tool_call_started")
    assert tool_started.tool_name == "add"
    assert tool_started.tool_arguments == {"a": 2, "b": 3}

    tool_result = next(e for e in events if e.type == "tool_call_result")
    assert tool_result.tool_result == "5"

    print("final history length:", len(history))
    for m in history:
        print(" ", m)
    assert len(history) == 4  # user, assistant(tool req), tool_result, assistant(final)

    print("\nOK - agent loop mechanics verified offline")


if __name__ == "__main__":
    main()
