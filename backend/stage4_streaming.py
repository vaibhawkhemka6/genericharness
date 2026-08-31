"""Stage 4 smoke test: streaming, top to bottom - TRANSPORT
(`Claude.invoke_stream()`/`OpenAIChat.invoke_stream()`), ACCUMULATOR
(`_populate_stream_data()`/`_populate_assistant_message_from_stream_data()`,
`app/models/base.py`), LOOP (`Model.response_stream()`), and CONSUMER
(`handle_model_response_stream()`, `app/agent/_response.py`) - exercised
live via `Agent.run(stream=True)`.

Two scenarios, matching the two examples worked through by hand before this
stage was built:

  Example A - plain text streaming, no tools ("What is the capital of
  France?"): expect ModelRequestStartedEvent, N x RunContentEvent (one per
  text chunk), ModelRequestCompletedEvent (with real token counts),
  RunCompletedEvent. No ToolCall* events at all.

  Example B - streaming with a `get_weather` tool call ("What's the weather
  in Paris?"): expect the same opening ModelRequestStarted/(zero-or-more
  content)/ModelRequestCompleted trio for the tool-call turn, then
  ToolCallStartedEvent -> ToolCallCompletedEvent sandwiching the tool's
  execution, then a second ModelRequestStarted/.../ModelRequestCompleted
  trio (the model's follow-up turn that reads the tool result) with content
  deltas this time, then RunCompletedEvent.

`ANTHROPIC_API_KEY` is set, `OPENAI_API_KEY` is blank - same asymmetry as
every earlier stage script; both live scenarios run against `Claude`
(`OpenAIChat.invoke_stream()` exists and is exercised for import/shape only,
not live, same reasoning as prior stages).

Run: python stage4_streaming.py
"""

from dotenv import load_dotenv

load_dotenv(override=True)

from app.agent.agent import Agent
from app.models.anthropic.claude import Claude
from app.run.agent import (
    ModelRequestCompletedEvent,
    ModelRequestStartedEvent,
    RunCompletedEvent,
    RunContentEvent,
    RunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from app.run.base import RunStatus
from app.tools.decorator import tool


def show(n, title):
    print(f"\n{'='*70}\n STEP {n}  {title}\n{'='*70}")


# ══ 1 ══════════════════════════════════════════════════════════════
show(1, "Example A - plain text streaming, no tools")

text_agent = Agent(model=Claude(), description="You are concise.")
events = list(text_agent.run("What is the capital of France? Answer in one short sentence.", stream=True))

for e in events:
    if isinstance(e, RunContentEvent):
        print(f"  [{e.event}] {e.content!r}")
    else:
        print(f"  [{e.event}] {e}")

event_types = [e.event for e in events]
print("\nevent sequence:", event_types)

assert event_types[0] == RunStartedEvent().event
assert event_types[1] == ModelRequestStartedEvent().event
assert event_types[-1] == RunCompletedEvent().event
assert ModelRequestCompletedEvent().event in event_types
assert ToolCallStartedEvent().event not in event_types, "Example A must have no tool calls"
assert ToolCallCompletedEvent().event not in event_types

content_events = [e for e in events if isinstance(e, RunContentEvent)]
assert len(content_events) > 0, "expected at least one RunContentEvent chunk"
streamed_text = "".join(e.content for e in content_events)
print("\nstreamed text (joined):", repr(streamed_text))
assert "paris" in streamed_text.lower()

completed = [e for e in events if isinstance(e, ModelRequestCompletedEvent)][0]
print("token counts:", completed.input_tokens, completed.output_tokens, completed.total_tokens)
assert completed.input_tokens and completed.input_tokens > 0
assert completed.output_tokens and completed.output_tokens > 0

run_completed = events[-1]
assert isinstance(run_completed, RunCompletedEvent)
print("final RunCompletedEvent.content:", repr(run_completed.content))
assert run_completed.content == streamed_text, "RunCompletedEvent.content must equal the accumulated stream"
print("OK")


# ══ 2 ══════════════════════════════════════════════════════════════
show(2, "Example B - streaming WITH a tool call")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. Paris.
    """
    return f"22 celsius and sunny in {city}"


tool_agent = Agent(model=Claude(), description="Use tools when relevant.", tools=[get_weather])
events_b = list(tool_agent.run("What's the weather in Paris? Use the tool.", stream=True))

for e in events_b:
    if isinstance(e, RunContentEvent):
        print(f"  [{e.event}] {e.content!r}")
    elif isinstance(e, (ToolCallStartedEvent, ToolCallCompletedEvent)):
        t = e.tool
        print(f"  [{e.event}] tool={t.tool_name if t else None} args={t.tool_args if t else None} result={t.result if t else None}")
    else:
        print(f"  [{e.event}] {e}")

event_types_b = [e.event for e in events_b]
print("\nevent sequence:", event_types_b)

assert event_types_b[0] == RunStartedEvent().event
assert event_types_b[1] == ModelRequestStartedEvent().event
assert event_types_b[-1] == RunCompletedEvent().event
assert ToolCallStartedEvent().event in event_types_b, "Example B must have a tool call"
assert ToolCallCompletedEvent().event in event_types_b

started_idx = event_types_b.index(ToolCallStartedEvent().event)
completed_idx = event_types_b.index(ToolCallCompletedEvent().event)
assert started_idx < completed_idx, "ToolCallStarted must precede ToolCallCompleted"

tool_started = [e for e in events_b if isinstance(e, ToolCallStartedEvent)][0]
tool_completed = [e for e in events_b if isinstance(e, ToolCallCompletedEvent)][0]
print("\ntool_started.tool  :", tool_started.tool)
print("tool_completed.tool:", tool_completed.tool)
assert tool_started.tool is not None and tool_started.tool.tool_name == "get_weather"
assert tool_started.tool.result is None, "tool_call_started must fire BEFORE the result exists"
assert tool_completed.tool is not None and tool_completed.tool.result is not None
assert "22 celsius" in tool_completed.tool.result

# Two full model turns: one that decided to call the tool, one that read
# the tool result back and answered - each bracketed by its own
# ModelRequestStarted/ModelRequestCompleted pair.
model_started_count = event_types_b.count(ModelRequestStartedEvent().event)
model_completed_count = event_types_b.count(ModelRequestCompletedEvent().event)
print("model_request_started count  :", model_started_count)
print("model_request_completed count:", model_completed_count)
assert model_started_count == 2
assert model_completed_count == 2

run_completed_b = events_b[-1]
print("\nfinal RunCompletedEvent.content:", repr(run_completed_b.content))
assert run_completed_b.content
print("OK")


# ══ 3 ══════════════════════════════════════════════════════════════
show(3, "Agent.run(stream=True) vs Agent.run() - same terminal content, two shapes")

blocking_output = text_agent.run("What is the capital of Germany? One word.")
streaming_events = list(text_agent.run("What is the capital of Germany? One word.", stream=True))
streaming_final = streaming_events[-1]

print("blocking .content :", blocking_output.content)
print("streaming .content:", streaming_final.content)
assert blocking_output.status == RunStatus.completed
assert isinstance(streaming_final, RunCompletedEvent)
assert "germany" not in blocking_output.content.lower()  # sanity: it answered "Berlin", not echoing the question
print("OK - both entry points converge on the same kind of finished answer")


print("\nALL STAGE 4 STREAMING STEPS OK")
