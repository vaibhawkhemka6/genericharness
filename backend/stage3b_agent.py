"""Stage 3 Half B smoke test: message assembly (`app/agent/_messages.py`)
and run orchestration (`app/agent/_run.py`, `app/agent/_response.py`) built
on top of Half A's `Model.response()` loop (`stage3_loop.py`).

Follows the same build order as the implementation: user message, system
message, the session-history stub, get_run_messages() ordering, then
update_run_response() in isolation with fake data, and only then a live
Agent.run() through the whole stack - including the two gotchas called out
up front: the StopAgentRun-empty-content fallback, and what happens to a
failed run's partial transcript.

`ANTHROPIC_API_KEY` is set, `OPENAI_API_KEY` is blank - same asymmetry as
every earlier stage script; live steps run against `Claude`, OpenAIChat only
appears for the error-path step.

Run: python stage3b_agent.py
"""

from dotenv import load_dotenv

load_dotenv(override=True)

from app.agent._messages import get_run_messages, get_session_history_messages, get_system_message, get_user_message
from app.agent._response import update_run_response
from app.agent.agent import Agent
from app.models.anthropic.claude import Claude
from app.models.message import Message
from app.models.openai.chat import OpenAIChat
from app.models.response import ModelResponse
from app.run.agent import RunOutput
from app.run.base import RunStatus
from app.run.messages import RunMessages
from app.tools.decorator import tool


def show(n, title):
    print(f"\n{'='*70}\n STEP {n}  {title}\n{'='*70}")


# ══ 1 ══════════════════════════════════════════════════════════════
show(1, "get_user_message()  raw input -> Message")
print("str            :", get_user_message("hi there").content)
print("dict           :", get_user_message({"q": "weather?"}).content)
existing = Message(role="user", content="already a message")
print("Message passthrough is same object:", get_user_message(existing) is existing)
print("None           :", get_user_message(None))
assert get_user_message("hi").role == "user"
assert get_user_message(None) is None
print("OK")


# ══ 2 ══════════════════════════════════════════════════════════════
show(2, "get_system_message()  description + instructions")
agent_no_prompt = Agent(model=Claude())
print("neither set    :", get_system_message(agent_no_prompt))

agent_desc_only = Agent(model=Claude(), description="You are a helpful assistant.")
print("description only:", repr(get_system_message(agent_desc_only).content))

agent_full = Agent(
    model=Claude(),
    description="You are a helpful assistant.",
    instructions=["Be concise.", "Always answer in English."],
)
sm = get_system_message(agent_full)
print("both, list instructions:")
print(sm.content)
assert get_system_message(agent_no_prompt) is None
assert "Be concise." in sm.content and "You are a helpful assistant." in sm.content
print("OK")


# ══ 3 ══════════════════════════════════════════════════════════════
show(3, "get_session_history_messages()  stub, no DB")
history = [Message(role="user", content=f"turn {i}") for i in range(5)]
agent_hist = Agent(model=Claude(), session_history=history, num_history_messages=2)
trimmed = get_session_history_messages(agent_hist)
print("trimmed to last 2:", [m.content for m in trimmed])
print("explicit override wins over agent.session_history:",
      [m.content for m in get_session_history_messages(agent_hist, session_history=[Message(role="user", content="override")])])
print("no history at all:", get_session_history_messages(Agent(model=Claude())))
assert [m.content for m in trimmed] == ["turn 3", "turn 4"]
assert get_session_history_messages(Agent(model=Claude())) == []
print("OK")


# ══ 4 ══════════════════════════════════════════════════════════════
show(4, "get_run_messages()  system -> history -> user, in order")

rm_no_history = get_run_messages(agent_desc_only, "What's 2+2?")
print("no history  :", [m.role for m in rm_no_history.messages])
assert [m.role for m in rm_no_history.messages] == ["system", "user"]

rm_with_history = get_run_messages(agent_hist, "and now?")
print("with history:", [(m.role, m.content) for m in rm_with_history.messages])
# agent_hist has no description/instructions -> no system message; history
# (last 2, per num_history_messages=2) must land BEFORE the new user turn.
assert [m.content for m in rm_with_history.messages] == ["turn 3", "turn 4", "and now?"]

rm_list_input = get_run_messages(agent_desc_only, [Message(role="user", content="a"), Message(role="assistant", content="b")])
print("list-of-Message input:", [(m.role, m.content) for m in rm_list_input.messages])
assert rm_list_input.user_message is None
assert [m.content for m in rm_list_input.messages] == ["You are a helpful assistant.", "a", "b"]
print("OK - history sits between system and the current turn in every shape")


# ══ 5 ══════════════════════════════════════════════════════════════
show(5, "update_run_response()  pure data-transform, fake inputs")

# 5a. Ordinary case: model_response.content is already populated.
fake_messages = [
    Message(role="system", content="sys"),
    Message(role="user", content="hi"),
    Message(role="assistant", content="hello!"),
]
fake_model_response = ModelResponse(role="assistant", content="hello!")
run_output = RunOutput()
run_messages = RunMessages(messages=fake_messages)
update_run_response(run_output, fake_model_response, run_messages)
print("5a ordinary content :", run_output.content, "| status:", run_output.status)
assert run_output.content == "hello!"
assert run_output.status == RunStatus.completed
assert len(run_output.messages) == 3

# 5b. StopAgentRun-empty-content fallback: last model_response has no
# content (just a tool call), the real answer is on the tool message.
fake_messages_stop = [
    Message(role="user", content="finalize please"),
    Message(role="assistant", content=None, tool_calls=[{"id": "c1", "type": "function", "function": {"name": "finalize_answer", "arguments": "{}"}}]),
    Message(role="tool", content="FINAL: 42", tool_call_id="c1", tool_name="finalize_answer", stop_after_tool_call=True),
]
fake_model_response_empty = ModelResponse(role="assistant", content=None, tool_calls=fake_messages_stop[1].tool_calls)
run_output_stop = RunOutput()
update_run_response(run_output_stop, fake_model_response_empty, RunMessages(messages=fake_messages_stop))
print("5b StopAgentRun fallback content:", run_output_stop.content)
assert run_output_stop.content == "FINAL: 42", "empty-content fallback didn't pick up the tool message"

# 5c. add_to_agent_memory filtering.
fake_messages_filtered = [
    Message(role="user", content="hi"),
    Message(role="assistant", content="hello!", add_to_agent_memory=False),
]
run_output_filtered = RunOutput()
update_run_response(run_output_filtered, ModelResponse(content="hello!"), RunMessages(messages=fake_messages_filtered))
print("5c filtered messages count:", len(run_output_filtered.messages), "(dropped the add_to_agent_memory=False one)")
assert len(run_output_filtered.messages) == 1
print("OK")


# ══ 6 ══════════════════════════════════════════════════════════════
show(6, "Agent.run()  live, no tools")
simple_agent = Agent(model=Claude(), description="You are concise.")
run_output = simple_agent.run("What is 2+2? Answer in one word.")
print("content :", run_output.content)
print("status  :", run_output.status)
print("metrics :", run_output.metrics.to_dict() if run_output.metrics else None)
assert run_output.status == RunStatus.completed
assert run_output.content
assert run_output.metrics.total_tokens > 0
print("OK")


# ══ 7 ══════════════════════════════════════════════════════════════
show(7, "Agent.run()  live, WITH a tool - full round trip through resolve_tools()")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. Paris.
    """
    return f"22 celsius and sunny in {city}"


tool_agent = Agent(model=Claude(), description="Use tools when relevant.", tools=[get_weather])
run_output = tool_agent.run("What's the weather in Paris? Use the tool.")
print("content :", run_output.content)
print("tools   :", [(t.tool_name, t.result) for t in (run_output.tools or [])])
print("messages:", [m.role for m in run_output.messages])
assert run_output.status == RunStatus.completed
assert run_output.tools and run_output.tools[0].tool_name == "get_weather"
print("OK")


# ══ 8 ══════════════════════════════════════════════════════════════
show(8, "Agent.run()  live, StopAgentRun - proves the fallback works end-to-end")

from app.exceptions import StopAgentRun


@tool
def finalize_answer(answer: str) -> str:
    """Produce the final answer directly and end the run.

    Args:
        answer: The final answer text to give the user.
    """
    raise StopAgentRun("finalize_answer called", agent_message=f"FINAL: {answer}")


stop_agent = Agent(model=Claude(), description="When you have the answer, call finalize_answer with it - don't say anything else.", tools=[finalize_answer])
run_output = stop_agent.run("What is the capital of France? Call finalize_answer with just the city name.")
print("content :", run_output.content)
print("status  :", run_output.status)
assert run_output.status == RunStatus.completed
assert run_output.content and run_output.content.startswith("FINAL:")
print("OK - RunOutput.content came from the tool message, not a blank model_response.content")


# ══ 9 ══════════════════════════════════════════════════════════════
show(9, "Agent.run()  error path - OpenAIChat, blank key -> RunOutput, not a raised exception")

failing_agent = Agent(model=OpenAIChat())
run_output = failing_agent.run("hi")
print("status  :", run_output.status)
print("content :", str(run_output.content)[:100])
print("messages:", len(run_output.messages) if run_output.messages else 0, "(partial transcript kept, not discarded)")
assert run_output.status == RunStatus.error
assert run_output.messages is not None and len(run_output.messages) >= 1
print("OK - a model provider failure comes back as a RunOutput with status=error,")
print("     the caller never has to catch an exception just to know a run failed")


print("\nALL STAGE 3 HALF B STEPS OK")
