"""Stage 3 smoke test: `Model.response()` - the `while True` agent loop
built on top of Stage 2's `invoke()`, plus the control-flow exceptions in
`app/exceptions.py` that a tool can use to steer it.

Where Stage 2 (`stage2_models.py`) proved one round-trip works the same way
regardless of provider, Stage 3 proves the *loop* around that round-trip
works: a turn with no tool calls ends immediately, a turn with tool calls
runs them and feeds results back for another turn, a hallucinated tool name
becomes a message instead of a crash, `StopAgentRun` ends the loop early
from inside a tool, and `tool_call_limit` cuts off a runaway loop.

`ANTHROPIC_API_KEY` is set in this project's `.env`; `OPENAI_API_KEY` is
blank - same asymmetry as Stage 2, so every live step below runs against
`Claude`, not `OpenAIChat`. `OpenAIChat` only appears in step 0, exercising
the same error-path guarantee Stage 2 already covered (a bad/blank key
surfaces as `ModelProviderError`, not a raw SDK exception) - `response()`
doesn't change that contract, since it's just `invoke()` called in a loop.

Run: python stage3_loop.py
"""

from dotenv import load_dotenv

load_dotenv(override=True)

from app.exceptions import ModelProviderError, RetryAgentRun, StopAgentRun
from app.models.anthropic.claude import Claude
from app.models.message import Message
from app.models.openai.chat import OpenAIChat
from app.tools.decorator import tool


def show(n, title):
    print(f"\n{'='*70}\n STEP {n}  {title}\n{'='*70}")


claude_model = Claude()


# ══ 0 ══════════════════════════════════════════════════════════════
show(0, "response() SHARES invoke()'s ERROR PATH  OpenAIChat, blank key")
try:
    OpenAIChat().response([Message(role="user", content="hi")])
    print("UNEXPECTED: no error (a working OPENAI_API_KEY must be set)")
except ModelProviderError as e:
    print("caught ModelProviderError:", str(e)[:120])
    print("OK - response()'s first invoke() call still raises through unchanged")


# ══ 1 ══════════════════════════════════════════════════════════════
show(1, "NO TOOL CALLS  loop ends after one turn")
messages = [
    Message(role="system", content="You are concise."),
    Message(role="user", content="What is 2+2? Answer in one word."),
]
model_response = claude_model.response(messages)
print("content        :", model_response.content)
print("tool_executions:", model_response.tool_executions)
print("messages len   :", len(messages), "(system, user, assistant)")
assert model_response.tool_executions == []
assert len(messages) == 3
print("OK - no tool_calls on the first turn means response() breaks immediately")


# ══ 2 ══════════════════════════════════════════════════════════════
show(2, "TOOL CALL RESOLVED THROUGH THE REGISTRY  live round-trip")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. Paris.
    """
    return f"22 celsius and sunny in {city}"


registry = {get_weather.name: get_weather}
messages = [
    Message(role="system", content="Use the get_weather tool for weather questions."),
    Message(role="user", content="What's the weather in Paris? Use the tool."),
]
model_response = claude_model.response(messages, functions=registry)

print("final content    :", model_response.content)
print("tool_executions   :", [(te.tool_name, te.result, te.tool_call_error) for te in model_response.tool_executions])
print("messages len      :", len(messages), "(system, user, assistant#1, tool, assistant#2)")
assert any(te.tool_name == "get_weather" and not te.tool_call_error for te in model_response.tool_executions)
assert len(messages) == 5
print("OK - tool call executed, result fed back, model answered on a second turn")


# ══ 3 ══════════════════════════════════════════════════════════════
show(3, "HALLUCINATED TOOL NAME  _prepare_function_calls() handles it directly")
messages = [Message(role="assistant", content=None, tool_calls=[
    {"id": "call_x", "type": "function", "function": {"name": "does_not_exist", "arguments": "{}"}}
])]
function_calls = claude_model._prepare_function_calls(
    assistant_message=messages[0], messages=messages, functions=registry
)
print("function_calls returned :", function_calls, "(empty - nothing resolved)")
print("messages appended       :", [m.role for m in messages][1:])
print("error message content   :", messages[-1].content)
assert function_calls == []
assert messages[-1].tool_call_error is True
print("OK - an unresolved tool name becomes an error tool Message, not a crash")


# ══ 4 ══════════════════════════════════════════════════════════════
show(4, "StopAgentRun FROM A TOOL  loop ends early, even with more to say")


@tool
def finalize_answer(answer: str) -> str:
    """Produce the final answer directly and end the run.

    Args:
        answer: The final answer text to give the user.
    """
    raise StopAgentRun(f"finalize_answer called", agent_message=f"FINAL: {answer}")


registry2 = {finalize_answer.name: finalize_answer}
messages = [Message(role="assistant", content=None, tool_calls=[
    {"id": "call_y", "type": "function", "function": {"name": "finalize_answer", "arguments": '{"answer": "42"}'}}
])]
function_calls = claude_model._prepare_function_calls(
    assistant_message=messages[0], messages=messages, functions=registry2
)
tool_executions = claude_model.run_function_calls(function_calls=function_calls)
print("tool_execution.result           :", tool_executions[0].result)
print("tool_execution.stop_after_tool_call:", tool_executions[0].stop_after_tool_call)
assert tool_executions[0].result == "FINAL: 42"
assert tool_executions[0].stop_after_tool_call is True
print("OK - StopAgentRun's .agent_message becomes the ToolExecution result,")
print("     and .stop_execution flows through as stop_after_tool_call - this is")
print("     the flag response()'s loop checks to break instead of looping back")


# ══ 5 ══════════════════════════════════════════════════════════════
show(5, "RetryAgentRun FROM A TOOL  failed ToolExecution, but doesn't stop")


@tool
def flaky_tool(x: str) -> str:
    """A tool that always asks to be retried.

    Args:
        x: anything.
    """
    raise RetryAgentRun("upstream hiccup", agent_message="Upstream failed, please try again.")


registry3 = {flaky_tool.name: flaky_tool}
messages = [Message(role="assistant", content=None, tool_calls=[
    {"id": "call_z", "type": "function", "function": {"name": "flaky_tool", "arguments": '{"x": "y"}'}}
])]
function_calls = claude_model._prepare_function_calls(
    assistant_message=messages[0], messages=messages, functions=registry3
)
tool_executions = claude_model.run_function_calls(function_calls=function_calls)
print("tool_call_error                    :", tool_executions[0].tool_call_error)
print("result                              :", tool_executions[0].result)
print("stop_after_tool_call                :", tool_executions[0].stop_after_tool_call)
assert tool_executions[0].tool_call_error is True
assert tool_executions[0].stop_after_tool_call is False
print("OK - RetryAgentRun reports a failed ToolExecution but leaves stop_after_tool_call")
print("     False, so response()'s loop would keep going, feeding the failure back")


# ══ 6 ══════════════════════════════════════════════════════════════
show(6, "tool_call_limit  run_function_calls() cuts off a runaway loop")


@tool
def noop(n: int) -> str:
    """Do nothing.

    Args:
        n: anything.
    """
    return f"ok {n}"


noop_registry = {noop.name: noop}
tool_calls_raw = [
    {"id": f"call_{i}", "type": "function", "function": {"name": "noop", "arguments": f'{{"n": {i}}}'}}
    for i in range(5)
]
messages = [Message(role="assistant", content=None, tool_calls=tool_calls_raw)]
function_calls = claude_model._prepare_function_calls(
    assistant_message=messages[0], messages=messages, functions=noop_registry
)
tool_executions = claude_model.run_function_calls(function_calls=function_calls, tool_call_limit=2)
for te in tool_executions:
    print(f"  {te.tool_name} error={te.tool_call_error} result={te.result!r}")
ran = [te for te in tool_executions if not te.tool_call_error]
limited = [te for te in tool_executions if te.tool_call_error]
assert len(ran) == 2
assert len(limited) == 3
assert "limit" in limited[0].result.lower()
print("OK - 5 calls in, limit=2 -> first 2 actually ran, the rest came back as")
print("     synthetic failed ToolExecutions instead of being executed")


print("\nALL STAGE 3 STEPS OK")
