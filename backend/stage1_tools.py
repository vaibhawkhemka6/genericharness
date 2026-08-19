"""Watch a tool call travel steps 1-12.  python stage1_tools.py [--live]

Adapted from the user-supplied skeleton: package is `app` (not `myagent`),
and a few call sites diverge from what the skeleton assumed real Agno's
API looks like - each is flagged inline with a DIVERGENCE comment:

  - `models/openai/chat.py` and `models/anthropic/claude.py` export
    module-level `parse_provider_response()` functions, not `OpenAIChat`/
    `Claude` classes with a `._parse_provider_response()` method (this
    project's provider adapters are translation-layer functions, not full
    `Model` dataclasses yet - see those files' docstrings).
  - `FunctionCall.execute()` returns this project's own `ToolExecution`
    (`models/response.py`) directly, not a separate `FunctionExecutionResult`
    with `.status`/`.error` fields - so step 10 reads `.tool_call_error`/
    `.result` instead.
  - The hand-crafted OFFLINE `raw_openai`/`raw_anthropic` stand-ins need a
    few more attributes than the skeleton gave them (`role`, `usage`, a
    `model_dump()` on each tool call) because this project's
    `parse_provider_response()` reads all of those - a real SDK response
    object has them; a bare `SimpleNamespace` doesn't unless asked.
"""
import json, os, inspect
from types import SimpleNamespace
from typing import Optional, get_type_hints
from docstring_parser import parse

from app.tools.decorator import tool
from app.tools.function import Function, FunctionCall
from app.utils.functions import get_function_call
from app.utils.models.openai import format_tools_for_model as fmt_openai
from app.utils.models.claude import format_tools_for_model as fmt_claude
from app.models.message import Message

def show(n, title):
    print(f"\n{'='*70}\n STEP {n}  {title}\n{'='*70}")
def js(x):
    print(json.dumps(x, indent=2, default=str))


# ══ 1 ══════════════════════════════════════════════════════════════
show(1, "DECLARATION  @tool wraps the callable")

@tool
def get_weather(city: str, units: str = "celsius", verbose: Optional[bool] = None) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. Paris.
        units: Temperature units, celsius or fahrenheit.
        verbose: Include extended forecast detail.
    """
    return f"20 {units} and sunny in {city}"

print("type(get_weather)  :", type(get_weather).__name__)
print("is it callable?    :", callable(get_weather))
print("name               :", get_weather.name)
print("entrypoint         :", get_weather.entrypoint)
print("parameters so far  :", get_weather.parameters)


# ══ 2 ══════════════════════════════════════════════════════════════
show(2, "INTROSPECTION  what from_callable() read")

raw = get_weather.entrypoint
print("signature      :", inspect.signature(raw))
print("type hints     :", get_type_hints(raw))
print("defaults       :", {k: v.default for k, v in inspect.signature(raw).parameters.items()
                           if v.default is not inspect.Parameter.empty})
doc = parse(inspect.getdoc(raw))
print("doc summary    :", doc.short_description)
print("doc params     :", {p.arg_name: p.description for p in doc.params})
# DIVERGENCE: real Agno's from_callable() also strips agent/team/run_context
# and media (images/videos/audios/files) params by name or by injected type
# (see agno/tools/function.py's FRAMEWORK_INJECTED_PARAMS). This project's
# trimmed Function.from_callable() (tools/function.py) has no injection
# concept yet - there's no live Agent/Team to inject - so it strips only
# "self".
print("stripped (real Agno)    : agent, team, run_context, self, images, videos, audios, files")
print("stripped (this project) : self only -- no agent/team/run_context/media injection yet")


# ══ 3 ══════════════════════════════════════════════════════════════
show(3, "SCHEMA  python types -> JSON Schema")
js(get_weather.parameters)
print("\nrequired   :", get_weather.parameters["required"], " <- units/verbose have defaults")
print("verbose    :", get_weather.parameters["properties"]["verbose"],
      " <- Optional[bool] UNWRAPPED, not nullable")


# ══ 4 ══════════════════════════════════════════════════════════════
show(4, "REGISTRY  name -> Function")

def boom(x: str) -> str:
    """Always fails.

    Args:
        x: anything.
    """
    raise RuntimeError("upstream 503")

registry = {get_weather.name: get_weather}
bf = Function.from_callable(boom); registry[bf.name] = bf

for k, v in registry.items():
    print(f"  {k:<12} -> Function(entrypoint={v.entrypoint.__name__})")


# ══ 5 ══════════════════════════════════════════════════════════════
show(5, "SERIALIZATION  to_dict() -- entrypoint is DROPPED here")
openai_tools = fmt_openai(list(registry.values()))
js(openai_tools[1])
print("\nadvertised names :", sorted(t["function"]["name"] for t in openai_tools))
print("registry keys    :", sorted(registry))
print("MATCH            :", {t["function"]["name"] for t in openai_tools} == set(registry))
print("'entrypoint' in payload?", "entrypoint" in json.dumps(openai_tools))


# ══ 6 ══════════════════════════════════════════════════════════════
show(6, "TRANSPORT  same tool, two dialects")
claude_tools = fmt_claude(openai_tools)
print("--- OPENAI ---");    js(openai_tools[1])
print("--- ANTHROPIC ---"); js([t for t in claude_tools if t["name"] == "get_weather"][0])
print("\nopenai key   : 'parameters'   nested under 'function'")
print("anthropic key: 'input_schema'  flat, + additionalProperties:false")


# ══ 7 ══════════════════════════════════════════════════════════════
show(7, "SELECTION  model decides")
LIVE = os.getenv("OPENAI_API_KEY") and "--live" in os.sys.argv
if LIVE:
    from openai import OpenAI
    r = OpenAI().chat.completions.create(
        model="gpt-4o-mini", tools=openai_tools,
        messages=[{"role": "user", "content": "what's the weather in Paris?"}])
    print("LIVE finish_reason:", r.choices[0].finish_reason)
    raw_openai = r
else:
    print("OFFLINE - injecting a hand-crafted response")
    # DIVERGENCE: this project's parse_provider_response() (models/openai/chat.py)
    # reads response_message.role, response.usage, and calls .model_dump() on
    # each tool call - a real ChatCompletion has all three; a bare
    # SimpleNamespace needs them added by hand to stand in for one.
    raw_openai = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="tool_calls",
            message=SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(
                id="call_abc", type="function",
                function=SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}'),
                model_dump=lambda: {"id": "call_abc", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}})]))],
        usage=None,
    )

# DIVERGENCE: same reason as raw_openai above - parse_provider_response()
# (models/anthropic/claude.py) reads response.role and response.usage too.
raw_anthropic = SimpleNamespace(role="assistant", stop_reason="tool_use", usage=None, content=[
    SimpleNamespace(type="tool_use", id="toolu_xyz", name="get_weather",
                    input={"city": "Paris"})])

print("\nopenai    .message.tool_calls[0].function.arguments =",
      repr(raw_openai.choices[0].message.tool_calls[0].function.arguments))
print("anthropic .content[0].input                          =",
      repr(raw_anthropic.content[0].input), " <- DICT, not string")


# ══ 8 ══════════════════════════════════════════════════════════════
show(8, "PARSING  normalize both to one shape")
# DIVERGENCE: no OpenAIChat/Claude classes yet - these are the module-level
# translation functions from models/openai/chat.py and models/anthropic/claude.py.
from app.models.openai.chat import parse_provider_response as parse_openai_response
from app.models.anthropic.claude import parse_provider_response as parse_claude_response

o = parse_openai_response(raw_openai)
a = parse_claude_response(raw_anthropic)
print("--- from openai ---");    js(o.tool_calls)
print("--- from anthropic ---"); js(a.tool_calls)
print("\nboth arguments are str?",
      isinstance(o.tool_calls[0]["function"]["arguments"], str),
      isinstance(a.tool_calls[0]["function"]["arguments"], str))

tool_call = o.tool_calls[0]


# ══ 9 ══════════════════════════════════════════════════════════════
show(9, "RESOLUTION  string -> callable, via the registry")
fc = get_function_call(name=tool_call["function"]["name"],
                       arguments=tool_call["function"]["arguments"],
                       call_id=tool_call["id"], functions=registry)
print("looked up      :", tool_call["function"]["name"], "->", fc.function.name)
print("entrypoint BACK:", fc.function.entrypoint)
print("arguments      :", fc.arguments, type(fc.arguments).__name__)
print("call_id        :", fc.call_id)
print("error          :", fc.error)
print("call str       :", fc.get_call_str())


# ══ 10 ═════════════════════════════════════════════════════════════
show(10, "EXECUTION  finally run the python")
res = fc.execute()
# DIVERGENCE: execute() returns this project's ToolExecution (models/response.py),
# not a FunctionExecutionResult - so tool_call_error/result, not status/error.
print("tool_call_error :", res.tool_call_error)
print("result          :", res.result)


# ══ 11 ═════════════════════════════════════════════════════════════
show(11, "RESULT MESSAGE  the atomic pair")
assistant_msg = Message(role="assistant", content=None, tool_calls=[tool_call])
tool_msg = Message(role="tool", content=str(res.result), tool_call_id=fc.call_id,
                   tool_name=fc.function.name, tool_args=fc.arguments,
                   tool_call_error=res.tool_call_error)
print("[1] role=assistant  tool_calls[0].id =", tool_call["id"])
print("[2] role=tool       tool_call_id     =", tool_msg.tool_call_id)
print("    content         =", repr(tool_msg.content))
print("    PAIRED          :", tool_call["id"] == tool_msg.tool_call_id)


# ══ 12 ═════════════════════════════════════════════════════════════
show(12, "RE-SERIALIZATION  diverges again")
print("--- OPENAI ---")
js([{"role": "assistant", "content": None, "tool_calls": [tool_call]},
    {"role": "tool", "tool_call_id": tool_msg.tool_call_id, "content": tool_msg.content}])
print("--- ANTHROPIC ---")
js([{"role": "assistant", "content": [{"type": "tool_use", "id": tool_call["id"],
      "name": tool_call["function"]["name"],
      "input": json.loads(tool_call["function"]["arguments"])}]},
    {"role": "user", "content": [{"type": "tool_result",
      "tool_use_id": tool_msg.tool_call_id, "content": tool_msg.content}]}])
print("\nNOTE anthropic has NO tool role -> rewritten to 'user'")
print("NOTE input was json.loads'd -- reverses the dumps from step 8")


# ══ FAILURES ═══════════════════════════════════════════════════════
show("F", "FAILURE INJECTION  where step 9 and 10 catch things")

print("\n[hallucinated name]")
print("  ->", get_function_call("get_wather", '{"city":"P"}', "c1", registry), "(None)")

print("\n[single quotes]")
print("  ->", get_function_call("get_weather", "{'city': 'Paris'}", "c2", registry).arguments)

print("\n[entrypoint raises -- the 'boom' function registered in step 4]")
res_boom = get_function_call("boom", '{"x": "y"}', "c3", registry).execute()
print("  ->", res_boom.tool_call_error, repr(res_boom.result))
