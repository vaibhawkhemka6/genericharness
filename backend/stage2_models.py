"""Stage 2 smoke test: the `Model` ABC (`app/models/base.py`) and its two
concrete subclasses, `OpenAIChat` and `Claude`.

Where Stage 0/0.5 (`stage0_openai.py`, `stage0_anthropic.py`) called each
provider's translation functions (`format_message`, `parse_provider_response`,
`get_metrics`) by hand, and Stage 1 (`stage1_tools.py`) traced a tool call
through those same functions, Stage 2 proves the two providers are now
interchangeable behind one interface: construct either subclass, call
`.invoke(messages, assistant_message, tools)`, get a `ModelResponse` back
and the `assistant_message` populated - the caller doesn't need to know or
care which provider is underneath.

`ANTHROPIC_API_KEY` is set in this project's `.env`; `OPENAI_API_KEY` is
present but blank, so:
  - `Claude.invoke()` is exercised live, end-to-end.
  - `OpenAIChat.invoke()` is exercised against the *error path* instead -
    a blank/invalid key still has to fail as a `ModelProviderError`, not
    leak a raw SDK exception past the adapter boundary. That's the one
    behavior every provider adapter must share regardless of whether this
    machine happens to have a working key for it.

Run: python stage2_models.py
"""

from dotenv import load_dotenv

load_dotenv(override=True)

from app.exceptions import ModelProviderError
from app.models.anthropic.claude import Claude
from app.models.base import Model
from app.models.message import Message
from app.models.openai.chat import OpenAIChat


def show(n, title):
    print(f"\n{'='*70}\n STEP {n}  {title}\n{'='*70}")


# ══ 1 ══════════════════════════════════════════════════════════════
show(1, "ABC  Model can't be instantiated directly")
try:
    Model(id="x")
    print("UNEXPECTED: no error")
except TypeError as e:
    print("caught:", e)


# ══ 2 ══════════════════════════════════════════════════════════════
show(2, "CONSTRUCTION  two providers, one interface")
openai_model = OpenAIChat()
claude_model = Claude()
print(openai_model)
print(claude_model)
print("both are Model instances:", isinstance(openai_model, Model), isinstance(claude_model, Model))
print("get_provider()          :", openai_model.get_provider(), "/", claude_model.get_provider())


# ══ 3 ══════════════════════════════════════════════════════════════
show(3, "INVOKE (live)  Claude - same call shape either provider would use")
messages = [
    Message(role="system", content="You are concise."),
    Message(role="user", content="What is 2+2? Answer in one word."),
]
assistant_message = Message(role="assistant", content=None)

model_response = claude_model.invoke(messages, assistant_message)

print("ModelResponse.content :", model_response.content)
print("assistant.content     :", assistant_message.content)
print("assistant.role        :", assistant_message.role)
print("assistant.metrics     :", assistant_message.metrics.to_dict())

assert assistant_message.content == model_response.content
assert assistant_message.metrics.total_tokens > 0, "usage never mapped onto the message"
assert assistant_message.metrics.duration is not None, "invoke() must time itself"
print("\nOK - Claude.invoke() populated content, role, and metrics on the message")


# ══ 4 ══════════════════════════════════════════════════════════════
show(4, "INVOKE (error path)  OpenAIChat - same failure shape either provider would raise")
try:
    openai_model.invoke([Message(role="user", content="hi")], Message(role="assistant", content=None))
    print("UNEXPECTED: no error (a working OPENAI_API_KEY must be set)")
except ModelProviderError as e:
    print("caught ModelProviderError:", str(e)[:120])
    print("model_name/model_id       :", e.model_name, "/", e.model_id)
    print("\nOK - a provider SDK failure surfaces as our ModelProviderError, not a raw SDK exception")
