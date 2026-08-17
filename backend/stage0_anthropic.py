"""Stage 0.5 end-to-end smoke test for the Anthropic adapter: build a
RunOutput by hand using nothing but the ported types/functions, with a real
Anthropic call in the middle.

Adapted from the user-supplied skeleton: package is `app` (not `myagent`),
and the ported functions are named `parse_provider_response`/`get_metrics`
in `app/models/anthropic/claude.py` (the skeleton called them
`parse`/`usage_to_metrics`). Deliberately bypasses this project's own
`invoke()` helper in that same module - like the OpenAI skeleton, this test
calls `format_messages()` + the raw client + `parse_provider_response()` +
`get_metrics()` directly, to exercise the ported pieces individually rather
than the convenience wrapper around them.
"""

import uuid

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

from app.metrics import MessageMetrics, RunMetrics
from app.models.message import Message
from app.run.base import RunContext, RunStatus
from app.run.messages import RunMessages
from app.run.agent import RunInput, RunOutput
from app.models.anthropic.claude import parse_provider_response, get_metrics
from app.utils.models.claude import format_messages

MODEL_ID = "claude-sonnet-4-5-20250929"  # claude-sonnet-4-20250514 from the skeleton 404s on this account
client = Anthropic()

run_input = RunInput(input_content="What is 2+2? Answer in one word.")
run_context = RunContext(run_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))
run_output = RunOutput(
    run_id=run_context.run_id,
    session_id=run_context.session_id,
    input=run_input,
    status=RunStatus.running,
)

run_metrics = RunMetrics()
run_metrics.start_timer()

run_messages = RunMessages(
    system_message=Message(role="system", content="You are concise."),
    user_message=Message(role="user", content=run_input.input_content),
)
run_messages.messages = run_messages.get_input_messages()
run_context.messages = run_messages.messages

#print(run_messages.messages) ## List of messages
# ---- DIVERGENCE: whole list in, tuple out ----
chat_messages, system_message = format_messages(run_messages.messages)
assert all(m["role"] != "system" for m in chat_messages), "system leaked into messages"
assert system_message, "system was dropped"

# ---- DIVERGENCE: system is a kwarg; max_tokens is REQUIRED ----
raw = client.messages.create(
    model=MODEL_ID,
    system=system_message,
    messages=chat_messages,
    max_tokens=1024,
)

model_response = parse_provider_response(raw)
model_response.response_usage = get_metrics(raw.usage)

## Message is appended from model response 
run_messages.messages.append(Message(
    role="assistant",
    content=model_response.content,
    metrics=model_response.response_usage,
))

run_metrics.stop_timer()
run_metrics += model_response.response_usage
run_output.content = model_response.content
run_output.model = MODEL_ID
run_output.model_provider = "anthropic"
run_output.messages = run_messages.messages
run_output.metrics = run_metrics
run_output.status = RunStatus.completed

assert run_output.metrics.total_tokens > 0, "total_tokens computed? Anthropic omits it"
assert len(run_output.messages) == 3
assert RunOutput.from_dict(run_output.to_dict()) == run_output
print("OK", run_output.content, run_output.metrics.total_tokens)
