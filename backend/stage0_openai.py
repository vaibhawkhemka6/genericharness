"""Stage 0.5 end-to-end smoke test: build a RunOutput by hand, using nothing
but the ported types/functions, with a real OpenAI call in the middle.

Adapted from the user-supplied skeleton: package is `app` (not `myagent`),
and the three ported functions from `app/models/openai/chat.py` are named
`format_message`, `parse_provider_response`, `get_metrics` (the skeleton
called them `format_message`, `parse`, `usage_to_metrics`).
"""

import uuid

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

from app.metrics import MessageMetrics, RunMetrics
from app.models.message import Message
from app.models.response import ModelResponse
from app.run.base import RunContext, RunStatus
from app.run.messages import RunMessages
from app.run.agent import RunInput, RunOutput

# your three ported functions — from openai/chat.py:313, :803, :976
from app.models.openai.chat import format_message, parse_provider_response, get_metrics

MODEL_ID = "gpt-4o-mini"
client = OpenAI()

# 1-2. input + context
run_input = RunInput(input_content="What is 2+2? Answer in one word.")
run_context = RunContext(run_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))

# 3. EMPTY output up front — proves no field is secretly required
run_output = RunOutput(
    run_id=run_context.run_id,
    session_id=run_context.session_id,
    input=run_input,
    status=RunStatus.running,
)

# 4. run-level timer
run_metrics = RunMetrics()
run_metrics.start_timer()

# 5-8. build messages
run_messages = RunMessages(
    system_message=Message(role="system", content="You are concise."),
    user_message=Message(role="user", content=run_input.input_content),
)
run_messages.messages = run_messages.get_input_messages()

# 9. live ref — mutations below must show up in run_context
run_context.messages = run_messages.messages

# 10. WIRE FORMAT — 5 keys. NOT to_dict().
wire = [format_message(m) for m in run_messages.messages]
print("wire keys:", [sorted(w) for w in wire])

# 11. the call
raw = client.chat.completions.create(model=MODEL_ID, messages=wire)

# 12-13. provider -> normalized
model_response: ModelResponse = parse_provider_response(raw)
if raw.usage is not None:
    model_response.response_usage = get_metrics(raw.usage)

# 14-15. assistant message back onto the list
assistant = Message(
    role="assistant",
    content=model_response.content,
    metrics=model_response.response_usage or MessageMetrics(),
)
run_messages.messages.append(assistant)

# 16-20. fill the output
run_metrics.stop_timer()
if model_response.response_usage is not None:
    run_metrics += model_response.response_usage

run_output.content = model_response.content
run_output.content_type = "str"
run_output.model = MODEL_ID
run_output.model_provider = "openai"
run_output.messages = run_messages.messages
run_output.metrics = run_metrics
run_output.status = RunStatus.completed

# --- assertions ---
assert run_output.status == RunStatus.completed
assert run_output.content, "no content"
assert run_output.metrics.total_tokens > 0, "usage never mapped"
assert len(run_output.messages) == 3, f"expected 3, got {len(run_output.messages)}"
assert len(run_context.messages) == 3, "live ref broken"
assert RunOutput.from_dict(run_output.to_dict()) == run_output, "round-trip failed"

print("\ncontent:", run_output.content)
print("tokens :", run_output.metrics.total_tokens)
print("OK")
