"""Stage 3 Half B - `update_run_response()`: the pure data-transform that
turns a finished `Model.response()` call into the `RunOutput` `Agent.run()`
hands back. No model/tool calls happen here - by the time this runs,
`agent.model.response()` has already mutated `run_messages.messages` in
place with every assistant/tool message from every round, and
`model_response` is the `ModelResponse` from the last of those rounds.
Testable without a live model call: feed it a hand-built `ModelResponse`
and message list and check `RunOutput` comes out right.
"""

from __future__ import annotations

from app.metrics import RunMetrics
from app.models.response import ModelResponse
from app.run.agent import RunOutput
from app.run.base import RunStatus
from app.run.messages import RunMessages


def update_run_response(
    run_output: RunOutput,
    model_response: ModelResponse,
    run_messages: RunMessages,
) -> RunOutput:
    """Fill `run_output` in from `model_response` + `run_messages.messages`,
    mark it `RunStatus.completed`, and return it (mutated in place, same as
    everything else in this loop - returned too, for a one-line call site).
    """
    content = model_response.content

    # StopAgentRun-empty-content fallback (decided up front, not left to be
    # discovered as a bug): when a tool ends the run early via
    # StopAgentRun, model_response is still the assistant's tool-call turn
    # (content=None, tool_calls=[...]) - no further model turn ever runs to
    # restate an answer, since response()'s loop breaks right after that
    # tool executes. The actual answer only exists on the tool-result
    # Message response() appended (its content is StopAgentRun's
    # .agent_message, via Model.run_function_call()). So the fallback scans
    # run_messages.messages backwards for the last message with non-empty
    # content, regardless of role - not specifically "the last assistant
    # message": in exactly this case the last assistant message is the
    # empty one, and the real content is one message later, on a tool
    # message.
    if not content:
        for message in reversed(run_messages.messages):
            if message.content:
                content = message.content
                break

    run_output.content = content
    if content is not None and not isinstance(content, str):
        run_output.content_type = type(content).__name__

    # add_to_agent_memory (models/message.py) filters what gets reported
    # back to the caller / would get persisted later - a message a tool or
    # future hook marks False (none do yet; the field defaults True) is
    # dropped here rather than shown on RunOutput or written to a session
    # transcript down the line.
    run_output.messages = [m for m in run_messages.messages if m.add_to_agent_memory]

    run_output.tools = model_response.tool_executions or None
    run_output.status = RunStatus.completed

    # Token totals summed across every assistant turn this run made (one
    # per round of response()'s loop) - RunMetrics.__add__ sums tokens but
    # deliberately does *not* sum duration from each side (see metrics.py);
    # duration is left for _run.py to set from its own wall-clock Timer
    # around the whole model.response() call, not reconstructed here from
    # individual message durations.
    run_metrics = RunMetrics()
    for message in run_messages.messages:
        if message.role == "assistant":
            run_metrics = run_metrics + message.metrics
    run_output.metrics = run_metrics

    return run_output
