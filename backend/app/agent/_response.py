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

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

from app.metrics import RunMetrics
from app.models.response import ModelResponse, ModelResponseEvent, ToolExecution
from app.run.agent import (
    ModelRequestCompletedEvent,
    ModelRequestStartedEvent,
    RunCompletedEvent,
    RunContentEvent,
    RunOutput,
    RunOutputEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from app.run.base import RunStatus
from app.run.messages import RunMessages

if TYPE_CHECKING:
    from app.agent.agent import Agent
    from app.tools.function import Function


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


def handle_model_response_stream(
    agent: "Agent",
    run_output: RunOutput,
    run_messages: RunMessages,
    functions: Optional[Dict[str, "Function"]] = None,
) -> Iterator[RunOutputEvent]:
    """CONSUMER layer for streaming - the streaming sibling of
    `update_run_response()` above. Watches `agent.model.response_stream()`'s
    `ModelResponse` deltas one at a time and translates each into exactly
    one agent-level `RunOutputEvent` (`run/agent.py`) - same envelope-plus-
    marker dispatch the model layer itself uses (`ModelResponse.event`),
    one layer up.

    `response_stream()` (`models/base.py`) already mutates
    `run_messages.messages` in place as it goes, the same way `response()`
    does for the non-streaming path - so once the loop below ends, this
    function is in exactly the position `_run.py:run()` is in after its own
    `agent.model.response(...)` call returns, and can finish the same way:
    build one synthetic `ModelResponse` (accumulated content + collected
    `ToolExecution`s) and hand it to `update_run_response()` to do the one
    real reconstruction pass (StopAgentRun-empty-content fallback, token
    totals, `RunStatus.completed`) - no separate copy of that logic lives
    here.
    """
    accumulated_content = ""
    tool_executions: List[ToolExecution] = []

    for model_response_delta in agent.model.response_stream(
        messages=run_messages.messages,
        functions=functions,
        tool_call_limit=agent.tool_call_limit,
    ):
        if model_response_delta.event == ModelResponseEvent.model_request_started.value:
            yield ModelRequestStartedEvent(
                agent_id=agent.id,
                agent_name=agent.name,
                run_id=run_output.run_id,
                session_id=run_output.session_id,
                model=agent.model.id,
                model_provider=agent.model.get_provider(),
            )
            continue

        if model_response_delta.event == ModelResponseEvent.model_request_completed.value:
            yield ModelRequestCompletedEvent(
                agent_id=agent.id,
                agent_name=agent.name,
                run_id=run_output.run_id,
                session_id=run_output.session_id,
                model=agent.model.id,
                model_provider=agent.model.get_provider(),
                input_tokens=model_response_delta.input_tokens,
                output_tokens=model_response_delta.output_tokens,
                total_tokens=model_response_delta.total_tokens,
                time_to_first_token=model_response_delta.time_to_first_token,
            )
            continue

        if model_response_delta.event == ModelResponseEvent.tool_call_started.value:
            tool_execution = (model_response_delta.tool_executions or [None])[0]
            yield ToolCallStartedEvent(
                agent_id=agent.id,
                agent_name=agent.name,
                run_id=run_output.run_id,
                session_id=run_output.session_id,
                tool=tool_execution,
            )
            continue

        if model_response_delta.event == ModelResponseEvent.tool_call_completed.value:
            tool_execution = (model_response_delta.tool_executions or [None])[0]
            if tool_execution is not None:
                tool_executions.append(tool_execution)
            yield ToolCallCompletedEvent(
                agent_id=agent.id,
                agent_name=agent.name,
                run_id=run_output.run_id,
                session_id=run_output.session_id,
                tool=tool_execution,
            )
            continue

        # event is None here: a plain content delta (see ModelResponse's
        # envelope-plus-marker docstring in models/response.py). The raw
        # tool-call JSON fragments Phase A (_populate_stream_data) collects
        # never surface as their own event up here - a fragment-only delta
        # has no .content, so it simply falls through this branch unyielded.
        if model_response_delta.content:
            accumulated_content += model_response_delta.content
            yield RunContentEvent(
                agent_id=agent.id,
                agent_name=agent.name,
                run_id=run_output.run_id,
                session_id=run_output.session_id,
                content=model_response_delta.content,
            )

    final_model_response = ModelResponse(
        content=accumulated_content or None,
        tool_executions=tool_executions or None,
    )
    update_run_response(run_output=run_output, model_response=final_model_response, run_messages=run_messages)

    yield RunCompletedEvent(
        agent_id=agent.id,
        agent_name=agent.name,
        run_id=run_output.run_id,
        session_id=run_output.session_id,
        content=run_output.content,
        content_type=run_output.content_type,
        metrics=run_output.metrics,
    )
