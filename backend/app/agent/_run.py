"""Stage 3 Half B - the orchestration core `Agent.run()` dispatches to:
resolve `agent.tools` into the `Dict[str, Function]` shape `Model.response()`
wants, assemble `messages` via `get_run_messages()` (Half B's message-
building piece), make the one call into Half A (`agent.model.response()`),
and turn the result into a `RunOutput` via `update_run_response()`.

Trimmed against real Agno's `Agent._run()`/`Agent.arun()`: no streaming, no
session read-then-write-back (session_history is a stub the caller hands
in - see `agent.py`'s docstring), no hooks, no reasoning, no output
schema/parser model, no HITL pause/resume. Just: build the messages, call
the model, report what happened - including when the model call itself
fails, which is the one place this function has to do more than wire
Half A and Half B together.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel

from app.agent._messages import get_run_messages
from app.agent._response import update_run_response
from app.agent.agent import Agent
from app.exceptions import ModelProviderError, RunCancelledException
from app.metrics import Timer
from app.models.message import Message
from app.run.agent import RunInput, RunOutput
from app.run.base import RunStatus
from app.tools.function import Function
from app.tools.toolkit import Toolkit

logger = logging.getLogger(__name__)


def resolve_tools(agent: Agent) -> Dict[str, Function]:
    """Turn `agent.tools` (a mix of plain callables, `@tool`-decorated
    `Function`s, and `Toolkit`s - whatever `Toolkit.register()` already
    accepts) into the flat `Dict[str, Function]` `Model.response()` needs.

    A `Toolkit`'s own `get_functions()` does the same normalization
    `Toolkit.register()` already did at construction time; a bare callable
    or `Function` here gets the identical treatment `Toolkit.register()`
    would give it, just without needing a `Toolkit` wrapper around it.
    """
    functions: Dict[str, Function] = {}
    for entry in agent.tools or []:
        if isinstance(entry, Toolkit):
            functions.update(entry.get_functions())
        elif isinstance(entry, Function):
            functions[entry.name] = entry
        elif callable(entry):
            function = Function.from_callable(entry)
            functions[function.name] = function
        else:
            logger.warning(f"Skipping tool {entry!r} - not a callable, Function, or Toolkit")
    return functions


def run(
    agent: Agent,
    input: Union[str, List, Dict, Message, BaseModel, List[Message]],
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_history: Optional[List[Message]] = None,
) -> RunOutput:
    """One end-to-end `Agent.run()`: resolve tools, assemble messages, call
    the model loop, report the result - or, if the model call itself fails,
    report *that* instead of letting the exception fly past `Agent.run()`'s
    caller with no `RunOutput` to show for it.
    """
    run_id = str(uuid4())
    session_id = session_id or str(uuid4())

    run_messages = get_run_messages(agent=agent, input=input, session_history=session_history)

    run_output = RunOutput(
        run_id=run_id,
        agent_id=agent.id,
        agent_name=agent.name,
        session_id=session_id,
        user_id=user_id,
        input=RunInput(input_content=input),
        model=agent.model.id,
        model_provider=agent.model.get_provider(),
        status=RunStatus.running,
    )

    functions = resolve_tools(agent) or None

    timer = Timer()
    timer.start()
    try:
        model_response = agent.model.response(
            messages=run_messages.messages,
            functions=functions,
            tool_call_limit=agent.tool_call_limit,
        )
    except RunCancelledException as e:
        # Partial-transcript decision: kept, not discarded. response()
        # mutates run_messages.messages in place as it goes, so whatever
        # rounds completed before cancellation are already sitting there -
        # a cancelled run still hands back everything that happened up to
        # the cut, rather than losing it silently. Whether a *session/db*
        # layer later chooses to write an errored/cancelled run's messages
        # to storage is that layer's call to make (Stage 5), not this
        # function's - RunOutput.status is exactly the signal it needs to
        # decide that.
        timer.stop()
        logger.info(f"Run {run_id} cancelled: {e}")
        run_output.status = RunStatus.cancelled
        run_output.content = str(e)
        run_output.messages = list(run_messages.messages)
        return run_output
    except ModelProviderError as e:
        timer.stop()
        logger.warning(f"Run {run_id} failed: {e}")
        run_output.status = RunStatus.error
        run_output.content = str(e)
        run_output.messages = list(run_messages.messages)
        return run_output
    timer.stop()

    update_run_response(run_output=run_output, model_response=model_response, run_messages=run_messages)
    if run_output.metrics is not None:
        run_output.metrics.duration = timer.elapsed
    return run_output
