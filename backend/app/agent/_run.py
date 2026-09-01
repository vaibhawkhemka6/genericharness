"""Stage 3 Half B - the orchestration core `Agent.run()` dispatches to:
resolve `agent.tools` into the `Dict[str, Function]` shape `Model.response()`
wants, assemble `messages` via `get_run_messages()` (Half B's message-
building piece), make the one call into Half A (`agent.model.response()`),
and turn the result into a `RunOutput` via `update_run_response()`.

Trimmed against real Agno's `Agent._run()`/`Agent.arun()`: no hooks, no
reasoning, no output schema/parser model, no HITL pause/resume. Just: build
the messages, call the model, report what happened - including when the
model call itself fails, which is the one place this function has to do
more than wire Half A and Half B together.

Stage 5 addition: session read-then-write-back, via two one-line bookends
around the same core - `read_or_create_session()` (`agent/_storage.py`)
before `get_run_messages()` so history can flow in, and `_cleanup_and_store()`
(this module, below) on every terminal path - success, cancelled, and
error alike - so the run flows back out. Both are no-ops when `agent.db`
isn't set (`read_or_create_session()` returns `None`, `_cleanup_and_store()`
sees that `None` and does nothing), so an agent without a `db` runs exactly
as it did before this stage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel

from app.agent._messages import get_run_messages
from app.agent._response import handle_model_response_stream, update_run_response
from app.agent._session import save_session
from app.agent._storage import read_or_create_session
from app.agent.agent import Agent
from app.exceptions import ModelProviderError, RunCancelledException
from app.metrics import Timer
from app.models.message import Message
from app.run.agent import RunCancelledEvent, RunErrorEvent, RunInput, RunOutput, RunOutputEvent, RunStartedEvent
from app.run.base import RunStatus
from app.tools.function import Function
from app.tools.toolkit import Toolkit

if TYPE_CHECKING:
    from app.session.agent import AgentSession

logger = logging.getLogger(__name__)


def _cleanup_and_store(agent: Agent, session: Optional["AgentSession"], run_output: RunOutput) -> None:
    """Terminal-path bookend, called once from every exit point of `run()`
    and `run_stream()` (success, cancelled, error alike - matching the
    "persist everything, filter bad runs out of future context" decision
    `AgentSession.get_messages()` enforces on the read side). Appends
    `run_output` to `session` (dedup-by-`run_id`-or-append, see
    `AgentSession.upsert_run()`) and writes it back via `save_session()`.

    No-op when `session` is `None` - the exact signal `read_or_create_session()`
    returns when `agent.db` isn't configured, so a databaseless `Agent` takes
    this call and does nothing, unaffected by Stage 5 same as everywhere else
    in this module.
    """
    if session is None:
        return
    session.upsert_run(run_output)
    save_session(agent, session)


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

    session = read_or_create_session(agent, session_id, user_id)

    run_messages = get_run_messages(agent=agent, input=input, session=session, session_history=session_history)

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
        # the cut, rather than losing it silently. Stage 5: the run is still
        # persisted as-is (status=cancelled) via _cleanup_and_store() below -
        # AgentSession.get_messages()'s skip_statuses is what keeps it out of
        # *future* context, not a decision made here about whether to save it.
        timer.stop()
        logger.info(f"Run {run_id} cancelled: {e}")
        run_output.status = RunStatus.cancelled
        run_output.content = str(e)
        run_output.messages = list(run_messages.messages)
        _cleanup_and_store(agent, session, run_output)
        return run_output
    except ModelProviderError as e:
        timer.stop()
        logger.warning(f"Run {run_id} failed: {e}")
        run_output.status = RunStatus.error
        run_output.content = str(e)
        run_output.messages = list(run_messages.messages)
        _cleanup_and_store(agent, session, run_output)
        return run_output
    timer.stop()

    update_run_response(run_output=run_output, model_response=model_response, run_messages=run_messages)
    if run_output.metrics is not None:
        run_output.metrics.duration = timer.elapsed
    _cleanup_and_store(agent, session, run_output)
    return run_output


def run_stream(
    agent: Agent,
    input: Union[str, List, Dict, Message, BaseModel, List[Message]],
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_history: Optional[List[Message]] = None,
) -> Iterator[RunOutputEvent]:
    """Streaming twin of `run()` above - identical setup (resolve tools,
    assemble messages, build the `RunOutput` shell), but instead of making
    one blocking `agent.model.response(...)` call and returning once,
    `yield`s from `handle_model_response_stream()` (`agent/_response.py`,
    the CONSUMER layer) as the model streams. `run_output` is still built up
    in place the same way (mutated by `handle_model_response_stream()`'s
    trailing call to `update_run_response()`) - by the time the generator is
    exhausted, `run_output` reflects the same finished state `run()`'s
    return value would have, callers just also got to watch it happen.

    Same exception handling shape as `run()`, translated into one terminal
    event instead of one early `return`: `RunCancelledException` /
    `ModelProviderError` raised out of the streaming sub-loop still leave
    whatever rounds already ran sitting in `run_messages.messages` (same
    partial-transcript decision `run()` makes), reported here as
    `RunCancelledEvent`/`RunErrorEvent` instead of a `RunOutput.status`
    the caller has to notice after the fact.

    `RunStartedEvent` is yielded exactly once here, before anything else -
    it marks "the run as a whole has begun," which is not the same thing as
    `ModelRequestStartedEvent` (yielded once per iteration, inside
    `handle_model_response_stream()` -> `response_stream()`): a run with a
    tool-call round trip fires `ModelRequestStarted`/`ModelRequestCompleted`
    twice (once per model turn) but `RunStartedEvent` only once, at the very
    top.
    """
    run_id = str(uuid4())
    session_id = session_id or str(uuid4())

    session = read_or_create_session(agent, session_id, user_id)

    run_messages = get_run_messages(agent=agent, input=input, session=session, session_history=session_history)

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

    yield RunStartedEvent(
        agent_id=agent.id,
        agent_name=agent.name,
        run_id=run_id,
        session_id=session_id,
        model=agent.model.id,
        model_provider=agent.model.get_provider(),
    )

    functions = resolve_tools(agent) or None

    timer = Timer()
    timer.start()
    try:
        yield from handle_model_response_stream(
            agent=agent,
            run_output=run_output,
            run_messages=run_messages,
            functions=functions,
        )
    except RunCancelledException as e:
        timer.stop()
        logger.info(f"Run {run_id} cancelled: {e}")
        run_output.status = RunStatus.cancelled
        run_output.content = str(e)
        run_output.messages = list(run_messages.messages)
        _cleanup_and_store(agent, session, run_output)
        yield RunCancelledEvent(
            agent_id=agent.id,
            agent_name=agent.name,
            run_id=run_id,
            session_id=session_id,
            content=str(e),
        )
        return
    except ModelProviderError as e:
        timer.stop()
        logger.warning(f"Run {run_id} failed: {e}")
        run_output.status = RunStatus.error
        run_output.content = str(e)
        run_output.messages = list(run_messages.messages)
        _cleanup_and_store(agent, session, run_output)
        yield RunErrorEvent(
            agent_id=agent.id,
            agent_name=agent.name,
            run_id=run_id,
            session_id=session_id,
            content=str(e),
        )
        return
    timer.stop()

    if run_output.metrics is not None:
        run_output.metrics.duration = timer.elapsed
    _cleanup_and_store(agent, session, run_output)
