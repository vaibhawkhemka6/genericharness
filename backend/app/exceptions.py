"""Exceptions used by the agent loop and model layer, trimmed from Agno's
`agno/exceptions.py` to the classes this project currently needs.

`AgentRunException` is control-flow, not an error report: a tool can raise it
mid-loop to hand the agent loop a message to show the user/model and a flag
saying whether to keep iterating (`stop_execution`). `RetryAgentRun` and
`StopAgentRun` are the two named shapes of that same exception - they don't
add new fields or behavior, they just pin `stop_execution` to `False`/`True`
so tool code can write `raise StopAgentRun(...)` instead of remembering which
boolean means what. This is Stage 3's genuinely-new piece: Stage 0's
`AgentRunException` existed but nothing raised or caught it yet -
`tools/function.py:FunctionCall.execute()` now re-raises it (and
`RunCancelledException`) instead of swallowing it as an ordinary tool
failure, and `models/base.py:Model.run_function_call()` is what catches it.

`RunCancelledException` is a different kind of control flow: it doesn't
carry a message for the model at all, because there's no "next turn" for the
model to see it on - it means the whole `response()` loop should unwind, not
that this one tool call failed. `FunctionCall.execute()` re-raises it
untouched, and `run_function_call()` does too.

`ModelProviderError` wraps whatever a model provider's SDK raises (rate
limits, auth failures, malformed responses) into one shape the agent loop can
catch without knowing which provider it's talking to.
"""

from __future__ import annotations

from typing import List, Optional, Union

from app.models.message import Message


class AgentRunException(Exception):
    """Raised by a tool (or the agent loop itself) to interrupt a run in a
    controlled way - as opposed to an unexpected error bubbling up."""

    def __init__(
        self,
        exc,
        user_message: Optional[Union[str, Message]] = None,
        agent_message: Optional[Union[str, Message]] = None,
        messages: Optional[List[Union[dict, Message]]] = None,
        stop_execution: bool = False,
    ):
        super().__init__(exc)
        self.user_message = user_message
        self.agent_message = agent_message
        self.messages = messages
        self.stop_execution = stop_execution
        self.type = "agent_run_error"
        self.error_id = "agent_run_error"


class RetryAgentRun(AgentRunException):
    """Raised by a tool to say "this failed, but don't stop the run" - the
    loop should feed the failure back to the model and keep going, same as
    any other failed tool call, just via an explicit exception instead of
    returning an error string.

    Note: nothing at this stage actually re-invokes the same tool call
    automatically - "retry" here describes the *loop's* behavior (keep
    iterating) from the tool's point of view, not a literal re-execution
    mechanism. That naming matches real Agno, where this exception plays
    the same role.
    """

    def __init__(
        self,
        exc,
        user_message: Optional[Union[str, Message]] = None,
        agent_message: Optional[Union[str, Message]] = None,
        messages: Optional[List[Union[dict, Message]]] = None,
    ):
        super().__init__(
            exc, user_message=user_message, agent_message=agent_message, messages=messages, stop_execution=False
        )
        self.error_id = "retry_agent_run_error"


class StopAgentRun(AgentRunException):
    """Raised by a tool to say "the run should end after this call" - e.g. a
    tool that produces the agent's final answer directly and there's
    nothing left for the model to add. Same shape as `AgentRunException`,
    with `stop_execution` pinned to `True`."""

    def __init__(
        self,
        exc,
        user_message: Optional[Union[str, Message]] = None,
        agent_message: Optional[Union[str, Message]] = None,
        messages: Optional[List[Union[dict, Message]]] = None,
    ):
        super().__init__(
            exc, user_message=user_message, agent_message=agent_message, messages=messages, stop_execution=True
        )
        self.error_id = "stop_agent_run_error"


class RunCancelledException(Exception):
    """Raised when a run is cancelled out-of-band (e.g. a caller closing an
    SSE connection). Not tied to any particular tool call - unlike
    `AgentRunException`, there's no `.agent_message` to feed back to the
    model, because the loop isn't continuing to a next turn."""

    def __init__(self, message: str = "Operation cancelled by user"):
        super().__init__(message)
        self.type = "run_cancelled_error"
        self.error_id = "run_cancelled_error"


class ModelProviderError(Exception):
    """Raised when a model provider's API call fails (rate limit, auth,
    malformed request/response, etc.)."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        model_name: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.model_name = model_name
        self.model_id = model_id
        self.type = "model_provider_error"
        self.error_id = "model_provider_error"

    def __str__(self) -> str:
        return str(self.message)
