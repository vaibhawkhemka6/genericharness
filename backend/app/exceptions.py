"""Exceptions used by the agent loop and model layer, trimmed from Agno's
`agno/exceptions.py` to the two classes this project currently needs.

`AgentRunException` is control-flow, not an error report: a tool can raise it
mid-loop to hand the agent loop a message to show the user/model and a flag
saying whether to keep iterating (`stop_execution`). It's the mechanism a
future "retry this tool call" or "abort the run" feature would build on.

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
