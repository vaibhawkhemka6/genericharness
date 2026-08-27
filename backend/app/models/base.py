"""`Model` - the abstract base every provider adapter (`OpenAIChat`, `Claude`)
implements, mirroring Agno's `agno/models/base.py` (~3100 lines) trimmed to
one synchronous request/response round-trip plus the tool-call loop built on
top of it.

Still dropped vs. real Agno: streaming (sync or async), `ainvoke`, response
caching to disk, retry/backoff, structured-output (`response_format`)
plumbing, and every HITL flow (confirmation/user-input/external-execution) -
our `Function` (`tools/function.py`) has none of those fields to check, so
there's nothing for a loop to gate on even if it wanted to.

`OpenAIChat` (`models/openai/chat.py`) and `Claude` (`models/anthropic/claude.py`)
are the two concrete subclasses. Each still exports its translation
functions (`format_message`/`format_messages`, `parse_provider_response`,
`get_metrics`) as plain module-level functions rather than methods - the
Stage 0/0.5 test scripts (`stage0_openai.py`, `stage0_anthropic.py`) call
those directly, by hand, to walk through request/response translation step
by step, so they can't be folded into the class without breaking those
scripts. `invoke()` (Stage 2) wires client construction + those functions +
`_populate_assistant_message()` together into one round-trip.

Stage 3 addition: `response()`, the `while True` loop built on top of
`invoke()` - call the model, and if it asked for tools, run them and loop
back with the results appended, until a turn comes back with no tool calls
(or something says to stop early). Three helpers do the work:

    _prepare_function_calls()  ModelResponse.tool_calls (raw dicts) -> resolved FunctionCalls,
                                looking each one up in the `functions` registry
    run_function_calls()       iterates this turn's FunctionCalls, applying the one guard
                                this trimmed loop has left (tool_call_limit)
    run_function_call()        one call's try/except - routes the control-flow exceptions
                                a tool can deliberately raise (see app/exceptions.py)

`response()` builds tool declarations once, in the OpenAI-shaped canonical
form (`utils/models/openai.py:format_tools_for_model`), and hands that same
shape to every provider's `invoke()` - it's each subclass's job (not
`response()`'s) to convert further if its wire format differs, the way
`Claude.invoke()` runs it through `utils/models/claude.py:format_tools_for_model()`
before calling the Anthropic SDK. That's what keeps `response()` itself
provider-agnostic: it never needs to know which shape a given provider
actually wants on the wire.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.exceptions import AgentRunException, RunCancelledException
from app.models.message import Message
from app.models.response import ModelResponse, ToolExecution
from app.tools.function import Function, FunctionCall
from app.utils.functions import get_function_call
from app.utils.models.openai import format_tools_for_model as format_tools_canonical

logger = logging.getLogger(__name__)


@dataclass
class Model(ABC):
    """Provider-agnostic interface: one model call in, one `ModelResponse` out."""

    # ID of the model to use, e.g. "gpt-4o-mini" or "claude-sonnet-4-5-20250929".
    id: str
    # Human-readable name for this model. Not sent to the provider API.
    name: Optional[str] = None
    # Provider label, e.g. "OpenAI" or "Anthropic". Not sent to the provider API.
    provider: Optional[str] = None
    # API key override. If None, each adapter's get_client() falls back to
    # the provider SDK's normal env-var lookup (OPENAI_API_KEY / ANTHROPIC_API_KEY).
    api_key: Optional[str] = None
    # Role used for tool-result messages appended by response()'s loop.
    # Every provider's format_message(s) already knows how to translate a
    # role="tool" Message into its own wire shape (a flat "tool" entry for
    # OpenAI, a role="user" tool_result block for Anthropic) - see
    # DEFAULT_ROLE_MAP / ROLE_MAP in each models/openai/anthropic module.
    tool_message_role: str = "tool"

    def get_provider(self) -> str:
        return self.provider or self.__class__.__name__

    @abstractmethod
    def invoke(
        self,
        messages: List[Message],
        assistant_message: Message,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Send `messages` to the provider and return the parsed `ModelResponse`.

        Implementations are expected to: format `messages` into the
        provider's wire shape, call the SDK, time the call onto
        `assistant_message.metrics.duration`, parse the raw response via
        that module's `parse_provider_response()` into a `ModelResponse`,
        call `self._populate_assistant_message(assistant_message,
        model_response)` to write it onto `assistant_message` (every
        provider adapter does this the same way), and return the
        `ModelResponse` itself - not `_populate_assistant_message()`'s
        return value, which is `assistant_message`. `response()` below
        relies on getting the `ModelResponse` back (it reads
        `.response_usage`, which only exists there, not on `Message`).
        """
        raise NotImplementedError

    def _populate_assistant_message(self, assistant_message: Message, model_response: ModelResponse) -> Message:
        """Copy a parsed `ModelResponse` onto the `Message` the caller will
        append to history. Shared by every `invoke()` implementation so the
        response -> message mapping only lives in one place, not once per
        provider.

        Does *not* touch `assistant_message.metrics.duration` - each
        `invoke()` sets that itself from its own request timer, before the
        response even exists to parse from.
        """
        if model_response.role is not None:
            assistant_message.role = model_response.role
        if model_response.content is not None:
            assistant_message.content = model_response.content
        if model_response.tool_calls:
            assistant_message.tool_calls = model_response.tool_calls
        if model_response.response_usage is not None:
            assistant_message.metrics.input_tokens = model_response.response_usage.input_tokens
            assistant_message.metrics.output_tokens = model_response.response_usage.output_tokens
            assistant_message.metrics.total_tokens = model_response.response_usage.total_tokens
        return assistant_message

    def response(
        self,
        messages: List[Message],
        functions: Optional[Dict[str, Function]] = None,
        tool_call_limit: Optional[int] = None,
    ) -> ModelResponse:
        """The agent loop's model-facing entry point: call the model, and if
        it asked for tools, run them and feed the results back, repeating
        until a turn comes back with no tool calls (or a tool/limit says to
        stop early).

        Mutates `messages` in place - the assistant message from each model
        call and the tool-result messages from each round of execution are
        both appended directly, the same way every earlier stage script
        built up a message list by hand. Returns one `ModelResponse`: role/
        content/tool_calls/response_usage reflect the *last* model call
        (the one that ended the loop), while `tool_executions` accumulates
        across every round - summing per-round usage into a running total
        belongs to a run-level `RunMetrics` a future agent-loop stage wires
        up, not this method.
        """
        model_response = ModelResponse()
        tool_declarations = format_tools_canonical(list(functions.values())) if functions else None

        tool_call_count = 0
        while True:
            assistant_message = Message(role="assistant")
            this_response = self.invoke(messages, assistant_message, tools=tool_declarations)
            messages.append(assistant_message)

            model_response.role = this_response.role
            model_response.content = this_response.content
            model_response.tool_calls = this_response.tool_calls
            if this_response.response_usage is not None:
                model_response.response_usage = this_response.response_usage

            if not assistant_message.tool_calls:
                break

            function_calls_to_run = self._prepare_function_calls(
                assistant_message=assistant_message,
                messages=messages,
                functions=functions,
            )

            tool_executions = self.run_function_calls(
                function_calls=function_calls_to_run,
                tool_call_count=tool_call_count,
                tool_call_limit=tool_call_limit,
            )
            tool_call_count += len(tool_executions)

            model_response.tool_executions = (model_response.tool_executions or []) + tool_executions

            stop = False
            for tool_execution in tool_executions:
                messages.append(
                    Message(
                        role=self.tool_message_role,
                        content=str(tool_execution.result) if tool_execution.result is not None else "",
                        tool_call_id=tool_execution.tool_call_id,
                        tool_name=tool_execution.tool_name,
                        tool_args=tool_execution.tool_args,
                        tool_call_error=tool_execution.tool_call_error,
                        stop_after_tool_call=tool_execution.stop_after_tool_call,
                    )
                )
                if tool_execution.stop_after_tool_call:
                    stop = True

            if stop:
                break
            # Otherwise: loop back and call the model again with the tool
            # results now in `messages`.

        return model_response

    def _prepare_function_calls(
        self,
        assistant_message: Message,
        messages: List[Message],
        functions: Optional[Dict[str, Function]] = None,
    ) -> List[FunctionCall]:
        """Turn `assistant_message.tool_calls` (raw provider dicts) into
        resolved `FunctionCall`s, looking each one up in `functions` via
        `get_function_call()` (Stage 1, `utils/functions.py`).

        A tool call that doesn't resolve to a known `Function` at all
        (hallucinated name, or no `functions` registry given) doesn't become
        a `FunctionCall` - `get_function_call()` returns `None` for that
        case, so there's nothing to `execute()`. Instead an error tool
        `Message` is appended straight to `messages` here, so the model
        sees the failure on its next turn without `response()`'s caller
        having to special-case it. A call that resolves but has
        unparseable arguments *does* still become a `FunctionCall` (with
        `.error` set) - `FunctionCall.execute()` already turns that into a
        clean failed `ToolExecution` on its own (Stage 1), so there's no
        need to duplicate that handling here.
        """
        function_calls_to_run: List[FunctionCall] = []
        for tool_call in assistant_message.tool_calls or []:
            tool_call_id = tool_call.get("id")
            function_def = tool_call.get("function", {})
            tool_name = function_def.get("name")

            function_call = get_function_call(
                name=tool_name,
                arguments=function_def.get("arguments"),
                call_id=tool_call_id,
                functions=functions,
            )
            if function_call is None:
                messages.append(
                    Message(
                        role=self.tool_message_role,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content="Error: The requested tool does not exist or is not available.",
                        tool_call_error=True,
                    )
                )
                continue
            function_calls_to_run.append(function_call)
        return function_calls_to_run

    def run_function_calls(
        self,
        function_calls: List[FunctionCall],
        tool_call_count: int = 0,
        tool_call_limit: Optional[int] = None,
    ) -> List[ToolExecution]:
        """Run every `FunctionCall` this turn produced, in order, applying
        the one guard this trimmed loop has left: a running `tool_call_limit`
        across the whole `response()` call.

        Real Agno also gates each call here on HITL flags
        (`requires_confirmation`/`requires_user_input`/`external_execution`)
        before running it - dropped along with the rest of that flow, since
        our `Function` (`tools/function.py`) has none of those fields to
        check.
        """
        tool_executions: List[ToolExecution] = []
        for function_call in function_calls:
            if tool_call_limit is not None:
                tool_call_count += 1
                if tool_call_count > tool_call_limit:
                    logger.debug(
                        f"Tool call limit ({tool_call_limit}) reached. "
                        f"Skipping: {function_call.function.name} (call #{tool_call_count})"
                    )
                    tool_executions.append(
                        ToolExecution(
                            tool_call_id=function_call.call_id,
                            tool_name=function_call.function.name,
                            tool_args=function_call.arguments,
                            tool_call_error=True,
                            result=(
                                f"Tool call limit reached. Tool call {function_call.function.name} "
                                "not executed. Don't try to execute it again."
                            ),
                        )
                    )
                    continue

            tool_executions.append(self.run_function_call(function_call))
        return tool_executions

    def run_function_call(self, function_call: FunctionCall) -> ToolExecution:
        """Execute one resolved tool call and route the control-flow
        exceptions a tool can deliberately raise (`app/exceptions.py`).

        `FunctionCall.execute()` never raises for an *ordinary* tool
        failure - that already comes back as a failed `ToolExecution`.
        What it does raise (Stage 3 addition to `execute()`) is tool-code-
        initiated control flow:

          - `RunCancelledException` - re-raised untouched. A cancelled run
            isn't a tool result to report back to the model; it has to
            unwind the whole `response()` loop, so this doesn't catch it
            either.
          - `AgentRunException` (including `RetryAgentRun`/`StopAgentRun`,
            or a bare one a tool raises directly) - caught here and turned
            into a failed `ToolExecution`, using `.agent_message` as the
            result the model sees (falling back to `str(exc)` if none was
            given) and `.stop_execution` to set `stop_after_tool_call` -
            that's what `response()` checks to decide whether to break the
            loop after this round.
        """
        try:
            return function_call.execute()
        except RunCancelledException:
            raise
        except AgentRunException as e:
            logger.warning(f"{function_call.function.name} raised {type(e).__name__}: {e}")
            result = e.agent_message
            if isinstance(result, Message):
                result = result.content
            return ToolExecution(
                tool_call_id=function_call.call_id,
                tool_name=function_call.function.name,
                tool_args=function_call.arguments,
                tool_call_error=True,
                result=str(result) if result is not None else str(e),
                stop_after_tool_call=e.stop_execution,
            )

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "provider": self.get_provider()}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id!r}>"
