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

Stage 4 addition: `response_stream()`, the streaming twin of `response()`.
The one-sentence version - streaming doesn't change what happens in a loop
iteration, it changes the granularity of observing it. `response()`'s
while-loop shape (call the model, check tool_calls, run them, decide
continue-or-break) is reused byte-for-byte; only what happens *inside* one
iteration's model call changes, from one atomic `invoke()` to a chunk-by-
chunk `invoke_stream()` sub-loop. Four pieces, matching the four generic
layers this splits into:

    MessageData                        the ACCUMULATOR's shape - one field
                                        per thing a stream of deltas needs
                                        collecting into
    invoke_stream()                    the TRANSPORT boundary (abstract here,
                                        same role as invoke()) - raw provider
                                        chunk -> ModelResponse delta, per
                                        provider adapter
    _populate_stream_data()            ACCUMULATOR Phase A - blind append/
                                        concat of each delta into MessageData,
                                        called once per chunk
    _populate_assistant_message_       ACCUMULATOR Phase B - the one real
      from_stream_data() /             reconstruction pass, called once
      parse_tool_calls()               after the stream ends (parse_tool_calls
                                        is the no-op default; OpenAIChat
                                        overrides it to merge by-index JSON
                                        argument fragments - Claude doesn't
                                        need to, see its own module)
    response_stream()                  the LOOP - same exits as response(),
                                        watched per-chunk instead of per-call

The CONSUMER layer (translating these into agent-level `RunOutputEvent`s)
lives one level up, in `agent/_response.py:handle_model_response_stream()` -
`response_stream()` itself has no idea an Agent exists.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from app.exceptions import AgentRunException, RunCancelledException
from app.metrics import MessageMetrics
from app.models.message import Message
from app.models.response import ModelResponse, ModelResponseEvent, ToolExecution
from app.tools.function import Function, FunctionCall
from app.utils.functions import get_function_call
from app.utils.models.openai import format_tools_for_model as format_tools_canonical

logger = logging.getLogger(__name__)


@dataclass
class MessageData:
    """Stage 4 - the streaming ACCUMULATOR's shape: one field per thing a
    `response_stream()` chunk can carry, collected across however many
    deltas one model call takes. Mirrors Agno's `MessageData`
    (`models/base.py`) trimmed to the subset this project's `Message`/
    `ModelResponse` actually have fields for - no reasoning, citations,
    audio/image/video/file, or provider_data.

    `response_content`/`response_tool_calls` start empty/`[]`, not `None` -
    `_populate_stream_data()` does `stream_data.response_content +=
    delta.content` on every chunk; starting from `None` would make the very
    first `+=` raise instead of concatenate.
    """

    response_role: Optional[str] = None
    response_content: str = ""
    response_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    response_metrics: Optional[MessageMetrics] = None
    extra: Optional[Dict[str, Any]] = None


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

    @abstractmethod
    def invoke_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[ModelResponse]:
        """Stage 4 - the TRANSPORT boundary for streaming, same ABC role as
        `invoke()`: format `messages` into the provider's wire shape, open a
        streaming request, and `yield` one `ModelResponse` *delta* per raw
        chunk the SDK hands back (via that module's
        `parse_provider_response_delta()`).

        Unlike `invoke()`, implementations do NOT call
        `self._populate_assistant_message()` themselves - a single chunk is
        a fragment, not a finished response, so there's nothing coherent to
        write onto `assistant_message` yet. That happens once, after the
        stream ends, in `_populate_assistant_message_from_stream_data()`
        below - `response_stream()` calls that itself once this generator
        is exhausted.
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

    def response_stream(
        self,
        messages: List[Message],
        functions: Optional[Dict[str, Function]] = None,
        tool_call_limit: Optional[int] = None,
    ) -> Iterator[ModelResponse]:
        """Streaming twin of `response()` - same while-loop, same exits
        (no tool_calls -> break; a tool sets stop_after_tool_call -> break;
        otherwise loop back with results appended), same in-place mutation
        of `messages`. The one structural difference: `response()` returns
        one final `ModelResponse` after the whole loop finishes; this
        `yield`s a `ModelResponse` for every event as it happens - content
        deltas (`event=None`) interleaved with lifecycle markers
        (`event=<ModelResponseEvent>`) - and never returns anything itself.
        A caller who wants the equivalent of `response()`'s return value has
        to reconstruct it from what it observed (that reconstruction is
        `agent/_response.py:handle_model_response_stream()`'s job, not
        this method's).

        One iteration's model call is `invoke_stream()`'s chunk-by-chunk
        sub-loop instead of one atomic `invoke()`: each chunk goes through
        `_populate_stream_data()` (Phase A - blind accumulation into
        `MessageData`), and once the sub-loop ends,
        `_populate_assistant_message_from_stream_data()` (Phase B) does the
        one reconstruction pass - tool-call argument fragments only get
        `parse_tool_calls()`-merged/parsed here, never mid-stream.
        """
        tool_declarations = format_tools_canonical(list(functions.values())) if functions else None

        tool_call_count = 0
        while True:
            assistant_message = Message(role="assistant")
            stream_data = MessageData()
            stream_data.response_metrics = MessageMetrics()
            stream_data.response_metrics.start_timer()

            yield ModelResponse(event=ModelResponseEvent.model_request_started.value)

            for delta in self.invoke_stream(messages, assistant_message, tools=tool_declarations):
                for piece in self._populate_stream_data(stream_data, delta):
                    yield piece

            stream_data.response_metrics.stop_timer()
            self._populate_assistant_message_from_stream_data(assistant_message, stream_data)
            messages.append(assistant_message)

            llm_metrics = assistant_message.metrics
            yield ModelResponse(
                event=ModelResponseEvent.model_request_completed.value,
                input_tokens=llm_metrics.input_tokens if llm_metrics else None,
                output_tokens=llm_metrics.output_tokens if llm_metrics else None,
                total_tokens=llm_metrics.total_tokens if llm_metrics else None,
                time_to_first_token=llm_metrics.time_to_first_token if llm_metrics else None,
            )

            if not assistant_message.tool_calls:
                break

            function_calls_to_run = self._prepare_function_calls(
                assistant_message=assistant_message,
                messages=messages,
                functions=functions,
            )

            stop = False
            for function_call in function_calls_to_run:
                # tool_call_started fires before execution - all a consumer
                # can show at this point is "the model wants to call X with
                # these (already fully-merged, by Phase B) arguments", not a
                # result yet.
                yield ModelResponse(
                    event=ModelResponseEvent.tool_call_started.value,
                    tool_executions=[
                        ToolExecution(
                            tool_call_id=function_call.call_id,
                            tool_name=function_call.function.name,
                            tool_args=function_call.arguments,
                        )
                    ],
                )

                tool_execution, tool_call_count = self._run_one_function_call_with_limit(
                    function_call, tool_call_count, tool_call_limit
                )

                yield ModelResponse(
                    event=ModelResponseEvent.tool_call_completed.value,
                    tool_executions=[tool_execution],
                )

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

    def _populate_stream_data(self, stream_data: MessageData, model_response_delta: ModelResponse) -> Iterator[ModelResponse]:
        """ACCUMULATOR Phase A: blind append/concat of one delta into
        `stream_data`, called once per chunk `invoke_stream()` yields.
        "Blind" is the point - Pattern 2 (collect dumb, merge smart, in two
        phases) says the real reconstruction work waits for Phase B
        (`_populate_assistant_message_from_stream_data()`), after the
        stream ends and every fragment has arrived. Tool-call argument
        fragments in particular are just concatenated string pieces here -
        `json.loads()`-parsing a half-arrived JSON string would fail, so
        nothing here even tries.

        Yields the delta back up to `response_stream()`'s caller only when
        it actually carries something new (`should_yield`) - a chunk with
        only `role` set, or only a trailing usage update with no content,
        produces no yield: nothing changed that a consumer watching content
        deltas would want to see.
        """
        should_yield = False

        if model_response_delta.role is not None:
            stream_data.response_role = model_response_delta.role

        if model_response_delta.response_usage is not None:
            if stream_data.response_metrics is None:
                stream_data.response_metrics = MessageMetrics()
                stream_data.response_metrics.start_timer()
            # In-place-looking accumulation (see MessageMetrics.__add__'s
            # docstring in metrics.py for why plain __add__ + reassignment
            # is enough here, no __iadd__ needed).
            stream_data.response_metrics += model_response_delta.response_usage

        if model_response_delta.content is not None:
            stream_data.response_content += model_response_delta.content
            if stream_data.response_metrics is not None and stream_data.response_metrics.time_to_first_token is None:
                stream_data.response_metrics.set_time_to_first_token()
            should_yield = True

        if model_response_delta.tool_calls:
            stream_data.response_tool_calls.extend(model_response_delta.tool_calls)
            should_yield = True

        if should_yield:
            yield model_response_delta

    def _populate_assistant_message_from_stream_data(self, assistant_message: Message, stream_data: MessageData) -> None:
        """ACCUMULATOR Phase B: the one real reconstruction pass, called
        once after `invoke_stream()`'s chunk sub-loop ends - copies the now-
        complete `stream_data` onto `assistant_message`, the same message
        `response_stream()`'s loop appends to `messages` and checks
        `.tool_calls` on to decide whether to keep going.

        `parse_tool_calls()` only runs here, never in Phase A - it's the
        no-op default below unless a provider adapter overrides it (OpenAI
        merges by-index JSON argument fragments; Claude doesn't need to,
        each of its deltas already carries one complete tool call - see
        each module's own docstring).
        """
        if stream_data.response_role is not None:
            assistant_message.role = stream_data.response_role
        if stream_data.response_metrics is not None:
            assistant_message.metrics = stream_data.response_metrics
        if stream_data.response_content:
            assistant_message.content = stream_data.response_content
        if stream_data.response_tool_calls:
            assistant_message.tool_calls = self.parse_tool_calls(stream_data.response_tool_calls)

    def parse_tool_calls(self, tool_calls_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge/parse the raw tool-call fragments `_populate_stream_data()`
        collected in `stream_data.response_tool_calls` into the same
        one-dict-per-call shape `Message.tool_calls`/`ModelResponse.tool_calls`
        already have everywhere else. No-op default - a provider whose
        deltas already arrive as complete, whole tool calls (Claude) has
        nothing to merge; one that streams argument JSON in index-addressed
        fragments (OpenAI) overrides this to reassemble them.
        """
        return tool_calls_data

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

    def _run_one_function_call_with_limit(
        self,
        function_call: FunctionCall,
        tool_call_count: int,
        tool_call_limit: Optional[int],
    ) -> "tuple[ToolExecution, int]":
        """One call's worth of the `tool_call_limit` guard, factored out of
        `run_function_calls()` so `response_stream()` (Stage 4) can apply the
        exact same limit accounting per-call, one at a time, instead of only
        as a batch - it needs to `yield` a `tool_call_started`/
        `tool_call_completed` event around each individual call, which
        `run_function_calls()`'s all-at-once loop has no hook for.

        Returns the `ToolExecution` plus the updated running count, since
        Python has no `int` pass-by-reference - the caller's loop variable
        has to be threaded through explicitly.
        """
        if tool_call_limit is not None:
            tool_call_count += 1
            if tool_call_count > tool_call_limit:
                logger.debug(
                    f"Tool call limit ({tool_call_limit}) reached. "
                    f"Skipping: {function_call.function.name} (call #{tool_call_count})"
                )
                return (
                    ToolExecution(
                        tool_call_id=function_call.call_id,
                        tool_name=function_call.function.name,
                        tool_args=function_call.arguments,
                        tool_call_error=True,
                        result=(
                            f"Tool call limit reached. Tool call {function_call.function.name} "
                            "not executed. Don't try to execute it again."
                        ),
                    ),
                    tool_call_count,
                )

        return self.run_function_call(function_call), tool_call_count

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
            tool_execution, tool_call_count = self._run_one_function_call_with_limit(
                function_call, tool_call_count, tool_call_limit
            )
            tool_executions.append(tool_execution)
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
