"""Tool representation and execution, mirroring `agno/tools/function.py`
trimmed to the two types this project's agent loop needs:

    Function      a callable's name/description/JSON-schema-parameters,
                   built by introspecting the callable (`from_callable`)
    FunctionCall   one model-requested invocation of a Function - arguments
                   already resolved, ready to `execute()`

Trimmed against real Agno (a 2500+ line file): no caching, no pre/post
hooks or tool_hooks chain, no confirmation/user-input/external-execution
(HITL) flow, no `agent`/`team`/`run_context`/media parameter injection, no
generator-returning tools, no strict-mode schema post-processing beyond
what `get_json_schema` already does. Those all assume a live `Agent`/`Team`
this stage doesn't have yet. `execute()` also returns this project's own
`ToolExecution` (`models/response.py`) directly rather than a separate
`FunctionExecutionResult` type - one record type for "the outcome of a
tool call" is enough until something besides the agent loop needs to see a
different shape of it.
"""

from __future__ import annotations

import json
import logging
from inspect import getdoc, signature
from typing import Any, Callable, Dict, Optional, get_type_hints

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, ConfigDict

from app.metrics import ToolCallMetrics
from app.models.response import ToolExecution
from app.utils.json_schema import get_json_schema

logger = logging.getLogger(__name__)


def get_entrypoint_docstring(entrypoint: Callable) -> str:
    """A callable's docstring, short + long description joined - the same
    text used both as a tool's model-facing `description` (when none is
    given explicitly) and for parameter descriptions via `parse_docstring`."""
    docstring = getdoc(entrypoint)
    if not docstring:
        return ""

    parsed = parse_docstring(docstring)
    lines = []
    if parsed.short_description:
        lines.append(parsed.short_description)
    if parsed.long_description:
        lines.extend(parsed.long_description.split("\n"))
    return "\n".join(lines)


class Function(BaseModel):
    """A callable an agent can invoke, plus the JSON Schema describing its
    arguments to a model."""

    name: str
    description: Optional[str] = None
    # JSON Schema object describing the callable's parameters.
    parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    strict: Optional[bool] = None

    # The function to be called. Excluded from to_dict() - never sent to a
    # model, and not JSON-serializable.
    entrypoint: Optional[Callable] = None

    # If True, the agent stops executing after this tool call.
    stop_after_tool_call: bool = False
    # If True, the tool's result is also surfaced to the caller verbatim
    # (e.g. shown in a UI) rather than only fed back to the model.
    show_result: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_callable(cls, c: Callable, name: Optional[str] = None, strict: bool = False) -> "Function":
        """Build a `Function` by introspecting `c`'s signature, type hints,
        and docstring (Google/NumPy/reST style, via `docstring_parser`)."""
        function_name = name or c.__name__
        parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        try:
            sig = signature(c)
            type_hints = get_type_hints(c)
            param_type_hints = {n: type_hints.get(n) for n in sig.parameters if n != "self"}

            param_descriptions: Dict[str, str] = {}
            if docstring := getdoc(c):
                for param in parse_docstring(docstring).params:
                    param_descriptions[param.arg_name] = param.description or ""

            parameters = get_json_schema(
                type_hints=param_type_hints, param_descriptions=param_descriptions, strict=strict
            )

            if strict:
                # Structured-output APIs that support strict mode require
                # every property to be listed as required.
                parameters["required"] = list(parameters["properties"])
            else:
                parameters["required"] = [
                    n for n, p in sig.parameters.items() if p.default is p.empty and n != "self"
                ]
            # A param whose type had no JSON schema (skipped by get_json_schema)
            # can't be required - that would describe a property that doesn't exist.
            parameters["required"] = [n for n in parameters["required"] if n in parameters["properties"]]
        except Exception as e:
            logger.warning(f"Could not parse args for {function_name}: {e}")

        return cls(
            name=function_name,
            description=get_entrypoint_docstring(c),
            parameters=parameters,
            entrypoint=c,
            strict=strict,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict,
                "stop_after_tool_call": self.stop_after_tool_call,
                "show_result": self.show_result,
            }.items()
            if v is not None
        }


class FunctionCall(BaseModel):
    """One model-requested invocation of a `Function`, with its arguments
    already resolved (see `utils/functions.py:get_function_call`)."""

    function: Function
    arguments: Optional[Dict[str, Any]] = None
    call_id: Optional[str] = None
    result: Optional[Any] = None

    # Set by get_function_call() when the model's arguments couldn't be
    # parsed - execute() short-circuits to a failed ToolExecution instead
    # of calling the entrypoint with bad/missing arguments.
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_call_str(self) -> str:
        args = self.arguments or {}
        return f"{self.function.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"

    def execute(self) -> ToolExecution:
        """Run the entrypoint and return the outcome as a `ToolExecution` -
        never raises; a failing tool is a failed `ToolExecution`, not an
        exception the agent loop has to catch."""
        if self.error is not None:
            return ToolExecution(
                tool_call_id=self.call_id,
                tool_name=self.function.name,
                tool_args=self.arguments,
                tool_call_error=True,
                result=self.error,
                stop_after_tool_call=self.function.stop_after_tool_call,
            )

        if self.function.entrypoint is None:
            return ToolExecution(
                tool_call_id=self.call_id,
                tool_name=self.function.name,
                tool_args=self.arguments,
                tool_call_error=True,
                result="Entrypoint is not set",
                stop_after_tool_call=self.function.stop_after_tool_call,
            )

        metrics = ToolCallMetrics()
        metrics.start_timer()
        try:
            logger.debug(f"Running: {self.get_call_str()}")
            self.result = self.function.entrypoint(**(self.arguments or {}))
            result_str = self.result if isinstance(self.result, str) else json.dumps(self.result, default=str)
            tool_call_error = False
        except Exception as e:
            logger.warning(f"Error running tool {self.get_call_str()}: {e}")
            result_str = str(e)
            tool_call_error = True
        metrics.stop_timer()

        return ToolExecution(
            tool_call_id=self.call_id,
            tool_name=self.function.name,
            tool_args=self.arguments,
            tool_call_error=tool_call_error,
            result=result_str,
            metrics=metrics,
            stop_after_tool_call=self.function.stop_after_tool_call,
        )
