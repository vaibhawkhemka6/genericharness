"""Resolving a model's raw tool call into a runnable `FunctionCall`,
mirroring `agno/utils/functions.py:get_function_call` (the `cache_result`
decorator factory alongside it in real Agno is dropped - `Function` here
has no caching fields to drive it).

This is where a model's (occasionally sloppy) tool-call arguments get
repaired rather than rejected outright: `arguments` may arrive as JSON, as
a Python-literal string a model produced instead, or - for a no-argument
tool - as an empty list. String values of "true"/"false"/"none" are coerced
to their real types, since some models emit those instead of JSON booleans/
null.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from app.tools.function import Function, FunctionCall

logger = logging.getLogger(__name__)


def get_function_call(
    name: str,
    arguments: Optional[str] = None,
    call_id: Optional[str] = None,
    functions: Optional[Dict[str, Function]] = None,
) -> Optional[FunctionCall]:
    """Look up `name` in `functions` and build a `FunctionCall` for it,
    parsing `arguments` (a JSON string, as models send it) along the way.
    Returns `None` only if `functions`/`name` don't resolve to a known
    `Function` - a parse failure still returns a `FunctionCall`, but with
    `.error` set, so the caller can execute() it into a failed
    `ToolExecution` the model can see and retry from."""
    if functions is None:
        return None

    function_to_call = functions.get(name)
    if function_to_call is None:
        logger.error(f"Function {name} not found")
        return None

    function_call = FunctionCall(function=function_to_call)
    if call_id is not None:
        function_call.call_id = call_id
    if arguments is not None and arguments != "":
        try:
            try:
                parsed_arguments = json.loads(arguments)
            except Exception:
                import ast

                parsed_arguments = ast.literal_eval(arguments)
        except Exception as e:
            logger.error(f"Unable to decode function arguments:\n{arguments}\nError: {e}")
            function_call.error = (
                f"Error while decoding function arguments: {e}\n\n"
                f"Please make sure we can json.loads() the arguments and retry."
            )
            return function_call

        # Models sometimes pass [] instead of {} for a no-argument tool.
        if isinstance(parsed_arguments, list) and len(parsed_arguments) == 0:
            return function_call

        if not isinstance(parsed_arguments, dict):
            logger.error(f"Function arguments are not a valid JSON object: {arguments}")
            function_call.error = "Function arguments are not a valid JSON object.\n\n Please fix and retry."
            return function_call

        clean_arguments: Dict[str, Any] = {}
        for k, v in parsed_arguments.items():
            if isinstance(v, str):
                lowered = v.strip().lower()
                if lowered in ("none", "null"):
                    clean_arguments[k] = None
                elif lowered == "true":
                    clean_arguments[k] = True
                elif lowered == "false":
                    clean_arguments[k] = False
                else:
                    clean_arguments[k] = v
            else:
                clean_arguments[k] = v
        function_call.arguments = clean_arguments

    return function_call
