"""The `@tool` decorator, mirroring `agno/tools/decorator.py` trimmed to
plain synchronous functions.

Trimmed against real Agno: no sync/async/async-generator wrapper dispatch
(`Function.entrypoint` here is always the callable as written - async tools
are a later stage), no cache/pre-hook/post-hook/tool_hooks/HITL kwargs -
`Function` (`tools/function.py`) doesn't carry any of those yet either, so
accepting them here would be silently inert.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

from app.tools.function import Function

F = TypeVar("F", bound=Callable[..., Any])

_VALID_KWARGS = frozenset({"name", "description", "strict", "show_result", "stop_after_tool_call"})


@overload
def tool(func: F) -> Function: ...


@overload
def tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict: Optional[bool] = None,
    show_result: Optional[bool] = None,
    stop_after_tool_call: Optional[bool] = None,
) -> Callable[[F], Function]: ...


def tool(*args: Any, **kwargs: Any) -> Union[Function, Callable[[F], Function]]:
    """Turn a plain function into a `Function` the agent loop can call.

        @tool
        def get_weather(city: str) -> str:
            \"\"\"Get the current weather for a city.

            Args:
                city: The city to look up.
            \"\"\"
            ...

        @tool(name="weather", stop_after_tool_call=True)
        def get_weather(city: str) -> str:
            ...
    """
    invalid = set(kwargs) - _VALID_KWARGS
    if invalid:
        raise ValueError(f"Invalid tool configuration arguments: {invalid}. Valid arguments are: {sorted(_VALID_KWARGS)}")

    def decorator(func: F) -> Function:
        function = Function.from_callable(func, name=kwargs.get("name"), strict=bool(kwargs.get("strict")))
        if kwargs.get("description") is not None:
            function.description = kwargs["description"]
        if kwargs.get("show_result") is not None:
            function.show_result = kwargs["show_result"]
        if kwargs.get("stop_after_tool_call") is not None:
            function.stop_after_tool_call = kwargs["stop_after_tool_call"]
        return function

    # Bare @tool (no parens): args = (func,), kwargs = {}.
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])

    return decorator
