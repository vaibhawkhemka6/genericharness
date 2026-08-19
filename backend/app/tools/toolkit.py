"""A named group of tools, mirroring `agno/tools/toolkit.py` (500 lines)
trimmed to the same subset `tools/function.py`/`tools/decorator.py`
support: plain synchronous, non-cached tools.

Two ways to build one:

    web = Toolkit(name="web", tools=[search, fetch])   # plain functions

    class Web(Toolkit):                                 # @tool-decorated
        def __init__(self):
            super().__init__(name="web", tools=[self.search, self.fetch])

        @tool
        def search(self, query: str) -> str:
            \"\"\"Search the web.

            Args:
                query: The search query.
            \"\"\"
            ...

Trimmed against real Agno: no caching, hooks, confirmation/user-input/
external-execution (HITL), async variants, connect()/close() connection
management, or path-safety helpers - `Function`/`FunctionCall` don't carry
any of those fields yet either, so accepting them here would be silently
inert. `include_tools`/`exclude_tools` and the `stop_after_tool_call_tools`/
`show_result_tools` name-list overrides are kept since they're the one
piece of toolkit-level configuration that doesn't depend on any of the
above.
"""

from __future__ import annotations

import logging
from inspect import signature
from typing import Callable, Dict, List, Optional, Sequence, Union

from app.tools.function import Function

logger = logging.getLogger(__name__)


class Toolkit:
    def __init__(
        self,
        name: str = "toolkit",
        tools: Optional[Sequence[Union[Callable, Function]]] = None,
        instructions: Optional[str] = None,
        include_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
        stop_after_tool_call_tools: Optional[List[str]] = None,
        show_result_tools: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.tools: Sequence[Union[Callable, Function]] = tools or []
        self.instructions = instructions
        self.include_tools = include_tools
        self.exclude_tools = exclude_tools
        self.stop_after_tool_call_tools = stop_after_tool_call_tools or []
        self.show_result_tools = show_result_tools or []

        # Name -> Function, in registration order. What the agent loop
        # actually reads (get_functions()).
        self.functions: Dict[str, Function] = {}

        self._check_filters([self._tool_name(t) for t in self.tools])
        for t in self.tools:
            self.register(t)

    def _tool_name(self, tool: Union[Callable, Function]) -> str:
        return tool.name if isinstance(tool, Function) else tool.__name__

    def _check_filters(self, available: List[str]) -> None:
        if self.include_tools:
            missing = set(self.include_tools) - set(available)
            if missing:
                raise ValueError(f"Included tool(s) not present in the toolkit: {', '.join(missing)}")
        if self.exclude_tools:
            missing = set(self.exclude_tools) - set(available)
            if missing:
                raise ValueError(f"Excluded tool(s) not present in the toolkit: {', '.join(missing)}")

    def register(self, tool: Union[Callable, Function], name: Optional[str] = None) -> None:
        """Add one tool to `self.functions`, applying `include_tools`/
        `exclude_tools` and the toolkit-level `stop_after_tool_call_tools`/
        `show_result_tools` name-list overrides.

        `tool` can be a plain callable (a module-level function, or an
        already-bound instance method like `self.search`) or a `Function`
        (an `@tool`-decorated method, accessed as `self.search` where
        `search` is a class attribute - see `_bind_decorated_tool`).
        """
        function = self._bind_decorated_tool(tool) if isinstance(tool, Function) else Function.from_callable(tool)
        if name is not None:
            function.name = name

        tool_name = function.name
        if self.include_tools is not None and tool_name not in self.include_tools:
            return
        if self.exclude_tools is not None and tool_name in self.exclude_tools:
            return

        if tool_name in self.stop_after_tool_call_tools:
            function.stop_after_tool_call = True
        if tool_name in self.show_result_tools or function.stop_after_tool_call:
            function.show_result = True

        self.functions[function.name] = function

    def _bind_decorated_tool(self, function: Function) -> Function:
        """A `@tool`-decorated instance method's `Function.entrypoint` is
        the *unbound* function - the decorator runs at class-definition
        time, before any instance (and thus no `self`) exists. Accessing it
        later as `self.search` doesn't rebind it either, since `Function`
        isn't a descriptor.

        `Toolkit.__init__` only ever runs from inside a subclass's own
        `__init__` (via `super().__init__(...)`), so by the time this runs,
        `self` here *is* the instance those methods belong to - close over
        it to rebuild the same `Function` with `self` pre-applied, keeping
        the schema/description the decorator already computed rather than
        re-deriving them from the now-different (missing `self`) signature.
        """
        entrypoint = function.entrypoint
        if entrypoint is None:
            return function

        params = list(signature(entrypoint).parameters)
        if not params or params[0] != "self":
            return function

        def bound(*args, **kwargs):
            return entrypoint(self, *args, **kwargs)

        bound.__name__ = function.name
        return function.model_copy(update={"entrypoint": bound})

    def get_functions(self) -> Dict[str, Function]:
        return self.functions

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} functions={list(self.functions)}>"
