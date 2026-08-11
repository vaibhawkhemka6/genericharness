"""Tool registry: turns plain Python functions into model-callable tools.

Modeled on Agno's approach: decorate a normal function with @tool, and the
decorator introspects its signature + docstring to build a JSON schema the
model can call. No provider-specific code lives here - model adapters convert
Tool.to_json_schema() into whatever shape their API needs.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints

_PY_TYPE_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_ARG_LINE_RE = re.compile(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    if not doc:
        return "", {}

    lines = doc.strip().splitlines()
    summary_lines: list[str] = []
    arg_descriptions: dict[str, str] = {}
    in_args_section = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue
        if stripped.lower() in ("returns:", "return:", "raises:"):
            in_args_section = False
            continue
        if in_args_section:
            match = _ARG_LINE_RE.match(line)
            if match:
                arg_descriptions[match.group(1)] = match.group(2).strip()
        elif stripped:
            summary_lines.append(stripped)

    summary = " ".join(summary_lines).strip()
    return summary, arg_descriptions


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # {"properties": {...}, "required": [...]}
    fn: Callable

    def to_json_schema(self) -> dict[str, Any]:
        """Provider-neutral schema. Adapters reshape this as needed."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", []),
            },
        }

    def execute(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def tool(fn: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None) -> Tool:
    """Decorator that turns a function into a Tool with an auto-derived schema.

    Usage:
        @tool
        def get_current_datetime() -> str:
            '''Return the current date and time in ISO format.'''
            ...

        @tool
        def calculator(expression: str) -> str:
            '''Evaluate a basic arithmetic expression.

            Args:
                expression: The arithmetic expression to evaluate, e.g. "2 + 2 * 3".
            '''
            ...
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        summary, arg_descriptions = _parse_docstring(inspect.getdoc(func) or "")
        tool_description = description or summary or tool_name

        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            py_type = hints.get(param_name, str)
            json_type = _PY_TYPE_TO_JSON_TYPE.get(py_type, "string")
            prop: dict[str, Any] = {"type": json_type}
            if param_name in arg_descriptions:
                prop["description"] = arg_descriptions[param_name]
            properties[param_name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return Tool(
            name=tool_name,
            description=tool_description,
            parameters={"properties": properties, "required": required},
            fn=func,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


class ToolRegistry:
    """Holds a set of Tools and executes them by name."""

    def __init__(self, tools: Optional[list[Tool]] = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        self._tools[t.name] = t

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_json_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        t = self.get(name)
        if t is None:
            raise ValueError(f"Unknown tool: {name!r}. Available: {list(self._tools)}")
        return t.execute(**arguments)
