"""Python type hints -> JSON Schema, mirroring Agno's `agno/utils/json_schema.py`
trimmed to the type shapes this project's tools actually need.

Two entry points:

    get_json_schema_for_arg()  one type hint -> one JSON Schema fragment
    get_json_schema()          a whole {param_name: type_hint} dict -> the
                                 "parameters" object a tool declaration needs

Trimmed against real Agno: no Pydantic `BaseModel`/`@dataclass` field
introspection (nested-object tool args aren't a Stage 1 need - a tool
argument is a primitive, a list of one, a dict, a `Literal`, or an `Enum`),
so `inline_pydantic_schema` and its `$ref`-resolution machinery are dropped
along with it. Everything else - primitives, containers, `Literal`, `Enum`,
`Optional`/`Union` - mirrors the real function.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union, get_args, get_origin

logger = logging.getLogger(__name__)


def is_origin_union_type(origin: Any) -> bool:
    """True for both `typing.Union` and, on 3.10+, the `X | Y` `UnionType`."""
    if sys.version_info.minor >= 10:
        from types import UnionType  # type: ignore

        return origin in (Union, UnionType)
    return origin is Union


def get_json_type_for_py_type(arg: str) -> str:
    """Map a Python type's `__name__` onto its JSON Schema `"type"`."""
    if arg == "int":
        return "integer"
    elif arg in ("float", "complex", "Decimal"):
        return "number"
    elif arg in ("str", "string"):
        return "string"
    elif arg in ("bool", "boolean"):
        return "boolean"
    elif arg in ("NoneType", "None"):
        return "null"
    elif arg in ("list", "tuple", "set", "frozenset"):
        return "array"
    elif arg in ("dict", "mapping"):
        return "object"
    # Unrecognized types (custom classes, etc.) fall back to "object" rather
    # than raising - a tool with an odd argument type should still get a
    # schema, even an imprecise one.
    return "object"


def get_json_schema_for_arg(type_hint: Any) -> Optional[Dict[str, Any]]:
    """One type hint -> one JSON Schema fragment (no `description`, no name -
    that's `get_json_schema()`'s job, since only it has the parameter name)."""
    type_args = get_args(type_hint)
    type_origin = get_origin(type_hint)

    if type_origin is not None:
        if type_origin is Literal:
            # Order matters: bool before int, since bool is an int subclass.
            if type_args:
                if all(isinstance(arg, str) for arg in type_args):
                    return {"type": "string", "enum": list(type_args)}
                elif all(isinstance(arg, bool) for arg in type_args):
                    return {"type": "boolean", "enum": list(type_args)}
                elif all(isinstance(arg, int) and not isinstance(arg, bool) for arg in type_args):
                    return {"type": "integer", "enum": list(type_args)}
                elif all(isinstance(arg, (int, float)) and not isinstance(arg, bool) for arg in type_args):
                    return {"type": "number", "enum": list(type_args)}
                return {"enum": list(type_args)}
            return {"type": "string"}
        elif type_origin in (list, tuple, set, frozenset):
            items = get_json_schema_for_arg(type_args[0]) if type_args else {"type": "string"}
            return {"type": "array", "items": items}
        elif type_origin is dict:
            key_schema = get_json_schema_for_arg(type_args[0]) if type_args else {"type": "string"}
            value_schema = get_json_schema_for_arg(type_args[1]) if len(type_args) > 1 else {"type": "string"}
            return {"type": "object", "propertyNames": key_schema, "additionalProperties": value_schema}
        elif is_origin_union_type(type_origin):
            types = []
            for arg in type_args:
                try:
                    schema = get_json_schema_for_arg(arg)
                    if schema:
                        types.append(schema)
                except Exception:
                    continue
            return {"anyOf": types} if types else None

    if isinstance(type_hint, type) and issubclass(type_hint, Enum):
        return {"type": "string", "enum": [member.value for member in type_hint]}

    # Bare `dict` means "arbitrary key-value pairs" - allow any properties.
    if type_hint is dict:
        return {"type": "object", "additionalProperties": True}

    json_schema: Dict[str, Any] = {"type": get_json_type_for_py_type(getattr(type_hint, "__name__", "object"))}
    if json_schema["type"] == "object":
        json_schema["properties"] = {}
        json_schema["additionalProperties"] = False
    return json_schema


def get_json_schema(
    type_hints: Dict[str, Any],
    param_descriptions: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """`{param_name: type_hint}` (as returned by `typing.get_type_hints()`) ->
    the `{"type": "object", "properties": {...}}` schema a tool declaration's
    `parameters` field needs. `Optional[X]` is unwrapped to `X` - optionality
    is expressed by omission from the caller's `required` list, not encoded
    into the property's own schema."""
    json_schema: Dict[str, Any] = {"type": "object", "properties": {}}
    if strict:
        json_schema["additionalProperties"] = False

    for parameter_name, type_hint in type_hints.items():
        if parameter_name == "return":
            continue

        try:
            type_origin = get_origin(type_hint)
            type_args = get_args(type_hint)
            is_optional = (
                type_origin is Union and len(type_args) == 2 and any(arg is type(None) for arg in type_args)
            )
            if is_optional:
                type_hint = next(arg for arg in type_args if arg is not type(None))

            arg_json_schema = get_json_schema_for_arg(type_hint) if type_hint else {}

            if arg_json_schema is not None:
                if param_descriptions and param_descriptions.get(parameter_name):
                    arg_json_schema["description"] = param_descriptions[parameter_name]
                json_schema["properties"][parameter_name] = arg_json_schema
            else:
                logger.warning(f"Could not parse argument {parameter_name} of type {type_hint}")
        except Exception:
            logger.exception(f"Error processing argument {parameter_name}")
            continue

    return json_schema
