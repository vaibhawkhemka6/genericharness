"""Message formatting for the Anthropic Messages API, mirroring Agno's
`agno/utils/models/claude.py` trimmed to the single function this project
needs: turning our internal `List[Message]` into the `(chat_messages,
system_prompt)` tuple that `client.messages.create()` expects.

New file, not a trimmed copy of an existing one in this project - Anthropic
and OpenAI use different wire formats even though both are "chat completions
with tool calls" underneath. OpenAI's system/tool messages are just more
entries in one flat `messages` list (see `models/openai/chat.py`);
Anthropic pulls system messages out into a separate top-level `system`
string and represents tool results as `tool_result` content blocks nested
inside a role="user" message, since Anthropic has no role="tool". Agno keeps
a `utils/models/<provider>.py` per provider for exactly this kind of
format-specific plumbing, so this project does too.

Trimmed against real Agno: no media (images/audio/video/files), no
thinking/redacted-thinking blocks, no citations, no assistant-prefill
trailing-message workaround, no tool-result compression. Assistant/tool
content blocks are built as plain dicts rather than typed
`anthropic.types.TextBlock`/`ToolUseBlock` objects - the Anthropic SDK
accepts both on the way in, and dicts avoid importing/tracking those types
this early.

`format_tools_for_model()` (ported from `agno/utils/models/claude.py:669`,
trimmed - no `$defs`/`definitions` passthrough for MCP-style raw schemas,
no strict-mode flag) converts the OpenAI-shaped tool declarations built by
`utils/models/openai.py` into Anthropic's `{"name", "description",
"input_schema"}` shape. OpenAI's shape doubles as the canonical one here
because every provider adapter's tool declarations start from the same
`List[Function]` - see `utils/models/openai.py`'s docstring.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from app.models.message import Message

# Anthropic's role set is narrower than ours: no separate "developer" role,
# and tool results go back as a role="user" message with a tool_result
# content block, not a role="tool" message (Anthropic has no such role).
ROLE_MAP = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "user",
}


def format_messages(messages: List[Message]) -> Tuple[List[Dict[str, Any]], str]:
    """Split `messages` into the `(chat_messages, system_prompt)` tuple
    Anthropic's `messages.create()` expects.

    System messages don't go in `messages` at all - Anthropic takes them as
    a separate `system` string, so they're pulled out and concatenated.
    Everything else becomes a `{"role": ..., "content": [...]}` entry, with
    assistant tool calls rendered as `tool_use` blocks and tool results
    rendered as `tool_result` blocks. Claude requires strictly alternating
    user/assistant turns, so consecutive same-role messages - e.g. two tool
    results in a row - are merged into one turn afterwards.
    """
    chat_messages: List[Dict[str, Any]] = []
    system_messages: List[str] = []

    for message in messages:
        content: Any = message.content or ""

        if message.role == "system":
            if content:
                system_messages.append(content)  # type: ignore[arg-type]
            continue

        elif message.role == "user":
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

        elif message.role == "assistant":
            content = []
            if isinstance(message.content, str) and message.content.strip():
                content.append({"type": "text", "text": message.content})

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function = tool_call["function"]
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": function["name"],
                            "input": json.loads(function["arguments"]) if function.get("arguments") else {},
                        }
                    )

            # Skip empty assistant turns (nothing to say, no tool call).
            if not content:
                continue

        elif message.role == "tool":
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.get_content_string(),
                }
            ]

        chat_messages.append({"role": ROLE_MAP[message.role], "content": content})

    # Claude requires alternating user/assistant turns - merge consecutive
    # same-role messages (e.g. back-to-back tool results) into one turn.
    merged_messages: List[Dict[str, Any]] = []
    for msg in chat_messages:
        if merged_messages and merged_messages[-1]["role"] == msg["role"]:
            merged_messages[-1]["content"].extend(msg["content"])
        else:
            merged_messages.append(msg)

    return merged_messages, " ".join(system_messages)


def _ensure_additional_properties_false(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively set `additionalProperties: false` on every object schema,
    not just the root - Anthropic requires this at every level.

    Only applied to *structured* object schemas (those with `"properties"`) -
    a dict-typed schema uses `additionalProperties` as its value-type schema
    (see `json_schema.py:get_json_schema_for_arg`'s `dict` branch), so those
    are left alone.
    """
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()
    if "properties" in result:
        result["additionalProperties"] = False

    for key, value in result.items():
        if key == "properties" and isinstance(value, dict):
            result[key] = {k: _ensure_additional_properties_false(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = _ensure_additional_properties_false(value)
        elif key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
            result[key] = [_ensure_additional_properties_false(v) if isinstance(v, dict) else v for v in value]

    return result


def format_tools_for_model(tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI-shaped tool declarations (`{"type": "function",
    "function": {"name", "description", "parameters"}}`, as built by
    `utils/models/openai.py:format_tools_for_model`) into Anthropic's
    `{"name", "description", "input_schema"}` shape."""
    if not tools:
        return None

    parsed_tools: List[Dict[str, Any]] = []
    for tool_def in tools:
        if tool_def.get("type", "") != "function":
            parsed_tools.append(tool_def)
            continue

        func_def = tool_def.get("function", {})
        parameters: Dict[str, Any] = func_def.get("parameters", {})
        properties: Dict[str, Any] = parameters.get("properties", {})

        input_properties: Dict[str, Any] = {}
        for param_name, param_info in properties.items():
            input_properties[param_name] = _ensure_additional_properties_false(param_info.copy())
            if "description" not in input_properties[param_name]:
                input_properties[param_name]["description"] = ""

        input_schema: Dict[str, Any] = {
            "type": parameters.get("type", "object"),
            "properties": input_properties,
            "required": parameters.get("required", []),
            "additionalProperties": False,
        }

        parsed_tools.append(
            {
                "name": func_def.get("name") or "",
                "description": func_def.get("description") or "",
                "input_schema": input_schema,
            }
        )

    return parsed_tools
