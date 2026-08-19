"""Tool declarations for the OpenAI Chat Completions API.

New file, no equivalent in real Agno - OpenAI's tool-call wire format
(`{"type": "function", "function": {"name", "description", "parameters"}}`)
is also the *canonical* shape every other provider's `format_tools_for_model`
(e.g. `utils/models/claude.py`) converts from, so building it lives in its
own file for the same reason `utils/models/claude.py` does: provider-specific
wire-format plumbing, one file per provider.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.tools.function import Function


def format_tools_for_model(functions: List[Function]) -> Optional[List[Dict[str, Any]]]:
    if not functions:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": f.name,
                "description": f.description or "",
                "parameters": f.parameters,
            },
        }
        for f in functions
    ]
