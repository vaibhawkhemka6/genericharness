"""Stage 3 Half B - message assembly: turn `agent` config + raw `run()`
input into the `RunMessages` `Model.response()` will mutate in place.

Three pieces, run in a fixed order by `get_run_messages()`:

    get_system_message()             agent.description + agent.instructions -> one system Message
    get_session_history_messages()   a session-history stub (Stage 5 replaces this with a real lookup)
    get_user_message()               raw run() input -> one user Message

Trimmed against real Agno's `Agent.get_system_message()`/`get_user_message()`
(both live inside the ~5000-line `Agent` class there): no tool-use
instructions, no expected-output/markdown formatting, no date/timezone
injection, no memory or knowledge-base retrieval, no per-message media. Free
functions taking `agent` as a parameter rather than methods on `Agent` -
same reasoning as `models/*/chat.py`'s translation functions: lets a future
stage script call `get_system_message(agent)` directly, by hand, without
needing a live `Agent.run()` to exercise it.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from app.agent.agent import Agent
from app.models.message import Message
from app.run.messages import RunMessages


def get_system_message(agent: Agent) -> Optional[Message]:
    """`agent.description` then `agent.instructions`, blank-line joined.
    `instructions` may be a single string or a list of lines (rendered as a
    "- " bullet each, matching how most model providers expect a short
    instruction list to look). Returns `None` if the agent has neither -
    `get_run_messages()` then omits the system message entirely rather
    than sending an empty one.
    """
    sections: List[str] = []

    if agent.description:
        sections.append(agent.description.strip())

    if agent.instructions:
        if isinstance(agent.instructions, str):
            sections.append(agent.instructions.strip())
        else:
            sections.append("\n".join(f"- {line}" for line in agent.instructions))

    if not sections:
        return None

    return Message(role="system", content="\n\n".join(sections))


def get_user_message(input: Union[str, List, Dict, Message, BaseModel]) -> Optional[Message]:
    """Wrap raw `run()` input into a single `role="user"` `Message`.

    A `Message` passed in is returned as-is (its role stays whatever the
    caller set - not forced to "user"). A `List[Message]` is deliberately
    *not* handled here - `get_run_messages()` treats that shape as the
    whole turn already assembled by the caller, not a single value to wrap,
    and never calls this function for it.
    """
    if input is None:
        return None
    if isinstance(input, Message):
        return input
    if isinstance(input, str):
        content = input
    elif isinstance(input, BaseModel):
        content = input.model_dump_json(exclude_none=True)
    else:
        content = json.dumps(input, ensure_ascii=False, default=str)
    return Message(role="user", content=content)


def get_session_history_messages(
    agent: Agent,
    session_history: Optional[List[Message]] = None,
) -> List[Message]:
    """Stand-in for a real session/db history lookup (Stage 5 - `session/`,
    `db/` don't exist yet). `session_history` is just a plain list the
    caller already has in memory - falls back to `agent.session_history`
    (itself just a stub field, see `agent.py`) if the caller doesn't pass
    one - then trims to the last `agent.num_history_messages`, if set.

    Always returns a list, never `None` - `get_run_messages()` extends
    directly with this, it shouldn't have to null-check it.
    """
    history = session_history if session_history is not None else agent.session_history
    if not history:
        return []
    if agent.num_history_messages is not None:
        return list(history[-agent.num_history_messages :])
    return list(history)


def get_run_messages(
    agent: Agent,
    input: Union[str, List, Dict, Message, BaseModel, List[Message]],
    session_history: Optional[List[Message]] = None,
) -> RunMessages:
    """The 4-step assembler: system -> history -> user -> done.

    Order matters and is the entire point of this function: history has to
    land *between* the system message and the current turn, not after it,
    or the model sees this turn's question before it sees what was already
    discussed.

    `RunMessages.messages` (not `get_input_messages()`) is the field this
    project's loop actually reads - `Model.response()` mutates that list
    directly - so that's where the system/history/user order above is
    guaranteed. `system_message`/`user_message` are also set, for callers
    that want the pieces individually.

    DIVERGENCE from `RunMessages`'s own docstring (`run/messages.py`,
    unmodified this stage): `extra_messages` there is documented as
    "messages added after the system and user messages". This function
    repurposes it to hold *history* instead - added between system and
    user - since that's the only other list-shaped field `RunMessages` has,
    and this project isn't touching `run/messages.py` this stage.
    `get_input_messages()` (system, user, extra - in that order) is
    therefore NOT history-correct for this project's use of
    `extra_messages`; nothing in this project calls it, only `.messages`.
    """
    run_messages = RunMessages()

    run_messages.system_message = get_system_message(agent)

    history = get_session_history_messages(agent=agent, session_history=session_history)
    run_messages.extra_messages = history or None

    if isinstance(input, list) and input and all(isinstance(item, Message) for item in input):
        # Caller already built the whole turn as Message objects (e.g.
        # replaying/forking a conversation) - use it as-is, no single
        # user_message to extract.
        turn_messages = list(input)
        run_messages.user_message = None
    else:
        user_message = get_user_message(input)
        run_messages.user_message = user_message
        turn_messages = [user_message] if user_message is not None else []

    ordered: List[Message] = []
    if run_messages.system_message is not None:
        ordered.append(run_messages.system_message)
    ordered.extend(history)
    ordered.extend(turn_messages)

    run_messages.messages = ordered
    return run_messages
