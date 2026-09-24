"""`agent_event_stream()` - the one function bridging Stage 4's
`Agent.run(stream=True)` (a sync generator of `RunOutputEvent` dataclasses)
to `sse_starlette.EventSourceResponse`'s expected shape (an iterable of
`{"event": ..., "data": ...}` dicts), mirroring real Agno's router-level SSE
framing around the identical `RunOutputEvent` vocabulary this project
already built in `run/agent.py`.

No new event vocabulary, no new run logic - this file's entire job is wire
format. `RunOutputEvent`'s member dataclasses (`RunStartedEvent`,
`RunContentEvent`, ..., `run/agent.py`) are all *plain* `@dataclass`es whose
own nested fields (`ToolExecution`, `RunMetrics`) are themselves plain
dataclasses too - so `dataclasses.asdict()` alone, with no per-field custom
serialization, already recurses correctly into every nested value. That's
why this file doesn't need a bespoke `to_dict()` per event type the way
`RunOutput`/`AgentSession` do (those hold `Message`s, a Pydantic type
`asdict()` doesn't know how to flatten).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Dict, Iterator, Optional

from wind.agent.agent import Agent
from wind.run.agent import RunOutputEvent

logger = logging.getLogger(__name__)


def agent_event_stream(
    agent: Agent,
    message: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Iterator[Dict[str, str]]:
    """Runs `agent.run(message, ..., stream=True)` and yields one SSE-ready
    dict per `RunOutputEvent` - `event` is the dataclass's own `.event`
    discriminator string (`RunEvent`, `run/agent.py`), `data` is that whole
    event JSON-encoded.

    Defensive `try/except` around the loop: `run_stream()` (`agent/_run.py`)
    already turns every exception it knows how to handle
    (`RunCancelledException`, `ModelProviderError`) into a terminal
    `RunCancelledEvent`/`RunErrorEvent` itself, so nothing should normally
    reach here - but an open SSE connection has no other way to report a
    truly unexpected exception (there's no HTTP status code left to set,
    headers are already flushed), so one is caught here, logged, and turned
    into a final synthetic `OSStreamError` frame rather than the connection
    just going silent mid-stream.
    """
    try:
        for event in agent.run(message, session_id=session_id, user_id=user_id, stream=True):
            event_dict: RunOutputEvent = event  # type: ignore[assignment]
            yield {"event": event_dict.event, "data": json.dumps(asdict(event_dict), default=str)}
    except Exception as exc:  # pragma: no cover - defensive, see docstring
        logger.exception("Unhandled exception mid-stream for agent %s", agent.id)
        yield {
            "event": "OSStreamError",
            "data": json.dumps({"content": str(exc)}),
        }
