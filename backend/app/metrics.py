"""Token usage + timing metrics, mirroring Agno's `agno/metrics.py`
(trimmed to the token/duration/timer subset this project needs - no audio,
cache, or cost accounting).

Leaf module - imports nothing from the rest of the project, so every other
module can depend on it without creating an import cycle.

`__add__` (on `MessageMetrics` and `RunMetrics`) is the load-bearing method
here: a run makes one model call per tool-call iteration, each producing
its own `MessageMetrics`, and those need to accumulate into a single
per-run total. Without `__add__`, that summing logic (token totals, "keep
whichever side has a value" for duration/time-to-first-token) would get
duplicated everywhere a caller needs a running total - the agent loop, a
session-level rollup, tests - instead of living in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from time import perf_counter, time
from typing import Any, Dict, Optional


class Timer:
    """Wall-clock timer for measuring a run's (or a single message's) duration."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

    def start(self) -> float:
        self.start_time = perf_counter()
        return self.start_time

    def stop(self) -> float:
        self.end_time = perf_counter()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time
        return self.end_time

    @property
    def elapsed(self) -> float:
        """Elapsed seconds so far - final if stopped, still-running if not."""
        if self.elapsed_time is not None:
            return self.elapsed_time
        if self.start_time is not None:
            return perf_counter() - self.start_time
        return 0.0


@dataclass
class ToolCallMetrics:
    """Timing for a single tool call, mirroring Agno's `ToolCallMetrics`
    (`metrics.py:115`) - unlike `MessageMetrics`, a tool call has no token
    counts of its own, only how long it took to run. Kept as its own type
    rather than reusing `MessageMetrics` with zeroed token fields: a tool
    call's `Timer` starts/stops independently of any model call, and
    `ToolExecution.metrics` (`models/response.py`) should say "this ran for
    N seconds" without implying "this consumed tokens."
    """

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None

    def __post_init__(self) -> None:
        self._timer = Timer()

    def start_timer(self) -> None:
        self._timer.start()
        if self.start_time is None:
            self.start_time = time()

    def stop_timer(self, set_duration: bool = True) -> None:
        self._timer.stop()
        if set_duration:
            self.duration = self._timer.elapsed
        if self.end_time is None:
            self.end_time = time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallMetrics":
        return cls(
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            duration=data.get("duration"),
        )


@dataclass
class BaseMetrics:
    """Raw token counts shared by every metrics type."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class MessageMetrics(BaseMetrics):
    """Usage + timing for a single model response message."""

    duration: Optional[float] = None
    time_to_first_token: Optional[float] = None

    def __add__(self, other: "MessageMetrics") -> "MessageMetrics":
        result = MessageMetrics(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )
        # Sum durations/time-to-first-token if both sides have one; otherwise
        # keep whichever side actually has a value.
        if self.duration is not None and other.duration is not None:
            result.duration = self.duration + other.duration
        else:
            result.duration = self.duration if self.duration is not None else other.duration

        if self.time_to_first_token is not None and other.time_to_first_token is not None:
            result.time_to_first_token = self.time_to_first_token + other.time_to_first_token
        else:
            result.time_to_first_token = (
                self.time_to_first_token if self.time_to_first_token is not None else other.time_to_first_token
            )
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Compact dict form - drops None and zero-valued fields."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) not in (None, 0)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageMetrics":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RunMetrics(MessageMetrics):
    """Usage + timing accumulated across an entire Agent.run() - i.e. every
    model call made across however many tool-call iterations it took."""

    def __post_init__(self) -> None:
        self._timer = Timer()

    def start_timer(self) -> None:
        self._timer.start()

    def stop_timer(self, set_duration: bool = True) -> None:
        self._timer.stop()
        if set_duration:
            self.duration = self._timer.elapsed

    def __add__(self, other: "MessageMetrics") -> "RunMetrics":  # type: ignore[override]
        """Fold one turn's `MessageMetrics` into this run's running total.

        Token counts sum. `duration` stays owned by this run's own timer
        (not summed from `other`) - the timer already measures the whole
        run's wall-clock time regardless of how many turns it took. The new
        result carries the *same* `Timer` instance forward (rather than
        starting a fresh one via `__post_init__`), so
        `run_metrics = run_metrics + turn_metrics` in a loop still reflects
        the original `start_timer()` call when `stop_timer()` is eventually
        called on it.
        """
        result = RunMetrics(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )
        result._timer = self._timer
        result.duration = self.duration
        result.time_to_first_token = (
            self.time_to_first_token if self.time_to_first_token is not None else other.time_to_first_token
        )
        return result
