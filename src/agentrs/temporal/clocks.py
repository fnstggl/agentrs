"""Three clocks, kept separate everywhere a time is stored.

* environment clock — when an event actually happened in the environment
  (``env_event_time``; known exactly only when the environment reports it);
* observation clock — when our sensors collected information (``observed_time``);
* training clock — when the learner used the information (``ingested_wall``; always real time).

Environment and observation times use the *time base* of their source: ``"real"`` (wall clock,
UTC epoch seconds) or ``"virtual:<id>"`` (a deterministic accelerated clock of a controlled
environment we run ourselves). Virtual elapsed time is never presented as real elapsed time:
records carry their time base and the ledger refuses to compare times across bases.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol


class Clock(Protocol):
    time_base: str

    def now(self) -> float: ...


class RealClock:
    time_base = "real"

    def now(self) -> float:
        return time.time()


class VirtualClock:
    """Deterministic clock driven explicitly by the test runner / environment."""

    def __init__(self, start: float, clock_id: str = "lab"):
        self._t = float(start)
        self._lock = threading.Lock()
        self.time_base = f"virtual:{clock_id}"

    def now(self) -> float:
        with self._lock:
            return self._t

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("virtual time cannot go backwards")
        with self._lock:
            self._t += seconds
            return self._t

    def set(self, t: float) -> None:
        with self._lock:
            if t < self._t:
                raise ValueError("virtual time cannot go backwards")
            self._t = float(t)


class ExternalClock:
    """Reads the time of an environment that owns its own clock (e.g. a SocialWebLab server)."""

    def __init__(self, fn, time_base: str):
        self._fn = fn
        self.time_base = time_base

    def now(self) -> float:
        return float(self._fn())


def is_virtual(time_base: str) -> bool:
    return time_base.startswith("virtual:")
