"""Real-world results are kept apart from synthetic results by construction.

* `RealWorldResult` is deliberately not a dict, not JSON-serialisable, not picklable and not copyable. The kernel's
  encoders (`core.canonical.to_jsonable`, `canonical_bytes`) raise on it, so it cannot become an event payload, a
  tool argument, a snapshot or a row of any store; `json.dump` and `pickle.dumps` raise as well.
* The only way out is `export_results(path, results)`, which writes a `*.realworld.json` file whose header and
  every row carry `"origin": "real-world"`.
* `assert_no_real_world(obj)` is the check a synthetic results writer runs on what it is about to write: it
  raises `ContaminationError` if a `RealWorldResult`, or any exported row carrying the origin tag, is inside.
* `SealedHoldoutGuard` counts evaluations per sealed holdout and refuses beyond the stated budget; the count is
  kept in a hash-chained ledger so that restarting the process does not reset it.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Iterable

from .log import ChainedLog

ORIGIN = "real-world"
SUFFIX = ".realworld.json"


class ContaminationError(Exception):
    """Real-world data was about to be mixed into synthetic results (or the other way round)."""


class HoldoutBudgetExceeded(PermissionError):
    pass


class RealWorldResult:
    """One measured outcome of a real-world experiment. Created only by `RealWorldBridge.measure` /
    `evaluate_holdout`; immutable; refuses every generic serialiser."""

    __slots__ = ("_d",)

    def __init__(self, _token: object, **fields):
        if _token is not _MINT:
            raise TypeError("RealWorldResult objects are created by the bridge's measurement functions only")
        object.__setattr__(self, "_d", dict(fields, origin=ORIGIN))

    def __getattr__(self, k):
        try:
            return object.__getattribute__(self, "_d")[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        raise AttributeError("real-world results are immutable")

    def __repr__(self) -> str:
        d = object.__getattribute__(self, "_d")
        return f"<RealWorldResult {d.get('experiment')}/{d.get('measure')} = {d.get('value')!r}>"

    def _refuse(self, *a, **k):
        raise ContaminationError("a real-world result cannot be serialised, copied or stored by generic means; "
                                 "use crucible.realworld.export_results")

    __reduce__ = __reduce_ex__ = __getstate__ = __copy__ = __deepcopy__ = _refuse   # pickle / copy / deepcopy


_MINT = object()


def _mint(**fields) -> RealWorldResult:
    return RealWorldResult(_MINT, **fields)


def is_real_world(x: Any) -> bool:
    if isinstance(x, RealWorldResult):
        return True
    if isinstance(x, dict):
        return x.get("origin") == ORIGIN or any(is_real_world(v) for v in x.values())
    if isinstance(x, (list, tuple, set, frozenset)):
        return any(is_real_world(v) for v in x)
    return False


def assert_no_real_world(obj: Any, where: str = "synthetic results") -> None:
    if is_real_world(obj):
        raise ContaminationError(f"real-world data must not be written into {where}")


def export_results(path: str, results: Iterable[RealWorldResult]) -> str:
    """Write results to a dedicated `*.realworld.json` file (never into a synthetic results file or store)."""
    results = list(results)
    if not str(path).endswith(SUFFIX):
        raise ContaminationError(f"real-world results are only written to files named *{SUFFIX}")
    if not all(isinstance(r, RealWorldResult) for r in results):
        raise ContaminationError("only RealWorldResult objects can be exported here (no synthetic rows)")
    if os.path.exists(path):
        try:
            with open(path) as f:
                if json.load(f).get("origin") != ORIGIN:
                    raise ContaminationError(f"{path} exists and is not a real-world results file")
        except (ValueError, AttributeError):
            raise ContaminationError(f"{path} exists and is not a real-world results file")
    rows = [dict(object.__getattribute__(r, "_d")) for r in results]
    doc = {"origin": ORIGIN, "schema": "crucible-realworld-results/1",
           "note": "Measured against real systems under an operator authorisation. Not comparable with, and never "
                   "to be merged into, synthetic Crucible results.", "written": time.time(), "results": rows}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


class SealedHoldoutGuard:
    """Counts evaluations against sealed holdouts. `budgets` maps a holdout name to the number of evaluations that
    may ever be run against it; the count lives in `ledger_path` (hash-chained, verified on open)."""

    def __init__(self, budgets: dict[str, int], ledger_path: str | None = None, log: ChainedLog | None = None):
        if (ledger_path is None) == (log is None):
            raise ValueError("give exactly one of ledger_path or log")
        self.budgets = {str(k): int(v) for k, v in budgets.items()}
        self.log = log if log is not None else ChainedLog(ledger_path)

    def used(self, holdout: str) -> int:
        return self.log.count("holdout", holdout=holdout)

    def remaining(self, holdout: str) -> int:
        if holdout not in self.budgets:
            return 0
        return max(0, self.budgets[holdout] - self.used(holdout))

    def charge(self, holdout: str, purpose: str = "") -> int:
        """Record one evaluation (before it runs: a crashed evaluation still counts). Returns evaluations left."""
        if holdout not in self.budgets:
            self.log.append({"kind": "refusal", "why": f"holdout {holdout!r} is not in the stated budget", "t": time.time()})
            raise HoldoutBudgetExceeded(f"holdout {holdout!r} has no stated budget")
        used = self.used(holdout)
        if used >= self.budgets[holdout]:
            self.log.append({"kind": "refusal", "why": f"holdout {holdout!r} budget {self.budgets[holdout]} exhausted",
                             "t": time.time()})
            raise HoldoutBudgetExceeded(f"holdout {holdout!r}: budget of {self.budgets[holdout]} evaluation(s) is used up")
        self.log.append({"kind": "holdout", "holdout": holdout, "n": used + 1, "budget": self.budgets[holdout],
                         "purpose": purpose[:300], "t": time.time()})
        return self.budgets[holdout] - used - 1

    def evaluate(self, holdout: str, fn: Callable[[], Any], purpose: str = "") -> Any:
        self.charge(holdout, purpose)
        return fn()
