"""Attention and time as a finite, recorded resource (Crucible 5.0, contract 1.6).

Each individual has a daily attention budget in minutes. Columns of the population table (`pop`, float32 unless
noted), declared with `declare(world, n)` or by including `COLUMNS` in the population's own `columns.declare`:

    attention_budget_min   minutes available per day
    attention_left_min     minutes left today
    attention_spent_min    minutes spent since the last reset (the ledger the invariant checks)
    attention_day   int32  simulated day of the last reset (-1 before the first)

`spend(ctx, idx, minutes, activity=...)` grants each request at most what is left (requests of the same individual
in one call are served in order) and records the whole cohort as ONE `cols.batch` event of kind
`attention.<activity>`. `reset(ctx, idx=None)` is the daily reset the population runs (`attention.reset`).

Invariant (`check`, also run by `Cols.check_invariants`, so `InvariantMonitor` reports it as `module.cols`):
left >= 0, spent >= 0, spent <= budget and left <= budget (with float32 rounding tolerance), and no reset dated in the
future. `violations(world)` returns the same findings as `crucible.evaluation.invariants.Violation` objects.
"""
from __future__ import annotations

import re

import numpy as np

from ..core.errors import InvalidAction, NotFound
from ..core.kernel import Ctx
from . import columns

TABLE = "pop"
COLUMNS = {
    "attention_budget_min": ("float32", (), 0.0),
    "attention_left_min": ("float32", (), 0.0),
    "attention_spent_min": ("float32", (), 0.0),
    "attention_day": ("int32", (), -1),
}
DAY = 86400
REL_TOL = 1e-5      # float32 accumulation over a day of many small grants
ABS_TOL = 1e-3      # minutes


def declare(world, n: int, table: str = TABLE) -> None:
    """Declare the attention columns of `table` (in Module.install)."""
    columns.declare(world, table, n, COLUMNS)


def _cols(world, table: str):
    try:
        return (columns.col(world, table, "attention_budget_min"), columns.col(world, table, "attention_left_min"),
                columns.col(world, table, "attention_spent_min"), columns.col(world, table, "attention_day"))
    except NotFound:
        raise NotFound(f"table {table} has no attention columns (attention.declare)") from None


def granted_minutes(left: np.ndarray, idx: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    """What `spend` grants for each request (in request order): requests of one individual are served in order,
    each getting min(request, what is left after the earlier ones). Pure; float64 arithmetic."""
    ix = np.asarray(idx, dtype=np.int64)
    m = np.asarray(minutes, dtype=np.float64)
    L = left[ix].astype(np.float64)
    if ix.size <= 1 or np.unique(ix).size == ix.size:
        return np.minimum(m, L)
    order = np.argsort(ix, kind="stable")
    ixs, ms, Ls = ix[order], m[order], L[order]
    first = np.ones(ixs.size, dtype=bool)
    first[1:] = ixs[1:] != ixs[:-1]
    group = np.cumsum(first) - 1
    cs = np.cumsum(ms)
    start = cs[first] - ms[first]                   # cumulative sum before each group
    c_incl = cs - start[group]                      # cumulative request within the group, inclusive
    c_excl = c_incl - ms
    g_sorted = np.minimum(Ls, c_incl) - np.minimum(Ls, c_excl)
    g = np.empty_like(g_sorted)
    g[order] = g_sorted
    return g


def spend(ctx: Ctx, idx: np.ndarray, minutes: np.ndarray, *, activity: str, table: str = TABLE) -> np.ndarray:
    """Spend attention for a cohort; returns the granted minutes per request (float32, <= what was left). One
    `cols.batch` event of kind `attention.<activity>` records it."""
    if not isinstance(activity, str) or not re.fullmatch(r"[a-z][a-z0-9_.:-]{0,52}", activity):
        raise InvalidAction(f"bad activity name {activity!r}")
    ix = np.asarray(idx)
    m = np.asarray(minutes, dtype=np.float64)
    if ix.ndim != 1 or m.shape != ix.shape:
        raise InvalidAction("idx and minutes must be one-dimensional and of equal length")
    if m.size and (not np.all(np.isfinite(m)) or float(m.min()) < 0):
        raise InvalidAction("minutes must be finite and >= 0")
    budget, left, spent, day = _cols(ctx.world, table)
    n = left.shape[0]
    ix = columns._norm_idx(ix, n)
    g = granted_minutes(left, ix, m)
    if ix.size == 0:
        return g.astype(np.float32)
    # one aggregated grant per individual: min(left, total request) is exactly representable against `left`, so the
    # float32 subtraction can never take `left` below zero
    u, inv = np.unique(ix, return_inverse=True)
    req_tot = np.zeros(u.size, dtype=np.float64)
    np.add.at(req_tot, inv, m)
    g_tot = np.minimum(left[u].astype(np.float64), req_tot).astype(np.float32)
    columns.batch(ctx, table, f"attention.{activity}", u, {"attention_left_min": -g_tot, "attention_spent_min": g_tot},
                  mode="add", extra={"activity": activity, "requests": int(ix.size),
                                     "requested_min": float(m.sum()), "granted_min": float(g_tot.astype(np.float64).sum())})
    return g.astype(np.float32)


def reset(ctx: Ctx, idx: np.ndarray | None = None, *, budget: np.ndarray | float | None = None,
          table: str = TABLE) -> str:
    """Daily reset (run by the population): left = budget, spent = 0, day = today, for `idx` (default: everyone).
    With `budget`, the budget is set first. Returns the batch REF."""
    b, left, spent, day = _cols(ctx.world, table)
    n = left.shape[0]
    ix = np.arange(n, dtype=np.int64) if idx is None else columns._norm_idx(idx, n)
    if budget is None:
        bud = b[ix].astype(np.float32)
    else:
        bud = np.broadcast_to(np.asarray(budget, dtype=np.float32), ix.shape).copy()
        if bud.size and (not np.all(np.isfinite(bud)) or float(bud.min()) < 0):
            raise InvalidAction("budget must be finite and >= 0")
    today = ctx.now // DAY
    vals = {"attention_left_min": bud, "attention_spent_min": np.zeros(ix.size, np.float32),
            "attention_day": np.full(ix.size, today, np.int32)}
    if budget is not None:
        vals["attention_budget_min"] = bud
    return columns.batch(ctx, table, "attention.reset", ix, vals, mode="set", extra={"day": int(today)})


def check(world, table: str = TABLE) -> list[dict]:
    """Attention law violations as InvariantMonitor-compatible records {kind, where, seq, t, amounts, detail}."""
    try:
        budget, left, spent, day = _cols(world, table)
    except NotFound:
        return []
    out = []
    tol = budget.astype(np.float64) * REL_TOL + ABS_TOL

    def v(kind, mask, detail, **amounts):
        bad = np.flatnonzero(mask)
        if bad.size:
            out.append({"kind": kind, "where": f"{table}[{int(bad[0])}]", "seq": int(world.seq), "t": int(world.now),
                        "amounts": {"rows": int(bad.size), **{k: float(a[bad[0]]) for k, a in amounts.items()}},
                        "detail": detail})
    v("attention.negative_left", left < 0, "attention left below zero", left=left)
    v("attention.negative_spent", spent < 0, "negative attention spent", spent=spent)
    v("attention.overspent", spent.astype(np.float64) > budget.astype(np.float64) + tol,
      "more attention spent since the last reset than the daily budget", spent=spent, budget=budget)
    v("attention.left_above_budget", left.astype(np.float64) > budget.astype(np.float64) + tol,
      "more attention left than the daily budget", left=left, budget=budget)
    v("attention.future_reset", day > world.now // DAY, "attention reset dated in the future", day=day)
    return out


def violations(world, table: str = TABLE) -> list:
    """`check` as crucible.evaluation.invariants.Violation objects (evaluator side)."""
    from ..evaluation.invariants import Violation
    return [Violation(**d) for d in check(world, table)]
