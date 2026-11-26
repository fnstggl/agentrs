"""Columnar state with batch events (Crucible 5.0, DECISION_LOG D-3).

Population-scale state lives in the kernel's dense arrays (`StateStore.arrays`, part of every snapshot and state
digest), one array per column, named `"<table>.<column>"`; row `i` of every column of a table is individual `i`.

* `declare(world, table, n, columns)` (in `Module.install`) creates the columns with their dtype, trailing shape
  and fill value. Like `declare_table`, this is a deterministic function of the world spec, so creation, replay and
  regeneration start from the same arrays; a snapshot load replaces them with the snapshot's.
* `batch(ctx, table, kind, idx, values, mode=...)` is how a cohort decision changes columns. The handler writes the
  row indices and new values to a content-addressed array blob (`artifacts.put_array_blob`) and emits ONE
  `cols.batch {table, kind, ref, n, cols, mode, extra}` event. The `cols.batch` applier loads the blob, checks its
  SHA-256 (the REF is inside the hash-chained payload, so the chain covers every individual decision), and applies
  it: `set` (last write wins for repeated rows), `add` (repeated rows accumulate), `max`, `min`.
* `grow(ctx, table, n_new)` appends rows with the fill values (`cols.grow`).

Writes inside a transaction journal only the touched rows (`StateStore.array_rows_mut`), so a cohort update of a
10M-row column costs O(cohort), and a failed command rolls the columns back exactly.

Kernel module `cols` (no tools) owns the two appliers; a world that uses columns lists it (it depends on
`artifacts`).
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.canonical import canonical_bytes, to_jsonable
from ..core.errors import InvalidAction, InvariantViolation, NotFound, ProtocolError
from ..core.kernel import Ctx, Module, applier, register_module
from . import artifacts, bus

MODES = ("set", "add", "max", "min")
NAME_RE = r"[a-z][a-z0-9_]{0,47}"
_INT_KINDS = "iub"


def _registry(world) -> dict:
    return world.runtime.setdefault("cols", {})


def declare(world, table: str, n: int, columns: dict[str, tuple]) -> None:
    """Declare columns of `table` with `n` rows: {name: (dtype, trailing_shape, fill)}. Call in Module.install.
    Declaring more columns of an existing table is allowed (same row count); redeclaring a column differently is not."""
    import re
    if not re.fullmatch(NAME_RE, table):
        raise ProtocolError(f"bad column table name {table!r}")
    n = int(n)
    if n < 0:
        raise ProtocolError("row count must be >= 0")
    reg = _registry(world)
    t = reg.setdefault(table, {"columns": {}})
    st = world.state
    who = st.declaring or "kernel"     # runtime attribution (crucible.substrate.primitives scans it)
    by = t.setdefault("by", {})
    for name, spec in columns.items():
        by.setdefault(name, who)
        if not re.fullmatch(NAME_RE, name):
            raise ProtocolError(f"bad column name {name!r}")
        dtype, shape, fill = spec
        dt = np.dtype(dtype)
        if dt.hasobject or dt.kind not in "iufb":
            raise ProtocolError(f"column {table}.{name}: dtype {dt} is not numeric")
        shape = tuple(int(x) for x in shape)
        key = f"{table}.{name}"
        norm = (dt.str, shape, fill)
        old = t["columns"].get(name)
        if old is not None:
            if (np.dtype(old[0]).str, tuple(old[1]), old[2]) != norm:
                raise ProtocolError(f"column {key} already declared as {old}")
            continue
        existing = [c for c in t["columns"]]
        if existing:
            n0 = st.arrays[f"{table}.{existing[0]}"].shape[0]
            if n0 != n:
                raise ProtocolError(f"table {table} has {n0} rows, not {n}")
        t["columns"][name] = (dt.str, shape, fill)
        if key not in st.arrays:
            st.set_array(key, np.full((n, *shape), fill, dtype=dt))


def tables(world) -> dict[str, dict]:
    """{table: {n, columns: {name: (dtype, shape, fill)}}} declared in this world."""
    out = {}
    for t, rec in sorted(_registry(world).items()):
        out[t] = {"n": rows(world, t), "columns": dict(rec["columns"])}
    return out


def rows(world, table: str) -> int:
    t = _registry(world).get(table)
    if not t or not t["columns"]:
        raise NotFound(f"no column table {table}")
    first = next(iter(t["columns"]))
    return int(world.state.arrays[f"{table}.{first}"].shape[0])


def col(world, table: str, name: str) -> np.ndarray:
    """Read-only view of one column."""
    key = f"{table}.{name}"
    if key not in world.state.arrays or name not in _registry(world).get(table, {}).get("columns", {}):
        raise NotFound(f"no column {key}")
    return world.state.array(key)


def _norm_idx(idx, n: int) -> np.ndarray:
    a = np.asarray(idx)
    if a.ndim != 1:
        raise InvalidAction("idx must be one-dimensional")
    if a.size and a.dtype.kind not in "iu":
        raise InvalidAction(f"idx must be integers, not {a.dtype}")
    a = a.astype(np.int64, copy=False)
    if a.size and (int(a.min()) < 0 or int(a.max()) >= n):
        raise InvalidAction(f"row index out of range [0, {n})")
    return a


def batch(ctx: Ctx, table: str, kind: str, idx: np.ndarray, values: dict[str, np.ndarray], *, mode: str = "set",
          vis="internal", extra: dict | None = None) -> str:
    """Record one cohort update of `table` as a content-addressed blob plus one `cols.batch` event; returns the blob
    REF. `values[c]` has shape (len(idx), *trailing shape of c) and is cast to the column dtype."""
    if mode not in MODES:
        raise InvalidAction(f"mode must be one of {MODES}")
    import re
    if not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_.:-]{0,63}", kind):
        raise InvalidAction(f"bad batch kind {kind!r}")
    reg = _registry(ctx.world).get(table)
    if not reg:
        raise NotFound(f"no column table {table}")
    if not values:
        raise InvalidAction("a batch needs at least one column")
    n = rows(ctx.world, table)
    ix = _norm_idx(idx, n)
    order = None
    if ix.size > 1 and np.any(ix[1:] < ix[:-1]):
        order = np.argsort(ix, kind="stable")      # stable: repeated rows keep their order (last write wins)
        ix = ix[order]
    arrays = {"idx": ix.astype(np.int32) if n < 2**31 else ix}
    names = sorted(values)
    for c in names:
        spec = reg["columns"].get(c)
        if spec is None:
            raise NotFound(f"no column {table}.{c}")
        dt, shape, _ = spec
        v = np.asarray(values[c])
        if v.dtype.kind not in "iufb":
            raise InvalidAction(f"values of {c} are not numeric")
        if v.shape != (len(idx), *shape):
            if v.ndim == 0 and not shape:
                v = np.full(len(idx), v)
            else:
                raise InvalidAction(f"values of {c} have shape {v.shape}, need {(len(idx), *shape)}")
        with np.errstate(over="ignore", invalid="ignore"):     # out-of-range values are refused just below
            v = v.astype(np.dtype(dt), copy=False)
        if v.dtype.kind == "f" and not np.all(np.isfinite(v)):
            raise InvalidAction(f"values of {c} are not finite in {np.dtype(dt)} (NaN, infinite or out of range)")
        if mode != "set" and np.dtype(dt).kind == "b":
            raise InvalidAction(f"mode {mode} needs a numeric column, {c} is boolean")
        arrays[f"v.{c}"] = v[order] if order is not None else v
    try:
        extra = to_jsonable(extra or {})       # a private copy: the caller may reuse its dict
        canonical_bytes(extra)
    except (TypeError, ValueError):
        raise InvalidAction("extra must be JSON-native (finite numbers)") from None
    if not isinstance(extra, dict):
        raise InvalidAction("extra must be an object")
    ref = artifacts.put_array_blob(ctx, arrays)
    ctx.emit("cols.batch", {"table": table, "kind": kind, "ref": ref, "n": int(ix.size), "cols": names, "mode": mode,
                            "extra": extra}, vis)
    return ref


def grow(ctx: Ctx, table: str, n_new: int) -> None:
    """Append `n_new` rows with the declared fill values to every column of `table` (event `cols.grow`)."""
    n_new = int(n_new)
    if n_new <= 0:
        raise InvalidAction("n_new must be positive")
    n0 = rows(ctx.world, table)
    ctx.emit("cols.grow", {"table": table, "n_new": n_new, "n_before": n0}, vis="internal")


def _apply_rows(st, key: str, ix: np.ndarray, v: np.ndarray, mode: str) -> None:
    a = st.arrays[key]
    k = ix.size
    if k == 0:
        return
    dup = bool(k > 1 and np.any(ix[1:] == ix[:-1]))
    if dup:
        last = np.ones(k, dtype=bool)
        last[:-1] = ix[1:] != ix[:-1]           # last occurrence of every row (idx is sorted, stable)
        first = np.ones(k, dtype=bool)
        first[1:] = ix[1:] != ix[:-1]
        rows_u = ix[last]
    else:
        rows_u = ix
    # journal only the touched rows unless the update covers most of the column (then one copy is cheaper)
    row_bytes = a.nbytes // max(1, a.shape[0])
    if rows_u.size * (row_bytes + 8) * 2 < a.nbytes:
        a = st.array_rows_mut(key, rows_u)
    else:
        a = st.array_mut(key)
    with np.errstate(over="ignore", invalid="ignore"):      # an overflowing sum is refused just below
        _write_rows(a, ix, rows_u, v, mode, dup, last if dup else None, first if dup else None)
    if mode == "add" and a.dtype.kind == "f" and not np.all(np.isfinite(a[rows_u])):
        # the command is refused (and rolled back): an overflowing float column is never committed
        raise InvalidAction(f"column {key} overflows {a.dtype}")


def _write_rows(a, ix, rows_u, v, mode, dup, last, first) -> None:
    if mode == "set":
        a[rows_u] = v[last] if dup else v
    elif mode == "add":
        if dup:
            starts = np.flatnonzero(first)
            a[rows_u] += np.add.reduceat(v, starts, axis=0).astype(a.dtype, copy=False)
        else:
            a[rows_u] += v
    elif mode == "max":
        if dup:
            np.maximum.at(a, ix, v)
        else:
            a[rows_u] = np.maximum(a[rows_u], v)
    elif mode == "min":
        if dup:
            np.minimum.at(a, ix, v)
        else:
            a[rows_u] = np.minimum(a[rows_u], v)


bus.event_type("cols.batch", S.obj({
    "table": {"type": "string", "pattern": NAME_RE}, "kind": {"type": "string", "pattern": r"[a-z][a-z0-9_.:-]{0,63}"},
    "ref": {"type": "string", "pattern": artifacts.REF_RE.pattern}, "n": {"type": "integer", "minimum": 0},
    "cols": {"type": "array", "items": {"type": "string", "pattern": NAME_RE}, "minItems": 1, "maxItems": 256},
    "mode": {"type": "string", "enum": list(MODES)}, "extra": {"type": "object", "additionalProperties": True}},
    ["table", "kind", "ref", "n", "cols", "mode", "extra"]), owner="cols",
    doc="one cohort update of a column table; the row indices and values are in the array blob `ref`")
bus.event_type("cols.grow", S.obj({"table": {"type": "string", "pattern": NAME_RE},
                                   "n_new": {"type": "integer", "minimum": 1},
                                   "n_before": {"type": "integer", "minimum": 0}},
                                  ["table", "n_new", "n_before"]), owner="cols",
               doc="rows appended to every column of a table, with the declared fill values")


@register_module
class Cols(Module):
    """Owner of the column appliers (`cols.batch`, `cols.grow`). No tools: columns change only through recorded
    cohort decisions of the modules that own the tables."""
    name = "cols"
    host = "cols"
    version = "1"
    depends = ("artifacts",)

    def install(self, world):
        bus.enforce(world, owner="cols")

    @applier("cols.batch")
    def ap_batch(self, ev):
        p = ev.payload
        w = self.world
        reg = _registry(w).get(p["table"])
        if not reg:
            raise InvariantViolation(f"cols.batch for undeclared table {p['table']}")
        try:
            arrs = artifacts.get_array_blob(w, p["ref"])
        except NotFound as e:
            raise InvariantViolation(f"cols.batch seq {ev.seq}: {e}") from None
        ix = arrs["idx"].astype(np.int64, copy=False)
        if ix.size != p["n"]:
            raise InvariantViolation(f"cols.batch seq {ev.seq}: blob has {ix.size} rows, event says {p['n']}")
        if ix.size > 1 and np.any(ix[1:] < ix[:-1]):
            raise InvariantViolation(f"cols.batch seq {ev.seq}: row indices not sorted")
        n = rows(w, p["table"])
        if ix.size and (int(ix[0]) < 0 or int(ix[-1]) >= n):
            raise InvariantViolation(f"cols.batch seq {ev.seq}: row index out of range")
        st = w.state
        for c in p["cols"]:
            v = arrs.get(f"v.{c}")
            spec = reg["columns"].get(c)
            if v is None or spec is None:
                raise InvariantViolation(f"cols.batch seq {ev.seq}: column {c} missing")
            if v.dtype.str != np.dtype(spec[0]).str or v.shape != (ix.size, *spec[1]):
                raise InvariantViolation(f"cols.batch seq {ev.seq}: column {c} has dtype/shape {v.dtype}{v.shape}")
            _apply_rows(st, f"{p['table']}.{c}", ix, v, p["mode"])

    @applier("cols.grow")
    def ap_grow(self, ev):
        p = ev.payload
        w = self.world
        reg = _registry(w).get(p["table"])
        if not reg:
            raise InvariantViolation(f"cols.grow for undeclared table {p['table']}")
        st = w.state
        for c, (dt, shape, fill) in reg["columns"].items():
            key = f"{p['table']}.{c}"
            a = st.arrays[key]
            if a.shape[0] != p["n_before"]:
                raise InvariantViolation(f"cols.grow seq {ev.seq}: {key} has {a.shape[0]} rows, expected "
                                         f"{p['n_before']}")
            st.set_array(key, np.concatenate([a, np.full((p["n_new"], *shape), fill, dtype=np.dtype(dt))]))

    def check_invariants(self, world):
        errs = []
        for t, rec in _registry(world).items():
            ns = {world.state.arrays[f"{t}.{c}"].shape[0] for c in rec["columns"] if f"{t}.{c}" in world.state.arrays}
            if len(ns) > 1:
                errs.append(f"columns of {t} have different row counts {sorted(ns)}")
            for c, (dt, shape, _) in rec["columns"].items():
                a = world.state.arrays.get(f"{t}.{c}")
                if a is None:
                    errs.append(f"column {t}.{c} missing")
                elif a.dtype.str != np.dtype(dt).str or tuple(a.shape[1:]) != tuple(shape):
                    errs.append(f"column {t}.{c} has dtype/shape {a.dtype}{a.shape}")
                elif a.dtype.kind == "f" and not np.all(np.isfinite(a)):
                    errs.append(f"column {t}.{c} has non-finite values")
        from . import attention
        errs += [f"{v['kind']} at {v['where']}: {v['detail']} {v['amounts']}" for v in attention.check(world)]
        return errs
