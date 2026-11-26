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

Write ownership (5.0, skeptic S1). Every column has an owner: the module that declared it. A batch may write a
column only when the module running the context (`Ctx.module`: the endpoint's, the process's, the subscriber's) is
its owner, a module of the same package as the owner (the population package's modules share the population
table), or a module the owner granted it to (`grant`). Columns that belong to an approved interface
(`register_interface`, e.g. the attention ledger) are written only through that interface (`attention.spend/reset`),
by any module, or by the module that declared them (initialising the rows it creates). System code without a module (operator tools) may write any column. The writer is recorded in the
event (`by`), and the applier re-checks it in live execution, so emitting `cols.batch` by hand cannot bypass it.
Integer values are range-checked: a cast or an `add` that would wrap is refused, never committed.

Kernel module `cols` (no tools) owns the two appliers; a world that uses columns lists it (it depends on
`artifacts`).
"""
from __future__ import annotations

import sys

import numpy as np

from ..core import schema as S
from ..core.canonical import canonical_bytes, to_jsonable
from ..core.errors import Forbidden, InvalidAction, InvariantViolation, NotFound, ProtocolError
from ..core.kernel import Ctx, Module, applier, register_module
from . import artifacts, bus

MODES = ("set", "add", "max", "min")
#: column name -> interface that alone may write it (registered by crucible.substrate.attention)
_IFACES: dict[str, str] = {}
#: interface -> the Python module that implements it (the module that registered it). Wave 3b (kcal): an interface
#: token is a capability of its implementation, not a string any module code may pass - `batch(..., _iface=...)` is
#: refused unless called from that module, so the ownership rule cannot be bypassed by naming an interface.
_IFACE_IMPL: dict[str, str] = {}


def register_interface(name: str, cols) -> None:
    """Columns `cols` (by name, in any table) are written only through interface `name`, which the calling Python
    module implements (only its code may pass ``_iface=name`` to `batch`)."""
    impl = sys._getframe(1).f_globals.get("__name__", "")
    prev = _IFACE_IMPL.setdefault(name, impl)
    if prev != impl:
        raise ProtocolError(f"interface {name} is already implemented by {prev}")
    for c in cols:
        _IFACES[c] = name


def grant(world, table: str, cols, writer: str) -> None:
    """Let module `writer` write columns `cols` of `table` (call in the owner module's install)."""
    reg = _registry(world).get(table)
    if not reg:
        raise NotFound(f"no column table {table}")
    for c in cols:
        if c not in reg["columns"]:
            raise NotFound(f"no column {table}.{c}")
        reg.setdefault("grants", {}).setdefault(c, set()).add(writer)


def _package(name: str | None) -> str | None:
    from ..core.kernel import MODULES
    cls = MODULES.get(name) if name else None
    if cls is None:
        return None
    parts = cls.__module__.split(".")
    return ".".join(parts[:2]) if parts[0] == "crucible" and len(parts) > 1 else cls.__module__


def may_write(world, table: str, col: str, writer: str | None, iface: str | None = None) -> str | None:
    """None if `writer` may write `table.col` (through `iface`), else the reason it may not."""
    reg = _registry(world).get(table) or {}
    need = _IFACES.get(col)
    owner = reg.get("by", {}).get(col)
    if need is not None:
        # the interface, or the declaring module itself (it initialises the rows it creates); the interface's own
        # invariant (attention conservation) checks the result either way
        if iface == need or (writer is not None and writer == owner):
            return None
        return f"column {table}.{col} is written only through the {need} interface"
    if writer is None:
        return None                     # system code (operator tools, genesis of the kernel)
    if owner is None or writer == owner or writer in reg.get("grants", {}).get(col, ()):
        return None
    pw, po = _package(writer), _package(owner)
    if pw is not None and pw == po:
        return None
    return f"module {writer} may not write column {table}.{col} (owned by {owner})"


def _int_range_error(dt: np.dtype, v: np.ndarray) -> str | None:
    if v.size == 0 or np.dtype(dt).kind not in "iu":
        return None
    info = np.iinfo(dt)
    if v.dtype.kind == "f":
        if not np.all(np.isfinite(v)):
            return "not finite"
        lo, hi = float(v.min()), float(v.max())
    elif v.dtype.kind in "iu":
        lo, hi = int(v.min()), int(v.max())
    else:
        return None
    if lo < info.min or hi > info.max:
        return f"outside the range of {np.dtype(dt)} [{info.min}, {info.max}]"
    return None
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
          vis="internal", extra: dict | None = None, _iface: str | None = None) -> str:
    """Record one cohort update of `table` as a content-addressed blob plus one `cols.batch` event; returns the blob
    REF. `values[c]` has shape (len(idx), *trailing shape of c) and is cast to the column dtype. Refused (Forbidden)
    when the running module may not write a column (see the module docstring)."""
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
    writer = getattr(ctx, "module", None)
    if _iface is not None:
        impl = _IFACE_IMPL.get(_iface)
        caller = sys._getframe(1).f_globals.get("__name__", "")
        if impl is None or caller != impl:
            raise Forbidden(f"interface {_iface} is implemented by {impl or 'nobody'}; {caller} may not write "
                            "through it")
    for c in names:
        spec = reg["columns"].get(c)
        if spec is None:
            raise NotFound(f"no column {table}.{c}")
        why = may_write(ctx.world, table, c, writer, _iface)
        if why:
            raise Forbidden(why)
        dt, shape, _ = spec
        v = np.asarray(values[c])
        if v.dtype.kind not in "iufb":
            raise InvalidAction(f"values of {c} are not numeric")
        if v.shape != (len(idx), *shape):
            if v.ndim == 0 and not shape:
                v = np.full(len(idx), v)
            else:
                raise InvalidAction(f"values of {c} have shape {v.shape}, need {(len(idx), *shape)}")
        bad = _int_range_error(np.dtype(dt), v)
        if bad:
            raise InvalidAction(f"values of {table}.{c} are {bad}")
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
    payload = {"table": table, "kind": kind, "ref": ref, "n": int(ix.size), "cols": names, "mode": mode,
               "extra": extra}
    if writer is not None:
        payload["by"] = writer
    if _iface is not None:
        payload["iface"] = _iface
    ctx.emit("cols.batch", payload, vis)
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
    if mode == "add" and a.dtype.kind in "iu":
        # integer sums must not wrap: checked in exact (int64) or float64 arithmetic before anything is written
        rows_all = np.unique(ix)
        acc = np.zeros((rows_all.size, *a.shape[1:]), dtype=np.float64 if a.dtype.itemsize >= 8 else np.int64)
        np.add.at(acc, np.searchsorted(rows_all, ix), v.astype(acc.dtype))
        tot = a[rows_all].astype(acc.dtype) + acc
        bad = _int_range_error(a.dtype, tot)
        if bad:
            raise InvalidAction(f"column {key}: the sum would be {bad}")
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
    "mode": {"type": "string", "enum": list(MODES)}, "extra": {"type": "object", "additionalProperties": True},
    "by": {"type": "string", "pattern": r"[a-z][a-z0-9_]{0,63}"}, "iface": {"type": "string", "pattern": NAME_RE}},
    ["table", "kind", "ref", "n", "cols", "mode", "extra"]), owner="cols",
    doc="one cohort update of a column table; the row indices and values are in the array blob `ref`; `by` is the "
        "module that wrote it, `iface` the approved interface it went through")
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
        live = getattr(w, "emitting", None)
        if live is not None:
            writer = getattr(live, "module", None)
            if p.get("by") != writer and not (writer is None and "by" not in p):
                raise Forbidden(f"cols.batch claims writer {p.get('by')!r} but {writer!r} emitted it")
            for c in p["cols"]:
                why = may_write(w, p["table"], c, writer, p.get("iface"))
                if why:
                    raise Forbidden(why)
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


from . import attention as _attention  # noqa: E402,F401  (registers the attention interface)
