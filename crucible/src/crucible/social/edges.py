"""Growable row tables on the substrate's columnar state, for sparse population-scale relations.

A relation (follows, shares, community memberships, conversation participants) is a column table whose rows are
relation records: e.g. table `follow` with columns src, dst, t, active, via. Rows are appended in place; the table
keeps spare capacity and doubles it with `cols.grow` when it fills, so appending k rows costs O(k) amortised.
Every change is a recorded cohort decision: `columns.batch` (one `cols.batch` event, the rows in a
content-addressed blob) plus, for appends, one `social.alloc` event that advances the table's row counter. Nothing
writes these columns any other way.

`EdgeTable(name, columns)` is the schema; `declare(world, cap)` in Module.install; `append(ctx, kind, values)`
returns the new row indices; `update(ctx, kind, rows, values)`; `view(world)` returns read-only column views of
the used rows.
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.errors import InvalidAction
from ..core.kernel import Ctx, Module, applier, register_module
from ..substrate import bus, columns


class EdgeTable:
    def __init__(self, name: str, cols: dict[str, tuple[str, tuple, float | int | bool]]):
        self.name = name
        self.cols = dict(cols)

    def declare(self, world, capacity: int = 1024) -> None:
        columns.declare(world, self.name, max(16, int(capacity)), self.cols)

    def used(self, world) -> int:
        rec = world.state.tables.get("social_alloc", {}).get(self.name)
        return 0 if rec is None else int(rec["used"])

    def capacity(self, world) -> int:
        return columns.rows(world, self.name)

    def view(self, world) -> dict[str, np.ndarray]:
        n = self.used(world)
        return {c: columns.col(world, self.name, c)[:n] for c in self.cols}

    def col(self, world, c: str) -> np.ndarray:
        return columns.col(world, self.name, c)[:self.used(world)]

    def append(self, ctx: Ctx, kind: str, values: dict[str, np.ndarray], *, extra: dict | None = None) -> np.ndarray:
        k = None
        for v in values.values():
            k = len(v)
            break
        if not k:
            return np.zeros(0, dtype=np.int64)
        n0 = self.used(ctx.world)
        cap = self.capacity(ctx.world)
        if n0 + k > cap:
            columns.grow(ctx, self.name, max(cap, n0 + k - cap))
        rows = np.arange(n0, n0 + k, dtype=np.int64)
        columns.batch(ctx, self.name, kind, rows, values, mode="set", extra=extra)
        ctx.emit("social.alloc", {"table": self.name, "used": n0 + k}, vis="internal")
        return rows

    def update(self, ctx: Ctx, kind: str, rows: np.ndarray, values: dict[str, np.ndarray], *, mode: str = "set",
               extra: dict | None = None) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if not rows.size:
            return
        if int(rows.max()) >= self.used(ctx.world) or int(rows.min()) < 0:
            raise InvalidAction(f"{self.name}: row out of range")
        columns.batch(ctx, self.name, kind, rows, values, mode=mode, extra=extra)


bus.event_type("social.alloc", S.obj({"table": {"type": "string"}, "used": {"type": "integer", "minimum": 0}},
                                     ["table", "used"]), owner="social_rows",
               doc="the row counter of a social relation table advanced (rows were appended by a cols.batch)")


@register_module
class SocialRows(Module):
    """Owner of the row counters of the social relation tables (`social_alloc`). No tools."""
    name = "social_rows"
    host = "social_rows"
    version = "1"
    depends = ("cols",)

    def install(self, world):
        world.state.declare_table("social_alloc")
        bus.enforce(world, owner="social_rows")

    @applier("social.alloc")
    def ap_alloc(self, ev):
        p = ev.payload
        st = self.world.state
        old = st.tables["social_alloc"].get(p["table"])
        if old is not None and p["used"] < old["used"]:
            raise InvalidAction(f"{p['table']}: row counter cannot go back")
        cap = columns.rows(self.world, p["table"])
        if p["used"] > cap:
            raise InvalidAction(f"{p['table']}: {p['used']} rows used of capacity {cap}")
        st.put("social_alloc", p["table"], {"used": p["used"]})

    def check_invariants(self, world):
        errs = []
        for t, rec in world.state.tables["social_alloc"].items():
            try:
                cap = columns.rows(world, t)
            except Exception:  # noqa: BLE001
                errs.append(f"row counter of unknown table {t}")
                continue
            if rec["used"] > cap:
                errs.append(f"{t}: {rec['used']} rows used of capacity {cap}")
        return errs


def pair_keys(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact uint64 keys of node pairs (population nodes < 2**32, other nodes 2**40 + k with k < 2**31)."""
    from .nodes import OFFSET
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    def compact(x):
        big = x >= OFFSET
        if np.any(x[~big] >= (1 << 31)) or np.any(x < 0):
            raise InvalidAction("node id outside the compact key range")
        out = x.astype(np.uint64)
        out[big] = (np.uint64(1) << np.uint64(31)) + (x[big] - OFFSET).astype(np.uint64)
        if np.any(out >= (np.uint64(1) << np.uint64(32))):
            raise InvalidAction("node id outside the compact key range")
        return out
    return (compact(a) << np.uint64(32)) | compact(b)
