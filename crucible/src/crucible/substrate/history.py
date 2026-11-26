"""Historical queries over a world's recorded history (Crucible 5.0, workstream A; MASTER_SPEC Part 10.4 "historical
queries", Gate P long-horizon persistence, learner provenance).

Everything is computed from what the kernel records - snapshots plus the hash-chained event log plus the
content-addressed blobs - never from a separate copy that could drift:

* `state_at(source, t=None, seq=None)` -> a read-only World as it was right after event `seq`, or at simulated time
  `t` (after every event with event time <= t). Built from the nearest earlier snapshot of the world (or genesis) and
  replay of the events after it, with the hash chain verified. The world can be read (tables, columns, artifacts,
  `observe`) but never written: its store refuses writes.
* `events_of(source, entity, ...)` -> the recorded events that involve `entity`: it acted, it is named in the payload,
  or the event was addressed to it (`vis`). For a population individual `p<i>` the cohort decisions of its rows are
  included too (`cols.batch` events whose array blob lists row i), so an individual's history covers what it decided
  as a member of a cohort, not only events naming it.
* `record_history(source, table, rid, ...)` -> every change of one state record: [(seq, t, type, record | None)].
* `column_history(source, table, column, row, ...)` -> every change of one column cell: [(seq, t, kind, value)].
* `mainline(events)` / `mainline_digest(events)` -> the main line of a log: the events not emitted by a
  same-transaction subscriber (`via`), at their main-line positions. Two worlds that differ only by observers,
  validators or removed couplings have the same main line exactly when the rest of the world behaved identically
  (D-21; counterfactual removal, Gate O / SP-10.3).

`source` is a kernel World (with or without a store; a store-less world answers from its in-memory log, which
must reach back to genesis) or a pair `(store, world_id)`.
"""
from __future__ import annotations

import bisect
import copy
from typing import Any, Iterable

import numpy as np

from ..core.canonical import canonical_bytes, chain_hash
from ..core.errors import NotFound, ProtocolError
from ..core.kernel import GENESIS_HASH, Event, World, WorldSpec

__all__ = ["state_at", "seq_at_time", "events_of", "record_history", "column_history", "mainline",
           "mainline_digest", "ReadOnlyStore", "mentions"]


# ---------------------------------------------------------------------------------------------------- sources
class ReadOnlyStore:
    """A view of a store that serves reads (events, snapshots, blobs) and refuses every write, so a historical world
    can never append to, or overwrite anything in, the store it was rebuilt from."""

    kind = "read-only"

    def __init__(self, store):
        self._s = store

    def __getattr__(self, name):
        if name in ("read_events", "get_blob", "blob_sizes", "world_info", "list_snapshots", "get_snapshot",
                    "latest_snapshot", "last_seq", "list_worlds", "lineage"):
            return getattr(self._s, name)
        raise ProtocolError(f"historical worlds are read-only (store.{name} refused)")

    def put_blob(self, key, data):
        raise ProtocolError("historical worlds are read-only (put_blob refused)")

    def append_events(self, world_id, events):
        raise ProtocolError("historical worlds are read-only (append_events refused)")


class _Source:
    def __init__(self, source):
        if isinstance(source, World):
            self.world = source
            if source.store is not None:
                source.flush()
                self.store, self.wid = source.store, source.world_id
            else:
                if source.log_base != 0:
                    raise ProtocolError("this store-less world's in-memory log does not reach back to genesis")
                self.store, self.wid = None, source.world_id
            self.spec = source.spec
        else:
            store, wid = source
            info = store.world_info(wid)
            if info is None:
                raise NotFound(f"no world {wid} in the store")
            self.world, self.store, self.wid = None, store, wid
            self.spec = WorldSpec.from_dict(info["spec"])
        self.parent = (self.store.world_info(self.wid) or {}).get("parent") if self.store is not None else None

    @property
    def last_seq(self) -> int:
        if self.store is not None:
            return int(self.store.last_seq(self.wid))
        return self.world.seq

    def events(self, from_seq: int = 0, to_seq: int | None = None, types=None) -> list[dict]:
        if self.store is not None:
            try:
                return self.store.read_events(self.wid, from_seq, to_seq, types=types)
            except TypeError:       # a store without type filtering
                evs = self.store.read_events(self.wid, from_seq, to_seq)
                return [e for e in evs if types is None or e["type"] in set(types)]
        to_seq = self.world.seq if to_seq is None else to_seq
        ts = None if types is None else set(types)
        return [e.to_dict() for e in self.world.log[from_seq:to_seq] if ts is None or e.type in ts]

    def snapshots(self) -> list[dict]:
        return self.store.list_snapshots(self.wid) if self.store is not None else []

    def blobs(self):
        return ReadOnlyStore(self.store) if self.store is not None else None


# ---------------------------------------------------------------------------------------------------- time -> seq
def seq_at_time(source, t: int) -> int:
    """The seq of the last event with event time <= t (event times never decrease along the log)."""
    src = source if isinstance(source, _Source) else _Source(source)
    if src.store is None:
        ts = [e.t for e in src.world.log]
        return bisect.bisect_right(ts, int(t))
    seq = 0
    for e in src.events(0, None):     # a store without a time index: one pass
        if e["t"] > t:
            break
        seq = e["seq"]
    return seq


# ---------------------------------------------------------------------------------------------------- past state
def state_at(source, t: int | None = None, seq: int | None = None, *, runtime: bool = False,
             verify: bool = True) -> World:
    """The world right after event `seq` (or at simulated time `t`), rebuilt from the nearest earlier snapshot plus
    replay with the hash chain verified. `runtime=True` also runs every Module.on_load (runtime caches, e.g. rebuilt
    neural weights) - needed only to call module APIs that use them. The result refuses writes to the store."""
    src = _Source(source)
    if (t is None) == (seq is None):
        raise ProtocolError("give exactly one of t or seq")
    if seq is None:
        seq = seq_at_time(src, int(t))
    seq = int(seq)
    last = src.last_seq
    if seq < 0 or seq > last:
        raise NotFound(f"world {src.wid} has events 1..{last}, not {seq}")
    base_blob, base_seq = None, 0
    snaps = [s for s in src.snapshots() if s["seq"] <= seq]
    if snaps:
        best = max(snaps, key=lambda s: s["seq"])
        base_blob, base_seq = src.store.get_snapshot(best["id"])[1], best["seq"]
    elif src.parent:
        raise ProtocolError(f"fork {src.wid} has no snapshot at or before seq {seq}")
    evs = src.events(base_seq, seq)
    w = World.replay(src.spec, evs, world_id=f"{src.wid}@{seq}", verify=verify, base_blob=base_blob,
                     store=src.blobs(), load=runtime)
    if w.seq != seq:
        raise ProtocolError(f"replay reached seq {w.seq}, not {seq}")
    w.runtime["history"] = {"source": src.wid, "seq": seq, "t": w.now, "base_snapshot_seq": base_seq,
                            "replayed": len(evs)}
    return w


# ---------------------------------------------------------------------------------------------------- entity history
def mentions(x: Any, entity: str) -> bool:
    """True if `entity` is a string value (or key) anywhere in the JSON value `x`."""
    if isinstance(x, str):
        return x == entity
    if isinstance(x, dict):
        return any(k == entity or mentions(v, entity) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return any(mentions(v, entity) for v in x)
    return False


def _involves(e: dict, entity: str) -> str | None:
    if e["actor"] == entity:
        return "actor"
    vis = e["vis"]
    if isinstance(vis, (list, tuple)) and entity in vis:
        return "addressed"
    if mentions(e["payload"], entity):
        return "named"
    return None


def events_of(source, entity: str, *, since_seq: int = 0, to_seq: int | None = None, types: Iterable[str] | None = None,
              rows: bool = True, row_tables: Iterable[str] = ("pop",), limit: int | None = None) -> list[dict]:
    """Recorded events involving `entity`, oldest first: each is the stored event dict plus `role` ("actor",
    "addressed", "named", or "cohort" for a cohort decision over the individual's row, with `row` and the
    individual's values from the batch blob in `values`)."""
    from . import artifacts, ids
    src = _Source(source)
    types = None if types is None else set(types)
    row = ids.index(entity) if rows else None
    row_tables = set(row_tables)
    reader = None
    out = []
    for e in src.events(since_seq, to_seq):
        if types is not None and e["type"] not in types:
            continue
        role = _involves(e, entity)
        rec = None
        if role is not None:
            rec = dict(e, role=role)
        elif row is not None and e["type"] == "cols.batch" and e["payload"]["table"] in row_tables:
            if reader is None:
                reader = _BlobReader(src)
            arrs = reader.arrays(e["payload"]["ref"])
            ix = arrs["idx"]
            j = np.searchsorted(ix, row)
            if j < ix.size and int(ix[j]) == row:
                hits = np.flatnonzero(ix == row)
                vals = {c: arrs[f"v.{c}"][hits].tolist() for c in e["payload"]["cols"]}
                rec = dict(e, role="cohort", row=int(row), values=vals)
        if rec is not None:
            out.append(rec)
            if limit is not None and len(out) >= limit:
                break
    return out


class _BlobReader:
    """Reads array blobs of a recorded world (its store, or the live world's own artifacts) with hash checks."""

    def __init__(self, src: _Source):
        self.src = src
        self.world = src.world

    def arrays(self, ref: str) -> dict:
        from . import artifacts
        if self.world is not None:
            try:
                return artifacts.get_array_blob(self.world, ref)     # its cache, store or lineage's pool (D-78)
            except NotFound:
                if self.src.store is None:
                    raise
        h = artifacts.parse_ref(ref)
        data = artifacts.read_store_blob(self.src.store, h)
        if data is None:
            raise NotFound(f"bytes of {ref} are not available in the store")
        return artifacts.npz_arrays(data)


# ---------------------------------------------------------------------------------------------------- record history
def _replayer(src: _Source, from_seq: int) -> tuple[World, int]:
    if from_seq <= 0:
        return World.replay(src.spec, [], world_id=f"{src.wid}@hist", store=src.blobs(), load=False), 0
    w = state_at(src.world if src.world is not None else (src.store, src.wid), seq=from_seq)
    return w, from_seq


def record_history(source, table: str, rid: str, *, from_seq: int = 0, to_seq: int | None = None) -> list[dict]:
    """Every change of record `table/rid` between from_seq and to_seq: [{seq, t, type, record}] with the record as
    it was right after that event (None when it was deleted). The first entry (seq = from_seq) is its value at the
    start of the window, when it existed."""
    src = _Source(source)
    w, base = _replayer(src, from_seq)
    if table not in w.state.tables:
        raise NotFound(f"no table {table}")
    cur = copy.deepcopy(w.state.tables[table].get(rid))
    out = [{"seq": base, "t": w.now, "type": None, "record": cur}] if cur is not None else []
    for d in src.events(base, to_seq):
        ev = Event.from_dict(d)
        w.state.begin()
        w._apply(ev)
        w.state.commit()
        w.seq = ev.seq
        new = w.state.tables[table].get(rid)
        if new != cur:
            cur = copy.deepcopy(new)
            out.append({"seq": ev.seq, "t": ev.t, "type": ev.type, "record": cur})
    return out


def column_history(source, table: str, column: str, row: int, *, from_seq: int = 0,
                   to_seq: int | None = None) -> list[dict]:
    """Every change of one column cell: [{seq, t, kind, value}] (value as a JSON list/number), starting with its value
    at from_seq. Only cohort events (`cols.batch` / `cols.grow`) change columns, so only those are applied after
    the start state is rebuilt."""
    src = _Source(source)
    w, base = _replayer(src, from_seq)
    key = f"{table}.{column}"
    if key not in w.state.arrays:
        raise NotFound(f"no column {key}")
    def val():
        a = w.state.arrays[key]
        return a[row].tolist() if row < a.shape[0] else None
    cur = val()
    out = [{"seq": base, "t": w.now, "kind": None, "value": cur}]
    for d in src.events(base, to_seq):
        ev = Event.from_dict(d)
        w.state.begin()
        w._apply(ev)
        w.state.commit()
        w.seq = ev.seq
        if ev.type in ("cols.batch", "cols.grow") and ev.payload.get("table") == table:
            new = val()
            if new != cur:
                cur = new
                out.append({"seq": ev.seq, "t": ev.t, "kind": ev.payload.get("kind", "grow"), "value": cur})
    return out


# ---------------------------------------------------------------------------------------------------- main line
def _as_dict(e) -> dict:
    return e.to_dict() if isinstance(e, Event) else e


def mainline(events: Iterable, *, after_genesis: bool = False) -> list[dict]:
    """The main line of a log: bodies of the events that no same-transaction subscriber emitted, with `seq` replaced
    by the main-line position (`mseq`). With `after_genesis`, the genesis note (which carries the spec hash, so it
    differs between worlds with different module lists) is left out."""
    out = []
    pos = 0
    for e in events:
        d = _as_dict(e)
        if d.get("via") is not None:
            continue
        pos += 1
        if after_genesis and pos == 1 and d["type"] == "note" and d["payload"].get("what") == "genesis":
            continue
        out.append({"mseq": pos, "t": d["t"], "type": d["type"], "actor": d["actor"], "cmd": d["cmd"],
                    "payload": d["payload"], "vis": d["vis"]})
    return out


def mainline_digest(events: Iterable, *, after_genesis: bool = False) -> str:
    """A hash chain over `mainline(events)`: equal for two logs exactly when their main lines are identical."""
    h = GENESIS_HASH
    for b in mainline(events, after_genesis=after_genesis):
        h = chain_hash(h, canonical_bytes(b))
    return h
