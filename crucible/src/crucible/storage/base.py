"""Durable event store interface.

The store is append-only for events: `(world_id, seq)` is unique and appends must be contiguous, so two kernel
processes that both believe they own a world cannot both commit seq N (the second append fails with
`StoreConflict`). Snapshots are immutable blobs addressed by id. A world records its parent (world, seq, hash)
when it was forked, which gives a lineage tree of experiments.

Backends: `MemoryStore` (tests, fast RL), `SQLiteStore` (default, single file, WAL), `PostgresStore`
(multi-process, the primary backend for long-running canonical worlds). All three run the same conformance
tests (tests/test_storage.py).

Threat model. The hash chain makes every *local* alteration of a stored log detectable on load (a changed,
deleted, reordered or re-labelled event breaks the chain at that point). It cannot, by itself, detect someone with
write access to the store who rewrites an event and recomputes the hash of every later event, or who simply drops
the newest events: the altered log is a perfectly valid chain with a different head. No in-world participant has
that access, and write access to the store is outside what the chain alone protects against. What closes the gap
is an *anchor*: the head of a world's log - (world id, sequence number, hash) - kept somewhere the store's writer
cannot change (the experiment's results file, a ticket, another machine). `make_anchor` exports one;
`verify_anchor` checks a store against it and reports a rewritten suffix, a truncated tail, a broken chain or a
different world. Record an anchor whenever results are reported, and verify it before a stored world is trusted
again. Snapshots are zlib-compressed pickles: load them only from stores you would run code from.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any


class StoreConflict(Exception):
    pass


class Store(ABC):
    kind = "abstract"

    @abstractmethod
    def create_world(self, world_id: str, spec: dict, parent: dict | None) -> None: ...

    @abstractmethod
    def world_info(self, world_id: str) -> dict | None: ...

    @abstractmethod
    def list_worlds(self) -> list[dict]: ...

    @abstractmethod
    def set_status(self, world_id: str, status: str) -> None: ...

    @abstractmethod
    def append_events(self, world_id: str, events: list[dict]) -> None: ...

    @abstractmethod
    def read_events(self, world_id: str, from_seq: int = 0, to_seq: int | None = None) -> list[dict]: ...

    @abstractmethod
    def last_seq(self, world_id: str) -> int: ...

    @abstractmethod
    def put_snapshot(self, sid: str, world_id: str, seq: int, hash: str, blob: bytes) -> None: ...

    @abstractmethod
    def get_snapshot(self, sid: str) -> tuple[dict, bytes]: ...

    @abstractmethod
    def list_snapshots(self, world_id: str) -> list[dict]: ...

    @abstractmethod
    def acquire(self, world_id: str, owner: str, ttl_s: float = 60.0) -> bool: ...

    @abstractmethod
    def release(self, world_id: str, owner: str) -> None: ...

    def put_blob(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def get_blob(self, key: str) -> bytes | None:
        raise NotImplementedError

    def latest_snapshot(self, world_id: str) -> dict | None:
        snaps = self.list_snapshots(world_id)
        return max(snaps, key=lambda s: s["seq"]) if snaps else None

    def lineage(self, world_id: str) -> list[dict]:
        """The world and its ancestors, nearest first. Always terminates: a parent that is not in this store (a
        fork taken into a fresh store), and a record that names itself or an earlier world of the chain as its
        parent (written by versions that let a fork keep its parent's id), end the walk."""
        out = []
        seen = set()
        cur = self.world_info(world_id)
        while cur is not None and cur["id"] not in seen:
            out.append(cur)
            seen.add(cur["id"])
            par = cur.get("parent")
            cur = self.world_info(par["world"]) if par else None
        return out

    def close(self) -> None:
        pass


def dumps(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


# ====================================================================== anchors (tamper evidence beyond the chain)
ANCHOR_SCHEMA = "crucible-anchor/1"


def make_anchor(world) -> dict:
    """The head of a world's event log as a small JSON record to keep OUTSIDE the store (see the module
    docstring). `world` is a kernel World (anything with world_id, seq, last_hash and spec.hash())."""
    return {"schema": ANCHOR_SCHEMA, "world_id": world.world_id, "seq": int(world.seq), "hash": world.last_hash,
            "spec": world.spec.hash()}


def verify_anchor(store: Store, anchor: dict, world_id: str | None = None) -> list[str]:
    """Check a store against an anchor recorded earlier. Returns the list of problems (empty = the stored log up
    to the anchored event is exactly the one that was anchored). Detects: a rewritten suffix (every hash valid,
    another head), a truncated tail, any break of the chain before the anchor, a different world or
    specification. Events appended after the anchor are fine - the log is append-only."""
    from ..core.canonical import canonical_bytes, chain_hash, digest
    if not isinstance(anchor, dict) or anchor.get("schema") != ANCHOR_SCHEMA:
        return ["not an anchor record"]
    wid = world_id or anchor["world_id"]
    info = store.world_info(wid)
    if info is None:
        return [f"no world {wid} in the store"]
    problems = []
    if digest(info["spec"]) != anchor.get("spec"):
        problems.append("the stored world specification is not the anchored one")
    parent = info.get("parent")
    h, base = ("0" * 32, 0) if not parent else (parent["hash"], int(parent["seq"]))
    last = base
    at_anchor = h if anchor["seq"] == base else None
    for e in store.read_events(wid, from_seq=base, to_seq=int(anchor["seq"])):
        if e["seq"] != last + 1:
            problems.append(f"gap in the stored log at {last + 1}")
            break
        h = chain_hash(h, canonical_bytes({k: e[k] for k in ("seq", "t", "type", "actor", "cmd", "payload", "vis")}))
        if h != e["hash"]:
            problems.append(f"hash chain broken at seq {e['seq']}")
            break
        last = e["seq"]
        if last == anchor["seq"]:
            at_anchor = h
    if not problems:
        if last < anchor["seq"]:
            problems.append(f"the stored log ends at seq {last}, before the anchored seq {anchor['seq']} (truncated)")
        elif at_anchor != anchor["hash"]:
            problems.append(f"the event at the anchored seq {anchor['seq']} has another hash than the anchor: the "
                            "log was rewritten")
    return problems


def now() -> float:
    return time.time()
