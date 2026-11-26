"""Retention and compaction of a stored world's blobs and snapshots (Crucible 5.0, workstream A; D-21, audit A2).

Every cohort decision (`cols.batch`), neural checkpoint, training-example batch and recorded inference writes a
content-addressed system array blob (`art/<sha256>` in the world's store). A 30-day development-profile run wrote
about 37k of them (162 MB); a 365-day run would write about 2 GB, most of it small, compressible arrays. This module
bounds that growth without giving up anything replay, regeneration or recounting needs:

* `pack(world)` packs the individually stored system array blobs (owner `sys`, kind `batch`) into compressed
  chunks `pack/<sha256 of the chunk>` (LZMA by default) in creation order, and replaces each blob's own key by a
  fixed-size pointer. Every read still checks the blob's SHA-256 (and the chunk's), so packing never changes a
  byte any world reads; replay from genesis, regeneration and audience recounts stay exact. Blobs that a module
  will rebuild runtime state from (`Module.blob_roots`, e.g. the latest neural checkpoints, now or at a retained
  snapshot) and blobs already declared superseded stay unpacked: they are read on every load, are mostly
  incompressible weights, and are the ones that may be retired later.
* `collect(world)` drops blobs their owner module declares superseded (`Module.superseded_blobs`, e.g. a
  checkpoint replaced by a later one) - never a blob that a `cols.batch` event applies, never a root of the current
  state or of any retained snapshot (each snapshot records its roots), and nothing at all when the store holds
  other worlds (forks) that are not given in `others`. Only modules can declare blobs superseded: unknown blobs are
  always kept.
* `prune_snapshots(world, keep_last, keep_every_s)` keeps the most recent snapshots and one per period.
* `compact(world)` = collect + pack (+ optional snapshot pruning); `stats(world)` reports the store's contents.

Cost per compaction (D-78). System array blobs are not world state (`crucible.substrate.artifacts`): their REFs
live only in the events that use them. Every compaction reads only the events written since the previous one
(`_scan`: system-blob REFs in order of first appearance, and the REFs `cols.batch` applies) and looks up the sizes
of exactly the blobs it considers (indexed key lookups), so a weekly compaction costs O(new events + pending blobs),
not O(blobs) or O(log). The scan's position is also kept in the store (`ret/<world>/scan` and append-only
`ret/<world>/batch/<i>` segments, checked against the event hash at that position), so the first compaction in a new
process does not re-read the log.

What is NOT bounded here, and why: the event log itself (the hash-chained record of every decision; SQLite stores
large payloads compressed) and the state's metadata row per participant artifact. Both grow linearly with activity,
as any exact record must.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import pickle
import re
import zlib
from typing import Iterable

from ..core.errors import ProtocolError
from . import artifacts as A

MiB = 1 << 20
CODECS = {"lzma": "x", "zlib": "z"}


def _store(world):
    st = world.store
    if st is None:
        raise ProtocolError("retention works on a stored world (a store-less world keeps its blobs in the process "
                            "pool, released when the world is closed)")
    for m in ("blob_sizes", "delete_blob"):
        if not hasattr(st, m):
            raise ProtocolError(f"store {type(st).__name__} has no {m}()")
    return st


_REF_HEX = re.compile(r"sha256:([0-9a-f]{64})")
_SCAN_WINDOW = 50_000          # events read per store query while scanning


def _blob_size(store, key: str) -> int | None:
    """Size of one blob by an indexed key lookup (blob_sizes with the full key as prefix)."""
    return store.blob_sizes(key).get(key)


def _system_hex(world, h: str) -> bool:
    """Whether REF hex `h` can be a system array blob: the state records no participant artifact with these bytes
    (rows of system batch blobs exist only in stores written before D-78)."""
    a = world.state.tables["artifacts"].get(A.REF_PREFIX + h) if "artifacts" in world.state.tables else None
    return a is None or (a["owner"] == "sys" and a["kind"] == "batch")


def _scan_key(world) -> str:
    return f"ret/{world.world_id}/scan"


def _load_scan(world, store) -> dict | None:
    """The scan position persisted in the store, if it still matches the world's log (same event hash there)."""
    raw = store.get_blob(_scan_key(world))
    if not raw:
        return None
    try:
        d = json.loads(raw.decode())
        seq = int(d["seq"])
        if seq > world.seq:
            return None
        if seq > 0:
            evs = store.read_events(world.world_id, seq - 1, seq)
            if len(evs) != 1 or evs[0].get("hash") != d["hash"]:
                return None
        batch = set()
        for i in range(int(d["segments"])):
            seg = store.get_blob(f"ret/{world.world_id}/batch/{i}")
            if seg is None or len(seg) % 32:
                return None
            batch.update(A.REF_PREFIX + seg[j:j + 32].hex() for j in range(0, len(seg), 32))
        return {"store": store, "seq": seq, "hash": d["hash"], "pending": dict.fromkeys(d["pending"]),
                "batch": batch, "segments": int(d["segments"]), "new_batch": []}
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        return None


def _save_scan(world, store, sc: dict) -> None:
    if sc["new_batch"]:
        store.put_blob(f"ret/{world.world_id}/batch/{sc['segments']}",
                       b"".join(bytes.fromhex(r[len(A.REF_PREFIX):]) for r in sc["new_batch"]))
        sc["segments"] += 1
        sc["new_batch"] = []
    store.put_blob(_scan_key(world), json.dumps({"seq": sc["seq"], "hash": sc["hash"], "segments": sc["segments"],
                                                 "pending": list(sc["pending"])}).encode())


def _scan(world, store) -> dict:
    """Read the events written since the previous call (D-78): REFs of system array blobs in order of first
    appearance (`pending`: candidates for packing until they are packed, dropped or gone) and the REFs `cols.batch`
    events apply (`batch`: never dropped). The world's runtime keeps the result and the position; the store keeps a
    copy so a new process continues where the last one stopped."""
    sc = world.runtime.get("retention.scan")
    if sc is None or sc["store"] is not store or sc["seq"] > world.seq:
        sc = _load_scan(world, store) or {"store": store, "seq": 0, "hash": None, "pending": {}, "batch": set(),
                                          "segments": 0, "new_batch": []}
        world.runtime["retention.scan"] = sc
    pos = sc["seq"]
    while pos < world.seq:
        evs = store.read_events(world.world_id, pos, min(pos + _SCAN_WINDOW, world.seq))
        if not evs:
            break
        for e in evs:
            p = e["payload"]
            if e["type"] == "cols.batch":
                ref = p["ref"]
                if ref not in sc["batch"]:
                    sc["batch"].add(ref)
                    sc["new_batch"].append(ref)
            for h in _REF_HEX.findall(json.dumps(p, separators=(",", ":"), default=str)):
                if h not in sc["pending"] and _system_hex(world, h):
                    sc["pending"][h] = None
            pos = int(e["seq"])
            sc["hash"] = e.get("hash")
    sc["seq"] = pos
    return sc


def roots(world) -> set[str]:
    """REFs of blobs the modules of `world` rebuild runtime state from in its current state (Module.blob_roots)."""
    out: set[str] = set()
    for mod in world.modules.values():
        fn = getattr(mod, "blob_roots", None)
        if fn is not None:
            out.update(r for r in fn(world) if r)
    return out


def superseded(world) -> set[str]:
    out: set[str] = set()
    for mod in world.modules.values():
        fn = getattr(mod, "superseded_blobs", None)
        if fn is not None:
            out.update(r for r in fn(world) if r)
    return out


def _batch_refs(world, store) -> set[str]:
    """REFs that `cols.batch` events apply (their appliers read them on every replay): never dropped. Incremental
    (`_scan`): a periodic compaction reads only the events written since the previous one."""
    return _scan(world, store)["batch"]


_SNAP_META: dict[tuple, dict] = {}     # (store path, snapshot id, hash) -> {"roots", "t"}: snapshots are immutable


def _snap_meta(store, sid: str) -> dict:
    """The roots and simulated time a snapshot recorded (read once per process: unpickling every retained snapshot
    at every compaction cost O(state x snapshots))."""
    key = (getattr(store, "path", id(store)), sid)
    m = _SNAP_META.get(key)
    if m is None:
        d = pickle.loads(zlib.decompress(store.get_snapshot(sid)[1]))
        r = d.get("blob_roots")
        if "t" in d:
            t = int(d["t"])
        else:
            from ..core.state import StateStore
            t = int(StateStore.loads(d["state"]).meta.get("clock", 0))
        m = {"roots": None if r is None else frozenset(r), "t": t}
        if len(_SNAP_META) > 4096:
            _SNAP_META.clear()
        _SNAP_META[key] = m
    return m


def snapshot_roots(store, sid: str) -> set[str] | None:
    """The blob roots a snapshot recorded when it was taken (None for snapshots older than this mechanism)."""
    r = _snap_meta(store, sid)["roots"]
    return None if r is None else set(r)


# ---------------------------------------------------------------------------------------------------- pack
def pack(world, *, chunk_bytes: int = 32 * MiB, codec: str = "lzma", level: int = 1,
         min_bytes: int = 4 * MiB) -> dict:
    """Pack the unpacked system array blobs of a stored world into compressed chunks. Returns
    {packed, chunks, raw_bytes, stored_bytes, skipped_roots}. Crash-safe: a chunk is written before any pointer to
    it, and a blob is always readable either in full or through its pointer. Fewer than `min_bytes` of unpacked
    blobs are left for the next call (chunks stay large)."""
    store = _store(world)
    if codec not in CODECS:
        raise ProtocolError(f"codec must be one of {sorted(CODECS)}")
    world.flush()
    sc = _scan(world, store)
    # never packed: blobs a load rebuilds from (now or from a retained snapshot) and blobs already superseded -
    # they are the ones collect() may drop later, and neural weights hardly compress anyway
    keep_refs = roots(world) | superseded(world)
    for snap in store.list_snapshots(world.world_id):
        keep_refs |= snapshot_roots(store, snap["id"]) or set()
    keep = {r[len(A.REF_PREFIX):] for r in keep_refs}
    todo = []
    for h in list(sc["pending"]):
        if h in keep:
            continue                          # stays a candidate: roots change, superseded blobs may be dropped
        n = _blob_size(store, A.blob_key(h))
        if n is None or n == A.PTR_LEN:
            del sc["pending"][h]              # gone (dropped, never stored) or already packed
            continue
        todo.append((h, n))
    out = {"packed": 0, "chunks": 0, "raw_bytes": 0, "stored_bytes": 0, "skipped_roots": len(keep)}
    if sum(n for _, n in todo) < min_bytes:
        _save_scan(world, store, sc)
        return out
    i = 0
    while i < len(todo):
        members, raw, size = [], [], 0
        while i < len(todo) and (size < chunk_bytes or not members):
            h, _n = todo[i]
            data = store.get_blob(A.blob_key(h))
            i += 1
            if data is None or A.sha256_hex(data) != h:
                continue          # missing or damaged: never packed (reads keep reporting it unavailable)
            members.append((h, size, len(data)))
            raw.append(data)
            size += len(data)
        if not members:
            continue
        blob = b"".join(raw)
        comp = lzma.compress(blob, preset=level) if codec == "lzma" else zlib.compress(blob, max(1, level))
        ph = hashlib.sha256(comp).hexdigest()
        store.put_blob(f"pack/{ph}", comp)
        if A.sha256_hex(store.get_blob(f"pack/{ph}") or b"") != ph:
            raise ProtocolError("chunk did not round-trip through the store; nothing was repointed")
        for h, off, n in members:
            store.put_blob(A.blob_key(h), A.pack_pointer(ph, off, n, CODECS[codec]))
            sc["pending"].pop(h, None)
        out["packed"] += len(members)
        out["chunks"] += 1
        out["raw_bytes"] += size
        out["stored_bytes"] += len(comp)
    _save_scan(world, store, sc)
    return out


# ---------------------------------------------------------------------------------------------------- collect
def collect(world, *, others: Iterable = (), dry_run: bool = False) -> dict:
    """Drop the blobs modules declare superseded that nothing retained still needs. Returns
    {dropped, bytes, kept_roots, declared, reason?}."""
    store = _store(world)
    world.flush()
    others = list(others)
    ids = {world.world_id, *(o.world_id for o in others)}
    foreign = sorted(w["id"] for w in store.list_worlds() if w["id"] not in ids)
    declared = superseded(world)
    out = {"dropped": 0, "bytes": 0, "declared": len(declared), "kept_roots": 0}
    if foreign:
        out["reason"] = f"store also holds {len(foreign)} other world(s) ({', '.join(foreign[:3])}...): kept all"
        return out
    if not declared:
        return out
    needed = roots(world) | _batch_refs(world, store)
    for o in others:
        needed |= roots(o) | _batch_refs(o, store)
    unknown_snaps = 0
    for wid in ids:
        for snap in store.list_snapshots(wid):
            r = snapshot_roots(store, snap["id"])
            if r is None:
                unknown_snaps += 1
            else:
                needed |= r
    if unknown_snaps:
        out["reason"] = f"{unknown_snaps} snapshot(s) without recorded roots: kept all"
        return out
    drop = sorted(declared - needed)
    out["kept_roots"] = len(declared & needed)
    for ref in drop:
        key = A.blob_key(ref[len(A.REF_PREFIX):])
        n = _blob_size(store, key)                  # one indexed lookup per candidate, never a scan of every blob
        if n is None:
            continue
        if n == A.PTR_LEN:
            continue          # already packed: its bytes live in a shared chunk; nothing to gain by dropping
        out["dropped"] += 1
        out["bytes"] += n
        if not dry_run:
            store.delete_blob(key)
    return out


# ---------------------------------------------------------------------------------------------------- snapshots
def prune_snapshots(world, *, keep_last: int = 3, keep_every_s: int | None = 30 * 86400,
                    dry_run: bool = False) -> dict:
    """Keep the `keep_last` newest snapshots of the world plus the first one in every `keep_every_s` period of
    simulated time; delete the rest. Returns {kept, deleted, bytes}."""
    store = _store(world)
    if not hasattr(store, "delete_snapshot"):
        raise ProtocolError(f"store {type(store).__name__} has no delete_snapshot()")
    snaps = sorted(store.list_snapshots(world.world_id), key=lambda s: s["seq"])
    keep = {s["id"] for s in snaps[-keep_last:]} if keep_last > 0 else set()
    if keep_every_s:
        seen = set()
        for s in snaps:
            t = _snapshot_time(store, s["id"])
            period = t // int(keep_every_s)
            if period not in seen:
                seen.add(period)
                keep.add(s["id"])
    out = {"kept": 0, "deleted": 0, "bytes": 0}
    for s in snaps:
        if s["id"] in keep:
            out["kept"] += 1
            continue
        out["deleted"] += 1
        out["bytes"] += int(s.get("bytes") or 0)
        if not dry_run:
            store.delete_snapshot(s["id"])
    return out


def _snapshot_time(store, sid: str) -> int:
    return _snap_meta(store, sid)["t"]


# ---------------------------------------------------------------------------------------------------- all together
def compact(world, *, prune: bool = False, **pack_kw) -> dict:
    """collect + pack (+ prune_snapshots when `prune`)."""
    out = {"collect": collect(world)}
    if prune:
        out["snapshots"] = prune_snapshots(world)
    out["pack"] = pack(world, **pack_kw)
    return out


def stats(world) -> dict:
    """What the world's store holds: blobs (full, packed pointers, chunks) and snapshots, in bytes."""
    store = _store(world)
    arts = store.blob_sizes("art/")
    packs = store.blob_sizes("pack/")
    ptr = sum(1 for n in arts.values() if n == A.PTR_LEN)
    snaps = store.list_snapshots(world.world_id)
    return {"blobs": len(arts), "blobs_packed": ptr, "blob_bytes_unpacked": sum(n for n in arts.values()
                                                                               if n != A.PTR_LEN),
            "pointer_bytes": ptr * A.PTR_LEN, "chunks": len(packs), "chunk_bytes": sum(packs.values()),
            "snapshots": len(snaps), "snapshot_bytes": sum(int(s.get("bytes") or 0) for s in snaps),
            "live_blob_bytes": sum(arts.values()) + sum(packs.values())}
