from __future__ import annotations

import copy
import threading

from .base import Store, StoreConflict, now


class MemoryStore(Store):
    kind = "memory"

    def __init__(self) -> None:
        self.worlds: dict[str, dict] = {}
        self.events: dict[str, list[dict]] = {}
        self.snaps: dict[str, tuple[dict, bytes]] = {}
        self.blobs: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def create_world(self, world_id, spec, parent):
        with self._lock:
            if world_id in self.worlds:
                raise StoreConflict(f"world {world_id} exists")
            self.worlds[world_id] = {"id": world_id, "spec": copy.deepcopy(spec), "parent": copy.deepcopy(parent), "created": now(),
                                     "status": "active", "writer": None, "lease_until": 0.0}
            self.events[world_id] = []

    def world_info(self, world_id):
        w = self.worlds.get(world_id)
        return None if w is None else copy.deepcopy(w)  # callers must not be able to mutate the stored spec

    def list_worlds(self):
        return [self.world_info(w) for w in sorted(self.worlds)]

    def set_status(self, world_id, status):
        self.worlds[world_id]["status"] = status

    def append_events(self, world_id, events):
        with self._lock:
            log = self.events[world_id]
            base = self._base(world_id)
            last = log[-1]["seq"] if log else base
            for e in events:
                if e["seq"] != last + 1:
                    raise StoreConflict(f"non-contiguous append to {world_id}: {e['seq']} after {last}")
                last = e["seq"]
            log.extend(copy.deepcopy(events))

    def _base(self, world_id):
        par = self.worlds[world_id]["parent"]
        return par["seq"] if par else 0

    def read_events(self, world_id, from_seq=0, to_seq=None, types=None):
        types = None if types is None else set(types)
        return [copy.deepcopy(e) for e in self.events[world_id]
                if e["seq"] > from_seq and (to_seq is None or e["seq"] <= to_seq)
                and (types is None or e["type"] in types)]

    def last_seq(self, world_id):
        log = self.events[world_id]
        return log[-1]["seq"] if log else self._base(world_id)

    def put_snapshot(self, sid, world_id, seq, hash, blob):
        self.snaps.setdefault(sid, ({"id": sid, "world": world_id, "seq": seq, "hash": hash, "created": now(),
                                     "bytes": len(blob)}, bytes(blob)))

    def get_snapshot(self, sid):
        meta, blob = self.snaps[sid]
        return dict(meta), blob

    def list_snapshots(self, world_id):
        return sorted((dict(m) for m, _ in self.snaps.values() if m["world"] == world_id),
                      key=lambda m: (m["seq"], m["id"]))  # same order as the SQL backends

    def acquire(self, world_id, owner, ttl_s=60.0):
        with self._lock:
            w = self.worlds[world_id]
            if w["writer"] in (None, owner) or w["lease_until"] < now():
                w["writer"], w["lease_until"] = owner, now() + ttl_s
                return True
            return False

    def release(self, world_id, owner):
        with self._lock:
            w = self.worlds[world_id]
            if w["writer"] == owner:
                w["writer"], w["lease_until"] = None, 0.0

    def put_blob(self, key, data):
        self.blobs[key] = bytes(data)

    def get_blob(self, key):
        return self.blobs.get(key)

    def delete_snapshot(self, sid):
        self.snaps.pop(sid, None)

    def delete_blob(self, key):
        self.blobs.pop(key, None)

    def blob_sizes(self, prefix=""):
        return {k: len(v) for k, v in sorted(self.blobs.items()) if k.startswith(prefix)}
