"""Caching and world records of space representations.

* `RepCache` - memory LRU + disk (`$CRUCIBLE_CACHE/space/...`), keyed by the full cache key (artifact SHA-256, space
  id and version, modality, derived preprocessing hash). A file holds a magic number, a JSON header (key, features,
  shapes, SHA-256 of the float bytes) and the float32 vector and chunk matrix; a read verifies all of them (a
  corrupt file is deleted and counted) and writes are atomic, so processes and worktrees can share the directory.
* Kernel module `space` (table `space_items`): what a world has recorded. A representation computed with a ctx is
  recorded through `crucible.substrate.inference.recorded` and stored here by one `space.recorded` event; replay,
  regeneration, snapshots and forks of the world then read the recorded vector instead of running a model.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict

import numpy as np

from ...core import schema as S
from ...core.kernel import Module, applier, endpoint, register_module

MAGIC = b"CRCSPC01"
TABLE = "space_items"


def cache_root() -> str:
    base = os.environ.get("CRUCIBLE_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "crucible")
    return os.path.join(base, "space")


def b64(a: np.ndarray, dtype: str = "<f4") -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def unb64(s: str, shape, dtype: str = "<f4") -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=dtype).astype(np.float32).reshape(shape)


def clean_features(f: dict) -> dict:
    """JSON-native features: floats rounded to 6 significant digits, NaN/inf -> None, strings kept."""
    out = {}
    for k, v in sorted(f.items()):
        if isinstance(v, str):
            out[k] = v
        elif v is None:
            out[k] = None
        else:
            x = float(v)
            out[k] = float(f"{x:.6g}") if np.isfinite(x) else None
    return out


class RepCache:
    def __init__(self, root: str | None = None, max_memory: int = 50_000, disk: bool = True):
        self.root = root or cache_root()
        self.max_memory, self.disk = int(max_memory), bool(disk)
        self._mem: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {"memory_hits": 0, "disk_hits": 0, "misses": 0, "puts": 0, "corrupt": 0}

    def path(self, key: str) -> str:
        h = hashlib.sha256(key.encode()).hexdigest()
        parts = key.split("|")
        slug = re.sub(r"[^A-Za-z0-9._@-]+", "_", parts[1] if len(parts) > 1 else "misc")[:100]
        return os.path.join(self.root, slug, h[:2], h + ".rep")

    def get(self, key: str):
        with self._lock:
            hit = self._mem.get(key)
            if hit is not None:
                self._mem.move_to_end(key)
                self.stats["memory_hits"] += 1
                return hit
        if self.disk:
            hit = self._read(key)
            if hit is not None:
                with self._lock:
                    self.stats["disk_hits"] += 1
                    self._remember(key, hit)
                return hit
        with self._lock:
            self.stats["misses"] += 1
        return None

    def put(self, key: str, vec: np.ndarray, chunks: np.ndarray, features: dict):
        vec = np.array(vec, dtype=np.float32).reshape(-1)
        chunks = np.array(chunks, dtype=np.float32).reshape(-1, vec.size)
        if not (np.all(np.isfinite(vec)) and np.all(np.isfinite(chunks))):
            raise ValueError(f"refusing to cache a non-finite representation for {key}")
        vec.setflags(write=False)
        chunks.setflags(write=False)
        feats = clean_features(features)
        entry = (vec, chunks, feats)
        with self._lock:
            self.stats["puts"] += 1
            self._remember(key, entry)
        if self.disk:
            self._write(key, entry)
        return entry

    def drop(self, key: str) -> None:
        with self._lock:
            self._mem.pop(key, None)
        try:
            os.remove(self.path(key))
        except FileNotFoundError:
            pass

    def _remember(self, key, entry) -> None:
        self._mem[key] = entry
        self._mem.move_to_end(key)
        while len(self._mem) > self.max_memory:
            self._mem.popitem(last=False)

    def _write(self, key: str, entry) -> None:
        vec, chunks, feats = entry
        raw = vec.astype("<f4").tobytes() + chunks.astype("<f4").tobytes()
        head = json.dumps({"key": key, "dim": int(vec.size), "k": int(chunks.shape[0]), "features": feats,
                           "sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True).encode()
        p = self.path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(MAGIC + len(head).to_bytes(4, "little") + head + raw)
        os.replace(tmp, p)

    def _read(self, key: str):
        p = self.path(key)
        try:
            with open(p, "rb") as fh:
                blob = fh.read()
        except OSError:
            return None
        try:
            if blob[:8] != MAGIC:
                raise ValueError("magic")
            n = int.from_bytes(blob[8:12], "little")
            head = json.loads(blob[12:12 + n].decode())
            raw = blob[12 + n:]
            d, k = int(head["dim"]), int(head["k"])
            if head.get("key") != key or len(raw) != 4 * d * (1 + k) or hashlib.sha256(raw).hexdigest() != head["sha256"]:
                raise ValueError("verification")
            arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            vec, chunks = arr[:d], arr[d:].reshape(k, d)
            if not (np.all(np.isfinite(vec)) and np.all(np.isfinite(chunks))):
                raise ValueError("non-finite")
        except (ValueError, KeyError, UnicodeDecodeError):
            with self._lock:
                self.stats["corrupt"] += 1
            try:
                os.remove(p)
            except OSError:
                pass
            return None
        vec.setflags(write=False)
        chunks.setflags(write=False)
        return vec, chunks, head["features"]


_DEFAULT: dict[str, RepCache] = {}
_DLOCK = threading.Lock()


def default_cache() -> RepCache:
    root = cache_root()
    with _DLOCK:
        c = _DEFAULT.get(root)
        if c is None:
            disk = os.environ.get("CRUCIBLE_EMBED_DISK_CACHE", "1") not in ("0", "false", "no")
            c = _DEFAULT[root] = RepCache(root, disk=disk)
        return c


# ============================================================================================== world records
def world_entry(world, key: str):
    """(vec, chunks, features) recorded in `world` under `key`, or None."""
    if world is None or "space" not in getattr(world, "modules", {}):
        return None
    rec = world.state.get(TABLE, key)
    if rec is None:
        return None
    dec = world.runtime.setdefault("space.decoded", {})
    hit = dec.get(key)
    if hit is not None and hit[0] == rec["sha256"]:
        return hit[1]
    d = int(rec["dim"])
    vec = unb64(rec["vec"], (d,))
    if rec.get("chunks"):
        chunks = unb64(rec["chunks"], (-1, d), dtype="<f2")
    else:
        chunks = vec[None, :].copy()
    vec.setflags(write=False)
    chunks.setflags(write=False)
    entry = (vec, chunks, rec["features"])
    dec[key] = (rec["sha256"], entry)
    return entry


@register_module
class SpaceRecords(Module):
    """Kernel module `space`: the world's record of shared-space representations (see the module docstring)."""
    name = "space"
    host = "space"
    version = "1"

    def install(self, world):
        world.state.declare_table(TABLE, index=("ref",))

    @applier("space.recorded")
    def ap_recorded(self, ev):
        st = self.world.state
        for it in ev.payload["items"]:
            if st.has(TABLE, it["key"]):
                continue
            st.put(TABLE, it["key"], {k: it[k] for k in ("ref", "modality", "space", "dim", "vec", "chunks", "sha256",
                                                          "features")} | {"t": ev.t})

    @endpoint("info", S.obj(), doc="The shared semantic space of this world: its id, version and vector layout "
              "(topic, affect and quantity blocks). Nothing about any artifact or participant.", cost_s=0,
              read_only=True)
    def ep_info(self, ctx):
        from . import DIM, SPACE_ID, SPACE_VERSION, layout
        return {"space": SPACE_ID, "version": SPACE_VERSION, "dim": DIM, "layout": layout()}

    def check_invariants(self, world):
        errs = []
        for key, rec in world.state.scan(TABLE):
            raw = base64.b64decode(rec["vec"])
            if hashlib.sha256(raw).hexdigest() != rec["sha256"]:
                errs.append(f"space item {key[:60]} does not match its digest")
            elif len(raw) != 4 * int(rec["dim"]) or not np.all(np.isfinite(np.frombuffer(raw, dtype="<f4"))):
                errs.append(f"space item {key[:60]} has a bad vector")
            if not key.startswith(rec["ref"] + "|"):
                errs.append(f"space item {key[:60]} is filed under the wrong artifact")
        return errs
