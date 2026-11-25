"""Embedding cache: memory (LRU) and disk, keyed by (artifact SHA-256, encoder id, encoder version, preprocessing hash).

The key is `crucible.semantic.encoders.cache_key(ref, modality)` = `"{ref}|{encoder_id}@{version}|{preprocessing_hash}"`
(INTERFACE_CONTRACTS section 4). Because artifacts are content-addressed, changed content is a different `ref` and
therefore a different key: a stale vector can never be returned for new content. A new encoder version or a
different preprocessing setting likewise changes the key, so representations are recomputed exactly when "content
changes or a new representation is required" (MASTER_SPEC 4.3) and reused across participants and worlds otherwise
(13.5).

Disk layout: `$CRUCIBLE_CACHE/embeddings/<encoder slug>/<hh>/<sha256(key)>.emb` (default `$CRUCIBLE_CACHE` is
`~/.cache/crucible`). A file holds a magic number, a JSON header with the full key, dimension and the SHA-256 of the
vector bytes, then the float32 little-endian vector. A read verifies all three; a file that fails verification is
counted as corrupt, deleted and treated as a miss. Writes are atomic (temporary file + rename), so concurrent
processes sharing the directory never read a partial entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import OrderedDict

import numpy as np

MAGIC = b"CRCEMB01"


def cache_root() -> str:
    base = os.environ.get("CRUCIBLE_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "crucible")
    return os.path.join(base, "embeddings")


def _slug(key: str) -> str:
    parts = key.split("|")
    enc = parts[1] if len(parts) > 1 else "misc"
    return re.sub(r"[^A-Za-z0-9._@-]+", "_", enc)[:120]


def encode_vector(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(vec, dtype="<f4")).tobytes()


def vector_sha256(vec: np.ndarray) -> str:
    return hashlib.sha256(encode_vector(vec)).hexdigest()


class EmbeddingCache:
    """Thread-safe two-level cache. `get` returns a read-only float32 vector or None; `put` stores in both levels."""

    def __init__(self, root: str | None = None, max_memory: int = 200_000, disk: bool = True):
        self.root = root or cache_root()
        self.max_memory = int(max_memory)
        self.disk = bool(disk)
        self._mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {"memory_hits": 0, "disk_hits": 0, "misses": 0, "puts": 0, "corrupt": 0, "disk_writes": 0}

    # ------------------------------------------------------------------------------------------ paths
    def path(self, key: str) -> str:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.root, _slug(key), h[:2], h + ".emb")

    # ------------------------------------------------------------------------------------------ access
    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            v = self._mem.get(key)
            if v is not None:
                self._mem.move_to_end(key)
                self.stats["memory_hits"] += 1
                return v
        if self.disk:
            v = self._read(key)
            if v is not None:
                with self._lock:
                    self.stats["disk_hits"] += 1
                    self._remember(key, v)
                return v
        with self._lock:
            self.stats["misses"] += 1
        return None

    def put(self, key: str, vec: np.ndarray, meta: dict | None = None) -> np.ndarray:
        v = np.array(vec, dtype=np.float32, copy=True).reshape(-1)
        if v.size == 0 or not np.all(np.isfinite(v)):
            raise ValueError(f"refusing to cache an empty or non-finite vector for {key}")
        v.setflags(write=False)
        with self._lock:
            self.stats["puts"] += 1
            self._remember(key, v)
        if self.disk:
            self._write(key, v, meta or {})
        return v

    def contains(self, key: str) -> bool:
        with self._lock:
            if key in self._mem:
                return True
        return self.disk and os.path.exists(self.path(key))

    def drop(self, key: str) -> None:
        """Remove one entry from both levels."""
        with self._lock:
            self._mem.pop(key, None)
        try:
            os.remove(self.path(key))
        except FileNotFoundError:
            pass

    def clear_memory(self) -> None:
        with self._lock:
            self._mem.clear()

    def __len__(self) -> int:
        return len(self._mem)

    # ------------------------------------------------------------------------------------------ internals
    def _remember(self, key: str, v: np.ndarray) -> None:
        self._mem[key] = v
        self._mem.move_to_end(key)
        while len(self._mem) > self.max_memory:
            self._mem.popitem(last=False)

    def _write(self, key: str, v: np.ndarray, meta: dict) -> None:
        p = self.path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        raw = encode_vector(v)
        head = json.dumps({"key": key, "dim": int(v.size), "dtype": "<f4",
                           "sha256": hashlib.sha256(raw).hexdigest(), "meta": meta}, sort_keys=True).encode()
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(MAGIC + len(head).to_bytes(4, "little") + head + raw)
        os.replace(tmp, p)
        with self._lock:
            self.stats["disk_writes"] += 1

    def _read(self, key: str) -> np.ndarray | None:
        p = self.path(key)
        try:
            with open(p, "rb") as fh:
                blob = fh.read()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            if blob[:8] != MAGIC:
                raise ValueError("magic")
            n = int.from_bytes(blob[8:12], "little")
            head = json.loads(blob[12:12 + n].decode())
            raw = blob[12 + n:]
            if head.get("key") != key or head.get("dtype") != "<f4" or len(raw) != 4 * int(head["dim"]) or \
                    hashlib.sha256(raw).hexdigest() != head.get("sha256"):
                raise ValueError("verification")
            v = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            if not np.all(np.isfinite(v)):
                raise ValueError("non-finite")
        except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
            with self._lock:
                self.stats["corrupt"] += 1
            try:
                os.remove(p)
            except OSError:
                pass
            return None
        v.setflags(write=False)
        return v

    def meta(self, key: str) -> dict | None:
        """The header stored with a disk entry (None if absent or invalid)."""
        try:
            with open(self.path(key), "rb") as fh:
                blob = fh.read(1 << 16)
            n = int.from_bytes(blob[8:12], "little")
            return json.loads(blob[12:12 + n].decode())
        except (OSError, ValueError):
            return None


_DEFAULT: dict[str, EmbeddingCache] = {}
_DLOCK = threading.Lock()


def default_cache() -> EmbeddingCache:
    """The process-wide cache for the current `$CRUCIBLE_CACHE` (a new one if the variable changed)."""
    root = cache_root()
    with _DLOCK:
        c = _DEFAULT.get(root)
        if c is None:
            disk = os.environ.get("CRUCIBLE_EMBED_DISK_CACHE", "1") not in ("0", "false", "no")
            c = _DEFAULT[root] = EmbeddingCache(root, disk=disk)
        return c
