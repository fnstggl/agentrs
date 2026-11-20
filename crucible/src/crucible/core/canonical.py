"""Canonical encoding and hashing.

Everything that is hashed (events, state digests, snapshot ids, spec hashes) goes through `canonical_bytes`,
which produces a byte string that does not depend on dict insertion order, set iteration order or the
Python hash seed. Floats are encoded with `repr` (shortest round-trip form), so the same float always gives
the same bytes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _norm(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_norm(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return {"__nd__": str(x.dtype), "shape": list(x.shape), "sha": hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=16).hexdigest()}
    if isinstance(x, bytes):
        return {"__bytes__": hashlib.blake2b(x, digest_size=16).hexdigest(), "n": len(x)}
    if hasattr(x, "items") and not isinstance(x, (str, bytes)):  # MappingProxyType etc.
        return {str(k): _norm(v) for k, v in x.items()}
    return x


def canonical_bytes(x: Any) -> bytes:
    # surrogatepass: unpaired surrogates (valid in Python str and JSON escapes) must hash, not raise
    return json.dumps(_norm(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8", "surrogatepass")


def canonical_json(x: Any) -> str:
    return canonical_bytes(x).decode()


def digest(x: Any, size: int = 16) -> str:
    return hashlib.blake2b(canonical_bytes(x), digest_size=size).hexdigest()


def chain_hash(prev: str, body: bytes) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(prev.encode())
    h.update(body)
    return h.hexdigest()


def to_jsonable(x: Any) -> Any:
    """Plain JSON-compatible deep copy (MappingProxyType → dict, tuple → list, numpy scalars → Python)."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(to_jsonable(v) for v in x)
    if hasattr(x, "items"):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON-able: {type(x).__name__}")
