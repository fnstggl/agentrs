"""Exact sets of individuals held in world state (workstream C2; DECISION_LOG D-5).

A set of population indices (the viewers of one item, the identities exposed to it, a creator's followers) is one
dense array of the kernel's `StateStore`, so it is part of every snapshot, fork and state digest, and every change is
journaled (a failed command rolls it back exactly). Two exact representations, chosen by size:

* **sparse**: a sorted array of unique int32 indices (int64 when the universe exceeds 2**31), for small sets;
* **dense**: a bitset (uint8, little bit order: bit j of byte b is individual 8*b + j), used once the sorted array
  would be larger than the bitset (more than n/32 members); 1.25 MB per 10M identities.

Both answer membership and enumerate members exactly; conversion is lossless. These functions are called only by
appliers (they write state) and by evaluator-side readers.
"""
from __future__ import annotations

import numpy as np

PREFIX = "audset."


def key(kind: str, k: int | str) -> str:
    return f"{PREFIX}{kind}.{k}"


def _dense_len(n: int) -> int:
    return (int(n) + 7) // 8


def is_dense(a: np.ndarray) -> bool:
    return a.dtype == np.uint8


def members(st, name: str) -> np.ndarray:
    """Sorted int64 members."""
    a = st.arrays.get(name)
    if a is None:
        return np.zeros(0, dtype=np.int64)
    if is_dense(a):
        return np.flatnonzero(np.unpackbits(a, bitorder="little")).astype(np.int64)
    return a.astype(np.int64)


def size(st, name: str) -> int:
    a = st.arrays.get(name)
    if a is None:
        return 0
    if is_dense(a):
        return int(np.unpackbits(a, bitorder="little").sum(dtype=np.int64))
    return int(a.size)


def contains(st, name: str, m: np.ndarray) -> np.ndarray:
    """Membership of each index in `m` (any order, repeats allowed)."""
    m = np.asarray(m, dtype=np.int64)
    a = st.arrays.get(name)
    if a is None or m.size == 0:
        return np.zeros(m.shape, dtype=bool)
    if is_dense(a):
        out = np.zeros(m.shape, dtype=bool)
        ok = m < a.size * 8
        mm = m[ok]
        out[ok] = ((a[mm >> 3] >> (mm & 7).astype(np.uint8)) & 1).astype(bool)
        return out
    if a.size == 0:
        return np.zeros(m.shape, dtype=bool)
    pos = np.searchsorted(a, m)
    pos = np.minimum(pos, a.size - 1)
    return a[pos] == m


def _sparse_dtype(n: int):
    return np.int32 if n < 2 ** 31 else np.int64


def add(st, name: str, m: np.ndarray, n_universe: int) -> np.ndarray:
    """Add the indices `m` (sorted, unique, int64) to the set; returns the mask of those that were not members.
    Journaled state write: call only from an applier."""
    m = np.asarray(m, dtype=np.int64)
    if m.size == 0:
        return np.zeros(0, dtype=bool)
    new = ~contains(st, name, m)
    nm = m[new]
    if nm.size == 0:
        return new
    a = st.arrays.get(name)
    dl = _dense_len(max(n_universe, int(nm[-1]) + 1))
    if a is not None and is_dense(a):
        if a.size < dl:                                   # the universe grew: extend the bitset (journaled)
            st.set_array(name, np.concatenate([a, np.zeros(dl - a.size, dtype=np.uint8)]))
        _set_bits(st, name, nm)
        return new
    cur = a if a is not None else np.zeros(0, dtype=_sparse_dtype(n_universe))
    if 4 * (cur.size + nm.size) > dl:                     # the bitset is now smaller: convert (lossless)
        bits = np.zeros(dl, dtype=np.uint8)
        allm = np.concatenate([cur.astype(np.int64), nm])
        np.bitwise_or.at(bits, allm >> 3, (1 << (allm & 7)).astype(np.uint8))
        st.set_array(name, bits)
        return new
    merged = np.insert(cur, np.searchsorted(cur, nm), nm.astype(cur.dtype))
    st.set_array(name, merged)
    return new


def _set_bits(st, name: str, nm: np.ndarray) -> None:
    b = nm >> 3
    bits = (1 << (nm & 7)).astype(np.uint8)
    first = np.ones(b.size, dtype=bool)
    first[1:] = b[1:] != b[:-1]
    starts = np.flatnonzero(first)
    vals = np.bitwise_or.reduceat(bits, starts)
    ub = b[starts]
    a = st.array_rows_mut(name, ub)
    a[ub] |= vals


def nbytes(st) -> int:
    return int(sum(a.nbytes for k, a in st.arrays.items() if k.startswith(PREFIX)))
