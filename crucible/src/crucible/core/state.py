"""Authoritative world state: tables of records plus dense arrays, with journaled transactions.

Rules (enforced where cheap, documented where not):
* Records are plain JSON-like dicts. `get` returns a read-only view; changes go through `put`/`patch`/`delete`,
  which journal the previous value so that a failed command rolls back exactly.
* Nested containers inside a record must be treated as immutable: build a new list/dict and `patch` it.
* Iteration is always in sorted-id order so that handlers are deterministic regardless of hash seeds.
* Dense arrays (background populations) are copied into the journal on first write inside a transaction.
* Only event appliers mutate state. Handlers read state and emit events (see kernel.py).
"""
from __future__ import annotations

import pickle
from types import MappingProxyType
from typing import Any, Iterable, Iterator

import numpy as np

from .canonical import canonical_bytes
from .errors import InvariantViolation, ProtocolError

import hashlib


def _ikey(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        raise ProtocolError("indexed fields must be scalars")
    return v


class StateStore:
    def __init__(self) -> None:
        self.tables: dict[str, dict[str, dict]] = {}
        self.index_fields: dict[str, list[str]] = {}
        self.indexes: dict[tuple[str, str], dict[Any, set[str]]] = {}
        self.arrays: dict[str, np.ndarray] = {}
        self.meta: dict[str, Any] = {}
        self._journal: list | None = None
        self._arr_saved: set[str] | None = None
        self.ctr_dirty: dict[str, int] = {}
        # 5.0 (D-21): counters allocated while a same-transaction subscriber runs (side_depth > 0) are recorded
        # separately, so the kernel can keep them off the main line (see World._commit)
        self.ctr_side: dict[str, int] = {}
        self.side_depth = 0
        # 5.0: which module declared which table (set by the kernel around Module.install; runtime only - not part of
        # snapshots or digests). Used by crucible.substrate.primitives to find duplicate shared facts.
        self.declaring: str | None = None
        self.declared_by: dict[str, list[str]] = {}
        self.array_declared_by: dict[str, str] = {}     # dense array -> the module that created it at install
        # 5.0 (wave-3b repair, SP-2.3b): metadata keys (exact, or a prefix ending in "." or ":") a module declared it
        # owns (`declare_meta` in Module.install; runtime only, like declared_by)
        self.meta_declared_by: dict[str, str] = {}

    # ------------------------------------------------------------------ schema
    def declare_table(self, name: str, index: Iterable[str] = ()) -> None:
        if self.declaring is not None:
            owners = self.declared_by.setdefault(name, [])
            if self.declaring not in owners:
                owners.append(self.declaring)
        self.tables.setdefault(name, {})
        fields = self.index_fields.setdefault(name, [])
        for f in index:
            if f in fields:
                continue
            fields.append(f)
            idx: dict[Any, set[str]] = {}
            for rid, rec in self.tables[name].items():
                idx.setdefault(_ikey(rec.get(f)), set()).add(rid)
            self.indexes[(name, f)] = idx

    def declare_meta(self, *keys: str) -> None:
        """Declare metadata keys the installing module owns: an exact key, or a prefix ending in "." or ":" (5.0,
        wave-3b repair). Keys namespaced by the module's own name or host (`<name>.x`, `<name>:x`) need no
        declaration (crucible.substrate.integrity.meta_owner)."""
        if self.declaring is None:
            raise ProtocolError("declare_meta outside Module.install")
        for k in keys:
            if not isinstance(k, str) or not k:
                raise ProtocolError(f"bad metadata key {k!r}")
            self.meta_declared_by.setdefault(k, self.declaring)

    # ------------------------------------------------------------------ reads
    def get(self, table: str, rid: str) -> MappingProxyType | None:
        rec = self.tables[table].get(rid)
        return None if rec is None else MappingProxyType(rec)

    def has(self, table: str, rid: str) -> bool:
        return rid in self.tables[table]

    def count(self, table: str) -> int:
        return len(self.tables[table])

    def ids(self, table: str) -> list[str]:
        return sorted(self.tables[table])

    def scan(self, table: str) -> Iterator[tuple[str, MappingProxyType]]:
        t = self.tables[table]
        for rid in sorted(t):
            yield rid, MappingProxyType(t[rid])

    def find_ids(self, table: str, field: str, value: Any) -> list[str]:
        idx = self.indexes.get((table, field))
        if idx is None:
            raise ProtocolError(f"no index {table}.{field}")
        return sorted(idx.get(value, ()))

    def find(self, table: str, field: str, value: Any) -> list[MappingProxyType]:
        t = self.tables[table]
        return [MappingProxyType(t[r]) for r in self.find_ids(table, field, value)]

    def find_one(self, table: str, field: str, value: Any) -> MappingProxyType | None:
        ids = self.find_ids(table, field, value)
        return MappingProxyType(self.tables[table][ids[0]]) if ids else None

    # ------------------------------------------------------------------ writes (journaled)
    def _jr(self, entry: tuple) -> None:
        if self._journal is not None:
            self._journal.append(entry)

    def put(self, table: str, rid: str, record: dict) -> None:
        t = self.tables[table]
        old = t.get(rid)
        new = dict(record)
        self._jr(("rec", table, rid, old))
        self._reindex(table, rid, old, new)
        t[rid] = new

    def insert(self, table: str, rid: str, record: dict) -> None:
        if rid in self.tables[table]:
            raise InvariantViolation(f"duplicate id {table}/{rid}")
        self.put(table, rid, record)

    def patch(self, table: str, rid: str, **changes: Any) -> dict:
        old = self.tables[table].get(rid)
        if old is None:
            raise InvariantViolation(f"patch of missing record {table}/{rid}")
        new = {**old, **changes}
        self.put(table, rid, new)
        return new

    def delete(self, table: str, rid: str) -> None:
        t = self.tables[table]
        old = t.get(rid)
        if old is None:
            return
        self._jr(("rec", table, rid, old))
        self._reindex(table, rid, old, None)
        del t[rid]

    def _reindex(self, table: str, rid: str, old: dict | None, new: dict | None) -> None:
        for f in self.index_fields.get(table, ()):
            idx = self.indexes[(table, f)]
            ov = None if old is None else old.get(f)
            nv = None if new is None else new.get(f)
            if old is not None and new is not None and ov == nv:
                continue
            if old is not None:
                s = idx.get(ov)
                if s is not None:
                    s.discard(rid)
                    if not s:
                        del idx[ov]
            if new is not None:
                idx.setdefault(_ikey(nv), set()).add(rid)

    def meta_get(self, key: str, default: Any = None) -> Any:
        return self.meta.get(key, default)

    def meta_set(self, key: str, value: Any) -> None:
        self._jr(("meta", key, None, self.meta.get(key, _MISSING)))
        self.meta[key] = value

    def next_counter(self, name: str) -> int:
        """Counters are allocated by handlers; the kernel records their final values in a `ctr` event at commit
        so that replay reproduces them exactly."""
        k = "ctr:" + name
        v = self.meta.get(k, 0) + 1
        self.meta_set(k, v)
        if self._journal is not None:
            if self.side_depth:
                self.ctr_side[k] = v
            else:
                self.ctr_dirty[k] = v
        return v

    def new_id(self, prefix: str) -> str:
        return f"{prefix}_{self.next_counter('id:' + prefix):06d}"

    def array(self, name: str) -> np.ndarray:
        a = self.arrays[name]
        v = a.view()
        v.flags.writeable = False
        return v

    def array_mut(self, name: str) -> np.ndarray:
        if self._journal is not None and name not in self._arr_saved:
            self._arr_saved.add(name)
            self._jr(("arr", name, None, self.arrays[name].copy() if name in self.arrays else _MISSING))
        return self.arrays[name]

    def array_rows_mut(self, name: str, rows: np.ndarray) -> np.ndarray:
        """5.0: the mutable array for a write that touches only `rows` (indices along axis 0). Journals just those
        rows (indices and previous values) instead of a copy of the whole array, so a cohort update of a 10M-row
        column costs O(len(rows)). Rows may repeat (the old values are taken before any write, so restoring them in
        one assignment is exact). Falls back to nothing extra when the whole array is already journaled."""
        if self._journal is not None and name not in self._arr_saved:
            a = self.arrays[name]
            r = np.array(rows, dtype=np.int64, copy=True)
            self._jr(("rows", name, r, a[r].copy()))
        return self.arrays[name]

    def set_array(self, name: str, value: np.ndarray) -> None:
        if self.declaring is not None and name not in self.arrays:
            # 5.0: arrays created while a module installs are attributed to it, like tables (runtime only)
            self.array_declared_by.setdefault(name, self.declaring)
        if self._journal is not None and name not in self._arr_saved:
            self._arr_saved.add(name)
            self._jr(("arr", name, None, self.arrays[name] if name in self.arrays else _MISSING))
        self.arrays[name] = np.ascontiguousarray(value)

    # ------------------------------------------------------------------ transactions
    @property
    def in_tx(self) -> bool:
        return self._journal is not None

    def begin(self) -> None:
        if self._journal is not None:
            raise ProtocolError("nested begin; use savepoint()")
        self._journal = []
        self._arr_saved = set()
        self.ctr_dirty = {}
        self.ctr_side = {}
        self.side_depth = 0

    def savepoint(self) -> int:
        if self._journal is None:
            raise ProtocolError("savepoint outside transaction")
        # arrays saved earlier in the transaction must be saved again: rollback_to(mark) can only restore
        # copies journaled after the mark
        self._arr_saved = set()
        return len(self._journal)

    def rollback_to(self, mark: int) -> None:
        j = self._journal
        while len(j) > mark:
            self._undo(j.pop())

    def commit(self) -> None:
        self._journal = None
        self._arr_saved = None

    def rollback(self) -> None:
        if self._journal is None:
            return
        self.rollback_to(0)
        self._journal = None
        self._arr_saved = None
        self.ctr_dirty = {}
        self.ctr_side = {}
        self.side_depth = 0

    def _undo(self, e: tuple) -> None:
        kind = e[0]
        if kind == "rec":
            _, table, rid, old = e
            cur = self.tables[table].get(rid)
            self._reindex(table, rid, cur, old)
            if old is None:
                self.tables[table].pop(rid, None)
            else:
                self.tables[table][rid] = old
        elif kind == "meta":
            _, key, _, old = e
            if old is _MISSING:
                self.meta.pop(key, None)
            else:
                self.meta[key] = old
        elif kind == "arr":
            _, name, _, old = e
            if old is _MISSING:
                self.arrays.pop(name, None)
            else:
                self.arrays[name] = old
            if self._arr_saved is not None:
                self._arr_saved.discard(name)
        elif kind == "rows":
            _, name, rows, old = e
            self.arrays[name][rows] = old

    # ------------------------------------------------------------------ snapshot / digest
    def dumps(self) -> bytes:
        if self._journal is not None:
            raise ProtocolError("snapshot inside a transaction")
        return pickle.dumps((self.tables, self.index_fields, self.arrays, self.meta), protocol=5)

    @classmethod
    def loads(cls, blob: bytes) -> "StateStore":
        s = cls()
        tables, index_fields, arrays, meta = pickle.loads(blob)
        s.tables, s.arrays, s.meta = tables, arrays, meta
        for name, fields in index_fields.items():
            s.index_fields[name] = []
            s.declare_table(name, fields)
        return s

    def digest(self) -> str:
        """Order-independent content digest of the whole state (used to verify replay and forks)."""
        h = hashlib.blake2b(digest_size=16)
        for name in sorted(self.tables):
            h.update(b"T" + name.encode())
            t = self.tables[name]
            for rid in sorted(t):
                h.update(rid.encode())
                h.update(canonical_bytes(t[rid]))
        for name in sorted(self.arrays):
            a = np.ascontiguousarray(self.arrays[name])
            h.update(b"A" + name.encode() + str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.tobytes())
        h.update(canonical_bytes({k: v for k, v in self.meta.items()}))
        return h.hexdigest()

    def table_digests(self) -> dict[str, str]:
        out = {}
        for name in sorted(self.tables):
            h = hashlib.blake2b(digest_size=8)
            t = self.tables[name]
            for rid in sorted(t):
                h.update(rid.encode())
                h.update(canonical_bytes(t[rid]))
            out[name] = h.hexdigest()
        return out


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"

    def __reduce__(self):
        return (_get_missing, ())


def _get_missing():
    return _MISSING


_MISSING = _Missing()
