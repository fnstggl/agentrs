"""Durable event store interface.

The store is append-only for events: `(world_id, seq)` is unique and appends must be contiguous, so two kernel
processes that both believe they own a world cannot both commit seq N (the second append fails with
`StoreConflict`). Snapshots are immutable blobs addressed by id. A world records its parent (world, seq, hash)
when it was forked, which gives a lineage tree of experiments.

Backends: `MemoryStore` (tests, fast RL), `SQLiteStore` (default, single file, WAL), `PostgresStore`
(multi-process, the primary backend for long-running canonical worlds). All three run the same conformance
tests (tests/test_storage.py).
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
        out = []
        cur = self.world_info(world_id)
        while cur is not None:
            out.append(cur)
            par = cur.get("parent")
            cur = self.world_info(par["world"]) if par else None
        return out

    def close(self) -> None:
        pass


def dumps(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def now() -> float:
    return time.time()
