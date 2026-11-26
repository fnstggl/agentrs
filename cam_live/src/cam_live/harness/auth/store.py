"""Persistent state of the authorization service: one SQLite file (WAL) under ``<run_dir>/auth/``.

Only the authorization service's uid can open it (directory 0700, file 0600). Every table is small and keyed for
the lookups the policy path makes on each decision.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Iterable, List, Optional, Sequence

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, task_hash TEXT NOT NULL, mode TEXT NOT NULL,
    task_json TEXT NOT NULL, pinned_at REAL NOT NULL, pinned_by TEXT NOT NULL, started_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS live_records (record_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, record_json TEXT NOT NULL,
    registered_at REAL NOT NULL, registered_by TEXT NOT NULL, revoked_at REAL);
CREATE TABLE IF NOT EXISTS grants (nonce TEXT PRIMARY KEY, action_hash TEXT NOT NULL, task_id TEXT NOT NULL,
    issued_at REAL NOT NULL, expires_at REAL NOT NULL, epoch INTEGER NOT NULL, status TEXT NOT NULL,
    used_at REAL, exec_until REAL, closed_at REAL, approval_id TEXT NOT NULL DEFAULT '',
    reserve_json TEXT NOT NULL DEFAULT '{}', effects_json TEXT NOT NULL DEFAULT '[]', n_primitives INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS grants_by_hash ON grants(action_hash, status);
CREATE INDEX IF NOT EXISTS grants_by_task ON grants(task_id, status);
CREATE TABLE IF NOT EXISTS approvals (approval_id TEXT PRIMARY KEY, bind_hash TEXT NOT NULL, action_hash TEXT NOT NULL,
    task_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL,
    resolved_at REAL, resolved_by TEXT NOT NULL DEFAULT '', note TEXT NOT NULL DEFAULT '', preview_json TEXT NOT NULL,
    action_json TEXT NOT NULL, reasons_json TEXT NOT NULL, policy_ids_json TEXT NOT NULL, consumed_at REAL);
CREATE INDEX IF NOT EXISTS approvals_by_bind ON approvals(bind_hash, created_at);
CREATE TABLE IF NOT EXISTS usage (task_id TEXT NOT NULL, k TEXT NOT NULL, v REAL NOT NULL, PRIMARY KEY (task_id, k));
CREATE TABLE IF NOT EXISTS usage_daily (task_id TEXT NOT NULL, day INTEGER NOT NULL, k TEXT NOT NULL, v REAL NOT NULL,
    PRIMARY KEY (task_id, day, k));
CREATE TABLE IF NOT EXISTS rate_events (id INTEGER PRIMARY KEY AUTOINCREMENT, k TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS rate_by_key ON rate_events(k, ts);
CREATE TABLE IF NOT EXISTS denied_intents (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
    target TEXT NOT NULL, effect TEXT NOT NULL, needles_json TEXT NOT NULL, operation TEXT NOT NULL,
    modality TEXT NOT NULL, policy_ids_json TEXT NOT NULL, action_hash TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS denied_by_task ON denied_intents(task_id, target);
CREATE TABLE IF NOT EXISTS content_log (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, kind TEXT NOT NULL,
    target TEXT NOT NULL, shingles_json TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS content_by_task ON content_log(task_id, kind, ts);
CREATE TABLE IF NOT EXISTS revocations (id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, by TEXT NOT NULL,
    ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS session (task_id TEXT NOT NULL, k TEXT NOT NULL, v TEXT NOT NULL, PRIMARY KEY (task_id, k));
"""


class Store:
    """Thin, thread-safe wrapper. Every write is committed before the call returns."""

    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(path)
        os.makedirs(d, mode=0o700, exist_ok=True)
        old = os.umask(0o077)
        try:
            self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30.0)
        finally:
            os.umask(old)
        self._lock = threading.RLock()
        self._depth, self._failed = 0, False
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_DDL)
        v = self.kv_get("schema_version")
        if v is None:
            self.kv_set("schema_version", str(SCHEMA_VERSION))
        elif int(v) != SCHEMA_VERSION:
            raise RuntimeError(f"authorization state schema {v} is not supported (expected {SCHEMA_VERSION})")

    # -- primitives ------------------------------------------------------------------------------------------
    def x(self, sql: str, args: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, tuple(args))

    def q(self, sql: str, args: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    def one(self, sql: str, args: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchone()

    def tx(self):
        return _Tx(self)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- small helpers ---------------------------------------------------------------------------------------
    def kv_get(self, k: str) -> Optional[str]:
        r = self.one("SELECT v FROM kv WHERE k=?", (k,))
        return None if r is None else r["v"]

    def kv_set(self, k: str, v: str) -> None:
        self.x("INSERT INTO kv(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))

    def kv_int(self, k: str, default: int = 0) -> int:
        v = self.kv_get(k)
        return default if v is None else int(v)


class _Tx:
    """``with store.tx():`` — one atomic, immediately locked transaction (holds the thread lock throughout).
    Re-entrant: an inner ``tx()`` joins the outer transaction."""

    def __init__(self, store: Store):
        self.s = store

    def __enter__(self):
        self.s._lock.acquire()
        if self.s._depth == 0:
            self.s._db.execute("BEGIN IMMEDIATE")
        self.s._depth += 1
        return self.s

    def __exit__(self, et, ev, tb):
        try:
            self.s._depth -= 1
            if self.s._depth == 0:
                self.s._db.execute("ROLLBACK" if (et or self.s._failed) else "COMMIT")
                self.s._failed = False
            elif et:
                self.s._failed = True          # the outermost level rolls everything back
        finally:
            self.s._lock.release()
        return False


def j(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def uj(text: Optional[str], default: Any = None) -> Any:
    if not text:
        return default
    return json.loads(text)


def in_clause(items: Iterable[Any]) -> str:
    return "(" + ",".join("?" for _ in items) + ")"
