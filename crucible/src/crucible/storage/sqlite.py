from __future__ import annotations

import json
import os
import sqlite3
import threading

from .base import Store, StoreConflict, dumps, now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS worlds (
  id TEXT PRIMARY KEY, spec TEXT NOT NULL, parent TEXT, created REAL NOT NULL, status TEXT NOT NULL,
  writer TEXT, lease_until REAL NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS events (
  world_id TEXT NOT NULL, seq INTEGER NOT NULL, t INTEGER NOT NULL, type TEXT NOT NULL, actor TEXT NOT NULL,
  cmd INTEGER, payload TEXT NOT NULL, vis TEXT NOT NULL, hash TEXT NOT NULL,
  PRIMARY KEY (world_id, seq)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS events_type ON events(world_id, type, seq);
CREATE TABLE IF NOT EXISTS snapshots (
  id TEXT PRIMARY KEY, world_id TEXT NOT NULL, seq INTEGER NOT NULL, hash TEXT NOT NULL, created REAL NOT NULL,
  bytes INTEGER NOT NULL, blob BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS snapshots_world ON snapshots(world_id, seq);
CREATE TABLE IF NOT EXISTS blobs (key TEXT PRIMARY KEY, data BLOB NOT NULL);
"""


class SQLiteStore(Store):
    kind = "sqlite"

    def __init__(self, path: str, synchronous: str = "NORMAL") -> None:
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._local = threading.local()
        self._sync = synchronous
        c = self._conn()
        c.executescript(_SCHEMA)
        c.commit()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(f"PRAGMA synchronous={self._sync}")
            c.execute("PRAGMA busy_timeout=30000")
            self._local.c = c
        return c

    def create_world(self, world_id, spec, parent):
        try:
            self._conn().execute("INSERT INTO worlds(id, spec, parent, created, status) VALUES (?,?,?,?,?)",
                                 (world_id, dumps(spec), dumps(parent) if parent else None, now(), "active"))
        except sqlite3.IntegrityError as e:
            raise StoreConflict(f"world {world_id} exists") from e

    def _row(self, r):
        return {"id": r[0], "spec": json.loads(r[1]), "parent": json.loads(r[2]) if r[2] else None, "created": r[3],
                "status": r[4], "writer": r[5], "lease_until": r[6]}

    def world_info(self, world_id):
        r = self._conn().execute("SELECT id, spec, parent, created, status, writer, lease_until FROM worlds WHERE id=?",
                                 (world_id,)).fetchone()
        return self._row(r) if r else None

    def list_worlds(self):
        rs = self._conn().execute("SELECT id, spec, parent, created, status, writer, lease_until FROM worlds ORDER BY id")
        return [self._row(r) for r in rs]

    def set_status(self, world_id, status):
        self._conn().execute("UPDATE worlds SET status=? WHERE id=?", (status, world_id))

    def _base(self, c, world_id):
        r = c.execute("SELECT parent FROM worlds WHERE id=?", (world_id,)).fetchone()
        if r is None:
            raise KeyError(world_id)
        return json.loads(r[0])["seq"] if r[0] else 0

    def append_events(self, world_id, events):
        if not events:
            return
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            r = c.execute("SELECT MAX(seq) FROM events WHERE world_id=?", (world_id,)).fetchone()
            last = r[0] if r[0] is not None else self._base(c, world_id)
            for i, e in enumerate(events):  # the whole batch must be contiguous, not just its first event
                if e["seq"] != last + 1 + i:
                    raise StoreConflict(f"non-contiguous append to {world_id}: {e['seq']} after {last + i}")
            c.executemany("INSERT INTO events(world_id, seq, t, type, actor, cmd, payload, vis, hash) VALUES (?,?,?,?,?,?,?,?,?)",
                          [(world_id, e["seq"], e["t"], e["type"], e["actor"], e["cmd"], dumps(e["payload"]),
                            dumps(e["vis"]), e["hash"]) for e in events])
            c.execute("COMMIT")
        except sqlite3.IntegrityError as e:
            c.execute("ROLLBACK")
            raise StoreConflict(str(e)) from e
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def read_events(self, world_id, from_seq=0, to_seq=None):
        q = "SELECT seq, t, type, actor, cmd, payload, vis, hash FROM events WHERE world_id=? AND seq>?"
        a: list = [world_id, from_seq]
        if to_seq is not None:
            q += " AND seq<=?"
            a.append(to_seq)
        q += " ORDER BY seq"
        return [{"seq": r[0], "t": r[1], "type": r[2], "actor": r[3], "cmd": r[4], "payload": json.loads(r[5]),
                 "vis": json.loads(r[6]), "hash": r[7]} for r in self._conn().execute(q, a)]

    def last_seq(self, world_id):
        c = self._conn()
        r = c.execute("SELECT MAX(seq) FROM events WHERE world_id=?", (world_id,)).fetchone()
        return r[0] if r[0] is not None else self._base(c, world_id)

    def put_snapshot(self, sid, world_id, seq, hash, blob):
        self._conn().execute("INSERT OR IGNORE INTO snapshots(id, world_id, seq, hash, created, bytes, blob) VALUES (?,?,?,?,?,?,?)",
                             (sid, world_id, seq, hash, now(), len(blob), sqlite3.Binary(blob)))

    def get_snapshot(self, sid):
        r = self._conn().execute("SELECT id, world_id, seq, hash, created, bytes, blob FROM snapshots WHERE id=?",
                                 (sid,)).fetchone()
        if r is None:
            raise KeyError(sid)
        return {"id": r[0], "world": r[1], "seq": r[2], "hash": r[3], "created": r[4], "bytes": r[5]}, bytes(r[6])

    def list_snapshots(self, world_id):
        rs = self._conn().execute("SELECT id, world_id, seq, hash, created, bytes FROM snapshots WHERE world_id=? ORDER BY seq",
                                  (world_id,))
        return [{"id": r[0], "world": r[1], "seq": r[2], "hash": r[3], "created": r[4], "bytes": r[5]} for r in rs]

    def acquire(self, world_id, owner, ttl_s=60.0):
        t = now()
        cur = self._conn().execute(
            "UPDATE worlds SET writer=?, lease_until=? WHERE id=? AND (writer IS NULL OR writer=? OR lease_until<?)",
            (owner, t + ttl_s, world_id, owner, t))
        return cur.rowcount == 1

    def release(self, world_id, owner):
        self._conn().execute("UPDATE worlds SET writer=NULL, lease_until=0 WHERE id=? AND writer=?", (world_id, owner))

    def put_blob(self, key, data):
        self._conn().execute("INSERT OR REPLACE INTO blobs(key, data) VALUES (?,?)", (key, sqlite3.Binary(data)))

    def get_blob(self, key):
        r = self._conn().execute("SELECT data FROM blobs WHERE key=?", (key,)).fetchone()
        return bytes(r[0]) if r else None

    def close(self):
        c = getattr(self._local, "c", None)
        if c is not None:
            c.close()
            self._local.c = None
