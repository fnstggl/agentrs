"""PostgreSQL event store (psycopg 3). Same contract as SQLiteStore; JSON payloads, BYTEA snapshots.

Payloads, visibility, specs and parents are stored as `json` (verbatim text), not `jsonb`: jsonb normalises
numbers (-0.0 becomes 0) and rejects NUL characters and unpaired surrogates, so a reloaded event would no longer hash to
its recorded chain hash (or could not be stored at all). Cast to jsonb in ad-hoc queries when needed.

Concurrency: appends run in a transaction that takes a per-world advisory lock and checks contiguity, so
concurrent writers to the same world serialise and the loser gets `StoreConflict`. The primary key
(world_id, seq) is the final guard.
"""
from __future__ import annotations

import json
import threading

from .base import Store, StoreConflict, dumps, now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS crucible_worlds (
  id TEXT PRIMARY KEY, spec JSON NOT NULL, parent JSON, created DOUBLE PRECISION NOT NULL, status TEXT NOT NULL,
  writer TEXT, lease_until DOUBLE PRECISION NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS crucible_events (
  world_id TEXT NOT NULL, seq BIGINT NOT NULL, t BIGINT NOT NULL, type TEXT NOT NULL, actor TEXT NOT NULL,
  cmd BIGINT, payload JSON NOT NULL, vis JSON NOT NULL, hash TEXT NOT NULL, PRIMARY KEY (world_id, seq));
CREATE INDEX IF NOT EXISTS crucible_events_type ON crucible_events(world_id, type, seq);
CREATE TABLE IF NOT EXISTS crucible_snapshots (
  id TEXT PRIMARY KEY, world_id TEXT NOT NULL, seq BIGINT NOT NULL, hash TEXT NOT NULL,
  created DOUBLE PRECISION NOT NULL, bytes BIGINT NOT NULL, blob BYTEA NOT NULL);
CREATE TABLE IF NOT EXISTS crucible_blobs (key TEXT PRIMARY KEY, data BYTEA NOT NULL);
ALTER TABLE crucible_events ADD COLUMN IF NOT EXISTS via TEXT;
"""


class PostgresStore(Store):
    kind = "postgres"

    def __init__(self, dsn: str) -> None:
        import psycopg  # noqa: F401  (optional dependency)
        self.dsn = dsn
        self._local = threading.local()
        with self._conn().transaction():
            # serialise schema creation: concurrent CREATE TABLE IF NOT EXISTS from several processes can fail
            self._conn().execute("SELECT pg_advisory_xact_lock(hashtext('crucible_schema'))")
            self._conn().execute(_SCHEMA)

    def _conn(self):
        import psycopg
        c = getattr(self._local, "c", None)
        if c is None or c.closed:
            c = psycopg.connect(self.dsn, autocommit=True)
            self._local.c = c
        return c

    def create_world(self, world_id, spec, parent):
        import psycopg
        try:
            self._conn().execute("INSERT INTO crucible_worlds(id, spec, parent, created, status) VALUES (%s,%s,%s,%s,%s)",
                                 (world_id, dumps(spec), dumps(parent) if parent else None, now(), "active"))
        except psycopg.errors.UniqueViolation as e:
            raise StoreConflict(f"world {world_id} exists") from e

    def _row(self, r):
        return {"id": r[0], "spec": r[1], "parent": r[2], "created": r[3], "status": r[4], "writer": r[5],
                "lease_until": r[6]}

    def world_info(self, world_id):
        r = self._conn().execute("SELECT id, spec, parent, created, status, writer, lease_until FROM crucible_worlds WHERE id=%s",
                                 (world_id,)).fetchone()
        return self._row(r) if r else None

    def list_worlds(self):
        rs = self._conn().execute("SELECT id, spec, parent, created, status, writer, lease_until FROM crucible_worlds ORDER BY id")
        return [self._row(r) for r in rs]

    def set_status(self, world_id, status):
        self._conn().execute("UPDATE crucible_worlds SET status=%s WHERE id=%s", (status, world_id))

    def _base(self, c, world_id):
        r = c.execute("SELECT parent FROM crucible_worlds WHERE id=%s", (world_id,)).fetchone()
        if r is None:
            raise KeyError(world_id)
        return r[0]["seq"] if r[0] else 0

    def append_events(self, world_id, events):
        import psycopg
        if not events:
            return
        c = self._conn()
        try:
            with c.transaction():
                c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (world_id,))
                r = c.execute("SELECT MAX(seq) FROM crucible_events WHERE world_id=%s", (world_id,)).fetchone()
                last = r[0] if r[0] is not None else self._base(c, world_id)
                for i, e in enumerate(events):  # the whole batch must be contiguous, not just its first event
                    if e["seq"] != last + 1 + i:
                        raise StoreConflict(f"non-contiguous append to {world_id}: {e['seq']} after {last + i}")
                with c.cursor() as cur:
                    with cur.copy("COPY crucible_events (world_id, seq, t, type, actor, cmd, payload, vis, hash, via) "
                                  "FROM STDIN") as cp:
                        for e in events:
                            cp.write_row((world_id, e["seq"], e["t"], e["type"], e["actor"], e["cmd"],
                                          dumps(e["payload"]), dumps(e["vis"]), e["hash"], e.get("via")))
        except psycopg.errors.UniqueViolation as e:
            raise StoreConflict(str(e)) from e

    def read_events(self, world_id, from_seq=0, to_seq=None, types=None):
        q = ("SELECT seq, t, type, actor, cmd, payload, vis, hash, via FROM crucible_events "
             "WHERE world_id=%s AND seq>%s")
        a: list = [world_id, from_seq]
        if to_seq is not None:
            q += " AND seq<=%s"
            a.append(to_seq)
        if types is not None:
            q += " AND type = ANY(%s)"
            a.append(sorted(set(types)))
        q += " ORDER BY seq"
        out = []
        for r in self._conn().execute(q, a):
            d = {"seq": r[0], "t": r[1], "type": r[2], "actor": r[3], "cmd": r[4], "payload": r[5], "vis": r[6],
                 "hash": r[7]}
            if r[8] is not None:
                d["via"] = r[8]
            out.append(d)
        return out

    def last_seq(self, world_id):
        c = self._conn()
        r = c.execute("SELECT MAX(seq) FROM crucible_events WHERE world_id=%s", (world_id,)).fetchone()
        return r[0] if r[0] is not None else self._base(c, world_id)

    def put_snapshot(self, sid, world_id, seq, hash, blob):
        self._conn().execute("INSERT INTO crucible_snapshots(id, world_id, seq, hash, created, bytes, blob) VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                             (sid, world_id, seq, hash, now(), len(blob), blob))

    def get_snapshot(self, sid):
        r = self._conn().execute("SELECT id, world_id, seq, hash, created, bytes, blob FROM crucible_snapshots WHERE id=%s",
                                 (sid,)).fetchone()
        if r is None:
            raise KeyError(sid)
        return {"id": r[0], "world": r[1], "seq": r[2], "hash": r[3], "created": r[4], "bytes": r[5]}, bytes(r[6])

    def list_snapshots(self, world_id):
        rs = self._conn().execute("SELECT id, world_id, seq, hash, created, bytes FROM crucible_snapshots WHERE world_id=%s ORDER BY seq",
                                  (world_id,))
        return [{"id": r[0], "world": r[1], "seq": r[2], "hash": r[3], "created": r[4], "bytes": r[5]} for r in rs]

    def acquire(self, world_id, owner, ttl_s=60.0):
        t = now()
        cur = self._conn().execute(
            "UPDATE crucible_worlds SET writer=%s, lease_until=%s WHERE id=%s AND (writer IS NULL OR writer=%s OR lease_until<%s)",
            (owner, t + ttl_s, world_id, owner, t))
        return cur.rowcount == 1

    def release(self, world_id, owner):
        self._conn().execute("UPDATE crucible_worlds SET writer=NULL, lease_until=0 WHERE id=%s AND writer=%s",
                             (world_id, owner))

    def put_blob(self, key, data):
        self._conn().execute("INSERT INTO crucible_blobs(key, data) VALUES (%s,%s) ON CONFLICT (key) DO UPDATE SET data=EXCLUDED.data",
                             (key, data))

    def get_blob(self, key):
        r = self._conn().execute("SELECT data FROM crucible_blobs WHERE key=%s", (key,)).fetchone()
        return bytes(r[0]) if r else None

    def delete_blob(self, key):
        self._conn().execute("DELETE FROM crucible_blobs WHERE key=%s", (key,))

    def blob_sizes(self, prefix=""):
        rs = self._conn().execute("SELECT key, octet_length(data) FROM crucible_blobs WHERE starts_with(key, %s) "
                                  "ORDER BY key", (prefix,))
        return {r[0]: int(r[1]) for r in rs}

    def close(self):
        c = getattr(self._local, "c", None)
        if c is not None and not c.closed:
            c.close()


def open_store(url: str) -> Store:
    """`memory://`, `sqlite:///path/to.db`, `postgresql://...`"""
    if url.startswith("memory:"):
        from .memory import MemoryStore
        return MemoryStore()
    if url.startswith("sqlite:"):
        # SQLAlchemy convention: sqlite:///relative/path.db, sqlite:////absolute/path.db, sqlite:///:memory:
        from .sqlite import SQLiteStore
        if url.startswith("sqlite:///"):
            path = url[len("sqlite:///"):]
        elif url.startswith("sqlite://"):
            path = url[len("sqlite://"):]
        else:
            path = url[len("sqlite:"):]
        return SQLiteStore(path or ":memory:")
    if url.startswith(("postgres://", "postgresql://")):
        return PostgresStore(url)
    raise ValueError(f"unknown store url {url!r}")
