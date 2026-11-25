"""`JsonlArchive`: a durable `ArchiveP` sink so the distributed stack runs end-to-end on its own.

Workstream E's Parquet archive (``memory/archive.py``) is the lifetime store of the integrated system; the learner
accepts any `ArchiveP`, and this class is the one used by the distributed-learning benchmarks and tests until that
archive is wired in. It is a real store, not a stand-in: append-only JSON-lines chunks (``chunk-<n>.jsonl``, rotated
every ``chunk_rows`` records) plus a SQLite index (event id → chunk, byte offset, length, env, actor, episode, step,
model version, attributed_to), fsync on `flush`, an explicit `StorageLimitExceeded` at ``max_bytes`` (never silent
dropping), rejection of non-REAL provenance, and chunked streaming `scan`.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Iterator, Optional

from ..interfaces import EventRecord, Provenance

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ev(event_id TEXT PRIMARY KEY, chunk INTEGER, off INTEGER, len INTEGER, env_id TEXT,
    actor_id TEXT, episode_id TEXT, step INTEGER, model_version INTEGER, attributed_to TEXT, t_sim REAL);
CREATE INDEX IF NOT EXISTS ev_ep ON ev(actor_id, episode_id, step);
CREATE INDEX IF NOT EXISTS ev_attr ON ev(attributed_to);
"""


class StorageLimitExceeded(RuntimeError):
    pass


class JsonlArchive:
    """Implements `ArchiveP`: append / get / scan / count / flush."""

    def __init__(self, root: str, chunk_rows: int = 65536, max_bytes: int = 200 * 2**30, fsync: bool = True):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.chunk_rows = chunk_rows
        self.max_bytes = max_bytes
        self.fsync = fsync
        self.db = sqlite3.connect(os.path.join(self.root, "index.sqlite"), timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)
        r = self.db.execute("SELECT MAX(chunk), COUNT(*) FROM ev").fetchone()
        self.chunk = int(r[0]) if r[0] is not None else 0
        self.rows_in_chunk = int(self.db.execute("SELECT COUNT(*) FROM ev WHERE chunk=?", (self.chunk,)).fetchone()[0])
        self._fh = None
        self._pending_rows: list = []
        self._bytes = sum(os.path.getsize(os.path.join(self.root, f)) for f in os.listdir(self.root)
                          if f.startswith("chunk-"))
        self._truncate_unindexed()

    def _path(self, chunk: int) -> str:
        return os.path.join(self.root, f"chunk-{chunk:06d}.jsonl")

    def _truncate_unindexed(self) -> None:
        """After a crash the last chunk may hold bytes written but never indexed: cut them off."""
        p = self._path(self.chunk)
        if not os.path.exists(p):
            return
        r = self.db.execute("SELECT MAX(off + len) FROM ev WHERE chunk=?", (self.chunk,)).fetchone()
        end = int(r[0]) if r[0] is not None else 0
        if os.path.getsize(p) > end:
            self._bytes -= os.path.getsize(p) - end
            with open(p, "r+b") as fh:
                fh.truncate(end)

    def _open(self):
        if self._fh is None:
            self._fh = open(self._path(self.chunk), "ab")
        return self._fh

    def append(self, records: list) -> None:
        for r in records:
            if r.provenance != Provenance.REAL:
                raise ValueError(f"archive accepts only REAL experience, got {r.provenance} ({r.event_id})")
        rows = []
        for r in records:
            if self.rows_in_chunk >= self.chunk_rows:
                if self._fh is not None:
                    self._fh.flush()
                    if self.fsync:
                        os.fsync(self._fh.fileno())
                    self._fh.close()
                    self._fh = None
                self.chunk += 1
                self.rows_in_chunk = 0
            line = ((r.__dict__.get("_wire") or r.to_json()) + "\n").encode()
            if self._bytes + len(line) > self.max_bytes:
                raise StorageLimitExceeded(f"archive {self.root} would exceed {self.max_bytes} bytes")
            fh = self._open()
            off = fh.tell()
            fh.write(line)
            self._bytes += len(line)
            self.rows_in_chunk += 1
            rows.append((r.event_id, self.chunk, off, len(line), r.env_id, r.actor_id, r.episode_id, r.step,
                         r.model_version, r.attributed_to, r.t_sim))
        self._pending_rows.extend(rows)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())
        rows = self._pending_rows
        if rows:
            with self.db:
                self.db.executemany("INSERT OR IGNORE INTO ev VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._pending_rows = []

    def _read(self, chunk: int, off: int, ln: int) -> EventRecord:
        if self._fh is not None and chunk == self.chunk:
            self._fh.flush()
        with open(self._path(chunk), "rb") as fh:
            fh.seek(off)
            return EventRecord.from_dict(json.loads(fh.read(ln)))

    def get(self, event_ids: list) -> list:
        """-> list aligned with event_ids (None where unknown)."""
        self.flush()
        out = []
        for eid in event_ids:
            r = self.db.execute("SELECT chunk, off, len FROM ev WHERE event_id=?", (eid,)).fetchone()
            out.append(self._read(*r) if r else None)
        return out

    def scan(self, batch: int = 1024, actor_id: Optional[str] = None, episode_id: Optional[str] = None,
             env_id: Optional[str] = None, min_version: Optional[int] = None,
             attributed_to: Optional[str] = None, **_ignored) -> Iterator[list]:
        """Stream records in insertion order, `batch` at a time, optionally filtered."""
        self.flush()
        where, args = [], []
        for col, v in (("actor_id", actor_id), ("episode_id", episode_id), ("env_id", env_id),
                       ("attributed_to", attributed_to)):
            if v is not None:
                where.append(f"{col}=?")
                args.append(v)
        if min_version is not None:
            where.append("model_version>=?")
            args.append(min_version)
        q = "SELECT chunk, off, len FROM ev" + (" WHERE " + " AND ".join(where) if where else "") + \
            " ORDER BY chunk, off"
        cur = self.db.execute(q, args)
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                return
            yield [self._read(*r) for r in rows]

    def count(self) -> int:
        self.flush()
        return int(self.db.execute("SELECT COUNT(*) FROM ev").fetchone()[0])

    def bytes(self) -> int:
        return self._bytes

    def close(self) -> None:
        self.flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self.db.close()


def make_jsonl_archive(root: str, **kw) -> JsonlArchive:
    """Factory usable by module path (``superlearner.cam_unified.dist.sinks:make_jsonl_archive``)."""
    return JsonlArchive(root, **kw)
