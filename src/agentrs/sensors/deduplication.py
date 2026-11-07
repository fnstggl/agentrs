"""Stable identities for observed actions.

* within a source: the action id is a hash of (source id, provider object id), so re-polling or a
  restarted collector never creates a second record;
* across sources: the same content by the same (pseudonymous) actor within a time window is the
  same action seen twice (e.g. a feed and a dataset export) — the later sighting is linked, not
  re-ingested;
* our own actions: registered with their provider object id when executed, so the public feed's
  report of our own post is recognised as ours and never re-labelled as a third-party action.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path


def stable_action_id(source_id: str, provider_object_id: str) -> str:
    return hashlib.sha256(f"{source_id}\x00{provider_object_id}".encode()).hexdigest()[:32]


class Deduplicator:
    def __init__(self, path: str | Path, window_s: float = 3600.0):
        self.window_s = window_s
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("CREATE TABLE IF NOT EXISTS seen (action_id TEXT PRIMARY KEY, content_hash TEXT, actor_ref TEXT,"
                         " at REAL, env_key TEXT)")
        self._db.execute("CREATE INDEX IF NOT EXISTS seen_hash ON seen(content_hash)")
        self._db.execute("CREATE TABLE IF NOT EXISTS aliases (env_key TEXT PRIMARY KEY, action_id TEXT)")
        self.stats = {"new": 0, "same_source_duplicate": 0, "cross_source_duplicate": 0, "own_action": 0}

    def register_alias(self, env_key: str, action_id: str) -> None:
        """env_key: '<environment instance>:<object id>' (e.g. our own post's item id)."""
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO aliases VALUES (?,?)", (env_key, action_id))

    def alias(self, env_key: str) -> str | None:
        with self._lock:
            r = self._db.execute("SELECT action_id FROM aliases WHERE env_key=?", (env_key,)).fetchone()
        return r[0] if r else None

    def check(self, action_id: str, content_hash: str, actor_ref: str | None, at: float,
              env_key: str | None = None) -> tuple[str, str | None]:
        """Returns (verdict, existing action id). verdict: new | same_source_duplicate |
        cross_source_duplicate | own_action. Registers the action when new."""
        with self._lock:
            if env_key is not None:
                own = self._db.execute("SELECT action_id FROM aliases WHERE env_key=?", (env_key,)).fetchone()
                if own and own[0] != action_id:  # registered by one of our own actions
                    self.stats["own_action"] += 1
                    return "own_action", own[0]
            if self._db.execute("SELECT 1 FROM seen WHERE action_id=?", (action_id,)).fetchone():
                self.stats["same_source_duplicate"] += 1
                return "same_source_duplicate", action_id
            r = self._db.execute("SELECT action_id FROM seen WHERE content_hash=? AND actor_ref IS ? AND ABS(at-?)<=?",
                                 (content_hash, actor_ref, at, self.window_s)).fetchone()
            if r:
                self.stats["cross_source_duplicate"] += 1
                return "cross_source_duplicate", r[0]
            self._db.execute("INSERT INTO seen VALUES (?,?,?,?,?)", (action_id, content_hash, actor_ref, at, env_key))
            if env_key is not None:
                self._db.execute("INSERT OR IGNORE INTO aliases VALUES (?,?)", (env_key, action_id))
            self.stats["new"] += 1
            return "new", None

    def forget(self, action_ids) -> None:
        with self._lock:
            for a in action_ids:
                self._db.execute("DELETE FROM seen WHERE action_id=?", (a,))
                self._db.execute("DELETE FROM aliases WHERE action_id=?", (a,))
