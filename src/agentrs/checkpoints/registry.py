"""Versioned checkpoint registry with evaluation-gated promotion and rollback (SQLite)."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, BaseModel


@dataclass
class CheckpointRecord:
    id: str
    component: str
    path: str
    parent: Optional[str]
    created: float
    status: str
    metrics: dict
    notes: str


class PromotionGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """Promote only if the candidate is not worse on held-out validation, does not regress
    on any regression-suite family by more than ``max_family_regression``, and keeps the
    invalid-action rate under ``max_invalid_rate``."""

    min_improvement: float = 0.0
    max_family_regression: float = 0.2
    max_invalid_rate: float = 0.3
    metric: str = "success_rate"

    def decide(self, cand: dict, prod: Optional[dict]) -> tuple[bool, str]:
        if cand.get("invalid_action_rate", 0.0) > self.max_invalid_rate:
            return False, f"invalid-action rate {cand['invalid_action_rate']:.2f} > {self.max_invalid_rate}"
        if prod is None:
            return True, "no production checkpoint yet"
        delta = cand[self.metric] - prod[self.metric]
        if delta < self.min_improvement:
            return False, f"{self.metric} {cand[self.metric]:.3f} vs production {prod[self.metric]:.3f}"
        for fam, v in prod.get("by_family", {}).items():
            c = cand.get("by_family", {}).get(fam)
            if c is not None and v - c > self.max_family_regression:
                return False, f"regression on {fam}: {c:.2f} vs {v:.2f}"
        return True, f"{self.metric} improved by {delta:+.3f}"


class CheckpointRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS checkpoints (id TEXT PRIMARY KEY, component TEXT, path TEXT,
                         parent TEXT, created REAL, status TEXT, metrics TEXT, notes TEXT)""")
            c.execute("CREATE TABLE IF NOT EXISTS production (component TEXT PRIMARY KEY, checkpoint_id TEXT, since REAL)")
            c.execute("CREATE TABLE IF NOT EXISTS events (time REAL, component TEXT, checkpoint_id TEXT, action TEXT, detail TEXT)")

    def _c(self):
        return sqlite3.connect(self.path, timeout=30)

    def _event(self, c, component, cid, action, detail=""):
        c.execute("INSERT INTO events VALUES (?,?,?,?,?)", (time.time(), component, cid, action, detail))

    def register(self, component: str, path: str, parent: Optional[str] = None, metrics: Optional[dict] = None,
                 notes: str = "") -> str:
        cid = f"{component}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        with self._c() as c:
            c.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)",
                      (cid, component, str(path), parent, time.time(), "candidate", json.dumps(metrics or {}), notes))
            self._event(c, component, cid, "register")
        return cid

    def get(self, cid: str) -> CheckpointRecord:
        with self._c() as c:
            r = c.execute("SELECT * FROM checkpoints WHERE id=?", (cid,)).fetchone()
        if r is None:
            raise KeyError(cid)
        return CheckpointRecord(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6] or "{}"), r[7])

    def update_metrics(self, cid: str, metrics: dict) -> None:
        rec = self.get(cid)
        with self._c() as c:
            c.execute("UPDATE checkpoints SET metrics=? WHERE id=?", (json.dumps({**rec.metrics, **metrics}), cid))

    def production(self, component: str) -> Optional[CheckpointRecord]:
        with self._c() as c:
            r = c.execute("SELECT checkpoint_id FROM production WHERE component=?", (component,)).fetchone()
        return self.get(r[0]) if r else None

    def promote(self, cid: str, detail: str = "") -> None:
        rec = self.get(cid)
        with self._c() as c:
            prev = c.execute("SELECT checkpoint_id FROM production WHERE component=?", (rec.component,)).fetchone()
            if prev:
                c.execute("UPDATE checkpoints SET status='retired' WHERE id=?", (prev[0],))
            c.execute("INSERT OR REPLACE INTO production VALUES (?,?,?)", (rec.component, cid, time.time()))
            c.execute("UPDATE checkpoints SET status='promoted' WHERE id=?", (cid,))
            self._event(c, rec.component, cid, "promote", detail)

    def reject(self, cid: str, reason: str = "") -> None:
        rec = self.get(cid)
        with self._c() as c:
            c.execute("UPDATE checkpoints SET status='rejected', notes=? WHERE id=?", (reason, cid))
            self._event(c, rec.component, cid, "reject", reason)

    def promotion_history(self, component: str) -> list[str]:
        with self._c() as c:
            rows = c.execute("SELECT checkpoint_id FROM events WHERE component=? AND action='promote' ORDER BY time",
                             (component,)).fetchall()
        return [r[0] for r in rows]

    def rollback(self, component: str, reason: str = "manual rollback") -> Optional[str]:
        """Restore the previously promoted checkpoint. Returns its id (None if nothing to restore)."""
        hist = self.promotion_history(component)
        cur = self.production(component)
        if cur is None:
            return None
        prev = [h for h in hist if h != cur.id]
        if not prev:
            return None
        target = prev[-1]
        with self._c() as c:
            c.execute("UPDATE checkpoints SET status='rolled_back' WHERE id=?", (cur.id,))
            c.execute("INSERT OR REPLACE INTO production VALUES (?,?,?)", (component, target, time.time()))
            c.execute("UPDATE checkpoints SET status='promoted' WHERE id=?", (target,))
            self._event(c, component, target, "rollback", f"from {cur.id}: {reason}")
        return target

    def list(self, component: Optional[str] = None) -> list[CheckpointRecord]:
        q = "SELECT * FROM checkpoints" + (" WHERE component=?" if component else "") + " ORDER BY created"
        with self._c() as c:
            rows = c.execute(q, (component,) if component else ()).fetchall()
        return [CheckpointRecord(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6] or "{}"), r[7]) for r in rows]

    def events(self) -> list[tuple]:
        with self._c() as c:
            return c.execute("SELECT * FROM events ORDER BY time").fetchall()
