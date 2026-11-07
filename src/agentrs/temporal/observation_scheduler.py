"""Durable, permission-aware scheduling of follow-up observations.

Jobs live in SQLite (survive process restarts) and are executed at-least-once with idempotent
effects (the ledger deduplicates observations by job id):

* **fixed horizons** — one job per configured horizon of the action's profile (idempotency key
  ``action:h:<label>``), with deterministic jitter;
* **event-driven** — ``on_event`` schedules an immediate check when an authorised feed reports
  activity, coalesced per time bucket so a burst of events costs one observation;
* **adaptive** — ``AdaptivePolicy`` uses the forecast stored for the action: it skips a fixed check
  whose predicted information gain is negligible (low predicted change and low uncertainty) and adds
  checks around the predicted median time of uncertain events;
* **budgets** — token bucket per source in the source's clock; a job that would exceed its source's
  budget stays pending; permissions are checked by the executor's fetcher (never bypassed here);
* **leases** — a claimed job is leased for ``lease_s`` wall seconds; expired leases (crashed worker)
  are recovered; failures retry with exponential backoff + jitter until ``max_attempts``, after
  which the horizon is recorded as a *missing* observation (→ censored label, never a failure);
* cancellation (deleted content, revoked source, closed window) and retention purging.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentrs.temporal.time_encoding import describe_duration

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, action_id TEXT NOT NULL, source_id TEXT NOT NULL, kind TEXT NOT NULL, horizon_label TEXT,
  horizon_s REAL, due_time REAL NOT NULL, time_base TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER DEFAULT 0,
  max_attempts INTEGER, next_attempt_time REAL, lease_owner TEXT, lease_until REAL, idem TEXT UNIQUE, priority REAL,
  created_wall REAL, completed_wall REAL, result_obs_id TEXT, error TEXT, provenance_json TEXT);
CREATE INDEX IF NOT EXISTS jobs_due ON jobs(status, due_time);
CREATE INDEX IF NOT EXISTS jobs_action ON jobs(action_id);
CREATE TABLE IF NOT EXISTS budgets (
  source_id TEXT PRIMARY KEY, capacity REAL, refill_per_s REAL, tokens REAL, updated_time REAL, spent REAL DEFAULT 0);
"""


class SchedulerConfig(BaseModel):
    max_attempts: int = 4
    backoff_base_s: float = 30.0
    backoff_max_s: float = 6 * 3600.0
    jitter_frac: float = 0.05  # +-5% of the horizon (deterministic per job)
    lease_s: float = 60.0
    event_coalesce_s: float = 60.0
    seed: int = 0


class AdaptivePolicy(BaseModel):
    """Decide which fixed checks are worth their cost, from the action's stored forecast."""

    min_change_prob: float = 0.03  # skip a check if every event's predicted P(change in interval) is below this
    max_uncertainty: float = 0.05  # ... and the ensemble std is below this
    add_around_median: bool = True
    keep_last: bool = True  # always observe at the end of the profile (window closure)

    def plan(self, horizons: Sequence[float], prediction: dict) -> tuple[list[float], list[float]]:
        """prediction: TemporalPrediction.to_dict() evaluated at ``horizons``.
        Returns (kept horizons, extra horizons)."""
        hp = {round(p["horizon_s"], 3): p for p in prediction.get("horizons", [])}
        kept, prev = [], None
        hs = sorted(horizons)
        for i, h in enumerate(hs):
            p = hp.get(round(h, 3))
            if p is None or (self.keep_last and i == len(hs) - 1) or prev is None:
                kept.append(h)
                prev = p
                continue
            change = max((p["event_prob"][k] - prev["event_prob"].get(k, 0.0)) for k in p["event_prob"])
            unc = max(p["event_std"].values()) if p["event_std"] else 0.0
            cnt_change = max([abs(p["count_mean"][k] - prev["count_mean"].get(k, 0.0)) for k in p["count_mean"]] + [0.0])
            if change < self.min_change_prob and unc < self.max_uncertainty and cnt_change < 0.5:
                continue
            kept.append(h)
            prev = p
        extra = []
        if self.add_around_median:
            for k, med in (prediction.get("median_time_s") or {}).items():
                ever = (prediction.get("event_ever") or {}).get(k, 0.0)
                if med and 0.2 < ever < 0.8 and hs and hs[0] < med < hs[-1]:
                    extra.append(float(med))
        return kept, extra


class TemporalObservationScheduler:
    def __init__(self, path: str | Path, cfg: SchedulerConfig | None = None):
        self.cfg = cfg or SchedulerConfig()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _q(self, sql, args=()):
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    def _x(self, sql, args=()):
        with self._lock:
            return self._db.execute(sql, tuple(args)).rowcount

    def _jitter(self, key: str, span: float) -> float:
        u = int(hashlib.md5(f"{self.cfg.seed}|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return (2 * u - 1) * span

    # ------------------------------------------------------------------ scheduling
    def _insert(self, action_id, source_id, kind, label, horizon_s, due, time_base, idem, priority=0.0,
                provenance=None) -> str | None:
        jid = uuid.uuid4().hex
        n = self._x("INSERT OR IGNORE INTO jobs (job_id, action_id, source_id, kind, horizon_label, horizon_s, due_time, "
                    "time_base, status, attempts, max_attempts, next_attempt_time, idem, priority, created_wall, "
                    "provenance_json) VALUES (?,?,?,?,?,?,?,?,'pending',0,?,?,?,?,?,?)",
                    (jid, action_id, source_id, kind, label, horizon_s, due, time_base, self.cfg.max_attempts, due, idem,
                     priority, time.time(), json.dumps(provenance or {})))
        return jid if n else None

    def schedule_horizons(self, action_id: str, source_id: str, action_time: float, horizons: Sequence[float],
                          time_base: str, kind: str = "horizon", provenance: dict | None = None) -> list[str]:
        out = []
        for h in horizons:
            label = describe_duration(h)
            due = action_time + h + (self._jitter(f"{action_id}:{label}", self.cfg.jitter_frac * h) if h > 0 else 0.0)
            j = self._insert(action_id, source_id, kind, label, float(h), max(action_time, due), time_base,
                             f"{action_id}:h:{label}", provenance=provenance)
            if j:
                out.append(j)
        return out

    def schedule_adaptive(self, action_id: str, source_id: str, action_time: float, horizons: Sequence[float],
                          time_base: str, prediction: dict, policy: AdaptivePolicy) -> dict:
        kept, extra = policy.plan(horizons, prediction)
        ids = self.schedule_horizons(action_id, source_id, action_time, kept, time_base)
        ids += self.schedule_horizons(action_id, source_id, action_time, extra, time_base, kind="adaptive")
        return {"kept": kept, "skipped": [h for h in horizons if h not in kept], "extra": extra, "jobs": ids}

    def on_event(self, action_id: str, source_id: str, at: float, time_base: str) -> str | None:
        """An authorised feed reported activity: observe now (coalesced per bucket)."""
        bucket = int(at // max(1.0, self.cfg.event_coalesce_s))
        return self._insert(action_id, source_id, "event", f"event@{bucket}", None, at, time_base,
                            f"{action_id}:ev:{bucket}", priority=1.0)

    # ------------------------------------------------------------------ budgets
    def set_budget(self, source_id: str, capacity: float, refill_per_s: float, now: float) -> None:
        self._x("INSERT INTO budgets VALUES (?,?,?,?,?,0) ON CONFLICT(source_id) DO UPDATE SET capacity=excluded.capacity,"
                " refill_per_s=excluded.refill_per_s", (source_id, capacity, refill_per_s, capacity, now))

    def _take_token(self, source_id: str, now: float) -> bool:
        r = self._q("SELECT * FROM budgets WHERE source_id=?", (source_id,))
        if not r:
            return True  # no budget configured for this source
        b = r[0]
        tokens = min(b["capacity"], b["tokens"] + max(0.0, now - b["updated_time"]) * b["refill_per_s"])
        if tokens < 1.0:
            self._x("UPDATE budgets SET tokens=?, updated_time=? WHERE source_id=?", (tokens, now, source_id))
            return False
        self._x("UPDATE budgets SET tokens=?, updated_time=?, spent=spent+1 WHERE source_id=?",
                (tokens - 1.0, now, source_id))
        return True

    # ------------------------------------------------------------------ execution
    def claim_due(self, now_by_base: dict[str, float], owner: str, limit: int = 64) -> list[dict]:
        """Lease due jobs. ``now_by_base`` maps each time base to its current time."""
        wall = time.time()
        out = []
        with self._lock:
            self.recover_expired_leases(wall)
            for base, now in now_by_base.items():
                rows = self._db.execute(
                    "SELECT * FROM jobs WHERE status='pending' AND time_base=? AND next_attempt_time<=? "
                    "ORDER BY priority DESC, due_time LIMIT ?", (base, now, limit * 4)).fetchall()
                for r in rows:
                    if len(out) >= limit:
                        break
                    if not self._take_token(r["source_id"], now):
                        continue
                    self._db.execute("UPDATE jobs SET status='leased', lease_owner=?, lease_until=?, attempts=attempts+1 "
                                     "WHERE job_id=? AND status='pending'", (owner, wall + self.cfg.lease_s, r["job_id"]))
                    d = dict(r)
                    d["attempts"] += 1
                    out.append(d)
        return out

    def complete(self, job_id: str, obs_id: str | None) -> None:
        self._x("UPDATE jobs SET status='done', completed_wall=?, result_obs_id=?, lease_owner=NULL WHERE job_id=?",
                (time.time(), obs_id, job_id))

    def fail(self, job_id: str, error: str, now: float) -> str:
        r = self._q("SELECT attempts, max_attempts FROM jobs WHERE job_id=?", (job_id,))
        if not r:
            return "unknown"
        att, mx = r[0]["attempts"], r[0]["max_attempts"]
        if att >= mx:
            self._x("UPDATE jobs SET status='failed', error=?, lease_owner=NULL, completed_wall=? WHERE job_id=?",
                    (error, time.time(), job_id))
            return "failed"
        delay = min(self.cfg.backoff_max_s, self.cfg.backoff_base_s * 2 ** (att - 1))
        delay *= 1.0 + 0.25 * random.Random(f"{job_id}:{att}").random()
        self._x("UPDATE jobs SET status='pending', error=?, lease_owner=NULL, next_attempt_time=? WHERE job_id=?",
                (error, now + delay, job_id))
        return "retry"

    def release(self, job_id: str) -> None:
        """Give a leased job back without counting an attempt (e.g. budget exhausted downstream)."""
        self._x("UPDATE jobs SET status='pending', lease_owner=NULL, attempts=MAX(0, attempts-1) WHERE job_id=?", (job_id,))

    def recover_expired_leases(self, wall: float | None = None) -> int:
        wall = wall if wall is not None else time.time()
        return self._x("UPDATE jobs SET status='pending', lease_owner=NULL WHERE status='leased' AND lease_until<?",
                       (wall,))

    def cancel(self, action_id: str, reason: str) -> int:
        return self._x("UPDATE jobs SET status='cancelled', error=? WHERE action_id=? AND status IN ('pending','leased')",
                       (reason, action_id))

    def cancel_source(self, source_id: str, reason: str) -> int:
        return self._x("UPDATE jobs SET status='cancelled', error=? WHERE source_id=? AND status IN ('pending','leased')",
                       (reason, source_id))

    def purge_finished(self, older_than_wall: float) -> int:
        return self._x("DELETE FROM jobs WHERE status IN ('done','failed','cancelled') AND completed_wall<?",
                       (older_than_wall,))

    def jobs(self, action_id: str | None = None) -> list[dict]:
        if action_id:
            return [dict(r) for r in self._q("SELECT * FROM jobs WHERE action_id=? ORDER BY due_time", (action_id,))]
        return [dict(r) for r in self._q("SELECT * FROM jobs ORDER BY due_time")]

    def next_due(self, time_base: str) -> float | None:
        r = self._q("SELECT MIN(next_attempt_time) AS t FROM jobs WHERE status='pending' AND time_base=?", (time_base,))
        return r[0]["t"] if r and r[0]["t"] is not None else None

    def stats(self) -> dict:
        rows = self._q("SELECT status, kind, COUNT(*) AS n FROM jobs GROUP BY status, kind")
        out: dict[str, Any] = {}
        for r in rows:
            out[f"{r['kind']}:{r['status']}"] = r["n"]
        out["budget_spent"] = {r["source_id"]: r["spent"] for r in self._q("SELECT source_id, spent FROM budgets")}
        return out


Fetcher = Callable[[dict], tuple[dict, str]]  # job -> (payload, completeness); raise to fail


class ObservationWorker:
    """Executes leased jobs: fetch through the source's (permission-checked) fetcher, write the
    observation to the ledger idempotently, complete or fail the job."""

    def __init__(self, scheduler: TemporalObservationScheduler, ledger, fetchers: dict[str, Fetcher],
                 owner: str | None = None):
        self.s = scheduler
        self.ledger = ledger
        self.fetchers = fetchers
        self.owner = owner or f"worker-{uuid.uuid4().hex[:6]}"
        self.stats = {"done": 0, "retry": 0, "failed": 0, "missing": 0}

    def run_due(self, now_by_base: dict[str, float], limit: int = 256) -> int:
        from agentrs.temporal.delayed_outcome_ledger import ObservationRecord

        jobs = self.s.claim_due(now_by_base, self.owner, limit)
        for j in jobs:
            now = now_by_base[j["time_base"]]
            f = self.fetchers.get(j["source_id"])
            try:
                if f is None:
                    raise RuntimeError(f"no fetcher for source {j['source_id']}")
                payload, completeness = f(j)
            except Exception as e:  # noqa: BLE001 - any fetch failure is retried / recorded
                st = self.s.fail(j["job_id"], f"{type(e).__name__}: {e}", now)
                self.stats["retry" if st == "retry" else "failed"] += 1
                if st == "failed":
                    oid = self.ledger.record_observation(ObservationRecord(
                        action_id=j["action_id"], actual_time=now, payload={}, horizon_label=j["horizon_label"],
                        intended_time=j["due_time"], completeness="missing", source_id=j["source_id"],
                        job_id=j["job_id"], idem=f"job:{j['job_id']}", time_base=j["time_base"]))
                    self.stats["missing"] += 1
                continue
            oid = self.ledger.record_observation(ObservationRecord(
                action_id=j["action_id"], actual_time=now, payload=payload, horizon_label=j["horizon_label"],
                intended_time=j["due_time"], completeness=completeness, source_id=j["source_id"], job_id=j["job_id"],
                idem=f"job:{j['job_id']}", time_base=j["time_base"]))
            self.s.complete(j["job_id"], oid)
            self.stats["done"] += 1
        return len(jobs)
