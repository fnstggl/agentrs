"""Durable record of every action and everything later learned about its consequences.

SQLite (WAL) tables:

* ``actions``       — one row per action (ours or a third party's): source, actor type, context and
                      the complete action (incl. content), generating checkpoint, permissions, time base
                      (``real`` or ``virtual:<id>``), outcome window, training eligibility;
* ``forecasts``     — predictions made at an information cutoff. **Insert-only**: SQLite triggers
                      abort any UPDATE, and any DELETE outside a recorded governance purge. A new
                      prediction after new information is a new row with a later cutoff;
* ``observations``  — what a sensor saw at a given time (intended vs actual observation time,
                      completeness, the job that collected it), idempotent per key;
* ``events``        — subsequent observable events (replies, edits, moderation, ...), with the
                      environment time when known and the time we observed them; deduplicated per
                      (source, provider event id);
* ``rewards``       — verified reward increments with maturity time and generating checkpoint;
* ``ingest``        — training clock: when the learner consumed which label version;
* ``governance``    — revocations, deletions, purges (who/why/when).

All environment / observation times are in the action's time base; mixing bases is refused.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
  action_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_type TEXT NOT NULL, actor_type TEXT NOT NULL,
  actor_ref TEXT, env_id TEXT, env_version TEXT, time_base TEXT NOT NULL, action_time REAL NOT NULL,
  observed_time REAL NOT NULL, goal TEXT, context_text TEXT, context_json TEXT, action_str TEXT NOT NULL,
  action_json TEXT, semantic_json TEXT, content_hash TEXT, policy_checkpoint TEXT, provenance_json TEXT,
  permission_json TEXT, split TEXT, grp TEXT, profile TEXT, window_end REAL, status TEXT DEFAULT 'open',
  close_reason TEXT, closed_time REAL, training_eligible INTEGER DEFAULT 1, created_wall REAL);
CREATE INDEX IF NOT EXISTS actions_source ON actions(source_id);
CREATE TABLE IF NOT EXISTS forecasts (
  forecast_id TEXT PRIMARY KEY, action_id TEXT NOT NULL, kind TEXT NOT NULL, info_cutoff_time REAL NOT NULL,
  cutoff_elapsed REAL NOT NULL, created_time REAL NOT NULL, created_wall REAL NOT NULL, checkpoint TEXT,
  horizons_json TEXT, prediction_json TEXT NOT NULL, history_hash TEXT);
CREATE INDEX IF NOT EXISTS forecasts_action ON forecasts(action_id);
CREATE TABLE IF NOT EXISTS governance_lock (k INTEGER);
CREATE TRIGGER IF NOT EXISTS forecasts_immutable_u BEFORE UPDATE ON forecasts
  BEGIN SELECT RAISE(ABORT, 'forecasts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS forecasts_immutable_d BEFORE DELETE ON forecasts
  WHEN (SELECT COUNT(*) FROM governance_lock) = 0
  BEGIN SELECT RAISE(ABORT, 'forecasts are immutable'); END;
CREATE TABLE IF NOT EXISTS observations (
  obs_id TEXT PRIMARY KEY, action_id TEXT NOT NULL, horizon_label TEXT, intended_time REAL, actual_time REAL NOT NULL,
  env_event_time REAL, elapsed REAL NOT NULL, payload_json TEXT, completeness TEXT, source_id TEXT, job_id TEXT,
  ingest_wall REAL, idem TEXT UNIQUE);
CREATE INDEX IF NOT EXISTS obs_action ON observations(action_id);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, action_id TEXT NOT NULL, type TEXT NOT NULL, env_event_time REAL, observed_time REAL NOT NULL,
  actor_ref TEXT, by_original_actor INTEGER, payload_json TEXT, source_id TEXT, provider_event_id TEXT,
  ingest_wall REAL, UNIQUE(source_id, provider_event_id));
CREATE INDEX IF NOT EXISTS events_action ON events(action_id);
CREATE TABLE IF NOT EXISTS rewards (
  reward_id TEXT PRIMARY KEY, action_id TEXT NOT NULL, kind TEXT NOT NULL, value REAL NOT NULL, env_time REAL,
  maturity_time REAL, checkpoint TEXT, objective TEXT, detail_json TEXT, idem TEXT UNIQUE, created_wall REAL);
CREATE TABLE IF NOT EXISTS ingest (
  action_id TEXT NOT NULL, label_version TEXT NOT NULL, ingested_wall REAL NOT NULL, trainer TEXT, train_step INTEGER);
CREATE INDEX IF NOT EXISTS ingest_action ON ingest(action_id);
CREATE TABLE IF NOT EXISTS governance (
  at_wall REAL, source_id TEXT, action_id TEXT, op TEXT, reason TEXT, detail_json TEXT);
"""


class ActionRecord(BaseModel):
    action_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_id: str
    source_type: str  # self | third_party | demo
    actor_type: str  # agent | human | simulated_user | software_agent | system
    actor_ref: str | None = None  # pseudonymous id
    env_id: str = ""
    env_version: str = ""
    time_base: str = "real"
    action_time: float
    observed_time: float  # when we first observed it (== action_time for our own actions)
    goal: str | None = None
    context_text: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    action_str: str
    action: dict[str, Any] = Field(default_factory=dict)  # operation, content, parameters
    semantic: dict[str, Any] | None = None  # SemanticAction dump for our own content actions
    policy_checkpoint: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    permission: dict[str, Any] = Field(default_factory=dict)
    split: str = "train"
    group: str = ""
    profile: str = ""
    window_end: float | None = None
    training_eligible: bool = True


class ObservationRecord(BaseModel):
    action_id: str
    actual_time: float  # observation clock
    payload: dict[str, Any]  # {"counts": {...}, "state": str, "continuous": {...}, "occurred": {...}}
    horizon_label: str | None = None
    intended_time: float | None = None
    env_event_time: float | None = None
    completeness: str = "complete"  # complete | partial | missing
    source_id: str = ""
    job_id: str | None = None
    idem: str | None = None
    time_base: str | None = None  # if given, must equal the action's time base


class EventRecord(BaseModel):
    action_id: str
    type: str
    observed_time: float
    env_event_time: float | None = None
    actor_ref: str | None = None
    by_original_actor: bool | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    source_id: str = ""
    provider_event_id: str | None = None
    time_base: str | None = None


class LedgerError(RuntimeError):
    pass


def content_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class DelayedOutcomeLedger:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ helpers
    def _q(self, sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    def _x(self, sql: str, args: Iterable = ()) -> int:
        with self._lock:
            return self._db.execute(sql, tuple(args)).rowcount

    def _base(self, action_id: str) -> str:
        r = self._q("SELECT time_base FROM actions WHERE action_id=?", (action_id,))
        if not r:
            raise LedgerError(f"unknown action {action_id}")
        return r[0]["time_base"]

    # ------------------------------------------------------------------ actions
    def record_action(self, rec: ActionRecord) -> str:
        """Idempotent: re-recording an existing action id returns it unchanged."""
        if rec.observed_time < rec.action_time - 1e-6:
            raise LedgerError("an action cannot be observed before it happened")
        content = str(rec.action.get("content") or "")
        self._x("""INSERT OR IGNORE INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rec.action_id, rec.source_id, rec.source_type, rec.actor_type, rec.actor_ref, rec.env_id,
                 rec.env_version, rec.time_base, rec.action_time, rec.observed_time, rec.goal, rec.context_text,
                 json.dumps(rec.context), rec.action_str, json.dumps(rec.action),
                 json.dumps(rec.semantic) if rec.semantic is not None else None, content_hash(content),
                 rec.policy_checkpoint, json.dumps(rec.provenance), json.dumps(rec.permission), rec.split, rec.group,
                 rec.profile, rec.window_end, "open", None, None, int(rec.training_eligible), time.time()))
        return rec.action_id

    def action(self, action_id: str) -> dict | None:
        r = self._q("SELECT * FROM actions WHERE action_id=?", (action_id,))
        if not r:
            return None
        d = dict(r[0])
        for k in ("context_json", "action_json", "semantic_json", "provenance_json", "permission_json"):
            d[k[:-5]] = json.loads(d.pop(k)) if d.get(k) else None
        return d

    def actions(self, where: str = "1=1", args: Iterable = ()) -> list[dict]:
        ids = [r["action_id"] for r in self._q(f"SELECT action_id FROM actions WHERE {where} ORDER BY action_time",
                                               args)]
        return [self.action(i) for i in ids]

    def close_action(self, action_id: str, at: float, reason: str) -> None:
        self._x("UPDATE actions SET status='closed', close_reason=?, closed_time=? WHERE action_id=? AND status='open'",
                (reason, at, action_id))

    def set_eligibility(self, action_id: str, eligible: bool, reason: str) -> None:
        self._x("UPDATE actions SET training_eligible=? WHERE action_id=?", (int(eligible), action_id))
        self._gov(None, action_id, "eligibility", reason, {"eligible": eligible})

    # ------------------------------------------------------------------ forecasts (immutable)
    def record_forecast(self, action_id: str, prediction: dict, checkpoint: str, info_cutoff_time: float,
                        created_time: float, horizons: list[float], kind: str = "initial",
                        history_hash: str | None = None) -> str:
        a = self.action(action_id)
        if a is None:
            raise LedgerError(f"unknown action {action_id}")
        if info_cutoff_time > created_time + 1e-6:
            raise LedgerError("a forecast cannot use information from after its creation time")
        if info_cutoff_time < a["action_time"] - 1e-6:
            raise LedgerError("information cutoff before the action")
        fid = uuid.uuid4().hex
        self._x("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (fid, action_id, kind, info_cutoff_time, info_cutoff_time - a["action_time"], created_time, time.time(),
                 checkpoint, json.dumps(horizons), json.dumps(prediction), history_hash))
        return fid

    def forecasts(self, action_id: str) -> list[dict]:
        rows = self._q("SELECT * FROM forecasts WHERE action_id=? ORDER BY info_cutoff_time, created_wall", (action_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["prediction"] = json.loads(d.pop("prediction_json"))
            d["horizons"] = json.loads(d.pop("horizons_json"))
            out.append(d)
        return out

    # ------------------------------------------------------------------ observations / events
    def record_observation(self, rec: ObservationRecord) -> str | None:
        a = self.action(rec.action_id)
        if a is None:
            raise LedgerError(f"unknown action {rec.action_id}")
        if a["status"] == "purged":
            return None
        if rec.time_base is not None and rec.time_base != a["time_base"]:
            raise LedgerError(f"time base {rec.time_base} != action's {a['time_base']}")
        idem = rec.idem or f"{rec.action_id}:{rec.job_id or ''}:{rec.actual_time:.3f}"
        oid = uuid.uuid4().hex
        n = self._x("INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (oid, rec.action_id, rec.horizon_label, rec.intended_time, rec.actual_time, rec.env_event_time,
                     rec.actual_time - a["action_time"], json.dumps(rec.payload), rec.completeness, rec.source_id,
                     rec.job_id, time.time(), idem))
        if n == 0:
            r = self._q("SELECT obs_id FROM observations WHERE idem=?", (idem,))
            return r[0]["obs_id"] if r else None
        return oid

    def observations(self, action_id: str, observed_before: float | None = None) -> list[dict]:
        sql, args = "SELECT * FROM observations WHERE action_id=?", [action_id]
        if observed_before is not None:
            sql += " AND actual_time <= ?"
            args.append(observed_before)
        rows = self._q(sql + " ORDER BY actual_time", args)
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            out.append(d)
        return out

    def record_event(self, rec: EventRecord) -> bool:
        """Returns False for duplicates (same provider event id from the same source)."""
        a = self.action(rec.action_id)
        if a is None:
            raise LedgerError(f"unknown action {rec.action_id}")
        if rec.time_base is not None and rec.time_base != a["time_base"]:
            raise LedgerError(f"time base {rec.time_base} != action's {a['time_base']}")
        pid = rec.provider_event_id or uuid.uuid4().hex
        n = self._x("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (uuid.uuid4().hex, rec.action_id, rec.type, rec.env_event_time, rec.observed_time, rec.actor_ref,
                     None if rec.by_original_actor is None else int(rec.by_original_actor), json.dumps(rec.payload),
                     rec.source_id, pid, time.time()))
        return n > 0

    def events(self, action_id: str, observed_before: float | None = None) -> list[dict]:
        sql, args = "SELECT * FROM events WHERE action_id=?", [action_id]
        if observed_before is not None:
            sql += " AND observed_time <= ?"
            args.append(observed_before)
        rows = self._q(sql + " ORDER BY COALESCE(env_event_time, observed_time), observed_time", args)
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            out.append(d)
        return out

    # ------------------------------------------------------------------ rewards
    def record_reward(self, action_id: str, kind: str, value: float, env_time: float, maturity_time: float,
                      checkpoint: str | None, objective: str, idem: str, detail: dict | None = None) -> bool:
        n = self._x("INSERT OR IGNORE INTO rewards VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (uuid.uuid4().hex, action_id, kind, float(value), env_time, maturity_time, checkpoint, objective,
                     json.dumps(detail or {}), idem, time.time()))
        return n > 0

    def rewards(self, action_id: str) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM rewards WHERE action_id=? ORDER BY maturity_time", (action_id,))]

    # ------------------------------------------------------------------ training clock
    def label_version(self, action_id: str) -> str:
        r = self._q("""SELECT (SELECT COUNT(*) FROM observations WHERE action_id=?) AS o,
                              (SELECT COUNT(*) FROM events WHERE action_id=?) AS e,
                              (SELECT status FROM actions WHERE action_id=?) AS s""", (action_id,) * 3)[0]
        return f"o{r['o']}e{r['e']}{r['s']}"

    def mark_ingested(self, action_ids: Iterable[str], trainer: str, step: int) -> None:
        now = time.time()
        with self._lock:
            for a in action_ids:
                self._db.execute("INSERT INTO ingest VALUES (?,?,?,?,?)", (a, self.label_version(a), now, trainer, step))

    def changed_since_ingest(self, trainer: str) -> list[str]:
        """Actions with new labels since this trainer last consumed them (progressive updates)."""
        out = []
        for r in self._q("SELECT action_id FROM actions WHERE training_eligible=1 AND status!='purged'"):
            a = r["action_id"]
            last = self._q("SELECT label_version FROM ingest WHERE action_id=? AND trainer=? ORDER BY ingested_wall DESC "
                           "LIMIT 1", (a, trainer))
            if not last or last[0]["label_version"] != self.label_version(a):
                out.append(a)
        return out

    # ------------------------------------------------------------------ governance
    def _gov(self, source_id, action_id, op, reason, detail=None):
        self._x("INSERT INTO governance VALUES (?,?,?,?,?,?)",
                (time.time(), source_id, action_id, op, reason, json.dumps(detail or {})))

    def redact_action(self, action_id: str, reason: str) -> None:
        """Remove stored content (e.g. the author deleted it and the source requires propagation)
        and exclude the action from training; outcome metadata without content is kept."""
        a = self.action(action_id)
        if a is None:
            return
        act = dict(a["action"] or {})
        act["content"] = None
        act["redacted"] = True
        self._x("UPDATE actions SET action_json=?, action_str=?, context_text='', semantic_json=NULL, "
                "training_eligible=0 WHERE action_id=?", (json.dumps(act), "[redacted]", action_id))
        self._x("UPDATE events SET payload_json='{}' WHERE action_id=?", (action_id,))
        self._gov(a["source_id"], action_id, "redact", reason)

    def purge_source(self, source_id: str, reason: str) -> dict:
        """Delete everything that came from a source (revoked authorisation / deletion request),
        including its forecasts — the only path that may delete forecasts, and it is logged."""
        ids = [r["action_id"] for r in self._q("SELECT action_id FROM actions WHERE source_id=?", (source_id,))]
        return self.purge_actions(ids, reason, source_id=source_id)

    def purge_actions(self, ids: list[str], reason: str, source_id: str | None = None) -> dict:
        counts = {"actions": len(ids)}
        with self._lock:
            self._db.execute("BEGIN")
            try:
                self._db.execute("INSERT INTO governance_lock VALUES (1)")
                for t in ("forecasts", "observations", "events", "rewards", "ingest"):
                    n = 0
                    for a in ids:
                        n += self._db.execute(f"DELETE FROM {t} WHERE action_id=?", (a,)).rowcount
                    counts[t] = n
                for a in ids:
                    self._db.execute("DELETE FROM actions WHERE action_id=?", (a,))
                self._db.execute("DELETE FROM governance_lock")
                self._db.execute("INSERT INTO governance VALUES (?,?,?,?,?,?)",
                                 (time.time(), source_id, None, "purge", reason, json.dumps(counts)))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return counts

    def governance_log(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM governance ORDER BY at_wall")]

    def stats(self) -> dict:
        r = {}
        for t in ("actions", "forecasts", "observations", "events", "rewards", "ingest"):
            r[t] = self._q(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"]
        return r
