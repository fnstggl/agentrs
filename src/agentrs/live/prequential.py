"""Real-time (prequential) scoring and the stratified baseline it is compared against.

Every forecast is stored in the ledger when the post is first seen, before any outcome exists.
Later, once the follow-up observation at horizon ``h`` has actually been made, the outcome at ``h``
is final and one row per (post, event, horizon) is written here with the model's stored forecast,
the baseline's stored forecast (made at the same moment from outcomes known at that moment) and the
truth. Nothing is ever re-forecast after the fact, so the scores cannot use hindsight.

Baseline: empirical rates per stratum (source / post kind × author audience bucket), shrunk toward
the kind's and then the global rate; counts as the stratum's geometric mean. It is refitted from the
rows already scored here, i.e. only from outcomes known before each forecast was made.
"""

from __future__ import annotations

import math
import random
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from agentrs.evaluation.metrics import auroc
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger
from agentrs.temporal.outcome_labels import build_examples
from agentrs.temporal.temporal_evaluation import event_truth
from agentrs.temporal.temporal_world_model import TemporalSpec
from agentrs.temporal.time_encoding import describe_duration

SCHEMA = """
CREATE TABLE IF NOT EXISTS ev (
  action_id TEXT, source TEXT, stratum TEXT, kind TEXT, event TEXT, horizon_s REAL, action_time REAL,
  forecast_time REAL, p_model REAL, p_base REAL, y INTEGER, checkpoint TEXT, scored_wall REAL,
  PRIMARY KEY (action_id, event, horizon_s));
CREATE TABLE IF NOT EXISTS ct (
  action_id TEXT, source TEXT, stratum TEXT, kind TEXT, count TEXT, horizon_s REAL, at_s REAL, action_time REAL,
  forecast_time REAL, pred_model REAL, pred_base REAL, actual REAL, checkpoint TEXT, scored_wall REAL,
  PRIMARY KEY (action_id, count, horizon_s));
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v REAL);
CREATE INDEX IF NOT EXISTS ev_ft ON ev(forecast_time);
"""


def audience_bucket(ctx: dict) -> str:
    v = ctx.get("author_followers", ctx.get("author_karma"))
    if v is None:
        return "unknown"
    v = int(v)
    if v <= 0:
        return "0"
    return f"1e{min(5, int(math.log10(v)))}"


def stratum_of(action: dict) -> tuple[str, str, str]:
    """(source, kind, stratum)."""
    ctx = action.get("context") or {}
    grp = action.get("grp") or action.get("group") or "unknown"
    return action["source_id"], grp, f"{grp}|{audience_bucket(ctx)}"


class PrequentialStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    def q(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, tuple(args)).fetchall()

    def x(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, tuple(args)).rowcount

    def get(self, k: str, default: float = 0.0) -> float:
        r = self.q("SELECT v FROM meta WHERE k=?", (k,))
        return float(r[0]["v"]) if r else default

    def put(self, k: str, v: float) -> None:
        self.x("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))


class StratifiedBaseline:
    def __init__(self, store: PrequentialStore, spec: TemporalSpec, m: float = 20.0):
        self.store, self.spec, self.m = store, spec, m
        self.ev: dict[tuple, tuple[float, float]] = {}  # (level key, event, h) -> (positives, n)
        self.ct: dict[tuple, tuple[float, float]] = {}  # (level key, count, h) -> (sum log1p, n)
        self.fitted_wall = 0.0

    def fit(self) -> None:
        ev, ct = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
        for r in self.store.q("SELECT kind, stratum, event, horizon_s, SUM(y) AS k, COUNT(*) AS n FROM ev "
                              "GROUP BY stratum, event, horizon_s"):
            for key in (("s", r["stratum"]), ("k", r["kind"]), ("g", "")):
                e = ev[(key, r["event"], r["horizon_s"])]
                e[0] += r["k"]
                e[1] += r["n"]
        for r in self.store.q("SELECT kind, stratum, count, horizon_s, actual FROM ct"):
            for key in (("s", r["stratum"]), ("k", r["kind"]), ("g", "")):
                c = ct[(key, r["count"], r["horizon_s"])]
                c[0] += math.log1p(max(0.0, r["actual"]))
                c[1] += 1
        self.ev = {k: tuple(v) for k, v in ev.items()}
        self.ct = {k: tuple(v) for k, v in ct.items()}
        self.fitted_wall = time.time()

    def _rate(self, table, kind, stratum, name, h, prior):
        p = prior
        for key in (("g", ""), ("k", kind), ("s", stratum)):  # global -> kind -> stratum, each shrunk to the last
            s, n = table.get((key, name, h), (0.0, 0.0))
            p = (s + self.m * p) / (n + self.m)
        return p

    def predict(self, action: dict, horizons: list[float]) -> dict:
        _, kind, stratum = stratum_of(action)
        hs = []
        for h in horizons:
            hs.append({"horizon_s": h, "horizon": describe_duration(h),
                       "event_prob": {e: self._rate(self.ev, kind, stratum, e, h, 0.5) for e in self.spec.events},
                       "count_mean": {c: math.expm1(self._rate(self.ct, kind, stratum, c, h, 0.0))
                                      for c in self.spec.counts}})
        return {"horizons": hs, "checkpoint": "baseline-stratified", "meta": {"stratum": stratum}}


def _pred_at(pred: dict, h: float) -> Optional[dict]:
    for hp in pred.get("horizons", []):
        if abs(hp["horizon_s"] - h) < 1e-6:
            return hp
    return None


class PrequentialScorer:
    def __init__(self, ledger: DelayedOutcomeLedger, spec: TemporalSpec, store: PrequentialStore):
        self.ledger, self.spec, self.store = ledger, spec, store
        with ledger._lock:
            ledger._db.execute("CREATE INDEX IF NOT EXISTS obs_ingest ON observations(ingest_wall)")
            ledger._db.execute("CREATE INDEX IF NOT EXISTS ev_ingest ON events(ingest_wall)")

    def update(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        since = self.store.get("scored_through", 0.0)
        ids = [r["action_id"] for r in self.ledger._q(
            "SELECT DISTINCT action_id FROM observations WHERE ingest_wall>? AND ingest_wall<=?", (since - 5, now))]
        n_ev = n_ct = 0
        for aid in ids:
            a = self.ledger.action(aid)
            if a is None or not a["training_eligible"]:
                continue  # deleted at source (redacted) or purged
            fc = {f["kind"]: f for f in self.ledger.forecasts(aid) if f["kind"] in ("initial", "baseline")}
            if "initial" not in fc or "baseline" not in fc:
                continue
            obs = [o for o in self.ledger.observations(aid, observed_before=now) if o["completeness"] == "complete"]
            if not obs:
                continue
            seen = max(o["elapsed"] for o in obs)
            exs = build_examples(self.ledger, aid, self.spec, as_of=now, cutoffs=(0.0,))
            if not exs:
                continue
            ex = exs[0]
            source, kind, stratum = stratum_of(a)
            fm, fb = fc["initial"], fc["baseline"]
            for h in fm["horizons"]:
                if seen < h * 0.98:  # outcome at h is final only once the follow-up at h was made
                    continue
                pm, pb = _pred_at(fm["prediction"], h), _pred_at(fb["prediction"], h)
                if pm is None or pb is None:
                    continue
                for e in self.spec.events:
                    y = event_truth(ex, e, h)
                    if y is None or e not in pm["event_prob"]:
                        continue
                    n_ev += self.store.x("INSERT OR IGNORE INTO ev VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                         (aid, source, stratum, kind, e, h, a["action_time"], fm["created_time"],
                                          float(pm["event_prob"][e]), float(pb["event_prob"][e]), int(y),
                                          fm["checkpoint"], now))
                o = min((o for o in obs if o["elapsed"] >= h * 0.98), key=lambda o: o["elapsed"])
                if o["elapsed"] > h * 1.5 + 600:
                    continue  # the observation is too late to stand for horizon h
                for c, v in (o["payload"].get("counts") or {}).items():
                    if c not in pm["count_mean"]:
                        continue
                    n_ct += self.store.x("INSERT OR IGNORE INTO ct VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                         (aid, source, stratum, kind, c, h, o["elapsed"], a["action_time"],
                                          fm["created_time"], float(pm["count_mean"][c]), float(pb["count_mean"][c]),
                                          float(v), fm["checkpoint"], now))
        self.store.put("scored_through", now)
        return {"actions": len(ids), "event_rows": n_ev, "count_rows": n_ct}

    # ------------------------------------------------------------------ reports
    @staticmethod
    def _brier(p, y):
        return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))

    @staticmethod
    def _ll(p, y):
        p = np.clip(np.asarray(p), 1e-4, 1 - 1e-4)
        y = np.asarray(y)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _paired_ci(self, rows, reps: int = 400, seed: int = 0):
        """95% bootstrap CI of Brier(baseline) - Brier(model), resampling posts (clusters)."""
        by = defaultdict(list)
        for r in rows:
            by[r["action_id"]].append((r["p_base"] - r["y"]) ** 2 - (r["p_model"] - r["y"]) ** 2)
        groups = [np.asarray(v) for v in by.values()]
        if len(groups) < 20:
            return None
        sums = np.array([g.sum() for g in groups])
        lens = np.array([len(g) for g in groups])
        rng = random.Random(seed)
        idx = list(range(len(groups)))
        stats = []
        for _ in range(reps):
            s = [rng.choice(idx) for _ in idx]
            stats.append(sums[s].sum() / lens[s].sum())
        return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

    def report(self, since_forecast_time: float = 0.0, ci: bool = True) -> dict:
        rows = self.store.q("SELECT * FROM ev WHERE forecast_time>=?", (since_forecast_time,))
        out = {"n_event_rows": len(rows), "posts_scored": len({r["action_id"] for r in rows}), "by": {}, "overall": {}}
        groups = defaultdict(list)
        for r in rows:
            groups[(r["source"], r["event"], r["horizon_s"])].append(r)
        for (src, e, h), rs in sorted(groups.items()):
            p, b, y = [r["p_model"] for r in rs], [r["p_base"] for r in rs], [r["y"] for r in rs]
            bm, bb = self._brier(p, y), self._brier(b, y)
            out["by"].setdefault(src, {}).setdefault(e, {})[describe_duration(h)] = {
                "n": len(rs), "pos_rate": round(float(np.mean(y)), 4), "brier_model": round(bm, 5),
                "brier_base": round(bb, 5), "skill": round(1 - bm / bb, 4) if bb > 0 else None,
                "auroc_model": auroc(p, y), "auroc_base": auroc(b, y),
                "logloss_model": round(self._ll(p, y), 4), "logloss_base": round(self._ll(b, y), 4)}
        for src in {r["source"] for r in rows}:
            rs = [r for r in rows if r["source"] == src]
            p, b, y = [r["p_model"] for r in rs], [r["p_base"] for r in rs], [r["y"] for r in rs]
            bm, bb = self._brier(p, y), self._brier(b, y)
            out["overall"][src] = {"n": len(rs), "posts": len({r["action_id"] for r in rs}),
                                   "brier_model": round(bm, 5), "brier_base": round(bb, 5),
                                   "skill": round(1 - bm / bb, 4) if bb > 0 else None,
                                   "brier_gain_ci95": self._paired_ci(rs) if ci else None,
                                   "auroc_model": auroc(p, y), "auroc_base": auroc(b, y)}
        crow = self.store.q("SELECT * FROM ct WHERE forecast_time>=?", (since_forecast_time,))
        cg = defaultdict(list)
        for r in crow:
            cg[(r["source"], r["count"], r["horizon_s"])].append(r)
        out["counts"] = {}
        for (src, c, h), rs in sorted(cg.items()):
            lm = np.mean([abs(math.log1p(max(0, r["pred_model"])) - math.log1p(r["actual"])) for r in rs])
            lb = np.mean([abs(math.log1p(max(0, r["pred_base"])) - math.log1p(r["actual"])) for r in rs])
            out["counts"].setdefault(src, {}).setdefault(c, {})[describe_duration(h)] = {
                "n": len(rs), "mean_actual": round(float(np.mean([r["actual"] for r in rs])), 3),
                "logmae_model": round(float(lm), 4), "logmae_base": round(float(lb), 4)}
        return out

    def trend(self, bucket_s: float = 3600.0) -> list[dict]:
        """Skill by the time the forecasts were made: does the learner get better as it runs?"""
        rows = self.store.q("SELECT source, forecast_time, p_model, p_base, y FROM ev")
        if not rows:
            return []
        t0 = min(r["forecast_time"] for r in rows)
        g = defaultdict(list)
        for r in rows:
            g[(r["source"], int((r["forecast_time"] - t0) // bucket_s))].append(r)
        out = []
        for (src, k), rs in sorted(g.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            p, b, y = [r["p_model"] for r in rs], [r["p_base"] for r in rs], [r["y"] for r in rs]
            bm, bb = self._brier(p, y), self._brier(b, y)
            out.append({"source": src, "bucket": k, "start": t0 + k * bucket_s, "n": len(rs),
                        "brier_model": round(bm, 5), "brier_base": round(bb, 5),
                        "skill": round(1 - bm / bb, 4) if bb > 0 else None})
        return out
