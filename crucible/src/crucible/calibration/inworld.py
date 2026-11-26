"""In-world validation of the fitted behaviour (MASTER_SPEC 5.2 / 5.5; requirements SP-5.0 and SP-5.5b). Owner:
workstream E. The population workstream wires the runtime decision models into the world (D-15); this module
measures what a world then does and compares it with held-out data.

Two drivers produce the same statistics, so model drift and integration drift can be told apart:

* `simulate_models` - the runtime decision models alone (daily_return@1 + extra_sessions@1), forward-simulated
  from the burn-in state of KuaiRand's held-out (test) users over the next days of the data, and on beyond the
  data's horizon (stationarity; no data to compare with there);
* `world_activity` - the activity a running world recorded through decisions whose provenance names those models
  (population decision batches, `record_decisions`), so the comparison measures the integrated world.

`compare(sim, ref)` gives four reports against the held-out users' real activity: the activity distribution, the
retention curve conditional on prior activity, heterogeneity across individuals and long-horizon drift. Each report
states its statistic, its prespecified tolerance (`TOLERANCE`) and whether it is within it; a report is never
"passed" by construction.

In-world randomised experiments are analysed from recorded decisions that carry an experiment label
(``extra.experiment = {"id": ..., "arm": ...}``, written by the world's mechanism that ran the experiment):

* `random_exposure_experiment`: items inserted at random into feeds (arm ``random``), opened or not through
  content_open@1; compared with KuaiRand's random-exposure log (open rate, and the odds ratio between items of the
  individual's own topics and other items);
* `ad_holdout_experiment`: campaign eligibility randomised across individuals (arms ``eligible`` / ``holdout``),
  visits through ad_response@2; the in-world complier log-odds shift per attended display (eligibility as the
  instrument for display, the E5 likelihood with each stratum's displayed-complier effect scaled by the recorded
  mean attention A / A_ref of its displayed individuals) compared with Criteo's shift (model
  ad_exposure_response@1) by a two-sample z test. Criteo's shift is, by the runtime's transport assumption
  (A_ref = 1, record advertising.attention_composition), the effect of a fully attended display; the unadjusted
  in-world shift and the intention-to-treat difference are reported alongside.

Which recorded decisions count (`audit`). A decision batch counts only when (a) it is an organic population decision
or a participant's own decision under the population's provenance rules (`crucible.population.check_provenance`:
listed by the population's own session batch and inside its cohort, or recorded by a tool command of that
individual), re-derived batch by batch here and cross-checked against `check_provenance`'s own totals, and (b) the
runtime decision model it names (or one serving the fitted model it names) evaluated at least as many rows in this
process as the counted batches claim (`runtime.usage()`; reset it before building the world). A batch stamped with
fitted-model provenance by a system call or module code, or one the runtime never evaluated, is not counted.

What the population must record for the in-world driver (contract, RUNTIME.md "In-world validation"): each daily
activation decision as a population decision batch whose provenance model id is ``daily_return@1`` (or the fitted id
``return_daily@1``) with a boolean column ``active``; sessions per active day with model ``extra_sessions@1`` (or
``session_rate@1``) and an integer column ``sessions``; feed opens with model ``content_open@1`` and a boolean column
``opened``; ad visits with ``ad_response@2`` and columns ``visit``, ``displayed``, ``organic_visit_probability`` and
``attended_probability``. A world that records none of them is reported as not wired (`NotWired`), never as validated.
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import runtime

DAY = 86400
# prespecified tolerances of the in-world reports (set before any in-world run; see the module docstring). Repair
# pass (2026-11-26): the heterogeneity statistic was the distribution of individuals' active-day shares, a function of
# the active-days histogram that `activity_distribution` already compares (its KS equalled that report's TV); it is
# now the distribution of individuals' own session intensity (mean sessions per active day). The change was made after
# the model-only run had been seen; the tolerances are unchanged.
TOLERANCE = {"activity_tv": 0.10, "sessions_tv": 0.10, "retention_max_abs": 0.05, "heterogeneity_ks": 0.10,
             "heterogeneity_sd_ratio": (0.8, 1.25), "drift_mean_abs": 0.03, "stationarity_abs": 0.05}
ACTIVITY_MODELS = ("daily_return@1", "return_daily@1")
SESSION_MODELS = ("extra_sessions@1", "session_rate@1")
HISTORY_BINS = ((0, 1), (2, 3), (4, 5), (6, 7))


class NotWired(RuntimeError):
    """The world records no decision of the fitted model the measurement needs."""


# ====================================================================== held-out reference (KuaiRand test users)
def kuairand_reference(first_sim_day=None, horizon: int = 7) -> dict:
    """Real daily activity of KuaiRand's held-out (test) users: burn-in up to `first_sim_day` - 1 (default
    2022-05-02, i.e. history up to 05-01), then `horizon` days to compare with (must end by 2022-05-08)."""
    from . import kuairand as K
    raw = K.load_raw()
    users, rows, sess = K.activity_matrices(raw)
    b = np.array([K.user_bucket(u) for u in users])
    test = users[K.split_of_bucket(b) == "test"]
    d0 = K.day_num(2022, 5, 2) if first_sim_day is None else int(first_sim_day)
    if d0 + horizon - 1 > K.day_num(2022, 5, 8):
        raise ValueError("the comparison window ends after the data (2022-05-08)")
    return {"users": test, "active": rows[test] > 0, "rows": rows[test], "sessions": sess[test], "d0": d0,
            "horizon": int(horizon), "source": "kuairand_pure test users (user bucket >= 85), both logs",
            "calendar_d0": K.D0.isoformat()}


# ====================================================================== statistics
def _hist(x, bins) -> np.ndarray:
    h = np.bincount(np.clip(np.asarray(x, int), 0, bins - 1), minlength=bins).astype(float)
    return h / max(h.sum(), 1.0)


def _ks(a, b) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not a.size or not b.size:
        return float("nan")
    grid = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / a.size -
                                np.searchsorted(b, grid, side="right") / b.size)))


def activity_stats(active: np.ndarray, sessions: np.ndarray, d0: int, horizon: int) -> dict:
    """Statistics of the window d0..d0+horizon-1 of per-individual day matrices [n, D] (history before d0)."""
    A = np.asarray(active, bool)[:, d0:d0 + horizon]
    S = np.asarray(sessions, float)[:, d0:d0 + horizon]
    hist7 = np.asarray(active, bool)[:, max(d0 - 7, 0):d0].sum(1)
    per_active = S[A]
    days_active = A.sum(1)
    seen = days_active > 0
    intensity = (S * A).sum(1)[seen] / days_active[seen]        # an individual's own mean sessions per active day
    ret = {}
    for lo, hi in HISTORY_BINS:
        m = (hist7 >= lo) & (hist7 <= hi)
        ret[f"{lo}-{hi}"] = {"n": int(m.sum()), "curve": A[m].mean(0).tolist() if m.any() else []}
    return {"n": int(A.shape[0]), "active_days_hist": _hist(A.sum(1), horizon + 1).tolist(),
            "sessions_per_active_day_hist": _hist(per_active, 11).tolist() if per_active.size else [],
            "daily_active_share": A.mean(0).tolist(), "user_session_intensity": intensity,
            "user_session_intensity_sd": float(intensity.std()) if intensity.size else float("nan"),
            "retention_by_history": ret}


def compare(sim: dict, ref: dict) -> dict:
    """The four in-world (or model-only) validation reports of `sim` against the held-out reference `ref` (both
    from `activity_stats`)."""
    T = TOLERANCE
    tv_a = 0.5 * float(np.abs(np.subtract(sim["active_days_hist"], ref["active_days_hist"])).sum())
    tv_s = (0.5 * float(np.abs(np.subtract(sim["sessions_per_active_day_hist"],
                                           ref["sessions_per_active_day_hist"])).sum())
            if sim["sessions_per_active_day_hist"] and ref["sessions_per_active_day_hist"] else float("nan"))
    diffs, w = [], []
    for k, r in ref["retention_by_history"].items():
        s = sim["retention_by_history"].get(k) or {}
        if r["n"] and s.get("n"):
            diffs.append(np.max(np.abs(np.subtract(s["curve"], r["curve"]))))
            w.append(r["n"])
    ret = float(np.average(diffs, weights=w)) if diffs else float("nan")
    ks = _ks(sim["user_session_intensity"], ref["user_session_intensity"])
    sdr = sim["user_session_intensity_sd"] / max(ref["user_session_intensity_sd"], 1e-12)
    drift = float(np.mean(np.abs(np.subtract(sim["daily_active_share"], ref["daily_active_share"]))))
    ok = lambda v, t: bool(np.isfinite(v) and v <= t)
    return {
        "activity_distribution": {"active_days_tv": tv_a, "sessions_per_active_day_tv": tv_s,
                                  "tolerance": [T["activity_tv"], T["sessions_tv"]],
                                  "within_tolerance": ok(tv_a, T["activity_tv"]) and ok(tv_s, T["sessions_tv"]),
                                  "sim": sim["active_days_hist"], "data": ref["active_days_hist"]},
        "retention_curve": {"weighted_max_abs_diff_by_history": ret, "tolerance": T["retention_max_abs"],
                            "within_tolerance": ok(ret, T["retention_max_abs"]),
                            "sim": sim["retention_by_history"], "data": ref["retention_by_history"]},
        "heterogeneity": {"user_session_intensity_ks": ks, "sd_ratio": float(sdr),
                          "statistic": "distribution across individuals (active at least once in the window) of "
                                       "their own mean sessions per active day: KS distance and sd ratio",
                          "tolerance": [T["heterogeneity_ks"], list(T["heterogeneity_sd_ratio"])],
                          "within_tolerance": ok(ks, T["heterogeneity_ks"]) and
                          T["heterogeneity_sd_ratio"][0] <= sdr <= T["heterogeneity_sd_ratio"][1]},
        "long_horizon_drift": {"daily_active_share_mean_abs_diff": drift, "tolerance": T["drift_mean_abs"],
                               "within_tolerance": ok(drift, T["drift_mean_abs"]),
                               "sim": sim["daily_active_share"], "data": ref["daily_active_share"],
                               "beyond_data": sim.get("beyond_data")}}


# ====================================================================== model-only driver
def _weekday_flag(day: int, calendar_d0: str) -> float:
    import datetime as _dt
    return float((_dt.date.fromisoformat(calendar_d0) + _dt.timedelta(days=int(day))).weekday() >= 5)


def simulate_models(ref: dict, beyond_days: int = 60, seed: int = 0) -> dict:
    """Forward-simulate daily_return@1 (activity) and extra_sessions@1 (sessions per active day) from the burn-in
    history of the reference individuals over the comparison window and `beyond_days` more days. Feed exposures of
    a simulated session are the individual's own mean exposures per session over its burn-in (its state; the
    session models have no exposure output). Returns `activity_stats` of the window plus the beyond-data drift."""
    ret, ses = runtime.get("daily_return@1"), runtime.get("extra_sessions@1")
    rng = np.random.default_rng(seed)
    d0, H = ref["d0"], ref["horizon"]
    n = ref["active"].shape[0]
    D = d0 + H + beyond_days
    A = np.zeros((n, D), bool); S = np.zeros((n, D)); R = np.zeros((n, D))
    A[:, :d0], S[:, :d0], R[:, :d0] = ref["active"][:, :d0], ref["sessions"][:, :d0], ref["rows"][:, :d0]
    hs, ha = S[:, :d0].sum(1), A[:, :d0].sum(1)
    rows_per_session = np.where(hs > 0, R[:, :d0].sum(1) / np.maximum(hs, 1), 0.0)
    for d in range(d0, D):
        lo = max(d - 7, 0)
        hist = A[:, :d]
        any_ = hist.any(1)
        last = np.full(n, 14.0)
        idx_last = d - 1 - np.argmax(hist[:, ::-1], axis=1)
        last[any_] = np.minimum(d - idx_last[any_], 14)
        inp = {"active_days_7d": A[:, lo:d].sum(1).astype(float), "days_since_last_active": last,
               "feed_exposures_7d": R[:, lo:d].sum(1), "sessions_7d": S[:, lo:d].sum(1),
               "active_yesterday": A[:, d - 1].astype(float),
               "weekend": np.full(n, _weekday_flag(d, ref["calendar_d0"]))}
        act = ret.sample(inp, rng)
        extra = ses.sample({"extra_sessions_observed": np.maximum(hs - ha, 0.0), "active_days_observed": ha}, rng)
        A[:, d] = act
        S[:, d] = np.where(act, 1 + extra, 0)
        R[:, d] = np.where(act, rng.poisson(np.maximum(rows_per_session * S[:, d], 0.0)), 0)
        hs = hs + S[:, d]; ha = ha + A[:, d]
    st = activity_stats(A, S, d0, H)
    share = A[:, d0:].mean(0)
    st["beyond_data"] = {"days": int(D - d0), "first_week_share": float(share[:7].mean()),
                         "last_week_share": float(share[-7:].mean()),
                         "abs_change": float(abs(share[-7:].mean() - share[:7].mean())),
                         "tolerance": TOLERANCE["stationarity_abs"],
                         "within_tolerance": bool(abs(share[-7:].mean() - share[:7].mean())
                                                  <= TOLERANCE["stationarity_abs"]),
                         "note": "beyond 2022-05-08 there is no data: stationarity of the simulated daily active "
                                 "share is reported, not validated"}
    st["models"] = [ret.id, ses.id]
    return st


def model_only_reports(horizon: int = 7, beyond_days: int = 60, seed: int = 0) -> dict:
    ref = kuairand_reference(horizon=horizon)
    rs = activity_stats(ref["active"], ref["sessions"], ref["d0"], ref["horizon"])
    sim = simulate_models(ref, beyond_days=beyond_days, seed=seed)
    out = compare(sim, rs)
    out["_meta"] = {"driver": "model_only", "reference": ref["source"], "individuals": int(rs["n"]),
                    "window_days": horizon, "beyond_days": beyond_days, "models": ["daily_return@1",
                                                                                    "extra_sessions@1"]}
    return out


# ====================================================================== which recorded decisions count
POP_TABLE = "pop"
PARTICIPANT_MODEL = "participant"
# Population-table batches that are not individual decisions (state set up or kept by the substrate). Everything
# else on the population table is a decision or a behavioural update and needs decision provenance (SP-5.0). Each
# entry: (kind or kind prefix ending in "."/"*", reason).
NON_DECISION_KINDS = (
    ("pop.init", "genesis of the individuals (generated initial state)"),
    ("popweb.genesis", "genesis of the individuals' known sites"),
    ("work.genesis", "genesis of the individuals' employment state"),
    ("crowd.init", "genesis of the social-scale state"),
    ("pop_adaptive.assign", "research-design assignment of adaptive participants to policy groups"),
    ("pop_strategic.assign", "research-design assignment of strategic participants to policy groups"),
    ("attention.*", "attention ledger: minutes an activity used (consequence of the session decision) and the daily "
                    "reset of the budget"),
    ("pop.wake", "scheduling: an individual's next activation set by an event addressed to it"),
    ("pop_wake", "scheduling (interface name of pop.wake)"),
    ("wake.*", "scheduling of the next activation by a platform notification"),
    ("pop.deposit", "bank bookkeeping: accrued income materialised into the individual's account"),
    ("feed.inbox_seen", "delivery bookkeeping: inbox ring marked as shown"),
    ("feed.inbox_read", "delivery bookkeeping: inbox ring marked as read"),
    ("share.deliver", "delivery of another individual's share (that share is the decision)"),
    ("feed.activity", "ring buffer of contacts' activity (consequence of their decisions)"),
    ("work.day_started", "bookkeeping: the work day an individual started"),
    ("work.shop_woken", "bookkeeping: a planned shopping wake-up consumed"),
)


def non_decision_reason(kind: str) -> str | None:
    for k, why in NON_DECISION_KINDS:
        if (k.endswith("*") and kind.startswith(k[:-1])) or kind == k:
            return why
    return None


def _events(world) -> list:
    return world.events_range(0)


def _ev_dict(e) -> dict:
    return e if isinstance(e, dict) else e.to_dict()


def _batch(world, ref) -> dict:
    from ..substrate import artifacts
    return artifacts.get_array_blob(world, ref if isinstance(ref, str) else ref.payload["ref"])


def _serving() -> dict:
    """{runtime decision id: fitted model id} and {fitted model id: [decision ids]}."""
    by_decision = {sid: sp["model"] for sid, sp in runtime.specs().items()}
    by_model: dict[str, list] = {}
    for sid, mid in sorted(by_decision.items()):
        by_model.setdefault(mid, []).append(sid)
    return {"decision": by_decision, "model": by_model}


def _resolve(prov: dict, fitted: set, serving: dict) -> tuple[str | None, list]:
    """(fitted model id, runtime decision ids that can have produced it) named by a decision's provenance."""
    mid, ver = str(prov.get("model_id", "")), str(prov.get("model_version", ""))
    for c in (mid, f"{mid}@{ver}", f"{mid.split('@')[0]}@{ver}"):
        if c in serving["decision"]:
            return serving["decision"][c], [c]
        if c in fitted:
            return c, list(serving["model"].get(c, []))
    return None, []


class Audit:
    """Decision batches of a world classified under the population's provenance rules, batch by batch, with the
    runtime cross-check. Build it once per measurement: `audit(world)`."""

    def __init__(self, world, usage: dict | None = None):
        from .registry import list_models
        evs = _events(world)
        self.world = world
        self.usage = dict(runtime.usage() if usage is None else usage)
        fitted, serving = set(list_models()), _serving()
        listed: dict[str, np.ndarray] = {}
        cmd_actor: dict[int, tuple] = {}
        for e in evs:
            d = _ev_dict(e)
            p = d["payload"]
            if d["type"] == "cols.batch" and p["table"] == POP_TABLE and p["kind"] == "pop.session":
                refs = (p.get("extra") or {}).get("decisions") or []
                if refs:
                    cohort = _batch(world, p["ref"])["idx"].astype(np.int64)
                    for r in refs:
                        listed[r] = cohort
            elif d["type"] == "cmd" and p.get("n") is not None:
                cmd_actor[int(p["n"])] = (d["actor"], bool(p.get("system")))
        models = world.state.tables.get("pop_decision_models", {})
        from ..substrate import ids
        self.batches = []
        for e in evs:
            d = _ev_dict(e)
            if d["type"] != "cols.batch":
                continue
            p = d["payload"]
            prov = (p.get("extra") or {}).get("decision")
            on_pop = p["table"] == POP_TABLE
            nd = non_decision_reason(p["kind"]) if on_pop else None
            if not on_pop and not isinstance(prov, dict):
                continue                       # not a population-table batch and no decision claimed
            rec = {"seq": d["seq"], "t": d["t"], "table": p["table"], "kind": p["kind"], "ref": p["ref"],
                   "non_decision": nd, "provenance": prov if isinstance(prov, dict) else None,
                   "experiment": (p.get("extra") or {}).get("experiment")}
            idx = _batch(world, p["ref"])["idx"].astype(np.int64)
            rec["rows"] = n = int(idx.size)
            rec["counted_rows"] = 0
            if not isinstance(prov, dict):
                rec["class"] = "missing"
            else:
                mk = f"{prov.get('model_id', '')}@{prov.get('model_version', '')}"
                rec["model_key"] = mk
                rec["fitted"], rec["decisions"] = _resolve(prov, fitted, serving)
                if mk not in models:
                    rec["class"] = "forced"
                    rec["why"] = "decision model not registered in this world"
                elif p["ref"] in listed and prov.get("model_id") != PARTICIPANT_MODEL:
                    if on_pop:
                        inside = np.isin(idx, listed[p["ref"]])
                        rec["counted_rows"] = int(inside.sum())
                        rec["class"] = "organic" if inside.all() else "partly_forced"
                        rec["inside"] = inside
                    else:                          # relation-table rows are not individuals: listing only
                        rec["counted_rows"] = n
                        rec["class"] = "organic"
                else:
                    ca = cmd_actor.get(d["cmd"]) if d.get("cmd") is not None else None
                    i = ids.index(ca[0]) if ca is not None else None
                    if ca is not None and not ca[1] and i is not None and \
                            prov.get("model_id") == PARTICIPANT_MODEL and (not on_pop or np.all(idx == i)):
                        rec["class"] = "participant"
                        rec["counted_rows"] = n
                    else:
                        rec["class"] = "forced"
                        rec["why"] = "not recorded by the population's own session nor by the individual's command"
            self.batches.append(rec)
        # the runtime cross-check: rows claimed by counted batches per runtime decision vs rows it evaluated
        claimed: dict[str, int] = {}
        for b in self.batches:
            if b["class"] in ("organic", "partly_forced") and b.get("fitted"):
                key = "|".join(sorted(b["decisions"])) or b["fitted"]
                claimed[key] = claimed.get(key, 0) + b["counted_rows"]
        self.claimed = claimed
        self.backed = {}
        for key, rows in claimed.items():
            ev = sum(self.usage.get(sid, 0) for sid in key.split("|") if sid in serving["decision"])
            self.backed[key] = {"claimed_rows": int(rows), "runtime_rows": int(ev), "ok": bool(ev > 0 and ev >= rows)}
        for b in self.batches:
            if b.get("fitted"):
                key = "|".join(sorted(b["decisions"])) or b["fitted"]
                b["runtime_backed"] = bool(self.backed.get(key, {}).get("ok"))
        self._consistency(evs)

    def _consistency(self, evs) -> None:
        """The batch-by-batch classification must agree with the population's own evaluator check on the population
        table (same events, every kind); a divergence means one of the two rules changed."""
        from ..population import check_provenance
        cp = check_provenance(self.world, events=evs, prefixes=("",))
        mine = [b for b in self.batches if b["table"] == POP_TABLE]
        organic = sum(b["counted_rows"] for b in mine if b["class"] in ("organic", "partly_forced"))
        part = sum(b["counted_rows"] for b in mine if b["class"] == "participant")
        missing = sum(1 for b in mine if b["class"] == "missing")
        got = {"organic_population_rows": organic, "participant_rows": part, "missing_provenance_batches": missing,
               "batches": len(mine)}
        bad = {k: (v, cp[k]) for k, v in got.items() if cp[k] != v}
        if bad:
            raise RuntimeError(f"inworld.Audit diverges from population.check_provenance: {bad}")
        self.check_provenance = cp

    # ------------------------------------------------------------------ views
    def counted(self, model_ids=None, experiment: str | None = None) -> list[dict]:
        """Organic or participant decision batches backed by the runtime, optionally naming one of `model_ids`
        (runtime decision ids or fitted ids) and labelled with an experiment id."""
        want = None if model_ids is None else set(model_ids)
        out = []
        for b in self.batches:
            if b["class"] not in ("organic", "partly_forced", "participant") or not b.get("runtime_backed"):
                continue
            if want is not None and not (want & ({b["fitted"]} | set(b["decisions"]) | {b["model_key"]})):
                continue
            if experiment is not None and (b.get("experiment") or {}).get("id") != experiment:
                continue
            out.append(b)
        return out

    def rows(self, b: dict) -> dict:
        """The batch's arrays restricted to its counted rows (rows outside the session cohort dropped)."""
        arr = _batch(self.world, b["ref"])
        keep = b.get("inside")
        if keep is None or keep.all():
            return arr
        return {k: (v[keep] if getattr(v, "shape", ()) and v.shape[0] == keep.size else v) for k, v in arr.items()}

    def fitted_used(self) -> list[str]:
        return sorted({b["fitted"] for b in self.counted() if b.get("fitted")})

    def decisions_without_provenance(self) -> list[dict]:
        return [b for b in self.batches if b["table"] == POP_TABLE and b["non_decision"] is None and
                b["class"] == "missing"]


def audit(world, usage: dict | None = None) -> Audit:
    return Audit(world, usage)


# ====================================================================== in-world driver
def world_activity(world, horizon: int = 7, au: Audit | None = None) -> dict:
    """Per-individual daily activity [n, days] from the world's counted daily-return and session decisions."""
    from ..substrate import columns
    au = au or audit(world)
    act = au.counted(ACTIVITY_MODELS)
    if not act:
        raise NotWired("no counted daily-activity decision (organic, runtime-backed, daily_return@1 / "
                       "return_daily@1 provenance)")
    n = columns.rows(world, POP_TABLE)
    days = int(max(b["t"] for b in act) // DAY) + 1
    A = np.zeros((n, days), bool); S = np.zeros((n, days))
    for b in act:
        r = au.rows(b)
        if "v.active" not in r:
            raise NotWired(f"daily-return decision batch {b['kind']} has no column 'active'")
        A[r["idx"], int(b["t"] // DAY)] |= r["v.active"].astype(bool)
    for b in au.counted(SESSION_MODELS):
        r = au.rows(b)
        if "v.sessions" in r:
            S[r["idx"], int(b["t"] // DAY)] = r["v.sessions"]
    S = np.where(A & (S == 0), 1, S)
    d0 = max(days - horizon, 7)
    if d0 + horizon > days:
        raise NotWired(f"the world ran {days} days with daily-return decisions; need {7 + horizon}")
    return {"active": A, "sessions": S, "d0": d0, "horizon": horizon}


def in_world_reports(world, horizon: int = 7, au: Audit | None = None) -> dict:
    wa = world_activity(world, horizon, au)
    ref = kuairand_reference(horizon=horizon)
    rs = activity_stats(ref["active"], ref["sessions"], ref["d0"], ref["horizon"])
    ws = activity_stats(wa["active"], wa["sessions"], wa["d0"], wa["horizon"])
    out = compare(ws, rs)
    out["_meta"] = {"driver": "in_world", "individuals": int(ws["n"]), "window_days": horizon,
                    "reference": ref["source"]}
    return out


# ====================================================================== in-world randomised experiments
def kuairand_random_exposure_reference() -> dict:
    """Open rate of KuaiRand's randomly inserted items (held-out test users), overall and for items of the user's
    own topics (user_tag_lift > 0) vs others; with standard errors."""
    from . import fit_e1, kuairand as K
    raw = K.load_raw()
    cf = K.choice_frame(raw)
    m = fit_e1.content_masks(cf)["rnd_test"]
    y = cf.loc[m, "consumed"].to_numpy(float)
    own = cf.loc[m, "user_tag_lift"].to_numpy(float) > 0
    p1, p0 = y[own].mean(), y[~own].mean()
    lor = math.log(p1 / (1 - p1)) - math.log(p0 / (1 - p0))
    se = math.sqrt(1 / (own.sum() * p1 * (1 - p1)) + 1 / ((~own).sum() * p0 * (1 - p0)))
    return {"open_rate": float(y.mean()), "open_rate_se": float(math.sqrt(y.mean() * (1 - y.mean()) / y.size)),
            "own_topic_log_odds_ratio": lor, "own_topic_log_odds_ratio_se": se, "rows": int(y.size)}


def random_exposure_experiment(world, experiment: str = "random_exposure", au: Audit | None = None) -> dict:
    au = au or audit(world)
    ev = au.counted(("content_open@1", "content_consumption@1"), experiment)
    if not ev:
        raise NotWired("no counted content_open@1 decision labelled with a random-exposure experiment")
    ys, own = [], []
    for b in ev:
        if (b.get("experiment") or {}).get("arm") != "random":
            continue
        r = au.rows(b)
        ys.append(r["v.opened"].astype(float))
        own.append(r["v.own_topic"].astype(bool) if "v.own_topic" in r else np.zeros(r["idx"].size, bool))
    if not ys:
        raise NotWired("the random-exposure experiment recorded no row of arm 'random'")
    y, o = np.concatenate(ys), np.concatenate(own)
    ref = kuairand_random_exposure_reference()
    p = y.mean()
    z_rate = (p - ref["open_rate"]) / math.sqrt(p * (1 - p) / y.size + ref["open_rate_se"] ** 2)
    out = {"rows": int(y.size), "open_rate": float(p), "reference": ref, "z_open_rate": float(z_rate)}
    if o.any() and (~o).any():
        p1, p0 = y[o].mean(), y[~o].mean()
        lor = math.log(p1 / (1 - p1)) - math.log(p0 / (1 - p0))
        se = math.sqrt(1 / (o.sum() * p1 * (1 - p1)) + 1 / ((~o).sum() * p0 * (1 - p0)))
        out["own_topic_log_odds_ratio"] = lor
        out["z_log_odds_ratio"] = (lor - ref["own_topic_log_odds_ratio"]) / math.sqrt(
            se ** 2 + ref["own_topic_log_odds_ratio_se"] ** 2)
    out["matching"] = bool(abs(out["z_open_rate"]) < 1.96 and abs(out.get("z_log_odds_ratio", 99.0)) < 1.96)
    return out


def _attention_profile(c: dict, abar: np.ndarray, theta: float) -> float:
    """E5 profile log-likelihood with stratum k's displayed-complier visit rate mu0 + abar_k (q(mu0) - mu0)."""
    from . import fit_e5 as E5
    from .predictors import _sigmoid
    tot = 0.0
    for k, ck in enumerate(E5._strata(c)):
        a = float(abar[k])

        def qfun(v, a=a):
            m = _sigmoid(v)
            return np.clip(m + a * (_sigmoid(v + theta) - m), 1e-9, 1 - 1e-9)
        tot += E5._stratum_profile("M1_constant_log_odds", [theta], ck, qfun=qfun)[0]
    return tot


def ad_holdout_estimate(z, displayed, visit, organic_visit_probability, attended_probability,
                        reference_attention: float = 1.0, strata: int = 5) -> dict:
    """The in-world analysis of a randomised-eligibility campaign: strata of the recorded organic visit probability,
    eligibility as the instrument for display, one log-odds shift per attended display (each stratum's
    displayed-complier effect scaled by the mean A / A_ref of its displayed rows) with profiled nuisances, its
    profile-likelihood 95% interval, the unadjusted shift (attention ignored) and the intention-to-treat difference."""
    from scipy.optimize import brentq, minimize_scalar
    from . import fit_e5 as E5
    z = np.asarray(z, bool)
    D = np.asarray(displayed, bool)
    y = np.asarray(visit, float)
    p = np.asarray(organic_visit_probability, float)
    A = np.asarray(attended_probability, float) / float(reference_attention)
    viol = {"displayed_in_holdout": int(np.count_nonzero(D & ~z)),
            "attention_without_display": int(np.count_nonzero(~D & (A > 0)))}
    D = D & z
    edges = np.quantile(p, np.linspace(0, 1, strata + 1))[1:-1]
    s = np.searchsorted(edges, p, side="right")
    K = edges.size + 1
    fr = {"z": z, "D": D, "y": y, "conv": np.zeros(y.size), "s": s}
    c = E5.cells(fr, K)
    nd = np.bincount(s[D], minlength=K).astype(float)
    abar = np.where(nd > 0, np.bincount(s[D], weights=A[D], minlength=K) / np.maximum(nd, 1), 0.0)
    if not np.all(c["nte"] > 0) or not np.all(c["ntn"] > 0) or not np.all(c["nc"] > 0) or np.any(abar <= 0):
        raise NotWired(f"the ad-holdout experiment has empty cells (control {c['nc'].tolist()}, displayed "
                       f"{c['nte'].tolist()}, eligible not displayed {c['ntn'].tolist()}, attention {abar.tolist()})")
    f = lambda t: -_attention_profile(c, abar, t)            # noqa: E731
    r = minimize_scalar(f, bounds=(0.0, 8.0), method="bounded", options={"xatol": 1e-4})
    th, ll = float(r.x), -float(r.fun)
    g = lambda t: 2 * (ll + f(t)) - 3.841458820694124         # noqa: E731 - profile LR = chi2_1(0.95)
    lo = 0.0 if g(0.0) <= 0 else float(brentq(g, 0.0, th)) if th > 0 else 0.0
    hi = float(brentq(g, th, 8.0)) if g(8.0) > 0 else 8.0
    raw = float(E5.fit_theta("M1_constant_log_odds", c, 0.0)[0])
    itt = float(y[z].mean() - y[~z].mean())
    p_bal = abs(float(np.mean(p[z])) - float(np.mean(p[~z]))) / max(float(np.std(p)) * math.sqrt(
        1 / max(z.sum(), 1) + 1 / max((~z).sum(), 1)), 1e-12)
    return {"rows": int(z.size), "eligible": int(z.sum()), "displayed": int(D.sum()), "strata": int(K),
            "mean_attention_displayed": abar.tolist(), "shift_attention_adjusted": th,
            "shift_ci95": [lo, hi], "shift_unadjusted": raw, "itt_visit": itt,
            "balance_z_organic_probability": float(p_bal), "protocol_violations": viol}


def ad_holdout_experiment(world, experiment: str = "ad_holdout", au: Audit | None = None) -> dict:
    """The E5 analysis of an in-world randomised campaign (`ad_holdout_estimate` on the counted ad_response@2
    decisions labelled with `experiment`), matched against Criteo's fitted shift by a two-sample z test."""
    from .registry import load, report
    au = au or audit(world)
    ev = au.counted(("ad_response@2", "ad_exposure_response@1"), experiment)
    if not ev:
        raise NotWired("no counted ad_response@2 decision labelled with an ad-holdout experiment")
    cols = {"z": [], "D": [], "y": [], "p0": [], "A": []}
    for b in ev:
        r = au.rows(b)
        arm = (b.get("experiment") or {}).get("arm")
        if arm not in ("eligible", "holdout"):
            continue
        need = [c for c in ("v.displayed", "v.visit", "v.organic_visit_probability", "v.attended_probability")
                if c not in r]
        if need:
            raise NotWired(f"ad-holdout decision batch {b['kind']} misses columns {need}")
        k = r["idx"].size
        cols["z"].append(np.full(k, arm == "eligible"))
        cols["D"].append(r["v.displayed"].astype(bool))
        cols["y"].append(r["v.visit"].astype(float))
        cols["p0"].append(r["v.organic_visit_probability"].astype(float))
        cols["A"].append(r["v.attended_probability"].astype(float))
    if not cols["z"]:
        raise NotWired("the ad-holdout experiment recorded no row of arm 'eligible' or 'holdout'")
    ref_att = float(runtime.specs()["ad_response@2"]["params"].get("reference_attention", 1.0))
    est = ad_holdout_estimate(*(np.concatenate(cols[k]) for k in ("z", "D", "y", "p0", "A")),
                              reference_attention=ref_att)
    return {**est, **match_criteo(est), "reference_attention": ref_att}


def match_criteo(est: dict) -> dict:
    from .registry import load, report
    ci = report("ad_exposure_response")["heldout_extra"]["kept_theta_ci95_fit_bootstrap"][0]
    a = float(load("ad_exposure_response").predictor.a)
    se_c = (ci[1] - ci[0]) / 3.92
    se_w = (est["shift_ci95"][1] - est["shift_ci95"][0]) / 3.92
    zz = (est["shift_attention_adjusted"] - a) / math.sqrt(se_c ** 2 + se_w ** 2)
    return {"criteo_shift": a, "criteo_ci95": ci, "z_shift": float(zz), "matching": bool(abs(zz) < 1.96)}


# ====================================================================== SP-5.0 measurements on a world
def provenance_metrics(world, au: Audit | None = None) -> dict:
    """Fitted models whose organic, runtime-backed decisions a world recorded; register claims of runtime use that
    fail verification or that this world did not exercise; population decision batches without provenance."""
    from collections import Counter
    from . import fidelity
    au = au or audit(world)
    used = au.fitted_used()
    claims = fidelity.inworld_claims()
    unverified = {rid: c for rid, c in claims.items()
                  if not c["ok"] or (c["kind"] == "default" and c["model"] not in used)}
    without = au.decisions_without_provenance()
    cls = Counter(b["class"] for b in au.batches if b["provenance"] is not None)
    return {"fitted_models_used_in_world": len(used),
            "claimed_runtime_uses_unverified": len(unverified),
            "decision_events_without_model_provenance": len(without),
            "decision_events_with_model_provenance": int(sum(cls.values())),
            "runtime_rows_evaluated": int(sum(au.usage.values())),
            "_used": used, "_without_by_kind": dict(Counter(b["kind"] for b in without)),
            "_provenance_classes": dict(cls),
            "_forced_with_fitted_provenance": dict(Counter(b["model_key"] for b in au.batches
                                                           if b["class"] == "forced" and b.get("fitted"))),
            "_runtime_backing": au.backed, "_unverified_claims": {k: v["problems"] or ["not exercised in this world"]
                                                                  for k, v in unverified.items()},
            "_check_provenance": au.check_provenance}


# ====================================================================== card scope of in-world uses
WORLD_PACKAGES = ("population", "social", "commerce", "economy", "innovation", "civ", "domains", "multiagent",
                  "semantic", "services")
WORLD_CALIBRATION_API = ("runtime",)          # the only calibration module world code may use
_FITTED_CALLS = ("predict", "predict_proba", "candidate_predict", "respond", "sample_types", "response_types")
_CHECKPOINT_STRINGS = ("calibration/models", "params.npz", "card.json")


def _docstring_nodes(tree) -> set:
    import ast
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and \
                    isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _resolve_from(module: str | None, level: int, package: str) -> str:
    if not level:
        return module or ""
    parts = package.split(".")
    base = parts[:len(parts) - level + 1] if level > 1 else parts
    return ".".join(base + ([module] if module else []))


def scan_source(text: str, package: str) -> list[tuple[int, str]]:
    """Static uses of fitted models outside the runtime contract in one world source file (`package` = the file's
    package, e.g. "crucible.population"), from its syntax tree (comments and docstrings never count):

    * an import of any crucible.calibration module other than `runtime`, or of a name from the calibration package
      other than `runtime` (`load`, `registry`, ...), absolute, relative or aliased;
    * an attribute of the calibration package other than `runtime` (`cal.load`, `getattr(cal, "load")`);
    * importlib / __import__ of a calibration module other than runtime;
    * `evaluate_features(...)` (held-out evaluation in feature space: bypasses the input contract and, with
      checkpoint=True, the contender the review serves);
    * a prediction call on a `.fitted` checkpoint (`x.fitted.predict(...)`);
    * a string naming checkpoint files (calibration/models, params.npz, card.json)."""
    import ast
    tree = ast.parse(text)
    docs = _docstring_nodes(tree)
    hits: list[tuple[int, str]] = []
    pkg_alias: set[str] = set()
    CAL = "crucible.calibration"

    def bad_module(m: str) -> bool:
        return (m == CAL + ".registry" or m.startswith(CAL + ".")) and m not in [f"{CAL}.{x}" for x in
                                                                                  WORLD_CALIBRATION_API]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if bad_module(a.name):
                    hits.append((node.lineno, f"imports {a.name}"))
                elif a.name == CAL:
                    pkg_alias.add(a.asname or "crucible")
        elif isinstance(node, ast.ImportFrom):
            mod = _resolve_from(node.module, node.level, package)
            if bad_module(mod):
                hits.append((node.lineno, f"imports from {mod}"))
            elif mod == CAL:
                for a in node.names:
                    if a.name not in WORLD_CALIBRATION_API:
                        hits.append((node.lineno, f"imports {a.name} from {CAL}"))
            elif mod == "crucible" or mod.endswith(".crucible"):
                for a in node.names:
                    if a.name == "calibration":
                        pkg_alias.add(a.asname or a.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            v = node.value
            if isinstance(v, ast.Name) and v.id in pkg_alias and node.attr not in WORLD_CALIBRATION_API + (
                    "calibration",):
                hits.append((node.lineno, f"uses calibration.{node.attr}"))
            if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "crucible" and \
                    v.attr == "calibration" and node.attr not in WORLD_CALIBRATION_API:
                hits.append((node.lineno, f"uses crucible.calibration.{node.attr}"))
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "getattr" and len(node.args) >= 2 and \
                    isinstance(node.args[0], ast.Name) and node.args[0].id in pkg_alias and \
                    isinstance(node.args[1], ast.Constant) and node.args[1].value not in WORLD_CALIBRATION_API:
                hits.append((node.lineno, f"getattr(calibration, {node.args[1].value!r})"))
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                m = node.args[0].value
                if m.startswith(".") and node.args[1:] and isinstance(node.args[1], ast.Constant):
                    m = _resolve_from(m.lstrip("."), len(m) - len(m.lstrip(".")), str(node.args[1].value))
                if (m == CAL or bad_module(m)) and m != CAL + ".runtime":
                    hits.append((node.lineno, f"imports {m} dynamically"))
            if name == "evaluate_features":
                hits.append((node.lineno, "evaluate_features (feature-space evaluation outside the contract)"))
            if isinstance(f, ast.Attribute) and f.attr in _FITTED_CALLS and isinstance(f.value, ast.Attribute) \
                    and f.value.attr == "fitted":
                hits.append((node.lineno, f"calls .fitted.{f.attr} (the checkpoint behind a decision model)"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            if any(k in node.value for k in _CHECKPOINT_STRINGS):
                hits.append((node.lineno, f"names checkpoint files ({node.value[:60]!r})"))
    return sorted(set(hits))


def static_scope_scan() -> dict:
    """{world source file: [(line, reason)]} over every world package."""
    import os
    from .registry import CRUCIBLE_ROOT
    base = os.path.join(CRUCIBLE_ROOT, "src", "crucible")
    hits = {}
    for pkg in WORLD_PACKAGES:
        for root, _, files in os.walk(os.path.join(base, pkg)):
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, base)
                package = ".".join(["crucible"] + rel[:-3].split(os.sep)[:-1])
                with open(path, encoding="utf-8") as fh:
                    h = scan_source(fh.read(), package)
                if h:
                    hits[rel] = h
    return hits


def out_of_scope_uses(scope_usage: dict | None = None) -> dict:
    """World uses of fitted models outside their cards' scope:

    * static: world source files that use a fitted checkpoint outside the runtime contract (`scan_source`); each
      file counts once;
    * dynamic (the runtime's scope ledger of this process, `runtime.scope_usage()`; reset it before the world run):
      every scoped decision model evaluated without a declared use (rows tagged "unscoped": its scope cannot be
      verified) counts once per decision id, and every refused out-of-scope attempt counts once per (decision, use).
    """
    static = static_scope_scan()
    led = runtime.scope_usage() if scope_usage is None else scope_usage
    unscoped = {d: u[runtime.UNSCOPED] for d, u in led.get("used", {}).items() if u.get(runtime.UNSCOPED)}
    refused = {f"{d}:{k}": n for d, r in led.get("refused", {}).items() for k, n in r.items()}
    return {"direct_checkpoint_loads": {k: [ln for ln, _ in v] for k, v in static.items()},
            "static_uses": {k: [f"{ln}: {why}" for ln, why in v] for k, v in static.items()},
            "scoped_models_evaluated_without_declared_use": unscoped, "refused_out_of_scope_attempts": refused,
            "out_of_card_scope_uses": len(static) + len(unscoped) + len(refused)}
