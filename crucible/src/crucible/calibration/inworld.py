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
  visits through ad_response@2; the in-world complier log-odds shift (eligibility as the instrument for display,
  the E5 estimator) compared with Criteo's (model ad_exposure_response@1).

What the population must record for the in-world driver (contract, RUNTIME.md "In-world validation"): each daily
activation decision as a population decision batch whose provenance model id is ``daily_return@1`` (or the fitted id
``return_daily@1``) with a boolean column ``active``; sessions per active day with model ``extra_sessions@1`` (or
``session_rate@1``) and an integer column ``sessions``; feed opens with model ``content_open@1`` and a boolean column
``opened``; ad visits with ``ad_response@2`` and boolean columns ``visit``, ``displayed``. A world that records none of
them is reported as not wired (`NotWired`), never as validated.
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import runtime

DAY = 86400
# prespecified tolerances of the in-world reports (set before any in-world run; see the module docstring)
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
    ret = {}
    for lo, hi in HISTORY_BINS:
        m = (hist7 >= lo) & (hist7 <= hi)
        ret[f"{lo}-{hi}"] = {"n": int(m.sum()), "curve": A[m].mean(0).tolist() if m.any() else []}
    return {"n": int(A.shape[0]), "active_days_hist": _hist(A.sum(1), horizon + 1).tolist(),
            "sessions_per_active_day_hist": _hist(per_active, 11).tolist() if per_active.size else [],
            "daily_active_share": A.mean(0).tolist(), "user_active_share": A.mean(1),
            "user_active_share_sd": float(A.mean(1).std()), "retention_by_history": ret}


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
    ks = _ks(sim["user_active_share"], ref["user_active_share"])
    sdr = sim["user_active_share_sd"] / max(ref["user_active_share_sd"], 1e-12)
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
        "heterogeneity": {"user_active_share_ks": ks, "sd_ratio": float(sdr),
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


# ====================================================================== in-world driver
def decision_events(world, model_ids) -> list:
    """cols.batch events whose decision provenance names one of `model_ids` (runtime decision or fitted id)."""
    want = set(model_ids)
    out = []
    for e in world.events_range(0):
        if e.type != "cols.batch":
            continue
        dec = (e.payload.get("extra") or {}).get("decision") or {}
        mid = str(dec.get("model_id", ""))
        ver = str(dec.get("model_version", ""))
        if mid in want or f"{mid}@{ver}" in want or mid.split("@")[0] + "@" + ver in want:
            out.append(e)
    return out


def _batch(world, e) -> dict:
    from ..substrate import artifacts
    return artifacts.get_array_blob(world, e.payload["ref"])


def world_activity(world, horizon: int = 7) -> dict:
    """Per-individual daily activity [n, days] from the world's recorded daily-return and session decisions."""
    from ..substrate import columns
    act_ev = decision_events(world, ACTIVITY_MODELS)
    if not act_ev:
        raise NotWired("no daily-activity decision recorded with daily_return@1 / return_daily@1 provenance")
    n = columns.rows(world, "pop")
    days = int(max(e.t for e in act_ev) // DAY) + 1
    A = np.zeros((n, days), bool); S = np.zeros((n, days))
    for e in act_ev:
        b = _batch(world, e)
        if "v.active" not in b:
            raise NotWired(f"daily-return decision batch {e.payload['kind']} has no column 'active'")
        A[b["idx"], int(e.t // DAY)] |= b["v.active"].astype(bool)
    for e in decision_events(world, SESSION_MODELS):
        b = _batch(world, e)
        if "v.sessions" in b:
            S[b["idx"], int(e.t // DAY)] = b["v.sessions"]
    S = np.where(A & (S == 0), 1, S)
    d0 = max(days - horizon, 7)
    if d0 + horizon > days:
        raise NotWired(f"the world ran {days} days with daily-return decisions; need {7 + horizon}")
    return {"active": A, "sessions": S, "d0": d0, "horizon": horizon}


def in_world_reports(world, horizon: int = 7) -> dict:
    wa = world_activity(world, horizon)
    ref = kuairand_reference(horizon=horizon)
    rs = activity_stats(ref["active"], ref["sessions"], ref["d0"], ref["horizon"])
    ws = activity_stats(wa["active"], wa["sessions"], wa["d0"], wa["horizon"])
    out = compare(ws, rs)
    out["_meta"] = {"driver": "in_world", "individuals": int(ws["n"]), "window_days": horizon,
                    "reference": ref["source"]}
    return out


# ====================================================================== in-world randomised experiments
def _labelled(world, model_ids, experiment: str) -> list:
    return [e for e in decision_events(world, model_ids)
            if ((e.payload.get("extra") or {}).get("experiment") or {}).get("id") == experiment]


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


def random_exposure_experiment(world, experiment: str = "random_exposure") -> dict:
    ev = _labelled(world, ("content_open@1", "content_consumption@1"), experiment)
    if not ev:
        raise NotWired("no content_open@1 decision labelled with a random-exposure experiment")
    ys, own = [], []
    for e in ev:
        b = _batch(world, e)
        if ((e.payload.get("extra") or {}).get("experiment") or {}).get("arm") != "random":
            continue
        ys.append(b["v.opened"].astype(float))
        own.append(b["v.own_topic"].astype(bool) if "v.own_topic" in b else np.zeros(b["idx"].size, bool))
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


def ad_holdout_experiment(world, experiment: str = "ad_holdout") -> dict:
    """The E5 estimator on an in-world randomised campaign: strata of the recorded organic visit probability,
    eligibility as the instrument for display, one log-odds shift with profiled nuisances; matched against the
    fitted shift's 95% interval (Criteo)."""
    from . import fit_e5 as E5
    from .registry import load, report
    ev = _labelled(world, ("ad_response@2", "ad_exposure_response@1"), experiment)
    if not ev:
        raise NotWired("no ad_response@2 decision labelled with an ad-holdout experiment")
    z, D, y, p0 = [], [], [], []
    for e in ev:
        b = _batch(world, e)
        arm = ((e.payload.get("extra") or {}).get("experiment") or {}).get("arm")
        k = b["idx"].size
        z.append(np.full(k, arm == "eligible")); D.append(b["v.displayed"].astype(bool) & (arm == "eligible"))
        y.append(b["v.visit"].astype(float)); p0.append(b["v.organic_visit_probability"].astype(float))
    fr = {"z": np.concatenate(z), "D": np.concatenate(D), "y": np.concatenate(y),
          "conv": np.zeros(sum(map(len, y)))}
    p = np.concatenate(p0)
    edges = np.quantile(p, np.linspace(0, 1, 6))[1:-1]
    fr["s"] = np.searchsorted(edges, p, side="right")
    c = E5.cells(fr, edges.size + 1)
    theta = E5.fit_theta("M1_constant_log_odds", c, 0.0)
    ci = report("ad_exposure_response")["heldout_extra"]["kept_theta_ci95_fit_bootstrap"][0]
    shift = float(theta[0])
    return {"rows": int(fr["z"].size), "in_world_shift": shift, "criteo_shift": float(load("ad_exposure_response").predictor.a),
            "criteo_ci95": ci, "matching": bool(ci[0] <= shift <= ci[1])}


# ====================================================================== SP-5.0 measurements on a world
POPULATION_DECISION_KINDS = ("pop.session", "pop.activation", "popweb.")


def provenance_metrics(world) -> dict:
    """Fitted models whose decisions a world recorded, and population decision batches without provenance."""
    from .registry import list_models
    from . import fidelity
    fitted = set(list_models())
    serving = {sid: sp["model"] for sid, sp in runtime.specs().items()}
    used, with_prov, without = set(), 0, []
    for e in world.events_range(0):
        if e.type != "cols.batch":
            continue
        p = e.payload
        dec = (p.get("extra") or {}).get("decision")
        if dec:
            with_prov += 1
            mid = str(dec.get("model_id", ""))
            cand = {mid, f"{mid}@{dec.get('model_version')}"}
            for c in cand:
                if c in fitted:
                    used.add(c)
                elif c in serving:
                    used.add(serving[c])
        elif p["kind"].startswith(POPULATION_DECISION_KINDS) and p["kind"] not in ("popweb.genesis", "popweb.seen"):
            without.append(p["kind"])
    claims = fidelity.inworld_claims()
    from collections import Counter
    return {"fitted_models_used_in_world": len(used), "claimed_runtime_uses_unverified":
            sum(1 for c in claims.values() if not c["ok"]),
            "decision_events_without_model_provenance": len(without),
            "decision_events_with_model_provenance": with_prov,
            "runtime_rows_evaluated": int(sum(runtime.usage().values())),
            "_used": sorted(used), "_without_by_kind": dict(Counter(without))}


# ====================================================================== card scope of in-world uses
WORLD_PACKAGES = ("population", "social", "commerce", "economy", "innovation", "civ", "domains", "multiagent",
                  "semantic", "services")
_DIRECT_LOAD = re.compile(r"(calibration|registry)\.load\(|from \.\.?calibration import[^\n]*\bload\b|"
                          r"from \.\.?calibration\.registry import[^\n]*\bload\b|cal\.load\b|getattr\(cal, \"load\"")


def out_of_scope_uses() -> dict:
    """World code that uses a fitted model outside its card's scope, measured statically: a world source file that
    loads a fitted checkpoint directly (registry / calibration.load) bypasses the runtime contract, its scope check
    and its feature formulas, so each such file counts; a reference to a runtime decision id is checked against the
    decision's machine-readable scope by the consumer at install (CalibratedOpen.check_scope)."""
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
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                lines = [i + 1 for i, line in enumerate(text.splitlines())
                         if _DIRECT_LOAD.search(line) and not line.lstrip().startswith("#")]
                if lines:
                    hits[os.path.relpath(path, base)] = lines
    return {"direct_checkpoint_loads": hits, "out_of_card_scope_uses": len(hits)}
