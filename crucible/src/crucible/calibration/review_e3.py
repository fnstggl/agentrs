"""E3 review of the kept calibration models (plan `docs/v5/calibration/plans/E3_review_plan.json`, workstream E).

For each kept model: the kept checkpoint (K), its best simple baseline rebuilt from its definition (B), the
simulator's existing rule-based behaviour mapped onto the same held-out rows (S; parameters as in the code, not
refitted) and, as a diagnostic, the rule's form with refitted parameters (SF) are scored on the card's test rows with
the card's primary metric; a long-horizon measure (forward simulation where outputs feed inputs, multi-day stability
otherwise) and the compute cost are measured; the plan's rule decides which contender is kept. Results go to
`docs/v5/calibration/reviews/<model>.json`. Offline code only.

    PYTHONPATH=src python -m crucible.calibration.review_e3 --model return_daily [--model ...]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time

import numpy as np

from . import fitlib as F
from . import metrics as M
from .fit_e4 import DRAWS, dense, measure, paired_boot, parts
from .registry import CRUCIBLE_ROOT, load

PLAN_PATH = "docs/v5/calibration/plans/E3_review_plan.json"
REVIEWS_DIR = os.path.join(CRUCIBLE_ROOT, "docs", "v5", "calibration", "reviews")
BUDGET_US = 100.0


def plan() -> dict:
    with open(os.path.join(CRUCIBLE_ROOT, PLAN_PATH), encoding="utf-8") as fh:
        return json.load(fh)


def timed(fn, n: int, reps: int = 5) -> float:
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) / max(n, 1) * 1e6)


def decide(contenders: dict, lower: bool = True) -> dict:
    """The plan's rule. contenders: name -> {primary, long_horizon (or None), compute_us, parts (or None), C}."""
    sign = 1 if lower else -1
    eligible = [k for k, c in contenders.items() if c.get("eligible_role", True) and c["compute_us"] <= BUDGET_US]
    order = sorted(eligible, key=lambda k: sign * contenders[k]["primary"])
    with_lh = [k for k in eligible if contenders[k].get("long_horizon") is not None]
    ref = min(with_lh, key=lambda k: contenders[k]["long_horizon"]) if with_lh else None
    rejected, kept = {}, None
    for k in order:
        c = contenders[k]
        if ref is not None and k != ref:
            if c.get("long_horizon") is None:
                kept = k
                break
            ci = paired_boot(c["parts"], contenders[ref]["parts"], c["C"], c.get("relative", False))
            if ci[0] > 0:
                rejected[k] = {"versus": ref, "difference_ci95": ci}
                continue
        kept = k
        break
    return {"eligible": eligible, "ordered_by_primary": order, "long_horizon_reference": ref,
            "drift_rejected": rejected, "kept_after_review": kept}


def write(name: str, review: dict) -> str:
    os.makedirs(REVIEWS_DIR, exist_ok=True)
    p = os.path.join(REVIEWS_DIR, f"{name}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(review, fh, indent=1, default=F._j)
        fh.write("\n")
    return p


def _row_ci(rows_a, rows_b, groups) -> list:
    ci = M.cluster_bootstrap(np.asarray(rows_a) - np.asarray(rows_b), groups)
    return [ci["lo"], ci["hi"]]


def assemble(name: str, model_id: str, primary: str, cont: dict, rule_text: str, extra: dict, t0: float,
             kept_id: str) -> dict:
    dec = decide(cont)
    out = {"schema": "crucible.v5.calibration_review/1", "model": model_id, "kept_by_card": kept_id,
           "plan": F.plan_record(PLAN_PATH), "review_commit": F.head_commit(),
           "reviewed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "primary_metric": primary,
           "rule_mapping": rule_text, "budget_us_per_row": BUDGET_US,
           "contenders": {k: {kk: vv for kk, vv in c.items() if kk not in ("parts", "C", "relative")}
                          for k, c in cont.items()},
           **dec, "kept_is_fitted_checkpoint": dec["kept_after_review"] == "K", "extra": extra,
           "seconds": time.time() - t0}
    return out


# ====================================================================== KuaiRand helpers
def kr_rule_inputs(raw, cf):
    """aff (primary-tag cosine with earlier plays), fam, seen, dur and the window competitors' utilities' inputs."""
    from . import kuairand as K
    L = raw["logs"]
    rows = L[(L["tab"] == 1)]
    rows = rows[rows["duration_s"] > 0]
    assert len(rows) == len(cf)
    std = L[L["is_rand"] == 0]
    tag = std["primary_tag"].to_numpy() + 1
    T = int(max(tag.max(), rows["primary_tag"].max() + 1)) + 1
    D = int(L["day"].max()) + 2
    U = int(L["user_id"].max()) + 1
    clk = std["is_click"].to_numpy(float)
    H = np.zeros((U, D, T), np.float32)
    np.add.at(H, (std["user_id"].to_numpy(), std["day"].to_numpy(), tag), clk)
    H = np.cumsum(H, axis=1) - H                                     # plays on earlier days
    ru, rd = rows["user_id"].to_numpy(), rows["day"].to_numpy()
    rt = rows["primary_tag"].to_numpy() + 1
    h = H[ru, rd]
    norm = np.linalg.norm(h, axis=1)
    aff = np.where((norm > 0) & (rt > 0), h[np.arange(len(rt)), rt] / np.maximum(norm, 1e-12), 0.0)
    del H
    fam = (cf["creator_familiarity"].to_numpy() > 0).astype(float)
    seen = (cf["item_seen_before"].to_numpy() > 0).astype(float)
    dur = cf["duration_s"].to_numpy(float)
    t = rows["time_ms"].to_numpy()
    return aff, fam, seen, dur, ru, t


def c2_open_probability(aff, fam, seen, dur, users, t_ms, sat: float = 0.5) -> np.ndarray:
    """C2 open rule with the next `window` rows of the same user within 30 minutes as competitors."""
    from ..social.consumption import BUILTIN as b
    util = (b["open_bias"] + b["open_affinity"] * aff + b["open_familiar"] * fam + b["open_seen"] * seen +
            b["open_sat"] * (sat - 0.5) + b["open_len"] * np.log(np.maximum(dur, 1.0) / 60.0))
    eu = np.exp(np.clip(util, -40, 40))
    order = np.lexsort((t_ms, users))
    eo, uo, to = eu[order], users[order], t_ms[order]
    win = eo.copy()
    for k in range(1, int(b["window"]) + 1):
        nxt_e = np.r_[eo[k:], np.zeros(k)]
        ok = np.r_[(uo[k:] == uo[:-k]) & (to[k:] - to[:-k] <= 30 * 60 * 1000), np.zeros(k, bool)]
        win += np.where(ok, nxt_e, 0.0)
    p = np.empty_like(eu)
    p[order] = eo / (math.exp(b["outside"]) + win)
    return np.clip(p, 1e-6, 1 - 1e-6)


def review_content(name: str) -> dict:
    from . import fit_e1 as E
    from . import kuairand as K
    t0 = time.time()
    f = load(name)
    raw = K.load_raw()
    cf = K.choice_frame(raw)
    cm = E.content_masks(cf)
    tr_key, te_key = ("rnd_train", "rnd_test") if name == "content_consumption" else ("obs_train", "obs_test")
    feats = f.features
    X = cf[feats].to_numpy(float)
    y = cf["is_click"].to_numpy(float)
    te, tr = cm[te_key], cm[tr_key]
    groups = cf.loc[te, "user_id"].to_numpy()
    day = cf.loc[te, "day"].to_numpy()
    pK = f.predict(X[te])
    pop_idx = [feats.index(c) for c in K.POPULARITY_FEATURES]
    bl = F.fit_logistic(X[tr][:, pop_idx], y[tr])
    pB = bl.predict(X[te][:, pop_idx])
    aff, fam, seen, dur, ru, tms = kr_rule_inputs(raw, cf)
    pS_all = c2_open_probability(aff, fam, seen, dur, ru, tms)
    pS = pS_all[te]
    Zs = np.column_stack([aff, fam, seen, np.log(np.maximum(dur, 1.0) / 60.0)])
    sf = F.fit_logistic(Zs[tr], y[tr])
    pSF = sf.predict(Zs[te])
    yt = y[te]
    gi, C = dense(groups)
    di, D = dense(day)
    cont = {}
    rows_K = M.row_log_loss(yt, pK)
    for k, p in (("K", pK), ("B", pB), ("S", pS), ("SF", pSF)):
        pt = parts(p, yt, gi, C, di, D)
        cont[k] = {"primary": M.log_loss(yt, p), "metrics": M.binary_report(yt, p),
                   "difference_vs_K_ci95": _row_ci(M.row_log_loss(yt, p), rows_K, groups),
                   "long_horizon": measure(pt), "parts": pt, "C": C,
                   "eligible_role": k != "SF"}
    n = min(100000, int(te.sum()))
    Xn = X[te][:n]
    from . import predictors as P
    portable = P.KINDS[f.predictor.kind].from_params(f.predictor.params())
    cont["K"]["compute_us"] = timed(lambda: portable.predict(Xn), n)
    cont["B"]["compute_us"] = timed(lambda: bl.predict(Xn[:, pop_idx]), n)
    sl = slice(0, n)
    cont["S"]["compute_us"] = timed(lambda: c2_open_probability(aff[sl], fam[sl], seen[sl], dur[sl], ru[sl], tms[sl]), n)
    cont["SF"]["compute_us"] = timed(lambda: sf.predict(Zs[:n]), n)
    rule = plan()["rule_mappings"][name]
    rep_card = f.metrics.get("log_loss")
    return assemble(name, f.id, "log_loss", cont, rule,
                    {"test_rows": int(te.sum()), "card_test_log_loss": rep_card,
                     "long_horizon_kind": "static: mean over test days of |mean predicted - mean observed|",
                     "SF_coefficients": {"aff": float(sf.w[0] / sf.scale[0]), "fam": float(sf.w[1] / sf.scale[1]),
                                         "seen": float(sf.w[2] / sf.scale[2]),
                                         "log_dur": float(sf.w[3] / sf.scale[3])}},
                    t0, f.card["selection"]["kept"])


def c2_watch_quantiles(aff, fam, seen, dur, consumed, levels, sat: float = 0.5) -> np.ndarray:
    """Exact quantiles of log1p(T) of the C2 watch rule (bounce mixture + truncated exponential) by bisection."""
    from ..social.consumption import BUILTIN as b
    pb = 1 / (1 + np.exp(-(b["bounce_bias"] + b["bounce_affinity"] * aff)))
    lam = b["hazard_per_s"] * np.exp(b["hazard_affinity"] * aff + b["hazard_familiar"] * fam +
                                     b["hazard_sat"] * (sat - 0.5) + b["hazard_seen"] * seen)

    def cdf(tt):
        bo = np.clip((np.minimum(tt, dur[:, None]) - 1.0) / 5.0, 0, 1)
        bo = np.where(tt >= dur[:, None], 1.0, bo)
        ex = np.where(tt >= dur[:, None], 1.0, 1 - np.exp(-lam[:, None] * tt))
        return pb[:, None] * bo + (1 - pb[:, None]) * ex
    lv = np.asarray(levels)[None, :]
    lo = np.zeros((len(aff), lv.shape[1]))
    hi = np.tile(dur[:, None], (1, lv.shape[1]))
    for _ in range(50):
        mid = (lo + hi) / 2
        below = cdf(mid) < lv
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    Q = np.log1p(hi)
    Q[consumed <= 0] = 0.0
    return Q


def review_watch(name: str = "watch_duration") -> dict:
    from . import fit_e1 as E
    from . import kuairand as K
    t0 = time.time()
    f = load(name)
    levels = np.asarray(f.card["quantile_levels"])
    raw = K.load_raw()
    cf = K.choice_frame(raw)
    wm = E.watch_masks(cf)
    te, tr = wm["test"], wm["train"]
    X = cf[f.features].to_numpy(float)
    y = np.log1p(cf["play_s"].to_numpy(float))
    QK = f.predict(X[te])
    # B1: binned quantiles by duration decile x consumed (rebuilt from the card's baseline definition)
    j_d, j_c = f.features.index("item_log_duration_s"), f.features.index("consumed")
    bq = F.fit_binned_quantiles(X[tr], y[tr], j_d, levels, n_bins=10, group_col=j_c)
    QB = bq.predict(X[te])
    aff, fam, seen, dur, ru, tms = kr_rule_inputs(raw, cf)
    consumed = cf["consumed"].to_numpy(float)
    QS = c2_watch_quantiles(aff[te], fam[te], seen[te], dur[te], consumed[te], levels)
    yt = y[te]
    groups = cf.loc[te, "user_id"].to_numpy()
    gi, C = dense(groups)
    di, D = dense(cf.loc[te, "day"].to_numpy())
    a, bq_ = list(np.round(levels, 6)).index(0.1), list(np.round(levels, 6)).index(0.9)
    rows_K = M.pinball_rows(yt, QK, levels)
    cont = {}
    for k, Q in (("K", QK), ("B", QB), ("S", QS)):
        inside = ((yt >= Q[:, a]) & (yt <= Q[:, bq_])).astype(float)
        pt = parts(inside, 0.8 * np.ones(yt.size), gi, C, di, D)
        cont[k] = {"primary": M.pinball(yt, Q, levels), "pit_ks": M.ks_uniform(M.pit_fast(yt, Q, levels)),
                   "coverage_80": float(inside.mean()),
                   "difference_vs_K_ci95": _row_ci(M.pinball_rows(yt, Q, levels), rows_K, groups),
                   "long_horizon": measure(pt), "parts": pt, "C": C}
    # consumed rows only (the simulator's watch rule applies to opened items)
    cm_ = consumed[te] > 0
    extra = {"consumed_rows_only": {k: M.pinball(yt[cm_], Q[cm_], levels) for k, Q in (("K", QK), ("B", QB), ("S", QS))},
             "long_horizon_kind": "static: mean over test days of |80% interval coverage - 0.8|",
             "S_quantiles": "exact quantiles of the C2 mixture by bisection (instead of the plan's 2,000-draw Monte "
                            "Carlo; same estimand without sampling error)"}
    n = min(100000, int(te.sum()))
    from . import predictors as P
    portable = P.KINDS[f.predictor.kind].from_params(f.predictor.params())
    cont["K"]["compute_us"] = timed(lambda: portable.predict(X[te][:n]), n)
    cont["B"]["compute_us"] = timed(lambda: bq.predict(X[te][:n]), n)
    cont["S"]["compute_us"] = timed(lambda: c2_watch_quantiles(aff[:n], fam[:n], seen[:n], dur[:n], consumed[:n], levels),
                                    n, reps=3)
    return assemble(name, f.id, "pinball", cont, plan()["rule_mappings"][name], extra, t0, f.card["selection"]["kept"])


# ---------------------------------------------------------------- population rhythm (pop) rule
def pop_rhythm(days: int = 30, n: int = 20000, seed: int = 0) -> dict:
    """Simulate the default population's activation rhythm and session_start decision (crucible.population.core)
    and return P(active on a day) and the mean extra sessions per active day over days 10..days."""
    from ..population import core as PC
    from ..population import models as PM
    rng = np.random.default_rng(seed)
    mix = PC.DEFAULT_MIX
    seg = rng.choice(len(PC.SEGMENTS), size=n, p=np.array([mix[s] for s in PC.SEGMENTS]) / sum(mix.values()))
    heavy = (seg == 1) | (seg == 3)
    employed = rng.random(n) < 0.68
    cs = np.where(employed, np.clip(rng.normal(9 * 60, 50, n), 6 * 60, 13 * 60), 0)
    cl = np.where(employed, np.clip(rng.normal(8 * 60, 40, n), 4 * 60, 10 * 60), 0)
    ds = np.clip(rng.normal(7 * 60, 55, n), 4 * 60, 11 * 60).astype(np.int64)
    dl = np.clip(rng.normal(16 * 60, 50, n), 13 * 60, 18 * 60).astype(np.int64)
    mean_gap = np.exp(rng.normal(np.where(heavy, math.log(7 * 3600), math.log(13 * 3600)), 0.45))
    session_min = np.exp(rng.normal(math.log(18.0), 0.45, n))
    budget = np.exp(rng.normal(np.where(heavy, math.log(120.0), math.log(75.0)), 0.35))
    d = PM._decisions()["session_start"]
    DAY = 86400
    nxt = PC.schedule_times(0, rng.uniform(0.05, 1.0, n) * mean_gap, ds, dl, rng)
    exp_ = np.zeros(n)
    sat = np.full(n, 0.5)
    sessions = np.zeros((n, days))
    left = budget.copy()
    last_day = np.zeros(n, np.int64)
    alive = np.ones(n, bool)
    t_end = days * DAY
    while True:
        due = alive & (nxt < t_end)
        if not due.any():
            break
        i = np.flatnonzero(due)
        now = nxt[i]
        dayi = now // DAY
        newday = dayi != last_day[i]
        left[i] = np.where(newday, budget[i], left[i])
        last_day[i] = dayi
        minute = (now % DAY) // 60
        in_c = (cl[i] > 0) & (((minute - cs[i]) % 1440) < cl[i])
        hour = minute / 60.0
        Fz = {"attention_ratio": left[i] / budget[i], "satisfaction": sat[i], "habit_max": np.zeros(i.size),
              "experience_log": np.log1p(exp_[i]), "in_commitment": in_c.astype(float), "woken": np.zeros(i.size),
              "hour_sin": np.sin(2 * math.pi * hour / 24), "hour_cos": np.cos(2 * math.pi * hour / 24)}
        start = (rng.random(i.size) < d.prob(Fz)) & (left[i] > 0.5)
        sessions[i[start], np.minimum(dayi[start], days - 1)] += 1
        exp_[i[start]] += 1
        left[i[start]] = np.maximum(left[i[start]] - session_min[i[start]], 0)
        gaps = mean_gap[i] * rng.exponential(1.0, i.size) * np.where(start, 1.0, 0.5) / (0.6 + 0.8 * sat[i])
        nxt[i] = PC.schedule_times(0, np.clip(gaps, 600, 3 * DAY), ds[i], dl[i], rng) + now
        alive[i] = nxt[i] < t_end
    S = sessions[:, 10:]
    act = S > 0
    return {"p_active": float(act.mean()), "extra_per_active_day": float((S[act] - 1).mean()),
            "days_measured": int(S.shape[1]), "individuals": n}


def _activity(raw):
    from . import kuairand as K
    users, rows, sess = K.activity_matrices(raw)
    return users, rows, sess


def review_return(name: str = "return_daily") -> dict:
    from . import fit_e1 as E
    from . import kuairand as K
    t0 = time.time()
    f = load(name)
    raw = K.load_raw()
    af = K.activity_frame(raw, K.day_num(2022, 4, 29), K.day_num(2022, 5, 8))
    rm = E.return_masks(af)
    te, tr = rm["test"], rm["train"]
    X = af[f.features].to_numpy(float)
    y = af["active"].to_numpy(float)
    yt = y[te]
    pK = f.predict(X[te])
    base0 = float(y[tr].mean())
    pB = (af.loc[te, "active_days_7d"].to_numpy() + 1) / 9.0                       # B1 own-rate rule (no fit)
    pr = pop_rhythm()
    from ..social.consumption import BUILTIN as b
    p_c2 = 1 - math.exp(-16 * 3600 / float(b["gap_mean_s"]))
    pS1 = np.full(yt.size, min(max(pr["p_active"], 1e-6), 1 - 1e-6))
    pS2 = np.full(yt.size, min(max(p_c2, 1e-6), 1 - 1e-6))
    S_best = "S1" if M.log_loss(yt, pS1) <= M.log_loss(yt, pS2) else "S2"
    groups = af.loc[te, "user_id"].to_numpy()
    rows_K = M.row_log_loss(yt, pK)
    # dynamic forward simulation of test users over 2022-05-01..05-08
    users_all, Rm, Sm = _activity(raw)
    test_users = np.unique(groups)
    d0, d1 = K.day_num(2022, 5, 1), K.day_num(2022, 5, 8)
    act = Rm[test_users] > 0
    hist_act = act[:, :d0]
    mean_rows = np.where(hist_act.sum(1) > 0, (Rm[test_users, :d0]).sum(1) / np.maximum(hist_act.sum(1), 1), np.nan)
    mean_sess = np.where(hist_act.sum(1) > 0, (Sm[test_users, :d0]).sum(1) / np.maximum(hist_act.sum(1), 1), np.nan)
    mean_rows = np.where(np.isnan(mean_rows), np.nanmean(mean_rows), mean_rows)
    mean_sess = np.where(np.isnan(mean_sess), np.nanmean(mean_sess), mean_sess)
    from . import predictors as P
    portable = P.KINDS[f.predictor.kind].from_params(f.predictor.params())

    def sim(pfun, seed=0):
        rng = np.random.default_rng(seed)
        acc = np.zeros((test_users.size, d1 - d0 + 1))
        dist = np.zeros(d1 - d0 + 2)
        for _ in range(DRAWS):
            A = act.astype(float).copy(); Rr = Rm[test_users].copy(); Ss = Sm[test_users].copy()
            for d in range(d0, d1 + 1):
                Xd = K.return_features_from(A > 0, Rr, Ss, d)
                a_ = rng.random(test_users.size) < pfun(Xd)
                A[:, d] = a_; Rr[:, d] = np.where(a_, mean_rows, 0); Ss[:, d] = np.where(a_, mean_sess, 0)
            acc += A[:, d0:d1 + 1] / DRAWS
            dist += np.bincount((A[:, d0:d1 + 1] > 0).sum(1), minlength=d1 - d0 + 2) / DRAWS
        return acc, dist
    obs = act[:, d0:d1 + 1].astype(float)
    obs_dist = np.bincount(obs.sum(1).astype(int), minlength=d1 - d0 + 2)
    funs = {"K": lambda Z: portable.predict(Z), "B": lambda Z: (Z[:, 0] + 1) / 9.0,
            "S": lambda Z: np.full(len(Z), pS1[0] if S_best == "S1" else pS2[0]),
            "B0_constant": lambda Z: np.full(len(Z), base0)}
    cont = {}
    gi = np.arange(test_users.size)
    per = np.tile(np.arange(d1 - d0 + 1), (test_users.size, 1))
    for k, p in (("K", pK), ("B", pB), ("S", pS1 if S_best == "S1" else pS2), ("B0_constant", np.full(yt.size, base0))):
        acc, dist = sim(funs[k])
        pt = parts(acc.ravel(), obs.ravel(), np.repeat(gi, d1 - d0 + 1), test_users.size, per.ravel(), d1 - d0 + 1)
        cont[k] = {"primary": M.log_loss(yt, p), "metrics": M.binary_report(yt, p),
                   "difference_vs_K_ci95": _row_ci(M.row_log_loss(yt, p), rows_K, groups),
                   "long_horizon": measure(pt), "active_days_tv": M.total_variation(dist, obs_dist),
                   "parts": pt, "C": test_users.size}
    n = min(100000, int(te.sum()))
    Xn = X[te][:n]
    cont["K"]["compute_us"] = timed(lambda: portable.predict(Xn), n)
    cont["B"]["compute_us"] = timed(lambda: (Xn[:, 0] + 1) / 9.0, n)
    cont["S"]["compute_us"] = timed(lambda: np.full(n, pS1[0]), n)
    cont["B0_constant"]["compute_us"] = cont["S"]["compute_us"]
    extra = {"S1_population_rhythm": {**pr, "test_log_loss": M.log_loss(yt, pS1)},
             "S2_c2_return_gap": {"p_active": p_c2, "test_log_loss": M.log_loss(yt, pS2)}, "S_is": S_best,
             "long_horizon_kind": "dynamic: forward simulation 2022-05-01..05-08 of the test users, mean daily "
                                  "|simulated - observed active rate|; also the TV distance of active-day counts",
             "observed_active_rate_by_day": obs.mean(0).tolist()}
    return assemble(name, f.id, "log_loss", cont, plan()["rule_mappings"][name], extra, t0,
                    f.card["selection"]["kept"])


def review_sessions(name: str = "session_rate") -> dict:
    from . import fit_e1 as E
    from . import kuairand as K
    t0 = time.time()
    f = load(name)
    raw = K.load_raw()
    g = E.session_days(raw)
    hc, he, tc, te_, users = E.session_test_arrays(g)
    shape, rate = f.predictor.shape, f.predictor.rate
    rK = f.predict(np.column_stack([hc, he]))
    D = lambda m, d: K.day_num(2022, m, d)       # noqa: E731
    hist = E.session_window(g, D(4, 22), D(5, 4))
    tr_users = hist.index[hist["split"] == "train"]
    glob = float(hist.loc[tr_users, "c"].sum() / hist.loc[tr_users, "e"].sum())
    pr = pop_rhythm()
    from ..social.consumption import BUILTIN as b
    c2 = 16 * 3600 / float(b["gap_mean_s"]) - 1
    dev = lambda r_: M.poisson_deviance(tc, np.maximum(r_ * te_, 1e-9))          # noqa: E731
    S_best = "S1" if dev(np.full(len(tc), pr["extra_per_active_day"])) <= dev(np.full(len(tc), c2)) else "S2"
    rS = np.full(len(tc), pr["extra_per_active_day"] if S_best == "S1" else c2)
    rates = {"K": rK, "B": np.full(len(tc), glob), "S": rS}

    def dev_rows(r_):
        mu = np.maximum(r_ * te_, 1e-9)
        return 2 * (np.where(tc > 0, tc * np.log(np.where(tc > 0, tc, 1) / mu), 0.0) - (tc - mu))
    # dynamic: per test user and active day in 2022-05-05..05-08, extra sessions drawn day by day
    gw = g[(g["day"] >= D(5, 5)) & (g["day"] <= D(5, 8))]
    uidx = {u: i for i, u in enumerate(users)}
    gw = gw[gw["user_id"].isin(uidx)]
    ui = gw["user_id"].map(uidx).to_numpy()
    dd = gw["day"].to_numpy() - D(5, 5)
    obs_extra = (gw["sessions"].to_numpy() - 1).astype(float)
    order = np.lexsort((dd, ui))
    ui, dd, obs_extra = ui[order], dd[order], obs_extra[order]

    def sim(kind, seed=0):
        rng = np.random.default_rng(seed)
        acc = np.zeros(ui.size)
        tot = np.zeros((DRAWS, len(users)))
        for r_ in range(DRAWS):
            c_add = np.zeros(len(users)); e_add = np.zeros(len(users))
            for d in range(4):
                m = dd == d
                u = ui[m]
                if kind == "K":
                    lam = rng.gamma(shape + hc[u] + c_add[u], 1.0 / (rate + he[u] + e_add[u]))
                else:
                    lam = rates[kind][u]
                x = rng.poisson(lam)
                acc[m] += x / DRAWS
                c_add[u] += x; e_add[u] += 1
                tot[r_, u] += x
        return acc, tot
    obs_tot = np.bincount(ui, weights=obs_extra, minlength=len(users))
    cont = {}
    gi = ui
    for k in ("K", "B", "S"):
        acc, tot = sim(k)
        pt = parts(acc, obs_extra, gi, len(users), dd, 4)
        cont[k] = {"primary": dev(rates[k]), "difference_vs_K_ci95": _row_ci(dev_rows(rates[k]), dev_rows(rK), users),
                   "long_horizon": measure(pt), "ks_user_totals": M.ks_two_sample(tot.ravel(), obs_tot),
                   "parts": pt, "C": len(users)}
    n = len(hc)
    cont["K"]["compute_us"] = timed(lambda: f.predict(np.column_stack([hc, he])), n)
    cont["B"]["compute_us"] = timed(lambda: np.full(n, glob), n)
    cont["S"]["compute_us"] = cont["B"]["compute_us"]
    extra = {"S1_population_rhythm": pr, "S2_c2_extra_per_active_day": c2, "S_is": S_best,
             "long_horizon_kind": "dynamic: extra sessions on each observed active day 2022-05-05..05-08 drawn day by "
                                  "day (K: gamma-Poisson posterior updated with the simulated counts), mean over days "
                                  "of |simulated - observed mean|; also KS of per-user totals",
             "window_note": "the plan text names 05-01..05-08; the card's test window (the rows the primary metric "
                            "uses) is 05-05..05-08, which is the window simulated"}
    return assemble(name, f.id, "poisson_deviance", cont, plan()["rule_mappings"][name], extra, t0,
                    f.card["selection"]["kept"])


def review_news(name: str = "news_click") -> dict:
    from . import mind
    t0 = time.time()
    f = load(name)
    data = mind.load()
    df, _ = mind.build(data)
    m = mind.news_masks(df, data["beh"])
    X = df[f.features].to_numpy(float)
    y = df["y"].to_numpy(float)
    te, tr = m["test"], m["train"]
    yt = y[te]
    pK = f.predict(X[te])
    pop_idx = [f.features.index(c) for c in mind.POP_FEATURES]
    bl = F.fit_logistic(X[tr][:, pop_idx], y[tr])
    pB = bl.predict(X[te][:, pop_idx])
    from ..social.consumption import BUILTIN as b
    tc = df["text_cos"].to_numpy()
    imp = df["imp"].to_numpy()
    util = b["open_bias"] + b["open_affinity"] * tc + b["open_len"] * math.log(120 / 60)
    eu = np.exp(np.clip(util, -40, 40))
    win = eu.copy()
    for k in range(1, int(b["window"]) + 1):
        ok = np.r_[imp[k:] == imp[:-k], np.zeros(k, bool)]
        win += np.where(ok, np.r_[eu[k:], np.zeros(k)], 0.0)
    pS = np.clip(eu / (math.exp(b["outside"]) + win), 1e-6, 1 - 1e-6)[te]
    sf = F.fit_logistic(tc[tr][:, None], y[tr])
    pSF = sf.predict(tc[te][:, None])
    groups = df.loc[te, "user_id"].to_numpy()
    gi, C = dense(groups)
    di, D = dense(df.loc[te, "hour"].to_numpy() // 4)
    rows_K = M.row_log_loss(yt, pK)
    cont = {}
    for k, p in (("K", pK), ("B", pB), ("S", pS), ("SF", pSF)):
        pt = parts(p, yt, gi, C, di, D)
        cont[k] = {"primary": M.log_loss(yt, p), "metrics": M.binary_report(yt, p),
                   "difference_vs_K_ci95": _row_ci(M.row_log_loss(yt, p), rows_K, groups),
                   "long_horizon": measure(pt), "parts": pt, "C": C, "eligible_role": k != "SF"}
    n = min(100000, int(te.sum()))
    from . import predictors as P
    portable = P.KINDS[f.predictor.kind].from_params(f.predictor.params())
    Xn = X[te][:n]
    cont["K"]["compute_us"] = timed(lambda: portable.predict(Xn), n)
    cont["B"]["compute_us"] = timed(lambda: bl.predict(Xn[:, pop_idx]), n)
    cont["S"]["compute_us"] = timed(lambda: np.exp(b["open_affinity"] * tc[:n]) / (1 + np.exp(tc[:n])), n)
    cont["SF"]["compute_us"] = cont["B"]["compute_us"]
    return assemble(name, f.id, "log_loss", cont, plan()["rule_mappings"][name],
                    {"long_horizon_kind": "static: mean over 4-hour blocks of the test day of |mean predicted - mean "
                                          "observed|"}, t0, f.card["selection"]["kept"])


def review_commercial(name: str = "commercial_response") -> dict:
    from . import criteo
    from . import scoring_e2 as S
    from .commercial import ARMS, OUTCOMES
    t0 = time.time()
    f = load("commercial_response@1")
    lab = criteo.splits()
    idx = criteo.audit_indices(lab)
    data = criteo.load()
    Xa = np.asarray(data["X"][idx], dtype=np.float64)
    t = np.asarray(data["treatment"][idx]).astype(bool)
    yv = np.asarray(data["visit"][idx]).astype(float)
    yc = np.asarray(data["conversion"][idx]).astype(float)
    tr = np.flatnonzero(lab == 0)
    pv0 = float(np.asarray(data["visit"])[tr].mean())
    pc0 = float(np.asarray(data["conversion"])[tr].mean())

    def pooled(P_):
        cols = {f"{o}_{a}": P_[:, i] for i, (o, a) in enumerate([(o, a) for o in OUTCOMES for a in ARMS])}
        pv = np.where(t, cols["visit_treated"], cols["visit_control"])
        pc = np.where(t, cols["conversion_treated"], cols["conversion_control"])
        return pv, pc
    PK = f.predict(Xa)
    PB = f.candidate_predict(f.card["selection"]["baseline"], Xa)
    PS = np.column_stack([np.full(len(idx), pv0), np.full(len(idx), pv0), np.full(len(idx), pc0),
                          np.full(len(idx), pc0)])
    cont = {}
    vK, cK = pooled(PK)
    rowsK = M.row_log_loss(yv, vK) + M.row_log_loss(yc, cK)
    dec = np.clip((np.argsort(np.argsort(vK)) * 10) // len(vK), 0, 9)
    cell = dec * 2 + t.astype(int)
    for k, P_ in (("K", PK), ("B", PB), ("S", PS)):
        pv, pc = pooled(P_)
        rows_ = M.row_log_loss(yv, pv) + M.row_log_loss(yc, pc)
        prim = S.log_loss(yv, pv) + S.log_loss(yc, pc)
        pt = parts(pv, yv, np.arange(len(idx)), len(idx), cell, 20)
        cont[k] = {"primary": float(prim), "difference_vs_K_ci95": _row_ci(rows_, rowsK, np.arange(len(idx))),
                   "long_horizon": measure(pt), "parts": pt, "C": len(idx)}
    n = len(idx)
    cont["K"]["compute_us"] = timed(lambda: f.predict(Xa), n, reps=3)
    cont["B"]["compute_us"] = timed(lambda: f.candidate_predict(f.card["selection"]["baseline"], Xa), n, reps=3)
    cont["S"]["compute_us"] = timed(lambda: np.tile([pv0, pv0, pc0, pc0], (n, 1)), n)
    rep_primary = load("commercial_response@1").card
    return assemble(name, f.id, "primary_pooled_logloss_visit_plus_conversion", cont,
                    plan()["rule_mappings"][name],
                    {"rows": "the report's seeded 100,000-user test audit subsample (Criteo rows have no time stamp; "
                             "the full-test number is the fit's)",
                     "pooled_train_rates": {"visit": pv0, "conversion": pc0},
                     "long_horizon_kind": "static: no time stamps; mean over (arm x decile of K's visit prediction) "
                                          "cells of |mean predicted - observed| visit rate",
                     "card_kept": rep_primary["selection"]["kept"]}, t0, f.card["selection"]["kept"])


def review_software(name: str) -> dict:
    from . import fit_software as FS
    from . import scoring_e2 as S
    t0 = time.time()
    f = load(f"{name}@1")
    with np.load(FS.tables_path()) as z:
        tab = {k: z[k] for k in z.files}
    base = f.card["selection"]["baseline"]
    cont, extra = {}, {}
    if name == "software_deep_engagement":
        a, b_ = FS.SPLIT_HOURS["test"]
        m = (tab["deep/t"] >= a * 3600) & (tab["deep/t"] < b_ * 3600)
        X, y = tab["deep/X"][m], tab["deep/y"][m].astype(float)
        pK, pB = f.predict(X), f.candidate_predict(base, X)
        groups = tab["deep/actor"][m]
        gi, C = dense(groups)
        di, D = dense(tab["deep/t"][m] // (4 * 3600))
        rows_K = M.row_log_loss(y, pK)
        for k, p in (("K", pK), ("B", pB)):
            pt = parts(p, y, gi, C, di, D)
            cont[k] = {"primary": float(S.log_loss(y, p)), "difference_vs_K_ci95": _row_ci(M.row_log_loss(y, p), rows_K,
                                                                                            groups),
                       "long_horizon": measure(pt), "parts": pt, "C": C}
        primary = "log_loss"
        extra["long_horizon_kind"] = "static: mean over 4-hour blocks of the test half-day of |mean predicted - observed|"
    else:
        X2, y2, r2 = tab["growth/d2/X"], tab["growth/d2/y"].astype(float), tab["growth/d2/repo"]
        X1, r1 = tab["growth/d1/X"], tab["growth/d1/repo"]
        y1 = tab["growth/d1/y"].astype(float)              # observed D2 stars of the D1 repositories
        pK, pB = f.predict(X2), f.candidate_predict(base, X2)

        def dev_rows(yy, mu):
            mu = np.maximum(mu, 1e-9)
            return 2 * (np.where(yy > 0, yy * np.log(np.where(yy > 0, yy, 1) / mu), 0.0) - (yy - mu))
        both = np.intersect1d(r1, r2)
        i1 = np.searchsorted(r1, both) if np.all(np.diff(r1) > 0) else np.array([np.flatnonzero(r1 == x)[0] for x in both])
        o2 = np.argsort(r2)
        i2 = o2[np.searchsorted(r2[o2], both)]
        o1 = np.argsort(r1)
        i1 = o1[np.searchsorted(r1[o1], both)]
        names = [n_ for n_, _, _ in FS.GROWTH_FEATURES]
        j_today = names.index("stars_today")
        alpha = f.dispersion

        def sim(predf, nb: bool, seed=0):
            rng = np.random.default_rng(seed)
            tot = np.zeros((DRAWS, both.size))
            for d_ in range(DRAWS):
                mu1 = np.maximum(predf(X1[i1]), 1e-12)
                s2 = rng.poisson(rng.gamma(1 / alpha, mu1 * alpha)) if nb else rng.poisson(mu1)
                Xd = X2[i2].copy()
                Xd[:, j_today] = np.log1p(s2)
                mu2 = np.maximum(predf(Xd), 1e-12)
                tot[d_] = rng.poisson(rng.gamma(1 / alpha, mu2 * alpha)) if nb else rng.poisson(mu2)
            return tot
        obs3 = y2[i2]
        for k, predf, nb, p in (("K", f.predict, True, pK), ("B", lambda Z: f.candidate_predict(base, Z), False, pB)):
            tot = sim(predf, nb)
            pt = parts(tot.mean(0), obs3, np.arange(both.size), both.size, np.zeros(both.size, int), 1)
            cont[k] = {"primary": float(S.poisson_deviance(y2, p)),
                       "difference_vs_K_ci95": _row_ci(dev_rows(y2, p), dev_rows(y2, pK), r2),
                       "long_horizon": measure(pt, relative=True), "relative": True,
                       "ks_log1p_repo_d3": M.ks_two_sample(np.log1p(tot.ravel()), np.log1p(obs3)),
                       "parts": pt, "C": both.size}
        primary = "mean_poisson_deviance"
        extra.update({"long_horizon_kind": "dynamic: D1 -> D2 -> D3 two-step simulation for repositories in both the "
                                           "D1 and D2 tables (D2 stars drawn, fed into stars_today; other D2 activity "
                                           "as observed); |simulated - observed total D3 stars| / observed",
                      "repositories": int(both.size), "observed_d2_stars_of_d1_repos": float(y1.sum())})
        X = X2
    n = min(100000, len(X))
    cont["K"]["compute_us"] = timed(lambda: f.predict(X[:n]), n)
    cont["B"]["compute_us"] = timed(lambda: f.candidate_predict(base, X[:n]), n)
    extra["rule"] = "no simulator rule exists for this behaviour (no simulated developer population); B and K only"
    return assemble(name, f.id, primary, cont, plan()["rule_mappings"][name], extra, t0, f.card["selection"]["kept"])


REVIEWS = {"content_consumption": lambda: review_content("content_consumption"),
           "content_consumption_feed": lambda: review_content("content_consumption_feed"),
           "watch_duration": review_watch, "return_daily": review_return, "session_rate": review_sessions,
           "news_click": review_news, "commercial_response": review_commercial,
           "software_deep_engagement": lambda: review_software("software_deep_engagement"),
           "software_repo_growth": lambda: review_software("software_repo_growth")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, choices=sorted(REVIEWS))
    a = ap.parse_args(argv)
    for name in a.model:
        t0 = time.time()
        r = REVIEWS[name]()
        p = write(name, r)
        print(name, "->", os.path.relpath(p, CRUCIBLE_ROOT), "kept_after_review", r["kept_after_review"],
              f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
