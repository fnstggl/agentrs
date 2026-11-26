"""E4 fits: six new behavioural domains and the shared-space news model, under the committed plan
`docs/v5/calibration/plans/E4_new_domains_plan.json` (workstream E). Offline fitting code only.

    PYTHONPATH=src python -m crucible.calibration.fit_e4 --model star_cascade [--model ...]

Every model follows the plan's global rules: candidates fitted on the train split; selection among candidates within
the compute budget by the validation primary metric, overruled only when the selected candidate's validation
long-horizon measure is significantly worse than the best one's; test once; confirmation against the simple
baselines; cluster-bootstrap intervals; comparison with the simulator's rule-based behaviour (`rule_mapping`).
Cards are written through `fitlib.finish` (registry SCHEMA), so the gates measure them like every other model.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

from . import fitlib as F
from . import metrics as M
from . import predictors as P
from . import predictors_e4 as P4
from .registry import CRUCIBLE_ROOT

PLAN_PATH = "docs/v5/calibration/plans/E4_new_domains_plan.json"
VERSION = 1
BUDGET_US = 100.0
N_BOOT = 200
DRAWS = 20
RULE_TEXT = ("E4: among candidates with compute <= 100 us/row, order by the validation primary metric (ties within "
             "1e-4 to the simpler); keep the first unless its validation long-horizon measure is worse than the "
             "smallest one with a paired cluster-bootstrap 95% interval of the difference entirely above 0; keep a "
             "simple baseline instead if it is strictly better on test")


def spec(name: str) -> dict:
    with open(os.path.join(CRUCIBLE_ROOT, PLAN_PATH), encoding="utf-8") as fh:
        return json.load(fh)["models"][name]


# ====================================================================== shared machinery
def portable_copy(pred):
    """The predictor as the registry loads it (NumPy path, no in-process scikit-learn shortcut)."""
    return P.KINDS[pred.kind].from_params(pred.params())


def time_us(pred, X, reps: int = 5) -> float:
    q = portable_copy(pred)
    X = np.ascontiguousarray(np.asarray(X, float)[:100000])
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        q.predict(X)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) / max(len(X), 1) * 1e6)


def dense(groups) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    inv = inv.reshape(-1)
    return inv, int(inv.max() + 1) if inv.size else 0


def parts(values, target, cl, C, per, D) -> tuple:
    """Per-cluster per-period sums (prediction or simulation, observation, count) as dense [C, D] arrays."""
    idx = np.asarray(cl, np.int64) * D + np.asarray(per, np.int64)
    sp = np.bincount(idx, weights=np.asarray(values, float), minlength=C * D).reshape(C, D)
    so = np.bincount(idx, weights=np.asarray(target, float), minlength=C * D).reshape(C, D)
    n = np.bincount(idx, minlength=C * D).reshape(C, D).astype(float)
    return sp, so, n


def measure(pt, w=None, relative: bool = False) -> float:
    """Mean over periods of |mean predicted - mean observed| (relative: divided by the mean observed)."""
    sp, so, n = pt
    if w is None:
        a, b, c = sp.sum(0), so.sum(0), n.sum(0)
    else:
        a, b, c = w @ sp, w @ so, w @ n
    ok = c > 0
    if not ok.any():
        return float("nan")
    d = np.abs(a[ok] - b[ok]) / c[ok]
    if relative:
        d = d / np.maximum(b[ok] / c[ok], 1e-12)
    return float(d.mean())


def paired_boot(pa, pb, C: int, relative: bool = False, n: int = N_BOOT, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
        diffs[i] = measure(pa, w, relative) - measure(pb, w, relative)
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def rank(val: dict, order: list, lower: bool) -> list:
    sign = 1 if lower else -1
    remaining, out = list(order), []
    while remaining:
        best = min(sign * val[k] for k in remaining)
        pick = next(k for k in remaining if sign * val[k] <= best + 1e-4)
        out.append(pick)
        remaining.remove(pick)
    return out


def select_e4(cands: dict, primary: str, baselines: list, order: list, lower: bool, compute: dict, lh: dict,
              lh_parts: dict, C: int, relative: bool = False) -> dict:
    sign = 1 if lower else -1
    val = {k: float(cands[k]["val"][primary]) for k in cands}
    test = {k: float(cands[k]["test"][primary]) for k in cands}
    ranked = rank(val, order, lower)
    eligible = [k for k in ranked if compute[k] <= BUDGET_US]
    with_lh = [k for k in eligible if lh.get(k) is not None and np.isfinite(lh[k])]
    ref = min(with_lh, key=lambda k: lh[k]) if with_lh else None
    rejected, selected = {}, None
    for k in eligible:
        if ref is not None and k != ref and k in with_lh:
            ci = paired_boot(lh_parts[k], lh_parts[ref], C, relative)
            if ci[0] > 0:
                rejected[k] = {"versus": ref, "difference_ci95": ci, "long_horizon": lh[k], "reference": lh[ref]}
                continue
        selected = k
        break
    if selected is None:
        raise RuntimeError("no eligible candidate")
    best_base = min(baselines, key=lambda b: sign * test[b])
    kept, reason = selected, "selected on validation and confirmed on test: no simple baseline is better"
    if sign * test[best_base] < sign * test[selected]:
        kept = best_base
        reason = f"{selected} was selected on validation but the baseline {best_base} is better on test; kept"
    elif selected in baselines:
        reason = "a simple baseline was selected on validation and is kept"
    if rejected:
        reason += f"; rejected for long-horizon drift: {sorted(rejected)}"
    return {"rule": RULE_TEXT, "rule_version": "E4", "primary_metric": primary, "lower_is_better": lower,
            "validation": val, "test": test, "baselines": list(baselines), "candidates": list(order),
            "ranked_on_validation": ranked, "compute_us_per_row": {k: float(compute[k]) for k in order},
            "compute_budget_us": BUDGET_US, "eligible": eligible,
            "long_horizon_validation": {k: (None if lh.get(k) is None else float(lh[k])) for k in order},
            "long_horizon_reference": ref, "drift_rejected": rejected, "selected_on_validation": selected,
            "best_baseline_on_test": best_base, "kept": kept, "kept_is_baseline": kept in baselines,
            "reason": reason}


def finish(name: str, domain: str, s: dict, cands: dict, sel: dict, features: list, target: str, splits: dict,
           heldout: list, bc: dict, calibration: dict, extra: dict, ident_note: str, limits: str,
           fmap: dict, t0: float, cmd: str, card_extra: dict, rule: dict, long_horizon_test: dict,
           population: str, platform: str, period: str) -> str:
    rep = {"schema": "crucible.v5.calibration_report/1", "model": name, "version": VERSION, "domain": domain,
           "datasets": [s["dataset"]], "identification": s["identification"], "estimand": s["estimand"],
           "population": population, "platform": platform, "period": period, "plan": F.plan_record(PLAN_PATH),
           "splits": splits, "heldout_splits": heldout, "candidates": F.strip(cands), "selection": sel,
           "baseline_comparison": bc, "calibration": calibration, "heldout_extra": extra,
           "rule_comparison": rule, "long_horizon_test": long_horizon_test,
           "identification_note": ident_note, "extrapolation_limits": limits, "feature_map": fmap,
           "fit": {"commit": F.head_commit(), "seconds": time.time() - t0, "command": cmd}}
    kept = cands[sel["kept"]]["pred"]
    return F.finish(name, VERSION, domain, kept, features, target, rep,
                    dict(card_extra, module="crucible.calibration.predictors_e4"))


def bc_binary(sel, cands, y, groups, preds) -> dict:
    k, b = sel["kept"], sel["best_baseline_on_test"]
    lk, lb = M.row_log_loss(y, preds[k]), M.row_log_loss(y, preds[b])
    ci_k, ci_d = M.cluster_bootstrap(lk, groups), M.cluster_bootstrap(lk - lb, groups)
    return {"metric": sel["primary_metric"], "kept": k, "baseline": b, "kept_test": sel["test"][k],
            "baseline_test": sel["test"][b], "difference": sel["test"][k] - sel["test"][b],
            "ci95": [ci_d["lo"], ci_d["hi"]], "kept_ci95": [ci_k["lo"], ci_k["hi"]],
            "bootstrap": {"resamples": ci_d["n_boot"], "clusters": ci_d["clusters"]}}


def bc_rows(sel, rows_k, rows_b, groups) -> dict:
    k, b = sel["kept"], sel["best_baseline_on_test"]
    ci_k, ci_d = M.cluster_bootstrap(rows_k, groups), M.cluster_bootstrap(rows_k - rows_b, groups)
    return {"metric": sel["primary_metric"], "kept": k, "baseline": b, "kept_test": sel["test"][k],
            "baseline_test": sel["test"][b], "difference": sel["test"][k] - sel["test"][b],
            "ci95": [ci_d["lo"], ci_d["hi"]], "kept_ci95": [ci_k["lo"], ci_k["hi"]],
            "bootstrap": {"resamples": ci_d["n_boot"], "clusters": ci_d["clusters"]}}


def binary_metrics(y, p) -> dict:
    return M.binary_report(y, p)


class Sub:
    """A fitted predictor on a subset of columns (exported as predictors_e4.Subset)."""

    def __init__(self, inner, cols):
        self.inner, self.cols = inner, list(cols)

    def predict(self, X):
        return self.inner.predict(np.asarray(X, float)[:, self.cols])

    def export(self):
        inner = self.inner
        if hasattr(inner, "_sk"):
            inner = portable_copy(inner)
        return P4.Subset(inner, self.cols)


def _export(pred):
    return pred.export() if isinstance(pred, Sub) else pred


def binary_fit(name: str, X: dict, y: dict, groups: dict, periods: dict, features: list, baseline_cols: list,
               cmd: str, t0: float) -> tuple:
    """Fit B0 constant, B1 logistic on `baseline_cols`, M1 logistic, M2 GBDT; select by the E4 rule."""
    b0 = F.fit_constant(y["train"])
    b1 = Sub(F.fit_logistic(X["train"][:, baseline_cols], y["train"]), baseline_cols)
    m1 = F.fit_logistic(X["train"], y["train"])
    m2 = F.fit_gbdt(X["train"], y["train"])
    s = spec(name)
    names = list(s["candidates"])
    preds = dict(zip(names, (b0, b1, m1, m2)))
    cx = dict(zip(names, ("baseline: constant", f"baseline: logistic on {len(baseline_cols)} features",
                          f"L2 logistic, {len(features)} features", f"GBDT, {getattr(m2, 'n_iter', 0)} trees")))
    npar = {names[0]: 1, names[1]: len(baseline_cols) + 1, names[2]: len(features) + 1, names[3]: F.n_params(m2)}
    pv = {k: np.asarray(p.predict(X["val"]), float) for k, p in preds.items()}
    pt = {k: np.asarray(p.predict(X["test"]), float) for k, p in preds.items()}
    cands = {k: {"complexity": cx[k], "n_params": npar[k], "pred": _export(preds[k]),
                 "val": binary_metrics(y["val"], pv[k]), "test": binary_metrics(y["test"], pt[k])} for k in names}
    compute = {k: time_us(_export(preds[k]), X["val"]) for k in names}
    gv, Cv = dense(groups["val"])
    perv, Dv = dense(periods["val"])
    lhp = {k: parts(pv[k], y["val"], gv, Cv, perv, Dv) for k in names}
    lh = {k: measure(lhp[k]) for k in names}
    sel = select_e4(cands, "log_loss", names[:2], names, True, compute, lh, lhp, Cv)
    gt, Ct = dense(groups["test"])
    pert, Dt = dense(periods["test"])
    lh_test = {k: measure(parts(pt[k], y["test"], gt, Ct, pert, Dt)) for k in names}
    return cands, sel, pv, pt, lh_test


# ====================================================================== GH Archive: participation and cascades
PART_TYPES = ("IssuesEvent", "IssueCommentEvent", "PullRequestEvent", "PullRequestReviewEvent",
              "PullRequestReviewCommentEvent")


def _windowed(ent, mask, hour, eq, hq, width=24):
    keys = np.sort(ent[mask].astype(np.int64) * 128 + hour[mask])
    lo = eq.astype(np.int64) * 128 + np.maximum(hq - width, 0)
    hi = eq.astype(np.int64) * 128 + hq
    return (np.searchsorted(keys, hi, "left") - np.searchsorted(keys, lo, "left")).astype(np.float64)


def _distinct_in_window(owner, other, hour, mask, q_owner, q_hour, width=24, H=96):
    """For each query (owner o, hour h): number of distinct `other` values with an event of `owner` o in hours
    [h - width, h). Exact: union of (event hour, event hour + width] intervals per (owner, other) pair, counted with
    a difference array per owner."""
    o = owner[mask].astype(np.int64); x = other[mask].astype(np.int64); h = hour[mask].astype(np.int64)
    uo, oi = np.unique(o, return_inverse=True)
    oi = oi.reshape(-1)
    order = np.lexsort((h, x, oi))
    oi, x, h = oi[order], x[order], h[order]
    keep = np.ones(oi.size, bool)
    keep[1:] = (oi[1:] != oi[:-1]) | (x[1:] != x[:-1]) | (h[1:] != h[:-1])
    oi, x, h = oi[keep], x[keep], h[keep]
    same_prev = np.zeros(oi.size, bool)
    same_prev[1:] = (oi[1:] == oi[:-1]) & (x[1:] == x[:-1])
    same_next = np.zeros(oi.size, bool)
    same_next[:-1] = same_prev[1:]
    start = ~same_prev | (h - np.r_[h[:1], h[:-1]] > width)
    end = ~same_next | (np.r_[h[1:], h[-1:]] - h > width)
    diff = np.zeros((uo.size, H + width + 2), np.int32)
    np.add.at(diff, (oi[start], h[start] + 1), 1)
    np.add.at(diff, (oi[end], h[end] + width + 1), -1)
    cover = np.cumsum(diff, axis=1)
    qi = np.searchsorted(uo, q_owner)
    ok = (qi < uo.size) & (uo[np.minimum(qi, uo.size - 1)] == q_owner)
    out = np.zeros(q_owner.size)
    out[ok] = cover[qi[ok], np.asarray(q_hour)[ok]]
    return out


def github_participation_table():
    from . import gharchive as G
    ev = G.load_events()
    C = G.TYPE_CODE
    typ, act, flags = ev["type"], ev["action"], ev["flags"]
    t = ev["t"].astype(np.int64)
    pos = ev["pos"]
    bot = (flags & G.FLAG_BOT) != 0
    owner = (flags & G.FLAG_OWNER) != 0
    org = (flags & G.FLAG_ORG) != 0
    hour = np.clip(t // 3600, 0, G.HOURS - 1)
    _, repo = np.unique(ev["repo"], return_inverse=True)
    _, actor = np.unique(ev["actor"], return_inverse=True)
    repo = repo.astype(np.int64); actor = actor.astype(np.int64)
    R = int(repo.max()) + 1
    _, pair = np.unique(actor * R + repo, return_inverse=True)
    pair = pair.reshape(-1).astype(np.int64)
    del ev
    isp = np.isin(typ, [C[n] for n in PART_TYPES]) & ~bot
    cand = np.flatnonzero(isp & ~owner)
    order = np.lexsort((pos[cand], t[cand]))
    cand = cand[order]
    _, first = np.unique(pair[cand], return_index=True)
    u = cand[np.sort(first)]
    u = u[(t[u] >= 24 * 3600) & (t[u] < 72 * 3600)]
    # label: another participation event of the same pair within 24 h after the unit (archive order breaks ties)
    pe = np.flatnonzero(isp)
    rank_ = np.empty(t.size, dtype=np.int64)
    rank_[np.lexsort((pos, t))] = np.arange(t.size)
    comp = np.sort(pair[pe] * (1 << 26) + rank_[pe])
    q = pair[u] * (1 << 26) + rank_[u]
    j = np.searchsorted(comp, q, "right")
    jj = np.minimum(j, comp.size - 1)
    nxt_rank = comp[jj] & ((1 << 26) - 1)
    same = (j < comp.size) & ((comp[jj] >> 26) == pair[u])
    t_of_rank = np.empty(t.size, np.int64)
    t_of_rank[rank_] = t
    y = (same & (t_of_rank[np.where(same, nxt_rank, 0)] <= t[u] + 24 * 3600)).astype(np.float64)
    H = hour[u]
    is_issue = np.isin(typ, [C["IssuesEvent"], C["IssueCommentEvent"]])
    is_pr = np.isin(typ, [C["PullRequestEvent"], C["PullRequestReviewEvent"], C["PullRequestReviewCommentEvent"]])
    star = (typ == C["WatchEvent"]) & ~bot
    allev = np.ones(t.size, bool)
    feats = {
        "first_is_issue_open": ((typ[u] == C["IssuesEvent"]) & (act[u] == G.ACTION_CODE["opened"])).astype(float),
        "first_is_issue_comment": (typ[u] == C["IssueCommentEvent"]).astype(float),
        "first_is_pr_open": ((typ[u] == C["PullRequestEvent"]) & (act[u] == G.ACTION_CODE["opened"])).astype(float),
        "first_is_pr_review": np.isin(typ[u], [C["PullRequestReviewEvent"],
                                               C["PullRequestReviewCommentEvent"]]).astype(float),
        "repo_issue_events_24h": np.log1p(_windowed(repo, is_issue, hour, repo[u], H)),
        "repo_pr_events_24h": np.log1p(_windowed(repo, is_pr, hour, repo[u], H)),
        "repo_participants_24h": np.log1p(_distinct_in_window(repo, actor, hour, isp, repo[u], H)),
        "repo_owner_events_24h": np.log1p(_windowed(repo, owner & ~bot, hour, repo[u], H)),
        "repo_stars_24h": np.log1p(_windowed(repo, star, hour, repo[u], H)),
        "repo_pushes_24h": np.log1p(_windowed(repo, typ == C["PushEvent"], hour, repo[u], H)),
        "repo_org": org[u].astype(float),
        "actor_events_24h": np.log1p(_windowed(actor, allev, hour, actor[u], H)),
        "actor_participation_repos_24h": np.log1p(_distinct_in_window(actor, repo, hour, isp, actor[u], H)),
        "actor_repo_events_24h": np.log1p(_windowed(pair, allev, hour, pair[u], H)),
        "hour_sin": np.sin(2 * np.pi * (H % 24) / 24), "hour_cos": np.cos(2 * np.pi * (H % 24) / 24)}
    # developers active in hours 0-48 (for the new-developer report)
    seen48 = np.zeros(int(actor.max()) + 1, bool)
    seen48[actor[t < 48 * 3600]] = True
    return {"X": feats, "y": y, "hour": H, "actor": actor[u], "repo": repo[u], "new_dev": ~seen48[actor[u]]}


def fit_participation(cmd: str, t0: float) -> str:
    name = "repo_participation_return"
    s = spec(name)
    tab = github_participation_table()
    feats = s["features"]
    X = np.column_stack([tab["X"][f] for f in feats])
    y, H = tab["y"], tab["hour"]
    sp = {"train": (H >= 24) & (H < 48), "val": (H >= 48) & (H < 60), "test": (H >= 60) & (H < 72)}
    sp["test_new_developers"] = sp["test"] & tab["new_dev"]
    Xs = {k: X[m] for k, m in sp.items()}
    ys = {k: y[m] for k, m in sp.items()}
    gs = {k: tab["actor"][m] for k, m in sp.items()}
    per = {k: H[m] // 4 for k, m in sp.items()}
    bcols = [feats.index(f) for f in ("first_is_issue_open", "first_is_issue_comment", "first_is_pr_open",
                                      "first_is_pr_review")]
    cands, sel, pv, pt, lh_test = binary_fit(name, Xs, ys, gs, per, feats, bcols, cmd, t0)
    pk = pt[sel["kept"]]
    # rule: the population's participate decision at mean preferences, one constant probability per read
    from ..population import models as PM
    d = PM._decisions()["participate"]
    z = d.bias + d.coef["sociability"] * 0.5 + d.coef["writing_skill"] * 0.5 + d.coef["page_utility"] * 0.0 + \
        d.coef["goal_social"] * 0.0
    p_rule = 1.0 / (1.0 + math.exp(-z))
    rule = {"mapping": s["rule_mapping"], "rule_probability": p_rule,
            "test": binary_metrics(ys["test"], np.full(ys["test"].size, p_rule)),
            "kept_minus_rule_log_loss": float(sel["test"][sel["kept"]] -
                                              M.log_loss(ys["test"], np.full(ys["test"].size, p_rule)))}
    newd = sp["test_new_developers"]
    extra = {"test_new_developers": {k: binary_metrics(ys["test_new_developers"],
                                                       cands[k]["pred"].predict(Xs["test_new_developers"]))
                                     for k in cands}}
    rows = {k: int(m.sum()) for k, m in sp.items()}
    splits = {"train": {"definition": s["splits"]["train"], "rows": rows["train"]},
              "val": {"definition": s["splits"]["validation"], "rows": rows["val"]},
              "test": {"definition": s["splits"]["test"], "rows": rows["test"]},
              "test_new_developers": {"definition": s["splits"]["test_new_developers"], "rows": int(newd.sum())}}
    limits = ("GH Archive public GitHub events, 2024-10-07..10 UTC (Monday to Thursday): participation by non-bot, "
              "non-owner accounts in public issue and pull-request discussions. The simulator may use this model for "
              "the probability that an individual who has just taken part in a community or project discussion takes "
              "part again in the same community within a day, as a function of the community's recent activity and "
              "maintainer activity and of the individual's own activity. It may NOT infer: (1) participation in "
              "non-software communities, private spaces or other platforms; (2) horizons beyond 24 h or retention "
              "over weeks (the window has four days); (3) causal effects of maintainer responses or community size "
              "(observational: active repositories attract active developers); (4) the content or quality of "
              "contributions (only event types are seen); 'first participation' is first within the window, so "
              "earlier history before 2024-10-07 is unknown (left-censored).")
    ident = ("Observational: who participates where is self-selected; associations of activity with return are not "
             "effects of making a community more active.")
    return finish(name, s["domain"], s, cands, sel, feats, "returned: another participation event on the same "
                  "repository within 24 h", splits, ["test", "test_new_developers"],
                  bc_binary(sel, cands, ys["test"], gs["test"], pt),
                  {"ece": M.ece(ys["test"], pk), "brier": M.brier(ys["test"], pk), "mean_pred": float(pk.mean()),
                   "mean_obs": float(ys["test"].mean()), "reliability": M.calibration_table(ys["test"], pk)},
                  extra, ident, limits, {f: f"{f}: see the E4 plan, models.{name}.feature_definitions" for f in feats}, t0, cmd,
                  {"mechanisms": ["community.return_participation"], "prior_rate": float(ys["train"].mean()),
                   "feature_quantiles": F.feature_quantiles(Xs["train"], feats)},
                  rule, {"stability_4h_blocks": lh_test},
                  "public GitHub accounts (non-bot, non-owner) participating in public repositories",
                  "GitHub (github.com) issues and pull requests", "2024-10-07..10 UTC")


def github_cascade_table():
    from . import gharchive as G
    ev = G.load_events()
    C = G.TYPE_CODE
    typ, flags = ev["type"], ev["flags"]
    t = ev["t"].astype(np.int64)
    bot = (flags & G.FLAG_BOT) != 0
    org = (flags & G.FLAG_ORG) != 0
    hour = np.clip(t // 3600, 0, G.HOURS - 1)
    _, repo = np.unique(ev["repo"], return_inverse=True)
    repo = repo.astype(np.int64)
    del ev
    star = (typ == C["WatchEvent"]) & ~bot
    srepo = np.unique(repo[star])
    idx = -np.ones(int(repo.max()) + 1, np.int64)
    idx[srepo] = np.arange(srepo.size)
    Rs = srepo.size
    S = np.zeros((Rs, G.HOURS), np.float32)
    np.add.at(S, (idx[repo[star]], hour[star]), 1)
    in_s = idx[repo] >= 0
    Fk = np.zeros((Rs, G.HOURS), np.float32)
    m = in_s & (typ == C["ForkEvent"]) & ~bot
    np.add.at(Fk, (idx[repo[m]], hour[m]), 1)
    Ev = np.zeros((Rs, G.HOURS), np.float32)
    np.add.at(Ev, (idx[repo[in_s]], hour[in_s]), 1)
    orgr = np.zeros(Rs, bool)
    orgr[idx[repo[in_s & org]]] = True
    create = in_s & (typ == C["CreateEvent"]) & ((flags & G.FLAG_REPO_CREATE) != 0)
    created_h = np.full(Rs, np.inf)
    np.minimum.at(created_h, idx[repo[create]], hour[create].astype(float))
    glob = S.sum(0)
    return {"S": S, "F": Fk, "E": Ev, "org": orgr, "created_h": created_h, "global": glob}


CASCADE_FEATURES = ["stars_prev_1h", "stars_prev_3h", "stars_prev_24h", "star_decay_2h", "star_decay_12h",
                    "forks_prev_24h", "events_prev_24h", "repo_org", "created_in_window", "age_hours_log",
                    "global_stars_prev_1h", "hour_sin", "hour_cos"]


def cascade_features(S: np.ndarray, h: int, tab: dict, rows: np.ndarray) -> np.ndarray:
    """Features of repositories `rows` for hour h from star history S[:, :h] (observed or simulated)."""
    lo = max(h - 24, 0)
    s1 = S[rows, h - 1]
    s3 = S[rows, max(h - 3, 0):h].sum(1)
    s24 = S[rows, lo:h].sum(1)
    ages = h - np.arange(lo, h)
    d2 = (S[rows, lo:h] * np.exp(-ages / 2.0)).sum(1)
    d12 = (S[rows, lo:h] * np.exp(-ages / 12.0)).sum(1)
    f24 = tab["F"][rows, lo:h].sum(1)
    e24 = tab["E"][rows, lo:h].sum(1)
    ch = tab["created_h"][rows]
    made = ch < h
    age = np.where(made, h - np.where(made, ch, 0), 0.0)
    g1 = tab["global"][h - 1] / 1000.0
    hh = h % 24
    n = rows.size
    return np.column_stack([np.log1p(s1), np.log1p(s3), np.log1p(s24), np.log1p(d2), np.log1p(d12), np.log1p(f24),
                            np.log1p(e24), tab["org"][rows].astype(float), made.astype(float), np.log1p(age),
                            np.full(n, np.log1p(g1)), np.full(n, math.sin(2 * math.pi * hh / 24)),
                            np.full(n, math.cos(2 * math.pi * hh / 24))])


def fit_cascade(cmd: str, t0: float) -> str:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import PoissonRegressor
    name = "star_cascade"
    s = spec(name)
    tab = github_cascade_table()
    S = tab["S"]
    feats = s["features"]
    assert feats == CASCADE_FEATURES
    Xl, yl, hl, rl = [], [], [], []
    for h in range(24, 96):
        rows = np.flatnonzero(S[:, h - 24:h].sum(1) > 0)
        Xl.append(cascade_features(S, h, tab, rows).astype(np.float32))
        yl.append(S[rows, h])
        hl.append(np.full(rows.size, h, np.int16))
        rl.append(rows)
    X = np.concatenate(Xl).astype(np.float64); y = np.concatenate(yl).astype(np.float64)
    H = np.concatenate(hl); Rr = np.concatenate(rl)
    del Xl, yl
    sp = {"train": (H >= 24) & (H < 60), "val": (H >= 60) & (H < 72), "test": (H >= 72)}
    Xs = {k: X[m] for k, m in sp.items()}
    ys = {k: y[m] for k, m in sp.items()}
    i24 = feats.index("stars_prev_24h")
    b0 = P4.Persistence(i24, 1.0 / 24.0, "expm1")
    sc1 = Xs["train"][:, [i24]]
    mu1, sd1 = sc1.mean(0), sc1.std(0)
    g1 = PoissonRegressor(alpha=1e-4, max_iter=1000).fit((sc1 - mu1) / sd1, ys["train"])
    b1 = P4.Subset(P4.Linear(g1.coef_, g1.intercept_, mu1, sd1, "log"), [i24])
    mu, sd = Xs["train"].mean(0), Xs["train"].std(0)
    sd[sd < 1e-12] = 1.0
    g = PoissonRegressor(alpha=1e-4, max_iter=1000).fit((Xs["train"] - mu) / sd, ys["train"])
    m1 = P4.Linear(g.coef_, g.intercept_, mu, sd, "log")
    hgb = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                        min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=20, random_state=0)
    hgb.fit(Xs["train"], ys["train"])
    m2 = P4.TreeRegressor.from_sklearn(hgb, link="log")
    k = min(20000, len(Xs["val"]))
    diff = float(np.max(np.abs(portable_copy(m2).predict(Xs["val"][:k]) - hgb.predict(Xs["val"][:k]))))
    if diff > 1e-6 * max(1.0, float(np.max(hgb.predict(Xs["val"][:k])))):
        raise AssertionError(f"exported regression trees differ from scikit-learn by {diff}")
    m2._sk = hgb
    names = list(s["candidates"])
    preds = dict(zip(names, (b0, b1, m1, m2)))

    def cm(yy, mu_):
        mu_ = np.maximum(mu_, 1e-9)
        q = np.quantile(mu_, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q[1:-1], mu_, "right"), 0, 9)
        err = [abs(mu_[b == i].mean() - yy[b == i].mean()) / max(yy.mean(), 1e-12) for i in range(10) if (b == i).any()]
        return {"poisson_deviance": M.poisson_deviance(yy, mu_), "count_calibration_error": float(np.mean(err)),
                "mean_pred": float(mu_.mean()), "mean_obs": float(yy.mean()), "n": int(yy.size)}
    pv = {k_: np.asarray(p.predict(Xs["val"]), float) for k_, p in preds.items()}
    pt = {k_: np.asarray(p.predict(Xs["test"]), float) for k_, p in preds.items()}
    cands = {k_: {"complexity": c, "n_params": n_, "pred": preds[k_] if k_ != names[3] else m2,
                  "val": cm(ys["val"], pv[k_]), "test": cm(ys["test"], pt[k_])}
             for k_, c, n_ in zip(names, ("baseline rule: stars_prev_24h / 24", "baseline: Poisson GLM, 1 feature",
                                          f"Poisson GLM, {len(feats)} features",
                                          f"GBDT Poisson, {hgb.n_iter_} trees"),
                                  (0, 2, len(feats) + 1, F.n_params(m2)))}
    compute = {k_: time_us(preds[k_], Xs["val"]) for k_ in names}
    # NB2 dispersion of each candidate from its training residuals (method of moments); 0 = Poisson
    alpha = {}
    for k_, p in preds.items():
        mu_ = np.maximum(p.predict(Xs["train"][:200000]), 1e-9)
        yy = ys["train"][:200000]
        alpha[k_] = 0.0 if k_ == names[0] else max(0.0, float(np.mean((yy - mu_) ** 2 - mu_) / np.mean(mu_ ** 2)))

    def simulate(pred, h0: int, h1: int, a: float, seed: int):
        rows = np.flatnonzero(S[:, h0 - 24:h0].sum(1) > 0)
        rng = np.random.default_rng(seed)
        obs = S[rows, h0:h1].astype(float)
        sims = np.zeros((DRAWS, rows.size, h1 - h0))
        for d_ in range(DRAWS):
            Ssim = S.copy() if d_ == 0 else Ssim
            Ssim[:, h0:] = S[:, h0:]
            Ssim[rows, h0:h1] = 0
            for h in range(h0, h1):
                act = Ssim[rows, h - 24:h].sum(1) > 0
                mu_ = np.zeros(rows.size)
                if act.any():
                    mu_[act] = np.maximum(pred.predict(cascade_features(Ssim, h, tab, rows[act]).astype(float)), 0)
                if a > 0:
                    lam = rng.gamma(1.0 / a, np.maximum(mu_, 1e-12) * a)
                else:
                    lam = mu_
                draw = rng.poisson(lam)
                Ssim[rows, h] = draw
                sims[d_, :, h - h0] = draw
        return rows, obs, sims

    def lh_parts(pred, a, h0, h1, seed=0):
        rows, obs, sims = simulate(pred, h0, h1, a, seed)
        R_, D_ = obs.shape
        mean_sim = sims.mean(0)
        return (mean_sim, obs, np.ones_like(obs)), rows, obs, sims
    lhp, lh = {}, {}
    for k_ in names:
        pt_, _, _, _ = lh_parts(preds[k_], alpha[k_], 60, 72)
        lhp[k_] = pt_
        lh[k_] = measure(pt_, relative=True)
    Cv = lhp[names[0]][0].shape[0]
    sel = select_e4(cands, "poisson_deviance", names[:2], names, True, compute, lh, lhp, Cv, relative=True)
    kp = preds[sel["kept"]]
    pt_k, rows_t, obs_t, sims_t = lh_parts(kp, alpha[sel["kept"]], 72, 96)
    ks = M.ks_two_sample(np.log1p(sims_t.sum(2).ravel()), np.log1p(obs_t.sum(1)))
    lh_test = {"kept_relative_hourly_drift": measure(pt_k, relative=True), "kept_ks_log1p_repo_totals": ks,
               "simulated_total": float(sims_t.sum(2).mean(0).sum()), "observed_total": float(obs_t.sum()),
               "repositories": int(rows_t.size)}
    for k_ in names:
        if k_ != sel["kept"]:
            ptk, _, ob, sm = lh_parts(preds[k_], alpha[k_], 72, 96)
            lh_test[f"{k_}_relative_hourly_drift"] = measure(ptk, relative=True)
            lh_test[f"{k_}_ks_log1p_repo_totals"] = M.ks_two_sample(np.log1p(sm.sum(2).ravel()), np.log1p(ob.sum(1)))
    rows_dev = {k_: M.poisson_deviance(ys["test"], pt[k_]) for k_ in names}
    groups = Rr[sp["test"]]

    def dev_rows(yy, mu_):
        mu_ = np.maximum(mu_, 1e-9)
        return 2 * (np.where(yy > 0, yy * np.log(np.where(yy > 0, yy, 1) / mu_), 0.0) - (yy - mu_))
    dk = dev_rows(ys["test"], pt[sel["kept"]]); db = dev_rows(ys["test"], pt[sel["best_baseline_on_test"]])
    rule = {"mapping": s["rule_mapping"], "comparable_rule": None,
            "persistence_rule_test_deviance": rows_dev[names[0]]}
    splits = {"train": {"definition": s["splits"]["train"], "rows": int(sp["train"].sum())},
              "val": {"definition": s["splits"]["validation"], "rows": int(sp["val"].sum())},
              "test": {"definition": s["splits"]["test"], "rows": int(sp["test"].sum())}}
    limits = ("GH Archive public GitHub events, 2024-10-07..10 UTC: hourly non-bot stars of repositories that "
              "received a star in the previous 24 hours. The simulator may use this model for the next-hour arrival "
              "rate of endorsements (stars, saves, upvotes) of an item given its own recent endorsement history and "
              "activity, i.e. self-exciting attention dynamics on a platform with a trending surface. It may NOT "
              "infer: (1) why attention arrives (observational: contagion through followers, outside referrals from "
              "news sites and GitHub's own trending page are not separated); (2) cascades of other content kinds or "
              "platforms; (3) horizons beyond a day or the long-term popularity of an item; (4) the effect of "
              "promotion; (5) items with no endorsement in the last 24 hours (outside the units). Weekday data "
              "only.")
    ident = ("Observational: the dependence of next-hour stars on recent stars mixes contagion, referral bursts and "
             "recommender (trending) feedback; it is a reduced-form self-excitation, not a causal diffusion effect.")
    return finish(name, s["domain"], s, cands, sel, feats, "stars: non-bot stars of the repository in the next hour",
                  splits, ["test"], bc_rows(sel, dk, db, groups),
                  {"count_calibration_error": cands[sel["kept"]]["test"]["count_calibration_error"],
                   "mean_pred": cands[sel["kept"]]["test"]["mean_pred"],
                   "mean_obs": cands[sel["kept"]]["test"]["mean_obs"]},
                  {"nb_dispersion": alpha}, ident, limits, {f: f"{f}: see the E4 plan, models.{name}.feature_definitions" for f in feats}, t0, cmd,
                  {"mechanisms": ["social.endorsement_cascade"], "nb_alpha": alpha[sel["kept"]],
                   "feature_quantiles": F.feature_quantiles(Xs["train"], feats)},
                  rule, lh_test, "public GitHub repositories with recent stars",
                  "GitHub (github.com) stars", "2024-10-07..10 UTC")


# ====================================================================== Online Retail II
def _retail_days():
    return {"t0": _dt.date(2009, 12, 1)}


def _day(y, m, d) -> int:
    return (_dt.date(y, m, d) - _dt.date(2009, 12, 1)).days


def _cust_bucket(c: np.ndarray) -> np.ndarray:
    return np.array([int(hashlib.sha256(f"e4:{int(x)}".encode()).hexdigest()[:8], 16) % 100 for x in c])


def retail_frame():
    import pandas as pd
    from . import retail as RT
    d = RT.load()
    df = pd.DataFrame({k: d[k] for k in ("t", "invoice", "cancel", "product", "quantity", "price", "customer",
                                         "country")})
    df["day"] = df["t"] // 86400
    df["code"] = np.asarray([p["code"] for p in d["products"]], dtype=object)[df["product"].to_numpy()]
    from .gharchive import canonical_ids
    known = df["customer"].to_numpy() >= 0
    cid = np.full(len(df), -1, np.int64)
    cid[known] = canonical_ids(df["customer"].to_numpy()[known])
    df["cid"] = cid
    uk = d["countries"].index("United Kingdom") if "United Kingdom" in d["countries"] else -1
    df["uk"] = (df["country"].to_numpy() == uk).astype(float)
    return df


def customer_table(df):
    import pandas as pd
    k = df[df["cid"] >= 0].copy()
    k["spend"] = k["quantity"] * k["price"]
    sales = k[k["cancel"] == 0]
    g = sales.groupby(["cid", "day"]).agg(spend=("spend", "sum"), lines=("spend", "size"),
                                         products=("product", "nunique"), uk=("uk", "max")).reset_index()
    g = g.sort_values(["cid", "day"]).reset_index(drop=True)
    c = g["cid"].to_numpy(); dd = g["day"].to_numpy()
    first = np.r_[True, c[1:] != c[:-1]]
    last = np.r_[c[1:] != c[:-1], True]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
    n_before = np.arange(len(g)) - grp_start
    prev_day = np.where(first, -1, np.r_[-1, dd[:-1]])
    next_day = np.where(last, 10 ** 6, np.r_[dd[1:], 10 ** 6])
    first_day = dd[grp_start]
    cum_spend = g.groupby("cid")["spend"].cumsum().to_numpy()
    # cancellation share of all lines of the customer up to the day
    k2 = k.groupby(["cid", "day"]).agg(lines_all=("cancel", "size"), cancels=("cancel", "sum")).reset_index()
    k2 = k2.sort_values(["cid", "day"])
    k2["cum_lines"] = k2.groupby("cid")["lines_all"].cumsum()
    k2["cum_cancels"] = k2.groupby("cid")["cancels"].cumsum()
    g = g.merge(k2[["cid", "day", "cum_lines", "cum_cancels"]], on=["cid", "day"], how="left")
    gap_mean = np.where(n_before >= 2, (prev_day - first_day) / np.maximum(n_before - 1, 1), 400.0)
    month = np.array([(_dt.date(2009, 12, 1) + _dt.timedelta(days=int(x))).month for x in dd])
    X = {"log_orders_before": np.log1p(n_before), "log_tenure_days": np.log1p(dd - first_day),
         "log_recency_days": np.log1p(np.where(first, 400, dd - prev_day)),
         "log_mean_gap_days": np.log1p(gap_mean), "log_spend_today": np.log1p(np.maximum(g["spend"], 0)),
         "log_spend_total": np.log1p(np.maximum(cum_spend, 0)), "log_lines_today": np.log1p(g["lines"]),
         "log_products_today": np.log1p(g["products"]),
         "cancel_share": (g["cum_cancels"] / g["cum_lines"]).to_numpy(float), "uk": g["uk"].to_numpy(float),
         "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12)}
    y = ((next_day - dd) <= 90).astype(float)
    return X, y, dd, c, month


def fit_customer_return(cmd: str, t0: float) -> str:
    name = "customer_return"
    s = spec(name)
    df = retail_frame()
    X, y, day, cid, month = customer_table(df)
    feats = s["features"]
    Xm = np.column_stack([X[f] for f in feats])
    bucket = _cust_bucket(np.unique(cid))[np.searchsorted(np.unique(cid), cid)]
    trc, vac, tec = bucket < 70, (bucket >= 70) & (bucket < 85), bucket >= 85
    end = _day(2011, 9, 10)
    sp = {"train": trc & (day <= _day(2011, 3, 31)),
          "val": vac & (day >= _day(2010, 10, 1)) & (day <= _day(2011, 3, 31)),
          "test": tec & (day >= _day(2011, 4, 1)) & (day <= end),
          "test_time_only": trc & (day >= _day(2011, 4, 1)) & (day <= end)}
    Xs = {k: Xm[m] for k, m in sp.items()}
    ys = {k: y[m] for k, m in sp.items()}
    gs = {k: cid[m] for k, m in sp.items()}
    ym = (np.array([(_dt.date(2009, 12, 1) + _dt.timedelta(days=int(x))).year for x in day]) * 12 + month)
    per = {k: ym[m] for k, m in sp.items()}
    bcols = [feats.index("log_orders_before"), feats.index("log_recency_days")]
    cands, sel, pv, pt, lh_test = binary_fit(name, Xs, ys, gs, per, feats, bcols, cmd, t0)
    pk = pt[sel["kept"]]
    # retention curves of the test months' cohorts (30/60/90 days): predicted (90 d only) vs observed
    rule = {"mapping": s["rule_mapping"],
            "note": ("patrons return within 90 days with probability 1 unless dormant; the population has no "
                     "order-history churn rule; the implied constant is evaluated at 0.99 (log loss needs p < 1)"),
            "test": binary_metrics(ys["test"], np.full(ys["test"].size, 0.99))}
    extra = {"test_time_only": {k: binary_metrics(ys["test_time_only"], cands[k]["pred"].predict(Xs["test_time_only"]))
                                for k in cands}}
    splits = {k2: {"definition": s["splits"]["validation" if k2 == "val" else k2], "rows": int(m.sum()),
                   "users": int(np.unique(cid[m]).size)} for k2, m in sp.items()}
    limits = ("UCI Online Retail II: 5,942 customers of one UK online gift-ware retailer (many of them wholesalers), "
              "2009-12-01..2011-12-09. The simulator may use this model for the probability that a customer who "
              "bought from a shop today buys from the same shop again within 90 days, given its order history "
              "(frequency, recency, tenure, spend, basket breadth, returns). It may NOT infer: (1) retention for "
              "subscriptions, digital products or consumers of other shops (one retailer, mostly business "
              "customers); (2) responses to price, service quality or marketing (not features; observational); "
              "(3) horizons other than 90 days; (4) behaviour after 2011 or outside the UK-centred customer base; "
              "lines without a customer id (23%) are excluded, so guest buyers are not represented.")
    ident = ("Observational: customers' order histories and their returns share unobserved causes (business size, "
             "assortment fit); feature effects are associations.")
    return finish(name, s["domain"], s, cands, sel, feats, "returned: another order within 90 days", splits,
                  ["test", "test_time_only"], bc_binary(sel, cands, ys["test"], gs["test"], pt),
                  {"ece": M.ece(ys["test"], pk), "brier": M.brier(ys["test"], pk), "mean_pred": float(pk.mean()),
                   "mean_obs": float(ys["test"].mean()), "reliability": M.calibration_table(ys["test"], pk)},
                  extra, ident, limits, {f: f"{f}: see the E4 plan, models.{name}.feature_definitions" for f in feats}, t0, cmd,
                  {"mechanisms": ["commerce.customer_retention"], "prior_rate": float(ys["train"].mean()),
                   "feature_quantiles": F.feature_quantiles(Xs["train"], feats)},
                  rule, {"stability_months": lh_test}, "5,942 customers of one UK online gift retailer",
                  "the retailer's non-store online shop", "2009-12-01..2011-12-09")


def demand_table(df):
    import pandas as pd
    s = df[(df["cancel"] == 0) & (df["quantity"] > 0) & (df["price"] > 0)].copy()
    s = s[s["code"].astype(str).str[:1].str.isdigit()]
    s["week"] = (s["day"] + 1) // 7
    last_week = (_day(2011, 11, 28) + 1) // 7
    s = s[s["week"] <= last_week]
    units = s.groupby(["product", "week"])["quantity"].sum()
    retail = s[s["quantity"] <= 12].groupby(["product", "week"])["price"].median()
    cust = s[s["cid"] >= 0].groupby(["product", "week"])["cid"].apply(lambda x: set(x))
    prods = np.sort(s["product"].unique())
    W = last_week + 1
    pi = {p: i for i, p in enumerate(prods)}
    U = np.zeros((prods.size, W)); Pr = np.full((prods.size, W), np.nan)
    ui = units.index.to_frame().to_numpy()
    U[[pi[p] for p in ui[:, 0]], ui[:, 1]] = units.to_numpy()
    ri = retail.index.to_frame().to_numpy()
    Pr[[pi[p] for p in ri[:, 0]], ri[:, 1]] = retail.to_numpy()
    # carry the retail price forward
    for w in range(1, W):
        m = np.isnan(Pr[:, w])
        Pr[m, w] = Pr[m, w - 1]
    Cu = [[set() for _ in range(W)] for _ in range(prods.size)]
    for (p, w), v in cust.items():
        Cu[pi[p]][w] = v
    NC = np.zeros((prods.size, W))
    for i in range(prods.size):
        for w in range(1, W):
            NC[i, w] = len(set().union(*Cu[i][max(w - 4, 0):w]))
    return prods, U, Pr, NC, W


DEMAND_FEATURES = ["log_rel_price", "log_units_prev_week", "log_units_prev_4w", "log_customers_prev_4w",
                   "product_log_mean_units", "week_sin", "week_cos", "weeks_since_first_sale_log"]


def demand_rows(U, Pr, NC, ref, pmean, first, w, rows, Usim=None):
    Uh = U if Usim is None else Usim
    prev1 = Uh[rows, w - 1] if w >= 1 else np.zeros(rows.size)
    prev4 = Uh[rows, max(w - 4, 0):w].mean(1) if w >= 1 else np.zeros(rows.size)
    nc = NC[rows, w]
    woy = ((w + 48) % 52) / 52.0               # week 0 = 2009-11-30, ISO week 49
    return np.column_stack([np.log(Pr[rows, w] / ref[rows]), np.log1p(prev1), np.log1p(prev4), np.log1p(nc),
                            pmean[rows], np.full(rows.size, math.sin(2 * math.pi * woy)),
                            np.full(rows.size, math.cos(2 * math.pi * woy)), np.log1p(w - first[rows])])


def fit_product_demand(cmd: str, t0: float) -> str:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    name = "product_demand"
    s = spec(name)
    df = retail_frame()
    prods, U, Pr, NC, W = demand_table(df)
    wk = lambda y, m, d: (_day(y, m, d) + 1) // 7       # noqa: E731
    tr_end, va0, va1, te0, te1 = wk(2011, 3, 21), wk(2011, 3, 28), wk(2011, 6, 20), wk(2011, 6, 27), wk(2011, 11, 28)
    sold = U > 0
    first = np.argmax(sold, axis=1)
    has_price = ~np.isnan(Pr)
    tr_weeks = np.zeros(W, bool); tr_weeks[:tr_end + 1] = True
    n_tr_sales = (sold & tr_weeks).sum(1)
    eligible = n_tr_sales >= 20
    va_sales = (sold[:, va0:va1 + 1]).sum(1)
    newp = (~eligible) & (va_sales >= 4)
    alive = (np.arange(W)[None, :] >= first[:, None]) & has_price
    yfull = np.log1p(U)
    ref = np.ones(prods.size)
    pmean = np.zeros(prods.size)
    for i in range(prods.size):
        m = alive[i] & tr_weeks
        if m.any():
            ref[i] = np.median(Pr[i, m])
            pmean[i] = yfull[i, m].mean()
    def build(weeks, sel_rows):
        Xl, yl, gl, wl = [], [], [], []
        for w in weeks:
            rows = np.flatnonzero(sel_rows & alive[:, w] & (w >= first + 1))
            if rows.size == 0:
                continue
            Xl.append(demand_rows(U, Pr, NC, ref, pmean, first, w, rows))
            yl.append(yfull[rows, w]); gl.append(rows); wl.append(np.full(rows.size, w))
        return np.concatenate(Xl), np.concatenate(yl), np.concatenate(gl), np.concatenate(wl)
    Xs, ys, gs, ws = {}, {}, {}, {}
    for k, weeks, rs in (("train", range(1, tr_end + 1), eligible), ("val", range(va0, va1 + 1), eligible),
                         ("test", range(te0, te1 + 1), eligible)):
        Xs[k], ys[k], gs[k], ws[k] = build(weeks, rs)
    # new products: product mean from their own weeks before the row
    pmean_new = pmean.copy()
    feats = s["features"]
    assert feats == DEMAND_FEATURES
    i_pm, i_p4, i_pr = feats.index("product_log_mean_units"), feats.index("log_units_prev_4w"), feats.index("log_rel_price")
    b0 = P4.Persistence(i_pm, 1.0, "identity")
    b1 = P4.Persistence(i_p4, 1.0, "identity")
    mu, sd = Xs["train"].mean(0), Xs["train"].std(0); sd[sd < 1e-12] = 1.0
    rg = Ridge(alpha=1.0).fit((Xs["train"] - mu) / sd, ys["train"])
    m1 = P4.Linear(rg.coef_, rg.intercept_, mu, sd, "identity")
    hgb = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                        min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=20, random_state=0)
    hgb.fit(Xs["train"], ys["train"])
    m2 = P4.TreeRegressor.from_sklearn(hgb, link="identity")
    k_ = min(20000, len(Xs["val"]))
    if float(np.max(np.abs(portable_copy(m2).predict(Xs["val"][:k_]) - hgb.predict(Xs["val"][:k_])))) > 1e-8:
        raise AssertionError("exported regression trees differ from scikit-learn")
    m2._sk = hgb
    names = list(s["candidates"])
    preds = dict(zip(names, (b0, b1, m1, m2)))

    def price_coef(pred, X):
        a, b = X.copy(), X.copy()
        a[:, i_pr] -= 0.05; b[:, i_pr] += 0.05
        return float(np.mean(pred.predict(b) - pred.predict(a)) / 0.10)

    def met(yy, pp, pred, X):
        e = pp - yy
        return {"mse": float(np.mean(e ** 2)), "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
                "price_coefficient": price_coef(pred, X), "n": int(yy.size)}
    pv = {k: preds[k].predict(Xs["val"]) for k in names}
    pt = {k: preds[k].predict(Xs["test"]) for k in names}
    cands = {k: {"complexity": c, "n_params": n_, "pred": preds[k], "val": met(ys["val"], pv[k], preds[k], Xs["val"]),
                 "test": met(ys["test"], pt[k], preds[k], Xs["test"])}
             for k, c, n_ in zip(names, ("baseline: product training mean", "baseline: mean of the previous 4 weeks",
                                         f"ridge, {len(feats)} features", f"GBDT, {hgb.n_iter_} trees"),
                                 (0, 0, len(feats) + 1, F.n_params(m2)))}
    compute = {k: time_us(preds[k], Xs["val"]) for k in names}
    resid = {k: ys["train"] - preds[k].predict(Xs["train"]) for k in names}

    def forward(pred, res, w0, w1, rows_sel, seed=0):
        rng = np.random.default_rng(seed)
        rows_all = np.flatnonzero(rows_sel)
        cl, pe, so = [], [], []
        acc = {}
        for d_ in range(DRAWS):
            Us = U.copy()
            for w in range(w0, w1 + 1):
                rows = rows_all[alive[rows_all, w] & (w >= first[rows_all] + 1)]
                if rows.size == 0:
                    continue
                Xw = demand_rows(U, Pr, NC, ref, pmean, first, w, rows, Usim=Us)
                yy = pred.predict(Xw) + rng.choice(res, size=rows.size)
                Us[rows, w] = np.maximum(np.expm1(yy), 0.0)
                key = w
                acc.setdefault(key, [rows, np.zeros(rows.size), U[rows, w]])
                acc[key][1] += Us[rows, w] / DRAWS
        for w, (rows, sim, obs) in acc.items():
            cl.append(rows); pe.append(np.full(rows.size, w)); so.append((sim, obs))
        rows = np.concatenate(cl); per_ = np.concatenate(pe)
        sim = np.concatenate([a for a, _ in so]); obs = np.concatenate([b for _, b in so])
        ci, C_ = dense(rows); pi_, D_ = dense(per_)
        return parts(sim, obs, ci, C_, pi_, D_), C_
    lhp, lh = {}, {}
    for k in names:
        lhp[k], Cv = forward(preds[k], resid[k], va0, va1, eligible)
        lh[k] = measure(lhp[k], relative=True)
    sel = select_e4(cands, "mse", names[:2], names, True, compute, lh, lhp, Cv, relative=True)
    kp = preds[sel["kept"]]
    ptk, _ = forward(kp, resid[sel["kept"]], te0, te1, eligible)
    lh_test = {"kept_relative_weekly_drift": measure(ptk, relative=True)}
    for k in names:
        if k != sel["kept"]:
            p2, _ = forward(preds[k], resid[k], te0, te1, eligible)
            lh_test[f"{k}_relative_weekly_drift"] = measure(p2, relative=True)
    # new products (reported): product mean from own weeks before each row
    Xn, yn, gn, wn = build(range(te0, te1 + 1), newp)
    for j, (r, w) in enumerate(zip(gn, wn)):
        m = alive[r, :w]
        Xn[j, i_pm] = yfull[r, :w][m].mean() if m.any() else 0.0
    extra = {"test_new_products": {k: met(yn, preds[k].predict(Xn), preds[k], Xn) for k in names},
             "price_response": {"M1_linear_coefficient": float(m1.w[i_pr] / m1.scale[i_pr]),
                                "kept_mean_change_for_10pct_price_increase":
                                    float(np.mean(kp.predict(np.column_stack([Xs["test"][:, :i_pr],
                                                                              Xs["test"][:, i_pr] + math.log(1.1),
                                                                              Xs["test"][:, i_pr + 1:]]))
                                                  - kp.predict(Xs["test"])))}}
    # product-cluster bootstrap of M1's price coefficient
    rng = np.random.default_rng(0)
    up = np.unique(gs["train"])
    coefs = []
    Ztr = (Xs["train"] - mu) / sd
    for b in range(50):
        pick = rng.choice(up, size=up.size)
        cnt = np.bincount(np.searchsorted(up, pick), minlength=up.size)
        wts = cnt[np.searchsorted(up, gs["train"])]
        rb = Ridge(alpha=1.0).fit(Ztr, ys["train"], sample_weight=wts)
        coefs.append(rb.coef_[i_pr] / sd[i_pr])
    extra["price_response"]["M1_coefficient_ci95_product_bootstrap_50"] = [float(np.quantile(coefs, 0.025)),
                                                                           float(np.quantile(coefs, 0.975))]
    # rule: the population's purchase decision applied to the recent-mean baseline
    from ..population import models as PM
    dsn = PM._decisions()["purchase"]
    c_ = dsn.coef
    z0 = dsn.bias + c_["page_relevance"] * 0.5 + c_["price_sensitivity"] * 0.5 + c_["resource_satisfaction"] * 0.5 + \
        c_["belief_shops"] * 0.5 + c_["impulsiveness"] * 0.5
    prr = 1 - np.exp(Xs["test"][:, i_pr])
    sig = lambda z: 1 / (1 + np.exp(-z))           # noqa: E731
    rule_units = np.expm1(Xs["test"][:, i_p4]) * sig(z0 + c_["price_ratio"] * prr) / sig(z0)
    rule_y = np.log1p(np.maximum(rule_units, 0))
    rule = {"mapping": s["rule_mapping"], "z0": float(z0),
            "test": {"mse": float(np.mean((rule_y - ys["test"]) ** 2)), "mae": float(np.mean(np.abs(rule_y - ys["test"])))},
            "implied_price_coefficient": float(np.mean((np.log1p(np.expm1(Xs["test"][:, i_p4]) *
                                                                  sig(z0 + c_["price_ratio"] * (1 - np.exp(Xs["test"][:, i_pr] + 0.05))) / sig(z0))
                                                        - np.log1p(np.expm1(Xs["test"][:, i_p4]) *
                                                                   sig(z0 + c_["price_ratio"] * (1 - np.exp(Xs["test"][:, i_pr] - 0.05))) / sig(z0))) / 0.10))}
    er_k = (pt[sel["kept"]] - ys["test"]) ** 2
    er_b = (pt[sel["best_baseline_on_test"]] - ys["test"]) ** 2
    splits = {"train": {"definition": s["splits"]["train"], "rows": int(ys["train"].size),
                        "products": int(np.unique(gs["train"]).size)},
              "val": {"definition": s["splits"]["validation"], "rows": int(ys["val"].size)},
              "test": {"definition": s["splits"]["test"], "rows": int(ys["test"].size)},
              "test_new_products": {"definition": s["splits"]["test_new_products"], "rows": int(yn.size)}}
    limits = ("UCI Online Retail II, one UK online gift-ware retailer (many wholesale customers), 2009-12..2011-12, "
              f"{int(eligible.sum())} products with sales in at least 20 training weeks. The simulator may use this "
              "model for the weekly demand of a durable consumer product in an online shop given its recent sales "
              "and its price relative to its own usual price. It may NOT infer: (1) a causal price elasticity: "
              "prices were set by the retailer (seasonal re-pricing, clearance, promotions), so the price "
              "coefficient is an association and may be biased in either direction; (2) demand for services, "
              "subscriptions or digital goods; (3) competition between shops or substitution between products (not "
              "modelled); (4) prices far from a product's usual price (log relative prices outside the training "
              "quantiles); (5) other markets or years. Quantity discounts are excluded from the price by using "
              "retail-size lines (<= 12 units).")
    ident = ("Observational: retailer-set prices; demand shocks that cause re-pricing (season, clearance) confound "
             "the price association.")
    return finish(name, s["domain"], s, cands, sel, feats, "y = log1p(units sold in the week)", splits,
                  ["test", "test_new_products"], bc_rows(sel, er_k, er_b, gs["test"]),
                  {"bias": cands[sel["kept"]]["test"]["bias"], "calibration_error": abs(cands[sel["kept"]]["test"]["bias"])},
                  extra, ident, limits, {f: f"{f}: see the E4 plan, models.{name}.feature_definitions" for f in feats}, t0, cmd,
                  {"mechanisms": ["commerce.price_response"], "feature_quantiles": F.feature_quantiles(Xs["train"], feats),
                   "residual_sd": float(np.std(resid[sel["kept"]]))},
                  rule, lh_test, "products of one UK online gift retailer", "the retailer's online shop",
                  "2009-11-30..2011-12-04 (weeks)")


# ====================================================================== KuaiRand: sessions and creator return
def kuairand_sessions(raw):
    import pandas as pd
    from . import kuairand as K
    L = raw["logs"][["user_id", "time_ms", "day", "hourmin", "play_s", "is_click", "bucket"]].sort_values(
        ["user_id", "time_ms"], kind="mergesort")
    u = L["user_id"].to_numpy(); t = L["time_ms"].to_numpy()
    new = np.ones(len(L), bool)
    new[1:] = (u[1:] != u[:-1]) | (t[1:] - t[:-1] > K.SESSION_GAP_MS)
    L = L.assign(sid=np.cumsum(new))
    g = L.groupby("sid").agg(user=("user_id", "first"), start=("time_ms", "first"), end=("time_ms", "last"),
                             day=("day", "first"), hourmin=("hourmin", "first"), rows=("time_ms", "size"),
                             play=("play_s", "sum"), bucket=("bucket", "first")).reset_index(drop=True)
    g["y"] = np.log1p(g["play"].to_numpy() / 60.0)
    return g


SESSION_FEATURES = ["user_mean_log_minutes", "user_log_sessions_before", "user_log_rows_per_session", "active_days_7d",
                    "log_minutes_since_last_session", "sessions_earlier_today", "hour_sin", "hour_cos", "weekend"]


def session_features(g, y_override=None, pop_mean=None):
    """Features of every session from its user's earlier sessions (y_override replaces y for the history)."""
    from . import kuairand as K
    yv = g["y"].to_numpy() if y_override is None else y_override
    users = g["user"].to_numpy(); day = g["day"].to_numpy(); rows = g["rows"].to_numpy(float)
    D = int(day.max()) + 2
    U = int(users.max()) + 1
    sy = np.zeros((U, D)); sn = np.zeros((U, D)); sr = np.zeros((U, D))
    np.add.at(sy, (users, day), yv); np.add.at(sn, (users, day), 1); np.add.at(sr, (users, day), rows)
    cy = np.cumsum(sy, 1) - sy; cn = np.cumsum(sn, 1) - sn; cr = np.cumsum(sr, 1) - sr      # earlier days only
    pm = float(np.mean(yv)) if pop_mean is None else pop_mean
    n_before = cn[users, day]
    mean_y = np.where(n_before > 0, cy[users, day] / np.maximum(n_before, 1), pm)
    act = sn > 0
    a7 = np.zeros(len(g))
    for d in np.unique(day):
        m = day == d
        a7[m] = act[users[m], max(d - 7, 0):d].sum(1)
    prev_end = np.r_[-1, g["end"].to_numpy()[:-1]]
    same_user_prev = np.r_[False, users[1:] == users[:-1]]
    gap_min = np.where(same_user_prev, (g["start"].to_numpy() - prev_end) / 60000.0, 10080.0)
    earlier_today = g.groupby(["user", "day"]).cumcount().to_numpy()
    hour = (g["hourmin"].to_numpy() // 100) % 24
    wd = np.array([(K.D0 + _dt.timedelta(days=int(d))).weekday() for d in day])
    return np.column_stack([mean_y, np.log1p(n_before), np.log1p(np.where(n_before > 0, cr[users, day] /
                                                                            np.maximum(n_before, 1), 0.0)),
                            a7, np.log1p(np.clip(gap_min, 0, 10080)), earlier_today, np.sin(2 * np.pi * hour / 24),
                            np.cos(2 * np.pi * hour / 24), (wd >= 5).astype(float)])


def c2_session_rule(levels, durations, n: int = 100000, seed: int = 0) -> np.ndarray:
    """Quantiles of log1p(minutes watched in a session) under the C2 session loop at default state."""
    from ..social.consumption import BUILTIN as b
    rng = np.random.default_rng(seed)
    k = 10
    heavy = rng.random(n) < 0.4
    budget = np.exp(rng.normal(np.where(heavy, math.log(120.0), math.log(75.0)), 0.35)) * 60.0
    dur = rng.choice(durations, size=(n, k))
    aff = 0.2
    sat = np.full(n, 0.5)
    util = b["open_bias"] + b["open_affinity"] * aff + b["open_sat"] * (sat[:, None] - 0.5) + \
        b["open_len"] * np.log(np.maximum(dur, 1.0) / 60.0)
    eu = np.exp(util)
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(eu, 1)], 1)
    J = np.arange(k)
    win = csum[:, np.minimum(J + int(b["window"]), k)] - csum[:, J]
    p_open = eu / (math.exp(b["outside"]) + win)
    p_bounce = 1 / (1 + math.exp(-(b["bounce_bias"] + b["bounce_affinity"] * aff)))
    lam = b["hazard_per_s"] * math.exp(b["hazard_affinity"] * aff)
    elapsed = np.zeros(n); watched = np.zeros(n); active = np.ones(n, bool)
    for p in range(k):
        hp = active & (elapsed + b["scroll_s"] <= budget)
        elapsed = np.where(hp, elapsed + b["scroll_s"], elapsed)
        o = hp & (rng.random(n) < p_open[:, p])
        bnc = rng.random(n) < p_bounce
        T = np.where(bnc, 1 + 5 * rng.random(n), rng.exponential(1 / lam, n))
        T = np.minimum(np.minimum(T, dur[:, p]), np.maximum(budget - elapsed, 0))
        T = np.where(o, T, 0.0)
        elapsed += T; watched += T
        vw = o & (T >= np.minimum(30.0, 0.5 * dur[:, p]))
        cont = rng.random(n) < 1 / (1 + np.exp(-(b["continue_bias"] + b["continue_sat"] * (sat - 0.5) +
                                                  b["continue_view"] * vw + b["continue_minutes"] * elapsed / 60.0)))
        active = active & np.where(hp, cont, True) & (elapsed < budget - 1e-9)
    return np.quantile(np.log1p(watched / 60.0), levels)


def fit_session_length(cmd: str, t0: float) -> str:
    from . import kuairand as K
    name = "session_length"
    s = spec(name)
    levels = np.asarray(s["quantile_levels"])
    raw = K.load_raw()
    g = kuairand_sessions(raw)
    split_u = K.split_of_bucket(g["bucket"].to_numpy())
    d = g["day"].to_numpy()
    D = lambda m_, dd: K.day_num(2022, m_, dd)      # noqa: E731
    sp = {"train": (split_u == "train") & (d >= D(4, 22)) & (d <= D(5, 2)),
          "val": (split_u == "val") & (d >= D(4, 29)) & (d <= D(5, 2)),
          "test": (split_u == "test") & (d >= D(5, 3)) & (d <= D(5, 8))}
    pop_mean = float(g["y"].to_numpy()[sp["train"]].mean())
    X = session_features(g, pop_mean=pop_mean)
    y = g["y"].to_numpy()
    feats = s["features"]
    assert feats == SESSION_FEATURES
    Xs = {k: X[m] for k, m in sp.items()}
    ys = {k: y[m] for k, m in sp.items()}
    b0 = F.fit_binned_quantiles(Xs["train"], ys["train"], 0, levels, n_bins=1)
    b1 = F.fit_binned_quantiles(Xs["train"], ys["train"], 0, levels, n_bins=10)
    m1, h1 = F.fit_quantile_net(Xs["train"], ys["train"], Xs["val"], ys["val"], levels, hidden=(), seed=0, threads=1)
    m2, h2 = F.fit_quantile_net(Xs["train"], ys["train"], Xs["val"], ys["val"], levels, hidden=(32, 32), seed=0,
                                threads=1)
    names = list(s["candidates"])
    preds = dict(zip(names, (b0, b1, m1, m2)))

    def qm(yy, Q):
        u = M.pit_fast(yy, Q, levels)
        return {"pinball": M.pinball(yy, Q, levels), "pit_ks": M.ks_uniform(u),
                "coverage_80": M.coverage(yy, Q, levels, 0.1, 0.9), "coverage_90": M.coverage(yy, Q, levels, 0.05, 0.95),
                "n": int(yy.size)}
    Qv = {k: preds[k].predict(Xs["val"]) for k in names}
    Qt = {k: preds[k].predict(Xs["test"]) for k in names}
    cands = {k: {"complexity": c, "n_params": F.n_params(preds[k]), "pred": preds[k], "val": qm(ys["val"], Qv[k]),
                 "test": qm(ys["test"], Qt[k])}
             for k, c in zip(names, ("baseline: global quantiles", "baseline: quantiles by own-history decile",
                                     "monotone quantile network, linear", "monotone quantile network 2x32"))}
    compute = {k: time_us(preds[k], Xs["val"]) for k in names}

    def forward(pred, mask, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(mask)
        days = np.unique(d[idx])
        sim_acc = np.zeros(idx.size)
        for _ in range(DRAWS):
            yy = y.copy()
            for dd in days:
                Xd = session_features(g, y_override=yy, pop_mean=pop_mean)
                m = idx[d[idx] == dd]
                draw = P.interpolate_quantiles(pred.predict(Xd[m]), levels, rng.random(m.size))
                yy[m] = draw
            sim_acc += yy[idx] / DRAWS
        ci, C_ = dense(g["user"].to_numpy()[idx]); pi_, D_ = dense(d[idx])
        return parts(sim_acc, y[idx], ci, C_, pi_, D_), C_
    lhp, lh = {}, {}
    for k in names:
        lhp[k], Cv = forward(preds[k], sp["val"])
        lh[k] = measure(lhp[k])
    sel = select_e4(cands, "pinball", names[:2], names, True, compute, lh, lhp, Cv)
    kp = preds[sel["kept"]]
    ptk, _ = forward(kp, sp["test"])
    lh_test = {"kept_daily_drift": measure(ptk)}
    for k in names:
        if k != sel["kept"]:
            lh_test[f"{k}_daily_drift"] = measure(forward(preds[k], sp["test"])[0])
    durs = raw["logs"]["duration_s"].to_numpy()
    durs = durs[np.isfinite(durs) & (durs > 0)]
    qrule = c2_session_rule(levels, durs)
    rule = {"mapping": s["rule_mapping"], "rule_quantiles": qrule.tolist(),
            "test": qm(ys["test"], np.tile(qrule, (ys["test"].size, 1)))}
    gk = g["user"].to_numpy()[sp["test"]]
    rk = M.pinball_rows(ys["test"], Qt[sel["kept"]], levels)
    rb = M.pinball_rows(ys["test"], Qt[sel["best_baseline_on_test"]], levels)
    Qk = Qt[sel["kept"]]
    splits = {k: {"definition": s["splits"]["validation" if k == "val" else k], "rows": int(m.sum()),
                  "users": int(np.unique(g["user"].to_numpy()[m]).size)} for k, m in sp.items()}
    limits = ("KuaiRand-Pure: 27,285 users of the Kuaishou short-video app (China), featured feed, sessions of "
              "2022-04-22..05-08 (30-minute idle gap). The simulator may use this model for how many minutes an "
              "individual spends watching in one feed session, given its own earlier sessions and the time of day. "
              "It may NOT infer: (1) total app time (only plays of the 7,583 pool videos are logged, so sessions are "
              "shorter than real app sessions); (2) other platforms or content kinds (articles, long video); (3) how "
              "session length responds to what is shown (no content inputs; use content_open and watch_time per "
              "item for that); (4) other populations or periods (one month including a holiday).")
    ident = "Observational: individual differences in session length are associations with the individual's history."
    return finish(name, s["domain"], s, cands, sel, feats, "y = log1p(minutes of logged play in the session)",
                  splits, ["test"], bc_rows(sel, rk, rb, gk),
                  {"pit_ks": cands[sel["kept"]]["test"]["pit_ks"], "coverage_80": cands[sel["kept"]]["test"]["coverage_80"],
                   "coverage_90": cands[sel["kept"]]["test"]["coverage_90"],
                   "reliability": [{"level": float(lv), "mean_pred": float(lv),
                                    "mean_obs": float(np.mean(ys["test"] <= Qk[:, i]))} for i, lv in enumerate(levels)]},
                  {"fit_history": {"M1": h1, "M2": h2}}, ident, limits, {f: f"{f}: see the E4 plan, models.{name}.feature_definitions" for f in feats},
                  t0, cmd, {"mechanisms": ["attention.session_length"], "quantile_levels": levels.tolist(),
                            "feature_quantiles": F.feature_quantiles(Xs["train"], feats), "population_mean_y": pop_mean},
                  rule, lh_test, "27,285 Kuaishou users (selected sample)", "Kuaishou featured feed (tab 1 and others)",
                  "2022-04-22..05-08")


CREATOR_FEATURES = ["log_plays_author_before", "log_plays_author_today", "log_days_since_first_play", "author_log_pop_7d",
                    "user_log_plays_7d", "user_log_authors_7d", "follows_author", "liked_author_today",
                    "watch_fraction_today", "weekend"]


def creator_table(raw):
    import pandas as pd
    from . import kuairand as K
    L = raw["logs"]
    plays = L[L["is_click"] == 1][["user_id", "author", "day", "play_s", "duration_s", "is_like"]].copy()
    plays["wf"] = np.minimum(plays["play_s"] / plays["duration_s"].where(plays["duration_s"] > 0, np.nan), 3.0)
    g = plays.groupby(["user_id", "author", "day"]).agg(n=("day", "size"), likes=("is_like", "max"),
                                                       wf=("wf", "mean")).reset_index()
    g = g.sort_values(["user_id", "author", "day"]).reset_index(drop=True)
    ua = g["user_id"].to_numpy().astype(np.int64) * 100000 + g["author"].to_numpy()
    dd = g["day"].to_numpy()
    first = np.r_[True, ua[1:] != ua[:-1]]
    last = np.r_[ua[1:] != ua[:-1], True]
    start = np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
    cumn = g.groupby(["user_id", "author"])["n"].cumsum().to_numpy() - g["n"].to_numpy()
    first_day = dd[start]
    next_day = np.where(last, 10 ** 6, np.r_[dd[1:], 10 ** 6])
    y = ((next_day > dd) & (next_day <= dd + 7)).astype(float)
    # follows: any is_follow on a video of the author on or before d
    fol = L[L["is_follow"] == 1].groupby(["user_id", "author"])["day"].min()
    fk = fol.index.get_level_values(0).to_numpy().astype(np.int64) * 100000 + fol.index.get_level_values(1).to_numpy()
    fmap = dict(zip(fk, fol.to_numpy()))
    fday = np.array([fmap.get(x, 10 ** 6) for x in ua])
    # author popularity and user activity in the previous 7 days
    Dn = int(L["day"].max()) + 2
    act_sets = {x: set(L.loc[L["day"] == x, "user_id"].unique()) for x in range(Dn)}
    act7 = np.array([len(set().union(*[act_sets.get(z, set()) for z in range(x - 7, x)])) for x in range(Dn)], float)
    ap = plays.groupby(["author", "day"]).size()
    A = int(L["author"].max()) + 2
    AP = np.zeros((A, Dn)); AP[ap.index.get_level_values(0), ap.index.get_level_values(1)] = ap.to_numpy()
    cAP = np.concatenate([np.zeros((A, 1)), np.cumsum(AP, 1)], 1)
    au = g["author"].to_numpy()
    pop7 = (cAP[au, dd] - cAP[au, np.maximum(dd - 7, 0)]) / np.maximum(act7[dd], 1.0) * 1000.0
    up = plays.groupby(["user_id", "day"]).size()
    Uu = int(L["user_id"].max()) + 1
    UP = np.zeros((Uu, Dn)); UP[up.index.get_level_values(0), up.index.get_level_values(1)] = up.to_numpy()
    cUP = np.concatenate([np.zeros((Uu, 1)), np.cumsum(UP, 1)], 1)
    us = g["user_id"].to_numpy()
    uplays7 = cUP[us, dd] - cUP[us, np.maximum(dd - 7, 0)]
    nauth = np.zeros(len(g))
    pa = g[["user_id", "author", "day"]]
    for x in np.unique(dd):
        w = pa[(pa["day"] < x) & (pa["day"] >= x - 7)].groupby("user_id")["author"].nunique()
        m = dd == x
        nauth[m] = w.reindex(us[m]).fillna(0).to_numpy()
    wd = np.array([(K.D0 + _dt.timedelta(days=int(x))).weekday() for x in dd])
    X = np.column_stack([np.log1p(cumn), np.log1p(g["n"]), np.log1p(dd - first_day), np.log1p(pop7),
                         np.log1p(uplays7), np.log1p(nauth), (fday <= dd).astype(float), g["likes"].to_numpy(float),
                         np.nan_to_num(g["wf"].to_numpy(float)), (wd >= 5).astype(float)])
    # exposures of the user to the author's videos in the previous 7 days (for the rule)
    ex = L.groupby(["user_id", "author", "day"]).size()
    ek = ex.index.to_frame().to_numpy()
    return X, y, dd, us, g["author"].to_numpy(), ek, ex.to_numpy()


def fit_creator_return(cmd: str, t0: float) -> str:
    from . import kuairand as K
    name = "creator_return"
    s = spec(name)
    raw = K.load_raw()
    X, y, dd, us, au, ek, exn = creator_table(raw)
    feats = s["features"]
    assert feats == CREATOR_FEATURES
    bmap = {u: K.user_bucket(u) for u in np.unique(us)}
    split_u = K.split_of_bucket(np.array([bmap[u] for u in us]))
    D = lambda m_, d_: K.day_num(2022, m_, d_)      # noqa: E731
    sp = {"train": (split_u == "train") & (dd <= D(4, 24)),
          "val": (split_u == "val") & (dd >= D(4, 18)) & (dd <= D(4, 24)),
          "test": (split_u == "test") & (dd >= D(4, 25)) & (dd <= D(5, 1))}
    Xs = {k: X[m] for k, m in sp.items()}
    ys = {k: y[m] for k, m in sp.items()}
    gs = {k: us[m] for k, m in sp.items()}
    per = {k: dd[m] for k, m in sp.items()}
    bcols = [feats.index("log_plays_author_before"), feats.index("log_plays_author_today")]
    cands, sel, pv, pt, lh_test = binary_fit(name, Xs, ys, gs, per, feats, bcols, cmd, t0)
    pk = pt[sel["kept"]]
    # rule: per-exposure open probability of the C2 open rule (familiar creator) and the user's recent exposures
    from ..social.consumption import BUILTIN as b
    u_ = b["open_bias"] + b["open_affinity"] * 0.2 + b["open_familiar"] * 1.0 + b["open_len"] * math.log(120 / 60)
    p1 = math.exp(u_) / (math.exp(b["outside"]) + (1 + b["window"]) * math.exp(u_))
    import pandas as pd
    exs = pd.Series(exn, index=pd.MultiIndex.from_arrays([ek[:, 0], ek[:, 1], ek[:, 2]]))
    tm = sp["test"]
    n_exp = np.zeros(tm.sum())
    ut, at, dt_ = us[tm], au[tm], dd[tm]
    for lag in range(1, 8):
        idx = pd.MultiIndex.from_arrays([ut, at, dt_ - lag])
        n_exp += exs.reindex(idx).fillna(0).to_numpy()
    p_rule = np.clip(1 - (1 - p1) ** np.maximum(n_exp, 1), 1e-6, 1 - 1e-6)
    rule = {"mapping": s["rule_mapping"], "per_exposure_open_probability": p1,
            "test": binary_metrics(ys["test"], p_rule)}
    splits = {k: {"definition": s["splits"]["validation" if k == "val" else k], "rows": int(m.sum()),
                  "users": int(np.unique(us[m]).size)} for k, m in sp.items()}
    limits = ("KuaiRand-Pure: 27,285 users of the Kuaishou short-video app (China), featured feed, 2022-04-08..05-08. "
              "The simulator may use this model for the probability that an individual who consumed a creator's "
              "item today consumes another item of the same creator within 7 days, given its familiarity, follow and "
              "like signals and both sides' activity. It may NOT infer: (1) the individual's preference net of "
              "exposure (re-exposure by the production recommender and by follow feeds is part of the outcome); "
              "(2) item-level repeat consumption (rare in the data); (3) other platforms, content kinds or periods; "
              "(4) horizons other than 7 days; plays count only pool videos.")
    ident = ("Observational: the outcome mixes the recommender's re-exposure and the individual's choice; feature "
             "effects are associations.")
    return finish(name, s["domain"], s, cands, sel, feats, "returned: plays another video of the creator within 7 days",
                  splits, ["test"], bc_binary(sel, cands, ys["test"], gs["test"], pt),
                  {"ece": M.ece(ys["test"], pk), "brier": M.brier(ys["test"], pk), "mean_pred": float(pk.mean()),
                   "mean_obs": float(ys["test"].mean()), "reliability": M.calibration_table(ys["test"], pk)},
                  {}, ident, limits, {f: f"{f}: see the E4 plan, models.{name}.feature_definitions" for f in feats}, t0, cmd,
                  {"mechanisms": ["feed.creator_return"], "prior_rate": float(ys["train"].mean()),
                   "feature_quantiles": F.feature_quantiles(Xs["train"], feats)},
                  rule, {"stability_days": lh_test}, "27,285 Kuaishou users (selected sample)",
                  "Kuaishou short-video app", "2022-04-08..05-08")


# ====================================================================== MIND with the shared semantic encoder
NEWS_SEM_FEATURES = ["text_cos", "text_cos_z", "cat_share", "log_hist_len", "pop_log_clicks_per_1k", "age_log_hours",
                     "slate_size_log"]


def fit_news_semantic(cmd: str, t0: float, emb_path: str) -> str:
    from . import mind
    name = "news_click_semantic"
    s = spec(name)
    data = mind.load()
    z = np.load(emb_path)
    if list(z["news_id"]) != list(data["news"]["news_id"].astype(str)):
        raise ValueError("embedding rows are not in mind.load() news order")
    E = z["emb"].astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    df, meta = mind.build(data, E=E)
    m = mind.news_masks(df, data["beh"])
    feats = s["features"]
    assert feats == NEWS_SEM_FEATURES
    X = df[feats].to_numpy(float)
    y = df["y"].to_numpy(float)
    sp = {"train": m["train"], "val": m["val"], "test": m["test"]}
    Xs = {k: X[v] for k, v in sp.items()}
    ys = {k: y[v] for k, v in sp.items()}
    gs = {k: df.loc[v, "user_id"].to_numpy() for k, v in sp.items()}
    per = {k: (df.loc[v, "hour"].to_numpy() // 4) for k, v in sp.items()}
    bcols = [feats.index(f) for f in ("pop_log_clicks_per_1k", "age_log_hours", "slate_size_log")]
    cands, sel, pv, pt, lh_test = binary_fit(name, Xs, ys, gs, per, feats, bcols, cmd, t0)
    pk = pt[sel["kept"]]
    # rule: C2 open rule with the MiniLM affinity, window = next 3 items of the impression
    from ..social.consumption import BUILTIN as b
    te = df.loc[m["test"], ["imp", "text_cos"]].reset_index(drop=True)
    util = b["open_bias"] + b["open_affinity"] * te["text_cos"].to_numpy() + b["open_len"] * math.log(120 / 60)
    eu = np.exp(np.clip(util, -40, 40))
    imp = te["imp"].to_numpy()
    win = np.zeros(eu.size)
    for k in range(int(b["window"]) + 1):
        sh = np.r_[eu[k:], np.zeros(k)]
        same = np.r_[imp[k:] == imp[:imp.size - k] if k else np.ones(imp.size, bool), np.zeros(k, bool)]
        win += np.where(same, sh, 0.0)
    p_rule = np.clip(eu / (math.exp(b["outside"]) + win), 1e-6, 1 - 1e-6)
    rule = {"mapping": s["rule_mapping"], "test": binary_metrics(ys["test"], p_rule)}
    old = None
    try:
        from .registry import load as _l
        o = _l("news_click@1")
        old = o.metrics.get("log_loss")
    except Exception:          # noqa: BLE001 - a reference number only
        old = None
    extra = {"reference_news_click_v1_test_log_loss": old,
             "ranking_test": M.group_ranking(ys["test"], pk, df.loc[m["test"], "imp"].to_numpy())}
    splits = {k: {"definition": s["splits"]["validation" if k == "val" else k], "rows": int(v.sum()),
                  "users": int(df.loc[v, "user_id"].nunique())} for k, v in sp.items()}
    limits = ("MIND-small: 50,000 sampled Microsoft News users (US, English), impressions 2019-11-09..15 chosen by the "
              "production recommender. The simulator may use this model for the probability that an individual "
              "opens an article shown in a slate, given the cosine of the article's text and its consumption history "
              "in the shared MiniLM-L6 space, its category share, popularity per active individual, age and slate "
              "size. It may NOT infer: (1) clicks on video or social posts (content_open); (2) other populations, "
              "languages or periods; (3) the causal effect of showing an item (no randomised exposure); (4) dwell or "
              "reading time; (5) affinities from other encoders or projections that do not preserve MiniLM cosines. "
              "Licence: Microsoft Research terms, non-commercial; acquisition route unresolved (BLOCKERS B-1).")
    ident = ("Observational: the production recommender chose the impressions, so click probabilities are conditional "
             "on that selection; feature effects are associations.")
    return finish(name, s["domain"], s, cands, sel, feats, "clicked: the user opened the news item", splits, ["test"],
                  bc_binary(sel, cands, ys["test"], gs["test"], pt),
                  {"ece": M.ece(ys["test"], pk), "brier": M.brier(ys["test"], pk), "mean_pred": float(pk.mean()),
                   "mean_obs": float(ys["test"].mean()), "reliability": M.calibration_table(ys["test"], pk)},
                  extra, ident, limits, {f: f"{f}: see the E4 plan, models.{name}.feature_definitions" for f in feats}, t0, cmd,
                  {"mechanisms": ["feed.article_open_semantic"], "prior_rate": meta["prior_rate"],
                   "feature_quantiles": F.feature_quantiles(Xs["train"], feats),
                   "text_representation": {"encoder": str(z["encoder"][0]), "text": "title + ' ' + abstract",
                                           "profile": "L2-normalised mean of the clicked history's embeddings"}},
                  rule, {"stability_4h_blocks": lh_test}, "50,000 sampled Microsoft News users (US, English)",
                  "Microsoft News (msn.com)", "2019-11-09..15")


# ---------------------------------------------------------------- held-out rows for the reproducers (same code paths)
def demand_test_rows(df=None):
    """(X, y) of product_demand's test split, built exactly as fit_product_demand builds them."""
    df = retail_frame() if df is None else df
    prods, U, Pr, NC, W = demand_table(df)
    wk = lambda y, m, d: (_day(y, m, d) + 1) // 7       # noqa: E731
    tr_end, te0, te1 = wk(2011, 3, 21), wk(2011, 6, 27), wk(2011, 11, 28)
    sold = U > 0
    first = np.argmax(sold, axis=1)
    has_price = ~np.isnan(Pr)
    tr_weeks = np.zeros(W, bool); tr_weeks[:tr_end + 1] = True
    eligible = (sold & tr_weeks).sum(1) >= 20
    alive = (np.arange(W)[None, :] >= first[:, None]) & has_price
    yfull = np.log1p(U)
    ref = np.ones(prods.size)
    pmean = np.zeros(prods.size)
    for i in range(prods.size):
        m = alive[i] & tr_weeks
        if m.any():
            ref[i] = np.median(Pr[i, m])
            pmean[i] = yfull[i, m].mean()
    Xl, yl = [], []
    for w in range(te0, te1 + 1):
        rows = np.flatnonzero(eligible & alive[:, w] & (w >= first + 1))
        if rows.size:
            Xl.append(demand_rows(U, Pr, NC, ref, pmean, first, w, rows))
            yl.append(yfull[rows, w])
    return np.concatenate(Xl), np.concatenate(yl)


def cascade_test_rows():
    """(X, y) of star_cascade's test split (hours 72-95), built exactly as fit_cascade builds them."""
    tab = github_cascade_table()
    S = tab["S"]
    Xl, yl = [], []
    for h in range(72, 96):
        rows = np.flatnonzero(S[:, h - 24:h].sum(1) > 0)
        Xl.append(cascade_features(S, h, tab, rows).astype(np.float32))
        yl.append(S[rows, h])
    return np.concatenate(Xl).astype(np.float64), np.concatenate(yl).astype(np.float64)


FITS = {"repo_participation_return": fit_participation, "star_cascade": fit_cascade,
        "customer_return": fit_customer_return, "product_demand": fit_product_demand,
        "session_length": fit_session_length, "creator_return": fit_creator_return,
        "news_click_semantic": fit_news_semantic}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, choices=sorted(FITS))
    ap.add_argument("--embeddings", default=None, help="MiniLM embeddings of MIND news (news_click_semantic)")
    a = ap.parse_args(argv)
    for name in a.model:
        t0 = time.time()
        cmd = "PYTHONPATH=src python -m crucible.calibration.fit_e4 --model " + name
        if name == "news_click_semantic":
            out = FITS[name](cmd, t0, a.embeddings)
        else:
            out = FITS[name](cmd, t0)
        print(name, "->", out, f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
