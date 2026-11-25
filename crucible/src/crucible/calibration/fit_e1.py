"""Fit, compare and register the E1 behavioural models (content consumption and browsing) exactly as prespecified
in `docs/v5/calibration/plans/E1_evaluation_plan.json`.

    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "ws-E1: fit" -- \\
        python -m crucible.calibration.fit_e1 --models content,watch,return,session,news

Each model writes `docs/v5/calibration/models/<name>/{model_card.json, params.npz}` and
`docs/v5/calibration/reports/<name>.{json,md}`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from . import fitlib as F
from . import kuairand as K
from . import metrics as M
from . import predictors as P

PLAN = "docs/v5/calibration/plans/E1_evaluation_plan.json"
VERSION = 1
D = K.day_num

KR_POP = ("27,285 anonymised Kuaishou users chosen by the KuaiRand authors (China, 2022); interactions only with "
          "the 7,583 videos of the random-exposure pool")
KR_PLATFORM = "Kuaishou short-video app, single-column featured feed (tab 1)"
KR_PERIOD = "2022-04-08 to 2022-05-08"


def _split_info(df, mask, definition):
    return {"definition": definition, "rows": int(mask.sum()), "users": int(df.loc[mask, "user_id"].nunique())}


def _fit_meta(t0, cmd):
    return {"commit": F.head_commit(), "seconds": time.time() - t0, "command": cmd}


def _bootstrap(y, p_kept, p_base, groups):
    lk = M.row_log_loss(y, p_kept)
    lb = M.row_log_loss(y, p_base)
    ci_k = M.cluster_bootstrap(lk, groups)
    ci_d = M.cluster_bootstrap(lk - lb, groups)
    return ci_k, ci_d


# ====================================================================== content consumption (choice)
CHOICE_LIMITS = (
    "Evidence: KuaiRand-Pure, {pop}; platform {plat}; period {per}. The simulator may use this model for the "
    "probability that an individual keeps watching an item its feed shows, as a function of the individual's "
    "activity, consume rate, topic affinity and creator familiarity and of the item's popularity, consume rate, age "
    "and duration. It may NOT infer: (1) behaviour on platforms other than an autoplay short-video feed (search, "
    "long-form video, articles: use news_click for article choice); (2) item ages beyond 30 days (all pool videos "
    "were uploaded 2022-04-09..11, so age is confounded with calendar time) or durations outside 6 s - 20 min; "
    "(3) responses of other populations or periods (one Chinese app, one month in 2022, a selected user sample, a "
    "holiday period 2022-04-30..05-04 inside the data); (4) the effect of social context, prices or advertising, "
    "which are not features; (5) quantities on another scale: activity counts cover only pool videos, so map "
    "scale-dependent inputs by quantile (feature_quantiles). Topic affinity is measured from one primary tag; with "
    "embedding-based affinity the mapping is by quantile, not by value. Repeat consumption is rare in the data "
    "(item_played_before is set on 0.007% of rows), so its coefficient is weakly identified.")


def content_masks(cf) -> dict:
    """The prespecified splits of the two choice models (plan: content_consumption, content_consumption_feed)."""
    day = cf["day"].to_numpy()
    sp = cf["split_user"].to_numpy()
    std = cf["is_rand"].to_numpy() == 0
    rnd = ~std
    inr = lambda a, b: (day >= a) & (day <= b)          # noqa: E731
    return {
        "obs_train": std & (sp == "train") & inr(D(2022, 4, 12), D(2022, 5, 2)),
        "obs_val": std & (sp == "val") & inr(D(2022, 4, 29), D(2022, 5, 2)),
        "obs_test": std & (sp == "test") & inr(D(2022, 5, 3), D(2022, 5, 8)),
        "obs_test_time_only": std & (sp == "train") & inr(D(2022, 5, 3), D(2022, 5, 8)),
        "rnd_train": rnd & (sp == "train") & inr(D(2022, 4, 22), D(2022, 5, 2)),
        "rnd_val": rnd & (sp == "val") & inr(D(2022, 4, 29), D(2022, 5, 2)),
        "rnd_test": rnd & (sp == "test") & inr(D(2022, 5, 3), D(2022, 5, 8)),
        "rnd_transport": rnd & (sp == "test") & inr(D(2022, 4, 22), D(2022, 5, 8)),
    }


def watch_masks(cf) -> dict:
    day = cf["day"].to_numpy()
    sp = cf["split_user"].to_numpy()
    std = cf["is_rand"].to_numpy() == 0
    inr = lambda a, b: (day >= a) & (day <= b)          # noqa: E731
    return {"train": (sp == "train") & ((std & inr(D(2022, 4, 12), D(2022, 5, 2))) |
                                        (~std & inr(D(2022, 4, 22), D(2022, 5, 2)))),
            "val": (sp == "val") & inr(D(2022, 4, 29), D(2022, 5, 2)),
            "test": (sp == "test") & inr(D(2022, 5, 3), D(2022, 5, 8))}


def return_masks(af) -> dict:
    day = af["day"].to_numpy()
    sp = af["split_user"].to_numpy()
    inr = lambda a, b: (day >= a) & (day <= b)          # noqa: E731
    return {"train": (sp == "train") & inr(D(2022, 4, 29), D(2022, 5, 4)),
            "val": (sp == "val") & inr(D(2022, 5, 1), D(2022, 5, 4)),
            "test": (sp == "test") & inr(D(2022, 5, 5), D(2022, 5, 8)),
            "test_time_only": (sp == "train") & inr(D(2022, 5, 5), D(2022, 5, 8))}


def fit_content(raw, cf, t0, cmd) -> list[str]:
    feats = K.CHOICE_FEATURES
    pop_idx = [feats.index(f) for f in K.POPULARITY_FEATURES]
    X = cf[feats].to_numpy(float)
    y = cf["is_click"].to_numpy(float)
    std = cf["is_rand"].to_numpy() == 0
    rnd = ~std
    m = content_masks(cf)
    Xs = {k: X[v] for k, v in m.items()}
    ys = {k: y[v] for k, v in m.items()}
    # observational fits (shared by both models)
    o_const = F.fit_constant(ys["obs_train"])
    o_pop = F.fit_logistic(Xs["obs_train"][:, pop_idx], ys["obs_train"])
    o_lr = F.fit_logistic(Xs["obs_train"], ys["obs_train"])
    o_gb = F.fit_gbdt(Xs["obs_train"], ys["obs_train"])
    # randomized-exposure fits
    r_const = F.fit_constant(ys["rnd_train"])
    r_pop = F.fit_logistic(Xs["rnd_train"][:, pop_idx], ys["rnd_train"])
    r_lr = F.fit_logistic(Xs["rnd_train"], ys["rnd_train"])
    r_gb = F.fit_gbdt(Xs["rnd_train"], ys["rnd_train"])
    off = F.fit_offset(o_gb, Xs["rnd_train"], ys["rnd_train"])
    o_gb_recal = P.TreeEnsemble(**{k: v for k, v in _te_kwargs(o_gb).items()}, offset=off)
    o_gb_recal._sk = o_gb._sk

    class Sub:          # a popularity-only logistic as a predictor on the full feature matrix
        def __init__(self, base, idx):
            self.base, self.idx = base, idx

        def predict(self, Z):
            return self.base.predict(np.asarray(Z)[:, self.idx])

    def cand(pred, complexity, val, test):
        pr = pred if not isinstance(pred, tuple) else Sub(*pred)
        c = {"complexity": complexity, "n_params": F.n_params(pred if not isinstance(pred, tuple) else pred[0]),
             "val": M.binary_report(ys[val], pr.predict(Xs[val])),
             "test": M.binary_report(ys[test], pr.predict(Xs[test])), "pred": pr}
        return c

    written = []
    # ---------------- content_consumption_feed (observational)
    cands = {
        "B0_constant": cand(o_const, "baseline: constant", "obs_val", "obs_test"),
        "B1_popularity": cand((o_pop, pop_idx), "baseline: popularity logistic (2 features)", "obs_val", "obs_test"),
        "O_logistic": cand(o_lr, "L2 logistic, 13 features", "obs_val", "obs_test"),
        "O_gbdt": cand(o_gb, f"GBDT, {o_gb.n_iter} trees x 31 leaves", "obs_val", "obs_test"),
    }
    sel = F.select(cands, "log_loss", ["B0_constant", "B1_popularity"],
                   complexity_order=["B0_constant", "B1_popularity", "O_logistic", "O_gbdt"])
    kept = sel["kept"]
    kp = cands[kept]["pred"]
    bp = cands[sel["best_baseline_on_test"]]["pred"]
    groups = cf.loc[m["obs_test"], "user_id"].to_numpy()
    ci_k, ci_d = _bootstrap(ys["obs_test"], kp.predict(Xs["obs_test"]), bp.predict(Xs["obs_test"]), groups)
    extra = {
        "test_time_only (train users, later days)": {k: M.binary_report(ys["obs_test_time_only"],
                                                                         c["pred"].predict(Xs["obs_test_time_only"]))
                                                     for k, c in cands.items()},
        "transport to random exposure (test users, 04-22..05-08)": {
            k: M.binary_report(ys["rnd_transport"], c["pred"].predict(Xs["rnd_transport"])) for k, c in cands.items()},
    }
    rep = _choice_report("content_consumption_feed", cf, m, ["obs_train", "obs_val", "obs_test", "obs_test_time_only",
                                                              "rnd_transport"], cands, sel, ci_k, ci_d, extra,
                         "observational",
                         "P(consume | the production recommender exposed the video in the featured feed, features)",
                         ("Observational: the exposures were chosen by Kuaishou's production recommender, which "
                          "selects on preferences the features do not capture. The model therefore predicts "
                          "engagement under that recommender and overstates the consume rate of exposures that are "
                          "not so selected (see 'transport to random exposure'). Use it as a learned-engagement "
                          "predictor (e.g. a recommender's scoring target), not as the individual's choice rule; the "
                          "choice rule is content_consumption, identified under random exposure."),
                         ys["obs_test"], kp.predict(Xs["obs_test"]), t0, cmd)
    written.append(F.finish("content_consumption_feed", VERSION, "content_selection", _pred_of(kp), feats,
                            "is_click: the individual kept watching (> ~7 s) a video the production recommender showed",
                            rep, {"prior_rate": cf.attrs["prior_rate"],
                                  "feature_quantiles": F.feature_quantiles(Xs["obs_train"], feats),
                                  "mechanisms": ["social.learned_engagement_score"]}))
    # ---------------- content_consumption (randomized exposure)
    cands_r = {
        "B0_constant": cand(r_const, "baseline: constant", "rnd_val", "rnd_test"),
        "B1_popularity": cand((r_pop, pop_idx), "baseline: popularity logistic (2 features)", "rnd_val", "rnd_test"),
        "O_logistic": cand(o_lr, "L2 logistic fitted on observational logs", "rnd_val", "rnd_test"),
        "O_gbdt": cand(o_gb, "GBDT fitted on observational logs", "rnd_val", "rnd_test"),
        "O_gbdt_recalibrated": cand(o_gb_recal, f"observational GBDT + intercept {off:+.3f} fitted on random "
                                                "exposure", "rnd_val", "rnd_test"),
        "R_logistic": cand(r_lr, "L2 logistic fitted on random exposure", "rnd_val", "rnd_test"),
        "R_gbdt": cand(r_gb, f"GBDT fitted on random exposure, {r_gb.n_iter} trees", "rnd_val", "rnd_test"),
    }
    sel_r = F.select(cands_r, "log_loss", ["B0_constant", "B1_popularity"],
                     complexity_order=["B0_constant", "B1_popularity", "R_logistic", "O_logistic",
                                       "O_gbdt_recalibrated", "R_gbdt", "O_gbdt"])
    kp = cands_r[sel_r["kept"]]["pred"]
    bp = cands_r[sel_r["best_baseline_on_test"]]["pred"]
    groups = cf.loc[m["rnd_test"], "user_id"].to_numpy()
    ci_k, ci_d = _bootstrap(ys["rnd_test"], kp.predict(Xs["rnd_test"]), bp.predict(Xs["rnd_test"]), groups)
    coef = {f: {"observational": float(o_lr.w[i]), "randomized": float(r_lr.w[i])} for i, f in enumerate(feats)}
    coef["_intercept"] = {"observational": o_lr.b, "randomized": r_lr.b}
    rep_rows = cf["item_played_before"].to_numpy() > 0
    repeat = {}
    for nm, mm in (("standard", std), ("random", rnd)):
        a = mm & rep_rows
        b = mm & ~rep_rows
        repeat[nm] = {"rows_repeat": int(a.sum()), "consume_rate_repeat": float(y[a].mean()) if a.any() else None,
                      "rows_first": int(b.sum()), "consume_rate_first": float(y[b].mean())}
    extra = {
        "observational vs randomized logistic coefficients (standardised features)": coef,
        "observational models on all random-exposure rows of test users (04-22..05-08)": {
            k: M.binary_report(ys["rnd_transport"], cands_r[k]["pred"].predict(Xs["rnd_transport"]))
            for k in ("O_logistic", "O_gbdt", "O_gbdt_recalibrated", "R_logistic", "R_gbdt")},
        "repeat consumption (item consumed before by the same individual)": repeat,
        "reliability table of the kept model on test": M.calibration_table(ys["rnd_test"], kp.predict(Xs["rnd_test"])),
    }
    rep = _choice_report("content_consumption", cf, m, ["rnd_train", "rnd_val", "rnd_test", "rnd_transport"], cands_r,
                         sel_r, ci_k, ci_d, extra, "randomized_exposure",
                         "P(an individual consumes (keeps watching) a video shown in the featured feed | exposure, "
                         "features), with exposure not selected on the individual's preferences",
                         ("Randomised exposure: the evaluated (and, for R_* candidates, fitted) rows are videos the "
                          "platform inserted into feeds uniformly at random from the pool, so whether an item was "
                          "shown is independent of the individual's unobserved taste. Conditional consume "
                          "probabilities are therefore not inflated by recommender selection. The feature effects "
                          "remain associations among randomly shown items (the features themselves - popularity, "
                          "affinity - were not randomised), so they are not causal effects of changing an item's "
                          "popularity or an individual's interests."),
                         ys["rnd_test"], kp.predict(Xs["rnd_test"]), t0, cmd)
    written.append(F.finish("content_consumption", VERSION, "content_selection", _pred_of(kp), feats,
                            "is_click: the individual kept watching (> ~7 s) a video shown in the feed",
                            rep, {"prior_rate": cf.attrs["prior_rate"],
                                  "feature_quantiles": F.feature_quantiles(Xs["rnd_train"], feats),
                                  "mechanisms": ["population.content_choice", "population.continue_watching"]}))
    return written


def _te_kwargs(te):
    return {"baseline": te.baseline, "roots": te.roots, "feature": te.feature, "threshold": te.threshold,
            "left": te.left, "right": te.right, "is_leaf": te.is_leaf, "value": te.value,
            "missing_left": te.missing_left}


def _pred_of(p):
    """The registered predictor: a popularity-only baseline is re-expressed on the full feature vector."""
    if hasattr(p, "base") and hasattr(p, "idx"):
        k = len(K.CHOICE_FEATURES)
        w = np.zeros(k)
        mean = np.zeros(k)
        scale = np.ones(k)
        for j, i in enumerate(p.idx):
            w[i], mean[i], scale[i] = p.base.w[j], p.base.mean[j], p.base.scale[j]
        return P.Logistic(w, p.base.b, mean, scale)
    return p


def _choice_report(name, cf, m, split_names, cands, sel, ci_k, ci_d, extra, ident, estimand, ident_note, yte, pte,
                   t0, cmd):
    defs = {"obs_train": "standard tab-1 rows, train users, 2022-04-12..05-02",
            "obs_val": "standard tab-1 rows, validation users, 2022-04-29..05-02",
            "obs_test": "standard tab-1 rows, test users (held out), 2022-05-03..05-08 (later time)",
            "obs_test_time_only": "standard tab-1 rows, train users, 2022-05-03..05-08",
            "rnd_train": "random-exposure tab-1 rows, train users, 2022-04-22..05-02",
            "rnd_val": "random-exposure tab-1 rows, validation users, 2022-04-29..05-02",
            "rnd_test": "random-exposure tab-1 rows, test users (held out), 2022-05-03..05-08 (later time)",
            "rnd_transport": "random-exposure tab-1 rows, test users, 2022-04-22..05-08"}
    return {
        "schema": "crucible.v5.calibration_report/1", "model": name, "version": VERSION, "domain": "content_selection",
        "datasets": ["kuairand_pure"], "identification": ident, "estimand": estimand,
        "population": KR_POP, "platform": KR_PLATFORM, "period": KR_PERIOD,
        "plan": F.plan_record(PLAN),
        "splits": {k: _split_info(cf, m[k], defs[k]) for k in split_names},
        "heldout_splits": [s for s in split_names if "test" in s or "transport" in s],
        "candidates": F.strip(cands), "selection": sel,
        "baseline_comparison": {"metric": "log_loss", "kept": sel["kept"], "baseline": sel["best_baseline_on_test"],
                                "kept_test": sel["test"][sel["kept"]],
                                "baseline_test": sel["test"][sel["best_baseline_on_test"]],
                                "difference": sel["test"][sel["kept"]] - sel["test"][sel["best_baseline_on_test"]],
                                "ci95": [ci_d["lo"], ci_d["hi"]], "kept_ci95": [ci_k["lo"], ci_k["hi"]],
                                "bootstrap": {"resamples": ci_d["n_boot"], "clusters": ci_d["clusters"]}},
        "calibration": {"ece": M.ece(yte, pte), "brier": M.brier(yte, pte), "mean_pred": float(np.mean(pte)),
                        "mean_obs": float(np.mean(yte)), "reliability": M.calibration_table(yte, pte)},
        "heldout_extra": extra,
        "missing_data": {"tab1_rows_dropped_without_duration": cf.attrs["dropped_no_duration"]},
        "identification_note": ident_note,
        "extrapolation_limits": CHOICE_LIMITS.format(pop=KR_POP, plat=KR_PLATFORM, per=KR_PERIOD),
        "feature_map": {f: K.FEATURE_MAP[f] for f in K.CHOICE_FEATURES},
        "fit": _fit_meta(t0, cmd),
    }


# ====================================================================== watch duration
WATCH_LIMITS = (
    "Evidence: KuaiRand-Pure ({pop}); {plat}; {per}. The model gives the distribution of log1p(watch seconds) of one "
    "feed exposure given whether it was consumed, the item's duration and the affinity/popularity inputs. The "
    "simulator may draw watch time for autoplay short-video exposures with durations of 6 s to 20 min. It may NOT "
    "infer: watch time for long-form video, live streams, articles or audio; durations outside the data range; "
    "effects of interventions on watch time (all associations are observational); other populations or periods. "
    "Quantiles outside the 2%-98% levels are extrapolated linearly over one quantile gap, so extreme replays "
    "(play ratios above about 4) are not reproduced; the skip branch (consumed = 0) is dominated by the platform's "
    "autoplay, so it describes how long an autoplaying video runs before the individual swipes away.")


def fit_watch(cf, t0, cmd) -> list[str]:
    plan = __import__("json").load(open(os.path.join(F.CRUCIBLE_ROOT, PLAN)))
    levels = np.asarray(plan["models"]["watch_duration"]["quantile_levels"], float)
    feats = K.WATCH_FEATURES
    X = cf[feats].to_numpy(float)
    y = np.log1p(cf["play_s"].to_numpy(float))
    m = watch_masks(cf)
    Xtr, ytr, Xva, yva, Xte, yte = X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]], X[m["test"]], y[m["test"]]
    ci = feats.index("consumed")
    di = feats.index("item_log_duration_s")
    b0 = F.fit_binned_quantiles(Xtr, ytr, di, levels, n_bins=1)
    b1 = F.fit_binned_quantiles(Xtr, ytr, di, levels, n_bins=10, group_col=ci)
    m1, h1 = F.fit_quantile_net(Xtr, ytr, Xva, yva, levels, hidden=())
    m2, h2 = F.fit_quantile_net(Xtr, ytr, Xva, yva, levels, hidden=(32, 32))
    te_dur = cf.loc[m["test"], "duration_s"].to_numpy()
    te_watch = cf.loc[m["test"], "play_s"].to_numpy()
    te_cons = Xte[:, ci]
    te_rand = cf.loc[m["test"], "is_rand"].to_numpy()
    dec = np.quantile(cf.loc[m["train"], "duration_s"], np.linspace(0, 1, 11)[1:-1])

    def dist_metrics(pred, Xs, ys, dur, watch, seed=0):
        Q = pred.predict(Xs)
        rng = np.random.default_rng(seed)
        ysim = pred.sample(Xs, rng.random(len(Xs)))
        wsim = np.expm1(np.maximum(ysim, 0))
        pit = M.pit_fast(ys, Q, levels)
        b = np.searchsorted(dec, dur)
        comp_err, view_err, per_bin = [], [], []
        thr = np.minimum(30.0, 0.5 * dur)
        for k in range(10):
            mk = b == k
            if mk.sum() < 50:
                continue
            co, cs = float(np.mean(watch[mk] >= dur[mk])), float(np.mean(wsim[mk] >= dur[mk]))
            vo, vs = float(np.mean(watch[mk] >= thr[mk])), float(np.mean(wsim[mk] >= thr[mk]))
            comp_err.append(abs(co - cs))
            view_err.append(abs(vo - vs))
            per_bin.append({"duration_decile": k, "rows": int(mk.sum()), "completion_obs": co, "completion_sim": cs,
                            "view_obs": vo, "view_sim": vs,
                            "watch_q50_obs": float(np.median(watch[mk])), "watch_q50_sim": float(np.median(wsim[mk])),
                            "watch_q90_obs": float(np.quantile(watch[mk], .9)),
                            "watch_q90_sim": float(np.quantile(wsim[mk], .9))})
        return {"n": int(len(ys)), "pinball": M.pinball(ys, Q, levels), "pit_ks": M.ks_uniform(pit),
                "coverage_80": M.coverage(ys, Q, levels, 0.1, 0.9), "coverage_90": M.coverage(ys, Q, levels, 0.05, 0.95),
                "wasserstein_watch_s": M.wasserstein1(watch, wsim),
                "completion_rate_abs_err": float(np.mean(comp_err)), "view_rate_abs_err": float(np.mean(view_err)),
                "pit_histogram": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(), "by_duration_decile": per_bin}

    def cand(pred, complexity, extra=None):
        v = dist_metrics(pred, Xva, yva, cf.loc[m["val"], "duration_s"].to_numpy(),
                         cf.loc[m["val"], "play_s"].to_numpy())
        t = dist_metrics(pred, Xte, yte, te_dur, te_watch)
        c = {"complexity": complexity, "n_params": F.n_params(pred), "pred": pred,
             "val": {k: v[k] for k in v if not isinstance(v[k], list)},
             "test": {k: t[k] for k in t if not isinstance(t[k], list)}, "test_detail": t}
        if extra:
            c["training"] = extra
        return c

    cands = {"B0_unconditional": cand(b0, "baseline: unconditional quantiles"),
             "B1_duration_bins": cand(b1, "baseline: quantiles per duration decile x consumed"),
             "M1_quantile_linear": cand(m1, "linear monotone quantiles", h1),
             "M2_quantile_mlp": cand(m2, "2 x 32 tanh monotone quantile network", h2)}
    sel = F.select({k: {"val": c["val"], "test": c["test"]} for k, c in cands.items()}, "pinball",
                   ["B0_unconditional", "B1_duration_bins"],
                   complexity_order=["B0_unconditional", "B1_duration_bins", "M1_quantile_linear", "M2_quantile_mlp"])
    kp = cands[sel["kept"]]["pred"]
    bp = cands[sel["best_baseline_on_test"]]["pred"]
    groups = cf.loc[m["test"], "user_id"].to_numpy()
    lk = M.pinball_rows(yte, kp.predict(Xte), levels)
    lb = M.pinball_rows(yte, bp.predict(Xte), levels)
    ci_k, ci_d = M.cluster_bootstrap(lk, groups), M.cluster_bootstrap(lk - lb, groups)
    subsets = {}
    for nm, mk in (("consumed", te_cons == 1), ("skipped", te_cons == 0), ("standard_log", te_rand == 0),
                   ("random_exposure_log", te_rand == 1)):
        d = dist_metrics(kp, Xte[mk], yte[mk], te_dur[mk], te_watch[mk])
        subsets[nm] = {k: d[k] for k in d if not isinstance(d[k], list)}
    kept_detail = cands[sel["kept"]]["test_detail"]
    rep = {
        "schema": "crucible.v5.calibration_report/1", "model": "watch_duration", "version": VERSION,
        "domain": "watch_duration", "datasets": ["kuairand_pure"], "identification": "observational",
        "estimand": "conditional distribution of y = log1p(play_time_s) of a feed exposure given consumed, duration "
                    "and affinity features; 21 quantile levels 0.02..0.98",
        "population": KR_POP, "platform": KR_PLATFORM, "period": KR_PERIOD, "plan": F.plan_record(PLAN),
        "splits": {"train": _split_info(cf, m["train"], "tab-1 rows of both logs, train users, standard 04-12..05-02 "
                                                        "and random 04-22..05-02"),
                   "val": _split_info(cf, m["val"], "tab-1 rows of both logs, validation users, 04-29..05-02"),
                   "test": _split_info(cf, m["test"], "tab-1 rows of both logs, test users (held out), 05-03..05-08")},
        "heldout_splits": ["test"],
        "candidates": {k: {kk: vv for kk, vv in c.items() if kk not in ("pred", "test_detail")}
                       for k, c in cands.items()},
        "selection": sel,
        "baseline_comparison": {"metric": "pinball", "kept": sel["kept"], "baseline": sel["best_baseline_on_test"],
                                "kept_test": sel["test"][sel["kept"]],
                                "baseline_test": sel["test"][sel["best_baseline_on_test"]],
                                "difference": sel["test"][sel["kept"]] - sel["test"][sel["best_baseline_on_test"]],
                                "ci95": [ci_d["lo"], ci_d["hi"]], "kept_ci95": [ci_k["lo"], ci_k["hi"]],
                                "bootstrap": {"resamples": ci_d["n_boot"], "clusters": ci_d["clusters"]}},
        "calibration": {"pit_ks": kept_detail["pit_ks"], "coverage_80": kept_detail["coverage_80"],
                        "coverage_90": kept_detail["coverage_90"], "pit_histogram": kept_detail["pit_histogram"]},
        "heldout_extra": {"kept model by duration decile (test)": kept_detail["by_duration_decile"],
                          "kept model by subset (test)": subsets},
        "identification_note": ("Observational association: watch time given features among exposures chosen by the "
                                "recommender (standard log) and random insertions (random log), pooled. It is not "
                                "the causal effect of changing an item's duration or an individual's affinity."),
        "extrapolation_limits": WATCH_LIMITS.format(pop=KR_POP, plat=KR_PLATFORM, per=KR_PERIOD),
        "feature_map": {f: K.FEATURE_MAP[f] for f in feats}, "fit": _fit_meta(t0, cmd),
    }
    rep["feature_map"]["target"] = "watch_s = expm1(sample); a view in the simulator needs watch_s >= min(30, 0.5 duration)"
    return [F.finish("watch_duration", VERSION, "watch_duration", kp, feats, "y = log1p(play_time_s)", rep,
                     {"quantile_levels": levels.tolist(), "prior_rate": cf.attrs["prior_rate"],
                      "feature_quantiles": F.feature_quantiles(Xtr, feats),
                      "mechanisms": ["population.watch_duration"],
                      "sampling": "FittedModel.sample(X, rng) draws y; watch_s = expm1(max(y, 0))"})]


# ====================================================================== daily return
RETURN_LIMITS = (
    "Evidence: KuaiRand-Pure ({pop}); activity = at least one logged exposure that day in the standard or "
    "random-exposure log (Kuaishou app, 2022-04-22..05-08). The simulator may use the model for the daily "
    "probability that an individual comes back to a feed platform given its recent activity pattern, and the "
    "fitted relation between recent activity and return (habit persistence). It may NOT infer: absolute activity "
    "levels of other platforms (an active day here requires a logged pool-video exposure, and the random-insertion "
    "rate rose during the period, so detected activity rises over time); long-horizon churn beyond 17 days; "
    "effects of notifications, content quality or social ties (not inputs); populations other than this selected "
    "sample of Kuaishou users. The training days include the 2022-04-30..05-04 Labour Day holiday in China, the "
    "test days do not.")


def fit_return(raw, t0, cmd) -> list[str]:
    af = K.activity_frame(raw, D(2022, 4, 29), D(2022, 5, 8))
    feats = K.RETURN_FEATURES
    X = af[feats].to_numpy(float)
    y = af["active"].to_numpy(float)
    m = return_masks(af)
    Xs = {k: X[v] for k, v in m.items()}
    ys = {k: y[v] for k, v in m.items()}
    a7 = feats.index("active_days_7d")

    class OwnRate:
        kind = "own_rate"

        def predict(self, Z):
            return (np.asarray(Z)[:, a7] + 1.0) / 9.0
    b0 = F.fit_constant(ys["train"])
    b1 = OwnRate()
    m1 = F.fit_logistic(Xs["train"], ys["train"])
    m2 = F.fit_gbdt(Xs["train"], ys["train"])

    def cand(pred, complexity, npar):
        return {"complexity": complexity, "n_params": npar, "pred": pred,
                "val": M.binary_report(ys["val"], pred.predict(Xs["val"])),
                "test": M.binary_report(ys["test"], pred.predict(Xs["test"]))}

    cands = {"B0_constant": cand(b0, "baseline: constant", 1),
             "B1_own_rate": cand(b1, "baseline rule: (active_days_7d + 1) / 9", 0),
             "M1_logistic": cand(m1, "L2 logistic, 6 features", F.n_params(m1)),
             "M2_gbdt": cand(m2, f"GBDT, {m2.n_iter} trees", F.n_params(m2))}
    sel = F.select(cands, "log_loss", ["B0_constant", "B1_own_rate"],
                   complexity_order=["B0_constant", "B1_own_rate", "M1_logistic", "M2_gbdt"])
    if sel["kept"] == "B1_own_rate":
        raise RuntimeError("the rule baseline was kept; it has no registered predictor kind")
    kp = cands[sel["kept"]]["pred"]
    bp = cands[sel["best_baseline_on_test"]]["pred"]
    groups = af.loc[m["test"], "user_id"].to_numpy()
    ci_k, ci_d = _bootstrap(ys["test"], kp.predict(Xs["test"]), bp.predict(Xs["test"]), groups)
    # forward simulation of the 4 test days
    users, rows, sess = K.activity_matrices(raw)
    test_users = np.array([u for u in users if K.split_of_bucket(np.array([K.user_bucket(u)]))[0] == "test"])
    d_first, d_last = D(2022, 5, 5), D(2022, 5, 8)
    obs_active = rows[test_users][:, d_first:d_last + 1] > 0
    sim = {}
    for nm, pred in (("kept", kp), ("B0_constant", b0), ("B1_own_rate", b1)):
        counts = np.zeros(5)
        ret = np.zeros(4)
        base_n = 0
        for draw in range(20):
            rng = np.random.default_rng(1000 + draw)
            A = rows[test_users][:, :d_first].copy() > 0
            R = rows[test_users][:, :d_first].copy()
            S = sess[test_users][:, :d_first].copy()
            A = np.concatenate([A, np.zeros((len(test_users), 4), bool)], 1)
            R = np.concatenate([R, np.zeros((len(test_users), 4))], 1)
            S = np.concatenate([S, np.zeros((len(test_users), 4))], 1)
            for d in range(d_first, d_last + 1):
                Xd = K.return_features_from(A, R, S, d)
                act = rng.random(len(test_users)) < pred.predict(Xd)
                A[:, d] = act
                # rows / sessions of a simulated active day: resample one of the individual's past active days
                for i in np.flatnonzero(act):
                    past = np.flatnonzero(A[i, :d] & (R[i, :d] > 0))
                    j = past[rng.integers(len(past))] if len(past) else None
                    R[i, d] = R[i, j] if j is not None else 1.0
                    S[i, d] = S[i, j] if j is not None else 1.0
            sa = A[:, d_first:d_last + 1]
            counts += np.bincount(sa.sum(1), minlength=5)
            was = rows[test_users][:, d_first - 1] > 0
            ret += sa[was].mean(0)
            base_n += 1
        sim[nm] = {"active_days_distribution": (counts / counts.sum()).tolist(), "retention_curve": (ret / base_n).tolist()}
    obs_counts = np.bincount(obs_active.sum(1), minlength=5)
    was = rows[test_users][:, d_first - 1] > 0
    obs = {"active_days_distribution": (obs_counts / obs_counts.sum()).tolist(),
           "retention_curve": obs_active[was].mean(0).tolist()}
    for nm in sim:
        sim[nm]["tv_distance_active_days"] = M.total_variation(sim[nm]["active_days_distribution"],
                                                               obs["active_days_distribution"])
        sim[nm]["retention_max_abs_err"] = float(np.max(np.abs(np.array(sim[nm]["retention_curve"]) -
                                                               np.array(obs["retention_curve"]))))
    rep = {
        "schema": "crucible.v5.calibration_report/1", "model": "return_daily", "version": VERSION,
        "domain": "browsing_frequency", "datasets": ["kuairand_pure"], "identification": "observational",
        "estimand": "P(an individual is active on day d | activity history before d)",
        "population": KR_POP, "platform": "Kuaishou app (feed activity from both logs)",
        "period": "2022-04-22 to 2022-05-08", "plan": F.plan_record(PLAN),
        "splits": {"train": _split_info(af, m["train"], "train users, days 04-29..05-04"),
                   "val": _split_info(af, m["val"], "validation users, days 05-01..05-04"),
                   "test": _split_info(af, m["test"], "test users (held out), days 05-05..05-08 (later time)"),
                   "test_time_only": _split_info(af, m["test_time_only"], "train users, days 05-05..05-08")},
        "heldout_splits": ["test", "test_time_only"],
        "candidates": F.strip(cands), "selection": sel,
        "baseline_comparison": {"metric": "log_loss", "kept": sel["kept"], "baseline": sel["best_baseline_on_test"],
                                "kept_test": sel["test"][sel["kept"]],
                                "baseline_test": sel["test"][sel["best_baseline_on_test"]],
                                "difference": sel["test"][sel["kept"]] - sel["test"][sel["best_baseline_on_test"]],
                                "ci95": [ci_d["lo"], ci_d["hi"]], "kept_ci95": [ci_k["lo"], ci_k["hi"]],
                                "bootstrap": {"resamples": ci_d["n_boot"], "clusters": ci_d["clusters"]}},
        "calibration": {"ece": M.ece(ys["test"], kp.predict(Xs["test"])), "brier": M.brier(ys["test"], kp.predict(Xs["test"])),
                        "reliability": M.calibration_table(ys["test"], kp.predict(Xs["test"]))},
        "heldout_extra": {
            "test_time_only": {k: M.binary_report(ys["test_time_only"], c["pred"].predict(Xs["test_time_only"]))
                               for k, c in cands.items()},
            "forward simulation of test users over 05-05..05-08 (20 draws)": {"observed": obs, "simulated": sim},
            "per-user heterogeneity: distribution of active days 04-22..05-04 (all users)":
                np.bincount((rows[:, D(2022, 4, 22):D(2022, 5, 4) + 1] > 0).sum(1)[users], minlength=14).tolist(),
        },
        "identification_note": ("Observational: return is predicted from the individual's own past activity "
                                "(persistence and heterogeneity). Nothing here identifies what would make an "
                                "individual return more often."),
        "extrapolation_limits": RETURN_LIMITS.format(pop=KR_POP),
        "feature_map": {f: K.FEATURE_MAP[f] for f in feats}, "fit": _fit_meta(t0, cmd),
    }
    return [F.finish("return_daily", VERSION, "browsing_frequency", kp, feats, "active: at least one logged "
                     "exposure on the day", rep,
                     {"feature_quantiles": F.feature_quantiles(Xs["train"], feats),
                      "mechanisms": ["population.daily_return"]})]


# ====================================================================== sessions per active day
SESSION_LIMITS = (
    "Evidence: KuaiRand-Pure ({pop}); sessions reconstructed from logged exposures with a 30-minute gap rule "
    "(2022-04-22..05-08). The simulator may draw each individual's propensity to start extra sessions on an "
    "active day from the fitted Gamma distribution (heterogeneity across individuals) and update it from observed "
    "behaviour with the posterior mean. It may NOT infer the absolute number of app sessions: only sessions that "
    "contain a logged pool-video exposure are seen, so true session counts are higher; nor session behaviour on "
    "other platforms, periods or populations. The gamma-Poisson form assumes a stable individual rate over two "
    "weeks.")


def _nb_fit(c, e):
    from scipy.optimize import minimize
    from scipy.special import gammaln
    c = np.asarray(c, float)
    e = np.asarray(e, float)

    def nll(th):
        k, r = np.exp(th)
        return -np.sum(gammaln(k + c) - gammaln(k) - gammaln(c + 1) + k * np.log(r / (r + e)) + c * np.log(e / (r + e)))
    mu = c.sum() / e.sum()
    if (c < 0).any() or (e <= 0).any():
        raise ValueError("counts must be >= 0 and exposures > 0")
    res = minimize(nll, x0=np.log([1.0, 1.0 / max(mu, 1e-3)]), method="L-BFGS-B", bounds=[(-9, 9), (-9, 9)])
    if not res.success:
        raise RuntimeError(f"negative-binomial fit did not converge: {res.message}")
    k, r = np.exp(res.x)
    return float(k), float(r), float(-res.fun)


def _nb_loglik(c, e, k, r):
    from scipy.special import gammaln
    return gammaln(k + c) - gammaln(k) - gammaln(c + 1) + k * np.log(r / (r + e)) + c * np.log(e / (r + e))


def _pois_loglik(c, mu):
    from scipy.special import gammaln
    return c * np.log(mu) - mu - gammaln(c + 1)


def session_days(raw) -> pd.DataFrame:
    g = K.sessions_by_day(raw)
    g["extra"] = g["sessions"] - 1
    g["bucket"] = g["user_id"].map({u: K.user_bucket(u) for u in g["user_id"].unique()})
    g["split_user"] = K.split_of_bucket(g["bucket"].to_numpy())
    return g


def session_window(g, a, b) -> pd.DataFrame:
    """Per user: c = extra sessions and e = active days within days a..b."""
    w = g[(g["day"] >= a) & (g["day"] <= b)]
    return w.groupby("user_id").agg(c=("extra", "sum"), e=("day", "size"), split=("split_user", "first"))


def session_test_arrays(g):
    """(history counts, history exposures, target counts, target exposures, users) of the prespecified test split."""
    hist = session_window(g, D(2022, 4, 22), D(2022, 5, 4))
    targ = session_window(g, D(2022, 5, 5), D(2022, 5, 8))
    users = targ.index[targ["split"] == "test"]
    return (hist.reindex(users)["c"].fillna(0).to_numpy(), hist.reindex(users)["e"].fillna(0).to_numpy(),
            targ.loc[users, "c"].to_numpy(float), targ.loc[users, "e"].to_numpy(float), users)


def fit_sessions(raw, t0, cmd) -> list[str]:
    g = session_days(raw)

    def window(a, b):
        return session_window(g, a, b)

    hist = window(D(2022, 4, 22), D(2022, 5, 4))
    targ = window(D(2022, 5, 5), D(2022, 5, 8))
    vhist = window(D(2022, 4, 22), D(2022, 4, 30))
    vtarg = window(D(2022, 5, 1), D(2022, 5, 4))

    def evaluate(params, H, T, split, glob):
        k, r = params
        users = T.index[T["split"] == split] if split else T.index
        Hc = H.reindex(users)["c"].fillna(0).to_numpy()
        He = H.reindex(users)["e"].fillna(0).to_numpy()
        Tc, Te = T.loc[users, "c"].to_numpy(float), T.loc[users, "e"].to_numpy(float)
        # B0: one pooled rate for everybody; B1: the individual's raw mean (posterior with a vanishing prior)
        preds = {"B0_global_mean": np.full(len(users), glob),
                 "B1_own_mean": (1e-9 + Hc) / (1e-9 + He),
                 "M1_gamma_poisson": (k + Hc) / (r + He)}
        out = {}
        for nm, rate in preds.items():
            mu = np.maximum(rate * Te, 1e-9)
            out[nm] = {"n": int(len(users)), "poisson_deviance": M.poisson_deviance(Tc, mu),
                       "mae": float(np.mean(np.abs(Tc / Te - rate)))}
        out["_arrays"] = (Tc, Te, preds, users)
        return out

    tr_users = hist.index[hist["split"] == "train"]
    vh = vhist[vhist.index.isin(tr_users)]
    kv, rv, _ = _nb_fit(vh["c"], vh["e"])
    val = evaluate((kv, rv), vh, vtarg[vtarg.index.isin(tr_users)], None, float(vh["c"].sum() / vh["e"].sum()))
    k, r, ll = _nb_fit(hist.loc[tr_users, "c"], hist.loc[tr_users, "e"])
    glob = float(hist.loc[tr_users, "c"].sum() / hist.loc[tr_users, "e"].sum())
    test = evaluate((k, r), hist, targ, "test", glob)
    Tc, Te, preds, users = test.pop("_arrays")
    val.pop("_arrays")
    # population-level predictive check on held-out users (no own history): NB vs homogeneous Poisson
    nb_ll = float(np.mean(_nb_loglik(Tc, Te, k, r)))
    po_ll = float(np.mean(_pois_loglik(Tc, glob * Te)))
    rng = np.random.default_rng(0)
    sims_nb = rng.poisson(rng.gamma(k, 1 / r, size=(20, len(Te))) * Te) / Te
    sims_po = rng.poisson(glob * np.tile(Te, (20, 1))) / Te
    obs_rate = Tc / Te
    ks_nb = M.ks_two_sample(obs_rate, sims_nb.ravel())
    ks_po = M.ks_two_sample(obs_rate, sims_po.ravel())
    for nm in test:
        test[nm]["heldout_loglik_per_user"] = nb_ll if nm == "M1_gamma_poisson" else po_ll
        test[nm]["ks_user_rate_distribution"] = ks_nb if nm == "M1_gamma_poisson" else ks_po
    for nm in val:
        val[nm]["heldout_loglik_per_user"] = None
    cands = {"B0_global_mean": {"complexity": "baseline: one rate for everybody", "n_params": 1},
             "B1_own_mean": {"complexity": "baseline: individual's raw mean", "n_params": 0},
             "M1_gamma_poisson": {"complexity": "gamma-Poisson heterogeneity (2 parameters)", "n_params": 2}}
    for nm in cands:
        cands[nm]["val"] = {kk: vv for kk, vv in val[nm].items() if vv is not None}
        cands[nm]["test"] = test[nm]
    sel = F.select(cands, "poisson_deviance", ["B0_global_mean", "B1_own_mean"],
                   complexity_order=["B0_global_mean", "B1_own_mean", "M1_gamma_poisson"])
    # the baselines are limits of the same predictor: shape, rate -> infinity (one global rate) or -> 0 (own mean)
    kept_pred = {"M1_gamma_poisson": P.GammaPoisson(k, r), "B0_global_mean": P.GammaPoisson(1e9, 1e9 / glob),
                 "B1_own_mean": P.GammaPoisson(1e-9, 1e-9)}[sel["kept"]]
    groups = np.asarray(users)
    dev_rows = lambda rate: 2 * (np.where(Tc > 0, Tc * np.log(np.where(Tc > 0, Tc, 1) / np.maximum(rate * Te, 1e-9)), 0)  # noqa: E731
                                 - (Tc - rate * Te))
    lk = dev_rows(preds[sel["kept"]])
    lb = dev_rows(preds[sel["best_baseline_on_test"]])
    ci_k, ci_d = M.cluster_bootstrap(lk, groups), M.cluster_bootstrap(lk - lb, groups)
    # calibration of count predictions: observed vs predicted mean by decile of the predicted rate
    pr = preds[sel["kept"]]
    edges = np.quantile(pr, np.linspace(0, 1, 11)[1:-1])
    b = np.searchsorted(edges, pr)
    table = []
    for q in range(10):
        mk = b == q
        if mk.any():
            table.append({"decile": q, "users": int(mk.sum()), "pred_rate": float(np.average(pr[mk], weights=Te[mk])),
                          "obs_rate": float(Tc[mk].sum() / Te[mk].sum())})
    cal_err = float(sum(t["users"] * abs(t["pred_rate"] - t["obs_rate"]) for t in table) / sum(t["users"] for t in table))
    rep = {
        "schema": "crucible.v5.calibration_report/1", "model": "session_rate", "version": VERSION,
        "domain": "browsing_frequency", "datasets": ["kuairand_pure"], "identification": "observational",
        "estimand": "individual rate of extra sessions (sessions - 1) per active day, heterogeneous across individuals",
        "population": KR_POP, "platform": "Kuaishou app (sessions from both logs, 30-minute gap)",
        "period": "2022-04-22 to 2022-05-08", "plan": F.plan_record(PLAN),
        "splits": {"train": {"definition": "train users, active days 04-22..05-04", "users": int(len(tr_users))},
                   "val": {"definition": "train users: history 04-22..04-30 predicts 05-01..05-04",
                           "users": int(val["M1_gamma_poisson"]["n"])},
                   "test": {"definition": "test users (held out): history 04-22..05-04 predicts 05-05..05-08 (later time)",
                            "users": int(len(users))}},
        "heldout_splits": ["test"],
        "candidates": cands, "selection": sel,
        "baseline_comparison": {"metric": "poisson_deviance", "kept": sel["kept"],
                                "baseline": sel["best_baseline_on_test"], "kept_test": sel["test"][sel["kept"]],
                                "baseline_test": sel["test"][sel["best_baseline_on_test"]],
                                "difference": sel["test"][sel["kept"]] - sel["test"][sel["best_baseline_on_test"]],
                                "ci95": [ci_d["lo"], ci_d["hi"]], "kept_ci95": [ci_k["lo"], ci_k["hi"]],
                                "bootstrap": {"resamples": ci_d["n_boot"], "clusters": ci_d["clusters"]}},
        "calibration": {"count_calibration_error": cal_err, "reliability_by_predicted_decile": table},
        "heldout_extra": {"fitted population distribution": {"gamma_shape": k, "gamma_rate": r,
                                                             "mean_rate": k / r, "cv_rate": 1 / np.sqrt(k),
                                                             "train_loglik": ll},
                          "held-out users without their own history": {
                              "nb_loglik_per_user": nb_ll, "homogeneous_poisson_loglik_per_user": po_ll,
                              "ks_rate_distribution_nb": ks_nb, "ks_rate_distribution_poisson": ks_po}},
        "identification_note": "Observational description of between-individual heterogeneity; no causal claim.",
        "extrapolation_limits": SESSION_LIMITS.format(pop=KR_POP),
        "feature_map": {"count": "extra sessions (sessions - 1) the individual started on its active days so far",
                        "exposure": "number of active days so far; predict() returns the posterior mean rate; "
                                    "draw_rates(n, rng) gives new individuals' rates"},
        "fit": _fit_meta(t0, cmd),
    }
    return [F.finish("session_rate", VERSION, "browsing_frequency", kept_pred, ["count", "exposure"],
                     "extra sessions per active day", rep,
                     {"mechanisms": ["population.session_frequency"]})]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="content,watch,return,session,news")
    a = ap.parse_args(argv)
    want = set(a.models.split(","))
    cmd = "python -m crucible.calibration.fit_e1 --models " + a.models
    F.plan_record(PLAN)                 # refuse to fit without a committed plan
    out = []
    if want & {"content", "watch", "return", "session"}:
        t0 = time.time()
        raw = K.load_raw()
        print(f"loaded KuaiRand-Pure in {time.time() - t0:.0f} s", flush=True)
        cf = K.choice_frame(raw) if want & {"content", "watch"} else None
        for key, fn in (("content", lambda t: fit_content(raw, cf, t, cmd)), ("watch", lambda t: fit_watch(cf, t, cmd)),
                        ("return", lambda t: fit_return(raw, t, cmd)), ("session", lambda t: fit_sessions(raw, t, cmd))):
            if key in want:
                t1 = time.time()
                out += fn(t1)
                print(f"{key}: {time.time() - t1:.0f} s", flush=True)
    if "news" in want:
        from . import mind
        t1 = time.time()
        out += mind.fit_news(PLAN, VERSION, cmd, t1)
        print(f"news: {time.time() - t1:.0f} s", flush=True)
    for d in out:
        print("wrote", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
