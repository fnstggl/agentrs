"""Fit and evaluate commercial_response@1 on the Criteo Uplift v2.1 randomized experiment (workstream E2).

Follows docs/v5/calibration/plans/E2_commercial_response.plan.json, committed before this script first ran:
seeded 60/20/20 user-level split, a constant-effect baseline and two T-learner complexities (logistic, gradient-boosted
trees), selection on validation by the pooled log loss of visit plus conversion, confirmation on test by paired
bootstrap, uplift ranking (Qini, AUUC), intervention-response estimates, and the response-type table validated on
test. Writes the checkpoint and card to docs/v5/calibration/models/commercial_response/ and the report to
docs/v5/calibration/reports/commercial_response@1.json.

    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "ws-E2: fit commercial" -- \
        python -m crucible.calibration.fit_commercial
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from . import criteo, portable
from . import scoring_e2 as S
from .commercial import OUTCOMES, ARMS, OUTPUTS, TYPE_FIELDS, assign_cells
from .fitmeta_e2 import code_state, environment, now, plan_ref, write_card, write_checkpoint, write_report, digest
from .gbdt_e2 import truncate_to_best
from .registry_e2 import load as registry_load, rel, REPORTS_DIR

NAME = "commercial_response"
VERSION = 1
MODEL_ID = f"{NAME}@{VERSION}"
PLAN = "E2_commercial_response.plan.json"
AMENDMENTS = ("E2_amendment_1.plan.json",)
CANDIDATES = ("constant_effect", "t_logistic", "t_gbdt", "t_gbdt_reg")
BASELINE = "constant_effect"
COMPLEXITY = {"constant_effect": 0, "t_logistic": 1, "t_gbdt": 2, "t_gbdt_reg": 2}
GBDT_PARAMS = {   # t_gbdt as prespecified; t_gbdt_reg added by amendment 1 (stronger leaf regularisation)
    "t_gbdt": {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 200, "l2_regularization": 1.0,
               "max_iter": 300, "n_iter_no_change": 20},
    "t_gbdt_reg": {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 1000, "l2_regularization": 10.0,
                   "max_iter": 500, "n_iter_no_change": 30},
}
N_BOOT = 200
N_P0_BINS = 8
N_TAU_BINS = 8


def fit_sub(cand: str, Xtr, ytr, Xva, yva, threads: int):
    """Fit one (arm, outcome) sub-model. Returns (portable arrays, reference predict fn or None, info)."""
    from threadpoolctl import threadpool_limits
    if cand == "constant_effect":
        p = float(ytr.mean())
        return portable.export_constant(p, "identity"), (lambda X, p=p: np.full(len(X), p)), {"rate": p}
    if cand == "t_logistic":
        from sklearn.linear_model import LogisticRegression
        mean = Xtr.mean(axis=0); scale = Xtr.std(axis=0); scale[scale == 0] = 1.0
        with threadpool_limits(threads):
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-7)
            lr.fit((Xtr - mean) / scale, ytr)
        ref = lambda X, lr=lr, m=mean, s=scale: lr.predict_proba((X - m) / s)[:, 1]
        return portable.export_linear(lr.coef_, lr.intercept_, mean, scale, "logit"), ref, {"n_iter": int(lr.n_iter_[0])}
    if cand in ("t_gbdt", "t_gbdt_reg"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = GBDT_PARAMS[cand]
        with threadpool_limits(threads):
            hgb = HistGradientBoostingClassifier(**params, early_stopping=True, scoring="loss", random_state=0)
            hgb.fit(Xtr, ytr, X_val=Xva, y_val=yva)
        es = truncate_to_best(hgb)              # amendment 1: keep the best validation iteration
        def ref(X, hgb=hgb):
            with threadpool_limits(threads):
                return hgb.predict_proba(X)[:, 1]
        return portable.export_hgb(hgb, "logit"), ref, {"n_iter": int(hgb.n_iter_), "early_stopping": es}
    raise ValueError(cand)


def evaluate(P: dict[str, np.ndarray], yv, yc, t) -> tuple[dict, np.ndarray]:
    """Metrics of one candidate on one split. P keys '<outcome>_<arm>'. Returns (metrics, per-row primary loss)."""
    t = t.astype(bool)
    out = {"per_arm": {}, "pooled": {}, "uplift": {}}
    rows = np.zeros(t.size)
    for outcome, y in (("visit", yv), ("conversion", yc)):
        for arm, m in (("control", ~t), ("treated", t)):
            out["per_arm"][f"{outcome}_{arm}"] = S.binary_report(y[m], P[f"{outcome}_{arm}"][m])
        own = np.where(t, P[f"{outcome}_treated"], P[f"{outcome}_control"])
        out["pooled"][outcome] = S.binary_report(y, own)
        rows += S.log_loss_rows(y, own)
        tau = P[f"{outcome}_treated"] - P[f"{outcome}_control"]
        q = S.qini_coefficient(y, t, tau)
        out["uplift"][outcome] = {**q, "auuc_normalised": S.auuc_normalised(y, t, tau),
                                  "model_implied_ate": float(tau.mean()),
                                  "by_predicted_decile": S.uplift_by_decile(y, t, tau),
                                  "heterogeneity": {"std_effect": float(tau.std()),
                                                    "share_negative": float(np.mean(tau < 0)),
                                                    "quantiles": dict(zip(["q01", "q05", "q25", "q50", "q75", "q95", "q99"],
                                                                          np.quantile(tau, [.01, .05, .25, .5, .75, .95, .99]).tolist()))}}
    out["primary"] = float(rows.mean())
    return out, rows


def build_types(p0, tau_v, Pk: dict[str, np.ndarray], yv, yc, t):
    """Response-type table on the validation split from the kept model's predictions."""
    t = t.astype(bool)
    q = np.linspace(0, 1, N_P0_BINS + 1)
    p0_edges = np.quantile(p0, q); p0_edges[0], p0_edges[-1] = -np.inf, np.inf
    b = np.searchsorted(p0_edges[1:-1], p0, side="right")
    tau_edges = np.empty((N_P0_BINS, N_TAU_BINS + 1))
    for i in range(N_P0_BINS):
        m = b == i
        e = np.quantile(tau_v[m], np.linspace(0, 1, N_TAU_BINS + 1)) if m.any() else np.zeros(N_TAU_BINS + 1)
        e[0], e[-1] = -np.inf, np.inf
        tau_edges[i] = e
    cell = assign_cells(p0, tau_v, p0_edges, tau_edges)
    K = N_P0_BINS * N_TAU_BINS
    n = np.bincount(cell, minlength=K).astype(np.float64)
    tab = {"weight": n / n.sum()}
    for outcome in OUTCOMES:
        for arm, key in (("control", "p0"), ("treated", "p1")):
            s = np.bincount(cell, weights=Pk[f"{outcome}_{arm}"], minlength=K)
            tab[f"{key}_{outcome}"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    obs = {}
    for outcome, y in (("visit", yv), ("conversion", yc)):
        for arm, m in (("control", ~t), ("treated", t)):
            nn = np.bincount(cell[m], minlength=K).astype(np.float64)
            yy = np.bincount(cell[m], weights=y[m].astype(np.float64), minlength=K)
            obs[f"{outcome}_{arm}_n"] = nn
            obs[f"{outcome}_{arm}_rate"] = np.where(nn > 0, yy / np.maximum(nn, 1), np.nan)
    return tab, obs, p0_edges, tau_edges, cell


def type_validation(tab, cell_test, yv, yc, t, seed: int = 7) -> dict:
    """Observed held-out uplift per type vs the type's predicted uplift; stratified bootstrap of the correlation.

    Also reports the weighted least-squares calibration slope of observed on predicted type uplift (1 = calibrated
    spread; < 1 = predicted heterogeneity too large)."""
    t = t.astype(bool)
    K = tab["weight"].size
    out = {}
    rng = np.random.default_rng(seed)
    for outcome, y in (("visit", yv), ("conversion", yc)):
        n1 = np.bincount(cell_test[t], minlength=K).astype(np.float64)
        n0 = np.bincount(cell_test[~t], minlength=K).astype(np.float64)
        y1 = np.bincount(cell_test[t], weights=y[t].astype(np.float64), minlength=K)
        y0 = np.bincount(cell_test[~t], weights=y[~t].astype(np.float64), minlength=K)
        ok = (n1 > 0) & (n0 > 0) & (tab["weight"] > 0)
        obs = np.where(ok, y1 / np.maximum(n1, 1) - y0 / np.maximum(n0, 1), np.nan)
        pred = tab[f"p1_{outcome}"] - tab[f"p0_{outcome}"]
        w = tab["weight"]
        corr = S.weighted_corr(pred[ok], obs[ok], w[ok])
        wmae = float(np.sum(w[ok] * np.abs(pred[ok] - obs[ok])) / np.sum(w[ok]))
        boots = []
        r1 = np.where(n1 > 0, y1 / np.maximum(n1, 1), 0.0); r0 = np.where(n0 > 0, y0 / np.maximum(n0, 1), 0.0)
        for _ in range(N_BOOT):
            b1 = rng.binomial(n1.astype(np.int64), r1) / np.maximum(n1, 1)
            b0 = rng.binomial(n0.astype(np.int64), r0) / np.maximum(n0, 1)
            boots.append(S.weighted_corr(pred[ok], (b1 - b0)[ok], w[ok]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        se_obs = np.sqrt(r1 * (1 - r1) / np.maximum(n1, 1) + r0 * (1 - r0) / np.maximum(n0, 1))
        wk = w[ok] / w[ok].sum(); px = pred[ok]; oy = obs[ok]
        mx = np.dot(wk, px); my = np.dot(wk, oy)
        vx = np.dot(wk, (px - mx) ** 2)
        slope = float(np.dot(wk, (px - mx) * (oy - my)) / vx) if vx > 0 else float("nan")
        out[outcome] = {"types_compared": int(ok.sum()), "weighted_corr": corr, "weighted_corr_ci95": [float(lo), float(hi)],
                        "weighted_mae": wmae, "calibration_slope": slope,
                        "weighted_mean_predicted": float(mx), "weighted_mean_observed": float(my),
                        "mean_abs_z": float(np.mean(np.abs((pred[ok] - obs[ok]) / np.maximum(se_obs[ok], 1e-12)))),
                        "per_type": [{"type": int(k), "weight": float(w[k]), "predicted_uplift": float(pred[k]),
                                      "observed_uplift": float(obs[k]), "se": float(se_obs[k]), "n_treated": int(n1[k]),
                                      "n_control": int(n0[k])} for k in np.flatnonzero(ok)]}
    return out


def simulated_experiment(model, yv, yc, t, cell_test, seed: int = 11) -> dict:
    """A randomized experiment simulated from the type table, compared with the observed test split."""
    rng = np.random.default_rng(seed)
    n = t.size
    types = model.sample_types(rng, n)
    treated = rng.random(n) < float(t.mean())
    sim = model.respond(rng, types, treated)
    out = {"n": int(n), "treatment_ratio": float(treated.mean())}
    tab = model.response_types()
    for outcome, y in (("visit", yv), ("conversion", yc)):
        obs = S.ate(y, t); sm = S.ate(sim[outcome], treated)
        tau_type = tab[f"p1_{outcome}"] - tab[f"p0_{outcome}"]
        # quintiles of the type effect, by type weight
        order = np.argsort(tau_type, kind="stable"); cw = np.cumsum(tab["weight"][order])
        quint = np.empty(tau_type.size, dtype=np.int64); quint[order] = np.minimum((cw * 5 - 1e-12).astype(np.int64), 4)
        rows = []
        for qi in range(5):
            ms = np.isin(types, np.flatnonzero(quint == qi)); mo = np.isin(cell_test, np.flatnonzero(quint == qi))
            if ms.any() and mo.any() and treated[ms].any() and (~treated[ms]).any() and t[mo].any() and (~t[mo].astype(bool)).any():
                rows.append({"quintile": qi, "simulated_uplift": S.ate(sim[outcome][ms], treated[ms])["ate"],
                             "observed_uplift": S.ate(y[mo], t[mo])["ate"]})
        out[outcome] = {"observed": obs, "simulated": sm,
                        "control_rate_z": float((sm["control_rate"] - obs["control_rate"]) /
                                                math.sqrt(obs["control_rate"] * (1 - obs["control_rate"]) * (1 / (~treated).sum() + 1 / (~t.astype(bool)).sum()))),
                        "ate_difference": float(sm["ate"] - obs["ate"]),
                        "ate_difference_se": float(math.sqrt(sm["se"] ** 2 + obs["se"] ** 2)),
                        "uplift_by_type_effect_quintile": rows}
    return out


def main(threads: int = 4, max_rows: int | None = None) -> dict:
    t_start = time.time()
    started_at = now()
    state = code_state()
    plan = plan_ref(PLAN)
    amendments = [plan_ref(a) for a in AMENDMENTS]
    data = criteo.load()
    lab_full = criteo.splits()
    X = np.asarray(data["X"], dtype=np.float32)
    T = np.asarray(data["treatment"]).astype(bool); YV = np.asarray(data["visit"]); YC = np.asarray(data["conversion"])
    E = np.asarray(data["exposure"])
    lab = lab_full
    if max_rows:                                    # smoke runs only (written outside the repository)
        keep = np.zeros(lab.size, bool); keep[np.random.default_rng(0).choice(lab.size, max_rows, replace=False)] = True
        lab = np.where(keep, lab, 255)
    idx = {s: np.flatnonzero(lab == i) for i, s in enumerate(criteo.SPLIT_NAMES)}
    split_info = {"method": "seeded PCG64 permutation", "seed": criteo.SPLIT_SEED, "digest": criteo.split_digest(lab_full),
                  "counts": {s: int(v.size) for s, v in idx.items()},
                  "treated_share": {s: float(T[v].mean()) for s, v in idx.items()}}
    print("splits", split_info["counts"], flush=True)
    Xs = {s: X[v].astype(np.float64) for s, v in idx.items()}
    ys = {s: {"visit": YV[v], "conversion": YC[v]} for s, v in idx.items()}
    ts = {s: T[v] for s, v in idx.items()}

    arrays, fit_info, preds = {}, {}, {s: {c: {} for c in CANDIDATES} for s in ("validation", "test")}
    refs = {}
    for cand in CANDIDATES:
        for outcome in OUTCOMES:
            for ai, arm in enumerate(ARMS):
                key = f"{outcome}_{arm}"
                mtr = ts["train"] == bool(ai); mva = ts["validation"] == bool(ai)
                t0 = time.time()
                arr, ref, info = fit_sub(cand, Xs["train"][mtr], ys["train"][outcome][mtr],
                                         Xs["validation"][mva], ys["validation"][outcome][mva], threads)
                for s in ("validation", "test"):
                    preds[s][cand][key] = ref(Xs[s])
                chk = Xs["validation"][:100_000]
                diff = portable.check_equal(arr, preds["validation"][cand][key][:100_000], chk)
                if diff > 1e-9:
                    raise RuntimeError(f"portable export of {cand}/{key} differs from the estimator by {diff}")
                arrays.update(portable.pack(f"{cand}/{key}", arr))
                refs[(cand, key)] = ref
                fit_info[f"{cand}/{key}"] = {**info, "n_train": int(mtr.sum()), "positives": int(ys["train"][outcome][mtr].sum()),
                                             "wall_s": round(time.time() - t0, 1), "portable_max_abs_diff": diff}
                print(cand, key, fit_info[f"{cand}/{key}"], flush=True)

    # ------------------------------------------------------------------ evaluation, selection, confirmation
    results, rows = {}, {}
    for cand in CANDIDATES:
        results[cand] = {}
        for s in ("validation", "test"):
            m, r = evaluate(preds[s][cand], ys[s]["visit"], ys[s]["conversion"], ts[s])
            results[cand][s] = m
            rows[(cand, s)] = r
        print(cand, "val primary", results[cand]["validation"]["primary"], "test primary", results[cand]["test"]["primary"], flush=True)
    kept = min(CANDIDATES, key=lambda c: (results[c]["validation"]["primary"], COMPLEXITY[c]))
    comparisons = []
    for other in CANDIDATES:
        if other == kept:
            continue
        bs = S.paired_bootstrap_mean_diff(rows[(kept, "test")] - rows[(other, "test")], n_boot=N_BOOT, seed=1)
        comparisons.append({"kept": kept, "other": other, "metric": "pooled log loss visit + conversion (test)",
                            "kept_minus_other": bs, "kept_significantly_worse": bool(bs["ci95"][0] > 0),
                            "validation_kept_minus_other": results[kept]["validation"]["primary"] - results[other]["validation"]["primary"]})
    worse = sum(c["kept_significantly_worse"] for c in comparisons)

    # ------------------------------------------------------------------ uplift uncertainty, ATE, exposure
    test_t = ts["test"]
    Pk_test = preds["test"][kept]
    qini_ci = {o: S.qini_bootstrap(ys["test"][o], test_t, Pk_test[f"{o}_treated"] - Pk_test[f"{o}_control"], n_boot=N_BOOT, seed=3)
               for o in OUTCOMES}
    e_test = E[idx["test"]]
    pe = float(e_test[test_t].mean()); var_pe = pe * (1 - pe) / test_t.sum()
    intervention = {}
    for o in OUTCOMES:
        a = S.ate(ys["test"][o], test_t)
        cace = a["ate"] / pe
        se_c = math.sqrt(a["se"] ** 2 / pe ** 2 + a["ate"] ** 2 * var_pe / pe ** 4)
        intervention[o] = {"itt_difference_in_means": a, "model_implied_ate": results[kept]["test"]["uplift"][o]["model_implied_ate"],
                           "exposure_complier_effect_wald": {"estimate": cace, "se": se_c, "ci95": [cace - 1.96 * se_c, cace + 1.96 * se_c],
                                                             "exposure_rate_treated": pe,
                                                             "assumption": "exclusion restriction: eligibility affects outcomes only through exposure"}}

    # ------------------------------------------------------------------ response types (validation), checkpoint, card
    Pk_val = preds["validation"][kept]
    p0v = Pk_val["visit_control"]; tauv = Pk_val["visit_treated"] - p0v
    tab, obs_val, p0_edges, tau_edges, _ = build_types(p0v, tauv, Pk_val, ys["validation"]["visit"], ys["validation"]["conversion"], ts["validation"])
    for k in TYPE_FIELDS:
        arrays[f"types/{k}"] = np.nan_to_num(tab[k], nan=0.0)
    arrays["types/p0_edges"] = p0_edges; arrays["types/tau_edges"] = tau_edges
    for k, v in obs_val.items():
        arrays[f"types/validation_{k}"] = np.nan_to_num(v, nan=0.0)
    audit_idx = (criteo.audit_indices(lab_full) if not max_rows else
                 np.sort(np.random.default_rng(1).choice(idx["test"], 20000, replace=False)))
    pos_in_test = np.searchsorted(idx["test"], audit_idx)
    probe = pos_in_test[:256]
    arrays["probe/X"] = Xs["test"][probe]
    arrays["probe/pred"] = np.column_stack([preds["test"][kept][f"{o}_{a}"][probe] for o in OUTCOMES for a in ARMS])
    # tree candidates other than the kept one are not shipped (checkpoints stay < 5 MB); their metrics are in the report
    stored = [c for c in CANDIDATES if c == kept or not c.startswith("t_gbdt")]
    ckpt = write_checkpoint(NAME, {k: v for k, v in arrays.items()
                                   if k.split("/", 1)[0] not in CANDIDATES or k.split("/", 1)[0] in stored})

    def summary(m):
        return {"primary_pooled_logloss_visit_plus_conversion": m["primary"],
                **{f"{o}_pooled_{k}": m["pooled"][o][k] for o in OUTCOMES for k in ("log_loss", "brier", "ece", "auc")},
                **{f"{key}_{k}": m["per_arm"][key][k] for key in m["per_arm"] for k in ("log_loss", "brier", "ece", "auc")},
                **{f"{o}_qini_coefficient": m["uplift"][o]["qini_coefficient"] for o in OUTCOMES},
                **{f"{o}_auuc_normalised": m["uplift"][o]["auuc_normalised"] for o in OUTCOMES}}

    limits_list = [
        "Absolute rates and effect sizes are not representative: Criteo sub-sampled the data non-uniformly so that the true incrementality level cannot be deduced. Use relative structure (ranking, heterogeneity, the correlation of baseline propensity and effect) and re-base with transport() to the world's own rates.",
        "The data pool several incrementality tests of undisclosed advertisers, countries and dates; the population is users reachable by Criteo's display advertising, not a general population.",
        "Features f0..f11 are anonymised random projections with no simulator counterpart; the simulator may use only the response-type table, not the feature-level model.",
        "The effect is the intention-to-treat effect of eligibility for one advertiser's display campaign over one test period (exposure rate among treated users only %.1f%%). It does not identify the response to arbitrary ads, prices, content or repeated campaigns, and it says nothing about long-run, carry-over or saturation effects." % (100 * pe),
        "No timestamps: no time-based holdout, no dynamics, no seasonality; the holdout is user-level (rows are users) only. Near-duplicate feature vectors occur across rows, so held-out users can resemble training users closely.",
        "transport() assumes each type's treatment effect is a constant odds ratio when the baseline rate changes; this is an assumption, not an estimate.",
        "Visit and conversion are binary per user over the test period; frequency, basket size and revenue are not observed.",
    ]
    card = {
        "name": NAME, "version": VERSION, "id": MODEL_ID, "workstream": "E2", "domain": "commercial_intervention_response",
        "loader": "crucible.calibration.commercial:CommercialResponseModel",
        "features": list(criteo.FEATURES),
        "feature_docs": {f: "Criteo anonymised feature (random projection, float)" for f in criteo.FEATURES},
        "target": "visit and conversion under control and treatment (per-arm probabilities)",
        "outputs": OUTPUTS,
        "identification": {"kind": "randomized_experiment",
                           "statement": "Treatment was randomly assigned (ratio 0.85), so the per-arm probabilities and their difference given pre-treatment features are experimentally identified intention-to-treat effects. Exposure is not randomized; the complier (exposure) effect in the report is a Wald estimate under the exclusion restriction."},
        "dataset": {"name": criteo.DATASET, "synthetic": False, "url": criteo.URL, "sha256": criteo.SHA256,
                    "manifest": "docs/v5/calibration/datasets/criteo_uplift_v2.1.json", "license": criteo.LICENSE["spdx"]},
        "checkpoint": ckpt,
        "report": rel(os.path.join(REPORTS_DIR, f"{MODEL_ID}.json")),
        "plan": plan, "plan_amendments": amendments, "stored_candidates": stored,
        "fit": {"started_at": started_at, "code": "src/crucible/calibration/fit_commercial.py", **state,
                "command": "python -m crucible.calibration.fit_commercial", "threads": threads, "environment": environment(),
                "split": split_info, "sub_models": fit_info},
        "metrics": summary(results[kept]["test"]),
        "baseline_metrics": summary(results[BASELINE]["test"]),
        "candidates": {c: {"complexity": COMPLEXITY[c], "validation": summary(results[c]["validation"]),
                           "test": summary(results[c]["test"])} for c in CANDIDATES},
        "selection": {"rule": "lowest validation pooled log loss (visit + conversion); confirmed on test by paired bootstrap",
                      "kept": kept, "baseline": BASELINE, "comparisons": comparisons, "worse_model_kept": int(worse)},
        "calibration_metrics": ["log_loss", "brier", "ece_15_equal_mass_bins", "auc", "qini_coefficient", "auuc_normalised"],
        "intervention_response": intervention,
        "limits": " ".join(limits_list),
        "extrapolation_limits": limits_list,
        "simulator_mapping": {
            "use": "response types, not features: CommercialResponseModel.sample_types(rng, n) once per simulated individual (store the type index in a population column); when the individual is targeted by an advertising intervention during a campaign period it responds with respond(rng, types, treated=True), otherwise treated=False.",
            "visit": "a visit to the advertiser's product or storefront during the campaign period (one Bernoulli draw per individual per campaign period)",
            "conversion": "a purchase from that advertiser during the period; only after a visit",
            "treated": "the individual is eligible to be targeted by the advertiser's campaign (intention to treat); exposure (an ad actually shown) is a subset",
            "transport": "transport(control_visit_rate, control_conversion_rate) re-bases the table to the world's own baseline rates keeping each type's odds ratio",
            "features": "f0..f11 have no simulator counterpart; predict(X) exists only to evaluate and audit the model on Criteo rows",
        },
        "fidelity_mechanisms": ["advertising_intervention_response", "advertising_response_heterogeneity"],
        "provenance": {"dataset": criteo.DATASET, "dataset_sha256": criteo.SHA256, "source": criteo.URL,
                       "license": criteo.LICENSE, "fit_commit": state["commit"], "plan_commit": plan["commit"],
                       "citation": "Diemert, Betlei, Renaudin, Amini. A Large Scale Benchmark for Uplift Modeling. AdKDD 2018."},
        "response_types": {"K": int(np.sum(tab["weight"] > 0)), "construction": "validation split, 8 x 8 equal-mass bins of predicted control visit probability and predicted visit effect",
                           "usable": None},
    }
    write_card(NAME, card)

    # ------------------------------------------------------------------ round trip through the registry: audit + types validation
    model = registry_load(MODEL_ID)
    Xa = X[audit_idx].astype(np.float64)
    audit = {"indices": "criteo.audit_indices(splits()) (seeded random 100k test rows)", "n": int(audit_idx.size),
             "indices_digest": digest(audit_idx.astype(np.int64)), "candidates": {}}
    ta = T[audit_idx]; yva = YV[audit_idx]; yca = YC[audit_idx]
    for cand in stored:
        Pa = model.candidate_predict(cand, Xa)
        P = {f"{o}_{a}": Pa[:, i] for i, (o, a) in enumerate([(o, a) for o in OUTCOMES for a in ARMS])}
        ref = np.column_stack([preds["test"][cand][f"{o}_{a}"][pos_in_test] for o in OUTCOMES for a in ARMS])
        m, _ = evaluate(P, yva, yca, ta)
        audit["candidates"][cand] = {"primary": m["primary"],
                                     **{f"{o}_pooled_log_loss": m["pooled"][o]["log_loss"] for o in OUTCOMES},
                                     **{f"{o}_pooled_brier": m["pooled"][o]["brier"] for o in OUTCOMES},
                                     **{f"{o}_pooled_ece": m["pooled"][o]["ece"] for o in OUTCOMES},
                                     **{f"{o}_qini_coefficient": m["uplift"][o]["qini_coefficient"] for o in OUTCOMES},
                                     "max_abs_diff_vs_estimator": float(np.max(np.abs(Pa - ref)))}
    # test cells from the estimator's predictions (equal to the shipped portable model: see the audit's max_abs_diff)
    Pt0 = preds["test"][kept]["visit_control"]; Pt1 = preds["test"][kept]["visit_treated"]
    cell_test = assign_cells(Pt0, Pt1 - Pt0, p0_edges, tau_edges)
    tv = type_validation(model.response_types(), cell_test, ys["test"]["visit"], ys["test"]["conversion"], test_t)
    tv_honest = type_validation(model.response_types("honest"), cell_test, ys["test"]["visit"], ys["test"]["conversion"], test_t)
    usable = bool(tv["visit"]["weighted_corr_ci95"][0] > 0)
    sim = simulated_experiment(model, ys["test"]["visit"], ys["test"]["conversion"], test_t, cell_test)
    card["response_types"].update({"usable": usable, "acceptance_rule": "lower bound of the 95% bootstrap interval of the weighted type-level visit-uplift correlation on test > 0",
                                   "test_visit_weighted_corr": tv["visit"]["weighted_corr"],
                                   "test_visit_weighted_corr_ci95": tv["visit"]["weighted_corr_ci95"],
                                   "test_visit_calibration_slope": tv["visit"]["calibration_slope"],
                                   "test_conversion_weighted_corr": tv["conversion"]["weighted_corr"],
                                   "honest_estimate_test_visit_weighted_corr": tv_honest["visit"]["weighted_corr"],
                                   "honest_estimate_test_visit_calibration_slope": tv_honest["visit"]["calibration_slope"],
                                   "honest_estimate_test_visit_weighted_mae": tv_honest["visit"]["weighted_mae"],
                                   "model_estimate_test_visit_weighted_mae": tv["visit"]["weighted_mae"],
                                   "default_estimate": "model (prespecified); response_types('honest') is reported as a secondary analysis"})
    write_card(NAME, card)

    rep = {
        "schema": "crucible.v5.calibration.report/1", "model": MODEL_ID, "plan": plan, "plan_amendments": amendments,
        "dataset": card["dataset"],
        "identification": card["identification"], "split": split_info,
        "heldout": {"validation": {c: results[c]["validation"] for c in CANDIDATES}, "test": {c: results[c]["test"] for c in CANDIDATES}},
        "selection": card["selection"],
        "calibration_metrics": {c: {"test": {
            **{f"{o}_{k}": results[c]["test"]["pooled"][o][k] for o in OUTCOMES for k in ("log_loss", "brier", "ece")},
            **{f"{key}_ece": results[c]["test"]["per_arm"][key]["ece"] for key in results[c]["test"]["per_arm"]},
            **{f"{o}_reliability": S.reliability(ys["test"][o], np.where(ts["test"], preds["test"][c][f"{o}_treated"],
                                                                        preds["test"][c][f"{o}_control"])) for o in OUTCOMES}}}
            for c in CANDIDATES},
        "uplift_uncertainty": {"kept": kept, "qini_bootstrap_test": qini_ci},
        "intervention_response": intervention,
        "response_types": {"table": {k: arrays[f"types/{k}"] for k in TYPE_FIELDS},
                           "validation_observed": {k: arrays[f"types/validation_{k}"] for k in obs_val},
                           "test_validation": tv, "test_validation_honest_estimate": tv_honest,
                           "usable": usable, "simulated_experiment": sim},
        "audit": audit, "fit_info": fit_info, "extrapolation_limits": limits_list,
        "wall_s": round(time.time() - t_start, 1), "finished_at": now(),
    }
    write_report(MODEL_ID, rep)
    print("kept", kept, "worse_model_kept", worse, "types usable", usable, "wall", rep["wall_s"], flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke run on a subsample (set CRUCIBLE_E2_CAL_DIR)")
    a = ap.parse_args()
    if a.max_rows and not os.environ.get("CRUCIBLE_E2_CAL_DIR"):
        raise SystemExit("--max-rows writes a smoke model; set CRUCIBLE_E2_CAL_DIR to a scratch directory")
    main(a.threads, a.max_rows)
