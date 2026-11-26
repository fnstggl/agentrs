"""E5 fit: the advertising-response redesign (audit E skeptic S5) under the committed plan
`docs/v5/calibration/plans/E5_ad_exposure_plan.json` (workstream E). Offline fitting code only.

    PYTHONPATH=src python -m crucible.calibration.fit_e5            # ~10 min on one CPU

What is estimated: the effect of being *displayed* an advertiser's campaign on the log-odds of visiting the
advertiser, as a function of the displayed individual's organic (no-ad) visit log-odds. Criteo's randomised
eligibility (treatment) is the instrument for display (exposure); with one-sided non-compliance the complier
(displayed) baseline of each stratum is identified from the control arm and the eligible-but-never-displayed users,
and the displayed users' visit rate identifies the complier visit rate under display. Strata are bins of a
pre-treatment score (commercial_response@1's control-visit sub-model, fitted on the train split), so they are
balanced between the arms.

Why this replaces the response-type table of `commercial_response@1` as the in-world mechanism: the type table gave
every simulated individual a latent type drawn independently of its own state, an effect independent of whether it
saw any ad, and an 18% mass of negative responders that the held-out data do not support (audit E skeptic s3). Here
the effect is attached to a display of the campaign, scales with the individual's own organic propensity (world
state), is never negative, and composes with an attention model per display (relevance, repetition, familiarity) in
the runtime decision model `ad_response@2`.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

from . import criteo
from . import fit_e4 as E4
from . import fitlib as F
from . import metrics as M
from . import portable
from . import predictors as P
from . import predictors_e5 as P5
from .registry import CRUCIBLE_ROOT, load

PLAN_PATH = "docs/v5/calibration/plans/E5_ad_exposure_plan.json"
NAME = "ad_exposure_response"
VERSION = 1
N_STRATA = 10
MIN_EXPOSED = 300
CLIP = (1e-4, 0.999)
N_BOOT = 200
CANDS = ("B0_no_effect", "B1_constant_difference", "M1_constant_log_odds", "M2_linear_log_odds")
BASELINES = ["B0_no_effect", "B1_constant_difference"]
PRIMARY = "excess_log_loss"


def spec() -> dict:
    with open(os.path.join(CRUCIBLE_ROOT, PLAN_PATH), encoding="utf-8") as fh:
        return json.load(fh)["models"][NAME]


# ====================================================================== data
def split_rows(lab: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Row indices of the plan's splits (explore, fit, val, test); `lab` is criteo.splits() (or a synthetic label
    array of the same coding in unit tests)."""
    lab = criteo.splits() if lab is None else lab
    val = np.flatnonzero(lab == 1)
    r = np.arange(val.size) % 10
    return {"explore": val[r == 0], "fit": val[np.isin(r, (1, 3, 5, 7, 9))], "val": val[np.isin(r, (2, 4, 6, 8))],
            "test": np.flatnonzero(lab == 2)}


def score(model, d: dict, rows: np.ndarray, sub: str = "visit_control", chunk: int = 500_000) -> np.ndarray:
    """p0_hat (or another sub-model of the kept T-learner) for the given rows, in chunks (memory)."""
    out = np.empty(rows.size, np.float64)
    pm = model.submodel(f"{model.kept}/{sub}")
    for i in range(0, rows.size, chunk):
        X = np.asarray(d["X"][rows[i:i + chunk]], dtype=np.float32)
        out[i:i + chunk] = portable.predict(pm, X)
    return out


def frame(d: dict, rows: np.ndarray, s: np.ndarray) -> dict:
    return {"z": np.asarray(d["treatment"][rows]).astype(bool), "D": np.asarray(d["exposure"][rows]).astype(bool),
            "y": np.asarray(d["visit"][rows]).astype(np.float64),
            "conv": np.asarray(d["conversion"][rows]).astype(np.float64), "s": s.astype(np.int64)}


# ====================================================================== strata and nuisances
def cells(fr: dict, K: int, outcome: str = "y") -> dict[str, np.ndarray]:
    """Per stratum counts: control rows / successes, treated exposed, treated non-exposed."""
    s, z, D, y = fr["s"], fr["z"], fr["D"], fr[outcome]

    def cnt(mask):
        return (np.bincount(s[mask], minlength=K).astype(float),
                np.bincount(s[mask], weights=y[mask], minlength=K).astype(float))
    nc, yc = cnt(~z)
    nte, yte = cnt(z & D)
    ntn, ytn = cnt(z & ~D)
    return {"nc": nc, "yc": yc, "nte": nte, "yte": yte, "ntn": ntn, "ytn": ytn}


def nuisance(c: dict, clip: bool = True) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = c["yc"] / c["nc"]
        nt = c["nte"] + c["ntn"]
        e = c["nte"] / nt
        nu = c["ytn"] / c["ntn"]
        t1 = c["yte"] / c["nte"]
        mu0_raw = (cr - (1 - e) * nu) / e
        itt = (c["yte"] + c["ytn"]) / nt - cr
    mu0 = np.clip(np.nan_to_num(mu0_raw, nan=CLIP[0]), *CLIP) if clip else mu0_raw
    return {"control": cr, "exposure": e, "never_taker": nu, "exposed": t1, "mu0_raw": mu0_raw, "mu0": mu0,
            "itt": itt, "late": itt / e, "rows": c["nc"] + nt, "exposed_rows": c["nte"]}


def strata_edges(p0_fit: np.ndarray, fr_fit_raw: dict) -> np.ndarray:
    """10 equal-mass bins of p0_hat on the fit split, merged by the plan's rule (inner edges returned)."""
    inner = list(np.quantile(p0_fit, np.linspace(0, 1, N_STRATA + 1))[1:-1])

    def assign(edges):
        return np.searchsorted(np.asarray(edges), p0_fit, side="right")

    def bad(edges):
        K = len(edges) + 1
        fr = dict(fr_fit_raw, s=assign(edges))
        nz = nuisance(cells(fr, K), clip=False)
        return [(nz["exposed_rows"][k] < MIN_EXPOSED) or not (0 < nz["mu0_raw"][k] < 1) for k in range(K)]

    k = 0
    while True:
        b = bad(inner)
        K = len(inner) + 1
        if k >= K - 1:
            break
        if b[k]:
            del inner[k]                 # merge stratum k into its upper neighbour
            continue
        k += 1
    while len(inner) and bad(inner)[-1]:
        del inner[-1]                    # merge the top stratum downwards
    return np.asarray(inner, dtype=np.float64)


# ====================================================================== candidates: joint three-arm likelihood
# Per stratum s the data are three binomial cells: control (n_c, y_c), eligible displayed (n_te, y_te) and eligible
# never displayed (n_tn, y_tn). With e_s = n_te / (n_te + n_tn) the nuisances are the never-taker rate nu_s and the
# complier baseline mu0_s: control visits ~ Bin(n_c, (1 - e_s) nu_s + e_s mu0_s), never-takers ~ Bin(n_tn, nu_s),
# displayed ~ Bin(n_te, q(mu0_s; theta)). theta (the effect structure) is shared across strata and fitted by maximum
# likelihood with the nuisances profiled out; held-out splits are scored by the profile likelihood with theta fixed
# (nuisances re-profiled on the held-out split), as excess log loss per displayed row over the saturated fit.

def _ll(y, n, p) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (n - y) * np.log1p(-p)))


def q_of(name: str, theta, v) -> np.ndarray:
    """Visit probability of a displayed complier with baseline log-odds v (candidate `name`, parameters theta)."""
    v = np.asarray(v, dtype=np.float64)
    p0 = P._sigmoid(v)
    if name == "B0_no_effect":
        return p0
    if name == "B1_constant_difference":
        return np.clip(p0 + max(float(theta[0]), 0.0), 1e-6, 1 - 1e-6)
    if name == "M1_constant_log_odds":
        return P._sigmoid(v + max(float(theta[0]), 0.0))
    if name == "M2_linear_log_odds":
        return P._sigmoid(v + np.maximum(float(theta[0]) + float(theta[1]) * (v - float(theta[2])), 0.0))
    raise ValueError(name)


def _stratum_profile(name, theta, ck: dict, start=None) -> tuple[float, tuple]:
    from scipy.optimize import minimize
    n_te, n_tn = ck["nte"], ck["ntn"]
    e = n_te / max(n_te + n_tn, 1.0)

    def nll(x):
        u, v = x
        pi = (1 - e) * P._sigmoid(u) + e * P._sigmoid(v)
        return -(_ll(ck["yc"], ck["nc"], pi) + _ll(ck["ytn"], n_tn, P._sigmoid(u)) +
                 _ll(ck["yte"], n_te, q_of(name, theta, v)))
    if start is None:
        nu = np.clip((ck["ytn"] + 0.5) / (n_tn + 1.0), 1e-5, 1 - 1e-5)
        cr = (ck["yc"] + 0.5) / (ck["nc"] + 1.0)
        mu = np.clip((cr - (1 - e) * nu) / max(e, 1e-6), 1e-4, 0.999)
        start = (float(P5.logit(nu)), float(P5.logit(mu)))
    best = None
    for x0 in (start, (start[0], start[1] - 1.0), (start[0], start[1] + 1.0)):
        r = minimize(nll, np.asarray(x0, float), method="L-BFGS-B", bounds=[(-14, 14), (-14, 14)])
        if best is None or r.fun < best.fun:
            best = r
    return -float(best.fun), (float(best.x[0]), float(best.x[1]))


def _strata(c: dict) -> list[dict]:
    K = c["nc"].size
    return [{k: float(c[k][j]) for k in c} for j in range(K)]


def profile_ll(name, theta, c: dict) -> tuple[float, list]:
    out, nuis = 0.0, []
    for ck in _strata(c):
        ll, x = _stratum_profile(name, theta, ck)
        out += ll
        nuis.append(x)
    return out, nuis


def saturated_ll(c: dict) -> float:
    return sum(_ll(ck["yc"], ck["nc"], ck["yc"] / max(ck["nc"], 1)) + _ll(ck["ytn"], ck["ntn"], ck["ytn"] / max(ck["ntn"], 1))
               + _ll(ck["yte"], ck["nte"], ck["yte"] / max(ck["nte"], 1)) for ck in _strata(c))


def excess_log_loss(name, theta, c: dict) -> float:
    """(saturated - profile log-likelihood) per displayed row: 0 is a perfect account of the three cell rates of
    every stratum; larger is worse."""
    return (saturated_ll(c) - profile_ll(name, theta, c)[0]) / max(float(np.sum(c["nte"])), 1.0)


def fit_theta(name, c: dict, center: float) -> np.ndarray:
    from scipy.optimize import minimize, minimize_scalar
    if name == "B0_no_effect":
        return np.zeros(0)
    if name in ("B1_constant_difference", "M1_constant_log_odds"):
        hi = 0.5 if name.startswith("B1") else 8.0
        r = minimize_scalar(lambda t: -profile_ll(name, [t], c)[0], bounds=(0.0, hi), method="bounded",
                            options={"xatol": 1e-5})
        return np.array([float(r.x)])
    if name == "M2_linear_log_odds":
        a0 = float(fit_theta("M1_constant_log_odds", c, center)[0])
        r = minimize(lambda t: -profile_ll(name, [t[0], t[1], center], c)[0], np.array([a0, 0.0]),
                     method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 400})
        return np.array([float(r.x[0]), float(r.x[1]), center])
    raise ValueError(name)


def predictor(name, theta) -> P5.AdExposure:
    if name == "B0_no_effect":
        return P5.AdExposure("none")
    if name == "B1_constant_difference":
        return P5.AdExposure("difference", delta=float(theta[0]))
    if name == "M1_constant_log_odds":
        return P5.AdExposure("linear_logit", a=float(theta[0]), b=0.0, c=0.0)
    return P5.AdExposure("linear_logit", a=float(theta[0]), b=float(theta[1]), c=float(theta[2]))


def fit_candidates(nz_fit: dict, c_fit: dict) -> tuple[dict, dict, float]:
    """{name: theta}, {name: predictor}, the M2 centre (displayed-weighted mean moment logit baseline, fit split)."""
    l0 = P5.logit(nz_fit["mu0"])
    n = c_fit["nte"]
    center = float(np.sum(n * l0) / max(np.sum(n), 1.0))
    thetas = {k: fit_theta(k, c_fit, center) for k in CANDS}
    return thetas, {k: predictor(k, thetas[k]) for k in CANDS}, center


COMPLEXITY = {"B0_no_effect": ("baseline: no effect", 0), "B1_constant_difference": ("baseline: one difference", 1),
              "M1_constant_log_odds": ("one log-odds shift", 1), "M2_linear_log_odds": ("linear log-odds shift", 2)}


def boot_excess_diff(a: tuple, b: tuple, c: dict, n: int = N_BOOT, seed: int = 0) -> dict:
    """Poisson cell bootstrap of the excess log loss of candidates a and b = (name, theta) and their difference."""
    rng = np.random.default_rng(seed)
    da, dd = [], []
    for _ in range(n):
        cb = _pois(rng, c)
        ea, eb = excess_log_loss(a[0], a[1], cb), excess_log_loss(b[0], b[1], cb)
        da.append(ea)
        dd.append(ea - eb)
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"a_ci95": q(da), "difference_ci95": q(dd), "resamples": n}


def exposed_rows(fr: dict, nz: dict) -> tuple[np.ndarray, np.ndarray]:
    """(X = organic_logit of each treated exposed row's stratum on its own split, y = visit)."""
    m = fr["z"] & fr["D"]
    return P5.logit(nz["mu0"])[fr["s"][m]][:, None], fr["y"][m]


def implied_itt(pred, nz: dict) -> np.ndarray:
    q = pred.predict(P5.logit(nz["mu0"])[:, None])
    return nz["exposure"] * (q - nz["mu0"])


def weighted_mse(pred_itt, obs_itt, rows) -> float:
    w = rows / rows.sum()
    return float(np.sum(w * (np.asarray(pred_itt) - np.asarray(obs_itt)) ** 2))


# ====================================================================== bootstraps (cell-level Poisson)
def _pois(rng, c: dict) -> dict:
    out = {}
    for g in ("c", "te", "tn"):
        yy, nn = c[f"y{g}"], c[f"n{g}"]
        ys = rng.poisson(yy).astype(float)
        out[f"y{g}"], out[f"n{g}"] = ys, ys + rng.poisson(np.maximum(nn - yy, 0)).astype(float)
    return out


def stratum_shift_ci(c: dict, n: int = N_BOOT, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        nz = nuisance(_pois(rng, c))
        with np.errstate(divide="ignore", invalid="ignore"):
            draws.append(P5.logit(np.clip(nz["exposed"], 1e-6, 1 - 1e-6)) - P5.logit(nz["mu0"]))
    A = np.array(draws)
    return [[float(np.nanquantile(A[:, k], 0.025)), float(np.nanquantile(A[:, k], 0.975))] for k in range(A.shape[1])]


def conversion_ratio(fr: dict, n: int = N_BOOT, seed: int = 0) -> dict:
    """rho = (ITT_conversion / ITT_visit) / (control conversion rate / control visit rate), pooled."""
    z, y, cv = fr["z"], fr["y"], fr["conv"]
    cnt = np.array([[np.sum(~z & (y == 0)), np.sum(~z & (y == 1) & (cv == 0)), np.sum(~z & (cv == 1))],
                    [np.sum(z & (y == 0)), np.sum(z & (y == 1) & (cv == 0)), np.sum(z & (cv == 1))]], float)

    def rho(C):
        n0, n1 = C[0].sum(), C[1].sum()
        v0, v1 = (C[0, 1] + C[0, 2]) / n0, (C[1, 1] + C[1, 2]) / n1
        c0, c1 = C[0, 2] / n0, C[1, 2] / n1
        return ((c1 - c0) / (v1 - v0)) / (c0 / v0), (c1 - c0) / (v1 - v0), c0 / v0
    r, inc, org = rho(cnt)
    rng = np.random.default_rng(seed)
    bs = np.array([rho(rng.poisson(cnt).astype(float))[0] for _ in range(n)])
    return {"rho": float(r), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "conversions_per_incremental_visit": float(inc), "conversions_per_control_visit": float(org),
            "counts": cnt.astype(int).tolist()}


def time_us(pred, X, reps: int = 5) -> float:
    X = np.ascontiguousarray(np.asarray(X, float)[:100000])
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        pred.predict(X)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) / max(1, X.shape[0]) * 1e6)


# ====================================================================== main
def prepare(d: dict | None = None, model=None, lab=None, splits=("fit", "val", "test"), score_fn=None) -> dict:
    """Scores, strata and nuisances of the plan's splits (shared with the reproducer). `score_fn(model, d, rows,
    sub)` defaults to the plan's score (commercial_response@1's kept sub-model); synthetic-data tests pass the true
    propensity of their simulated trial instead."""
    d = criteo.load() if d is None else d
    model = load("commercial_response@1") if model is None else model
    score_fn = score if score_fn is None else score_fn
    rows = split_rows(lab)
    p0 = {k: score_fn(model, d, rows[k], "visit_control") for k in splits}
    raw_fit = frame(d, rows["fit"], np.zeros(rows["fit"].size))
    edges = strata_edges(p0["fit"], raw_fit)
    K = edges.size + 1
    fr = {k: frame(d, rows[k], np.searchsorted(edges, p0[k], side="right")) for k in splits}
    c = {k: cells(fr[k], K) for k in splits}
    nz = {k: nuisance(c[k]) for k in splits}
    return {"d": d, "model": model, "rows": rows, "p0": p0, "edges": edges, "K": K, "fr": fr, "cells": c, "nz": nz,
            "score_fn": score_fn}


def run(cmd: str = "python -m crucible.calibration.fit_e5", d: dict | None = None, lab=None,
        write: bool = True, score_fn=None):
    """The plan's fit. With `write=False` (synthetic-data unit tests) nothing is written and the report is
    returned."""
    t0 = time.time()
    plan = F.plan_record(PLAN_PATH) if write else {"path": PLAN_PATH, "dry_run": True}
    s = spec()
    S = prepare(d=d, lab=lab, score_fn=score_fn)
    d, model, rows, fr, c, nz, K = S["d"], S["model"], S["rows"], S["fr"], S["cells"], S["nz"], S["K"]
    print("strata", K, "edges", S["edges"].round(5).tolist(), "secs", round(time.time() - t0), flush=True)
    ffit = fr["fit"]
    late = float((ffit["y"][ffit["z"]].mean() - ffit["y"][~ffit["z"]].mean()) / ffit["D"][ffit["z"]].mean())
    thetas, preds, center = fit_candidates(nz["fit"], c["fit"])
    print("fitted", {k: np.round(v, 4).tolist() for k, v in thetas.items()}, "secs", round(time.time() - t0), flush=True)
    Xs, ys = {}, {}
    for k in ("fit", "val", "test"):
        Xs[k], ys[k] = exposed_rows(fr[k], nz[k])
    cands = {}
    for name in CANDS:
        pr = preds[name]
        cx, npar = COMPLEXITY[name]
        cands[name] = {"complexity": cx, "n_params": int(npar), "pred": pr, "theta": thetas[name].tolist()}
        for k, key in (("fit", "fit"), ("val", "val"), ("test", "test")):
            cands[name][key] = dict(M.binary_report(ys[k], pr.predict(Xs[k])),
                                    excess_log_loss=excess_log_loss(name, thetas[name], c[k]))
    compute = {k: time_us(preds[k], np.repeat(Xs["val"], 2, axis=0)) for k in CANDS}
    lh = {k: None for k in CANDS}
    sel = E4.select_e4(cands, PRIMARY, BASELINES, list(CANDS), True, compute, lh, {}, 1)
    kept = sel["kept"]
    pt = {k: preds[k].predict(Xs["test"]) for k in CANDS}
    best_base = sel["best_baseline_on_test"]
    bt = boot_excess_diff((kept, thetas[kept]), (best_base, thetas[best_base]), c["test"])
    bc = {"metric": PRIMARY, "kept": kept, "baseline": best_base, "kept_test": sel["test"][kept],
          "baseline_test": sel["test"][best_base], "difference": sel["test"][kept] - sel["test"][best_base],
          "ci95": bt["difference_ci95"], "kept_ci95": bt["a_ci95"],
          "bootstrap": {"resamples": bt["resamples"], "kind": "Poisson bootstrap of the stratum cells (iid users)"}}
    # ---- per-stratum tables, implied ITT, rule comparison
    tables = {}
    for k in ("fit", "val", "test"):
        z = nz[k]
        tables[k] = [{"stratum": j, "rows": int(z["rows"][j]), "exposed_rows": int(z["exposed_rows"][j]),
                      "exposure_rate": float(z["exposure"][j]), "control_visit_rate": float(z["control"][j]),
                      "never_taker_visit_rate": float(z["never_taker"][j]), "complier_baseline": float(z["mu0"][j]),
                      "complier_baseline_unclipped": float(z["mu0_raw"][j]), "exposed_visit_rate": float(z["exposed"][j]),
                      "itt": float(z["itt"][j]), "wald_late": float(z["late"][j]),
                      "complier_log_odds_shift": float(P5.logit(np.clip(z["exposed"][j], 1e-6, 1 - 1e-6))
                                                       - P5.logit(z["mu0"][j]))} for j in range(K)]
    shift_ci = stratum_shift_ci(c["test"])
    for j, ci in enumerate(shift_ci):
        tables["test"][j]["complier_log_odds_shift_ci95"] = ci
    negative = [j for j, ci in enumerate(shift_ci) if ci[1] < 0]
    itt_mse = {k: weighted_mse(implied_itt(preds[k], nz["test"]), nz["test"]["itt"], nz["test"]["rows"]) for k in CANDS}
    # simulator rules (plan rule_mapping): S1 runtime ad_response@1 as served, S2 its prespecified type table
    base = model.card["selection"]["baseline"]
    cc = model.candidate_predict(base, np.zeros((1, len(model.features))))[0]
    s1 = np.full(K, float(cc[1] - cc[0]))
    tau = S["score_fn"](model, S["d"], rows["test"], "visit_treated") - S["p0"]["test"]
    from .commercial import assign_cells
    cell = assign_cells(S["p0"]["test"], tau, model.arrays["types/p0_edges"], model.arrays["types/tau_edges"])
    tab = model.response_types("model")
    eff = (tab["p1_visit"] - tab["p0_visit"])[cell]
    st = fr["test"]["s"]
    s2 = np.bincount(st, weights=eff, minlength=K) / np.maximum(np.bincount(st, minlength=K), 1)
    rule = {"mapping": s["rule_mapping"], "metric": "weighted mean squared error of the per-stratum ITT on test",
            "test": {"S1_ad_response@1_served_constant_effect": weighted_mse(s1, nz["test"]["itt"], nz["test"]["rows"]),
                     "S2_ad_response@1_type_table": weighted_mse(s2, nz["test"]["itt"], nz["test"]["rows"]),
                     **{f"{k} (E5 candidate)": v for k, v in itt_mse.items()}},
            "per_stratum_test_itt": {"observed": nz["test"]["itt"].tolist(), "S1": s1.tolist(), "S2": s2.tolist(),
                                     **{k: implied_itt(preds[k], nz["test"]).tolist() for k in CANDS}},
            "not_comparable": "economy.work sponsored-search click (one hand-set probability): a click on a search "
                              "result, not a campaign display; no randomised counterpart in these data"}
    rho = {k: conversion_ratio(fr[k]) for k in ("fit", "test")}
    served_rho = rho["fit"]["rho"] if not (rho["fit"]["ci95"][0] <= 1.0 <= rho["fit"]["ci95"][1]) else 1.0
    conv_tables = {}
    for k in ("fit", "test"):
        zc = nuisance(cells(fr[k], K, "conv"))
        conv_tables[k] = [{"stratum": j, "complier_conversion_baseline": float(zc["mu0"][j]),
                           "complier_conversion_baseline_unclipped": float(zc["mu0_raw"][j]),
                           "exposed_conversion_rate": float(zc["exposed"][j]), "itt": float(zc["itt"][j])}
                          for j in range(K)]
    # parameter intervals of the kept candidate (Poisson cell bootstrap of the fit split, refitted)
    rng = np.random.default_rng(0)
    th_boot = []
    for _ in range(N_BOOT if thetas[kept].size else 0):
        th_boot.append(fit_theta(kept, _pois(rng, c["fit"]), center)[:2].tolist())
    theta_ci = ([[float(np.quantile(np.array(th_boot)[:, j], 0.025)), float(np.quantile(np.array(th_boot)[:, j], 0.975))]
                 for j in range(len(th_boot[0]))] if th_boot else [])
    pk = pt[kept]
    calib = {"ece": M.ece(ys["test"], pk), "brier": M.brier(ys["test"], pk), "mean_pred": float(pk.mean()),
             "mean_obs": float(ys["test"].mean()), "reliability": M.calibration_table(ys["test"], pk),
             "itt_weighted_mse": itt_mse[kept]}
    km = preds[kept]
    shift_grid = {f"{p:g}": float(km.shift(P5.logit([p]))[0]) if km.mode in ("linear_logit", "step_logit") else None
                  for p in (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8)}
    extra = {"strata": {"edges_p0_hat": S["edges"].tolist(), "K": K, "rule": s["strata"]},
             "per_stratum": tables, "conversion_per_stratum": conv_tables,
             "negative_effect_strata_test": negative,
             "negative_effects": ("no stratum's test interval lies entirely below 0: the floor at 0 agrees with the "
                                  "data" if not negative else f"strata {negative} have a test interval below 0: the "
                                  "floor at 0 is a known deviation from the data"),
             "incremental_visit_conversion_ratio": rho, "served_conversion_ratio": served_rho,
             "pooled_wald_late_fit": late, "kept_shift_at_organic_probability": shift_grid,
             "m2_center_logit": center, "theta": {k: thetas[k].tolist() for k in CANDS},
             "kept_theta_ci95_fit_bootstrap": theta_ci,
             "fit_split_metrics": {k: cands[k]["fit"] for k in CANDS}}
    splits = {k: {"definition": s["splits"]["validation" if k == "val" else k], "rows": int(rows[k].size),
                  "exposed_treated_rows": int(ys[k].size)} for k in ("fit", "val", "test")}
    splits["explore"] = {"definition": s["splits"]["explore"], "rows": int(rows["explore"].size)}
    limits = (
        "Criteo Uplift v2.1 (13,979,592 users of pooled randomised incrementality tests of Criteo display advertising; "
        "advertisers, countries and dates undisclosed; released 2018; sub-sampled non-uniformly by the publisher). "
        "The simulator may use this model for: the visit probability of an individual who is displayed (and, through "
        "ad_response@2, attends) an advertiser's display or feed campaign during a campaign period, as a log-odds shift "
        "of that individual's own organic visit probability. It may NOT infer: (1) effects for compliers unlike "
        "Criteo's - the displayed users were chosen by Criteo's bidder (exposure rises from about 0.1% to 15% of "
        "eligible users with the baseline score), so the effect is that of a targeted display on users a bidder "
        "reached; (2) effects of ad content, creative quality, price or relevance (not observed; ad_response@2 adds "
        "relevance and repetition only through a separate attention model and a declared assumption); (3) frequency, "
        "wear-in, wear-out, carry-over or repeated campaigns (no timestamps, one period per user); (4) negative "
        "effects (floored at 0); (5) absolute rates: the publisher's sub-sampling distorts levels; the log-odds shift "
        "is invariant to sub-sampling that depends on the outcome or on the arm only, not to sub-sampling that "
        "depends on exposure; (6) search advertising (a click on a sponsored result is a different decision). "
        "Strata are bins of a pre-treatment score; within-stratum heterogeneity is averaged, and the complier baseline "
        "of each stratum is a moment estimate (noisy where exposure is rare; clipped to [1e-4, 0.999]).")
    ident = ("Randomised eligibility (treatment) is the instrument for display (exposure). Assumptions: exclusion "
             "restriction (eligibility affects visits only through display), one-sided non-compliance (no control user "
             "is displayed: exposed_in_control = 0 in the conversion record) and monotonicity (trivial here). Under "
             "them, each stratum's complier baseline and complier visit rate under display are identified, and so is "
             "their log-odds difference. The score that defines strata uses pre-treatment features only and was fitted "
             "on the train split, so strata are balanced between the arms on the fit, validation and test splits.")
    fmap = {"organic_logit": ("log-odds of the displayed individual's organic (no-ad) probability of visiting the "
                              "advertiser during the campaign period: in the data the stratum's complier baseline, in "
                              "the simulator the individual's own organic visit probability (ad_response@2 input "
                              "organic_visit_probability)")}
    rep = {"schema": "crucible.v5.calibration_report/1", "model": NAME, "version": VERSION, "domain": s["domain"],
           "datasets": [s["dataset"]], "identification": s["identification"], "estimand": s["estimand"],
           "population": "13,979,592 users reachable by Criteo display advertising (pooled incrementality tests)",
           "platform": "Criteo display advertising (retargeting and prospecting campaigns of undisclosed advertisers)",
           "period": "undisclosed; released 2018", "plan": plan, "splits": splits, "heldout_splits": ["test"],
           "candidates": F.strip(cands), "selection": sel, "baseline_comparison": bc, "calibration": calib,
           "heldout_extra": extra, "rule_comparison": rule,
           "long_horizon_test": {"not_applicable": s["long_horizon"]}, "identification_note": ident,
           "extrapolation_limits": limits, "feature_map": fmap,
           "fit": {"commit": F.head_commit(), "seconds": time.time() - t0, "command": cmd}}
    if not write:
        return {"report": rep, "preds": preds, "S": S}
    card_extra = {"module": "crucible.calibration.predictors_e5",
                  "mechanisms": ["advertising_intervention_response", "advertising_response_heterogeneity",
                                 "advertising.attention_composition"],
                  "prior_rate": float(ys["fit"].mean()),
                  "feature_quantiles": F.feature_quantiles(Xs["fit"], ["organic_logit"]),
                  "served_conversion_ratio": served_rho, "reference_attention": 1.0,
                  "strata_edges_p0_hat": S["edges"].tolist()}
    out = F.finish(NAME, VERSION, s["domain"], km, ["organic_logit"], "visit: the displayed individual visits the "
                   "advertiser during the campaign period", rep, card_extra)
    print("kept", kept, "test", sel["test"], "secs", round(time.time() - t0), flush=True)
    return out


def main(argv=None) -> int:
    run(" ".join(["python -m crucible.calibration.fit_e5"] + list(argv or sys.argv[1:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
