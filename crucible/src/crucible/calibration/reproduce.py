"""Recompute the held-out test metrics of the registered models from the verified datasets and the saved
checkpoints, using only `registry.load` (the portable NumPy predictors) and the prespecified split functions.

This is how the gate tests confirm that a report's numbers are what the saved model actually achieves.
`reproduce_all()` covers every model listed in `REPRODUCERS`; each row is {metric, reported, recomputed, n, abs_diff,
scope}. Each toolchain's own metric functions are used (`metrics` for E1, `scoring_e2` for E2), because the reports
were computed with them."""
from __future__ import annotations

import os

import numpy as np

from . import kuairand as K
from . import metrics as M
from .registry import load

E1_MODELS = ("content_consumption", "content_consumption_feed", "watch_duration", "return_daily", "session_rate",
             "news_click")
E2_MODELS = ("commercial_response", "software_deep_engagement", "software_repo_growth")


def _row(name: str, recomputed: float, n: int) -> dict:
    f = load(name)
    metric = f.card["selection"]["primary_metric"]
    reported = float(f.metrics[metric])
    return {"metric": metric, "reported": reported, "recomputed": float(recomputed), "n": int(n),
            "abs_diff": abs(float(recomputed) - reported)}


def reproduce_kuairand(names=("content_consumption", "content_consumption_feed", "watch_duration", "return_daily",
                              "session_rate")) -> dict:
    from . import fit_e1 as E
    raw = K.load_raw()
    out = {}
    if {"content_consumption", "content_consumption_feed", "watch_duration"} & set(names):
        cf = K.choice_frame(raw)
        cm = E.content_masks(cf)
        y = cf["is_click"].to_numpy(float)
        for name, split in (("content_consumption", "rnd_test"), ("content_consumption_feed", "obs_test")):
            if name in names:
                f = load(name)
                X = cf.loc[cm[split], f.features].to_numpy(float)
                out[name] = _row(name, M.log_loss(y[cm[split]], f.predict(X)), len(X))
        if "watch_duration" in names:
            f = load("watch_duration")
            wm = E.watch_masks(cf)["test"]
            X = cf.loc[wm, f.features].to_numpy(float)
            yy = np.log1p(cf.loc[wm, "play_s"].to_numpy(float))
            out["watch_duration"] = _row("watch_duration", M.pinball(yy, f.predict(X),
                                                                     np.asarray(f.card["quantile_levels"])), len(X))
    if "return_daily" in names:
        f = load("return_daily")
        af = K.activity_frame(raw, K.day_num(2022, 4, 29), K.day_num(2022, 5, 8))
        rm = E.return_masks(af)["test"]
        X = af.loc[rm, f.features].to_numpy(float)
        out["return_daily"] = _row("return_daily", M.log_loss(af.loc[rm, "active"].to_numpy(float), f.predict(X)),
                                   len(X))
    if "session_rate" in names:
        f = load("session_rate")
        hc, he, tc, te, users = E.session_test_arrays(E.session_days(raw))
        rate = f.predict(np.column_stack([hc, he]))
        out["session_rate"] = _row("session_rate", M.poisson_deviance(tc, rate * te), len(users))
    return out


def reproduce_news() -> dict:
    from . import mind
    f = load("news_click")
    data = mind.load()
    df, _ = mind.build(data)
    m = mind.news_masks(df, data["beh"])["test"]
    X = df.loc[m, f.features].to_numpy(float)
    return {"news_click": _row("news_click", M.log_loss(df.loc[m, "y"].to_numpy(float), f.predict(X)), len(X))}


def _row_e2(f, metric: str, reported: float, recomputed: float, n: int, scope: str) -> dict:
    return {"metric": metric, "reported": float(reported), "recomputed": float(recomputed), "n": int(n),
            "abs_diff": abs(float(recomputed) - float(reported)), "scope": scope}


def reproduce_commercial() -> dict:
    """commercial_response@1: the kept candidate's primary metric (pooled visit + conversion log loss, each row
    scored by its own arm's model) on the report's seeded audit subsample of 100,000 test users (amendment 1, A1.3),
    about 12 s. The full-test number (2,795,919 users, 28 times the tree evaluations) is the fit's own."""
    from . import criteo
    from . import scoring_e2 as S
    from .commercial import ARMS, OUTCOMES
    from .fitmeta_e2 import digest
    from .registry import report
    f = load("commercial_response@1")
    rep = report(f.id)
    lab = criteo.splits()
    if criteo.split_digest(lab) != f.card["fit"]["split"]["digest"]:
        raise ValueError("commercial_response: the seeded split differs from the fit's")
    idx = criteo.audit_indices(lab)
    if digest(idx.astype(np.int64)) != rep["audit"]["indices_digest"]:
        raise ValueError("commercial_response: the audit subsample differs from the report's")
    data = criteo.load()
    X = np.asarray(data["X"][idx], dtype=np.float64)
    t = np.asarray(data["treatment"][idx]).astype(bool)
    P = f.predict(X)
    cols = {f"{o}_{a}": P[:, i] for i, (o, a) in enumerate([(o, a) for o in OUTCOMES for a in ARMS])}
    primary = sum(S.log_loss(np.asarray(data[o][idx]), np.where(t, cols[f"{o}_treated"], cols[f"{o}_control"]))
                  for o in OUTCOMES)
    kept = f.card["selection"]["kept"]
    return {"commercial_response": _row_e2(f, "primary_pooled_logloss_visit_plus_conversion",
                                           rep["audit"]["candidates"][kept]["primary"], primary, len(idx),
                                           "test audit subsample (seeded, 100,000 users)")}


def reproduce_software() -> dict:
    """software_deep_engagement@1 (log loss) and software_repo_growth@1 (mean Poisson deviance) on their whole test
    splits, from the evaluated tables built from the extracted GH Archive window (`fit_software.tables_path`)."""
    from . import fit_software as F
    from . import scoring_e2 as S
    p = F.tables_path()
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: acquire gharchive_2024-10-07_96h and run fit_software")
    with np.load(p) as z:
        tab = {k: z[k] for k in ("deep/t", "deep/X", "deep/y", "growth/d2/X", "growth/d2/y")}
    a, b = F.SPLIT_HOURS["test"]
    m = (tab["deep/t"] >= a * 3600) & (tab["deep/t"] < b * 3600)
    deep = load("software_deep_engagement@1")
    ll = S.log_loss(tab["deep/y"][m].astype(float), deep.predict(tab["deep/X"][m]))
    growth = load("software_repo_growth@1")
    dev = S.poisson_deviance(tab["growth/d2/y"], growth.predict(tab["growth/d2/X"]))
    return {"software_deep_engagement": _row_e2(deep, "log_loss", deep.metrics["log_loss"], ll, int(m.sum()),
                                                "test split (first stars on D2 12:00-24:00 UTC)"),
            "software_repo_growth": _row_e2(growth, "mean_poisson_deviance", growth.metrics["mean_poisson_deviance"],
                                            dev, len(tab["growth/d2/y"]), "test split (D2 -> D3)")}


REPRODUCERS = {"kuairand": reproduce_kuairand, "news": reproduce_news, "commercial": reproduce_commercial,
               "software": reproduce_software}


def reproduce_all() -> dict:
    """{model name: row} for every model the reproducers cover (E1's six and E2's three)."""
    out = {}
    for fn in REPRODUCERS.values():
        out.update(fn())
    return out
