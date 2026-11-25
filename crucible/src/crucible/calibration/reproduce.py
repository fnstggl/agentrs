"""Recompute the held-out test metrics of E1's registered models from the verified datasets and the saved
checkpoints, using only `registry.load` (the portable NumPy predictors) and the prespecified split functions.

This is how the gate tests confirm that a report's numbers are what the saved model actually achieves."""
from __future__ import annotations

import numpy as np

from . import kuairand as K
from . import metrics as M
from .registry import load

E1_MODELS = ("content_consumption", "content_consumption_feed", "watch_duration", "return_daily", "session_rate",
             "news_click")


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
