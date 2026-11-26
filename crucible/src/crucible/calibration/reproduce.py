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


def _rt(model_id: str, X: np.ndarray) -> np.ndarray:
    """Held-out predictions of a registered checkpoint through the runtime decision-model API (so every
    reproduction also checks that the runtime loads the checkpoint and evaluates held-out rows)."""
    from . import runtime
    dm = runtime.decision_for(model_id)
    return np.asarray(dm.evaluate_features(X, checkpoint=True).values, dtype=np.float64)


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
                out[name] = _row(name, M.log_loss(y[cm[split]], _rt(f.id, X)), len(X))
        if "watch_duration" in names:
            f = load("watch_duration")
            wm = E.watch_masks(cf)["test"]
            X = cf.loc[wm, f.features].to_numpy(float)
            yy = np.log1p(cf.loc[wm, "play_s"].to_numpy(float))
            out["watch_duration"] = _row("watch_duration", M.pinball(yy, _rt(f.id, X),
                                                                     np.asarray(f.card["quantile_levels"])), len(X))
    if "return_daily" in names:
        f = load("return_daily")
        af = K.activity_frame(raw, K.day_num(2022, 4, 29), K.day_num(2022, 5, 8))
        rm = E.return_masks(af)["test"]
        X = af.loc[rm, f.features].to_numpy(float)
        out["return_daily"] = _row("return_daily", M.log_loss(af.loc[rm, "active"].to_numpy(float), _rt(f.id, X)),
                                   len(X))
    if "session_rate" in names:
        f = load("session_rate")
        hc, he, tc, te, users = E.session_test_arrays(E.session_days(raw))
        rate = _rt(f.id, np.column_stack([hc, he]))
        out["session_rate"] = _row("session_rate", M.poisson_deviance(tc, rate * te), len(users))
    return out


def reproduce_news() -> dict:
    from . import mind
    f = load("news_click")
    data = mind.load()
    df, _ = mind.build(data)
    m = mind.news_masks(df, data["beh"])["test"]
    X = df.loc[m, f.features].to_numpy(float)
    return {"news_click": _row("news_click", M.log_loss(df.loc[m, "y"].to_numpy(float), _rt(f.id, X)), len(X))}


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
    P = _rt(f.id, X)
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
    ll = S.log_loss(tab["deep/y"][m].astype(float), _rt(deep.id, tab["deep/X"][m]))
    growth = load("software_repo_growth@1")
    dev = S.poisson_deviance(tab["growth/d2/y"], _rt(growth.id, tab["growth/d2/X"]))
    return {"software_deep_engagement": _row_e2(deep, "log_loss", deep.metrics["log_loss"], ll, int(m.sum()),
                                                "test split (first stars on D2 12:00-24:00 UTC)"),
            "software_repo_growth": _row_e2(growth, "mean_poisson_deviance", growth.metrics["mean_poisson_deviance"],
                                            dev, len(tab["growth/d2/y"]), "test split (D2 -> D3)")}


# ---------------------------------------------------------------- E4 models (fit_e4, plan E4_new_domains_plan.json)
E4_MODELS = ("customer_return", "product_demand", "repo_participation_return", "star_cascade", "creator_return",
             "session_length", "news_click_semantic")


def _e4_row(name: str, recomputed: float, n: int, scope: str) -> dict:
    r = _row(name, recomputed, n)
    r["scope"] = scope
    return r


def reproduce_retail() -> dict:
    from . import fit_e4 as E
    out = {}
    df = E.retail_frame()
    if _registered("customer_return"):
        Xd, y, day, cid, _ = E.customer_table(df)
        f = load("customer_return")
        X = np.column_stack([Xd[c] for c in f.features])
        uc = np.unique(cid)
        bucket = E._cust_bucket(uc)[np.searchsorted(uc, cid)]
        m = (bucket >= 85) & (day >= E._day(2011, 4, 1)) & (day <= E._day(2011, 9, 10))
        out["customer_return"] = _e4_row("customer_return", M.log_loss(y[m], _rt(f.id, X[m])), int(m.sum()),
                                         "test customers, 2011-04-01..09-10")
    if _registered("product_demand"):
        f = load("product_demand")
        X, y = E.demand_test_rows(df)
        p = _rt(f.id, X)
        out["product_demand"] = _e4_row("product_demand", float(np.mean((p - y) ** 2)), len(y),
                                        "test weeks 2011-06-27..11-28")
    return out


def reproduce_github_e4() -> dict:
    from . import fit_e4 as E
    out = {}
    if _registered("repo_participation_return"):
        f = load("repo_participation_return")
        tab = E.github_participation_table()
        X = np.column_stack([tab["X"][c] for c in f.features])
        H = tab["hour"]
        m = (H >= 60) & (H < 72)
        out["repo_participation_return"] = _e4_row("repo_participation_return",
                                                   M.log_loss(tab["y"][m], _rt(f.id, X[m])), int(m.sum()),
                                                   "first participations at hours 60-72")
    if _registered("star_cascade"):
        f = load("star_cascade")
        X, y = E.cascade_test_rows()
        out["star_cascade"] = _e4_row("star_cascade", M.poisson_deviance(y, _rt(f.id, X)), len(y),
                                      "repository-hours 72-95")
    return out


def reproduce_kuairand_e4() -> dict:
    from . import fit_e4 as E
    out = {}
    raw = K.load_raw()
    if _registered("creator_return"):
        f = load("creator_return")
        X, y, dd, us, _, _, _ = E.creator_table(raw)
        split_u = K.split_of_bucket(np.array([K.user_bucket(u) for u in us]))
        m = (split_u == "test") & (dd >= K.day_num(2022, 4, 25)) & (dd <= K.day_num(2022, 5, 1))
        out["creator_return"] = _e4_row("creator_return", M.log_loss(y[m], _rt(f.id, X[m])), int(m.sum()),
                                        "test users, 2022-04-25..05-01")
    if _registered("session_length"):
        f = load("session_length")
        g = E.kuairand_sessions(raw)
        split_u = K.split_of_bucket(g["bucket"].to_numpy())
        d = g["day"].to_numpy()
        tr = (split_u == "train") & (d >= K.day_num(2022, 4, 22)) & (d <= K.day_num(2022, 5, 2))
        X = E.session_features(g, pop_mean=float(g["y"].to_numpy()[tr].mean()))
        m = (split_u == "test") & (d >= K.day_num(2022, 5, 3)) & (d <= K.day_num(2022, 5, 8))
        lv = np.asarray(f.card["quantile_levels"])
        out["session_length"] = _e4_row("session_length", M.pinball(g["y"].to_numpy()[m], _rt(f.id, X[m]), lv),
                                        int(m.sum()), "test users, sessions 2022-05-03..05-08")
    return out


def mind_minilm_embeddings(data: dict) -> np.ndarray:
    """MiniLM-L6 embeddings of every MIND news text (title + ' ' + abstract) in `mind.load()` order, cached in the
    dataset directory (derived, never committed); computed with crucible.semantic.encoders.text when absent."""
    from .datasets import dataset_dir
    p = os.path.join(dataset_dir("mind_small"), "derived", "minilm_title_abstract.npz")
    ids = data["news"]["news_id"].astype(str).to_numpy()
    if os.path.exists(p):
        with np.load(p) as z:
            if list(z["news_id"]) == list(ids):
                return z["emb"].astype(np.float32)
    from ..semantic.encoders.text import TextEncoder
    text = [" ".join(t.split()) for t in (data["news"]["title"].fillna("") + " " +
                                          data["news"]["abstract"].fillna("")).tolist()]
    enc = TextEncoder()
    E = np.zeros((len(text), 384), np.float32)
    for s in range(0, len(text), 2048):
        E[s:s + 2048] = enc.embed_texts(text[s:s + 2048])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, emb=E.astype(np.float16), news_id=ids, encoder=np.array([enc.id]))
    return E.astype(np.float16).astype(np.float32)


def reproduce_news_semantic() -> dict:
    from . import mind
    if not _registered("news_click_semantic"):
        return {}
    f = load("news_click_semantic")
    data = mind.load()
    E = mind_minilm_embeddings(data)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    df, _ = mind.build(data, E=E)
    m = mind.news_masks(df, data["beh"])["test"]
    X = df.loc[m, f.features].to_numpy(float)
    return {"news_click_semantic": _e4_row("news_click_semantic", M.log_loss(df.loc[m, "y"].to_numpy(float),
                                                                           _rt(f.id, X)), len(X), "dev day")}


def _registered(name: str) -> bool:
    from .registry import list_models
    return any(m.split("@")[0] == name for m in list_models())


REPRODUCERS = {"kuairand": reproduce_kuairand, "news": reproduce_news, "commercial": reproduce_commercial,
               "software": reproduce_software, "retail": reproduce_retail, "github_e4": reproduce_github_e4,
               "kuairand_e4": reproduce_kuairand_e4, "news_semantic": reproduce_news_semantic}


def reproduce_all() -> dict:
    """{model name: row} for every model the reproducers cover (E1's six, E2's three and the E4 models)."""
    out = {}
    for fn in REPRODUCERS.values():
        out.update(fn())
    return out
