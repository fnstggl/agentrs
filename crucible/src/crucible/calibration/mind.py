"""MIND-small news-browsing choice model: P(click | a news item shown in an impression, the user's history).

Text affinity uses features computed here (TF-IDF of title + abstract reduced by truncated SVD); the semantic
workstream's neural encoders can replace the representation later, which is why the model also uses the scale-free
within-impression z-score of the affinity. All popularity and age features use only impressions of earlier hours.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from . import fitlib as F
from . import metrics as M
from .datasets import acquire

T0 = pd.Timestamp("2019-11-09 00:00:00")
FEATURES = ["text_cos", "text_cos_z", "cat_share", "subcat_share", "log_hist_len", "pop_ctr_24h",
            "pop_log_clicks_per_1k", "age_log_hours", "slate_size_log"]
POP_FEATURES = ["pop_ctr_24h", "pop_log_clicks_per_1k", "slate_size_log"]
FEATURE_MAP = {
    "text_cos": "cosine between the item's text embedding and the mean embedding of the individual's consumed items "
                "(here: 128-d LSA of TF-IDF; with another encoder map by quantile, card feature_quantiles)",
    "text_cos_z": "text_cos standardised within the slate the individual is shown (scale-free; use as is)",
    "cat_share": "(individual's past consumption in the item's category + s) / (past consumption + 1), s = population "
                 "share of the category",
    "subcat_share": "the same for the finer topic (subcategory)",
    "log_hist_len": "log1p(number of items in the individual's consumption history)",
    "pop_ctr_24h": "(clicks + 10 m) / (impressions + 10) of the item in the previous 24 simulated hours, m = card "
                   "prior_rate",
    "pop_log_clicks_per_1k": "log1p(1000 * clicks of the item in the previous 24 h / individuals active in those 24 h)",
    "age_log_hours": "log1p(hours since the item was first shown on the platform)",
    "slate_size_log": "log1p(number of items in the slate / page the individual is shown)",
}
LIMITS = (
    "Evidence: MIND-small, 50,000 sampled Microsoft News users (English, US market), impressions 2019-11-09..15 "
    "chosen by the production recommender. The simulator may use this model for the probability that an individual "
    "opens an article shown in a slate of news/article recommendations, given text affinity to its history, topic "
    "shares, popularity, freshness and slate size. It may NOT infer: (1) clicks on video or social posts (use "
    "content_consumption); (2) behaviour of populations, languages or periods other than US Microsoft News in "
    "November 2019; (3) the causal effect of showing an item (exposure was chosen by a recommender; there is no "
    "randomised exposure in MIND); (4) dwell or reading time (not in the data); (5) raw affinity values from a "
    "different encoder (use text_cos_z, or quantile-map text_cos). Popularity counts come from a 50k-user sample "
    "and are used per capita. News first shown on 2019-11-09 have left-censored ages.")


def _bucket(uid: str) -> int:
    return int(hashlib.sha256(f"e1:{uid}".encode()).hexdigest()[:8], 16) % 100


def load(root: str | None = None) -> dict:
    root = root or acquire("mind_small")
    ncols = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    bcols = ["impression_id", "user_id", "time", "history", "impressions"]
    out = {}
    for part in ("train", "dev"):
        d = os.path.join(root, "download", part)
        news = pd.read_csv(os.path.join(d, "news.tsv"), sep="\t", names=ncols, quoting=3)
        beh = pd.read_csv(os.path.join(d, "behaviors.tsv"), sep="\t", names=bcols, quoting=3)
        beh["hour"] = ((pd.to_datetime(beh["time"], format="%m/%d/%Y %I:%M:%S %p") - T0).dt.total_seconds()
                       // 3600).astype(int)
        beh["part"] = part
        out[part] = (news, beh)
    news = pd.concat([out["train"][0], out["dev"][0]]).drop_duplicates("news_id").reset_index(drop=True)
    beh = pd.concat([out["train"][1], out["dev"][1]], ignore_index=True)
    return {"news": news, "train_news_ids": set(out["train"][0]["news_id"]), "beh": beh, "root": root}


def embed(news: pd.DataFrame, train_ids: set, dim: int = 128, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    text = (news["title"].fillna("") + " " + news["abstract"].fillna("")).to_numpy()
    is_tr = news["news_id"].isin(train_ids).to_numpy()
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=3, max_features=50000)
    Xtr = vec.fit_transform(text[is_tr])
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    svd.fit(Xtr)
    E = svd.transform(vec.transform(text)).astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return E


def explode(beh: pd.DataFrame, nid: dict) -> pd.DataFrame:
    rows_imp, rows_news, rows_y = [], [], []
    for i, s in enumerate(beh["impressions"].to_numpy()):
        for tok in s.split():
            n, c = tok.rsplit("-", 1)
            rows_imp.append(i)
            rows_news.append(nid[n])
            rows_y.append(int(c))
    return pd.DataFrame({"imp": np.asarray(rows_imp, np.int64), "news": np.asarray(rows_news, np.int64),
                         "y": np.asarray(rows_y, np.int8)})


def build(data: dict, E: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Candidate rows with features. `E` [n_news, d] (rows in `data["news"]` order, L2-normalised) replaces the
    TF-IDF/LSA text representation (news_click_semantic uses the shared MiniLM encoder)."""
    news, beh = data["news"], data["beh"]
    nid = {n: i for i, n in enumerate(news["news_id"])}
    E = embed(news, data["train_news_ids"]) if E is None else np.asarray(E, np.float32)
    cand = explode(beh, nid)
    hour = beh["hour"].to_numpy()[cand["imp"].to_numpy()]
    H = int(hour.max()) + 2
    N = len(news)
    # hourly clicks / impressions per news, distinct active users per hour
    clicks = np.zeros((N, H), np.float32)
    imps = np.zeros((N, H), np.float32)
    np.add.at(clicks, (cand["news"].to_numpy(), hour), cand["y"].to_numpy())
    np.add.at(imps, (cand["news"].to_numpy(), hour), 1)
    cc = np.concatenate([np.zeros((N, 1), np.float32), np.cumsum(clicks, 1)], 1)     # cc[:, h] = sum over < h
    ci = np.concatenate([np.zeros((N, 1), np.float32), np.cumsum(imps, 1)], 1)
    users_by_hour = beh.groupby("hour")["user_id"].apply(set).to_dict()
    active24 = np.array([len(set().union(*[users_by_hour.get(x, set()) for x in range(h - 24, h)]))
                         for h in range(H + 1)], float)
    first_seen = np.full(N, np.inf)
    np.minimum.at(first_seen, cand["news"].to_numpy(), hour.astype(float))
    prior_rate = float(cand["y"].mean())
    n_ = cand["news"].to_numpy()
    lo = np.maximum(hour - 24, 0)
    c24 = cc[n_, hour] - cc[n_, lo]
    i24 = ci[n_, hour] - ci[n_, lo]
    # user profiles and category shares from the impression's history
    cat = pd.factorize(news["category"])[0]
    sub = pd.factorize(news["subcategory"])[0]
    hist_lists = [[nid[h] for h in s.split() if h in nid] if isinstance(s, str) else [] for s in beh["history"]]
    allh = np.fromiter((x for hl in hist_lists for x in hl), np.int64)
    s_cat = np.bincount(cat[allh], minlength=cat.max() + 1) / len(allh)
    s_sub = np.bincount(sub[allh], minlength=sub.max() + 1) / len(allh)
    prof_cache: dict = {}
    n_imp = len(beh)
    prof = np.zeros((n_imp, E.shape[1]), np.float32)
    hlen = np.zeros(n_imp)
    cat_counts = []
    for i, hl in enumerate(hist_lists):
        key = (beh["part"].iat[i], beh["user_id"].iat[i], len(hl))
        if key not in prof_cache:
            if hl:
                v = E[hl].mean(0)
                v /= max(np.linalg.norm(v), 1e-9)
            else:
                v = np.zeros(E.shape[1], np.float32)
            prof_cache[key] = (v, np.bincount(cat[hl], minlength=cat.max() + 1) if hl else None,
                               np.bincount(sub[hl], minlength=sub.max() + 1) if hl else None)
        prof[i] = prof_cache[key][0]
        hlen[i] = len(hl)
        cat_counts.append(key)
    imp = cand["imp"].to_numpy()
    tc = np.einsum("ij,ij->i", E[n_], prof[imp])
    df = pd.DataFrame({"imp": imp, "news": n_, "y": cand["y"].to_numpy(), "hour": hour})
    df["text_cos"] = tc
    g = df.groupby("imp")["text_cos"]
    mu, sd = g.transform("mean").to_numpy(), g.transform("std").fillna(0).to_numpy()
    df["text_cos_z"] = np.where(sd > 1e-9, (tc - mu) / np.where(sd > 1e-9, sd, 1), 0.0)
    cs = np.empty(len(df))
    ss = np.empty(len(df))
    ccat, csub = cat[n_], sub[n_]
    # category shares: the impression's history counts looked up per candidate (candidate rows sorted by impression)
    keys = [cat_counts[i] for i in range(n_imp)]
    start = np.searchsorted(imp, np.arange(n_imp))
    end = np.searchsorted(imp, np.arange(n_imp), side="right")
    for i in range(n_imp):
        a, b = start[i], end[i]
        if a == b:
            continue
        _, kc, ks = prof_cache[keys[i]]
        L = hlen[i]
        if kc is None:
            cs[a:b] = s_cat[ccat[a:b]]
            ss[a:b] = s_sub[csub[a:b]]
        else:
            cs[a:b] = (kc[ccat[a:b]] + s_cat[ccat[a:b]]) / (L + 1)
            ss[a:b] = (ks[csub[a:b]] + s_sub[csub[a:b]]) / (L + 1)
    df["cat_share"] = cs
    df["subcat_share"] = ss
    df["log_hist_len"] = np.log1p(hlen[imp])
    df["pop_ctr_24h"] = (c24 + 10 * prior_rate) / (i24 + 10)
    df["pop_log_clicks_per_1k"] = np.log1p(1000 * c24 / np.maximum(active24[hour], 1))
    df["age_log_hours"] = np.log1p(np.maximum(hour - first_seen[n_], 0))
    df["slate_size_log"] = np.log1p(df.groupby("imp")["imp"].transform("size").to_numpy())
    df["user_id"] = beh["user_id"].to_numpy()[imp]
    df["part"] = beh["part"].to_numpy()[imp]
    df["impression_id"] = beh["impression_id"].to_numpy()[imp]
    meta = {"prior_rate": prior_rate, "n_news": N, "embedding_dim": int(E.shape[1]),
            "empty_history_impressions": int((hlen == 0).sum()),
            "news_without_abstract": int(news["abstract"].isna().sum())}
    return df, meta


def news_masks(df: pd.DataFrame, beh: pd.DataFrame) -> dict:
    """The prespecified splits of news_click (plan: models.news_click.splits)."""
    train_users = set(beh.loc[beh["part"] == "train", "user_id"])
    ub = df["user_id"].map({u: _bucket(u) for u in df["user_id"].unique()}).to_numpy()
    part = df["part"].to_numpy()
    day = df["hour"].to_numpy() // 24                    # 0 = 2019-11-09
    fit_pool = (part == "train") & (ub < 80) & (day >= 1) & (day <= 4)
    imps = np.unique(df.loc[fit_pool, "imp"].to_numpy())
    rng = np.random.default_rng(0)
    chosen = np.sort(rng.choice(imps, size=min(50000, len(imps)), replace=False))
    m = {"train": fit_pool & np.isin(df["imp"].to_numpy(), chosen),
         "val": (part == "train") & (ub >= 80) & (day == 5),
         "test": part == "dev"}
    new_user = ~df["user_id"].isin(train_users).to_numpy()
    m["test_new_users"] = m["test"] & new_user
    m["test_returning_users"] = m["test"] & ~new_user
    return m


def fit_news(plan_path: str, version: int, cmd: str, t0: float) -> list[str]:
    data = load()
    df, meta = build(data)
    print(f"news features: {len(df)} candidate rows in {time.time() - t0:.0f} s", flush=True)
    m = news_masks(df, data["beh"])
    X = df[FEATURES].to_numpy(float)
    y = df["y"].to_numpy(float)
    pop_idx = [FEATURES.index(f) for f in POP_FEATURES]
    Xs = {k: X[v] for k, v in m.items()}
    ys = {k: y[v] for k, v in m.items()}
    gs = {k: df.loc[v, "imp"].to_numpy() for k, v in m.items()}
    b0 = F.fit_constant(ys["train"])
    b1 = F.fit_logistic(Xs["train"][:, pop_idx], ys["train"])
    m1 = F.fit_logistic(Xs["train"], ys["train"])
    m2 = F.fit_gbdt(Xs["train"], ys["train"])
    class Sub:
        def __init__(self, base, idx):
            self.base, self.idx = base, idx

        def predict(self, Z):
            return self.base.predict(np.asarray(Z)[:, self.idx])

    def report(pred, split):
        p = pred.predict(Xs[split])
        r = M.binary_report(ys[split], p)
        r.update(M.group_ranking(ys[split], p, gs[split]))
        return r

    preds = {"B0_constant": (b0, "baseline: constant", 1), "B1_popularity": (Sub(b1, pop_idx),
             "baseline: popularity logistic (3 features)", len(pop_idx) + 1),
             "M1_logistic": (m1, "L2 logistic, 9 features", F.n_params(m1)),
             "M2_gbdt": (m2, f"GBDT, {m2.n_iter} trees x 31 leaves", F.n_params(m2))}
    cands = {k: {"complexity": c, "n_params": n, "pred": p, "val": report(p, "val"), "test": report(p, "test")}
             for k, (p, c, n) in preds.items()}
    sel = F.select(cands, "log_loss", ["B0_constant", "B1_popularity"],
                   complexity_order=["B0_constant", "B1_popularity", "M1_logistic", "M2_gbdt"])
    kp = cands[sel["kept"]]["pred"]
    bp = cands[sel["best_baseline_on_test"]]["pred"]
    groups = df.loc[m["test"], "user_id"].to_numpy()
    pk, pb = kp.predict(Xs["test"]), bp.predict(Xs["test"])
    lk, lb = M.row_log_loss(ys["test"], pk), M.row_log_loss(ys["test"], pb)
    ci_k, ci_d = M.cluster_bootstrap(lk, groups), M.cluster_bootstrap(lk - lb, groups)

    def info(mask, d):
        return {"definition": d, "rows": int(mask.sum()), "users": int(df.loc[mask, "user_id"].nunique()),
                "impressions": int(df.loc[mask, "imp"].nunique())}

    rep = {
        "schema": "crucible.v5.calibration_report/1", "model": "news_click", "version": version,
        "domain": "content_selection", "datasets": ["mind_small"], "identification": "observational",
        "estimand": "P(click | news shown in an impression, the user's click history, text, category, popularity, age)",
        "population": "50,000 sampled Microsoft News users (English, US), anonymised",
        "platform": "Microsoft News (msn.com) news feed", "period": "2019-11-09 to 2019-11-15",
        "plan": F.plan_record(plan_path),
        "splits": {"train": info(m["train"], "50,000 train-file impressions of fit users, 2019-11-10..13 (seed 0)"),
                   "val": info(m["val"], "train-file impressions of validation users, 2019-11-14"),
                   "test": info(m["test"], "all dev-file impressions, 2019-11-15 (later day)"),
                   "test_new_users": info(m["test_new_users"], "dev impressions of users absent from the train file"),
                   "test_returning_users": info(m["test_returning_users"], "dev impressions of train-file users")},
        "heldout_splits": ["test", "test_new_users", "test_returning_users"],
        "candidates": F.strip(cands), "selection": sel,
        "baseline_comparison": {"metric": "log_loss", "kept": sel["kept"], "baseline": sel["best_baseline_on_test"],
                                "kept_test": sel["test"][sel["kept"]],
                                "baseline_test": sel["test"][sel["best_baseline_on_test"]],
                                "difference": sel["test"][sel["kept"]] - sel["test"][sel["best_baseline_on_test"]],
                                "ci95": [ci_d["lo"], ci_d["hi"]], "kept_ci95": [ci_k["lo"], ci_k["hi"]],
                                "bootstrap": {"resamples": ci_d["n_boot"], "clusters": ci_d["clusters"]}},
        "calibration": {"ece": M.ece(ys["test"], pk), "brier": M.brier(ys["test"], pk),
                        "mean_pred": float(pk.mean()), "mean_obs": float(ys["test"].mean()),
                        "reliability": M.calibration_table(ys["test"], pk)},
        "heldout_extra": {k: {c: report(cands[c]["pred"], k) for c in cands}
                          for k in ("test_new_users", "test_returning_users")},
        "missing_data": meta,
        "identification_note": ("Observational: MIND records the production recommender's impressions; which news "
                                "were shown depended on the user, so click probabilities are conditional on that "
                                "selection and the feature effects are associations, not effects of exposure."),
        "extrapolation_limits": LIMITS, "feature_map": FEATURE_MAP,
        "fit": {"commit": F.head_commit(), "seconds": time.time() - t0, "command": cmd},
    }
    kept_pred = kp
    if isinstance(kp, Sub):
        from . import predictors as P
        w = np.zeros(len(FEATURES))
        mean = np.zeros(len(FEATURES))
        scale = np.ones(len(FEATURES))
        for j, i in enumerate(pop_idx):
            w[i], mean[i], scale[i] = kp.base.w[j], kp.base.mean[j], kp.base.scale[j]
        kept_pred = P.Logistic(w, kp.base.b, mean, scale)
    return [F.finish("news_click", version, "content_selection", kept_pred, FEATURES,
                     "clicked: the user opened the news item", rep,
                     {"prior_rate": meta["prior_rate"], "feature_quantiles": F.feature_quantiles(Xs["train"], FEATURES),
                      "text_representation": json.loads(json.dumps(
                          {"method": "TF-IDF (sublinear, English stop words, min_df 3, <= 50k terms) of title + "
                                     "abstract fitted on train-file news -> truncated SVD 128 (seed 0) -> L2 norm",
                           "profile": "normalised mean of history vectors"})),
                      "mechanisms": ["population.article_choice"]})]
