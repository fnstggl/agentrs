"""Fit and evaluate the software-adoption models on GH Archive (workstream E2).

Follows docs/v5/calibration/plans/E2_software_adoption.plan.json, committed before this script first ran:

* software_deep_engagement@1 - P(fork / issue / pull request by the same developer on the same repository within
  24 h | the developer's first star on it); time-based holdout (train D1, validation D2 00-12, test D2 12-24),
  user-level holdout subset, strict-time refit check;
* software_repo_growth@1 - next-day stars of a repository given today's activity; train/validation on D1->D2
  (repository-level split), time-based test on D2->D3, repository-level holdout subset, NB2 dispersion and
  distributional checks;
* the new-repository adoption curve (descriptive, not fitted).

The evaluated tables are written to $CRUCIBLE_DATA/<dataset>/tables_v1.npz (not committed; digests in the reports).

    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "ws-E2: fit software" -- \
        python -m crucible.calibration.fit_software
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from . import gharchive as G
from . import portable
from . import scoring_e2 as S
from .fitmeta_e2 import code_state, digest, environment, now, plan_ref, write_card, write_checkpoint, write_report
from .gbdt_e2 import truncate_to_best
from .registry_e2 import REPORTS_DIR, load as registry_load, rel
from .software import DEEP_FEATURES, GROWTH_FEATURES, transform

PLAN = "E2_software_adoption.plan.json"
AMENDMENTS = ("E2_amendment_1.plan.json",)
TABLES_VERSION = "2"        # 2: built from pseudonymised events (gharchive extractor 2)
N_BOOT = 200
DAY = 86400
WINDOW_S = 24 * 3600
DEEP_TYPES = ("ForkEvent", "IssuesEvent", "IssueCommentEvent", "PullRequestEvent", "PullRequestReviewEvent",
              "PullRequestReviewCommentEvent")
SPLIT_HOURS = {"train": (24, 48), "validation": (48, 60), "test": (60, 72), "strict_train": (24, 36)}
REPO_SPLIT_SEED = 20261125


def tables_path(dest: str | None = None) -> str:
    return os.path.join(G.dataset_dir(dest), f"tables_v{TABLES_VERSION}.npz")


def splitmix64(x: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


# ----------------------------------------------------------------------------------------------- tables

def build_tables(ev: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Deterministic feature/label tables for both models (and the descriptive curve) from the event table."""
    C = G.TYPE_CODE
    typ = ev["type"]; flags = ev["flags"]; t = ev["t"].astype(np.int64); pos = ev["pos"]
    bot = (flags & G.FLAG_BOT) != 0; owner = (flags & G.FLAG_OWNER) != 0; org = (flags & G.FLAG_ORG) != 0
    hour = np.clip(t // 3600, 0, G.HOURS - 1)
    day = np.clip(t // DAY, 0, G.HOURS // 24 - 1)
    repo_ids, repo_first, repo = np.unique(ev["repo"], return_index=True, return_inverse=True); R = repo_ids.size
    _, actor = np.unique(ev["actor"], return_inverse=True)
    _, pair = np.unique(actor.astype(np.int64) * R + repo, return_inverse=True)
    repo = repo.astype(np.int64); actor = actor.astype(np.int64); pair = pair.astype(np.int64)
    isin = lambda names: np.isin(typ, [C[n] for n in names])
    grp = {"stars": (typ == C["WatchEvent"]) & ~bot, "forks": (typ == C["ForkEvent"]) & ~bot,
           "issue_events": isin(("IssuesEvent", "IssueCommentEvent")),
           "pr_events": isin(("PullRequestEvent", "PullRequestReviewEvent", "PullRequestReviewCommentEvent")),
           "pushes": typ == C["PushEvent"], "releases": typ == C["ReleaseEvent"], "events": np.ones(t.size, bool)}

    def windowed(ent, mask, eq, hq, width=24):
        keys = np.sort(ent[mask] * 128 + hour[mask])
        lo = eq * 128 + np.maximum(hq - width, 0); hi = eq * 128 + hq
        return (np.searchsorted(keys, hi, "left") - np.searchsorted(keys, lo, "left")).astype(np.float64)

    create = (typ == C["CreateEvent"]) & ((flags & G.FLAG_REPO_CREATE) != 0)
    created_t = np.full(R, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(created_t, repo[create], t[create])

    # ---- deep-engagement units: first star per (actor, repo) by a non-bot, non-owner actor
    idx_star = np.flatnonzero((typ == C["WatchEvent"]) & ~bot & ~owner)
    _, first = np.unique(pair[idx_star], return_index=True)
    u = idx_star[np.sort(first)]
    u = u[(t[u] >= 24 * 3600) & (t[u] < 72 * 3600)]
    rank = np.empty(t.size, dtype=np.int64); rank[np.lexsort((pos, t))] = np.arange(t.size)

    def label(names):
        md = np.flatnonzero(isin(names) & ~bot)
        comp = pair[md] * (1 << 26) + rank[md]
        o = np.argsort(comp, kind="stable"); comp = comp[o]; td = t[md][o]
        q = pair[u] * (1 << 26) + rank[u]
        j = np.searchsorted(comp, q, "right")
        jj = np.minimum(j, comp.size - 1)
        same = (j < comp.size) & ((comp[jj] >> 26) == pair[u])
        return (same & (td[jj] <= t[u] + WINDOW_S)).astype(np.int8)

    H = hour[u]
    raw = {"repo_stars_24h": windowed(repo, grp["stars"], repo[u], H), "repo_forks_24h": windowed(repo, grp["forks"], repo[u], H),
           "repo_issue_events_24h": windowed(repo, grp["issue_events"], repo[u], H),
           "repo_pr_events_24h": windowed(repo, grp["pr_events"], repo[u], H),
           "repo_pushes_24h": windowed(repo, grp["pushes"], repo[u], H),
           "repo_releases_24h": windowed(repo, grp["releases"], repo[u], H),
           "repo_events_24h": windowed(repo, grp["events"], repo[u], H),
           "repo_org": org[u].astype(np.float64)}
    ct = created_t[repo[u]]
    made = ct < t[u]
    raw["repo_created_in_window"] = made.astype(np.float64)
    raw["repo_age_hours"] = np.where(made, (t[u] - np.where(made, ct, 0)) / 3600.0, 0.0)
    for name, key in (("actor_events_24h", "events"), ("actor_stars_24h", "stars"), ("actor_forks_24h", "forks"),
                      ("actor_issue_events_24h", "issue_events"), ("actor_pr_events_24h", "pr_events"),
                      ("actor_pushes_24h", "pushes")):
        raw[name] = windowed(actor, grp[key], actor[u], H)
    raw["actor_repo_events_24h"] = windowed(pair, grp["events"], pair[u], H)
    hod = (t[u] % DAY) / 3600.0
    raw["hour_sin"] = np.sin(2 * np.pi * hod / 24); raw["hour_cos"] = np.cos(2 * np.pi * hod / 24)
    out = {"deep/X": transform(DEEP_FEATURES, raw), "deep/y": label(DEEP_TYPES),
           "deep/y_fork": label(("ForkEvent",)), "deep/y_issue": label(("IssuesEvent", "IssueCommentEvent")),
           "deep/y_pr": label(("PullRequestEvent", "PullRequestReviewEvent", "PullRequestReviewCommentEvent")),
           "deep/t": t[u], "deep/actor": actor[u]}

    # ---- repository growth units
    nd = G.HOURS // 24
    cnt = {k: np.bincount(repo[m] * nd + day[m], minlength=R * nd).reshape(R, nd).astype(np.float64) for k, m in grp.items()}
    org_repo = np.zeros(R, bool); org_repo[repo[org]] = True
    cday = np.where(created_t < np.iinfo(np.int64).max, created_t // DAY, -1)
    # repository split bucket, computed from the raw id at extraction (gharchive extractor 2 keeps no raw id)
    hashed = np.asarray(ev["repo_bucket"])[repo_first].astype(np.uint64)
    for d in (1, 2):
        m = ((cnt["stars"][:, d] + cnt["forks"][:, d]) >= 1) | (cday == d)
        r = np.flatnonzero(m)
        rawg = {"stars_today": cnt["stars"][r, d], "forks_today": cnt["forks"][r, d],
                "issue_events_today": cnt["issue_events"][r, d], "pr_events_today": cnt["pr_events"][r, d],
                "pushes_today": cnt["pushes"][r, d], "releases_today": cnt["releases"][r, d],
                "events_today": cnt["events"][r, d], "stars_yesterday": cnt["stars"][r, d - 1],
                "created_today": (cday[r] == d).astype(np.float64), "created_yesterday": (cday[r] == d - 1).astype(np.float64),
                "repo_org": org_repo[r].astype(np.float64)}
        out[f"growth/d{d}/X"] = transform(GROWTH_FEATURES, rawg)
        out[f"growth/d{d}/y"] = cnt["stars"][r, d + 1]
        out[f"growth/d{d}/repo"] = r
        out[f"growth/d{d}/train_bucket"] = (hashed[r] < np.uint64(80)).astype(np.int8)

    # ---- descriptive: adoption of repositories created on D1 (non-owner, non-bot stars, 48 h)
    newr = np.flatnonzero(cday == 1)
    ms = grp["stars"] & ~owner & (cday[repo] == 1)
    dt = t[ms] - created_t[repo[ms]]
    ok = (dt > 0) & (dt <= 48 * 3600)
    rr = repo[ms][ok]; hh = np.ceil(dt[ok] / 3600.0).astype(np.int64)
    out["curve/n_new_repos"] = np.array([newr.size])
    out["curve/stars_by_hour"] = np.bincount(hh, minlength=49)[1:49].astype(np.float64)
    per = np.zeros(R); np.add.at(per, rr[hh <= 24], 1); s24 = per[newr]
    per48 = np.zeros(R); np.add.at(per48, rr, 1); s48 = per48[newr]
    out["curve/stars_24h"] = s24; out["curve/stars_48h"] = s48
    return out


def table_digests(tab: dict[str, np.ndarray]) -> dict[str, str]:
    return {k: digest(np.ascontiguousarray(v)) for k, v in sorted(tab.items())}


# ----------------------------------------------------------------------------------------------- fitting helpers

def _hgb_cls(Xtr, ytr, Xva, yva, threads):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    with threadpool_limits(threads):
        m = HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
                                           max_iter=300, early_stopping=True, n_iter_no_change=20, scoring="loss",
                                           random_state=0).fit(Xtr, ytr, X_val=Xva, y_val=yva)
    return m, truncate_to_best(m)          # amendment 1: keep the best validation iteration


GROWTH_GBDT = {   # gbdt_poisson as prespecified; gbdt_poisson_reg added by amendment 1
    "gbdt_poisson": {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 100, "max_iter": 300,
                     "n_iter_no_change": 20},
    "gbdt_poisson_reg": {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 200, "l2_regularization": 10.0,
                         "max_iter": 500, "n_iter_no_change": 30},
}


def fit_deep(cand, Xtr, ytr, Xva, yva, threads):
    from threadpoolctl import threadpool_limits
    if cand == "constant_rate":
        p = float(ytr.mean())
        return portable.export_constant(p, "identity"), (lambda X: np.full(len(X), p)), {"rate": p}
    if cand == "logistic":
        from sklearn.linear_model import LogisticRegression
        mean = Xtr.mean(0); scale = Xtr.std(0); scale[scale == 0] = 1.0
        with threadpool_limits(threads):
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-7).fit((Xtr - mean) / scale, ytr)
        return (portable.export_linear(lr.coef_, lr.intercept_, mean, scale, "logit"),
                lambda X: lr.predict_proba((X - mean) / scale)[:, 1], {"n_iter": int(lr.n_iter_[0])})
    if cand == "gbdt":
        m, es = _hgb_cls(Xtr, ytr, Xva, yva, threads)
        return portable.export_hgb(m, "logit"), (lambda X: m.predict_proba(X)[:, 1]), {"n_iter": int(m.n_iter_), "early_stopping": es}
    raise ValueError(cand)


def fit_growth(cand, Xtr, ytr, Xva, yva, threads):
    from threadpoolctl import threadpool_limits
    if cand == "proportional":
        from scipy.optimize import minimize
        s = np.expm1(Xtr[:, 0]); y = ytr

        def f(ab):
            mu = ab[0] * s + ab[1]
            return float(np.mean(mu - y * np.log(mu))), np.array([np.mean((1 - y / mu) * s), np.mean(1 - y / mu)])
        res = minimize(f, x0=np.array([max(y.mean() / max(s.mean(), 1e-9) * 0.5, 1e-3), 0.1]), jac=True,
                       method="L-BFGS-B", bounds=[(0.0, None), (1e-6, None)])
        a, b = map(float, res.x)
        return portable.export_proportional(a, b, 0), (lambda X: a * np.expm1(X[:, 0]) + b), {"a": a, "b": b, "converged": bool(res.success)}
    if cand == "poisson_glm":
        from sklearn.linear_model import PoissonRegressor
        mean = Xtr.mean(0); scale = Xtr.std(0); scale[scale == 0] = 1.0
        with threadpool_limits(threads):
            pr = PoissonRegressor(alpha=1e-4, max_iter=3000, tol=1e-8).fit((Xtr - mean) / scale, ytr)
        return (portable.export_linear(pr.coef_, [pr.intercept_], mean, scale, "log"),
                lambda X: pr.predict((X - mean) / scale), {"n_iter": int(pr.n_iter_)})
    if cand in GROWTH_GBDT:
        from sklearn.ensemble import HistGradientBoostingRegressor
        with threadpool_limits(threads):
            m = HistGradientBoostingRegressor(loss="poisson", **GROWTH_GBDT[cand], early_stopping=True, scoring="loss",
                                              random_state=0).fit(Xtr, ytr, X_val=Xva, y_val=yva)
        es = truncate_to_best(m)
        return portable.export_hgb(m, "log"), (lambda X: m.predict(X)), {"n_iter": int(m.n_iter_), "early_stopping": es}
    raise ValueError(cand)


def deep_metrics(y, p) -> dict:
    return {**S.binary_report(y, p), "average_precision": S.average_precision(y, p),
            "reliability_by_decile": S.reliability(y, p, 10)}


def growth_metrics(y, mu) -> dict:
    return {"n": int(y.size), "mean_poisson_deviance": S.poisson_deviance(y, mu), "mae": float(np.mean(np.abs(y - mu))),
            "rmse_log1p": float(np.sqrt(np.mean((np.log1p(y) - np.log1p(mu)) ** 2))), "spearman": S.spearman(y, mu),
            "top_1pct_recall": S.top_recall(y, mu, 0.01), "mean_predicted": float(mu.mean()), "mean_observed": float(y.mean()),
            "calibration_by_decile": S.count_calibration(y, mu, 10)}


def select(results, rows, cands, complexity, key):
    kept = min(cands, key=lambda c: (results[c]["validation"][key], complexity[c]))
    comps = []
    for other in cands:
        if other == kept:
            continue
        bs = S.paired_bootstrap_mean_diff(rows[(kept, "test")] - rows[(other, "test")], n_boot=N_BOOT, seed=1)
        comps.append({"kept": kept, "other": other, "metric": f"{key} (test)", "kept_minus_other": bs,
                      "kept_significantly_worse": bool(bs["ci95"][0] > 0),
                      "validation_kept_minus_other": results[kept]["validation"][key] - results[other]["validation"][key]})
    return kept, comps, int(sum(c["kept_significantly_worse"] for c in comps))


POPULATION = ("public GitHub accounts and public repositories whose events reached GitHub's public events API and "
              "GH Archive between 2024-10-07 and 2024-10-10 UTC (Monday to Thursday); bot accounts excluded from units")
COMMON_LIMITS = [
    "Population: public GitHub, 2024-10-07..10 UTC (four weekdays of one autumn week). Developers on GitHub are skewed towards professional, English-speaking and North-American/European/Asian time zones; private repositories and private activity are invisible.",
    "GH Archive records the public events API, which drops events under load and omits some activity; counts are lower bounds.",
    "Stars are cheap signals inflated by automation and star farming; a star is not evidence of useful adoption (MASTER_SPEC 6.5). These models describe engagement dynamics, not usefulness.",
    "Observational associations only: nothing was randomized, so the models do not say what promotion, ranking or buying stars would cause.",
    "Short horizon: features look back 24 h and outcomes forward 24 h; long-run adoption, retention, maintenance and dependency incorporation are not observed.",
    "Bots are identified only by the '[bot]' login suffix; other automation accounts remain in the data.",
]


def main(threads: int = 4) -> dict:
    t_start = time.time()
    started_at = now()
    state = code_state()
    plan = plan_ref(PLAN)
    amendments = [plan_ref(a) for a in AMENDMENTS]
    ev = G.load_events()
    print("events", ev["t"].size, flush=True)
    t0 = time.time()
    tab = build_tables(ev)
    build_s = round(time.time() - t0, 1)
    del ev
    np.savez(tables_path(), **tab)
    digests = table_digests(tab)
    print("tables built", build_s, "s", flush=True)
    manifest = {"dataset": G.DATASET, "dataset_manifest": "docs/v5/calibration/datasets/gharchive_2024-10-07_96h.json"}
    reports = {}

    # =========================================================== deep engagement
    tt = tab["deep/t"]
    sp = {s: (tt >= a * 3600) & (tt < b * 3600) for s, (a, b) in SPLIT_HOURS.items()}
    X, y = tab["deep/X"], tab["deep/y"].astype(np.float64)
    seen_actors = np.unique(tab["deep/actor"][sp["train"] | sp["validation"]])
    new_user = np.isin(tab["deep/actor"], seen_actors, invert=True) & sp["test"]
    cands = ("constant_rate", "logistic", "gbdt"); complexity = {"constant_rate": 0, "logistic": 1, "gbdt": 2}
    arrays, fit_info, preds = {}, {}, {}
    for c in cands:
        t1 = time.time()
        arr, ref, info = fit_deep(c, X[sp["train"]], y[sp["train"]], X[sp["validation"]], y[sp["validation"]], threads)
        for s in ("validation", "test"):
            preds[(c, s)] = ref(X[sp[s]])
        diff = portable.check_equal(arr, preds[(c, "test")], X[sp["test"]])
        if diff > 1e-9:
            raise RuntimeError(f"portable export of {c} differs by {diff}")
        arrays.update(portable.pack(f"cand/{c}", arr))
        fit_info[c] = {**info, "wall_s": round(time.time() - t1, 1), "portable_max_abs_diff": diff}
        print("deep", c, fit_info[c], flush=True)
    results, rows = {}, {}
    for c in cands:
        results[c] = {s: deep_metrics(y[sp[s]], preds[(c, s)]) for s in ("validation", "test")}
        p_test = preds[(c, "test")]
        results[c]["test_new_users"] = deep_metrics(y[new_user], p_test[new_user[sp["test"]]])
        rows[(c, "test")] = S.log_loss_rows(y[sp["test"]], p_test)
    kept, comps, worse = select(results, rows, cands, complexity, "log_loss")
    arrays.update(portable.pack("model", portable.unpack(f"cand/{kept}", arrays)))
    # strict-time check: the kept class refitted on D1 00-12 only (all label windows close before the test period)
    arr_s, ref_s, info_s = fit_deep(kept, X[sp["strict_train"]], y[sp["strict_train"]], X[sp["validation"]], y[sp["validation"]], threads)
    strict = {"train_hours": SPLIT_HOURS["strict_train"], "n_train": int(sp["strict_train"].sum()), **info_s,
              "test": deep_metrics(y[sp["test"]], ref_s(X[sp["test"]]))}
    probe = np.flatnonzero(sp["test"])[:256]
    arrays["probe/X"] = X[probe]; arrays["probe/pred"] = preds[(kept, "test")][:256]
    name = "software_deep_engagement"; mid = f"{name}@1"
    ckpt = write_checkpoint(name, arrays)
    label_rates = {k: {s: float(tab[f"deep/{k}"][sp[s]].mean()) for s in ("train", "validation", "test")}
                   for k in ("y", "y_fork", "y_issue", "y_pr")}

    def dsum(m):
        return {k: m[k] for k in ("log_loss", "brier", "ece", "auc", "average_precision", "mean_predicted", "mean_observed", "n")}
    limits = [*COMMON_LIMITS,
              "Deep engagement is measured only as public fork / issue / pull-request events by the same account on the same repository within 24 h of the first star; later engagement, use without public events (cloning, installing, running) and engagement through other accounts are invisible.",
              "Stars by repository owners (login equals owner) and by '[bot]' accounts are excluded; only the first star per developer-repository pair in the window is a unit."]
    card = {
        "name": name, "version": 1, "id": mid, "workstream": "E2", "domain": "software_collaboration_and_adoption",
        "loader": "crucible.calibration.software:DeepEngagementModel",
        "features": [n for n, _, _ in DEEP_FEATURES],
        "feature_docs": {n: f"{d} [{tf}]" for n, tf, d in DEEP_FEATURES},
        "target": "deeper engagement (fork, issue or issue comment, pull request or review) by the same developer on the same repository within 24 h of the developer's first star",
        "outputs": ["p_deep_engagement_24h"],
        "identification": {"kind": "observational", "statement": "Associations in public GitHub activity; nothing was randomized."},
        "population": POPULATION,
        "dataset": {"name": G.DATASET, "synthetic": False, "url": G.BASE_URL, **manifest},
        "checkpoint": ckpt, "report": rel(os.path.join(REPORTS_DIR, f"{mid}.json")), "plan": plan, "plan_amendments": amendments,
        "fit": {"started_at": started_at, "code": "src/crucible/calibration/fit_software.py", **state,
                "command": "python -m crucible.calibration.fit_software", "threads": threads, "environment": environment(),
                "split": {k: list(v) for k, v in SPLIT_HOURS.items()}, "counts": {s: int(m.sum()) for s, m in sp.items()},
                "sub_models": fit_info, "tables": {"path": "$CRUCIBLE_DATA/" + G.DATASET + f"/tables_v{TABLES_VERSION}.npz", "build_s": build_s}},
        "metrics": dsum(results[kept]["test"]), "baseline_metrics": dsum(results["constant_rate"]["test"]),
        "candidates": {c: {"complexity": complexity[c], "validation": dsum(results[c]["validation"]),
                           "test": dsum(results[c]["test"]), "test_new_users": dsum(results[c]["test_new_users"])} for c in cands},
        "selection": {"rule": "lowest validation log loss; confirmed on test by paired bootstrap", "kept": kept,
                      "baseline": "constant_rate", "comparisons": comps, "worse_model_kept": worse},
        "calibration_metrics": ["log_loss", "brier", "ece_15_equal_mass_bins", "auc", "average_precision", "reliability_by_decile"],
        "limits": " ".join(limits), "extrapolation_limits": limits,
        "simulator_mapping": {
            "unit": "a developer (population individual or agent) starring/watching a repository in the software ecosystem for the first time",
            "repo_stars_24h": "star/watch actions on the repository in the previous 24 simulated hours",
            "repo_forks_24h": "code.fork calls on the repository in that window",
            "repo_issue_events_24h": "issue tracker events (issues.create, issues.comment, issues.close) on the repository's project",
            "repo_pr_events_24h": "pull-request / review events on the repository",
            "repo_pushes_24h": "code.commit calls (one push per commit batch)",
            "repo_releases_24h": "package or repository releases",
            "repo_events_24h": "all of the above plus any other public repository event",
            "repo_org": "repository owned by an organisation principal",
            "repo_created_in_window": "repository created by code.create_repo before the star (always observable in the simulator; in the data only creations inside the window were seen)",
            "repo_age_hours": "simulated hours since creation (the data only covered ages up to about 72 h; cap at 72 when using)",
            "actor_*": "the same counts for the developer's own actions in the previous 24 simulated hours",
            "hour_sin/hour_cos": "simulated hour of day",
            "output": "probability that the developer forks, files/comments an issue or opens/reviews a pull request on that repository within 24 simulated hours",
            "transform": "matrix_from_counts(raw) applies log1p to count features",
        },
        "fidelity_mechanisms": ["developer_light_to_deep_engagement"],
        "provenance": {"dataset": G.DATASET, "source": G.BASE_URL, "license": G.LICENSE, "fit_commit": state["commit"],
                       "plan_commit": plan["commit"]},
    }
    write_card(name, card)
    model = registry_load(mid)
    pt = model.predict(X[sp["test"]])
    audit = {"rows": "all test rows", "n": int(sp["test"].sum()), "max_abs_diff_vs_estimator": float(np.max(np.abs(pt - preds[(kept, "test")]))),
             "log_loss": S.log_loss(y[sp["test"]], pt), "baseline_log_loss": S.log_loss(y[sp["test"]], model.candidate_predict("constant_rate", X[sp["test"]]))}
    reports[mid] = {
        "schema": "crucible.v5.calibration.report/1", "model": mid, "plan": plan, "plan_amendments": amendments, "dataset": card["dataset"],
        "identification": card["identification"], "population": POPULATION,
        "split": card["fit"]["split"], "counts": card["fit"]["counts"], "label_rates": label_rates,
        "heldout": {"validation": {c: results[c]["validation"] for c in cands}, "test": {c: results[c]["test"] for c in cands},
                    "test_new_users": {c: results[c]["test_new_users"] for c in cands}},
        "selection": card["selection"], "strict_time_check": strict,
        "calibration_metrics": {c: {"test": {k: results[c]["test"][k] for k in ("log_loss", "brier", "ece", "reliability_by_decile")}} for c in cands},
        "audit": audit, "tables": {"digests": {k: v for k, v in digests.items() if k.startswith("deep/")}, "version": TABLES_VERSION},
        "fit_info": fit_info, "extrapolation_limits": limits,
    }

    # =========================================================== repository growth
    X1, y1 = tab["growth/d1/X"], tab["growth/d1/y"]; X2, y2 = tab["growth/d2/X"], tab["growth/d2/y"]
    b1 = tab["growth/d1/train_bucket"].astype(bool)
    gX = {"train": X1[b1], "validation": X1[~b1], "test": X2}; gy = {"train": y1[b1], "validation": y1[~b1], "test": y2}
    new_repo = np.isin(tab["growth/d2/repo"], tab["growth/d1/repo"][b1], invert=True)
    gc = ("proportional", "poisson_glm", "gbdt_poisson", "gbdt_poisson_reg")
    gcx = {"proportional": 0, "poisson_glm": 1, "gbdt_poisson": 2, "gbdt_poisson_reg": 2}
    garr, ginfo, gpred = {}, {}, {}
    for c in gc:
        t1 = time.time()
        arr, ref, info = fit_growth(c, gX["train"], gy["train"], gX["validation"], gy["validation"], threads)
        for s in ("train", "validation", "test"):
            gpred[(c, s)] = ref(gX[s])
        diff = float(np.max(np.abs(portable.predict(arr, gX["test"]) - gpred[(c, "test")]) / np.maximum(1.0, gpred[(c, "test")])))
        if diff > 1e-9:
            raise RuntimeError(f"portable export of {c} differs by {diff} (relative)")
        garr.update(portable.pack(f"cand/{c}", arr))
        ginfo[c] = {**info, "wall_s": round(time.time() - t1, 1), "portable_max_rel_diff": diff}
        print("growth", c, ginfo[c], flush=True)
    gres, grows = {}, {}
    for c in gc:
        gres[c] = {s: growth_metrics(gy[s], gpred[(c, s)]) for s in ("validation", "test")}
        gres[c]["test_new_repos"] = growth_metrics(gy["test"][new_repo], gpred[(c, "test")][new_repo])
        grows[(c, "test")] = S.poisson_deviance_rows(gy["test"], gpred[(c, "test")])
    gkept, gcomps, gworse = select(gres, grows, gc, gcx, "mean_poisson_deviance")
    garr.update(portable.pack("model", portable.unpack(f"cand/{gkept}", garr)))
    alpha = S.fit_nb_alpha(gy["train"], gpred[(gkept, "train")])
    garr["nb_alpha"] = np.array([alpha])
    rng = np.random.default_rng(5)
    sim = S.sample_nb(rng, gpred[(gkept, "test")], alpha)
    sim_pois = rng.poisson(np.maximum(gpred[(gkept, "test")], 1e-12))
    dist = {"observed": S.count_distribution(gy["test"]), "simulated_nb": S.count_distribution(sim),
            "simulated_poisson": S.count_distribution(sim_pois), "nb_alpha": alpha,
            "ks_log1p_nb": S.ks_distance(np.log1p(sim), np.log1p(gy["test"])),
            "ks_log1p_poisson": S.ks_distance(np.log1p(sim_pois), np.log1p(gy["test"])),
            "test_nb_loglik": S.nb_loglik(gy["test"], gpred[(gkept, "test")], alpha)}
    probe = np.arange(min(256, gX["test"].shape[0]))
    garr["probe/X"] = gX["test"][probe]; garr["probe/pred"] = gpred[(gkept, "test")][probe]
    gname = "software_repo_growth"; gmid = f"{gname}@1"
    gckpt = write_checkpoint(gname, garr)

    def gsum(m):
        return {k: m[k] for k in ("mean_poisson_deviance", "mae", "rmse_log1p", "spearman", "top_1pct_recall", "mean_predicted", "mean_observed", "n")}
    curve_n = int(tab["curve/n_new_repos"][0])
    s24, s48 = tab["curve/stars_24h"], tab["curve/stars_48h"]
    curve = {"n_new_repos_D1": curve_n, "share_with_star_24h": float(np.mean(s24 > 0)), "share_with_star_48h": float(np.mean(s48 > 0)),
             "mean_cumulative_stars_by_hour": (np.cumsum(tab["curve/stars_by_hour"]) / max(curve_n, 1)).tolist(),
             "stars_48h_distribution": S.count_distribution(s48),
             "note": "descriptive reference (not fitted): repositories created on 2024-10-08 UTC, non-owner non-bot stars within 48 h"}
    glimits = [*COMMON_LIMITS,
               "Units are repositories with at least one star or fork on the day (or created that day): the model says nothing about the large majority of repositories with no such activity.",
               "Next-day stars for the test day (2024-10-10) come from a Thursday; weekday and holiday patterns are not modelled.",
               "The negative-binomial dispersion was fitted on training rows; the simulated distribution's tail (q99.9) should be compared with the report before using the model for viral outliers."]
    gcard = {
        "name": gname, "version": 1, "id": gmid, "workstream": "E2", "domain": "software_collaboration_and_adoption",
        "loader": "crucible.calibration.software:RepoGrowthModel",
        "features": [n for n, _, _ in GROWTH_FEATURES], "feature_docs": {n: f"{d} [{tf}]" for n, tf, d in GROWTH_FEATURES},
        "target": "stars received by the repository on the next day (non-bot accounts)", "outputs": ["expected_stars_next_day"],
        "identification": {"kind": "observational", "statement": "Associations in public GitHub activity; nothing was randomized."},
        "population": POPULATION,
        "dataset": {"name": G.DATASET, "synthetic": False, "url": G.BASE_URL, **manifest},
        "checkpoint": gckpt, "report": rel(os.path.join(REPORTS_DIR, f"{gmid}.json")), "plan": plan, "plan_amendments": amendments,
        "fit": {"started_at": started_at, "code": "src/crucible/calibration/fit_software.py", **state,
                "command": "python -m crucible.calibration.fit_software", "threads": threads, "environment": environment(),
                "split": {"train": "D1 -> D2, repository hash bucket < 80", "validation": "D1 -> D2, bucket >= 80", "test": "D2 -> D3 (time-based)"},
                "counts": {s: int(v.size) for s, v in gy.items()}, "sub_models": ginfo},
        "metrics": gsum(gres[gkept]["test"]), "baseline_metrics": gsum(gres["proportional"]["test"]),
        "candidates": {c: {"complexity": gcx[c], "validation": gsum(gres[c]["validation"]), "test": gsum(gres[c]["test"]),
                           "test_new_repos": gsum(gres[c]["test_new_repos"])} for c in gc},
        "selection": {"rule": "lowest validation mean Poisson deviance; confirmed on test by paired bootstrap", "kept": gkept,
                      "baseline": "proportional", "comparisons": gcomps, "worse_model_kept": gworse},
        "dispersion": {"nb_alpha": alpha, "fitted_on": "train rows, kept model means, maximum likelihood"},
        "calibration_metrics": ["mean_poisson_deviance", "calibration_by_decile", "spearman", "top_1pct_recall", "distributional checks"],
        "limits": " ".join(glimits), "extrapolation_limits": glimits,
        "simulator_mapping": {
            "unit": "a repository (or package) in the software ecosystem on one simulated day",
            "stars_today": "star/watch actions by developers on the repository today", "forks_today": "code.fork calls today",
            "issue_events_today": "issue tracker events on its project today", "pr_events_today": "pull-request / review events today",
            "pushes_today": "code.commit calls today", "releases_today": "releases today", "events_today": "all repository events today",
            "stars_yesterday": "stars the previous simulated day", "created_today": "code.create_repo today",
            "created_yesterday": "code.create_repo the previous day", "repo_org": "owned by an organisation principal",
            "output": "expected stars tomorrow; sample(rng, X) draws counts with the fitted NB2 dispersion",
            "use": "a reference for the rate at which simulated developers discover repositories (to validate or calibrate the simulator's discovery and recommendation mechanisms), not a creator-assigned popularity",
            "transform": "matrix_from_counts(raw) applies log1p to count features",
        },
        "fidelity_mechanisms": ["repository_adoption_growth", "new_repository_adoption_curve"],
        "provenance": {"dataset": G.DATASET, "source": G.BASE_URL, "license": G.LICENSE, "fit_commit": state["commit"],
                       "plan_commit": plan["commit"]},
    }
    write_card(gname, gcard)
    gmodel = registry_load(gmid)
    mu = gmodel.predict(gX["test"])
    gaudit = {"rows": "all test rows", "n": int(gy["test"].size),
              "max_rel_diff_vs_estimator": float(np.max(np.abs(mu - gpred[(gkept, "test")]) / np.maximum(1.0, gpred[(gkept, "test")]))),
              "mean_poisson_deviance": S.poisson_deviance(gy["test"], mu),
              "baseline_mean_poisson_deviance": S.poisson_deviance(gy["test"], gmodel.candidate_predict("proportional", gX["test"]))}
    reports[gmid] = {
        "schema": "crucible.v5.calibration.report/1", "model": gmid, "plan": plan, "plan_amendments": amendments, "dataset": gcard["dataset"],
        "identification": gcard["identification"], "population": POPULATION, "split": gcard["fit"]["split"], "counts": gcard["fit"]["counts"],
        "heldout": {"validation": {c: gres[c]["validation"] for c in gc}, "test": {c: gres[c]["test"] for c in gc},
                    "test_new_repos": {c: gres[c]["test_new_repos"] for c in gc}},
        "selection": gcard["selection"],
        "calibration_metrics": {c: {"test": {k: gres[c]["test"][k] for k in ("mean_poisson_deviance", "calibration_by_decile")}} for c in gc},
        "distribution": dist, "new_repository_adoption_curve": curve, "audit": gaudit,
        "tables": {"digests": {k: v for k, v in digests.items() if k.startswith(("growth/", "curve/"))}, "version": TABLES_VERSION},
        "fit_info": ginfo, "extrapolation_limits": glimits,
    }
    for m, r in reports.items():
        r["wall_s"] = round(time.time() - t_start, 1); r["finished_at"] = now()
        write_report(m, r)
    print("deep kept", kept, "worse", worse, "| growth kept", gkept, "worse", gworse, "alpha", alpha,
          "wall", round(time.time() - t_start, 1), flush=True)
    return reports


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    main(ap.parse_args().threads)
