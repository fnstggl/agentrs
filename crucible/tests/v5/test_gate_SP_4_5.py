"""SP-4.5 (MASTER_SPEC Part 4.5): recommendation mechanisms - chronological, interest retrieval, collaborative
filtering, learned engagement, exploration, social graph, diversity-aware, reputation-sensitive, search-driven,
freshness - with delayed and noisy feedback. Owner: workstream C2.

Metric: recommendation_mechanisms >= 10. A mechanism counts when, inside a platform mixture of a persistent run
(2,500 identities, 30 creators, two platforms, 3 simulated days), it placed items that were shown (impressions
attributed to it) and viewed, and when, as the only distributor of a forked arm, it produced an exposure pattern
different from every other mechanism's. The learned mechanisms must have been refitted from delayed, noisy feedback,
and the platform analytics they and the bandit read must be delayed and noisy.
"""
import itertools

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social import experiments as X
from crucible.social.recsys import REGISTRY, training

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_5.py"
DAY = 86400
REQUIRED = ("chronological", "interest", "cf", "engagement", "explore", "social", "diversity", "reputation", "search",
            "fresh")

PLATFORMS = {
    "posts": {"k": 10, "recommenders": [{"name": "chronological", "weight": 0.6}, {"name": "interest", "weight": 0.8},
                                        {"name": "cf", "weight": 0.8}, {"name": "engagement", "weight": 0.8},
                                        {"name": "explore", "weight": 0.5}, {"name": "social", "weight": 0.8},
                                        {"name": "fresh", "weight": 0.5}, {"name": "trending", "weight": 0.4}]},
    "blog": {"k": 8, "recommenders": [{"name": "diversity", "weight": 0.8}, {"name": "reputation", "weight": 0.8},
                                          {"name": "search", "weight": 0.8}, {"name": "random", "weight": 0.2}]},
}


@pytest.fixture(scope="module")
def run():
    # the one population (crucible.population; the crowd stand-in is not used)
    w = World.create(X.pop_spec("sp45", 31, n=2500, platforms=PLATFORMS), strict=True)
    X.run(w, 2 * DAY)
    snap = w.snapshot_blob()
    s2 = w.seq
    X.run(w, 3 * DAY)
    return {"world": w, "day2": snap, "day2_seq": s2}


def test_recommendation_mechanisms(run):
    w = run["world"]
    imp = X.decision_rows(w, "imp", 0)
    con = X.decision_rows(w, "consume", 0)
    n = AUD.n_people(w)
    # views of impressions attributed to each mechanism: the (item, viewer) pair was placed by it, then viewed
    k_imp = imp["item"] * n + imp["viewer"]
    kv = np.unique(con["item"][con["is_view"]] * n + con["viewer"][con["is_view"]])
    placed = {}
    for name in sorted(REGISTRY):
        m = imp["src"] == AUD.SOURCE_CODE[name]
        keys = np.unique(k_imp[m])
        placed[name] = {"impressions": int(m.sum()), "views": int(np.isin(keys, kv, assume_unique=True).sum())}
    # each mechanism alone, from the same day-2 state, for half a day: its exposure pattern
    arms = {}
    for name in sorted(REGISTRY):
        a = World.from_blob(run["day2"], world_id=f"sp45-{name}", fork=True)
        for p in PLATFORMS:
            X.set_mixture(a, p, [{"name": name}])
        s0 = a.seq
        X.run(a, 2 * DAY + DAY // 2)
        arms[name] = X.pair_keys(a, X.decision_rows(a, "imp", s0))
    overlaps = {f"{x}|{y}": X.jaccard(arms[x], arms[y]) for x, y in itertools.combinations(sorted(arms), 2)}
    distinct = {nm: all(v < 0.9 for k, v in overlaps.items() if nm in k.split("|")) for nm in arms}
    working = [nm for nm in sorted(REGISTRY) if placed[nm]["impressions"] > 0 and placed[nm]["views"] > 0 and
               distinct[nm] and arms[nm].size > 0]
    # delayed, noisy feedback: refits of the learned mechanisms and the published analytics
    fits = [e.payload for e in w.log if e.type == "feed.model"]
    eng = [f for f in fits if f["rec"] == "engagement"]
    cf = [f for f in fits if f["rec"] == "cf"]
    eng_delay = min((f["t"] - f["meta"]["newest_example_t"] for f in eng), default=-1)
    eng_noise = float(np.mean([f["meta"]["flipped"] / f["meta"]["examples"] for f in eng])) if eng else 0.0
    hashes = {f["rec"]: set() for f in fits}
    for f in fits:
        hashes[f["rec"]].add(f["ref"])
    ni = AUD.n_items(w)
    pub_t = AUD.item_col(w, "pub_t")
    pub_v = AUD.item_col(w, "pub_views").astype(np.float64)
    # exact views at the time each published figure refers to, recounted from the recorded batches
    exact = np.zeros(ni)
    ok_t = con["is_view"] & (con["t"] <= pub_t[con["item"]])
    np.add.at(exact, con["item"][ok_t], 1)
    m = (pub_t >= 0) & (exact >= 50)
    logratio = np.log(np.maximum(pub_v[m], 1) / exact[m])
    analytics_delay = float(np.median(w.now - pub_t[pub_t >= 0])) if np.any(pub_t >= 0) else -1.0
    metrics = {"recommendation_mechanisms": len(working), "required_mechanisms_working":
               int(sum(1 for r in REQUIRED if r in working)), "registered_mechanisms": len(REGISTRY),
               "mechanisms": ",".join(working),
               "max_pairwise_exposure_overlap": round(max(overlaps.values()), 4),
               "engagement_refits": len(eng), "cf_refits": len(cf),
               "distinct_fitted_parameter_sets": int(sum(len(v) for v in hashes.values())),
               "engagement_min_feedback_age_s": int(eng_delay), "engagement_label_flip_rate": round(eng_noise, 4),
               "analytics_median_age_s": analytics_delay,
               "analytics_noise_log_sd": round(float(np.std(logratio)), 4) if logratio.size else 0.0,
               "analytics_items_compared": int(m.sum())}
    evidence.record("SP-4.5", metrics, command=CMD,
                    notes="Per mechanism, inside the mixtures: " + ", ".join(
                        f"{k}: {v['impressions']} imp / {v['views']} viewed" for k, v in placed.items()) +
                    ". Engagement is refitted every 6 h only on impressions at least delay_s old with label_noise of "
                    "the labels flipped; explore and engagement read the published analytics, which lag one to two "
                    "analytics intervals and carry log-normal noise (sd compared against the exact recount).")
    assert len(eng) >= 2 and len(cf) >= 2
    assert eng_delay >= 6 * 3600
    assert set(REQUIRED) <= set(REGISTRY)
