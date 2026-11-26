"""Gate SP-4.5b: the popularity of one item reduces the opportunities of competing content; feedback is delayed,
analytics noisy, audiences change, platform algorithms evolve endogenously and competitors adapt. Owner: workstream C.

One persistent run of the integrated social internet on the ONE population (`social_world`, 2,000 individuals, 4
simulated days; the posts platform reweights its own recommender mixture every day from its measured feedback and
runs learned recommenders refitted from recorded outcomes; members create on several platforms).

Attention competition (randomised, within one world so that both arms share every other draw): at day 3 a random
half of 1,600 individuals receives an experimenter's informative post (on the population's most read topic) in the
share inbox of the posts platform (as when many contacts shared it, without the notification an ordinary share sends)
and the other half does not. Outcome: what each individual was shown and consumed of OTHER items on posts during the
next 24 hours. Popularity takes finite
feed slots and attention: the competing content must get fewer opportunities with the treated - fewer impressions
(feed slots; the primary outcome, one-sided Mann-Whitney U test) and fewer views and seconds (reported).

Dynamic kinds, each observed on the run (not configured):
1. delayed feedback - the analytics a creator sees lag the exact counters (published values older than one tick);
2. noisy analytics - published values differ from the exact ones by the platform's multiplicative noise;
3. changing audiences - individuals' interests drift with what they consume (mean cosine to day 0 below 1) and the
   set of active individuals changes from day to day;
4. platform algorithm evolution - a platform changed its own distribution from its feedback (feed.self_changes);
5. learned recommenders refitted on new recorded outcomes (feed.model events with different parameters);
6. competitors adapt - creators change what they make from what their earlier pieces earned (per-medium value and
   medium choice change over the run).

Metrics (ACCEPTANCE_MANIFEST): attention_competition_effect_pvalue <= 0.01, dynamic_kinds_observed >= 5.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social import experiments as X
from crucible.social.worlds import social_world
from crucible.substrate import columns

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_5b.py"
DAY = 86400
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3 * 3600)]

POSTS = {"recommenders": [{"name": "interest", "weight": 0.8}, {"name": "engagement", "weight": 0.8},
                          {"name": "cf", "weight": 0.6}, {"name": "chronological", "weight": 0.6},
                          {"name": "social", "weight": 0.6}, {"name": "explore", "weight": 0.2}],
         "adapt": {"every_s": DAY, "eta": 0.6, "min_impressions": 150}}


def test_SP_4_5b_attention_competition_and_dynamics():
    plats = ["posts", "blog", "pics", "shorts"]
    w = World.create(social_world("development", 45, n=2000, platforms=plats,
                                  feed={"platforms": {"posts": POSTS}, "train_interval_s": 6 * 3600},
                                  creators={"prior": -0.4}, params={"log_retain": 6_000_000}), strict=True)
    n = AUD.n_people(w)
    interests0 = columns.col(w, "pop", "interests").astype(np.float32).copy()
    value0 = columns.col(w, "pop", "soc_med_value").astype(np.float32).copy()
    active_days = []
    for d in range(3):
        s0 = w.seq
        w.advance_time((d + 1) * DAY)
        imp = X.decision_rows(w, "imp", s0)
        active_days.append(set(np.unique(imp["viewer"]).tolist()))
    # ---------------------------------------------------------------- attention competition
    from crucible.population import semantics, writing
    lab = w.system_call("admin.join", {"handle": "lab", "endowment": 10_000, "password": "pw-lab"})["principal"]
    topic = int(np.argmax(semantics.popularity()))
    title, body = writing.compose(np.random.default_rng(9), topic, 6, 1.0)
    a = w.execute(lab, {"tool": "studio.write", "args": {"format": "text", "title": title, "body": body}})
    r = w.execute(lab, {"tool": "posts.publish", "args": {"ref": a.result["ref"], "title": title}})
    assert r.ok, r.error
    item = int(w.state.tables["content"][r.result["id"]]["aud_item"])
    rng = np.random.default_rng(45)
    sample = rng.choice(n, 1600, replace=False)
    treat, ctrl = np.sort(sample[:800]), np.sort(sample[800:])
    feed = w.modules["feed"]
    from crucible.social.consumption import group_into_rows, ring_push

    def assign(ctx):
        # the item sits in the share inbox of the treated (as when many of one's contacts shared it) - without the
        # notification of an ordinary share, so that the treated are not brought back more often than the control
        u_, cnt, (newi, newg) = group_into_rows(treat, [np.full(treat.size, item, np.int32),
                                                       np.ones(treat.size, np.uint8)], feed.R)
        columns.batch(ctx, "pop", "share.deliver", u_,
                      {"feed_inbox": ring_push(columns.col(w, "pop", "feed_inbox")[u_], newi, cnt),
                       "feed_inbox_gen": ring_push(columns.col(w, "pop", "feed_inbox_gen")[u_], newg, cnt)},
                      mode="set", extra={"deliveries": int(treat.size), "experiment": "attention_competition"})
    w._run_system("experiment.assign", assign)
    s0 = w.seq
    w.advance_time(w.now + DAY)
    con = X.decision_rows(w, "consume", s0, platform="posts")
    imp = X.decision_rows(w, "imp", s0, platform="posts")
    other = con["item"] != item
    views = np.zeros(n)
    secs = np.zeros(n)
    imps = np.zeros(n)
    np.add.at(views, con["viewer"][other & con["is_view"]], 1)
    np.add.at(secs, con["viewer"][other], con["watch_s"][other])
    np.add.at(imps, imp["viewer"][imp["item"] != item], 1)
    got_item = np.zeros(n, bool)
    got_item[AUD.viewers_of(w, item)] = True
    # the competing content's opportunities: its impressions (feed slots) and the views and seconds it received
    u = mannwhitneyu(imps[treat], imps[ctrl], alternative="less")
    uv = mannwhitneyu(views[treat], views[ctrl], alternative="less")
    us = mannwhitneyu(secs[treat], secs[ctrl], alternative="less")
    # ---------------------------------------------------------------- dynamic kinds
    st = w.state
    kinds = {}
    ni = AUD.n_items(w)
    pub_t = AUD.item_col(w, "pub_t")[:ni]
    stg_t = AUD.item_col(w, "stg_t")[:ni]
    exact = AUD.item_col(w, "views")[:ni].astype(np.float64)
    pubv = AUD.item_col(w, "pub_views")[:ni].astype(np.float64)
    stgv = AUD.item_col(w, "stg_views")[:ni].astype(np.float64)
    shown = pub_t >= 0
    age = (w.now - stg_t[shown]) if shown.any() else np.zeros(0)
    behind = int(np.sum(stgv[shown] < exact[shown]))
    kinds["delayed_feedback"] = bool(age.size and float(age.min()) > 0 and behind > 0)
    m = (stgv > 0) & (pubv > 0)
    noise_sd = float(np.std(np.log(pubv[m] / stgv[m]))) if m.sum() > 5 else 0.0
    kinds["noisy_analytics"] = noise_sd > 0.0
    interests1 = columns.col(w, "pop", "interests").astype(np.float32)
    cos = np.sum(interests0 * interests1, axis=1) / np.maximum(
        np.linalg.norm(interests0, axis=1) * np.linalg.norm(interests1, axis=1), 1e-9)
    jac = len(active_days[1] & active_days[2]) / max(len(active_days[1] | active_days[2]), 1)
    kinds["changing_audiences"] = bool(float(np.mean(cos)) < 0.9999 and jac < 1.0)
    kinds["platform_algorithm_evolution"] = X.self_changes(w) > 0
    fits = [e for e in w.log if e.type == "feed.model"]
    refs = {}
    for e in fits:
        refs.setdefault((e.payload["platform"], e.payload["rec"]), set()).add(e.payload["ref"])
    kinds["learned_recommenders_refitted"] = any(len(v) > 1 for v in refs.values())
    value1 = columns.col(w, "pop", "soc_med_value").astype(np.float32)
    media = {}
    for e in w.log:
        if e.type == "cols.batch" and e.payload.get("kind") == "pop_social.created":
            media.setdefault(int(e.t // DAY), []).extend((e.payload.get("extra") or {}).get("media") or [])
    kinds["competitors_adapt"] = bool(np.any(np.abs(value1 - value0) > 1e-6))
    observed = sorted(k for k, v in kinds.items() if v)
    metrics = {"attention_competition_effect_pvalue": float(u.pvalue), "dynamic_kinds_observed": len(observed),
               "kinds": ",".join(observed), "competing_views_treated_mean": round(float(views[treat].mean()), 4),
               "competing_views_control_mean": round(float(views[ctrl].mean()), 4),
               "competing_seconds_treated_mean": round(float(secs[treat].mean()), 2),
               "competing_seconds_control_mean": round(float(secs[ctrl].mean()), 2),
               "competing_seconds_pvalue": float(us.pvalue), "competing_views_pvalue": float(uv.pvalue),
               "competing_impressions_treated_mean": round(float(imps[treat].mean()), 4),
               "competing_impressions_control_mean": round(float(imps[ctrl].mean()), 4),
               "treated_who_viewed_the_popular_item": int(got_item[treat].sum()),
               "control_who_viewed_the_popular_item": int(got_item[ctrl].sum()),
               "analytics_log_noise_sd": round(noise_sd, 4), "analytics_min_age_s": int(age.min()) if age.size else -1,
               "items_whose_published_views_lag_exact": behind, "interest_mean_cosine_to_day0": round(float(np.mean(cos)), 6),
               "active_jaccard_day2_day3": round(jac, 4), "platform_self_changes": int(X.self_changes(w)),
               "recommender_fits": len(fits),
               "media_by_day": ";".join(f"{d}:{len(v)}" for d, v in sorted(media.items())),
               "creators_whose_medium_values_changed": int(np.sum(np.any(np.abs(value1 - value0) > 1e-6, axis=1)))}
    evidence.record("SP-4.5b", metrics, command=CMD, notes=(
        "One persistent run on 2,000 population members (4 platforms, 4 days). Attention competition: randomised "
        "inbox delivery of one popular item to 800 of 1,600 individuals at day 3; outcome = qualifying views of "
        "other items on posts in the next day (one-sided Mann-Whitney). Dynamic kinds observed in the run itself."))
    assert got_item[treat].sum() > got_item[ctrl].sum()
    assert w.check_invariants() == []
