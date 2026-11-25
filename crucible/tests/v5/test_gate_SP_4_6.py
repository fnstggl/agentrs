"""SP-4.6 (MASTER_SPEC Part 4.6): audience accounting distinguishes impressions, unique exposed, views, unique
viewers, watch duration, completion, repeat consumption, voluntary shares, subscribers and returning viewers, with a
documented view threshold. Owner: workstream C2.

Metrics: audience_quantities_distinguished >= 10, view_threshold_documented == true. Measured on a real run (2,000
identities, 30 creators, a following feed that shows already-seen items so repeat consumption occurs) and with
controlled consumption by population individuals acting through the feed tools.
"""
import os

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social import experiments as X
from crucible.substrate import ids

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_6.py"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAY = 86400

PLATFORMS = {
    "posts": {"k": 10, "recommenders": [{"name": "chronological", "weight": 1.0},
                                        {"name": "interest", "weight": 1.0},
                                        {"name": "trending", "weight": 0.5, "params": {"exclude_seen": False}},
                                        {"name": "social", "weight": 0.8}]},
    "articles": {"k": 8, "recommenders": [{"name": "interest", "weight": 1.0},
                                          {"name": "chronological", "weight": 0.8}]},
}


@pytest.fixture(scope="module")
def world():
    w = World.create(X.spec("sp46", 21, n=2000, creators=30, platforms=PLATFORMS), strict=True)
    X.run(w, 3 * DAY)
    return w


def _individual(w, items, exclude=()):
    """A non-creator individual never shown any of `items`, made a principal (so it can use the feed tools)."""
    creators = set(w.state.tables["crowd_creators"])
    shown = set(np.concatenate([AUD.exposed(w, it) for it in items]).tolist())
    i = next(i for i in range(AUD.n_people(w)) if str(i) not in creators and i not in exclude and i not in shown)
    w._run_system("sp46.principal", lambda ctx: ids.ensure_principal(ctx, i))
    return i, ids.pid(i)


def test_audience_quantities_and_view_threshold(world):
    w = world
    ni = AUD.n_items(w)
    Q = np.stack([np.array([AUD.audience(w, i)[q] for i in range(ni)], dtype=np.float64) for q in AUD.QUANTITIES])
    nonzero = Q.sum(axis=1) > 0
    distinct = [bool(nonzero[a] and all(not np.array_equal(Q[a], Q[b]) for b in range(len(Q)) if b != a))
                for a in range(len(Q))]
    q = {name: Q[j] for j, name in enumerate(AUD.QUANTITIES)}
    relations = {
        "exposure_not_view": bool(np.any(q["unique_exposed"] > q["unique_viewers"])),
        "impressions_above_unique_exposed": bool(np.any(q["impressions"] > q["unique_exposed"])),
        "repeat_views_not_unique": bool(np.any(q["repeat_views"] > 0) and
                                        np.array_equal(q["repeat_views"], q["views"] - q["unique_viewers"])),
        "completions_below_views": bool(np.any(q["completions"] < q["views"]) and np.all(q["completions"] <= q["views"])),
        "returning_within_unique": bool(np.any(q["returning_viewers"] > 0) and
                                        np.all(q["returning_viewers"] <= q["unique_viewers"])),
        "shares_and_subscribers_recorded": bool(q["shares"].sum() > 0 and q["subscribers"].sum() > 0),
        "watch_time_includes_non_views": True,
    }
    rc = AUD.recount(w)
    # watch time also counts partial viewings that did not reach the threshold
    con = X.decision_rows(w, "consume", 0)
    relations["watch_time_includes_non_views"] = bool(np.any(~con["is_view"] & (con["watch_s"] > 0)))

    # controlled consumption through the tools: the threshold, exposure-before-view, repeat and self-views
    art = next(i for i in range(ni) if AUD.item_col(w, "duration_s")[i] >= 90 and
               w.state.tables["aud_item_info"][str(i)]["platform"] == "articles")
    post = next(i for i in range(ni) if AUD.item_col(w, "duration_s")[i] < 50 and
                w.state.tables["aud_item_info"][str(i)]["platform"] == "posts")
    i1, p1 = _individual(w, (art, post))
    i2, p2 = _individual(w, (art, post), exclude=(i1,))
    checks = {}

    def watch(p, item, s):
        return w.execute(p, {"tool": "feed.watch", "args": {"item": item, "seconds": s}})

    v0 = AUD.audience(w, art)
    r = watch(p2, art, 60)
    checks["no_view_without_exposure"] = (not r.ok) and AUD.audience(w, art) == v0
    before = AUD.audience(w, art)
    assert w.execute(p1, {"tool": "feed.open", "args": {"item": art}}).ok
    after_open = AUD.audience(w, art)
    checks["exposure_counted_not_view"] = (after_open["impressions"] == before["impressions"] + 1 and
                                           after_open["views"] == before["views"])
    r29 = watch(p1, art, 29.0)
    a29 = AUD.audience(w, art)
    checks["29s_of_long_item_is_not_a_view"] = (r29.ok and r29.result["watched_s"] == 29.0 and
                                                 not r29.result["counted_as_view"] and
                                                 a29["views"] == before["views"] and
                                                 abs(a29["watch_s"] - before["watch_s"] - 29.0) < 1e-3)
    r30 = watch(p1, art, 30.0)
    a30 = AUD.audience(w, art)
    checks["30s_of_long_item_is_a_view"] = (r30.ok and r30.result["counted_as_view"] and
                                             a30["views"] == before["views"] + 1 and
                                             a30["unique_viewers"] == before["unique_viewers"] + 1)
    r30b = watch(p1, art, 30.0)
    a30b = AUD.audience(w, art)
    checks["second_view_is_a_repeat"] = (r30b.ok and a30b["views"] == a30["views"] + 1 and
                                         a30b["unique_viewers"] == a30["unique_viewers"] and
                                         a30b["repeat_views"] == a30["repeat_views"] + 1)
    d = float(AUD.item_col(w, "duration_s")[post])
    assert w.execute(p1, {"tool": "feed.open", "args": {"item": post}}).ok
    pb = AUD.audience(w, post)
    rlow = watch(p1, post, round(0.5 * d - 0.5, 3))
    rhalf = watch(p1, post, 0.5 * d)
    pa = AUD.audience(w, post)
    checks["half_of_short_item_is_a_view"] = (rlow.ok and not rlow.result["counted_as_view"] and rhalf.ok and
                                              rhalf.result["counted_as_view"] and pa["views"] == pb["views"] + 1)
    creator = w.state.tables["aud_item_info"][str(art)]["creator"]
    ci = ids.index(creator)
    assert w.execute(creator, {"tool": "feed.open", "args": {"item": art}}).ok
    cb = AUD.audience(w, art)
    rs = watch(creator, art, float(AUD.item_col(w, "duration_s")[art]))
    ca = AUD.audience(w, art)
    checks["creator_self_view_not_counted"] = (rs.ok and rs.result["watched_s"] >= 30.0 and
                                               ca["views"] == cb["views"] and ca["completions"] == cb["completions"]
                                               and ca["watch_s"] == cb["watch_s"])
    rc2 = AUD.recount(w)
    checks["recount_exact_after_controlled_views"] = rc2["total_mismatches"] == 0 and \
        rc2["views_without_prior_exposure"] == 0

    guide = open(os.path.join(ROOT, "docs/v5/components/C2.md")).read()
    documented = (AUD.VIEW_RULE in guide and AUD.VIEW_RULE in (AUD.__doc__ or "") and AUD.VIEW_MIN_S == 30.0 and
                  AUD.VIEW_MIN_FRAC == 0.5 and float(AUD.view_threshold(100.0)) == 30.0 and
                  float(AUD.view_threshold(20.0)) == 10.0 and all(checks.values()))
    metrics = {"audience_quantities_distinguished": int(sum(distinct)), "view_threshold_documented": bool(documented),
               "quantities": ",".join(AUD.QUANTITIES), "items": ni,
               "relations_holding": int(sum(relations.values())), "relations_checked": len(relations),
               "controlled_checks_passed": int(sum(checks.values())), "controlled_checks": len(checks),
               "recount_mismatches": rc["total_mismatches"], "repeat_views_total": int(q["repeat_views"].sum()),
               "returning_viewers_total": int(q["returning_viewers"].sum()), "view_rule": AUD.VIEW_RULE}
    evidence.record("SP-4.6", metrics, command=CMD,
                    notes="A quantity counts as distinguished when it is non-zero and its per-item values differ "
                          "from those of every other quantity. Relations: " +
                          ", ".join(f"{k}={v}" for k, v in relations.items()) + ". Controlled checks: " +
                          ", ".join(f"{k}={v}" for k, v in checks.items()))
    assert all(relations.values()), relations
    assert all(checks.values()), checks
