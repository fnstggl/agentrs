"""The article-open decision from workstream E's fitted news_click@1 model (crucible.social.consumption.CalibratedOpen)."""
import numpy as np
import pytest

import crucible
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social import experiments as X
from crucible.social.consumption import CalibratedOpen
from crucible.substrate import columns

crucible.load_all_modules()
DAY = 86400
PLAT = {"posts": {"k": 8, "recommenders": [{"name": "interest"}, {"name": "social"}]},
        "articles": {"k": 8, "recommenders": [{"name": "interest"}, {"name": "search"}, {"name": "fresh"}]}}


def _world(calibrated: bool):
    feed = {"calibrated": {"articles": {"open": "news_click@1"}}} if calibrated else {}
    w = World.create(X.spec("t-cal", 4, n=1500, creators=25, posts_per_day=3, platforms=PLAT, feed=feed,
                            crowd={"creator_platforms": {"posts": 0.4, "articles": 0.6}}), strict=True)
    X.run(w, DAY + 12 * 3600)
    return w


@pytest.fixture(scope="module")
def worlds():
    return _world(True), _world(False)


def test_calibrated_article_choice_runs_end_to_end(worlds):
    cal, base = worlds
    eng = cal.modules["feed"].engine
    assert set(eng.calibrated) == {"articles"} and isinstance(eng.calibrated["articles"], CalibratedOpen)
    assert eng.calibrated["articles"].model.card["provenance"]["datasets"] == ["mind_small"]
    rc = AUD.recount(cal)
    assert rc["total_mismatches"] == 0 and rc["views_without_prior_exposure"] == 0
    assert not cal.check_invariants()
    # the topic history the model reads is the individuals' own recorded views
    tv = columns.col(cal, "pop", "feed_topic_views")
    con = X.decision_rows(cal, "consume", 0)
    assert int(tv.sum()) == int(con["is_view"].sum())

    def open_rate(w, platform):
        imp = X.decision_rows(w, "imp", 0, platform=platform)
        c = X.decision_rows(w, "consume", 0, platform=platform)
        return c["item"].size / max(1, imp["item"].size)
    r_cal, r_base = open_rate(cal, "articles"), open_rate(base, "articles")
    assert 0.005 < r_cal < 0.9 and abs(r_cal - r_base) > 0.01          # a different decision model is in use
    # posts are untouched by the article model: both worlds use the built-in decision there
    assert 0.005 < open_rate(cal, "posts") < 0.9


def test_calibrated_open_probability_follows_affinity(worlds):
    cal, _ = worlds
    co = cal.modules["feed"].engine.calibrated["articles"]
    items = AUD.items_on(cal, "articles")
    idx = np.arange(0, 1500, 3, dtype=np.int64)
    rng = np.random.default_rng(0)
    feed = items[rng.integers(0, items.size, size=(idx.size, 8))]
    from crucible.social.recsys import features as F
    u = F.interests(cal, idx)
    E = F.item_emb(cal, feed.reshape(-1)).reshape(idx.size, 8, -1)
    aff = np.einsum("vd,vkd->vk", u, E).astype(np.float64)
    p = co.p_open(cal, "articles", idx, feed, aff, np.ones(feed.shape, bool))
    assert np.all((p > 0) & (p < 1))
    rho = np.corrcoef(np.argsort(np.argsort(aff.ravel())), np.argsort(np.argsort(p.ravel())))[0, 1]
    assert rho > 0.2                                                     # more relevant articles are opened more
