"""The article-open decision from workstream E's runtime decision model (crucible.social.consumption.CalibratedOpen
-> crucible.calibration.runtime `article_open_semantic@1` -> news_click_semantic@1).

The default C2 representation is lexical, not the MiniLM space the model's affinity is defined in, so a world must
declare the space. The end-to-end test below declares it for a C2 world only to exercise the wiring (decision
events, exact recount); it validates the mechanics, not the behaviour, because the C2 interests are not MiniLM
vectors (labelled here and in the fidelity register)."""
import numpy as np
import pytest

import crucible
from crucible.calibration.runtime import ContractError
from crucible.core.errors import ProtocolError
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social import experiments as X
from crucible.social.consumption import CalibratedOpen
from crucible.substrate import columns

crucible.load_all_modules()
DAY = 86400
PLAT = {"posts": {"k": 8, "recommenders": [{"name": "interest"}, {"name": "social"}]},
        "articles": {"k": 8, "recommenders": [{"name": "interest"}, {"name": "search"}, {"name": "fresh"}]}}
DECLARED = {"articles": {"open": "article_open_semantic@1", "affinity_space": "minilm-l6-cosine",
                         "content_kind": "article"}}


def _spec(feed):
    return X.spec("t-cal", 4, n=1500, creators=25, posts_per_day=3, platforms=PLAT, feed=feed,
                  crowd={"creator_platforms": {"posts": 0.4, "articles": 0.6}})


def _world(feed):
    w = World.create(_spec(feed), strict=True)
    X.run(w, DAY + 12 * 3600)
    return w


@pytest.fixture(scope="module")
def worlds():
    return _world({"calibrated": DECLARED}), _world({})


def test_retired_and_undeclared_configurations_are_refused():
    # the batch-rank news_click@1 path is retired; an undeclared affinity space is outside the model's contract
    for cal in ({"articles": {"open": "news_click@1"}}, {"articles": {"open": "article_open_semantic@1"}},
                # a news-article model on a platform of social posts is outside its scope (card limit 1)
                {"posts": {"open": "article_open_semantic@1", "affinity_space": "minilm-l6-cosine",
                           "content_kind": "post"}}):
        with pytest.raises((ContractError, ProtocolError)):
            World.create(_spec({"calibrated": cal}), strict=True)


def test_calibrated_article_choice_runs_end_to_end(worlds):
    cal, base = worlds
    eng = cal.modules["feed"].engine
    assert set(eng.calibrated) == {"articles"} and isinstance(eng.calibrated["articles"], CalibratedOpen)
    assert eng.calibrated["articles"].dm.model_id == "news_click_semantic@1"
    rc = AUD.recount(cal)
    assert rc["total_mismatches"] == 0 and rc["views_without_prior_exposure"] == 0
    assert not cal.check_invariants()
    tv = columns.col(cal, "pop", "feed_topic_views")
    con = X.decision_rows(cal, "consume", 0)
    assert int(tv.sum()) == int(con["is_view"].sum())

    def open_rate(w, platform):
        imp = X.decision_rows(w, "imp", 0, platform=platform)
        c = X.decision_rows(w, "consume", 0, platform=platform)
        return c["item"].size / max(1, imp["item"].size)
    r_cal, r_base = open_rate(cal, "articles"), open_rate(base, "articles")
    assert 0.001 < r_cal < 0.9 and abs(r_cal - r_base) > 0.005          # a different decision model is in use
    assert 0.005 < open_rate(cal, "posts") < 0.9                      # posts keep the built-in decision


def _slates(w, n_view=400, seed=0):
    from crucible.social.recsys import features as F
    items = AUD.items_on(w, "articles")
    idx = np.arange(0, 1500, 1500 // n_view, dtype=np.int64)[:n_view]
    rng = np.random.default_rng(seed)
    feed = items[rng.integers(0, items.size, size=(idx.size, 8))]
    u = F.interests(w, idx)
    E = F.item_emb(w, feed.reshape(-1)).reshape(idx.size, 8, -1)
    aff = np.einsum("vd,vkd->vk", u, E).astype(np.float64)
    return idx, feed, aff


def test_calibrated_open_is_row_wise_and_absolute(worlds):
    cal, _ = worlds
    co = cal.modules["feed"].engine.calibrated["articles"]
    idx, feed, aff = _slates(cal)
    has = np.ones(feed.shape, bool)
    p = co.p_open(cal, "articles", idx, feed, aff, has)
    assert np.all((p > 0) & (p < 1))
    # one viewer's probabilities do not depend on which other viewers share the batch
    for v in (0, 7, 123):
        alone = co.p_open(cal, "articles", idx[v:v + 1], feed[v:v + 1], aff[v:v + 1], has[v:v + 1])
        np.testing.assert_allclose(alone[0], p[v], rtol=0, atol=1e-12)
    half = co.p_open(cal, "articles", idx[::2], feed[::2], aff[::2], has[::2])
    np.testing.assert_allclose(half, p[::2], rtol=0, atol=1e-12)
    # uniformly more relevant content raises opening probabilities (the old batch-rank map gave exactly 0 change)
    up = co.p_open(cal, "articles", idx, feed, np.clip(aff + 0.2, -1, 1), has)
    assert float(np.mean(up - p)) > 0.0
    rho = np.corrcoef(np.argsort(np.argsort(aff.ravel())), np.argsort(np.argsort(p.ravel())))[0, 1]
    assert rho > 0.2                                                     # more relevant articles are opened more
