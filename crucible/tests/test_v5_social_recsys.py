"""Recommenders (crucible.social.recsys): the protocol, each mechanism's behaviour, fusion, schedules, finite
exposure, and the consumption engine's laws (attention, shares, contact graph)."""
import numpy as np
import pytest

import crucible
from crucible.core.errors import ProtocolError
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social import experiments as X
from crucible.social import scale, textrep
from crucible.social.recsys import REGISTRY, Mixture, Recommender, make, training
from crucible.social.recsys import features as F
from crucible.substrate import columns

crucible.load_all_modules()
DAY = 86400
ALL = {"posts": {"k": 8, "recommenders": [{"name": n, "weight": 1.0} for n in
                                          ("interest", "social", "chronological", "engagement", "cf", "explore",
                                           "fresh", "trending")]},
       "articles": {"k": 6, "recommenders": [{"name": n} for n in ("search", "diversity", "reputation", "random")]}}


@pytest.fixture(scope="module")
def world():
    w = World.create(X.spec("t-rec", 3, n=800, creators=20, posts_per_day=3, platforms=ALL,
                            feed={"train_interval_s": 6 * 3600}), strict=True)
    X.run(w, DAY + 12 * 3600)
    return w


def test_registry_and_protocol(world):
    assert {"chronological", "interest", "cf", "engagement", "explore", "social", "diversity", "reputation",
            "search", "fresh"} <= set(REGISTRY)
    w = world
    cand = AUD.items_on(w, "posts")
    viewers = np.arange(0, 400, dtype=np.int64)
    for name, cls in sorted(REGISTRY.items()):
        rec = make(name)
        assert isinstance(rec, Recommender) and rec.name == name and rec.version
        out = rec.rank(w, "posts", viewers, cand, 7, np.random.default_rng(0))
        assert out.shape == (viewers.size, 7) and out.dtype.kind == "i"
        ok = out >= 0
        assert np.all(np.isin(out[ok], cand))                       # only candidates, -1 padding
        for row in out:                                              # no item twice in a list
            r = row[row >= 0]
            assert r.size == np.unique(r).size
            assert np.all(row[r.size:] == -1)                        # padding only at the end
    with pytest.raises(ValueError):
        make("interest", {"no_such_param": 1})


def test_mechanisms_use_their_own_evidence(world):
    w = world
    cand = AUD.items_on(w, "posts")
    v = np.arange(0, 300, dtype=np.int64)
    rng = lambda: np.random.default_rng(1)                           # noqa: E731
    # chronological lists only items of creators each viewer follows, newest first
    ch = make("chronological").rank(w, "posts", v, cand, 5, rng())
    fol = F.creator_membership(w, "fo", cand, v)
    pub = columns.col(w, AUD.ITEMS, "published_at")
    for r, row in enumerate(ch):
        items = row[row >= 0]
        assert all(fol[r, np.searchsorted(cand, i)] for i in items)
        assert np.all(np.diff(pub[items]) <= 0)
    # interest ranking is ordered by cosine of the viewer's need-shifted interests with the item representation
    it = make("interest", {"need_weight": 0.0}).rank(w, "posts", v[:50], cand, 5, rng())
    u = F.interests(w, v[:50])
    for r, row in enumerate(it):
        s = F.item_emb(w, row[row >= 0]) @ u[r]
        assert np.all(np.diff(s) <= 1e-6)
    # the learned recommenders were fitted from recorded data and their parameters are world state
    assert training.model(w, "posts", "cf") is not None and training.model(w, "posts", "engagement") is not None
    # search retrieves only items whose text contains a query word
    srch = make("search")
    q = srch.queries(w, v[:20], rng())
    sr = srch.rank(w, "posts", v[:20], cand, 5, rng())
    from crucible.social.recsys.algorithms import _VOCAB
    for r, row in enumerate(sr):
        qw = {_VOCAB[j] for j in q[r]}
        for i in row[row >= 0]:
            assert qw & set(w.state.tables["aud_item_info"][str(int(i))]["terms"])
    # diversity: MMR lists are less self-similar than plain relevance lists
    dv = make("diversity", {"lam": 0.3}).rank(w, "posts", v[:100], cand, 6, rng())
    rel = make("interest").rank(w, "posts", v[:100], cand, 6, rng())

    def selfsim(lists):
        out = []
        for row in lists:
            E = F.item_emb(w, row[row >= 0])
            S = E @ E.T
            out.append((S.sum() - np.trace(S)) / max(1, S.size - len(S)))
        return float(np.mean(out))
    assert selfsim(dv) < selfsim(rel)


def test_mixture_fusion_schedule_and_override(world):
    w = World.from_blob(world.snapshot_blob(), world_id="t-rec-mix", fork=True)
    feed = w.modules["feed"]
    vid0, ents0 = feed.mixture_entries(w, "posts", w.now)
    X.set_mixture(w, "posts", [{"name": "random", "weight": 1.0}])
    vid1, ents1 = feed.mixture_entries(w, "posts", w.now)
    assert vid1 != vid0 and [e["name"] for e in ents1] == ["random"]
    assert X.mixture_changes(w) >= 1
    # an actor cannot change it
    rec = next(iter(w.state.tables["crowd_creators"].values()))
    r = w.execute(rec["principal"], {"tool": "feed.set_mixture", "args": {"platform": "posts",
                                                                         "recommenders": [{"name": "random"}]}})
    assert not r.ok and r.error["code"] == "forbidden"
    m = Mixture([{"name": "interest", "weight": 2.0}, {"name": "fresh", "weight": 1.0}])
    cand = AUD.items_on(w, "posts")
    out, src = m.rank(w, "posts", np.arange(50), cand, 6, np.random.default_rng(0))
    assert out.shape == (50, 6) and set(np.unique(src[out >= 0])) <= {AUD.SOURCE_CODE["interest"],
                                                                       AUD.SOURCE_CODE["fresh"]}
    with pytest.raises(ValueError):
        Mixture([{"name": "interest", "weight": 0.0}])
    with pytest.raises(ProtocolError):
        World.create(X.spec("t-bad", 1, n=50, creators=2, platforms={"posts": {"recommenders": [
            {"name": "interest"}], "schedule": [{"day": 0, "recommenders": [{"name": "fresh"}]}]}}))


def test_finite_exposure_popularity_takes_slots_from_competitors(world):
    """A session has k slots: an item that wins slots takes them from its competitors. A strongly matching post on
    the population's most common subject is published in a fork; after half a day of ordinary distribution and
    consumption its popularity (trending) and relevance (interest) win it slots, and every slot it holds is one a
    competitor no longer has."""
    w = World.from_blob(world.snapshot_blob(), world_id="t-fin", fork=True)
    mix = [{"name": "trending", "weight": 1.0}, {"name": "interest", "weight": 1.0}]
    X.set_mixture(w, "posts", mix)
    u = columns.col(w, "pop", "interests").astype(np.float64).mean(0)
    t = int(np.argmax(textrep.topic_centroids() @ u))
    creator = sorted(w.state.tables["crowd_creators"].items(), key=lambda kv: int(kv[0]))[0]
    one = np.zeros(len(textrep.topic_centroids()))
    one[t] = 1.0
    out = {}

    def pub(ctx):
        r = np.random.default_rng(5)
        title, text, tags = scale.write_text(r, one, 60, focus=0.95)
        return scale.publish_post(ctx, int(creator[0]), creator[1]["principal"], "posts", r, text=text, title=title,
                                  tags=tags)
    w._run_system("t.publish", lambda ctx: out.setdefault("i", pub(ctx)))
    new = out["i"]
    s0 = w.seq
    X.run(w, w.now + 12 * 3600)
    rows = X.decision_rows(w, "imp", s0, platform="posts")
    share_new = float((rows["item"] == new).mean())
    assert share_new > 0 and AUD.audience(w, new)["unique_viewers"] > 0
    # allocation: the same cohort and state, candidates with and without the popular item
    from crucible.social.consumption import candidate_pool
    pool = candidate_pool(w, "posts", 300, 7)
    assert new in pool
    m = Mixture(mix)
    cohort = np.arange(0, AUD.n_people(w), 3, dtype=np.int64)
    seen = F.membership(w, "vw", np.array([new]), cohort)[:, 0]
    cohort = cohort[~seen]                                       # people who have not viewed it yet
    fa, _ = m.rank(w, "posts", cohort, pool[pool != new], 8, np.random.default_rng(0))
    fb, _ = m.rank(w, "posts", cohort, pool, 8, np.random.default_rng(0))
    taken = int((fb == new).sum())
    assert taken > 0
    comp_a = int((fa >= 0).sum())
    comp_b = int(((fb >= 0) & (fb != new)).sum())
    assert comp_b == comp_a - taken                              # every slot it holds is a competitor's loss
    # and per session never more impressions than slots
    _, c = np.unique(rows["seq"] * AUD.n_people(w) + rows["viewer"], return_counts=True)
    assert c.max() <= 8


def test_attention_and_shares(world):
    w = world
    assert not [e for e in w.check_invariants() if "attention" in e]
    con = X.decision_rows(w, "consume", 0)
    budget = columns.col(w, "pop", "attention_budget_min").astype(np.float64)
    key = (con["t"] // DAY) * AUD.n_people(w) + con["viewer"]
    uk, inv = np.unique(key, return_inverse=True)
    assert np.all(np.bincount(inv, weights=con["watch_s"]) / 60.0 <= budget[uk % AUD.n_people(w)] + 1e-3)
    # shares reached the sharers' contacts: recipients' inboxes hold shared items, generation >= 1
    inbox = columns.col(w, "pop", "feed_inbox")
    gen = columns.col(w, "pop", "feed_inbox_gen")
    assert int((inbox >= 0).sum()) > 0 and int(gen[inbox >= 0].min()) >= 1
    imp = X.decision_rows(w, "imp", 0)
    assert int((imp["src"] == AUD.SOURCE_CODE["share"]).sum()) > 0


def test_contact_graph_and_homophily():
    g = scale.ContactGraph(10_000_000, 42, 500, 8, 4)
    i = np.array([0, 1, 499, 500, 9_999_999, 1234567])
    c = g.contacts(i)
    assert c.shape == (i.size, 12) and np.array_equal(c, g.contacts(i))      # deterministic, no storage
    assert np.all(c != i[:, None]) and np.all((c >= 0) & (c < 10_000_000))
    assert np.all(c[:, :8] // 500 == (i // 500)[:, None])                     # local contacts in the community
    w = World.create(X.spec("t-homo", 2, n=4000, creators=0, platforms={"posts": {}}, identity=False))
    u = columns.col(w, "pop", "interests").astype(np.float64)
    gr = w.runtime["crowd.graph"]
    idx = np.arange(0, 4000, 7)
    nb = gr.contacts(idx)[:, 0]
    rnd = np.random.default_rng(0).integers(0, 4000, idx.size)
    sim_nb = np.einsum("nd,nd->n", u[idx], u[nb]).mean()
    sim_rnd = np.einsum("nd,nd->n", u[idx], u[rnd]).mean()
    assert sim_nb > sim_rnd + 0.1                                             # neighbours share interests
