"""Audience accounting (crucible.social.audience): recorded decisions only, the view rule, exact recount, analytics
visibility, determinism (replay, regeneration, fork)."""
import numpy as np
import pytest

import crucible
from crucible.core.errors import CrucibleError, InvalidAction
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social import experiments as X
from crucible.substrate import ids

crucible.load_all_modules()
DAY = 86400
SMALL = {"posts": {"k": 8, "recommenders": [{"name": "interest"}, {"name": "chronological", "weight": 0.7},
                                            {"name": "social"}]},
         "articles": {"k": 6, "recommenders": [{"name": "search"}, {"name": "fresh"}]}}


@pytest.fixture(scope="module")
def world():
    w = World.create(X.spec("t-aud", 5, n=600, creators=15, posts_per_day=3, platforms=SMALL), strict=True)
    X.run(w, DAY + 6 * 3600)
    return w


def _sys(w, fn):
    out = {}
    w._run_system("test", lambda ctx: out.setdefault("r", fn(ctx)))
    return out.get("r")


def test_decision_validation(world):
    w = world
    ni = AUD.n_items(w)
    post = next(i for i in range(ni) if w.state.tables["aud_item_info"][str(i)]["platform"] == "posts")
    dur = float(AUD.item_col(w, "duration_s")[post])
    shown = AUD.exposed(w, post)
    fresh = next(i for i in range(AUD.n_people(w)) if i not in set(shown.tolist()))
    v = np.array([fresh])
    it = np.array([post])
    f = np.array([False])

    def consume(**kw):
        a = dict(platform="posts", viewers=v, items=it, watch_s=np.array([dur]), completed=f, shared=f, followed=f,
                 liked=f)
        a.update(kw)
        return lambda ctx: AUD.consumption(ctx, **a)
    seq = w.seq
    for bad in (consume(),                                        # never shown to this viewer
                consume(platform="articles"),                     # not on this platform
                consume(items=np.array([ni + 5]))):               # no such item
        with pytest.raises(InvalidAction):
            _sys(w, bad)
    assert w.seq == seq                                           # refused commands leave no trace
    a_pre = AUD.audience(w, post)
    _sys(w, lambda ctx: AUD.impressions(ctx, "posts", v, it, np.array([0])))
    for bad in (consume(watch_s=np.array([dur + 5.0])),          # longer than the item
                consume(watch_s=np.array([0.2 * dur]), completed=np.array([True])),
                consume(watch_s=np.array([0.0]), shared=np.array([True])),
                consume(watch_s=np.array([-1.0]))):
        with pytest.raises(InvalidAction):
            _sys(w, bad)
    a0 = AUD.audience(w, post)
    _sys(w, consume(watch_s=np.array([dur]), completed=np.array([True]), followed=np.array([True])))
    a1 = AUD.audience(w, post)
    assert a1["views"] == a0["views"] + 1 and a1["unique_viewers"] == a0["unique_viewers"] + 1
    assert a1["completions"] == a0["completions"] + 1
    assert a1["impressions"] == a_pre["impressions"] + 1 and a1["unique_exposed"] == a_pre["unique_exposed"] + 1


def test_no_tool_writes_a_counter(world):
    """Every tool of aud and feed, called by a creator on its own item with any arguments, leaves the item's
    counters unchanged unless it records the creator's own exposure (which is never a view)."""
    w = World.from_blob(world.snapshot_blob(), world_id="t-aud-fork", fork=True)
    rec = sorted(w.state.tables["crowd_creators"].items(), key=lambda kv: int(kv[0]))[0][1]
    creator = rec["principal"]
    mine = [int(i) for i in w.state.find_ids("aud_item_info", "creator", creator)]
    before = {i: AUD.audience(w, i) for i in mine}
    for tool, ep in sorted(w.endpoints.items()):
        if ep.host not in ("aud", "feed"):
            continue
        for i in mine[:2]:
            w.execute(creator, {"tool": tool, "args": {"item": i, "seconds": 999.0, "share": True, "follow": True,
                                                       "like": True, "platform": "posts", "q": "x", "k": 5,
                                                       "recommenders": []}})
    for i in mine:
        after = AUD.audience(w, i)
        for q in ("views", "unique_viewers", "completions", "repeat_views", "shares", "subscribers",
                  "returning_viewers", "watch_s"):
            assert after[q] == before[i][q], (i, q)


def test_recount_is_exact_and_independent(world):
    rc = AUD.recount(world)
    assert rc["total_mismatches"] == 0 and rc["views_without_prior_exposure"] == 0
    assert rc["impression_rows"] > 1000 and rc["distinct_viewers_checked"] > 100
    # the recount really is independent: tampering with one state counter, or one member of an exact set, is
    # detected (a fork has the same state; its history is the original's log)
    w = World.from_blob(world.snapshot_blob(), world_id="t-aud-tamper", fork=True)
    assert AUD.recount(w, events=world.log)["total_mismatches"] == 0
    i = int(np.argmax(AUD.item_col(w, "views")))
    w.state.arrays[f"{AUD.ITEMS}.views"][i] += 1
    assert AUD.recount(w, events=world.log)["mismatches"]["views"] == 1
    w.state.arrays[f"{AUD.ITEMS}.views"][i] -= 1
    from crucible.social import idset
    name = idset.key("vw", i)
    a = w.state.arrays[name]
    if idset.is_dense(a):
        a[0] ^= 1
    else:
        w.state.arrays[name] = np.sort(np.r_[a[1:], np.int32(max(int(a.max()) + 1, 0))]).astype(a.dtype)
    assert AUD.recount(w, events=world.log)["member_mismatches"]["viewers"] == 1


def test_analytics_are_delayed_noisy_and_private(world):
    w = world
    rec = sorted(w.state.tables["crowd_creators"].items(), key=lambda kv: int(kv[0]))[0][1]
    creator = rec["principal"]
    mine = [int(i) for i in w.state.find_ids("aud_item_info", "creator", creator)]
    i = max(mine, key=lambda x: AUD.audience(w, x)["views"])
    exact = AUD.audience(w, i)
    seen = AUD.audience(w, i, principal=creator)
    assert seen["exact"] is False and seen["as_of"] is not None and seen["as_of"] < w.now
    assert set(AUD.QUANTITIES) <= set(seen)
    r1 = w.execute(creator, {"tool": "aud.analytics", "args": {"item": i}})
    r2 = w.execute(creator, {"tool": "aud.analytics", "args": {"item": i}})
    assert r1.ok and r1.result == r2.result                      # one noisy draw per publication
    other = next(int(x) for x in w.state.tables["aud_item_info"] if int(x) not in mine)
    assert not w.execute(creator, {"tool": "aud.analytics", "args": {"item": other}}).ok
    pub = w.execute(creator, {"tool": "aud.item", "args": {"item": other}}).result
    assert set(pub) >= {"views", "title", "ref"} and "unique_viewers" not in pub
    assert exact["exact"] is True


def test_replay_regenerate_and_fork_are_exact():
    spec = X.spec("t-aud-det", 9, n=300, creators=8, posts_per_day=4, platforms={
        "posts": {"k": 6, "recommenders": [{"name": "interest"}, {"name": "engagement", "params": {"delay_s": 3600}},
                                           {"name": "cf", "params": {"min_pairs": 10}}, {"name": "explore"}]},
        "articles": {"k": 4, "recommenders": [{"name": "diversity"}, {"name": "reputation"}]}},
        feed={"train_interval_s": 4 * 3600})
    # not strict: a strict world registers validator subscribers that shift ctx.rng counters, and
    # World.regenerate always rebuilds a non-strict world (substrate issue reported in docs/v5/workstreams/C2.json)
    w = World.create(spec)
    X.run(w, 16 * 3600, step=3600)
    fits = [e for e in w.log if e.type == "feed.model"]
    assert fits                                                  # learned recommenders were fitted (recorded)
    evs = [e.to_dict() for e in w.log]
    r = World.replay(spec, evs)
    assert r.digest() == w.digest() and r.last_hash == w.last_hash
    g = World.regenerate(spec, evs)
    assert g.last_hash == w.last_hash and g.digest() == w.digest()
    f = World.from_blob(w.snapshot_blob(), world_id="t-fork", fork=True)
    assert f.digest() == w.digest()
    X.run(w, 20 * 3600, step=3600)
    X.run(f, 20 * 3600, step=3600)
    assert f.digest() == w.digest()                              # a fork continues identically
