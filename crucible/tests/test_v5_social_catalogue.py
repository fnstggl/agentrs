"""Regression test (workstream C, wave 3b; D-21 content_artifact migration): the audience module keeps no copy of the
catalogue's metadata for catalogue items; `audience.info` reads it from `content.content`, and a copy is an invariant
violation. Also: the population's own creation reaches several platforms, and its decisions carry provenance."""
from __future__ import annotations

import collections

import numpy as np
import pytest

import crucible
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social.worlds import social_world
from crucible.substrate import ids

crucible.load_all_modules()
pytestmark = [pytest.mark.timeout(1800)]


@pytest.fixture(scope="module")
def world():
    w = World.create(social_world("development", 9, n=150, platforms=["posts", "kb", "social", "letters"],
                                  creators={"prior": 0.8, "inbox_prior": 0.6}), strict=True)
    w.advance_time(30 * 3600)
    return w


def test_catalogue_items_keep_no_metadata_copy(world):
    w = world
    rows = w.state.tables["aud_item_info"]
    assert rows, "no distributed item"
    for k, r in rows.items():
        assert r["content_id"] is not None and not any(f in r for f in AUD.CATALOGUE_FIELDS), r
        c = w.state.tables["content"][r["content_id"]]
        merged = AUD.info(w, int(k))
        assert (merged["title"], merged["creator"], merged["platform"], merged["kind"], merged["ref"]) == \
            (c["title"], c["creator"], c["platform"], c["kind"], c["ref"])
        assert int(k) in AUD.items_of(w, c["creator"])
    assert w.check_invariants() == []
    f = w.fork("copy")
    k = next(iter(rows))
    f.state.patch("aud_item_info", k, title="a copy")             # what the invariant must refuse
    assert any("copy of the catalogue" in e for e in f.check_invariants())


def test_population_creates_on_several_platforms_with_provenance(world):
    w = world
    plats = collections.Counter(c["platform"] for c in w.state.tables["content"].values()
                                if ids.index(c["creator"]) is not None)
    assert len(plats) >= 3, plats
    from crucible.population import check_provenance
    pr = check_provenance(w, w.events_range(0))
    assert pr["forced_rows"] == 0 and pr["missing_provenance_batches"] == 0, pr
    kinds = {e.payload["kind"] for e in w.log if e.type == "cols.batch"}
    assert "pop_social.created" in kinds and "pop_social.inbox" in kinds
    rc = AUD.recount(w, events=w.events_range(0))
    assert rc["total_mismatches"] == 0 and rc["views_without_prior_exposure"] == 0
    assert int(np.sum(AUD.item_col(w, "views"))) > 0


def test_follows_have_one_owner_the_social_graph(world):
    """D-21 relationship migration (audience side): with the one population and the social network, a follow decided
    in the consumption engine is an edge of the social graph and nowhere else - the audience module keeps no
    follower sets (`fo:<creator>`) and no follower count; its `followers` / `is_follower` read the graph, and its
    per-item `subscribers` count the follow decisions, each of which is one new edge."""
    from crucible.social import graph, nodes
    from crucible.social.recsys import features as F
    from crucible.substrate import facts
    w = world
    assert AUD.graph_follows(w)
    assert "aud_creators.followers" not in w.state.arrays
    assert not [k for k in w.state.arrays if k.startswith("audset.fo.")]
    assert not any(k.startswith("aud.cols:aud_creators.followers") for k in facts.world_items(w))
    v = graph.FOLLOW.view(w)
    feed_edges = v["active"] & (v["via"] == graph.VIA_CODE["feed"])
    assert int(feed_edges.sum()) > 0, "no follow decided in a feed session"
    # every follow decided after consuming an item is one edge of the graph (and the recount agrees)
    assert int(np.sum(AUD.item_col(w, "follows"))) == int(feed_edges.sum())
    rc = AUD.recount(w, events=w.events_range(0))
    assert rc["mismatches"]["subscribers"] == 0
    # the audience module's follower API is a view of the graph
    codes = np.unique(AUD.item_col(w, "creator")[:AUD.n_items(w)])
    codes = codes[codes >= 0]
    for c in codes:
        node = F.creator_nodes(w, np.array([c]))[0]
        if node < 0:
            continue
        g = graph.followers(w, int(node))
        assert np.array_equal(AUD.followers(w, int(c)), g[g < nodes.OFFSET])
    some = AUD.followers(w, int(codes[0]))
    probe = np.arange(min(50, AUD.n_people(w)), dtype=np.int64)
    assert np.array_equal(AUD.is_follower(w, np.full(probe.size, codes[0]), probe), np.isin(probe, some))
