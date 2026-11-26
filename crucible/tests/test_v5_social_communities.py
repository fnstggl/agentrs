"""Regression test (workstream C, wave 3b; MASTER_SPEC 4.7 member-to-member interaction): in the population's
community activity (`pop_social`, social.popsocial), members who visit a community read its posts through the
consumption engine and may reply; the author of a post that gets a reply is notified and comes back to its
communities (a recorded `pop.wake` for `social.community`), the way a message brings its recipient back.

A small social world of the one population; the community, its first post and its members are set up through the
community module's and the catalogue's own APIs (an experimenter's setup, labelled), then the members' visit runs the
real activity handler as the population's tick runs it. CPU (texts of the fixed seed are cached after the first run)."""
from __future__ import annotations

import numpy as np
import pytest

import crucible
from crucible.core.kernel import World
from crucible.social.worlds import social_world
from crucible.substrate import artifacts, ids

crucible.load_all_modules()
pytestmark = [pytest.mark.timeout(1800)]


def test_a_reply_in_a_community_brings_the_author_back():
    from crucible import population
    from crucible.population import core as PC, writing
    from crucible.social import communities as CM
    from crucible.social import content as C
    w = World.create(social_world("development", 9, n=150, platforms=["posts", "community"],
                                  creators={"prior": -30.0, "inbox_prior": -30.0}), strict=True)
    w.advance_time(3600)
    author, members = 0, np.arange(1, 61, dtype=np.int64)
    rng = np.random.default_rng(3)
    made = {}

    def setup(ctx):
        p = ids.ensure_principal(ctx, author)
        cid = w.modules["community"].found(ctx, p, "harbour-talk", "Harbour Talk", "boats, harbours and tides",
                                           "People who talk about the harbour.")
        for i in members:
            ids.ensure_principal(ctx, int(i))
        n = int(w.state.tables["communities"][cid]["n"])
        w.modules["community"].join_many(ctx, members, np.full(members.size, n, np.int64), "population")
        for k in range(3):
            title, body = writing.compose(rng, 3 + k, 4, 0.9)
            ref = artifacts.put(ctx, f"{title}\n\n{body}\n".encode("utf-8"), "text/plain", owner=p,
                                meta={"title": title[:80]}, kind="text")
            C.module(w).publish(ctx, p, ref, "post", "community", title, community=cid, access="public")
        made["cid"] = cid
    w._run_system("experiment.setup", setup)
    cid = made["cid"]
    assert CM.members(w, int(w.state.tables["communities"][cid]["n"])).size == members.size + 1
    reg = PC._registry(w)["spec"]["social.community"]
    before = len([e for e in w.log if e.type == "content.comment"])

    def visit(ctx):
        with PC.running_as(ctx, reg["owner"]):
            return reg["handler"](ctx, members, ctx.now)
    w._run_system("experiment.visit", visit)
    replies = [e for e in w.log if e.type == "content.comment"][before:]
    assert replies, "no member replied to the community's posts"
    wakes = [e for e in w.log if e.type == "cols.batch" and e.payload.get("kind") == "pop.wake"
             and (e.payload.get("extra") or {}).get("activity") == "social.community"]
    assert wakes, "the author was not notified of the replies"
    from crucible.substrate import columns
    slot = reg["slot"]
    assert int(columns.col(w, "pop", "wake_act")[author]) == slot
    assert int(columns.col(w, "pop", "wake_at")[author]) <= w.now + 3600
    # the author comes back: its next activation is the community activity, run by the population's own tick
    w.advance_time(w.now + 3 * 3600)
    assert w.check_invariants() == []
    assert not [e for e in w.log if e.type == "proc.failed"]
    _ = population
