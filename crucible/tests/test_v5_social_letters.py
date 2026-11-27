"""Regression test (workstream C, wave 3b): a newsletter issue a population creator sends reaches its population
subscribers' inboxes, and its impressions are recorded when a subscriber is shown the issue in its own session -
never in the sender's command (social.creators `letters.send`).

Gate C.3 (30-day population run) found 1,818 impression rows without decision provenance: `letters.send` recorded
an impression for every population subscriber at delivery time, inside the sender's command, so the rows concerned
individuals who were not deciding anything (`population.check_provenance`: forced rows). Setup (labelled): 120
members of a small social world subscribe to one member's newsletter (experimenter's assignment through
`letters.subscribe_many`); the member writes and sends one issue with the ordinary tools as its own principal; the
notified subscribers come back by themselves."""
from __future__ import annotations

import numpy as np
import pytest

import crucible
from crucible import population
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social.worlds import social_world
from crucible.substrate import artifacts, ids

crucible.load_all_modules()
pytestmark = [pytest.mark.timeout(900)]


def _letters_impressions(w) -> int:
    n = 0
    for e in w.log:
        if e.type == "cols.batch" and e.payload.get("kind") == "impress.letters":
            n += int(artifacts.get_array_blob(w, e.payload["ref"])["idx"].size)
    return n


def test_a_newsletter_issue_is_an_impression_only_when_a_subscriber_is_shown_it():
    w = World.create(social_world("development", 12, n=160, platforms=["posts", "letters"]), strict=True)
    w.advance_time(3600)
    letters = w.modules["letters"]
    author = 3
    subs = np.array([i for i in range(160) if i != author][:120], np.int64)
    out = {}

    def setup(ctx):
        p = ids.ensure_principal(ctx, author)
        out["nl"] = ctx.call("letters.create", {"name": "Field notes", "description": "what I read this week"},
                             as_principal=p)["newsletter"]
        ln = int(ctx.state.tables["newsletters"][out["nl"]]["n"])
        letters.subscribe_many(ctx, subs, np.full(subs.size, ln, np.int64))
        ref = ctx.call("studio.write", {"format": "text", "title": "Field notes 1",
                                        "body": "Three things I read this week about the local football league and "
                                                "the new stadium plans, with my own notes on each."},
                       as_principal=p)["ref"]
        out["send"] = ctx.call("letters.send", {"newsletter": out["nl"], "ref": ref, "title": "Field notes 1"},
                               as_principal=p)
    before = _letters_impressions(w)
    w._run_system("test.newsletter", setup)
    assert out["send"]["population_reached"] == subs.size            # delivered to every population subscriber
    assert _letters_impressions(w) == before                         # ... and nobody was "shown" it by the send
    item = int(w.state.tables["content"][out["send"]["item"]]["aud_item"])
    w.advance_time(w.now + 4 * 3600)                                  # the notified come back by themselves
    shown = AUD.exposed(w, item)
    assert shown.size > 0 and _letters_impressions(w) > before
    assert set(shown.tolist()) <= set(subs.tolist())                  # exposed: subscribers shown it in a session
    prov = population.check_provenance(w)
    assert prov["forced_rows"] == 0, prov
    assert w.check_invariants() == []


def test_a_creator_with_followers_opens_a_newsletter_and_a_tier_from_a_short_post(monkeypatch):
    """A creator whose audience reached the thresholds (newsletter 5, paid tier 15 followers) may open its channels
    whatever it writes now - most pieces are short posts - not only when it writes an article (before, a creator
    who never wrote articles never opened one, and young civilisations had no newsletter or membership at all).
    Setup (labelled): 16 members are assigned to follow one member (experimenter); the opening probabilities are set
    to 1 for the test; the world has posts and newsletters but no blog (no article medium)."""
    from crucible.social import graph as G, nodes, popinbox as PI
    w = World.create(social_world("development", 13, n=120, platforms=["posts", "letters"]), strict=True)
    w.advance_time(1800)
    author = 5
    fans = np.array([i for i in range(120) if i != author][:16], np.int64)

    def follow(ctx):
        p = ids.ensure_principal(ctx, author)
        G.follow_batch(ctx, fans, np.full(fans.size, nodes.node_of(w, p)), via="other")
    w._run_system("test.follow", follow)
    monkeypatch.setattr(PI, "NEWSLETTER_P", 1.0)
    monkeypatch.setattr(PI, "TIER_P", 1.0)
    pop_social = w.modules["pop_social"]
    p = ids.pid(author)
    before = len(w.state.tables.get("content", {}))
    w._run_system("test.create", lambda ctx: pop_social.act_create(ctx, np.array([author], np.int64), ctx.now))
    made = [c for c in w.state.tables["content"].values() if c["creator"] == p]
    assert made and all(c["platform"] == "posts" for c in made) and len(w.state.tables["content"]) > before
    assert w.state.find_ids("newsletters", "owner", p)                 # opened from a short post
    assert "creators" in w.modules
    assert w.state.find_ids("creator_tiers", "creator", p)              # 16 followers >= 15: a paid tier
    assert not [e for e in w.log if e.type == "proc.failed"]
    assert w.check_invariants() == []
