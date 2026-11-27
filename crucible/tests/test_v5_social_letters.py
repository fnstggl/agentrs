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
