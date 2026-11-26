"""Integration test (workstream C, wave 3b; D-34, SP-5.5b input): the short-video platform of a social world decides
with workstream E's fitted model (content_open@1, through crucible.calibration.runtime) and runs a random-exposure
experiment whose labelled decisions `crucible.calibration.inworld.random_exposure_experiment` reads.

A strict, typed world (the civilisation setting) of 800 population members on posts and shorts for two simulated
days; the insertion rate is raised to 0.3 (evaluator-side configuration of the experiment) so that a small catalogue
yields enough randomly exposed rows. GPU slot recommended (the population's text path)."""
from __future__ import annotations

import collections

import numpy as np
import pytest

import crucible
from crucible.core.kernel import World
from crucible.social.worlds import MIXTURES, social_world

crucible.load_all_modules()
pytestmark = [pytest.mark.timeout(3600)]


def test_random_exposure_decisions_are_labelled_and_read_by_the_calibration_analysis():
    from crucible.calibration import inworld
    from crucible.social import audience as AUD
    shorts = {"recommenders": MIXTURES["shorts"], "random_insert": {"rate": 0.3, "max_age_days": 30}}
    w = World.create(social_world("development", 31, n=800, platforms=["posts", "shorts"],
                                  feed={"platforms": {"shorts": shorts}}, creators={"prior": 0.8},
                                  params={"typed_events": True}), strict=True)
    w.advance_time(2 * 86400)
    assert not [e for e in w.log if e.type == "proc.failed"]
    arms = collections.Counter()
    for b in w.log:
        if b.type == "cols.batch" and b.payload.get("kind") == "open.shorts":
            ex = b.payload["extra"]
            assert ex["decision"]["model_id"] == "content_open"          # decision provenance of the fitted model
            arms[ex["experiment"]["arm"]] += int(ex["rows"])
            assert ex["experiment"]["id"] == "random_exposure"
    assert arms["random"] > 0 and arms["recommended"] > arms["random"], arms
    # the randomly exposed slots are impressions of source `random`, recorded like any other
    from crucible.social import experiments as X
    imp = X.decision_rows(w, "imp", 0, platform="shorts")
    assert int(np.sum(imp["src"] == AUD.SOURCE_CODE["random"])) >= arms["random"]
    r = inworld.random_exposure_experiment(w)
    assert r["rows"] == arms["random"] and 0.0 <= r["open_rate"] <= 1.0
    assert np.isfinite(r["z_open_rate"]) and isinstance(r["matching"], bool)
    assert w.check_invariants() == []
