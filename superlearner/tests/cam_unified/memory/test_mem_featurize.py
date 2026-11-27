"""HashedEventEncoder: finite tokens for non-finite readings; entity ids never reach the hash."""
from __future__ import annotations

import math

import torch

from memhelpers import light_events
from superlearner.cam_unified.interfaces import Action, Observation, ObsKind
from superlearner.cam_unified.memory.featurize import HashedEventEncoder


def test_nonfinite_numbers_give_finite_event_tokens():
    torch.manual_seed(0)
    enc = HashedEventEncoder(32)
    evs = light_events(6)
    evs[1].observations[0].value = float("inf")
    evs[2].observations[0].value = float("nan")
    evs[3].result.cost = float("-inf")         # the actor-visible spend (the record's reward is not an input)
    evs[4].action = Action("set_lever", {"value": float("nan")})
    evs[5].result.duration = float("inf")
    out = enc(evs)
    assert torch.isfinite(out).all()
    feats = [enc.numeric(e)[0] for e in evs]
    assert all(math.isfinite(x) for f in feats for x in f)
    assert feats[0][-1] == 0 and all(f[-1] > 0 for f in feats[1:])      # the non-finite count is a feature
    legacy = HashedEventEncoder(32, view="legacy")                         # declared baseline: reads the reward
    evs[3].reward = float("-inf")
    assert torch.isfinite(legacy(evs)).all() and legacy.numeric(evs[3])[0][-1] > 0


def test_entity_ids_in_action_arguments_are_not_hashed():
    enc = HashedEventEncoder(32)
    a, b = light_events(2)
    for e, ent in ((a, "prod_17"), (b, "prod_93")):        # same event in two worlds with different entity ids
        e.observations = [Observation(ObsKind.NUMERIC, "", value=1.0, entity=ent)]
        e.action = Action("set_price", {"product": ent, "mode": "fast"})
    ta, tb = enc.tokens(a), enc.tokens(b)
    assert ta == tb and "argv:product=<entity>" in ta and not any("prod_" in t for t in ta)
    assert "argv:mode=fast" in ta                           # ordinary short arguments are still hashed
    b.action = Action("set_price", {"product": "prod_93", "mode": "slow"})
    assert enc.tokens(b) != ta
