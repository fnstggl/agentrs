"""Families that became prediction tasks or were added in the second wave-3b continuation (D-63/D-64/D-65
amendments): dl_delayed_adoption (new, 8.6.3), dl_short_term_damage v2, eng_delayed_controller v2 and
ch_tech_transition v3. Each test exercises the real world and its evaluator-side pieces."""
from __future__ import annotations

import numpy as np
import pytest

from crucible.rapid import claims_w3b as CW
from crucible.rapid import families as F
from crucible.rapid.categories import constants_fail_after_change
from crucible.rapid.dominance import search
from crucible.rapid.engines.primitive_graph import true_response
from crucible.rapid.references import play


def _truth(env):
    obj = env._config["objective"]
    return true_response(env._engine, obj["heldout"], obj["gauge"], ticks=int(obj.get("settle_ticks", 1)))


@pytest.mark.parametrize("seed", [0, 11])
def test_delayed_adoption_mechanism_claims_hold(seed):
    r = CW.claim_adoption_delayed_and_amplified(F.make("dl_delayed_adoption", seed))
    assert r["ok"], r
    assert r["first_awareness_effect_at"] == r["lag"] and r["adopters_under_low_spend"] == 0.0
    lo, hi = r["new_adopters_at_5_and_15_percent"]
    assert hi > lo > 0                                       # word of mouth from the installed base


def test_delayed_adoption_truth_has_the_threshold_and_saturation():
    env = F.make("dl_delayed_adoption", 0)
    t = _truth(env)
    held = [list(c.values())[0] for c in env._config["objective"]["heldout"]]
    assert held == sorted(held) and all(b >= a for a, b in zip(t, t[1:]))       # more spend, more adopters
    assert t[0] == 0.0 and t[-1] > 100.0                                        # gated below, saturating above
    assert t[-1] - t[-2] < t[2] - t[1]                                          # diminishing returns


@pytest.mark.parametrize("family", ["dl_delayed_adoption", "dl_short_term_damage", "eng_delayed_controller",
                                    "ch_tech_transition"])
def test_prediction_reference_beats_naive_predictors(family):
    seed = 0
    v_ref = play(F.make(family, seed))["value"]
    simple = search(F.make(family, seed), seed=seed)
    assert v_ref > 0.5 and simple["best_value"] <= 0.8 * v_ref, (v_ref, simple["best_label"], simple["best_value"])
    assert "naive_prediction" in simple["by_arm"] or simple["best_value"] <= 0.0


def test_wear_latch_identification_and_claim():
    env = F.make("dl_short_term_damage", 2)
    r = CW.identify_latch_level(env)
    assert r["ok"], r
    t = _truth(F.make("dl_short_term_damage", 2))
    held = [list(c.values())[0] for c in F.make("dl_short_term_damage", 2)._config["objective"]["heldout"]]
    intact = [y for d, y in zip(held, t) if d <= 2.0]
    worn = [y for d, y in zip(held, t) if d >= 7.0]
    assert min(intact) > 0 and max(worn) < max(intact) * 1.5     # long-run output of overdriven machines collapses


def test_delayed_plant_dead_time_is_identified():
    for seed in (0, 10):
        r = CW.identify_heater_delay(F.make("eng_delayed_controller", seed))
        assert r["ok"], r


def test_tech_transition_constants_must_be_revised():
    # the established process is the best constant early; after it becomes obsolete that constant falls short
    oks = [constants_fail_after_change("ch_tech_transition", s)["ok"] for s in (0, 1, 2)]
    assert sum(oks) >= 2, oks
    env = F.make("ch_tech_transition", 0)
    new = env._config["claim"]["new"]
    cost = next(n["lever"]["cost"] for n in env._config["nodes"] if n["id"] == new)
    assert cost > 0
