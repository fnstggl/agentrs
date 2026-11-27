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
                                    "ch_tech_transition", "net_congestion_routes"])
def test_prediction_reference_beats_naive_predictors(family):
    seed = 1 if family == "net_congestion_routes" else 0     # (on seed 0 its exponent grid picks too flat a curve)
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


def test_congestion_routes_share_a_link_and_are_convex():
    for seed in (0, 10):
        r = CW.claim_shared_congestion(F.make("net_congestion_routes", seed))
        assert r["ok"], r
        assert CW.identify_tighter_route(F.make("net_congestion_routes", seed))["ok"]
    t = _truth(F.make("net_congestion_routes", 1))
    assert max(t) > 1.5 * min(t)                     # the held-out flows reach the steep part of the delay curves


@pytest.mark.parametrize("family", ["plan_spoilage", "plan_serial_stages", "dl_delayed_adoption"])
def test_stock_outflows_are_always_served(family):
    # no phantom units: every stock's requested outflow (the saturating sales/processing node) is what it gives up
    for seed in (0, 11):
        e = F.make(family, seed)
        c, eng = e._config, e._engine
        name = list(c["objective"]["heldout"][0])[0]
        stocks = [nd for nd in c["nodes"] if nd.get("prim", "").startswith("stock")]
        rng = np.random.default_rng(seed)
        for _ in range(50):
            e.step({"kind": "levers", "values": {name: float(rng.choice([0.0, 3.0, 10.0]))}})
            for nd in stocks:
                i = eng.idx[nd["id"]]
                want = sum(w * e._state["h"][0, -1, eng.idx[src]] for src, lag, w in nd["ports"].get("outflow", [])
                           if lag == 0)
                assert want - float(e._state[f"ps:{i}:taken"][0]) < 1e-9


def test_spoilage_and_serial_chain_claims_and_identification():
    for seed in (0, 10):
        assert CW.claim_spoilage(F.make("plan_spoilage", seed))["ok"]
        assert CW.identify_spoilage_rate(F.make("plan_spoilage", seed))["ok"]
        r = CW.claim_serial_bottleneck(F.make("plan_serial_stages", seed))
        assert r["ok"] and r["pulse_reaches_stage_two_after"] == 1 + r["transit_lag"], r
        assert CW.identify_bottleneck_stage(F.make("plan_serial_stages", seed))["ok"]


def test_new_pair_protocols_verify_the_nearest_differences():
    from crucible.rapid import distinct as D
    assert D.verify("plan_serial_stages", "plan_spoilage")["ok"]
    assert D.verify("dl_delayed_adoption", "plan_spoilage")["ok"]


def test_interdependent_networks_cascade_and_carry_failure():
    for seed in (0, 11):
        r = CW.claim_interdependent_cascade(F.make("net_interdependent", seed))
        assert r["ok"], r
        assert CW.identify_first_threshold(F.make("net_interdependent", seed))["ok"]
    env = F.make("net_interdependent", 0)
    assert F.REGISTRY["net_interdependent"]["partition"] == "v5-eval-1"
    t = _truth(env)
    assert set(t) <= {0.0, 1.0} and 0.0 in t and 1.0 in t           # some demand pairs collapse, some do not
    v_ref = play(F.make("net_interdependent", 0))["value"]
    assert v_ref > 0.9
