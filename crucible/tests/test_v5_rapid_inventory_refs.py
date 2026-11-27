"""Regression tests for the 8.4.1 inventory references and oracles (wave 3b continuation): the public references of
plan_distribution and des_inventory must not buy stock that can no longer be sold before the end of the episode,
des_inventory's reference must batch its orders (every order pays a fixed cost), and the evaluator oracles must
remain upper references for the public references on the admission seeds."""
from __future__ import annotations

import numpy as np
import pytest

from crucible.rapid import families as F
from crucible.rapid.identify import PublicView
from crucible.rapid.oracles import oracle
from crucible.rapid.references import inventory, play


def _record_orders(env, lever: str) -> list[tuple[int, float]]:
    """Wrap env.step so that every order quantity sent through `lever` is recorded with its tick."""
    orig = env.step
    orders: list[tuple[int, float]] = []

    def step(action):
        acts = action.get("steps", [action]) if isinstance(action, dict) else []
        for a in acts:
            if a.get("kind") == "levers" and lever in a.get("values", {}):
                orders.append((env.t, float(a["values"][lever])))
        return orig(action)
    env.step = step
    return orders


@pytest.mark.parametrize("seed", [0, 11])
def test_distribution_reference_stops_ordering_when_orders_cannot_arrive_in_time(seed):
    env = F.make("plan_distribution", seed)
    c = env._config
    orders = _record_orders(env, c["order_name"])
    from crucible.rapid.evaluator_h2 import reference_distribution
    reference_distribution(PublicView(env), env)
    late = [q for t, q in orders if t + int(c["lead"]) + 2 >= env.horizon]
    assert orders and sum(q for _, q in orders) > 0
    assert all(q == 0.0 for q in late), late
    # the reference (closed loop, public information) improves on doing nothing by a wide margin
    wait = F.make("plan_distribution", seed)
    while not wait.done:
        wait.step(None)
    assert env.outcome()["value"] > wait.outcome()["value"] + 2000


def test_distribution_oracle_is_an_upper_reference_on_the_admission_seeds():
    for seed in (0, 1, 2):
        v_or = oracle(F.make("plan_distribution", seed))["value"]
        v_ref = play(F.make("plan_distribution", seed))["value"]
        assert v_or >= v_ref - 1e-6, (seed, v_or, v_ref)


@pytest.mark.parametrize("seed", [0, 10])
def test_inventory_reference_batches_orders_and_stops_before_the_end(seed):
    env = F.make("des_inventory", seed)
    c = env._config
    orders = _record_orders(env, c["order_name"])
    inventory(PublicView(env), env)
    placed = [(t, q) for t, q in orders if q > 0]
    # every order pays the fixed order cost: orders are batched, far fewer than the ticks of the episode
    assert 3 <= len(placed) <= env.horizon // 3, len(placed)
    # batches are larger than one tick's demand (the economic order quantity of the fixed cost and the holding cost)
    assert np.mean([q for _, q in placed]) > 1.5 * float(c["demand"])
    # nothing is ordered in the last ticks, where a delivery (at least the minimum lead time) could not be sold
    assert all(t + int(c["lead_min"]) < env.horizon for t, _ in placed)


def test_inventory_reference_reads_no_hidden_parameter():
    # the reference must behave identically when the hidden lead-time and season parameters are absent from the
    # config it can read (it only uses public names, the cost terms of the brief and the observations)
    env = F.make("des_inventory", 2)
    shadow = F.make("des_inventory", 2)
    public = {k: v for k, v in shadow._config.items() if k not in ("demand", "season_amp", "season_period",
                                                                  "season_phase", "lead_min", "lead_p")}

    class View:                                     # the same world, seen through a config without hidden keys
        _config = public
    inventory(PublicView(env), View)
    assert env.outcome()["value"] == play(F.make("des_inventory", 2))["value"]


def test_inventory_oracle_includes_batched_policies_and_bounds_the_reference():
    for seed in (0, 1, 2):
        orc = oracle(F.make("des_inventory", seed))
        v_ref = play(F.make("des_inventory", seed))["value"]
        assert orc["value"] >= v_ref - 1e-6, (seed, orc["value"], v_ref)


def test_distribution_world_accepts_a_replenishment_program():
    # Part 8.4 asks for programmatic actions: a base-stock and shipping rule deployed as code runs inside the world
    env = F.make("plan_distribution", 3)
    c = env._config
    src = (
        "def act(obs, caller):\n"
        "    x = obs['numbers']\n"
        f"    sites = {list(c['site_names'])!r}\n"
        "    pos = x.get('depot_stock', 0) + x.get('on_order', 0)\n"
        "    q = max(0, 40 - pos)\n"
        "    ship = {s: max(0, 6 - x.get(s + '.stock', 0)) for s in sites}\n"
        f"    return {{'kind': 'sequence', 'steps': [{{'kind': 'levers', 'values': {{{c['order_name']!r}: q}}}},\n"
        "            {'kind': 'allocate', 'amounts': ship}]}\n")
    assert "program" in env.action_space().kinds
    r = env.step({"kind": "program", "source": src, "ticks": 10})
    assert r["result"]["program"]["ticks_run"] == 10 and not r["result"]["program"]["error"], r["result"]
    assert env.t == 10 and env._state["in"] > 0                  # the program ordered stock
    assert env._state["out"] > 0                                  # and the shipped stock sold at the sites
    from crucible.rapid.categories import action_forms
    assert action_forms(["plan_distribution"])["programmatic"]


def test_drift_blanks_scores_only_doses_outside_the_settable_range():
    from crucible.rapid.categories import unobserved_conditions
    for seed in (0, 10):
        r = unobserved_conditions("lab_drift_blanks", seed)
        assert r["ok"] and r["outside_settable_range"] == r["heldout_conditions"] == 4, r
