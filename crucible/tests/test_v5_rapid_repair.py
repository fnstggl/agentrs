"""Regression tests for the wave-3b repair pass of workstream H (verifier findings on c09eb24a):

* the dominance search of prediction worlds includes generic hold-and-fit predictors (corners and middle of the
  settable box held for equal blocks, linear / additive quadratic / leave-one-out fits), not only random settings;
* compute in program worlds is billed in whole quanta per call, so the same spec and seed give the same outcome in
  fresh processes;
* the 8.10 requirement of a prediction family is measured on its scored channel;
* revenue in prediction worlds books realised quantities (a noisy view, scaled by the observation_noise axis);
* the spoilage claim has a margin; the inventory reference reads the cost terms from the public brief.

Each test exercises the real worlds and evaluator-side pieces."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from crucible.rapid import claims_w3b as CW
from crucible.rapid import dominance as D
from crucible.rapid import families as F
from crucible.rapid.categories import beliefs_fail_after_change, constants_fail_after_change
from crucible.rapid.engines import program_exec as PE
from crucible.rapid.engines.primitive_graph import true_response
from crucible.rapid.identify import PublicView
from crucible.rapid.references import inventory, play

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _truth(env):
    obj = env._config["objective"]
    return true_response(env._engine, obj["heldout"], obj["gauge"], ticks=int(obj.get("settle_ticks", 1)))


def _patched(family: str, mutate):
    """A context that rebuilds `family` with `mutate(config)` applied after its builder."""
    fam = F.FAMILIES[family]

    class Ctx:
        def __enter__(self):
            self.orig = fam.build
            fam.build = lambda rng, d, _b=self.orig: mutate(_b(rng, d))

        def __exit__(self, *exc):
            fam.build = self.orig
    return Ctx()


# ====================================================================== generic predictors
def test_hold_design_covers_the_corners_and_the_middle():
    props = {"a": {"minimum": 0.0, "maximum": 10.0}, "b": {"minimum": 1.0, "maximum": 5.0}}
    S = D.hold_settings(["a", "b"], props)
    assert {(s["a"], s["b"]) for s in S} >= {(0.0, 1.0), (0.0, 5.0), (10.0, 1.0), (10.0, 5.0), (5.0, 3.0)}
    assert len(D.hold_settings(["a"], props)) == 5


@pytest.mark.parametrize("family,seed", [("cx_balancing_loop", 0), ("plan_spoilage", 2)])
def test_prediction_search_runs_the_hold_and_fit_predictors(family, seed):
    res = D._naive_predictors(F.make(family, seed), np.random.default_rng(seed))
    assert {"mean", "linear", "hold_linear", "hold_quadratic", "hold_cv"} <= set(res)


@pytest.mark.parametrize("family", ["cx_balancing_loop", "plan_spoilage", "net_congestion_routes",
                                    "ch_tech_transition"])
def test_redesigned_families_beat_the_hold_and_fit_predictors(family):
    # verifier finding: these were within 0.8 of their reference under hold-and-fit on two of three seeds
    ratios = []
    for seed in (1, 10):
        v_wait = float(D.play(F.make(family, seed), lambda t, o: {})[0])
        v_ref = float(play(F.make(family, seed))["value"])
        nv = D._naive_predictors(F.make(family, seed), np.random.default_rng(np.random.SeedSequence([seed, 7919])))
        best = max(nv.values())
        assert v_ref > v_wait
        ratios.append((best - v_wait) / (v_ref - v_wait))
    assert max(ratios) <= 0.8, ratios


# ====================================================================== deterministic compute billing
def test_billing_is_quantised_per_call():
    assert PE.billed_units(0) == PE.billed_units(PE.QUANTUM_UNITS - 1) == PE.CALL_UNITS
    assert PE.billed_units(PE.QUANTUM_UNITS) == PE.CALL_UNITS + PE.QUANTUM_UNITS
    assert PE.billed_units(5 * PE.QUANTUM_UNITS + 7) == PE.CALL_UNITS + 5 * PE.QUANTUM_UNITS


_RUN = """
import json, sys
from crucible.rapid import families as F
from crucible.rapid.references import play
e = F.make(sys.argv[1], int(sys.argv[2]))
lines = []
orig = e.step
def step(a):
    r = orig(a)
    lines.append(int(r["observation"]["resources"].get("compute_lines_last_tick", 0) or 0))
    return r
e.step = step
print(json.dumps({"value": play(e)["value"], "lines": lines}))
"""


@pytest.mark.parametrize("family,seed", [("pe_stream_processor", 1), ("pe_contention_sched", 0)])
def test_program_world_outcomes_repeat_in_fresh_processes(family, seed):
    # verifier finding: the reference of pe_stream_processor seed 1 scored 1371.6, 1002.6, 221.1, ... across runs
    env = dict(os.environ, PYTHONPATH=SRC, OMP_NUM_THREADS="1", CRUCIBLE_METER_MEMO="off")
    runs = [json.loads(subprocess.run([sys.executable, "-c", _RUN, family, str(seed)], env=env, check=True,
                                      capture_output=True, text=True, timeout=600).stdout.strip().splitlines()[-1])
            for _ in range(2)]
    assert runs[0] == runs[1]
    assert sum(runs[0]["lines"]) > 0


# ====================================================================== 8.10 on the scored channel
def _cheaper_new_process(cfg):
    new = cfg["claim"]["new"]
    for n in cfg["nodes"]:
        if n["id"] == new:
            n["lever"]["cost"] *= 0.6
    return cfg


def test_tech_transition_revision_is_measured_on_predictions_not_income():
    # verifier finding: the old income-based check flipped when a cost that never enters the score was lowered
    for seed in (1, 12):
        base = constants_fail_after_change("ch_tech_transition", seed)
        t0 = _truth(F.make("ch_tech_transition", seed))
        with _patched("ch_tech_transition", _cheaper_new_process):
            mut = constants_fail_after_change("ch_tech_transition", seed)
            t1 = _truth(F.make("ch_tech_transition", seed))
        assert t0 == t1
        assert base["channel"] == "scored predictions" and base["ok"]
        assert mut == base


@pytest.mark.parametrize("family", ["hc_confounded", "mk_elastic", "cx_overdose", "cx_balancing_loop"])
def test_static_prediction_worlds_do_not_pass_the_revision_check(family):
    r = beliefs_fail_after_change(family, 0)
    assert not r["ok"] and r["relative_shortfall_of_early_truth"] < 1e-9


def test_drifting_taste_is_frozen_where_it_was_early():
    r = beliefs_fail_after_change("ch_drifting_ideal", 0)
    assert r["ok"] and r["relative_shortfall_of_early_truth"] > 0.25


# ====================================================================== realised revenue
def _revenue(family: str, seed: int, noise: float | None, ticks: int = 25):
    kw = {} if noise is None else {"observation_noise": noise}
    e = F.make(family, seed, **kw)
    eng = e._engine
    m = e._config["money"]
    props = e.action_space().kinds["levers"]["properties"]["values"]["properties"]
    rng = np.random.default_rng(1)
    rev, node = [], []
    for _ in range(ticks):
        vals = {k: float(p["minimum"] + rng.random() * (p["maximum"] - p["minimum"])) for k, p in props.items()}
        r = e.step({"kind": "levers", "values": vals})
        rev.append(r["observation"]["resources"]["revenue_cents_last_tick"])
        node.append(float(e._state["h"][0, -1, eng.idx[m["revenue_node"]]]))
    implied = np.minimum(np.maximum(np.asarray(node), 0.0), m["cap"]) * m["price"]
    return np.asarray(rev), implied


@pytest.mark.parametrize("family", ["plan_spoilage", "hc_confounded", "dl_short_term_damage", "mk_elastic"])
def test_prediction_revenue_is_a_noisy_view_scaled_by_the_noise_axis(family):
    rev, implied = _revenue(family, 0, None)
    rev10, implied10 = _revenue(family, 0, 3.0)
    on = implied > 0
    rel = np.abs(rev[on] / implied[on] - 1.0)
    rel10 = np.abs(rev10[implied10 > 0] / implied10[implied10 > 0] - 1.0)
    assert rel.max() > 1e-3                                   # no longer price x the noise-free node
    assert np.median(rel10) > 5 * np.median(rel)              # the observation_noise axis reaches revenue


def test_scalar_twin_still_matches_with_revenue_draws():
    from crucible.rapid.twin import compare
    for family in ("plan_spoilage", "hc_confounded"):
        r = compare(family, 0, ticks=20)
        assert r["ok"], (family, r)


def test_income_worlds_keep_exact_revenue():
    e = F.make("ch_silent_reversal", 0)                        # maximize_net: the score is the money itself
    assert "realisation_sd" not in e._config["money"]


# ====================================================================== the spoilage claim has a margin
def _linear_sales(cfg):
    s = next(n for n in cfg["nodes"] if n["id"] == cfg["claim"]["outcome"])
    s["params"]["cap"], s["params"]["half"] = 5000.0, 10000.0
    return cfg


def _barely_concave(cfg):
    s = next(n for n in cfg["nodes"] if n["id"] == cfg["claim"]["outcome"])
    s["params"]["cap"], s["params"]["half"] = 5000.0, 2000.0
    return cfg


@pytest.mark.parametrize("seed", [0, 2, 12])
def test_spoilage_claim_fails_without_saturation(seed):
    assert CW.claim_spoilage(F.make("plan_spoilage", seed))["ok"]
    for mutate in (_linear_sales, _barely_concave):
        with _patched("plan_spoilage", mutate):
            r = CW.claim_spoilage(F.make("plan_spoilage", seed))
        assert not r["ok"], (mutate.__name__, r)


# ====================================================================== inventory terms are public
def test_inventory_terms_are_in_the_brief_and_the_reference_needs_no_config_costs():
    for family in ("des_inventory", "des_perishable"):
        env = F.make(family, 2)
        brief = json.dumps(env.spec["brief"])
        c = env._config
        assert f"each order costs {c['order_fixed']:g} cents" in brief
        assert f"every unit on hand costs {c['holding_cost']:g} cents per tick" in brief
    env = F.make("des_inventory", 2)
    names = ("order_name", "gauge_names", "max_order")               # public: the action schema and the gauges

    class View:                                     # the world seen through its public names only
        _config = {k: v for k, v in F.make("des_inventory", 2)._config.items() if k in names}
    inventory(PublicView(env), View)
    assert env.outcome()["value"] == play(F.make("des_inventory", 2))["value"]
