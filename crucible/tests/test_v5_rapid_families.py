"""The Tier 1 family registry and its integration: make(), difficulty axes, splits, 4.0 families untouched,
fingerprints, intervention tests, the public identification procedures, the 4.0 numeric lever view (existing
learners unchanged), the kernel-hosted twin, and sandboxed controller programs."""
from __future__ import annotations

import json

import numpy as np
import pytest

from crucible.rapid import families as F
from crucible.rapid import fingerprints as FP
from crucible.rapid.mechview import kernel_world, lever_families, lever_mech, make_task
from crucible.sandbox.runner import capabilities

CONTRACT_KEYS = {"engine", "category", "split", "fingerprint", "doc"}


def test_registry_entries_carry_the_contract_fields():
    assert len(F.REGISTRY) == len(F.FAMILIES) >= 12
    for name, e in F.REGISTRY.items():
        assert CONTRACT_KEYS <= set(e), name
        assert e["category"] in F.CATEGORIES and e["split"] in ("train", "val", "test")
        assert e["distinct_because"] and e["action_kinds"] and set(e["axes"]) <= set(F.DIFFICULTY_AXES)
    cats = {e["category"] for e in F.REGISTRY.values()}
    from crucible.rapid.engines import ENGINE_KINDS
    assert len(cats) == 10 and {e["engine"] for e in F.REGISTRY.values()} == set(ENGINE_KINDS)


def test_make_respects_difficulty_axes_and_refuses_unknown_ones():
    env = F.make("net_routing", 1, graph_size=6, horizon=30, observation_noise=0.5)
    assert env.horizon == 30 and len(env._config["route_names"]) == 6 and env.difficulty["graph_size"] == 6
    with pytest.raises(ValueError):
        F.make("net_routing", 1, opponent_capability=0.9)
    a, b = F.build_config("hc_confounded", 3), F.build_config("hc_confounded", 3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(F.build_config("hc_confounded", 4), sort_keys=True) != json.dumps(a, sort_keys=True)


def test_v5_families_are_new_and_the_4_0_partitions_are_untouched():
    from crucible.mechanisms.base import MECH_FAMILIES, mech_families
    from crucible.mechanisms.sim import ensure_engines
    ensure_engines()
    assert (len(mech_families(split="train")), len(mech_families(split="val")),
            len(mech_families(split="test"))) == (7, 7, 11)
    assert not set(F.FAMILIES) & set(MECH_FAMILIES)
    assert all(m.startswith("rapid:") for m in [lever_mech("hc_confounded", 0)["family"]])
    assert {F.REGISTRY[n]["partition"] for n in F.families(split="val")} == {"v5-eval-1"}


def test_the_spec_object_holds_no_hidden_configuration():
    from crucible.rapid.admission import secrets
    for fam in ("hc_confounded", "des_maintenance", "net_cascade", "posm_unlock", "srch_tools"):
        env = F.make(fam, 0)
        text = json.dumps(env.spec, default=str)
        assert not [s for s in secrets(env) if s in text]
        assert "claim" not in text and "truth" not in text
        # no identity: the family name (which describes the mechanism), the seed, the engine, the category, the
        # split (which marks evaluation worlds) and the difficulty setting stay evaluator side
        assert fam not in text and not {"family", "seed", "engine", "category", "split", "difficulty"} & set(env.spec)
        r = env.step({"kind": "query", "target": "spec"}) if "query" in env.action_space().kinds else None
        if r is not None:
            assert fam not in json.dumps(r, default=str) and '"val"' not in json.dumps(r["result"], default=str)


def test_observation_only_is_the_default_and_objective_free_operation_works():
    env = F.make("net_routing", 0)
    assert env.reward_mode == "observation_only" and env.spec["experience_mode"] == "observation_only"
    free = F.make("net_routing", 0, objective=None)
    assert free.spec["objective"] == {"a0": None} and free.observe()["objective"] is None
    for _ in range(free.horizon + 3):                                   # runs to the horizon and refuses after
        r = free.step(None)
        assert "reward" not in r and "reward" not in r["result"]
    assert free.done and not r["result"]["ok"]
    assert free.outcome()["objective_mode"] == "none"
    with pytest.raises(ValueError):
        F.make("net_routing", 0, reward_mode="no_such_mode")


def test_fingerprints_ignore_names_seeds_and_coefficients_but_not_structure():
    a = FP.instance_fingerprint(F.make("hc_confounded", 0))
    b = FP.instance_fingerprint(F.make("hc_confounded", 7))
    assert FP.distance(FP.merge([a]), FP.merge([b])) == 0.0
    c = FP.instance_fingerprint(F.make("hc_confounded", 0, n_variables=3))
    assert FP.distance(FP.merge([a]), FP.merge([c])) == 0.0       # distractors are not structure (M.n1)
    h = FP.instance_fingerprint(F.make("hc_confounded", 0, horizon=200))
    assert FP.distance(FP.merge([a]), FP.merge([h])) == 0.0       # nor is the horizon
    fps = {n: F.REGISTRY[n]["fingerprint"] for n in F.REGISTRY}
    kept, dup = FP.counted(fps)
    assert not dup and len(kept) == len(fps)
    near = FP.nearest("mk_elastic", fps)
    assert near["nearest"].startswith("mk_") and "motifs" in near["differs_in"]


def test_legacy_fingerprints_are_far_from_v5_families():
    from crucible.rapid.admission import distinctness
    d = distinctness("hc_conditional")
    assert d["ok"] and d["distance_4_0"] > 0 and d["nearest_4_0"]


@pytest.mark.parametrize("family", ["hc_confounded", "mk_gate", "des_bottleneck", "net_diffusion", "posm_unlock",
                                    "srch_distractors", "coop_split_info", "sci_functional_law"])
def test_intervention_claims_hold(family):
    from crucible.rapid.claims import run
    res = run(family, 1)
    assert res and all(r["ok"] for r in res.values()), res


def test_lookalike_markets_obey_different_rules_behind_one_interface():
    from crucible.rapid.claims import lookalike_matrix
    m = lookalike_matrix(2)
    assert m["all_own"] and m["distinct"] and m["same_interface"], m


@pytest.mark.parametrize("family", ["hc_confounded", "mk_signal", "des_maintenance", "srch_field", "posm_fault_diagnosis",
                                    "coop_split_info"])
def test_public_identification_answers_the_question(family):
    from crucible.rapid.identify import run
    oks = [run(family, s)["ok"] for s in (0, 1, 2)]
    assert sum(oks) >= 2, oks


def test_identification_gets_only_a_public_view():
    from crucible.rapid.identify import PublicView
    v = PublicView(F.make("net_routing", 0))
    with pytest.raises(AttributeError):
        v._state                                                       # noqa: B018
    with pytest.raises(AttributeError):
        v.outcome()


def test_oracle_beats_defaults_and_trivial_play():
    from crucible.rapid import policies as P
    from crucible.rapid.oracles import fork, oracle
    for fam in ("hc_conditional", "des_inventory", "posm_sign_switch", "srch_field"):
        env = F.make(fam, 0)
        o = oracle(fork(env))["value"]
        d = P.run(fork(env), P.wait)["value"]
        triv = max(P.run(fork(env), P.constant(a))["value"] for _, a in P.canonical_actions(env.action_space()))
        assert o > d and o >= triv, fam


# ---------------------------------------------------------------- the 4.0 numeric view
def test_lever_view_runs_existing_research_learners_unchanged():
    from crucible.research.envs import MechVecEnv, run_episodes
    from crucible.research.learner import EnvSpec
    from crucible.research.learners import make as make_learner
    fams = ["hc_confounded", "net_routing", "posm_sign_switch", "des_inventory", "mk_elastic"]
    assert set(fams) <= set(lever_families())
    tasks = [make_task(f, s) for f in fams for s in (0, 1)]
    spec = EnvSpec()
    for name in ("hold", "random", "probe_exploit", "hill_climb"):
        env = MechVecEnv(tasks, spec)
        learner = make_learner(name, spec, seed=0)
        recs = run_episodes(env, learner)
        assert len(recs) == len(tasks) and all(np.isfinite(r["net_cents"]) for r in recs)
        assert env.reset().shape == (len(tasks), spec.obs_dim)


def test_lever_view_learner_updates_its_weights():
    import torch
    from crucible.research.envs import MechVecEnv
    from crucible.research.learner import EnvSpec
    from crucible.research.learners import make as make_learner
    spec = EnvSpec()
    env = MechVecEnv([make_task("net_routing", s) for s in range(4)], spec)
    lr = make_learner("ppo_gru", spec, seed=0, rollout_steps=128)
    before = [p.detach().clone() for p in lr.net.parameters()]
    obs = env.reset()
    lr.reset_context(env.n)
    while not env.done.all():
        a = lr.act(obs, explore=True)
        nxt, rew, done, _ = env.step(a)
        lr.observe(obs, a, rew, done, nxt)
        obs = nxt
    out = lr.update()
    assert out["updates"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, lr.net.parameters()))


def test_validated_task_scores_against_the_private_reference():
    from crucible.research.envs import MechVecEnv, run_episodes
    from crucible.research.learner import EnvSpec
    from crucible.research.learners import make as make_learner
    t = make_task("posm_sign_switch", 0, validate=True)
    assert t.validated and t.reference_net > t.default_net
    pub = t.public()
    assert "reference" not in json.dumps(pub) and pub["goal_net_cents"] == t.goal_net_cents
    rec = run_episodes(MechVecEnv([t], EnvSpec()), make_learner("hold", EnvSpec()))[0]
    assert rec["score"] is not None


def test_lever_view_simulator_equals_the_rapid_world():
    from crucible.mechanisms.sim import MechSim
    from crucible.rapid.env import world_seed_of
    for fam in ("hc_confounded", "posm_sign_switch", "des_perishable"):
        mech = lever_mech(fam, 2)
        sim = MechSim(mech, world_seed_of(fam, 2), "r1")
        env = F.make(fam, 2)
        rng = np.random.default_rng(0)
        lev = sim.engine.levers(0)
        for t in range(25):
            x = {lv.name: float(lv.clip(rng.uniform(lv.lo, lv.hi))) for lv in lev}
            s = sim.step(x)
            r = env.step({"kind": "levers", "values": x})
            nums = r["observation"]["numbers"]
            for g, v in s.readings[0].items():
                assert nums.get(g, 0.0) == v, (fam, t, g)
            assert s.revenue[0] == float(round(r["observation"]["resources"]["revenue_cents_last_tick"]))


def test_kernel_twin_matches_the_batch_simulator_with_ledger_money():
    from crucible.core.kernel import World
    from crucible.mechanisms.sim import cents
    from crucible.rapid.batch import BatchRapid
    w, actor, mech = kernel_world("net_routing", 3)
    fast = BatchRapid("net_routing", [3])
    rng = np.random.default_rng(5)
    for t in range(20):
        vals = fast.lo + rng.random(fast.lo.shape) * (fast.hi - fast.lo)
        settings = {n: float(v) for n, v in zip(fast.lever_names[0], vals[0])}
        assert w.execute(actor, {"tool": "rig.set", "args": {"settings": settings}}).ok
        w.advance_time(w.now + 3600)
        rep = w.execute(actor, {"tool": "rig.read", "args": {}}).result["report"]
        out = fast.step(vals)
        assert [rep["readings"][g] for g in mech["config"]["rapid"]["gauge_names"]] == list(out["readings"][0])
        assert rep["revenue_cents"] == cents(out["revenue"][0]) and rep["cost_cents"] == cents(out["cost"][0])
    assert w.check_invariants() == []
    events = [e.to_dict() for e in w.events_range(0)]
    assert World.replay(w.spec, events).digest() == w.digest()
    spec = w.execute(actor, {"tool": "rig.spec", "args": {}}).result
    assert "c2" not in json.dumps(spec) and "assign" not in json.dumps(spec)


@pytest.mark.sandbox
def test_program_actions_run_a_sandboxed_controller():
    caps = capabilities()
    if caps["landlock_abi"] < 1 or not caps["seccomp"] or not caps["root"]:
        pytest.fail("the sandbox (Landlock + seccomp, run as root) is required for program actions")
    env = F.make("eng_delayed_controller", 1)
    h = next(iter(env.action_space().kinds["levers"]["properties"]["values"]["properties"]))
    tg = env._config["gauges"][0]["name"]
    src = ("def act(observation, caller):\n"
           f"    T = observation['numbers'].get('{tg}', 20.0)\n"
           f"    return {{'kind': 'levers', 'values': {{'{h}': 8.0 if T < 30 else 2.0}}}}\n")
    r = env.step({"kind": "program", "source": src, "ticks": 6})
    assert r["result"]["ok"] and r["result"]["program"]["ticks_run"] == 6 and env.t == 6
    assert all(a["kind"] == "levers" for a in r["result"]["program"]["actions"])
    bad = env.step({"kind": "program", "source": "def act(o, c):\n    return {'kind': 'program'}\n", "ticks": 2})
    assert not bad["result"]["ok"] and "allowed kinds" in bad["result"]["error"]
    crash = env.step({"kind": "program", "source": "def act(o, c):\n    raise RuntimeError('x')\n", "ticks": 2})
    assert not crash["result"]["ok"] and "RuntimeError" in crash["result"]["error"]
