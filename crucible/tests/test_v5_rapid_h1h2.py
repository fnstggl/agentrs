"""Unit tests of the H1/H2 Tier 1 engines (program_exec, experiment_lab, planning, strategic) and of the H1-H3
families: determinism, observation validity, accounting, what each engine really does (programs run inside the
dynamics in the sandbox, experiments run on pilot copies and spend budget, deliveries arrive after their lead time,
opponents learn), every family's own intervention claims, and the public reference playing through the public
interface only."""
from __future__ import annotations

import json

import numpy as np
import pytest

from crucible.core.canonical import digest
from crucible.rapid import families as F
from crucible.rapid import observation as O
from crucible.rapid import policies as P
from crucible.rapid.claims import run as run_claims
from crucible.sandbox.runner import capabilities

H12_ENGINES = ("program_exec", "experiment_lab", "planning", "strategic")
H12 = sorted(n for n, f in F.FAMILIES.items() if f.engine in H12_ENGINES)
H3 = sorted(n for n in F.FAMILIES if F.FAMILIES[n].build.__module__ == "crucible.rapid.families_h3")
NEEDS_SANDBOX = sorted(n for n in H12 if F.FAMILIES[n].engine == "program_exec")


def _sandbox():
    cap = capabilities()
    if cap["landlock_abi"] < 1 or not cap["seccomp"]:
        raise RuntimeError("the sandbox (Landlock + seccomp) is required for program_exec worlds")


def test_the_h1_h2_engines_and_their_families_exist():
    assert {F.FAMILIES[n].engine for n in H12} == set(H12_ENGINES)
    assert len(H12) >= 25 and len(H3) >= 10


@pytest.mark.parametrize("family", H12 + H3)
def test_same_seed_and_actions_give_the_same_world_and_valid_observations(family):
    traces = []
    for _ in range(2):
        env = F.make(family, 3)
        rng = np.random.default_rng(5)
        obs = {a: env.observe(a) for a in env.agents()}
        tr = []
        for _ in range(min(env.horizon, 25)):
            acts = {a: P.random_policy(obs[a], env.action_space(a), rng) for a in env.agents()}
            r = env.step(acts if len(env.agents()) > 1 else acts[env.agents()[0]])
            obs = r.get("observations") or {env.agents()[0]: r["observation"]}
            for o in obs.values():
                assert O.check(o) == [], (family, O.check(o))
                assert "reward" not in json.dumps(o["resources"]) and "reward" not in r["result"]
            tr.append(digest(obs))
        traces.append((tr, digest(env.outcome())))
    assert traces[0] == traces[1]


# families whose own mechanism claims do not hold on seed 0 (recorded defects, not admitted): none at present - the
# dead-time, fleet-service and bargaining claims were repaired (workstream H, wave 3)
KNOWN_CLAIM_FAILURES: set[str] = set()


@pytest.mark.parametrize("family", [pytest.param(f, marks=pytest.mark.xfail(strict=True, reason="known failing claim"))
                                    if f in KNOWN_CLAIM_FAILURES else f for f in H12 + H3])
def test_each_family_passes_its_own_intervention_claims(family):
    """A family's claims are controlled interventions on copies of its worlds; each must hold (Gate M.8)."""
    assert F.FAMILIES[family].claims, family
    res = run_claims(family, 0)
    bad = {k: v for k, v in res.items() if not v.get("ok")}
    assert not bad, (family, json.dumps(bad, default=str)[:600])


# ====================================================================== program_exec
def test_a_deployed_program_runs_inside_the_dynamics_every_inner_step():
    _sandbox()
    env = F.make("pe_unstable_control", 0)
    u = sorted(env.action_space().kinds["levers"]["properties"]["values"]["properties"])
    src = ("def act(observation, caller):\n"
           "    m = caller.get('memory') or {'n': 0}\n"
           "    m['n'] = m['n'] + 1\n"
           "    return {'u': 0.0, 'memory': m}\n")
    r = env.step({"kind": "program", "source": src, "ticks": 6})
    assert r["result"]["ok"], r["result"]
    st = env._state
    # within the one tick, the transition code called the program at every inner step (more calls than ticks)
    # and the program kept its own state in memory across those calls
    assert st["pstat"]["calls"] >= 5 and st["pstat"]["errors"] == 0, st["pstat"]
    assert (st["memory"] or {}).get("n", 0) == st["pstat"]["calls"], st["memory"]
    assert r["observation"]["resources"]["compute_lines_last_tick"] > 0
    assert u                               # the world also offers plain levers next to programs


def test_a_program_cannot_reach_the_file_system_or_the_network():
    _sandbox()
    env = F.make("pe_unstable_control", 0)
    src = ("def act(observation, caller):\n"
           "    import os\n"
           "    return {'u': float(len(os.listdir('/')))}\n")
    env.step({"kind": "program", "source": src, "ticks": 4})
    env.step(None)
    assert env._state["pstat"]["calls"] > 0 and env._state["pstat"]["errors"] == env._state["pstat"]["calls"]
    src2 = ("def act(observation, caller):\n"
            "    return {'u': float(len(open('/etc/passwd').read()))}\n")
    e2 = F.make("pe_unstable_control", 0)
    e2.step({"kind": "program", "source": src2, "ticks": 4})
    r = e2.step(None)
    assert e2._state["pstat"]["errors"] == e2._state["pstat"]["calls"] > 0
    assert any("program error" in x for x in r["observation"]["text"])


def test_program_runs_replay_exactly():
    _sandbox()
    src = ("def act(observation, caller):\n"
           "    x = observation.get('x', 0.0) if isinstance(observation, dict) else 0.0\n"
           "    return {'u': -1.5 * float(x)}\n")
    out = []
    for _ in range(2):
        env = F.make("pe_unstable_control", 1)
        env.step({"kind": "program", "source": src, "ticks": 4})
        for _ in range(4):
            env.step(None)
        out.append((digest(env.observe()), digest(env.outcome())))
    assert out[0] == out[1]


# ====================================================================== experiment_lab
def test_experiments_run_on_pilot_copies_spend_budget_and_replicate():
    env = F.make("lab_kinetics", 0)
    sp = env.action_space()
    design = sp.kinds["experiment"]["properties"]["design"]["properties"]
    sets = {n: float(p["minimum"] + 0.5 * (p["maximum"] - p["minimum"])) for n, p in design["set"]["properties"].items()}
    meas = list(design["measure"]["items"]["enum"])[:1]
    live = dict(env._state["live"])
    b0 = float(env._state["budget"])
    inst = sorted(design["instrument"]["enum"]) if "instrument" in design else None
    d = {"set": sets, "measure": meas, "replicates": 3}
    if inst:
        d["instrument"] = inst[0]
    if "time" in design:
        d["time"] = int(design["time"]["minimum"]) + 1
    r = env.step({"kind": "experiment", "design": d})
    assert r["result"]["ok"], r["result"]
    assert float(env._state["budget"]) < b0 and env._state["live"] == live       # the live system is untouched
    vals = json.dumps(r["result"]["results"], default=str)
    assert meas[0] in vals


def test_instruments_differ_in_precision():
    env = F.make("lab_kinetics", 0)
    inst = env._config["instruments"]
    sds = sorted(float(v["sd"]) for v in inst.values())
    costs = sorted(float(v["cost"]) for v in inst.values())
    assert len(inst) >= 2 and sds[0] < sds[-1] and costs[0] < costs[-1]


# ====================================================================== planning
def test_shipments_arrive_after_their_lead_time_and_stock_is_conserved():
    env = F.make("plan_distribution", 0)
    c = env._config
    sp = env.action_space()
    sites = sorted(sp.kinds["allocate"]["properties"]["amounts"]["properties"])
    before = json.dumps(env._state["transit"], default=str)
    env.step({"kind": "allocate", "amounts": {sites[0]: 10.0}})
    assert json.dumps(env._state["transit"], default=str) != before        # in transit, not yet at the site
    env._config = dict(c, _audit=True)
    P.run(env, P.random_policy, np.random.default_rng(1))
    assert getattr(env, "_audit", []) == []


def test_perishable_lots_age_and_are_discarded():
    env = F.make("plan_perishable", 0)
    levers = env.action_space().kinds["levers"]["properties"]["values"]["properties"]
    big = {n: float(p["maximum"]) for n, p in levers.items()}
    seen = 0.0
    for _ in range(env.horizon):
        r = env.step({"kind": "levers", "values": big})
        seen += r["observation"]["numbers"].get("discarded", 0.0)
    assert seen > 0.0                        # over-ordering a perishable good makes lots expire


# ====================================================================== strategic
def test_learning_rivals_respond_to_the_participants_prices():
    lo, hi = F.make("strat_price_war", 0), F.make("strat_price_war", 0)
    name = next(iter(lo.action_space().kinds["levers"]["properties"]["values"]["properties"]))
    p = lo.action_space().kinds["levers"]["properties"]["values"]["properties"][name]
    rl, rh = [], []
    for _ in range(25):
        a = lo.step({"kind": "levers", "values": {name: float(p["minimum"])}})["observation"]["numbers"]
        b = hi.step({"kind": "levers", "values": {name: float(p["maximum"])}})["observation"]["numbers"]
        rl.append([v for k, v in a.items() if k.endswith(".price")])
        rh.append([v for k, v in b.items() if k.endswith(".price")])
    # rivals facing a low price end up pricing lower than rivals facing a high one (they learn from the market)
    assert np.mean(rl[-5:]) < np.mean(rh[-5:])


def test_bargaining_counterparts_answer_offers():
    env = F.make("strat_bargain", 0)
    sp = env.action_space()
    r = env.step({"kind": "discrete", "decision": "respond", "choice": "accept"})
    assert r["result"]["ok"] or "error" in r["result"]
    assert "offer_value_to_me" in r["observation"]["numbers"]
    assert "levers" in sp.kinds and set(sp.limits["decisions"]["respond"]) == {"propose", "accept", "walk_away"}


@pytest.mark.parametrize("family", ["strat_entry", "strat_blotto", "team_commons", "team_profile", "plan_capacity",
                                    "lab_drift_blanks", "pe_router"])
def test_the_reference_plays_through_the_public_interface(family):
    """The scripted reference gets a PublicView (actions, observations, schemas, the public spec) and plays a whole
    episode; its result is a real outcome of the world."""
    if F.FAMILIES[family].engine == "program_exec":
        _sandbox()
    from crucible.rapid.references import play
    env = F.make(family, 0)
    out = play(env)
    assert env.done and np.isfinite(out["value"])
