"""The Tier 1 engines: determinism, accounting, independence of their transition code, and what each one
really does (event queues, graph algorithms, categorical hidden-state transitions, exhaustible opportunities)."""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from crucible.core.canonical import digest
from crucible.rapid import families as F
from crucible.rapid import policies as P
from crucible.rapid.batch import BatchRapid, batchable
from crucible.rapid.engines import ENGINE_KINDS, SPEC_ENGINE_KINDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONE_PER_ENGINE = ["hc_conditional", "des_inventory", "net_diffusion", "posm_fault_diagnosis", "srch_tools"]


V5_ENGINES = {"primitive_graph", "discrete_event", "graph_dynamical", "state_machine", "search", "program_exec",
              "experiment_lab", "planning", "strategic"}
# what an engine module may import besides the contract: program_exec runs agent-written controllers inside the
# dynamics through the sandbox (Landlock + seccomp), which is infrastructure, not another engine's transition code
EXTRA_IMPORTS = {"program_exec": {"...sandbox.runner"}}


def test_nine_engines_registered_and_all_eight_spec_kinds_covered():
    assert set(ENGINE_KINDS) == V5_ENGINES
    assert all(v in ENGINE_KINDS for v in SPEC_ENGINE_KINDS.values())
    assert sorted(SPEC_ENGINE_KINDS) == sorted(["discrete-event", "graph dynamical", "search/information",
                                                "executable program", "multi-agent strategic",
                                                "scientific experiment", "planning/resource",
                                                "partially observed state machine"])
    assert len(set(SPEC_ENGINE_KINDS.values())) == 8 and "primitive_graph" not in SPEC_ENGINE_KINDS.values()


def test_engine_modules_share_only_the_contract():
    edir = os.path.join(ROOT, "src", "crucible", "rapid", "engines")
    for name in ENGINE_KINDS:
        tree = ast.parse(open(os.path.join(edir, f"{name}.py")).read())
        rel = {("." * n.level) + (n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level}
        absolute = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        allowed = {"..", ".base", "..primitives", "..actions"} | EXTRA_IMPORTS.get(name, set())
        assert rel <= allowed, (name, rel - allowed)        # the contract, nothing else
        assert not any(m.startswith("crucible") for m in absolute), (name, absolute)
        assert not rel & {f".{o}" for o in ENGINE_KINDS if o != name}, name      # no other engine's code


@pytest.mark.parametrize("family", ONE_PER_ENGINE + ["coop_split_info", "net_repair", "des_maintenance"])
def test_same_seed_and_actions_give_the_same_world(family):
    traces = []
    for _ in range(2):
        env = F.make(family, 4)
        rng = np.random.default_rng(9)
        tr = []
        obs = {a: env.observe(a) for a in env.agents()}
        while not env.done:
            acts = {a: P.random_policy(obs[a], env.action_space(a), rng) for a in env.agents()}
            r = env.step(acts if len(env.agents()) > 1 else acts[env.agents()[0]])
            obs = r.get("observations") or {"a0": r["observation"]}
            tr.append(digest(obs))
        traces.append((tr, digest(env.outcome())))
    assert traces[0] == traces[1]


@pytest.mark.parametrize("family", sorted(F.FAMILIES))
def test_accounting_holds_every_tick_under_random_play(family):
    env = F.make(family, 2)
    env._config = dict(env._config, _audit=True)
    P.run(env, P.random_policy, np.random.default_rng(3), agents={a: P.random_policy for a in env.agents()[1:]})
    assert getattr(env, "_audit", []) == []
    for a in env.agents():
        led = env._ledger
        assert abs(led["cash"][a] - (env._config["endowment"] + led["revenue"][a] - led["cost"][a])) < 1e-6


def test_discrete_event_engine_runs_an_event_queue_in_continuous_time():
    env = F.make("des_bottleneck", 0)
    for _ in range(20):
        env.step(None)
    st = env._state
    assert st["clock"] == 20.0 and st["heap"] and all(isinstance(e[0], float) for e in st["heap"])
    times = sorted(e[0] for e in st["heap"])
    assert times[0] >= 20.0
    assert st["arrived"] == st["completed"] + sum(st["q"]) + sum(st["busy"])
    inv = F.make("des_perishable", 1)
    inv.step({"kind": "levers", "values": {inv._config["order_name"]: 40.0}})
    for _ in range(30):
        inv.step(None)
    s = inv._state
    assert s["delivered"] - s["sold"] - s["expired"] == sum(q for q, _ in s["lots"])


def test_graph_engine_dynamics_use_the_network():
    env = F.make("net_diffusion", 0)
    A = np.asarray(env._config["adj"])
    cl = np.asarray(env._config["cluster"])
    within = A[cl[:, None] == cl[None, :]].mean()
    across = A[cl[:, None] != cl[None, :]].mean()
    assert within > 5 * across                                 # a clustered graph
    from crucible.rapid.engines.graph_dynamical import components
    rep = F.make("net_repair", 0)
    comp = components(rep._config["n_nodes"], [tuple(e) for e in rep._config["edges"]])
    assert len(set(comp)) == 1                                 # the network starts connected
    r = F.make("net_routing", 0)
    for _ in range(10):
        r.step({"kind": "levers", "values": {n: 1.0 for n in r._config["route_names"]}})
    s = r._state
    assert abs(s["arrived"][0] - s["delivered"][0] - s["q1"].sum() - s["q2"].sum() - s["o1"].sum()) < 1e-9


def test_state_machine_transitions_follow_their_matrix():
    env = F.make("posm_sign_switch", 0, horizon=4000)
    T = np.asarray(env._config["T"])
    counts = np.zeros_like(T)
    prev = int(env._state["s"][0])
    for _ in range(4000):
        env.step(None)
        s = int(env._state["s"][0])
        counts[prev, s] += 1
        prev = s
    emp = counts / counts.sum(axis=1, keepdims=True)
    assert np.abs(emp - T).max() < 0.03


def test_search_opportunities_are_used_up():
    env = F.make("srch_field", 0)
    env.step({"kind": "discrete", "decision": "dig", "choice": "c2_3"})
    v1 = env.outcome()["revenue"]
    env.step({"kind": "discrete", "decision": "dig", "choice": "c2_3"})
    assert env.outcome()["revenue"] == v1 and env._state["left"][2, 3] == 0.0


@pytest.mark.parametrize("family", [f for f in sorted(F.FAMILIES) if batchable(f)])
def test_batch_path_equals_single_worlds_bit_for_bit(family):
    seeds = [0, 1, 2, 3, 5, 8]
    try:
        b = BatchRapid(family, seeds)
    except ValueError as e:                                    # instances with different wiring cannot share a batch
        assert "differ" in str(e)
        groups = {}
        for s in range(12):
            cfg = F.build_config(family, s)
            key = digest([[n["id"], n.get("prim"), sorted((p, [(t[0], t[1]) for t in ts]) for p, ts in
                                                         n.get("ports", {}).items()),
                           {k: v for k, v in n.get("params", {}).items() if isinstance(v, (int, str))}]
                          for n in cfg["nodes"]])
            groups.setdefault(key, []).append(s)
        seeds = max(groups.values(), key=len)
        b = BatchRapid(family, seeds)
    envs = [F.make(family, s) for s in seeds]
    rng = np.random.default_rng(1)
    for t in range(25):
        vals = b.lo + rng.random(b.lo.shape) * (b.hi - b.lo)
        out = b.step(vals)
        for i, e in enumerate(envs):
            r = e.step({"kind": "levers", "values": {n: float(v) for n, v in zip(b.lever_names[i], vals[i])}})
            eng = e._engine
            vis = [k for k, g in enumerate(getattr(eng, "gauges", [])) if not g.get("on_demand")] \
                if eng.kind == "primitive_graph" else list(range(out["readings"].shape[1]))
            got = list(r["observation"]["numbers"].values())
            want = [float(out["readings"][i, k]) for k in vis]
            want = [w for w in want if w == w]
            assert got == want, (family, t, i)
            res = r["observation"]["resources"]
            if "revenue_cents_last_tick" in res:                   # worlds with money
                assert res["revenue_cents_last_tick"] == float(out["revenue"][i])
