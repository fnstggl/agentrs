"""Gate SP-7.3 - new engine kinds (workstream H).

An engine kind counts when a registered engine implements it with its own transition code (it imports no other
engine), a run shows transition behaviour of that kind (an event queue in continuous time, a graph adjacency
driving dynamics, categorical hidden-state transitions, exhaustible search opportunities, a program called inside
the dynamics, learning opponents, experiments on pilot copies, deliveries arriving after their lead time), and at
least one family on it is admitted on seeds 0-2. The count is recorded as measured."""
from __future__ import annotations

import ast
import json
import os

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import families as F
from crucible.rapid import report as RP
from crucible.rapid.engines import ENGINE_KINDS, SPEC_ENGINE_KINDS

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _transition_markers(kind: str) -> bool:
    """Evidence from a run that the transition code is of the claimed kind."""
    if kind == "discrete_event":
        env = F.make("des_bottleneck", 0)
        n = 0
        for _ in range(10):
            env.step(None)
            n += len(env._state["heap"]) > 0
        # events were taken from a priority queue in continuous time within ticks
        return n == 10 and isinstance(env._state["heap"][0][0], float) and env._state["clock"] == 10.0
    if kind == "graph_dynamical":
        env = F.make("net_diffusion", 0)
        A = np.asarray(env._config["adj"])
        return A.shape[0] == A.shape[1] == len(env._config["theta"]) and (A == A.T).all() and A.sum() > 0
    if kind == "state_machine":
        env = F.make("posm_sign_switch", 0)
        T = np.asarray(env._config["T"])
        seen = set()
        for _ in range(80):
            env.step(None)
            seen.add(int(env._state["s"][0]))
        return np.allclose(T.sum(axis=1), 1.0) and len(seen) >= 2
    if kind == "search":
        env = F.make("srch_field", 0)
        before = float(env._state["left"].sum())
        r = env.step({"kind": "query", "target": "probe", "args": {"cell": "c0_0"}})
        env.step({"kind": "discrete", "decision": "dig", "choice": "c0_0"})
        return r["result"]["ok"] and float(env._state["left"].sum()) <= before
    if kind == "program_exec":
        env = F.make("pe_unstable_control", 0)
        src = ("def act(observation, caller):\n    m = caller.get('memory') or {'n': 0}\n    m['n'] += 1\n"
               "    return {'u': 0.0, 'memory': m}\n")
        r = env.step({"kind": "program", "source": src, "ticks": 4})
        # the world called the program at every inner step of the tick (more calls than ticks), in the sandbox
        return r["result"]["ok"] and env._state["pstat"]["calls"] >= 5 and \
            (env._state["memory"] or {}).get("n") == env._state["pstat"]["calls"]
    if kind == "strategic":
        env = F.make("strat_price_war", 0)
        name = next(iter(env.action_space().kinds["levers"]["properties"]["values"]["properties"]))
        before = json.dumps(env._state["V"], default=str)
        for _ in range(8):
            env.step({"kind": "levers", "values": {name: 6.0}})
        # rivals' learned values (propensities) changed with experience of the participant's prices
        return json.dumps(env._state["V"], default=str) != before
    if kind == "experiment_lab":
        env = F.make("lab_kinetics", 0)
        sp = env.action_space()
        dz = sp.kinds["experiment"]["properties"]["design"]["properties"]
        d = {"set": {n: float(p["minimum"]) for n, p in dz["set"]["properties"].items()},
             "measure": list(dz["measure"]["items"]["enum"])[:1], "replicates": 2}
        if "instrument" in dz:
            d["instrument"] = dz["instrument"]["enum"][0]
        if "time" in dz:
            d["time"] = int(dz["time"]["minimum"]) + 1
        live, b0 = dict(env._state["live"]), float(env._state["budget"])
        r = env.step({"kind": "experiment", "design": d})
        return r["result"]["ok"] and env._state["live"] == live and float(env._state["budget"]) < b0
    if kind == "planning":
        env = F.make("plan_distribution", 0)
        sites = sorted(env.action_space().kinds["allocate"]["properties"]["amounts"]["properties"])
        before = json.dumps(env._state["transit"], default=str)
        env.step({"kind": "allocate", "amounts": {sites[0]: 10.0}})
        moved = json.dumps(env._state["transit"], default=str) != before
        return moved and int(env._config.get("lead", 1) if "lead" in env._config else 1) >= 1
    return False


def test_SP_7_3_new_engine_kinds():
    adm = RP.admission_report(seeds=(0, 1, 2), jobs=int(os.environ.get("CRUCIBLE_RAPID_JOBS", "1")))
    admitted_engines = {r["engine"] for r in adm["reports"].values() if r["admitted"]}
    edir = os.path.join(ROOT, "src", "crucible", "rapid", "engines")
    implemented, missing = [], []
    for spec_kind, eng in SPEC_ENGINE_KINDS.items():
        if eng is None:
            missing.append(spec_kind)
            continue
        src = ast.parse(open(os.path.join(edir, f"{eng}.py")).read())
        imports = {("." * n.level) + (n.module or "") for n in ast.walk(src) if isinstance(n, ast.ImportFrom)}
        own = not (imports & {f".{o}" for o in ENGINE_KINDS if o != eng})
        if eng in ENGINE_KINDS and eng in admitted_engines and own and _transition_markers(eng):
            implemented.append(spec_kind)
        else:
            missing.append(spec_kind)
    record("SP-7.3", {"new_engine_kinds": len(implemented), "implemented": ",".join(implemented),
                      "not_counted": ",".join(missing),
                      "compositional_engine": "primitive_graph" if "primitive_graph" in admitted_engines else "none"},
           notes="A kind counts with its own transition code, a run showing behaviour of that kind and an admitted "
                 "family (seeds 0-2). The primitive_graph engine is the structural generator's compositional engine "
                 f"and is not counted among the eight. Not counted: {missing}.")
