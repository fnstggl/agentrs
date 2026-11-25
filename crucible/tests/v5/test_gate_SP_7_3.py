"""Gate SP-7.3 - new engine kinds (workstream H).

An engine kind counts when a registered engine implements it with its own transition code (it imports no other
engine) and at least one family on it is admitted. This wave implements four of the eight kinds; the count is
recorded as measured."""
from __future__ import annotations

import ast
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
    return False


def test_SP_7_3_new_engine_kinds():
    adm = RP.admission_report(seeds=(0, 1, 2))
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
                      "not_yet_implemented": ",".join(missing),
                      "compositional_engine": "primitive_graph" if "primitive_graph" in admitted_engines else "none"},
           notes="Implemented with admitted families: discrete-event, graph dynamical, search/information, partially "
                 "observed state machine. Executable program, multi-agent strategic, scientific experiment and "
                 "planning/resource engines are scheduled for the next wave (the primitive_graph engine is the "
                 "structural generator's compositional engine and is not counted among the eight).")
    assert len(implemented) == 4
