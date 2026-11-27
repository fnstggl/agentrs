"""SP-3.2 (MASTER_SPEC Part 3.2): four participant classes, counted by what they DO in a world of the one population
(crucible.population): a class counts when its members took recorded decisions - class 1 sessions (the population's
learning batch of the session's takers, `pop.learn`), class 2 decisions of a shared adaptive policy (example
batches of its activity and resource decisions), class 3 strategic decisions of their policy groups (ventures built
with ordinary tools), class 4 actions of its members through ordinary tools (events recorded with a class-4 member's
own principal as actor) during activations of the semantic participant runtime bound with
`population.register_class_runtime` (workstream D's `pop_semantic`, composed into the world when the module is
registered in this checkout; without it class-4 rows exist but do not act, and are not counted)."""
import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.core.kernel import MODULES, World
from crucible.population.worlds import population_world
from crucible.substrate import artifacts, columns

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-3.2"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_SP_3_2_participant_classes_act():
    import torch
    torch.set_num_threads(1)
    spec = population_world("development", 6, n=200, sites=6, tick_s=1800,
                            classes={"adaptive": 32, "strategic": 8, "semantic": 2})
    if "pop_semantic" in MODULES:
        spec.modules.append({"name": "pop_semantic", "config": {}})
    w = World.create(spec, strict=True)
    try:
        w.advance_time(5 * 86400)
        cls = columns.col(w, "pop", "class")
        acted = {1: 0, 2: 0, 3: 0, 4: 0}
        activations4 = 0
        for e in w.log:
            if e.type != "cols.batch" or e.payload["table"] != "pop":
                continue
            k = e.payload["kind"]
            if k == "pop.learn":
                idx = artifacts.get_array_blob(w, e.payload["ref"])["idx"].astype(np.int64)
                acted[1] += int(np.sum(cls[idx] == 1))
            elif k == "pop.session" and (e.payload.get("extra") or {}).get("class") == 4:
                activations4 += int(e.payload["n"])
        # class 4 acts when its members' own principals did something in the world through ordinary tools (events
        # recorded with a class-4 member as actor), not when the runtime was merely activated
        c4 = {f"p{int(i)}" for i in np.flatnonzero(cls == 4)}
        acts4 = [e for e in w.log if e.actor in c4]
        acted[4] = len(acts4) if activations4 else 0
        acted[2] = sum(int(e.payload["n"]) for e in w.log if e.type == "pop_adaptive.examples")
        acted[3] = sum(1 for e in w.log if e.type == "pop_strategic.exp")
        rows = {c: int(np.sum(cls == c)) for c in (1, 2, 3, 4)}
        m = {"participant_classes": int(sum(1 for c in acted if acted[c] > 0)),
             **{f"class{c}_rows": rows[c] for c in rows}, **{f"class{c}_decisions": acted[c] for c in acted},
             "class4_runtime_bound": bool(4 in (w.runtime.get("pop.class_runtimes") or {})),
             "class4_activations": int(activations4),
             "class4_action_event_types": ",".join(sorted({e.type for e in acts4}))[:400]}
        evidence.record("SP-3.2", m, command=CMD, notes=(
            "Class 1: individuals' sessions (pop.learn rows of class-1 individuals); class 2: decisions of the shared "
            "adaptive policies (activity and resource choices with realised utilities); class 3: strategic decisions "
            "(pop_strategic.exp); class 4: events its members' own principals caused through ordinary tools during "
            "activations of the bound semantic participant runtime. Class 4 runtime registered in this checkout: "
            f"{'pop_semantic' in MODULES}."))
    finally:
        w.close()
    assert rows[4] == 2
    crit = next(e for e in MF.load()["requirements"] if e["id"] == "SP-3.2")["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, m)] if not ok]
    assert not fails, (fails, m)
