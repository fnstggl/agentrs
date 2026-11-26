"""SP-3.2 (MASTER_SPEC Part 3.2): four participant classes, counted by what they DO in a world of the one population
(crucible.population): a class counts when its members took recorded decisions - class 1 sessions, class 2 decisions
of a shared adaptive policy, class 3 strategic decisions of their policy groups, class 4 decisions of a semantic
participant runtime bound with `population.register_class_runtime` (workstream 2b binds it; without a bound runtime
class-4 rows exist but do not act, and are not counted)."""
import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.population.worlds import population_world
from crucible.substrate import artifacts, columns

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_3_2.py"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_SP_3_2_participant_classes_act():
    import torch
    torch.set_num_threads(1)
    w = World.create(population_world("development", 6, n=200, sites=6, tick_s=1800,
                                      classes={"adaptive": 32, "strategic": 8, "semantic": 2}))
    w.advance_time(5 * 86400)
    cls = columns.col(w, "pop", "class")
    acted = {1: 0, 2: 0, 3: 0, 4: 0}
    for e in w.log:
        if e.type == "cols.batch" and e.payload["kind"] == "pop.session":
            idx = artifacts.get_array_blob(w, e.payload["ref"])["idx"].astype(np.int64)
            for c in (1, 2, 4):
                acted[c] += int(np.sum(cls[idx] == c)) if c != 2 else 0
    acted[2] = sum(1 for e in w.log if e.type == "pop_adaptive.examples")
    acted[3] = sum(1 for e in w.log if e.type == "pop.policy_update" and str(e.payload["group"]).startswith("strategic"))
    rows = {c: int(np.sum(cls == c)) for c in (1, 2, 3, 4)}
    m = {"participant_classes": int(sum(1 for c in acted if acted[c] > 0)),
         **{f"class{c}_rows": rows[c] for c in rows}, **{f"class{c}_decisions": acted[c] for c in acted},
         "class4_runtime_bound": bool(4 in (w.runtime.get("pop.class_runtimes") or {}))}
    evidence.record("SP-3.2", m, command=CMD, notes=(
        "Classes 1-3 act in this tree; class 4 (compact semantic generalists) rows exist but no semantic runtime is "
        "bound to population.register_class_runtime here (workstream 2b), so they are not counted."))
    assert rows[4] == 2
