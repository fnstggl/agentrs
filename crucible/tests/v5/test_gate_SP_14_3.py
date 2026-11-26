"""SP-14.3 - Adaptive curriculum over the ten named weakness categories with learning-progress, regret, uncertainty
and success measures and collapse prevention; the curriculum never accesses sealed evaluation results
(MASTER_SPEC Part 14.3; workstream N).

A curriculum (`crucible.research.v5.curriculum`) measures a learner on worlds it generates for every category -
results served by the independent evaluator process through its ``curriculum`` view (train and validation
partitions only) - and profiles all ten categories. Sealed access is measured three ways while it runs: the
evaluator counts sealed or evaluation results served to the curriculum view, the curriculum counts refused results
(we offer it evaluation and sealed results directly), and an audit hook counts opens of the sealed salt file by
this process.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import curriculum as CU
from crucible.research.v5 import curriculum_lab as CL
from crucible.research.v5 import distribution as D
from crucible.research.v5 import evaluator as EV

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
_OPENS = []


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], str) and "v5_sealed_salt" in args[0]:
        _OPENS.append(args[0])


def test_SP14_3_curriculum_categories_measures_and_sealed_isolation(tmp_path):
    sys.addaudithook(_hook)
    sealed = str(tmp_path / "sealed")
    ref = CL.reference_profile(n_per=3)
    cur = CU.Curriculum(seed=3, reference=ref)
    learner = CL.PlantedAdapter(0, arm=CL.PLANTED["lag_blind"])
    ev = EV.Evaluator.start(sealed_dir=sealed)
    served = 0
    try:
        reg = D.build_registry()
        evals = [w["id"] for w in reg.worlds(tier=1, family_partition="evaluation", source="crucible.rapid")][:1]
        ev.must("declare_sealed", "final", evals, 1, 1)
        rng = np.random.default_rng(3)
        n = 0
        while n < 90:
            it = cur.next()
            if "structure" not in it:
                continue
            e = cur.entry_for(it)
            sd = 1_000_000 + int(it["seed"]) % 900_000
            try:
                sc = CL._score(e, sd, learner, rng)
            except Exception:  # noqa: BLE001 - an unbuildable draw gives no measurement
                continue
            cur.observe({"partition": "validation", "structure": it["structure"], "category": it["category"],
                         "score": sc, "seen": False})
            n += 1
        # results the curriculum is offered but must refuse
        for part in ("evaluation", "sealed"):
            cur.observe({"partition": part, "category": "memory", "score": 0.9})
        t = ev.open_episode(reg.worlds(tier=1, family_partition="train", source="crucible.rapid")[0]["id"], 1_000_031)
        EV.run_remote_episode(ev, t, learner, rng)
        r = ev.result(t, view="curriculum")
        cur.observe({"partition": r["partition"], "category": "generalization", "score": r["score"], "seen": False})
        served = ev.must("sealed_status")["curriculum_sealed_served"]
    finally:
        ev.close()
    prof = cur.prof.profile()
    measured = [c for c in CU.WEAKNESSES if all(prof[c].get(m) is not None for m in CU.MEASURES)]
    measures = [m for m in CU.MEASURES if all(prof[c].get(m) is not None for c in measured)]
    accesses = served + len(_OPENS)
    st = cur.stats()
    record("SP-14.3", {"weakness_categories": len(measured), "curriculum_measures": len(measures),
                       "sealed_result_accesses": int(accesses), "refused_evaluation_results":
                       st["refused_results"]["evaluation"], "refused_sealed_results": st["refused_results"]["sealed"],
                       "random_share": float(st["random_share"]), "normalised_entropy": float(st["normalised_entropy"])},
           notes=f"Profile {({c: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in prof[c].items()} for c in CU.WEAKNESSES})}; "
                 f"weights {cur.weights()}")
    assert len(measured) >= 10 and len(measures) >= 4, prof
    assert accesses == 0 and st["refused_results"]["evaluation"] == 1 and st["refused_results"]["sealed"] == 1
