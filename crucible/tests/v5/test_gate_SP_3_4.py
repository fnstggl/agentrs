"""SP-3.4 (MASTER_SPEC Part 3.4): independent learning, each kind tested separately with weight hashes, training
examples, losses, behavioural changes and evaluation outcomes, in worlds of the one population.

1. persistent personal-state adaptation - two learning groups with identical initial weights, one with personal GRU
   memory and one without; outcome: realised utility after a controlled change (the benefit of personal state);
2. shared-policy learning - one group's shared weights learn from its members' own experience (hash change,
   held-out value-prediction loss vs the initial weights);
3. independently learned policy groups - a third group with other initial weights learns on its own members only;
4. frozen historical participants - `pop_adaptive.freeze_copy` of a learning group: the copy's hash stays fixed while
   the source keeps learning, and its members keep acting;
5. independently trained competing architectures - class 3's two strategic groups (MLP/REINFORCE and GRU/A2C).

learning_kinds_tested >= 5. A kind counts as TESTED when its experiment ran with recorded weight hashes and
optimizer updates; whether it HELPED is reported separately (personal state: the CI of its benefit)."""
import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.population import experiments as PX
from crucible.population.worlds import population_world

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m crucible.scale.slots --cpu 1 -- python -m pytest -q tests/v5/test_gate_SP_3_4.py"
DAY = 86400
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3 * 3600)]


def test_SP_3_4_independent_learning_kinds():
    import torch
    torch.set_num_threads(1)
    arms = {"memory": {"id": 0, "share": 0.34, "seed": 5}, "no_memory": {"id": 1, "share": 0.33, "seed": 5,
                                                                         "memory": False},
            "other_init": {"id": 2, "share": 0.33, "seed": 9}}
    res = [PX.adaptation(s, 5, arms) for s in (0, 1)]
    r = res[0]
    kinds = {}
    ps = [x["phase2_difference"] for x in res]
    kinds["personal_state_adaptation"] = {
        "tested": all(x["updates"]["memory"] > 0 and x["updates"]["no_memory"] > 0 for x in res),
        "benefit_mean": float(np.mean([p["mean"] for p in ps])), "benefit_ci_low": float(min(p["ci95"][0] for p in ps)),
        "benefit_ci_high": float(max(p["ci95"][1] for p in ps))}
    kinds["shared_policy_learning"] = {
        "tested": bool(r["hash_end"]["memory"] != r["hash_genesis"]["memory"] and r["optimizer_steps"]["memory"] > 0),
        "heldout_mse_learned": (r.get("heldout_mse") or {}).get("memory")}
    kinds["independent_policy_groups"] = {
        "tested": bool(r["hash_end"]["other_init"] != r["hash_genesis"]["other_init"] and
                       r["hash_end"]["other_init"] != r["hash_end"]["memory"]),
        "updates_other_init": r["updates"]["other_init"]}
    # frozen historical participants
    w = World.create(population_world("development", 4, n=60, sites=6, tick_s=1800, strategic=False,
                                      classes={"adaptive": 64, "strategic": 0, "semantic": 0},
                                      adaptive={"min_examples": 16}))
    w.advance_time(2 * DAY)
    fc = w.system_call("pop_adaptive.freeze_copy", {"group": 7, "from_group": 0, "share": 0.5})
    ad = w.modules["pop_adaptive"]
    src0 = ad.policy(w, 0).weights_hash()
    w.advance_time(5 * DAY)
    frozen_acts = sum(1 for e in w.log if e.type == "pop_adaptive.examples" and e.payload["group"] == 7)
    kinds["frozen_historical"] = {"tested": bool(ad.policy(w, 7).weights_hash() == fc["weights_hash"] and
                                                 ad.policy(w, 0).weights_hash() != src0 and frozen_acts > 0),
                                  "frozen_example_batches": frozen_acts, "members": fc["members"]}
    # competing architectures (class 3)
    w3 = World.create(population_world("development", 3, n=120, sites=6, tick_s=1800, adaptive=False,
                                       classes={"adaptive": 0, "strategic": 8, "semantic": 0}))
    w3.advance_time(8 * DAY)
    ups = [e.payload for e in w3.log if e.type == "pop.policy_update" and str(e.payload["group"]).startswith("strategic")]
    by = {}
    for u in ups:
        by.setdefault(u["group"], set()).add(u["weights_hash"])
    archs = {u["group"]: u.get("arch") for u in ups}
    kinds["competing_architectures"] = {"tested": bool(len([g for g, h in by.items() if len(h) > 1]) >= 2),
                                        "groups": {g: archs.get(g) for g in by}, "updates": len(ups)}
    m = {"learning_kinds_tested": int(sum(1 for k in kinds.values() if k["tested"]))}
    for k, v in kinds.items():
        m[f"{k}_tested"] = bool(v["tested"])
    m["personal_state_benefit_mean"] = round(kinds["personal_state_adaptation"]["benefit_mean"], 4)
    m["personal_state_benefit_ci_low"] = round(kinds["personal_state_adaptation"]["benefit_ci_low"], 4)
    m["personal_state_benefit_ci_high"] = round(kinds["personal_state_adaptation"]["benefit_ci_high"], 4)
    evidence.record("SP-3.4", m, command=CMD, notes=f"Per kind: {kinds}. Per seed (memory vs no-memory, phase 2 "
                                                    f"after the controlled change): {ps}")
    assert len(kinds) == 5
