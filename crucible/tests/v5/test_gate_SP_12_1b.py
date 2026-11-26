"""SP-12.1b: the self-play population mixes a shared training system and independently updated participants, with
diverse learning algorithms, unrelated objectives and adaptation during the civilisation's lifetime.

A short campaign (`_selfplay_kit.campaign`): alpha's three identities feed one shared REINFORCE policy (a shared
training system), beta's two one PPO policy; gamma (A2C, drive reach) and delta (A2C, drive adoption) are updated
independently; the semantic generalist pursues a text goal. Algorithms are counted among groups whose parameters
changed in the civilisation; unrelated objectives are learner identities whose own drive is not profit (and the
semantic participant's goal, reported separately); lifetime adaptation = updates in both halves of the run."""
import json

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.multiagent import campaign as C

from _selfplay_kit import campaign

pytestmark = [pytest.mark.slow, pytest.mark.timeout(5400)]
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-12.1b"
LEARNING = ("alpha", "beta", "gamma", "delta")


def test_SP_12_1b_heterogeneous_learning_population(tmp_path):
    run = campaign(str(tmp_path), "short")
    m = run["match"]
    ev = C.learning_evidence(run)
    updated = [g for g in LEARNING if ev[g]["updates_in_run"] >= 1]
    algos = sorted({m.groups[g].algo for g in updated})
    drives = {p: mm.drive for p, mm in m.members.items() if mm.role in ("current", "independent")}
    unrelated = sorted(p for p, d in drives.items() if d != "profit")
    shared = {g: len(m.groups[g].identities) for g in LEARNING}
    half = m.t0 + (m.world.now - m.t0) // 2
    t_of = {}
    for u in run["registry"].updates:
        if u["group"] in LEARNING and u["changed"] and u.get("algo") != "behaviour_cloning":
            t_of.setdefault(u["group"], []).append(u)
    # update times: the decisions an update trained on (provenance of the versions it produced)
    halves = {g: set() for g in LEARNING}
    for v in run["registry"].versions.values():
        if v["group"] in LEARNING and v["origin"] == "update":
            ts = [x["t_last"] for x in v["provenance"]["by_identity"].values()]
            halves[v["group"]].add("first" if max(ts) < half else "second")
    lifetime = sorted(g for g in LEARNING if halves[g] == {"first", "second"})
    record("SP-12.1b", {"distinct_learning_algorithms": len(algos), "algorithms": json.dumps(algos),
                        "participants_with_unrelated_objectives": len(unrelated),
                        "unrelated_objectives": json.dumps({p: drives[p] for p in unrelated}),
                        "independently_updated_participants": len(updated),
                        "shared_training_identities": json.dumps(shared),
                        "groups_adapting_through_lifetime": len(lifetime),
                        "semantic_goals": json.dumps(sorted({mm.goal for mm in m.members.values()
                                                             if mm.role == "semantic"}))},
           command=CMD, labels={"learned": True},
           notes="independently updated participants = policy groups updated in the civilisation, each its own "
                 "parameters/optimiser/generators (alpha and beta are shared training systems of several identities)")
    m.close()
    assert max(shared.values()) >= 2
    assert len(algos) >= 3 and len(unrelated) >= 2 and len(updated) >= 2
