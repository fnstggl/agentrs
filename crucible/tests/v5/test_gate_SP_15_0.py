"""SP-15.0 - Evaluation distinguishes memorisation, pretrained competence, context adaptation, persistent state,
neural weight learning, meta-learning and transferable efficiency (MASTER_SPEC Part 15; workstream N).

Each explanation has a control built to have it (`instrument.CONTROLS`: memoriser, pretrained, adapter, stateful,
weights, meta, transfer) on families where that capability was established independently. An explanation counts
as distinguished when the instrument (`instrument.controls_result`, five learner seeds) labels its control with it
and labels no control that is built without it (the hold control and the controls whose known capability excludes
it).
"""
from __future__ import annotations

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import instrument as I

pytestmark = [pytest.mark.slow, pytest.mark.timeout(14400)]
OWNER = {"memorisation": "memoriser", "pretrained_competence": "pretrained", "context_adaptation": "adapter",
         "persistent_state": "stateful", "weight_learning": "weights", "meta_learning": "meta",
         "transferable_efficiency": "transfer"}


def test_SP15_0_seven_explanations_distinguished():
    res = I.controls_result()
    c = res["controls"]
    dist = {}
    for exp, own in OWNER.items():
        found = exp in c[own]["labels"]
        wrong = [a for a, (must, must_not) in I.CONTROLS.items() if exp in must_not and exp in c[a]["labels"]]
        dist[exp] = {"found_on_its_control": found, "found_where_excluded": wrong}
    n = sum(1 for v in dist.values() if v["found_on_its_control"] and not v["found_where_excluded"])
    dev = {"seeds": {0, 1}, "eval_seeds": set(range(1_000_101, 1_000_111)), "world_seed0": {300}}
    d = res["design"]
    overlap = len(set(d["seeds"]) & dev["seeds"]) + len(set(d["eval_seeds"]) & dev["eval_seeds"]) + \
        int(d["world_seed0"] in dev["world_seed0"])
    record("SP-15.0", {"explanations_distinguished": n, "explanations": len(I.EXPLANATIONS),
                       "controls": len(c), "learner_seeds": len(res["design"]["seeds"]),
                       "seeds_shared_with_development_batteries": overlap},
           notes=f"Per explanation: {dist}; labels {({a: c[a]['labels'] for a in c})}")
    assert n == 7, dist
    assert overlap == 0, d             # confirmation, not a rerun of the development batteries (D-69 addendum 2)
