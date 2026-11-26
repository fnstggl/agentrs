"""Unit tests of the v5 research instrument (workstream N): partitions, sampler, statistics, censoring, safe loading,
claim scanning, curriculum guards and proposal safety. Fast; the measured gates are in tests/v5/test_gate_T.py,
test_gate_U.py and test_gate_SP_1[45]_*.py."""
from __future__ import annotations

import json

import numpy as np
import pytest

from crucible.research.v5 import curves as CV
from crucible.research.v5 import curriculum as CU
from crucible.research.v5 import distribution as D
from crucible.research.v5 import hstats as HS
from crucible.research.v5 import integrity as G
from crucible.research.v5 import learners as LR
from crucible.research.v5 import proposal as P
from crucible.research.v5 import sampler as S


@pytest.fixture(scope="module")
def reg():
    return D.build_registry()


def test_partitions_by_family_and_seed(reg):
    tr = reg.worlds(tier=1, family_partition="train", source="crucible.rapid")[0]["id"]
    ev = reg.worlds(tier=1, family_partition="evaluation", source="crucible.rapid")[0]["id"]
    assert reg.partition_of(tr, 5) == "train"
    assert reg.partition_of(tr, 1_000_005) == "validation"
    assert reg.partition_of(tr, 2_000_000_005) == "sealed"
    assert reg.partition_of(ev, 5) == "evaluation"
    with pytest.raises(D.PartitionError):
        reg.check_trainable(ev, 5)
    with pytest.raises(TypeError):
        reg.tables["partitions"][0]["name"] = "x"


def test_sampler_deterministic_and_resumable(reg):
    m = S.Mixture(tiers={1: 1.0}, name="t1")
    a, b = S.Sampler(reg, m, seed=3), S.Sampler(reg, m, seed=3)
    xs = [a.next()["item"] for _ in range(50)]
    assert xs == [b.next()["item"] for _ in range(50)]
    st = json.loads(json.dumps(a.state()))
    c = S.Sampler.resume(reg, st)
    assert c.next()["item"] == a.next()["item"]
    if not all(reg.tiers().values()):          # the default mixture never trains on fewer tiers silently
        with pytest.raises(S.TierUnavailable):
            S.Sampler(reg, S.DEFAULT_MIXTURE)
    else:
        assert S.Sampler(reg, S.DEFAULT_MIXTURE).targets == {1: 0.5, 2: 0.3, 3: 0.2}


def test_compute_deficit_scheduling(reg):
    tiers = [t for t, n in reg.tiers().items() if n]
    if len(tiers) < 2:
        tiers = [1]
    m = S.Mixture(tiers={t: 1.0 for t in tiers}, require_all_tiers=False, name="x")
    s = S.Sampler(reg, m, seed=1)
    for _ in range(200):
        sp = s.next()
        s.report(sp, cpu_s=0.01 * sp["tier"], interactions=5)
    sh = s.shares()["compute"]
    assert all(abs(v - 1 / len(sh)) < 0.05 for v in sh.values())


def test_crossed_interval_calibrated_naive_not():
    cr = HS.null_calibration("crossed", trials=300, n_seeds=3, n_worlds=12, seed=5)
    nv = HS.null_calibration("naive", trials=300, n_seeds=3, n_worlds=12, seed=5)
    assert cr["false_positive_rate"] <= 0.08 and nv["false_positive_rate"] > 0.3


def test_kaplan_meier_and_censored_acceleration():
    def run(nq, task, tau=64):
        return {"N_q": nq, "censored": nq is None, "tau": tau, "task": task, "learner_seed": 0}
    u = [run(None, "a"), run(None, "b"), run(32, "c")]
    e = [run(4, "a"), run(8, "b"), run(2, "c")]
    km = CV.kaplan_meier(u)
    assert km["median"] is None and km["censored"] == 2
    acc = CV.acceleration(u, e, n_boot=50)
    assert acc["median_ratio"]["kind"] == "lower_bound" and not acc["median_ratio"]["finite"]
    acc2 = CV.acceleration(e, u, n_boot=50)
    assert acc2["median_ratio"]["kind"] == "undefined"
    assert CV.n_q([{"budget": 0, "score": 0.1}, {"budget": 4, "score": 0.5}], 0.3)["N_q"] == 4
    assert CV.n_q([{"budget": 0, "score": 0.1}], 0.3)["censored"]


def test_params_hash_and_architecture_allow_list():
    w = LR.make("weights", 0)
    h = LR.params_hash(w.parameters())
    w.W2[0, 0] += 1.0
    assert LR.params_hash(w.parameters()) != h
    with pytest.raises(KeyError):
        LR.make("pickle_loader")


def test_claim_scanner_controls_and_clauses():
    for k in G.KINDS:
        assert G.control_check(k)["ok"], k
    t = "The viral cascades are empirically realistic. No realism is claimed for reach."
    assert len(G.scan_text(t, "realism_cascades")["claims"]) == 1


def test_curriculum_refuses_evaluation_results_and_keeps_random_share():
    cur = CU.Curriculum(seed=0)
    assert not cur.observe({"partition": "sealed", "category": "memory", "score": 1.0})
    assert not cur.observe({"partition": "evaluation", "category": "memory", "score": 1.0})
    assert cur.observe({"partition": "validation", "category": "memory", "score": 0.0})
    for _ in range(300):
        cur.next()
    assert cur.stats()["random_share"] >= 0.2
    with pytest.raises(ValueError):
        CU.Curriculum(random_share=0.1)
    rng = np.random.default_rng(0)
    for w in CU.TEMPLATES:
        assert CU.category_of(CU.targeted_structure(w, rng)) == w


def test_proposal_safety_refuses_partition_fields_and_free_text(reg):
    v = P.validate_proposal(reg, {"topology": "chain", "split": "val"})
    assert not v["accepted"] and not v["checks"]["safety"]
    v = P.validate_proposal(reg, {"topology": "chain", "motifs": ["saturation"], "pad": "x" * 9000})
    assert not v["accepted"]
    v = P.validate_proposal(reg, "not a structure")
    assert not v["accepted"]
