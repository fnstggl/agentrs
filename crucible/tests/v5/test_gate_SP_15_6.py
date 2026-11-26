"""SP-15.6 - Positive-control transfer experiment on newly added nontrivial families detected by the evaluator
(MASTER_SPEC Part 15.6; workstream N).

The genuine-transfer control (`learners.TransferLearner`: it learns *how to search* across tasks) is trained on
v5 rapid training families and then practises held-out v5 families it never saw. Its capability on those targets
was established independently first (`instrument.establish_transfer`, on seeds no protocol uses). The instrument
(`instrument.controls_result`: five learner seeds, crossed seed x task interval, prespecified effect) must label it
`transferable_efficiency`, and must *not* give that label to the known-negative control trained identically on
uninformative signals (`transfer_sham`).
"""
from __future__ import annotations

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import distribution as D
from crucible.research.v5 import instrument as I

pytestmark = [pytest.mark.slow, pytest.mark.timeout(14400)]


def test_SP15_6_positive_control_transfer():
    res = I.controls_result()
    d = res["design"]
    reg = D.build_registry()
    est = I.establish_transfer(reg, d["train_families"], [f for f in I.establish(reg) if f not in d["train_families"]])
    established = {f: est[f]["transfer_gain"] for f in d["transfer_families"]}
    tr = res["controls"]["transfer"]
    sham = res["controls"]["transfer_sham"]
    st = tr["stats"]["transferable_efficiency"]
    detected = "transferable_efficiency" in tr["labels"] and "transferable_efficiency" not in sham["labels"] and \
        all(v >= I.ESTABLISH_MIN for v in established.values())
    record("SP-15.6", {"positive_control_transfer_detected": bool(detected), "transfer_effect": float(st["estimate"]),
                       "transfer_ci_low": float(st["lo"]), "sham_transfer_effect":
                       float(sham["stats"]["transferable_efficiency"]["estimate"]), "learner_seeds": len(d["seeds"]),
                       "target_families": len(d["transfer_families"])},
           notes=f"Targets (new v5 families, held out of training) and their independently established gains: "
                 f"{established}; acceleration {st.get('acceleration', {}).get('median_ratio')}; labels transfer "
                 f"{tr['labels']}, sham {sham['labels']}")
    assert detected
