"""SP-5.4: a fidelity record for every important behavioural mechanism, with the population, platform and period of
its evidence and its limits (owner workstream E).

The important mechanisms are computed at test time (`crucible.calibration.fidelity.important_mechanisms`): every
mechanism a registered fitted model grounds, every behavioural mechanism of the Crucible 4.0 register, and every
kernel module a v5 behaviour package registers. Records come from `docs/v5/FIDELITY_REGISTER.md` and the 4.0
register."""
import os

from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.acceptance.v5.evidence import record
from crucible.calibration import fidelity as FI


def test_SP_5_4_fidelity_record_for_every_important_mechanism():
    m = FI.sp54_metrics()
    metrics = {k: v for k, v in m.items() if not k.startswith("_")}
    record("SP-5.4", metrics, command="PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-5.4",
           artifacts=[FI.REGISTER_MD, os.path.join(os.path.dirname(FI.REGISTER_MD), "..", "..", "src", "crucible",
                                                   "evaluation", "fidelity.py")],
           notes=f"missing: {m['_missing']}; incomplete: {m['_incomplete']}")
    # every v5 row is itself well formed and the empirically fitted rows cite registered datasets
    assert not m["_row_problems"], m["_row_problems"]
    assert m["empirically_fitted_rows"] >= 1
    crit = next(e for e in MF.load()["requirements"] if e["id"] == "SP-5.4")["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, (fails, m["_missing"], m["_incomplete"])


def test_SP_5_4_register_rejects_incomplete_records():
    good = {"mechanism": "x", "module": "pop", "category": "empirically fitted",
            "evidence": "kuairand_pure: Kuaishou featured feed, 27,285 users, 2022-04-22..05-08",
            "heldout": "log loss 0.44 vs 0.48", "limits": "one app, one month"}
    assert FI.record_problems(good) == []
    assert FI.record_problems(dict(good, category="calibrated"))            # not a v5 category
    assert FI.record_problems(dict(good, evidence="our own survey 2022"))    # no registered dataset
    assert FI.record_problems(dict(good, evidence="kuairand_pure"))          # no period
    assert FI.record_problems(dict(good, heldout=""))                        # empirical without held-out result
    assert FI.record_problems(dict(good, limits=""))
    assert FI.record_problems({"mechanism": "y", "module": "pop", "category": "plausible, unvalidated",
                               "evidence": "none", "heldout": "", "limits": "assumed"}) == []
