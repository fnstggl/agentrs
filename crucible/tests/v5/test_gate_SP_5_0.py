"""SP-5.0 (MASTER_SPEC Parts 3.2, 4.4, 5; owner workstream E): fitted behavioural models drive Class 1 participant
decisions in the running integrated world, every runtime use the fidelity register claims is exercised, and decision
events carry the model id and version (decision provenance).

Measured on a generated civilisation (`crucible.civ.generate("balanced", 3, population="development")`) run for two
simulated days (`crucible.calibration.inworld.provenance_metrics`): the fitted models named by the decision
provenance (`extra.decision`, a fitted model id or a runtime decision id that serves one) of batches that are organic
population decisions or participants' own decisions under the population's provenance rules (re-derived batch by
batch and cross-checked against `population.check_provenance`) and whose rows the runtime really evaluated in this
run (`runtime.usage()`, reset before the world is built); the register's in-world claims that fail verification or
that this world does not exercise; and every population-table batch outside the declared non-decision kinds
(`inworld.NON_DECISION_KINDS`: genesis, scheduling, attention and money bookkeeping, deliveries) recorded without
provenance. The population workstream wires the runtime models (D-15); until it does, and
until civ.generate builds again (the integration fix of the duplicate 'space' module), this test FAILS - it never
skips."""
import pytest

from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.acceptance.v5.evidence import record
from crucible.calibration import inworld, runtime

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-5.0"
DAYS = 2


def _criteria(req):
    return next(e for e in MF.load()["requirements"] if e["id"] == req)["criteria"]


@pytest.mark.slow
@pytest.mark.timeout(5400)
def test_SP_5_0_fitted_models_drive_decisions_in_the_integrated_world():
    import torch
    torch.set_num_threads(1)
    import crucible
    crucible.load_all_modules()
    from crucible import civ
    runtime.reset_usage()
    w = civ.generate("balanced", 3, population="development")
    try:
        w.advance_time(DAYS * 86400)
        m = inworld.provenance_metrics(w)
    finally:
        w.close()
    metrics = {k: v for k, v in m.items() if not k.startswith("_")}
    record("SP-5.0", metrics, command=CMD,
           notes=f"fitted models used: {m['_used']}; population decision batches without provenance by kind: "
                 f"{m['_without_by_kind']}; provenance classes: {m['_provenance_classes']}; forced batches with "
                 f"fitted provenance: {m['_forced_with_fitted_provenance']}; unverified claims: "
                 f"{m['_unverified_claims']}")
    # the measurement saw the world's decisions (not an empty event log)
    assert m["decision_events_with_model_provenance"] + m["decision_events_without_model_provenance"] > 0
    fails = [msg for c in _criteria("SP-5.0") for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, (fails, m["_used"], m["_without_by_kind"])
