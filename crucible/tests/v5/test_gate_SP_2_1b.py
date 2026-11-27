"""SP-2.1b (MASTER_SPEC Part 2.1, Mode C): a technological invention can change economic outcomes, participant
behaviour, resource demand and future technological opportunities, each confirmed by counterfactual removal in an
integrated world. Workstream J.

The invention (crucible.civ.experiments `invention`, profile balanced, 6 seeds, fork after 2 days, 8 days after
it): a scripted reference inventor (labelled scripted; crucible.innovation.reference.ServiceCreator) writes, deploys
and declares a service for one of the civilisation's published work formats at the best known solver tier. The
counterfactual arm admits the same inventor, who never builds it. Everything downstream is the world's: individuals
of the population find the declaration through search, the forum and their contacts, try it on their own work,
adopt it by their own accepted outcomes and pay per call; its compute loads its provider's datacenter, which draws
from the regional grid; the evaluator-side relation graph (crucible.innovation.relations) labels the needs the new
situation creates.

    invention_effect_kinds_confirmed   of the four kinds (crucible.civ.experiments.INVENTION_KINDS), those with a
                                       measure whose effect is sign-consistent over the seeds, whose paired 95% CI
                                       excludes zero and whose mean exceeds twice the placebo reshuffling noise
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "CRUCIBLE_CIV_WORKERS=3 PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_2_1b.py"


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_SP_2_1b_invention_changes_four_kinds_of_outcome():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates
    r = gates.invention()
    m = {"invention_effect_kinds_confirmed": r["invention_effect_kinds_confirmed"],
         "invariant_errors": r["invariant_errors"] if r["invariant_errors"] is not None else -1,
         "failed_process_runs_in_arms": r["proc_failed"] if r.get("proc_failed") is not None else -1}
    evidence.record("SP-2.1b", m, command=CMD, notes=f"scripted reference inventor (labelled); seeds {r['seeds']}; "
                    f"error {r['error']}; kinds {r['kinds']}")
    assert r["error"] is None, r["error"]
    assert m["invariant_errors"] == 0
    assert m["invention_effect_kinds_confirmed"] == 4, r["kinds"]
