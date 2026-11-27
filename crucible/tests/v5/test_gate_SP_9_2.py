"""SP-9.2 - Tier 2 domain Software ecosystem (MASTER_SPEC Part 9.2; workstream I-1).

Integrated runs through the learner interface (`base.section_metrics` / `base.run` -> `crucible.experience.make(2,
"software:<config>", seed)`, default reward mode) of `crucible.domains.software`: the reference maintainer (LABELLED
SCRIPTED, tool-only) on seeds 0 and 1 of the first training configuration over the 21-day horizon, while the
population's developers (module `devpop`) build their own applications, search the index, judge packages by their
own preferences, depend, run their own tests, report issues, hand-code, publish their own packages, upgrade and
execute. Mechanics are counted by the framework's recorded-event probes; adoption is the ablation-measured
adoption by execution of `innovation.pkg`. Controls: the null baseline, and the reference's playbook with a stub
package (documented names, code that does nothing): independent testing must keep developers from adopting it.
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import checker, evidence  # noqa: E402
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains import software as D  # noqa: E402

CMD = "PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/v5/test_gate_SP_9_2.py"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(5 * 3600)]


def _adopters(run) -> int:
    return int((run["outcome"].get("details") or {}).get("independent_adopters", 0))


def test_SP_9_2_software_ecosystem_mechanics():
    d = base.part9_domains()["software"]
    assert d is D.DOMAIN
    m, runs = base.section_metrics("software")
    cfg = d.reference_configs[0] if d.reference_configs else d.split("train")[0].name
    null = base.cached_run("software", cfg, 0, "null", None)
    stub = base.cached_run("software", cfg, 0, "stub_package", None)
    m.update({"mechanics_exercised_without_seat": null["mechanics"]["exercised"],
              "autonomous_actors_active": null["actors"]["active"],
              "population_activity_decisions_without_seat": null["population"]["activity_decisions"],
              "reference_independent_adopters_min": min(_adopters(r) for r in runs),
              "stub_package_independent_adopters": _adopters(stub),
              "stub_package_issue_events": stub["mechanics"]["counts"]["Issues"],
              "null_success": bool(null["success"]),
              "invariant_violations": sum(r["invariant_violations"] for r in runs + [null, stub]),
              "reference_script_world_reads": len(base.reads_world(D.reference_script)),
              "process_failures_all_runs": sum(r["proc_failed"] for r in runs + [null, stub])})
    evidence.record("SP-9.2", m, command=CMD, labels=dict(K.SCRIPTED),
                    notes=f"software:{cfg} seeds {[r['seed'] for r in runs]}, {runs[0]['days']} days; LABELLED "
                          f"SCRIPTED reference maintainer (tool-only). Mechanic counts: "
                          f"{[r['mechanics']['counts'] for r in runs]}; null run: {null['mechanics']['counts']}, actors "
                          f"{null['actors']['counts']}; outcomes {[r['outcome'].get('details') for r in runs]}; stub "
                          f"package: {stub['outcome'].get('details')}")
    assert m["required_mechanics_missing"] == 0, [r["mechanics"]["missing"] for r in runs]
    # domains-framework/3: every evidence function returned recorded events, no two mechanics share evidence
    assert m["unverified_evidence"] == "none" and m["indistinct_mechanics"] == "none", m
    assert m["autonomous_actors_active"] == len(d.actors) and m["population_activity_decisions_without_seat"] > 0
    assert m["process_failures_all_runs"] == 0 and m["invariant_violations"] == 0
    assert m["reference_independent_adopters_min"] > 0 and m["stub_package_independent_adopters"] == 0
    assert not m["null_success"]
    assert m["reference_script_world_reads"] == 0          # the generator itself is tool-only too
    assert not checker.unmet("SP-9.2", m), checker.unmet("SP-9.2", m)
