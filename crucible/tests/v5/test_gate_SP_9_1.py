"""SP-9.1 - Tier 2 domain Startup economy (MASTER_SPEC Part 9.1; workstream I-1).

Integrated runs through the learner interface (`crucible.domains.base.section_metrics` / `base.run` ->
`crucible.experience.make(2, "startup:<config>", seed)`, default reward mode) of `crucible.domains.startup`: the
reference founder (LABELLED SCRIPTED, tool-only) on seeds 0 and 1 of the first training configuration over the
75-day horizon, while the population's professionals do their own tasks, try services on them, subscribe, churn,
ask for help and recommend; incumbents bill and react. Every required mechanic is counted by the framework's
recorded-event probes. Controls on the same instance: the null baseline (the seat never acts: what the autonomous
participants exercise alone) and the reference's whole playbook with a service that ignores its input (demand must
depend on what the product does, not on the playbook).
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import checker, evidence  # noqa: E402
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains import startup as D  # noqa: E402

CMD = "PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/v5/test_gate_SP_9_1.py"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(4 * 3600)]


def _arr(run) -> float:
    return max([o["arr_usd"] for o in (run["outcome"].get("details") or {}).get("orgs", {}).values()] or [0.0])


def test_SP_9_1_startup_economy_mechanics():
    d = base.part9_domains()["startup"]
    assert d is D.DOMAIN
    m, runs = base.section_metrics("startup")
    cfg = d.reference_configs[0] if d.reference_configs else d.split("train")[0].name
    null = base.cached_run("startup", cfg, 0, "null", None)
    garbage = base.cached_run("startup", cfg, 0, "garbage_service", None)
    m.update({"mechanics_exercised_without_seat": null["mechanics"]["exercised"],
              "autonomous_actors_active": null["actors"]["active"],
              "population_activity_decisions_without_seat": null["population"]["activity_decisions"],
              "reference_arr_usd_min": min(_arr(r) for r in runs), "garbage_service_arr_usd": _arr(garbage),
              "null_success": bool(null["success"]), "garbage_service_success": bool(garbage["success"]),
              "invariant_violations": sum(r["invariant_violations"] for r in runs + [null, garbage]),
              "reference_script_world_reads": len(base.reads_world(D.reference_script)),
              "process_failures_all_runs": sum(r["proc_failed"] for r in runs + [null, garbage])})
    evidence.record("SP-9.1", m, command=CMD, labels=dict(K.SCRIPTED),
                    notes=f"startup:{cfg} seeds {[r['seed'] for r in runs]}, {runs[0]['days']} days; LABELLED "
                          f"SCRIPTED reference founder (tool-only). Mechanic counts (reference runs): "
                          f"{[r['mechanics']['counts'] for r in runs]}; null run: {null['mechanics']['counts']}, "
                          f"actors {null['actors']['counts']}; outcomes: {[r['outcome'].get('details') for r in runs]}; "
                          f"input-ignoring service: {garbage['outcome'].get('details')}")
    assert m["required_mechanics_missing"] == 0, [r["mechanics"]["missing"] for r in runs]
    assert m["autonomous_actors_active"] == len(d.actors) and m["population_activity_decisions_without_seat"] > 0
    assert m["process_failures_all_runs"] == 0 and m["invariant_violations"] == 0
    assert m["reference_arr_usd_min"] > 0 and m["garbage_service_arr_usd"] < 0.1 * m["reference_arr_usd_min"]
    assert not m["null_success"] and not m["garbage_service_success"]
    assert m["reference_script_world_reads"] == 0          # the generator itself is tool-only too
    assert not checker.unmet("SP-9.1", m), checker.unmet("SP-9.1", m)
