"""SP-9.4 - Tier 2 domain Community ecosystem (MASTER_SPEC Part 9.4; workstream I-1).

Integrated runs through the learner interface (`base.section_metrics` / `base.run` -> `crucible.experience.make(2,
"community:<config>", seed)`, default reward mode) of `crucible.domains.community`: the reference organiser
(LABELLED SCRIPTED, tool-only) on seeds 0 and 1 of the first training configuration over the 28-day horizon, while
the population visits, posts, replies, reports, moderates, joins, leaves and founds communities (`social.popsocial`)
and votes, proposes, schedules and attends events, invites and messages its contacts and moves away from
abandoned communities (`community.commlife`). Every required mechanic is counted by the framework's recorded-event
probes. Controls: the null baseline, and a founder who founds a community and never takes part (the members of an
owner-ruled community with an absent owner found a new home: sustained participation must need an organiser).
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import checker, evidence  # noqa: E402
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains import community as D  # noqa: E402

CMD = "PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/v5/test_gate_SP_9_4.py"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(6 * 3600)]


def test_SP_9_4_community_ecosystem_mechanics():
    d = base.part9_domains()["community"]
    assert d is D.DOMAIN
    m, runs = base.section_metrics("community")
    cfg = d.reference_configs[0] if d.reference_configs else d.split("train")[0].name
    null = base.cached_run("community", cfg, 0, "null", None)
    idle = base.cached_run("community", cfg, 0, "found_and_idle", None)
    m.update({"mechanics_exercised_without_seat": null["mechanics"]["exercised"],
              "autonomous_actors_active": null["actors"]["active"],
              "population_activity_decisions_without_seat": null["population"]["activity_decisions"],
              "reference_successes": sum(1 for r in runs if r["success"]),
              "idle_founder_success": bool(idle["success"]), "idle_founder_evolution_events": int(idle["mechanics"]["counts"]["Community evolution"]),
              "null_success": bool(null["success"]),
              "invariant_violations": sum(r["invariant_violations"] for r in runs + [null, idle]),
              "reference_script_world_reads": len(base.reads_world(D.reference_script)),
              "process_failures_all_runs": sum(r["proc_failed"] for r in runs + [null, idle])})
    evidence.record("SP-9.4", m, command=CMD, labels=dict(K.SCRIPTED),
                    notes=f"community:{cfg} seeds {[r['seed'] for r in runs]}, {runs[0]['days']} days; LABELLED "
                          f"SCRIPTED reference organiser (tool-only). Mechanic counts "
                          f"{[r['mechanics']['counts'] for r in runs]}; null {null['mechanics']['counts']}, actors "
                          f"{null['actors']['counts']}; outcomes {[r['outcome'].get('details') for r in runs]}; idle "
                          f"founder {idle['outcome'].get('details')}")
    assert m["required_mechanics_missing"] == 0, [r["mechanics"]["missing"] for r in runs]
    assert m["autonomous_actors_active"] == len(d.actors) and m["population_activity_decisions_without_seat"] > 0
    assert m["process_failures_all_runs"] == 0 and m["invariant_violations"] == 0
    assert not m["null_success"] and not m["idle_founder_success"]
    assert m["reference_script_world_reads"] == 0          # the generator itself is tool-only too
    assert not checker.unmet("SP-9.4", m), checker.unmet("SP-9.4", m)
