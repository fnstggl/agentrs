"""SP-9.5 - Tier 2 domain Scientific research (MASTER_SPEC Part 9.5; workstream I-1).

Integrated runs through the learner interface (`base.section_metrics` / `base.run` -> `crucible.experience.make(2,
"research:<config>", seed)`, default reward mode) of `crucible.domains.research_domain`: the reference researcher
(LABELLED SCRIPTED, tool-only) on seeds 0 and 1 of the first training configuration over the 30-day horizon, while
the population's researchers (module `scipop`) study the hidden-law phenomenon with their own money, publish and
replicate each other's papers and the seat's paper, and the standards bureau runs blind challenges at novel
conditions. Every required mechanic is counted by the framework's recorded-event probes. Controls: the null
baseline (what the autonomous participants exercise alone, with decision provenance), the other two training
configurations (12 days, seat idle), and an intervention arm without population researchers (no independent
replication is recorded and the seat's claim is never independently tested).
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import checker, evidence  # noqa: E402
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains import research_domain as RD  # noqa: E402

CMD = "PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/v5/test_gate_SP_9_5.py"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3 * 3600)]
EVALUATOR_KEYS = ("success", "score", "novel_nrmse", "best_novel_nrmse", "priority", "evaluation", "claim")


def test_SP_9_5_scientific_research_mechanics():
    d = base.part9_domains()["research"]
    assert d is RD.DOMAIN
    m, runs = base.section_metrics("research")
    cfg = d.split("train")[0].name
    null = base.run("research", cfg, 0, policy="null", keep_env=True)
    w = null["env"].world
    prov = K.decision_provenance(w, ("scipop.",))
    researchers = len(w.state.tables["scipop_r"])
    pop_papers = sum(1 for _, p in w.state.scan("sci_papers") if K.is_population(p["author"]))
    obs = null["env"].observe(null["seats"][0])
    leaks = K.observation_leaks(obs, EVALUATOR_KEYS)
    law_in_obs = any(w.modules["sci"]._phen_cfg(pid)["law"] in str(obs) for pid in w.modules["sci"].laws)
    null["env"].close()
    no_pop = base.run("research", cfg, 0, policy="reference",
                      overrides={"module_config": {"scipop": {"share": 0.0, "min": 0}}})
    others = {}
    for c in ("drift", "survey"):
        r2 = base.run("research", c, 1, policy="null", days=12)
        others[c] = {"exercised": r2["mechanics"]["exercised"], "missing": r2["mechanics"]["missing"],
                     "proc_failed": r2["proc_failed"], "invariant_violations": r2["invariant_violations"]}
    m.update({"mechanics_exercised_without_seat": null["mechanics"]["exercised"],
              "autonomous_actors_active": null["actors"]["active"],
              "population_researchers": researchers, "population_papers": pop_papers,
              "population_decision_batches": prov["batches"], "forced_decision_batches": prov["forced"],
              "observation_evaluator_keys": len(leaks), "hidden_law_in_observation": bool(law_in_obs),
              "ablated_independent_replications": no_pop["mechanics"]["counts"]["Independent replication"],
              "ablated_reference_success": bool(no_pop["success"]),
              "null_success": bool(null["success"]),
              "invariant_violations": sum(r["invariant_violations"] for r in runs + [null, no_pop]),
              "process_failures_all_runs": sum(r["proc_failed"] for r in runs + [null, no_pop])})
    evidence.record("SP-9.5", m, command=CMD, labels=dict(K.SCRIPTED),
                    notes=f"research:{cfg} seeds {[r['seed'] for r in runs]}, {runs[0]['days']} days; LABELLED "
                          f"SCRIPTED reference researcher (tool-only). Mechanic counts "
                          f"{[r['mechanics']['counts'] for r in runs]}; null {null['mechanics']['counts']}, actors "
                          f"{null['actors']['counts']}; outcomes {[r['outcome'].get('details') for r in runs]}; "
                          f"without population researchers: {no_pop['mechanics']['counts']}; other training "
                          f"configurations (seat idle, 12 days): {others}")
    assert m["required_mechanics_missing"] == 0, [r["mechanics"]["missing"] for r in runs]
    assert m["mechanics_exercised_without_seat"] >= 7 and m["autonomous_actors_active"] == len(d.actors)
    assert prov["batches"] > 0 and prov["forced"] == 0, prov
    assert researchers >= 10 and pop_papers >= 3
    assert not leaks and not law_in_obs
    assert m["ablated_independent_replications"] == 0 < m["events_independent_replication"]
    assert not m["ablated_reference_success"] and not m["null_success"]
    assert m["process_failures_all_runs"] == 0 and m["invariant_violations"] == 0
    for c, o in others.items():
        assert o["exercised"] >= 7 and o["proc_failed"] == 0 and o["invariant_violations"] == 0, (c, o)
    assert not checker.unmet("SP-9.5", m), checker.unmet("SP-9.5", m)
