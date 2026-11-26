"""SP-9.5 - Tier 2 domain Scientific research (MASTER_SPEC Part 9.5; workstream I-1).

Integrated runs of the domain through the learner interface (`crucible.domains.base.run` ->
`crucible.experience.make(2, "research:<config>", seed)`, default reward mode):

* the reference run (research:bench, seed 0, the 30-day horizon): the learner seat is driven by the domain's hidden
  reference (LABELLED SCRIPTED, tool-only), while the population's researchers (`research_domain`, module `scipop`)
  study the hidden-law phenomenon with their own money, publish, replicate each other's papers and the seat's paper,
  and the standards bureau runs blind challenges at novel conditions;
* the null-baseline run (same instance, the seat never acts): what the autonomous participants alone exercise;
* the other two training configurations (12 days, seed 1): structurally different worlds exercise the mechanics too.

Every required mechanic is counted by the framework's recorded-event probes (`base.mechanics_exercised`: event
types after genesis plus counts measured from rows the appliers wrote), never declared by the domain.
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import checker, evidence  # noqa: E402
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains import research_domain as RD  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
EVALUATOR_KEYS = ("success", "score", "novel_nrmse", "best_novel_nrmse", "priority", "evaluation", "claim")


def test_SP_9_5_scientific_research_mechanics():
    ref = base.run("research", "bench", 0, policy="reference", keep_env=True)
    env = ref["env"]
    w = env.world
    seat = ref["seats"][0]
    obs = env.observe(seat)
    leaks = K.observation_leaks(obs, EVALUATOR_KEYS)
    obs_only = (ref["reward_mode"] == "observation_only" and env.reward(seat) == 0.0 and not leaks
                and "evaluation" not in obs and obs.get("objective"))
    drv = ref["drive"]["agents"][seat]
    reachable = bool(env.tier == 2 and env.domain == "research" and drv["ok"] >= 3 and seat == "p:researcher")
    prov = K.decision_provenance(w, ("scipop.",))
    researchers = len(w.state.tables["scipop_r"])
    pop_papers = sum(1 for _, p in w.state.scan("sci_papers") if K.is_population(p["author"]))
    mech = ref["mechanics"]
    env.close()
    null = base.run("research", "bench", 0, policy="null")
    others = {}
    for cfg in ("drift", "survey"):
        r2 = base.run("research", cfg, 1, policy="null", days=12)
        others[cfg] = {"exercised": r2["mechanics"]["exercised"], "missing": r2["mechanics"]["missing"],
                       "proc_failed": r2["proc_failed"], "invariant_violations": r2["invariant_violations"]}
    metrics = {"required_mechanics_listed": mech["listed"], "required_mechanics_exercised": mech["exercised"],
               "required_mechanics_missing": len(mech["missing"]), "observation_only_default": bool(obs_only),
               "reachable_through_experience": reachable,
               "mechanics_exercised_without_seat": null["mechanics"]["exercised"],
               "population_researchers": researchers, "population_papers": pop_papers,
               "population_decision_batches": prov["batches"], "forced_decision_batches": prov["forced"],
               "autonomous_actors_active": null["actors"]["active"],
               "independent_replications": mech["counts"]["Independent replication"],
               "challenge_tests": mech["counts"]["Novel-condition evaluation"],
               "process_failures": ref["proc_failed"] + null["proc_failed"],
               "invariant_violations": ref["invariant_violations"] + null["invariant_violations"],
               "reference_success": bool(ref["success"]), "null_success": bool(null["success"])}
    evidence.record("SP-9.5", metrics, labels=dict(K.SCRIPTED),
                    notes=f"research:bench seed 0, {ref['days']} days; the learner seat is driven by the LABELLED "
                          f"SCRIPTED reference (tool-only). Mechanic counts (reference run): {mech['counts']}; "
                          f"null-baseline run: {null['mechanics']['counts']}; actors {null['actors']['counts']}. "
                          f"Other training configurations (seed 1, 12 days, seat idle): {others}. Evaluator: "
                          f"{ref['outcome'].get('details')}")
    assert not mech["missing"], mech
    assert null["mechanics"]["exercised"] >= 7, null["mechanics"]
    assert null["actors"]["active"] == null["actors"]["declared"] == 2
    assert prov["batches"] > 0 and prov["forced"] == 0, prov
    assert researchers >= 10 and pop_papers >= 3
    assert metrics["process_failures"] == 0 and metrics["invariant_violations"] == 0
    for cfg, o in others.items():
        assert o["exercised"] >= 7 and o["proc_failed"] == 0 and o["invariant_violations"] == 0, (cfg, o)
    assert not checker.unmet("SP-9.5", metrics), checker.unmet("SP-9.5", metrics)
