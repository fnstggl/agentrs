"""SP-9.5 - Tier 2 domain Scientific research (MASTER_SPEC Part 9.5; workstream I-1).

One integrated run of the domain's first training configuration (`research:bench`, seed 0, 30 simulated days)
built through `crucible.experience.make(2, ...)` with the default reward mode: the population's researchers
(`crucible.domains.research_domain`, module `scipop`) study the hidden-law phenomenon with their own money, publish,
replicate each other's papers and the learner seat's paper, and the standards bureau runs blind challenges at novel
conditions. The learner seat is driven by the domain's hidden reference (LABELLED SCRIPTED, tool-only through the
experience environment). Every required mechanic is measured on the recorded world by
`research_domain.mechanics` (events and state), never asserted by the domain. The other two training configurations
are run for 12 days and their mechanics reported (structurally different worlds exercise the same mechanics).
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible import experience  # noqa: E402
from crucible.acceptance.v5 import checker, evidence  # noqa: E402
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import research_domain as RD  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
EVALUATOR_KEYS = ("success", "score", "novel_nrmse", "first_accurate_paper", "best_novel_nrmse", "evaluation")


def _run(config: str, seed: int, days: float):
    env = experience.make(2, f"research:{config}", seed, semantic=False)      # default reward mode
    agent = env.agents()[0]
    obj = RD.DOMAIN.objective(env.world, agent, 1)
    pol = RD.reference(env.world, agent, obj)
    first_obs = env.observe(agent)
    out = K.drive(env, agent, pol, days=days, dt=6 * 3600)
    return env, agent, obj, pol, first_obs, out


def test_SP_9_5_scientific_research_mechanics():
    env, agent, obj, pol, first_obs, drive = _run("bench", 0, 30)
    w = env.world
    mech = RD.DOMAIN.mechanics(w)
    exercised = [m for m in RD.REQUIRED if mech[m]["exercised"]]
    missing = [m for m in RD.REQUIRED if not mech[m]["exercised"]]
    prov = K.decision_provenance(w, ("scipop.",))
    obs = env.observe(agent)
    leaks = K.observation_leaks(obs, EVALUATOR_KEYS) + K.observation_leaks(first_obs, EVALUATOR_KEYS)
    rewards = [env.reward(agent)]
    obs_only = env.reward_mode == "observation_only" and all(r == 0.0 for r in rewards) and not leaks
    ok_actions = sum(1 for x in pol.log if x["ok"])
    reachable = bool(env.tier == 2 and env.domain == "research" and ok_actions >= 3 and agent == "p:researcher")
    ev = env.verify()["evaluation"][agent]                                     # evaluator side only
    others = {}
    for cfg in ("drift", "survey"):
        e2, a2, _, _, _, _ = _run(cfg, 1, 12)
        m2 = RD.DOMAIN.mechanics(e2.world)
        others[cfg] = sorted(m for m in RD.REQUIRED if m2[m]["exercised"])
    researchers = len(w.state.tables["scipop_r"])
    pop_papers = sum(1 for _, p in w.state.scan("sci_papers") if K.is_population(p["author"]))
    metrics = {"required_mechanics_listed": len(RD.REQUIRED), "required_mechanics_exercised": len(exercised),
               "required_mechanics_missing": len(missing), "observation_only_default": bool(obs_only),
               "reachable_through_experience": reachable,
               "population_researchers": researchers, "population_papers": pop_papers,
               "population_decision_batches": prov["batches"], "forced_decision_batches": prov["forced"],
               "independent_population_replications": mech["Independent replication"][
                   "independent_reports_by_population"],
               "challenges": mech["Novel-condition evaluation"]["challenges"],
               "invariant_violations": len(w.check_invariants()),
               "reference_success": bool(ev["success"]), "reference_failures": len(drive["failures"])}
    evidence.record("SP-9.5", metrics, labels=dict(K.SCRIPTED),
                    notes=f"research:bench seed 0, 30 days; learner seat driven by the LABELLED SCRIPTED reference "
                          f"(tool-only). Mechanics: {mech}. Other training configurations (12 days, seed 1): "
                          f"{others}. Evaluator: {ev['details']}")
    assert not missing, missing
    assert prov["batches"] > 0 and prov["forced"] == 0, prov
    assert researchers >= 10 and pop_papers >= 3
    assert metrics["invariant_violations"] == 0
    for cfg, ex in others.items():
        assert len(ex) >= 7, (cfg, ex)
    assert not checker.unmet("SP-9.5", metrics), checker.unmet("SP-9.5", metrics)
