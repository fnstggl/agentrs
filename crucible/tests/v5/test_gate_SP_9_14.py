"""Gate SP-9.14 - Tier 2 domain Crisis recovery (MASTER_SPEC Part 9.14, workstream I-3).

Every required mechanic (cascading disruptions, scarce resources, prioritization, coordination, recovery,
resilience, independent service-restoration metrics) must be exercised by recorded events in an integrated run of
a generated configuration, with the shared population as the autonomous participants (households), observation_only
by default and the world reached through `crucible.experience.make(2, ...)`.

Measured on every configuration of the domain (three training, one compositional and one structural evaluation
configuration), each a freshly generated instance driven through the learner interface for the objective's 10 days
by the domain's scripted, tool-only coordinator (labelled scripted); a mechanic counts only when it is exercised in
EVERY configuration's run, judged from the recorded log (`crisis.mechanics`). The evaluator's counterfactual (the
same disruption with the operators acting alone) is part of the independent service-restoration metrics."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import checker
from crucible.acceptance.v5.evidence import record
from crucible.domains import _dom3 as K
from crucible.domains import crisis as C
from crucible.experience.leakage import find_keys

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_SP_9_14_crisis_recovery_mechanics_exercised():
    per, notes, evals = {}, [], {}
    mode_ok, reach_ok = True, True
    for cfg in [c.name for c in C.DOMAIN.configs]:
        env = K.make_env("crisis", cfg, 4)                  # the default reward mode
        a = env.agents()[0]
        mode_ok &= env.reward_mode == "observation_only"
        r = K.run_env(env, a, C.Reference(env.world, a), days=10, dt=2 * 3600)
        obs = env.observe(a)
        mode_ok &= r["reward_nonzero"] == 0 and env.reward(a) == 0.0 and not find_keys(obs)
        ver = env.verify()
        reach_ok &= r["actions_ok"] > 0 and a in (ver.get("evaluation") or {}) and not ver["invariant_violations"]
        m = C.mechanics(env.world)
        per[cfg] = {k: bool(v["exercised"]) for k, v in m.items()}
        evals[cfg] = ver["evaluation"][a]["details"]
        notes.append(f"{cfg}: " + ", ".join(f"{k}={v['events']}" for k, v in m.items()) +
                     f"; loss ratio vs uncoordinated {evals[cfg]['loss_ratio_vs_uncoordinated']}")
    exercised = [k for k in C.MECHANICS if all(per[c].get(k) for c in per)]
    metrics = {"required_mechanics_listed": len(C.DOMAIN.required_mechanics),
               "required_mechanics_exercised": len(exercised),
               "required_mechanics_missing": len(C.DOMAIN.required_mechanics) - len(exercised),
               "observation_only_default": bool(mode_ok), "reachable_through_experience": bool(reach_ok),
               "configurations_run": len(per),
               "mean_loss_ratio_vs_uncoordinated": round(sum(e["loss_ratio_vs_uncoordinated"] for e in evals.values())
                                                         / len(evals), 4)}
    record("SP-9.14", metrics, labels={"scripted": True},
           notes="Scripted tool-only coordinator (crisis.Reference), 10 simulated days per configuration, seed 4. "
                 + " | ".join(notes)[:6000])
    assert tuple(C.DOMAIN.required_mechanics) == C.MECHANICS and len(C.MECHANICS) == 7
    assert not checker.unmet("SP-9.14", metrics), (metrics, per)
