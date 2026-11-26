"""Gate SP-9.12 - Tier 2 domain Organizational ecosystem (MASTER_SPEC Part 9.12, workstream I-3).

Every required mechanic (organizations, differing goals, contracts, negotiation, coordination, shared resources,
governance, commitments, collective outcomes) must be exercised by recorded events in an integrated run of a
generated configuration, with the shared population as the autonomous participants (customers of the providers;
the providers' managers are autonomous too), observation_only by default and the world reached through
`crucible.experience.make(2, ...)`.

Measured on every configuration of the domain (three training, one compositional and one structural evaluation
configuration), each a freshly generated instance driven through the learner interface for the objective's 40 days
by the domain's scripted, tool-only convener (labelled scripted); a mechanic counts only when it is exercised in
EVERY configuration's run, judged from the recorded log (`organization.mechanics`)."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import checker
from crucible.acceptance.v5.evidence import record
from crucible.domains import _dom3 as K
from crucible.domains import organization as C
from crucible.experience.leakage import find_keys

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_SP_9_12_organizational_ecosystem_mechanics_exercised():
    per, notes, evals = {}, [], {}
    mode_ok, reach_ok = True, True
    for cfg in [c.name for c in C.DOMAIN.configs]:
        env = K.make_env("organization", cfg, 6)                  # the default reward mode
        a = env.agents()[0]
        mode_ok &= env.reward_mode == "observation_only"
        r = K.run_env(env, a, C.Reference(env.world, a), days=40, dt=6 * 3600)
        obs = env.observe(a)
        mode_ok &= r["reward_nonzero"] == 0 and env.reward(a) == 0.0 and not find_keys(obs)
        ver = env.verify()
        reach_ok &= r["actions_ok"] > 0 and a in (ver.get("evaluation") or {}) and not ver["invariant_violations"]
        m = C.mechanics(env.world)
        per[cfg] = {k: bool(v["exercised"]) for k, v in m.items()}
        evals[cfg] = ver["evaluation"][a]["details"]
        notes.append(f"{cfg}: " + ", ".join(f"{k}={v['events']}" for k, v in m.items()) +
                     f"; evaluation {evals[cfg]}")
    exercised = [k for k in C.MECHANICS if all(per[c].get(k) for c in per)]
    metrics = {"required_mechanics_listed": len(C.DOMAIN.required_mechanics),
               "required_mechanics_exercised": len(exercised),
               "required_mechanics_missing": len(C.DOMAIN.required_mechanics) - len(exercised),
               "observation_only_default": bool(mode_ok), "reachable_through_experience": bool(reach_ok),
               "configurations_run": len(per),
               "reference_service_min": round(min(e["request_success"] for e in evals.values()), 4)}
    record("SP-9.12", metrics, labels={"scripted": True},
           notes="Scripted tool-only convener (organization.Reference), 40 simulated days per configuration, seed 6. "
                 + " | ".join(notes)[:6000])
    assert tuple(C.DOMAIN.required_mechanics) == C.MECHANICS and len(C.MECHANICS) == 9
    assert not checker.unmet("SP-9.12", metrics), (metrics, per)
