"""Gate SP-9.13 - Tier 2 domain Digital marketplace (MASTER_SPEC Part 9.13, workstream I-3).

Every required mechanic (two-sided participation, discovery, transactions, trust, quality, disputes, pricing,
network effects, retention) must be exercised by recorded events in an integrated run of a generated
configuration, with the shared population as the autonomous participants (buyers; traders and incumbent operators
are autonomous too), observation_only by default and the world reached through `crucible.experience.make(2, ...)`.

Measured on every configuration of the domain (three training, one compositional and one structural evaluation
configuration), each a freshly generated instance driven through the learner interface for 45 days by the domain's
scripted, tool-only exchange operator (labelled scripted); a mechanic counts only when it is exercised in EVERY
configuration's run, judged from the recorded log (`marketplace_domain.mechanics`)."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import checker
from crucible.acceptance.v5.evidence import record
from crucible.domains import _dom3 as K
from crucible.domains import marketplace_domain as M
from crucible.experience.leakage import find_keys

pytestmark = [pytest.mark.slow, pytest.mark.timeout(5400)]


def test_SP_9_13_digital_marketplace_mechanics_exercised():
    per, notes, evals = {}, [], {}
    mode_ok, reach_ok = True, True
    for cfg in [c.name for c in M.DOMAIN.configs]:
        env = K.make_env("marketplace", cfg, 5)             # the default reward mode
        a = env.agents()[0]
        mode_ok &= env.reward_mode == "observation_only"
        r = K.run_env(env, a, M.Reference(env.world, a), days=45, dt=6 * 3600)
        obs = env.observe(a)
        mode_ok &= r["reward_nonzero"] == 0 and env.reward(a) == 0.0 and not find_keys(obs)
        ver = env.verify()
        reach_ok &= r["actions_ok"] > 0 and a in (ver.get("evaluation") or {}) and not ver["invariant_violations"]
        m = M.mechanics(env.world)
        per[cfg] = {k: bool(v["exercised"]) for k, v in m.items()}
        evals[cfg] = ver["evaluation"][a]
        notes.append(f"{cfg}: " + ", ".join(f"{k}={v['events']}" for k, v in m.items()) +
                     f"; evaluation {evals[cfg]['details']}")
    exercised = [k for k in M.MECHANICS if all(per[c].get(k) for c in per)]
    metrics = {"required_mechanics_listed": len(M.DOMAIN.required_mechanics),
               "required_mechanics_exercised": len(exercised),
               "required_mechanics_missing": len(M.DOMAIN.required_mechanics) - len(exercised),
               "observation_only_default": bool(mode_ok), "reachable_through_experience": bool(reach_ok),
               "configurations_run": len(per),
               "reference_successes": sum(bool(e["success"]) for e in evals.values())}
    record("SP-9.13", metrics, labels={"scripted": True},
           notes="Scripted tool-only exchange operator (marketplace_domain.Reference), 45 simulated days per "
                 "configuration, seed 5. " + " | ".join(notes)[:6000])
    assert tuple(M.DOMAIN.required_mechanics) == M.MECHANICS and len(M.MECHANICS) == 9
    assert not checker.unmet("SP-9.13", metrics), (metrics, per)
