"""Gate SP-9.11 - Tier 2 domain Information ecosystem (MASTER_SPEC Part 9.11, workstream I-3).

Every required mechanic (search, documents, knowledge artifacts, queries, source provenance, information quality,
updating facts, independent user satisfaction, verified retrieval performance) must be exercised by recorded events
in an integrated run of a generated configuration, with the shared population as the autonomous participants,
observation_only by default and the world reached through `crucible.experience.make(2, ...)`.

Measured here on every configuration of the domain (three training, one compositional and one structural evaluation
configuration), each a freshly generated instance driven through the learner interface by the domain's scripted,
tool-only reference (labelled scripted): a mechanic counts only when it is exercised in EVERY configuration's run,
judged from the world's recorded log (`information.mechanics`). Observation-only is checked on the default
constructor: the reward mode, a zero reward at every step and no evaluator key in any observation."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import checker
from crucible.acceptance.v5.evidence import record
from crucible.domains import _dom3 as K
from crucible.domains import information as I
from crucible.experience.leakage import find_keys

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_SP_9_11_information_ecosystem_mechanics_exercised():
    per, runs, notes = {}, {}, []
    mode_ok, reach_ok = True, True
    for cfg in [c.name for c in I.DOMAIN.configs]:
        env = K.make_env("information", cfg, 11)            # the default reward mode
        a = env.agents()[0]
        mode_ok &= env.reward_mode == "observation_only"
        pol = I.Reference(env.world, a)
        r = K.run_env(env, a, pol, days=20)
        obs = env.observe(a)
        leaked = find_keys(obs)
        mode_ok &= r["reward_nonzero"] == 0 and env.reward(a) == 0.0 and not leaked
        ver = env.verify()
        reach_ok &= r["actions_ok"] > 0 and a in (ver.get("evaluation") or {}) and not ver["invariant_violations"]
        m = I.mechanics(env.world)
        per[cfg] = {k: bool(v["exercised"]) for k, v in m.items()}
        runs[cfg] = {"actions_ok": r["actions_ok"], "events": env.world.seq,
                     "evaluation": ver["evaluation"][a]["details"]}
        notes.append(f"{cfg}: " + ", ".join(f"{k}={v['events']}" for k, v in m.items()))
    exercised = [k for k in I.MECHANICS if all(per[c].get(k) for c in per)]
    metrics = {"required_mechanics_listed": len(I.DOMAIN.required_mechanics),
               "required_mechanics_exercised": len(exercised),
               "required_mechanics_missing": len(I.DOMAIN.required_mechanics) - len(exercised),
               "observation_only_default": bool(mode_ok), "reachable_through_experience": bool(reach_ok),
               "configurations_run": len(per),
               "service_lookups_total": int(sum(v["evaluation"]["lookups"] for v in runs.values()))}
    record("SP-9.11", metrics, labels={"scripted": True},
           notes="Scripted tool-only reference (information.Reference), 20 simulated days per configuration, seed 11. "
                 + " | ".join(notes)[:6000])
    assert tuple(I.DOMAIN.required_mechanics) == I.MECHANICS and len(I.MECHANICS) == 9
    assert not checker.unmet("SP-9.11", metrics), (metrics, per)
