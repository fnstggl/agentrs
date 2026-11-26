"""Gate SP-9.15 - Tier 2 domain Research commercialization (MASTER_SPEC Part 9.15, workstream I-3).

Every required mechanic (research, experimentation, technical development, product creation, financing,
deployment, independent adoption, commercial sustainability) must be exercised by recorded events in an integrated
run of a generated configuration, with the shared population as the autonomous participants (practitioners who
adopt or reject the product on their own bills; labs, investors and apparatus operators are autonomous too),
observation_only by default, reachable through `crucible.experience.make(2, ...)`; and the domain must genuinely
connect scientific discovery and economic outcomes, confirmed by counterfactual removal of the discovery.

Mechanics: every configuration (three training, one compositional, one structural evaluation configuration), each a
freshly generated instance driven through the learner interface for the objective's 100 days by the domain's
scripted, tool-only founder (labelled scripted); a mechanic counts only when exercised in EVERY configuration's run.
Counterfactual removal: the same world (configuration and seed) driven by the same founder with the discovery
removed - it builds its advisor from common practice (the population's habitual setting, the most generous
knowledge available without the phenomenon's data) instead of from its measurements - on two seeds. The link is
confirmed when, on every seed, the discovery founder succeeds while the founder without it fails, and the
discovery at least doubles the retained independent adopters and raises the last 30 days' revenue."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import checker
from crucible.acceptance.v5.evidence import record
from crucible.domains import _dom3 as K
from crucible.domains import commercialization as R
from crucible.experience.leakage import find_keys

pytestmark = [pytest.mark.slow, pytest.mark.timeout(5400)]


def _run(cfg, seed, discovery=True):
    env = K.make_env("commercialization", cfg, seed)
    a = env.agents()[0]
    common = env.world.modules["rcomm"].c["own_setting"][0]
    pol = R.Reference(env.world, a, discovery=discovery, common_practice=common)
    r = K.run_env(env, a, pol, days=100, dt=12 * 3600)
    return env, a, r


def test_SP_9_15_research_commercialization_mechanics_and_discovery_link():
    per, notes = {}, []
    mode_ok, reach_ok = True, True
    for cfg in [c.name for c in R.DOMAIN.configs]:
        env, a, r = _run(cfg, 2)
        mode_ok &= env.reward_mode == "observation_only" and r["reward_nonzero"] == 0 and env.reward(a) == 0.0
        mode_ok &= not find_keys(env.observe(a))
        ver = env.verify()
        reach_ok &= r["actions_ok"] > 0 and a in (ver.get("evaluation") or {}) and not ver["invariant_violations"]
        m = R.mechanics(env.world)
        per[cfg] = {k: bool(v["exercised"]) for k, v in m.items()}
        notes.append(f"{cfg}: " + ", ".join(f"{k}={v['events']}" for k, v in m.items()) +
                     f"; evaluation {ver['evaluation'][a]['details']}")
    # counterfactual removal of the discovery, same worlds
    link = []
    for seed in (2, 3):
        env_a, a_a, _ = _run("linear_optimum", seed, True)
        env_b, a_b, _ = _run("linear_optimum", seed, False)
        ea = R.evaluate(env_a.world, a_a, R.objective(env_a.world, a_a, 1))
        eb = R.evaluate(env_b.world, a_b, R.objective(env_b.world, a_b, 1))
        da, db = ea["details"], eb["details"]
        ok = ea["success"] and not eb["success"] and da["adopters_retained"] >= 2 * max(1, db["adopters_retained"]) \
            and da["revenue_30d_cents"] > db["revenue_30d_cents"]
        link.append(ok)
        notes.append(f"counterfactual seed {seed}: with discovery {da}; without {db}")
    exercised = [k for k in R.MECHANICS if all(per[c].get(k) for c in per)]
    metrics = {"required_mechanics_listed": len(R.DOMAIN.required_mechanics),
               "required_mechanics_exercised": len(exercised),
               "required_mechanics_missing": len(R.DOMAIN.required_mechanics) - len(exercised),
               "observation_only_default": bool(mode_ok), "reachable_through_experience": bool(reach_ok),
               "discovery_to_economic_outcome_confirmed": bool(link and all(link)),
               "configurations_run": len(per), "counterfactual_seeds": len(link)}
    record("SP-9.15", metrics, labels={"scripted": True},
           notes="Scripted tool-only founder (commercialization.Reference), 100 simulated days per run. "
                 + " | ".join(notes)[:8000])
    assert tuple(R.DOMAIN.required_mechanics) == R.MECHANICS and len(R.MECHANICS) == 8
    assert not checker.unmet("SP-9.15", metrics), (metrics, per)
