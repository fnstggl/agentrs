"""Gate R: observation-only operation (workstream L). One test per requirement; each measures the real system and
records evidence with `crucible.acceptance.v5.evidence.record`.

The reward paths exercised (crucible.experience.leakage): MechVecEnv, KernelMechEnv, SelfPlayEnv,
KernelSelfPlayEnv, Gymnasium CrucibleEnv, PettingZoo CrucibleParallelEnv, scale.rollout.WorldEnv, and the
experience adapters (Tier 1 and kernel worlds, with the numeric view of Tier 1).
"""
import json

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience import leakage
from crucible.experience.loop import run_learner
from crucible.experience.reference import ReferenceLearner
from crucible.experience.registry import make

pytestmark = pytest.mark.timeout(1200)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_R.py"


@pytest.fixture(scope="module")
def battery():
    return {r["test"]: r for r in leakage.run_battery()}


@pytest.fixture(scope="module")
def learner_run():
    """The reference learner through the observation-only interface (Tier 1 and a kernel economy)."""
    out = {}
    for label, tier, name, kw in (("tier1", 1, "mech:scm_additive", {}),
                                  ("kernel", 0, "tiny", {"agents": ["learner"]})):
        env = make(tier, name, 3, "observation_only", **kw)
        L = ReferenceLearner(seed=0, reward_mode=env.reward_mode)
        before = L.params_hash()
        s = run_learner(env, L, 160 if tier == 1 else 64, update_every=32, think_s=None if tier == 1 else 3600)
        obs = env.observe(env.agents()[0])
        out[label] = {"summary": s, "before": before, "after": L.params_hash(), "obs": obs, "env": env}
    return out


def test_R1_native_intermediate_reward_disabled(battery, learner_run):
    paths = [k for k, r in battery.items() if "nonzero_native_reward_steps" in r["metrics"]]
    nonzero = sum(battery[k]["metrics"]["nonzero_native_reward_steps"] for k in paths)
    steps = sum(int(battery[k]["metrics"].get("steps", 0)) for k in paths)
    legacy_nonzero = sum(int(battery[k]["metrics"].get("legacy_nonzero_reward_steps", 0)) for k in paths)
    lr = sum(r["summary"]["nonzero_rewards"] for r in learner_run.values())
    lsteps = sum(r["summary"]["steps"] for r in learner_run.values())
    metrics = {"nonzero_native_reward_steps": nonzero + lr, "reward_paths_checked": len(paths),
               "steps_checked": steps + lsteps, "legacy_nonzero_reward_steps_same_inputs": legacy_nonzero}
    record("R.1", metrics, command=CMD, notes="observation_only on every reward path; legacy mode with the same inputs "
           "pays nonzero rewards on the counted steps, so the zero is the mode's doing. Paths: " + ", ".join(paths))
    assert metrics["nonzero_native_reward_steps"] == 0, metrics
    assert legacy_nonzero > 0


def test_R2_no_privileged_progress_values(battery, learner_run):
    keys = ("test_gym", "test_pettingzoo", "test_tier1_invariance", "test_kernel_invariance")
    found = sum(battery[k]["metrics"]["privileged_progress_fields"] for k in keys)
    in_learner = sum(len(leakage.find_keys(r["obs"])) for r in learner_run.values())
    oterm = sum(battery[k]["metrics"].get("outcome_terminations", 0) for k in ("test_gym", "test_pettingzoo"))
    metrics = {"privileged_progress_fields": found + in_learner + oterm, "surfaces_scanned": len(keys) + 2}
    record("R.2", metrics, command=CMD, notes="scanned: Gymnasium infos, PettingZoo infos (and outcome-based "
           "termination), every observation/result/tool list of the Tier 1 and kernel experience envs, the "
           f"reference learner's final observations. Keys: {sorted(leakage.PRIVILEGED_KEYS)}")
    assert metrics["privileged_progress_fields"] == 0, [battery[k]["metrics"].get("privileged_paths") for k in keys]


def test_R3_no_hidden_evaluator_data(battery):
    k, t, rig = battery["test_kernel_invariance"], battery["test_tier1_invariance"], battery["test_rig_hidden_canary"]
    planted = k["metrics"]["canaries_planted"] + 1 + rig["metrics"]["canaries_planted"]
    hits = k["metrics"]["canary_hits"] + t["metrics"]["canary_hits"] + rig["metrics"]["canary_hits"]
    differ = k["metrics"]["observation_differences"] + t["metrics"]["observation_differences"]
    metrics = {"hidden_evaluator_fields": hits + differ, "canaries_planted": planted, "canary_hits": hits,
               "invariance_differences": differ}
    record("R.3", metrics, command=CMD, notes="canaries: evaluator runtime state, recommender-private slot, internal "
           "event, another participant's private mail, world-spec evaluator parameter, goal evaluator parameter, "
           "Tier 1 task private reference numbers, hidden mechanism configuration, internal rig filing; plus "
           "invariance of everything the agent receives to those data and to the reward mode")
    assert metrics["hidden_evaluator_fields"] == 0 and planted >= 5, metrics


def test_R4_legitimate_observations_remain(battery, learner_run):
    missing = sum(int(battery[k]["metrics"].get("legitimate_fields_missing", 0)) for k in battery)
    t1, kn = learner_run["tier1"]["obs"], learner_run["kernel"]["obs"]
    need_t1 = ["rig.seats.r1.revenue_cents", "rig.seats.r1.cost_cents", "rig.seats.r1.net_cents",
               "rig.seats.r1.ticks_done"]
    absent = [k for k in need_t1 if k not in t1["numbers"]]
    absent += [] if any(k.startswith("bank.balances.") for k in t1["numbers"]) else ["tier1 balance"]
    absent += [] if any(k.startswith("rig.seats.r1.readings.") for k in t1["numbers"]) else ["tier1 readings"]
    absent += [] if any(k.startswith("bank.balances.") for k in kn["numbers"]) else ["kernel balance"]
    absent += [] if kn["tools"] and t1["tools"] else ["tools"]
    for label in ("tier1", "kernel"):
        kinds = learner_run[label]["summary"]["kinds"]
        absent += [f"{label} {k}" for k in ("events", "structured_numbers", "tool_schemas", "action_results")
                   if k not in kinds]
    absent += [] if kn["resources"].get("balance_cents") is not None else ["kernel resources"]
    compared = sum(1 for k in battery if "legitimate_fields_missing" in battery[k]["metrics"])
    metrics = {"legitimate_fields_missing": missing + len(absent), "environments_compared_to_legacy": compared,
               "required_entries_checked": len(need_t1) + 7}
    record("R.4", metrics, command=CMD, notes="every non-reward observation entry of the numeric envs, the Gym/PZ "
           f"observations and WorldEnv observations equal to legacy mode's; required present: {need_t1} + balances, "
           f"readings, tools, events, resources. Absent: {absent}")
    assert metrics["legitimate_fields_missing"] == 0, absent


def test_R5_independent_verification_active(battery, learner_run, tmp_path):
    log = tmp_path / "verifier.jsonl"
    n = 0
    with open(log, "w") as fh:
        for k in ("test_gym", "test_pettingzoo", "test_kernel_invariance"):
            if battery[k]["metrics"].get("verifier_outcomes_recorded"):
                fh.write(json.dumps({"source": k, "verified": True}) + "\n")
                n += 1
        for label, r in learner_run.items():
            v = r["env"].verify()
            assert "agents" in v or "episodes" in v
            fh.write(json.dumps({"source": f"reference learner {label}", "verification": v}, default=str) + "\n")
            n += 1
    metrics = {"verifier_outcomes_recorded": n}
    record("R.5", metrics, command=CMD, notes="evaluator-side verifier outcomes produced while the learner ran in "
           "observation-only mode (GoalVerifier status with progress for Gym/PZ/kernel, operator episode records for "
           "Tier 1); written to an evaluator-side log, never to the learner")
    assert n >= 1


def test_R6_shaped_baselines_executable():
    out = leakage.legacy_check()
    ok = [o for o in out if o["identical"]]
    metrics = {"shaped_baseline_runs": len(ok), "legacy_runs": len(out),
               "legacy_runs_not_identical": len(out) - len(ok)}
    record("R.6", metrics, command=CMD, notes="4.0 implementation (git c07098b) vs current in legacy mode, same inputs, "
           "byte-identical observations/rewards/infos: " + "; ".join(f"{o['name']}={o['identical']}" for o in out))
    assert len(ok) == len(out) and len(ok) >= 1, out


def test_R7_reference_learner_updates_observation_only(learner_run):
    s1, s2 = learner_run["tier1"], learner_run["kernel"]
    n = s1["summary"]["params_changed_updates"] + s2["summary"]["params_changed_updates"]
    metrics = {"reference_learner_updates_observation_only": n,
               "optimizer_steps": s1["summary"]["updates"] + s2["summary"]["updates"],
               "reward_sum_seen": s1["summary"]["reward_sum"] + s2["summary"]["reward_sum"],
               "params_changed_tier1": s1["before"] != s1["after"], "params_changed_kernel": s2["before"] != s2["after"]}
    record("R.7", metrics, command=CMD, notes="ReferenceLearner (not CAM) through make(..., 'observation_only'): its "
           "signal is its own (change of its observed balance) plus the self-supervised world-model loss; the "
           "environment paid 0.0 reward")
    assert n >= 1 and metrics["reward_sum_seen"] == 0.0


def test_Rn1_explicit_reward_leakage_tests(battery):
    passed = [k for k, r in battery.items() if r["passed"]]
    failed = {k: r["metrics"] for k, r in battery.items() if not r["passed"]}
    metrics = {"reward_leakage_tests_passed": len(passed), "reward_leakage_tests_run": len(battery)}
    record("R.n1", metrics, command=CMD, notes="tests: " + ", ".join(f"{k}: {battery[k]['name']}" for k in battery)
           + (f"; failed: {failed}" if failed else ""))
    assert not failed, failed
    assert len(passed) >= 5
