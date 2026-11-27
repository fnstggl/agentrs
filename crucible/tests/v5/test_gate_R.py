"""Gate R: observation-only operation (workstream L). One test per requirement; each measures the real system and
records evidence with `crucible.acceptance.v5.evidence.record`.

Two batteries run: the reward-path battery (`crucible.experience.leakage`: MechVecEnv, KernelMechEnv, SelfPlayEnv,
KernelSelfPlayEnv, Gymnasium CrucibleEnv, PettingZoo CrucibleParallelEnv, scale.rollout.WorldEnv, the experience
adapters - zero rewards, key names, canaries, invariance to evaluator-side data and to the non-legacy mode), and the
information-flow probes (`crucible.experience.probes`: exact-value, free-text, regression, counterfactual,
predicate-oracle, termination, tool-sweep, ... over the rapid and mechanism adapters, kernel worlds with v5 modules,
the innovation registry, self-play, numeric, Gymnasium/PettingZoo and AGENTRS surfaces), which try to RECOVER
evaluator values from everything a participant reaches - the defence key names and invariance cannot provide.
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
def probe_run():
    from crucible.experience import probes
    return probes.run_all(errors="record")


def _probe_leaks(run, *kinds) -> int:
    return sum(run["by_kind"][k]["leaks"] for k in kinds) + len(run["errors"])


@pytest.fixture(scope="module")
def learner_run():
    """The reference learner through the observation-only interface (Tier 1 and a kernel economy)."""
    out = {}
    for label, tier, name, kw in (("tier1", 1, "mech:scm_additive", {}), ("tier1_rapid", 1, "rapid:net_routing", {}),
                                  ("kernel", 0, "tiny", {"agents": ["learner"]})):
        env = make(tier, name, 3, "observation_only", **kw)
        L = ReferenceLearner(seed=0, reward_mode=env.reward_mode)
        before = L.params_hash()
        s = run_learner(env, L, 160 if tier == 1 else 64, update_every=32, think_s=None if tier == 1 else 3600)
        s["signal_sum"] = sum(float(u.get("signal_sum") or 0.0) for u in s["update_log"])
        obs = env.observe(env.agents()[0])
        out[label] = {"summary": s, "before": before, "after": L.params_hash(), "obs": obs, "env": env}
    return out


def test_R1_native_intermediate_reward_disabled(battery, learner_run, probe_run):
    paths = [k for k, r in battery.items() if "nonzero_native_reward_steps" in r["metrics"]]
    nonzero = sum(battery[k]["metrics"]["nonzero_native_reward_steps"] for k in paths)
    steps = sum(int(battery[k]["metrics"].get("steps", 0)) for k in paths)
    legacy_nonzero = sum(int(battery[k]["metrics"].get("legacy_nonzero_reward_steps", 0)) for k in paths)
    lr = sum(r["summary"]["nonzero_rewards"] for r in learner_run.values())
    lsteps = sum(r["summary"]["steps"] for r in learner_run.values())
    default_mode = probe_run["metrics"]["nonzero_scalar_rewards_default_mode"]
    metrics = {"nonzero_native_reward_steps": nonzero + lr + default_mode, "reward_paths_checked": len(paths),
               "steps_checked": steps + lsteps + probe_run["by_kind"]["reward_scalar"]["checked"],
               "legacy_nonzero_reward_steps_same_inputs": legacy_nonzero,
               "default_constructed_reward_paths_nonzero": default_mode}
    record("R.1", metrics, command=CMD, notes="observation_only on every reward path, and every environment of the "
           "probe surfaces constructed with its DEFAULT mode; legacy mode, selected explicitly with the same inputs, "
           "pays nonzero rewards on the counted steps, so the zero is the mode's doing. Paths: " + ", ".join(paths))
    assert metrics["nonzero_native_reward_steps"] == 0, metrics
    assert legacy_nonzero > 0


def test_R2_no_privileged_progress_values(battery, learner_run, probe_run):
    keys = ("test_gym", "test_pettingzoo", "test_tier1_invariance", "test_kernel_invariance")
    found = sum(battery[k]["metrics"]["privileged_progress_fields"] for k in keys)
    in_learner = sum(len(leakage.find_keys(r["obs"])) for r in learner_run.values())
    oterm = sum(battery[k]["metrics"].get("outcome_terminations", 0) for k in ("test_gym", "test_pettingzoo"))
    recovered = _probe_leaks(probe_run, "key_name", "exact_value", "free_text", "regression", "predicate_oracle",
                             "termination")
    metrics = {"privileged_progress_fields": found + in_learner + oterm + recovered,
               "surfaces_scanned": len(keys) + 2 + probe_run["metrics"]["surfaces_probed"],
               "probe_recoveries_of_evaluator_values": recovered,
               "probe_checks": sum(probe_run["by_kind"][k]["checked"] for k in ("key_name", "exact_value", "free_text",
                                                                                "regression", "predicate_oracle",
                                                                                "termination"))}
    record("R.2", metrics, command=CMD, notes="key names in Gymnasium/PettingZoo infos (and outcome-based "
           "termination), the Tier 1 and kernel experience envs and the reference learner's observations; and the "
           "information-flow probes that try to recover verifier progress, the mechanism's objective quantity, "
           "integrity-netted metrics (independent revenue, via goals/exp/charter/commitment predicates), outcome "
           f"termination: {probe_run['by_kind']}")
    assert metrics["privileged_progress_fields"] == 0, ([battery[k]["metrics"].get("privileged_paths") for k in keys],
                                                       [l for s_ in probe_run["surfaces"] for l in s_["leaks"]][:5])


def test_R3_no_hidden_evaluator_data(battery, probe_run):
    k, t, rig = battery["test_kernel_invariance"], battery["test_tier1_invariance"], battery["test_rig_hidden_canary"]
    planted = k["metrics"]["canaries_planted"] + 1 + rig["metrics"]["canaries_planted"] + \
        probe_run["by_kind"]["canary"]["checked"]
    hits = k["metrics"]["canary_hits"] + t["metrics"]["canary_hits"] + rig["metrics"]["canary_hits"]
    differ = k["metrics"]["observation_differences"] + t["metrics"]["observation_differences"]
    probe = _probe_leaks(probe_run, "canary", "counterfactual", "tool_sweep")
    metrics = {"hidden_evaluator_fields": hits + differ + probe, "canaries_planted": planted, "canary_hits": hits,
               "invariance_differences": differ, "probe_hidden_data_leaks": probe}
    record("R.3", metrics, command=CMD, notes="canaries: evaluator runtime state, recommender-private slot, internal "
           "event, another participant's private mail, world-spec evaluator parameter, goal evaluator parameter, "
           "Tier 1 task private reference numbers, hidden mechanism configuration, internal rig filing, other "
           "participants' private rapid messages, rapid objective thresholds; plus invariance of everything the agent "
           "receives to those data and to the non-legacy reward mode, and the probes' tool sweep (evaluator tools absent "
           "and refused) and counterfactual worlds")
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
    # the legacy side is normalised only for the reviewed evaluator-only fields (leakage.REVIEWED_EVALUATOR_FIELDS);
    # a field the registry hides without a review is compared (and counted above) and named here
    unreviewed = leakage.unreviewed_hidden_fields()
    metrics = {"legitimate_fields_missing": missing + len(absent), "environments_compared_to_legacy": compared,
               "required_entries_checked": len(need_t1) + 7, "unreviewed_hidden_fields": len(unreviewed)}
    record("R.4", metrics, command=CMD, notes="every non-reward observation entry of the numeric envs, the Gym/PZ "
           f"observations and WorldEnv observations equal to legacy mode's; required present: {need_t1} + balances, "
           f"readings, tools, events, resources. Absent: {absent}. Legacy side normalised only for the "
           f"{len(leakage.REVIEWED_EVALUATOR_FIELDS)} reviewed evaluator-only registry entries; unreviewed hidden "
           f"fields: {unreviewed}")
    assert metrics["legitimate_fields_missing"] == 0, absent
    assert not unreviewed, unreviewed


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
    s1, s2, s3 = learner_run["tier1"], learner_run["kernel"], learner_run["tier1_rapid"]
    n = sum(r["summary"]["params_changed_updates"] for r in learner_run.values())
    metrics = {"reference_learner_updates_observation_only": n,
               "optimizer_steps": sum(r["summary"]["updates"] for r in learner_run.values()),
               "reward_sum_seen": sum(r["summary"]["reward_sum"] for r in learner_run.values()),
               "params_changed_tier1": s1["before"] != s1["after"], "params_changed_kernel": s2["before"] != s2["after"],
               "params_changed_tier1_rapid": s3["before"] != s3["after"],
               "own_signal_nonzero_tier1_rapid": abs(s3["summary"]["signal_sum"]) > 0}
    record("R.7", metrics, command=CMD, notes="ReferenceLearner (not CAM) through make(..., 'observation_only'): its "
           "signal is its own (change of its observed balance) plus the self-supervised world-model loss; the "
           "environment paid 0.0 reward")
    assert n >= 1 and metrics["reward_sum_seen"] == 0.0


def test_Rn1_explicit_reward_leakage_tests(battery, probe_run):
    from crucible.experience import probes
    passed = [k for k, r in battery.items() if r["passed"]]
    failed = {k: r["metrics"] for k, r in battery.items() if not r["passed"]}
    surf_ok = [s_["name"] for s_ in probe_run["surfaces"] if s_["n_leaks"] == 0]
    surf_bad = {s_["name"]: s_["leaks"][:3] for s_ in probe_run["surfaces"] if s_["n_leaks"]}
    surf_bad.update(probe_run["errors"])
    pc = probes.positive_control()
    metrics = {"reward_leakage_tests_passed": len(passed) + len(surf_ok),
               "reward_leakage_tests_run": len(battery) + len(probes.SURFACES),
               "probe_kinds": probe_run["metrics"]["reward_leak_probe_kinds"],
               "positive_control_leak_kinds": len(pc["kinds_with_leaks"])}
    record("R.n1", metrics, command=CMD, notes="battery: " + ", ".join(f"{k}: {battery[k]['name']}" for k in battery)
           + f"; probe surfaces without leaks: {surf_ok}; positive control (the probes on the 4.0 contract) found "
             f"{pc['n_leaks']} leaks of kinds {pc['kinds_with_leaks']}"
           + (f"; failed: {failed} {surf_bad}" if failed or surf_bad else ""))
    assert not failed and not surf_bad, (failed, surf_bad)
    assert metrics["reward_leakage_tests_passed"] >= 5 and len(pc["kinds_with_leaks"]) >= 3
