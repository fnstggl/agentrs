"""Gate S: architecture-independent learner integration (replaces "Actual CAM integration", Clarification 1).

The learner is `crucible.experience.reference.ReferenceLearner` (a small recurrent network, not CAM), connected only
through the public `crucible.experience` interface. Tier 2 and Tier 3 are the v5 domains (`crucible.domains`,
workstream I) and civilisations (`crucible.civ`, workstream J), reached through the same `make(tier, ...)`; until
those packages land, S.2 and S.3 record zero actions and fail - they do not substitute another world type.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience import ExperienceStore, validate_observation
from crucible.experience.loop import run_learner
from crucible.experience.parallel import FarmRun, VecExperience
from crucible.experience.reference import ReferenceLearner
from crucible.experience.registry import TierUnavailable, civ_available, domains, make

pytestmark = pytest.mark.timeout(1800)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_S.py"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _learner(env, seed=0):
    return ReferenceLearner(seed=seed, reward_mode=env.reward_mode)


def test_S1_reference_learner_acts_in_tier1():
    env = make(1, "mech:scm_additive", 11, "observation_only")
    L = _learner(env)
    s = run_learner(env, L, 220, update_every=40)
    metrics = {"tier1_actions": s["actions"], "tier1_world_changes": s["world_changes"], "tier1_ok": s["ok"],
               "tier1_rejected": s["rejected"], "tier1_updates": s["updates"]}
    record("S.1", metrics, command=CMD, notes="Tier 1 = 4.0 hidden-mechanism facility scm_additive:11 through "
           "MechExperienceEnv (the kernel facility's own rig.* tools); world change = an executed action changed the "
           "settings the facility runs at")
    assert metrics["tier1_actions"] >= 100 and metrics["tier1_world_changes"] >= 1, metrics


def _tier_run(tier: int, names: list[str], steps: int) -> dict:
    acts = changes = 0
    worlds = []
    for name in names:
        env = make(tier, name, 0, "observation_only", agents=["learner"])
        L = _learner(env)
        s = run_learner(env, L, steps, update_every=32, think_s=3600)
        acts += s["actions"]
        changes += s["world_changes"]
        worlds.append(name)
    return {"actions": acts, "changes": changes, "worlds": worlds}


def test_S2_reference_learner_acts_in_tier2():
    doms = domains()
    if not doms:
        record("S.2", {"tier2_actions": 0, "tier2_world_changes": 0, "tier2_domains_present": 0}, command=CMD,
               notes="no v5 Tier 2 domain in this checkout (crucible.domains has no DOMAIN; workstream I). The adapter "
                     "path make(2, '<domain>[:<config>]', ...) -> KernelExperienceEnv is implemented and is exercised "
                     "as soon as a domain lands; no other world type is counted as Tier 2.")
        with pytest.raises(TierUnavailable):
            make(2, "startup_economy", 0, "observation_only")
        pytest.fail("Tier 2 domains (workstream I) are not present: tier2_actions = 0")
    r = _tier_run(2, sorted(doms), 120)
    record("S.2", {"tier2_actions": r["actions"], "tier2_world_changes": r["changes"],
                   "tier2_domains_present": len(doms)}, command=CMD, notes=f"domains: {r['worlds']}")
    assert r["actions"] >= 100 and r["changes"] >= 1, r


def test_S3_reference_learner_acts_in_tier3():
    if not civ_available():
        record("S.3", {"tier3_actions": 0, "tier3_world_changes": 0, "tier3_generator_present": False}, command=CMD,
               notes="no v5 civilisation generator in this checkout (crucible.civ.generate; workstream J). The adapter "
                     "path make(3, '<profile>', ...) -> KernelExperienceEnv is implemented; no other world type is "
                     "counted as Tier 3.")
        with pytest.raises(TierUnavailable):
            make(3, "default", 0, "observation_only")
        pytest.fail("Tier 3 civilisations (workstream J) are not present: tier3_actions = 0")
    import crucible.civ as civ
    profiles = sorted(getattr(civ, "PROFILES", {"default": None}))[:2]
    r = _tier_run(3, profiles, 120)
    record("S.3", {"tier3_actions": r["actions"], "tier3_world_changes": r["changes"],
                   "tier3_generator_present": True}, command=CMD, notes=f"profiles: {r['worlds']}")
    assert r["actions"] >= 100 and r["changes"] >= 1, r


def test_S4_experience_storage_and_updates(tmp_path):
    root = tmp_path / "exp"
    store = ExperienceStore(str(root), shard_records=250)
    n = 0
    for name, seed in (("mech:scm_additive", 21), ("mech:abm_utility", 22), ("mech:scm_delay", 23)):
        env = make(1, name, seed, "observation_only", validate=False)
        L = _learner(env, seed)
        s = run_learner(env, L, 360, store=store, update_every=60)
        n += s["records"]
    env = make(0, "tiny", 4, "observation_only", agents=["learner"])
    s = run_learner(env, _learner(env), 60, store=store, update_every=30, think_s=3600)
    n += s["records"]
    store.close()
    reopened = ExperienceStore(str(root))
    integrity = reopened.verify()
    stored = sum(1 for _ in reopened.iterate())
    fresh = ReferenceLearner(seed=99, reward_mode="observation_only")
    before = fresh.params_hash()
    u = fresh.learn_from_records(reopened.iterate(env={"tier": 1}), batch=128)
    metrics = {"experience_records": stored, "records_appended": n, "learner_parameter_updates": u["updates"],
               "params_changed": fresh.params_hash() != before, "sealed_shards": integrity["shards"],
               "store_bytes": reopened.stats()["bytes"]}
    record("S.4", metrics, command=CMD, notes="records appended by run_learner across 3 Tier 1 worlds and a kernel "
           "economy, store closed and reopened from disk, every line/shard/manifest hash verified, a fresh learner "
           "trained on the stored Tier 1 experience")
    assert stored == n >= 1000 and u["updates"] >= 1 and metrics["params_changed"], metrics


def test_S5_one_interface_many_world_types():
    """One learner instance (the same weights) across world types; every observation validated."""
    from crucible.research.tasks import make_task
    from crucible.worldgen import generate
    from crucible.experience.kernel import KernelExperienceEnv
    task = make_task("scm_additive", 5, validate=False)
    cfg, seed = task.world_config()
    worlds = [("tier1 mechanism facility (MechVecEnv)", 1, lambda: make(1, "mech:scm_additive", 5, "observation_only",
                                                                       task=task)),
              ("tier1 two-seat facility (SelfPlayEnv)", 1, lambda: make(1, "mech:abm_duopoly", 2, "observation_only",
                                                                       validate=False)),
              ("kernel-hosted facility (rig world)", 0, lambda: KernelExperienceEnv(
                  generate(cfg, seed), agents=["agent"], endowment=task.endowment, reward_mode="observation_only")),
              ("4.0 economy (tiny)", 0, lambda: make(0, "tiny", 1, "observation_only", agents=["learner"])),
              ("4.0 open digital economy", 0, lambda: make(0, "open_digital", 1, "observation_only",
                                                          agents=["learner"]))]
    for dn in sorted(domains())[:3]:
        worlds.append((f"v5 domain {dn}", 2, lambda dn=dn: make(2, dn, 0, "observation_only", agents=["learner"])))
    if civ_available():
        worlds.append(("v5 civilisation", 3, lambda: make(3, "default", 0, "observation_only", agents=["learner"])))
    L = ReferenceLearner(seed=0, reward_mode="observation_only")
    failures, ran, tiers = 0, [], set()
    action_shapes = set()
    for label, tier, mk in worlds:
        env = mk()
        s = run_learner(env, L, 40, update_every=20, think_s=None if tier == 1 else 3600)
        failures += s["schema_failures"]
        if s["actions"] > 0:
            ran.append(label)
            tiers.add(tier)
        obs = env.observe(env.agents()[0])
        failures += len(validate_observation(obs))
        action_shapes.add(tuple(sorted({"tool", "args"})))
    v5_tiers = sorted(t for t in tiers if t in (1, 2, 3))
    metrics = {"schema_validation_failures": failures, "world_types_one_interface": len(ran),
               "tiers_one_interface": len(v5_tiers), "action_shapes": len(action_shapes)}
    record("S.5", metrics, command=CMD, notes=f"world types run by one ReferenceLearner instance: {ran}; v5 tiers "
           f"covered: {v5_tiers} (Tier 2/3 appear only once crucible.domains / crucible.civ land)")
    assert failures == 0 and len(ran) >= 3, metrics
    assert v5_tiers == [1, 2, 3], f"the same interface across Tier 1, 2 and 3 needs the v5 domains and civilisations: {metrics}"


def test_S6_checkpoint_persistence(tmp_path):
    env = make(1, "mech:scm_gate", 7, "observation_only", validate=False)
    L = _learner(env, 3)
    run_learner(env, L, 120, update_every=30)
    path = str(tmp_path / "ref.pt")
    L.save(path)
    L2 = ReferenceLearner(seed=12345, reward_mode="observation_only")
    L2.load(path)
    mism = 0
    a, b = L.net.state_dict(), L2.net.state_dict()
    mism += sum(int(not (a[k] == b[k]).all()) for k in a) + int(set(a) != set(b))
    oa, ob = L.opt.state_dict(), L2.opt.state_dict()
    for k in oa["state"]:
        for f in oa["state"][k]:
            x, y = oa["state"][k][f], ob["state"][k][f]
            mism += int(not bool((x == y).all())) if hasattr(x, "shape") else int(x != y)
    mism += int(L.params_hash() != L2.params_hash()) + int(L.memory != L2.memory) + int(L.updates != L2.updates)
    # the same future: same sampled actions on the same observations, same update result
    env2 = make(1, "mech:scm_gate", 7, "observation_only", validate=False)
    obs = [env2.observe(env2.agents()[0])]
    for _ in range(10):
        env2.advance(env2.tick_s)
        obs.append(env2.observe(env2.agents()[0]))
    L.reset()
    L2.reset()
    for o in obs:
        mism += int(json.dumps(L.act(o, explore=True), sort_keys=True) != json.dumps(L2.act(o, explore=True),
                                                                                       sort_keys=True))
        L.observe(0.0, None, o)
        L2.observe(0.0, None, o)
    ua, ub = L.update(), L2.update()
    mism += int(ua["params_hash_after"] != ub["params_hash_after"])
    metrics = {"checkpoint_roundtrip_mismatches": mism, "checkpoint_bytes": os.path.getsize(path),
               "params": L.params()}
    record("S.6", metrics, command=CMD, notes="save -> load into a learner built with another seed: weights, optimiser "
           "state, RNG states, parameter-spec memory, counters; then identical sampled actions on 11 observations and "
           "an identical update")
    assert mism == 0, metrics


def test_S7_parallel_environments(tmp_path):
    # vectorised: one learner drives 8 live environments step by step
    envs = [make(1, f"mech:{f}", s, "observation_only", validate=False)
            for f, s in [("scm_additive", 1), ("scm_delay", 2), ("abm_utility", 3), ("scm_gate", 4)] * 2]
    vec = VecExperience(envs)
    L = ReferenceLearner(seed=0, reward_mode="observation_only")
    obs = vec.observe()
    steps = 0
    for _ in range(12):
        acts = [L.act(o) for o in obs]
        _, _, obs = vec.step(acts, [e.tick_s for e in envs])
        steps += vec.n
    # process-parallel: the crucible.scale RolloutFarm, 4 worker processes, 4 worlds at a time
    store = ExperienceStore(str(tmp_path / "farm"))
    worlds = [{"tier": 1, "name": f"mech:{f}", "seed": s, "reward_mode": "observation_only", "kw": {"validate": False}}
              for f, s in [("scm_additive", 1), ("scm_delay", 2), ("abm_utility", 3), ("scm_gate", 4)]]
    fr = FarmRun(L, worlds, n_workers=4, steps_per_task=60, store=store)
    try:
        fr.wait_ready()
        out = fr.run(rounds=2)
    finally:
        fr.close()
    store.close()
    metrics = {"parallel_environments": int(out["max_concurrent_environments"]),
               "vectorised_environments": vec.n, "vectorised_steps": steps,
               "farm_worker_processes": len(out["worker_pids"]), "farm_steps": out["steps"],
               "farm_failures": len(out["failures"]), "farm_learner_updates": out["updates"],
               "farm_param_versions": out["params_versions"]}
    record("S.7", metrics, command=CMD, notes="parallel_environments = most environments whose worker intervals "
           "overlapped in the RolloutFarm run (separate processes); also a vectorised run of 8 environments in one "
           "process; experience from the farm stored, the learner updated and new parameters published per round")
    assert metrics["parallel_environments"] >= 4 and metrics["farm_failures"] == 0, metrics


def test_S8_reproducible_commands(tmp_path):
    outs = []
    for i in range(2):
        d = tmp_path / f"run{i}"
        cmd = [sys.executable, "-m", "crucible.experience", "run", "--tier", "1", "--name", "mech:scm_additive",
               "--seed", "0", "--steps", "120", "--reward-mode", "observation_only", "--store", str(d / "store"),
               "--checkpoint", str(d / "ref.pt"), "--out", str(d / "out.json")]
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), OMP_NUM_THREADS="1")
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, timeout=600)
        outs.append(json.loads((d / "out.json").read_text()))
    keys = ("store_digest", "params_hash_after", "env_digest", "actions", "world_changes", "updates")
    same = all(outs[0][k] == outs[1][k] for k in keys)
    metrics = {"reproducible_commands_verified": bool(same), "digests_compared": len(keys),
               "store_digest": outs[0]["store_digest"], "params_hash": outs[0]["params_hash_after"]}
    record("S.8", metrics, command=" ".join(["python -m crucible.experience run --tier 1 --name mech:scm_additive",
                                             "--seed 0 --steps 120 --reward-mode observation_only --store ... "]),
           notes="the same command run twice in fresh processes: identical experience-store digest, learner parameter "
                 "hash, final world digest and counts")
    assert same, [(k, outs[0][k], outs[1][k]) for k in keys]
