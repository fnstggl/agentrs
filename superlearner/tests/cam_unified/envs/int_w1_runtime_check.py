"""W2 x W1 integration check (needs W1's build.py / runtime.py, i.e. a tree where the W2 branch is merged into W1's
main; not a pytest module). Run from <tree>/superlearner under the GPU lock (~7 GB):
    PYTHONPATH=src:../crucible/src python tests/cam_unified/envs/int_w1_runtime_check.py --out r.json --run-dir run

build_agent('dom_dev') + W2 factory.make_env (ow_tiny process-isolated with canaries, + 2 rapid lever worlds)
through W1's runtime.run_lifetime observation_only, 12 decisions, update_every 6. Measures: Transition.rewards is
None every round, every action's direct result in the next input with caused_by (and how many copies), labels
through take_labels -> LabelStore, core hash change, canaries absent from actor inputs and the run directory."""
import argparse
import json
import os
import sys
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--update-every", type=int, default=6)
    a = ap.parse_args()

    import torch

    HERE = os.getcwd()
    sys.path.insert(0, os.path.join(HERE, "tests/cam_unified/envs"))
    from test_env_real_e2e import _task  # noqa: E402

    from superlearner.cam_unified.build import ConcatLabels, SystemConfig, build_agent  # noqa: E402
    from superlearner.cam_unified.core.bench import rapid_lever_specs  # noqa: E402
    from superlearner.cam_unified.envs import integrity as I  # noqa: E402
    from superlearner.cam_unified.envs.factory import make_env  # noqa: E402
    from superlearner.cam_unified.envs.ow_convert import is_delivered_result, result_key  # noqa: E402
    from superlearner.cam_unified.interfaces import ObsKind  # noqa: E402
    from superlearner.cam_unified.runtime import ConcatEnv, GuardedEnv, run_lifetime  # noqa: E402
    from superlearner.cam_unified.stage1.harness_agent import core_hash  # noqa: E402

    t0 = time.time()
    os.makedirs(a.run_dir, exist_ok=True)
    canaries = {k: I.new_canary(k) for k in ("task", "task_attr", "state", "status", "label")}
    sc = SystemConfig(deterministic=False, archive_autoflush=False, fresh_max_steps=a.update_every + 1,
                      checkpoint_semantic_cache=False)
    sysm = build_agent("dom_dev", env_kind="mixed", device="cuda", run_dir=a.run_dir,
                       overrides={"grad_checkpoint": True}, system=sc, seed=0)
    build_s = time.time() - t0
    ow = make_env("openworld", tasks=[_task(canaries, a.n)], isolated=True, seed=0, max_wait_s=600,
                  evaluator_kw={"objective_factory": I.canary_objective_factory(
                      {"state": canaries["state"], "status": canaries["status"], "label": canaries["label"]})},
                  store_url=f"sqlite:///{os.path.join(os.path.dirname(a.run_dir), 'ow_world.db')}")
    specs = rapid_lever_specs(2, seed0=2000)
    rp = make_env("rapid", specs=specs)
    fam_canaries = sorted({s.name.split(":", 1)[1] for s in specs} | {f"{s.name}#{s.seed}" for s in specs})
    env = GuardedEnv(ConcatEnv([ow.adapter, rp.adapter]))
    seen = []
    h0 = core_hash(sysm.model)
    t1 = time.time()
    life = run_lifetime(sysm.agent, env, labels=ConcatLabels([ow.labels, rp.labels]), label_store=sysm.label_store,
                        learner=sysm.learner, max_decisions=a.n, update_every=a.update_every, require_cam_eval=False,
                        transcript=os.path.join(a.run_dir, "transcript.jsonl"), on_step=lambda s, tr: seen.append(tr))
    loop_s = time.time() - t1
    h1 = core_hash(sysm.model)

    delivery = []
    for k, tr in enumerate(seen):
        for b, rec in enumerate(tr.records):
            act = rec.action
            if act is None:
                continue
            nxt = tr.next_obs.observations[b]
            res = [o for o in nxt if o.caused_by == act.action_id and o.kind in (ObsKind.TOOL_RESULT, ObsKind.ERROR)]
            keys = [result_key(o) for o in res]
            delivery.append({"k": k, "b": b, "tool": act.tool, "ok": None if rec.result is None else rec.result.ok,
                             "in_next_input": bool(res), "copies": len(res), "distinct": len(set(keys)),
                             "delivered_copies": sum(is_delivered_result(o) for o in res),
                             "cost": None if rec.result is None else rec.result.cost})
    inputs = [tr.next_obs for tr in seen] + [[r.observations for r in tr.records] for tr in seen]
    all_canaries = list(canaries.values()) + fam_canaries
    hits_inputs = I.canary_hits(inputs, all_canaries)
    scan = I.scan_run_dir(a.run_dir, all_canaries)
    priv = I.privileged_keys(inputs)
    labs = list(life.labels_taken) if isinstance(life.labels_taken, list) else life.labels_taken
    out = {
        "what": "W2 x W1 integration: build_agent(dom_dev) + envs.factory.make_env (ow_tiny isolated + 2 rapid) through "
                "runtime.run_lifetime, observation_only",
        "B": env.B, "rounds": life.steps, "reward_mode": life.reward_mode, "build_s": round(build_s, 2),
        "loop_s": round(loop_s, 2), "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "rewards_none_every_round": all(tr.rewards is None for tr in seen),
        "record_rewards_zero": all(r.reward == 0.0 for tr in seen for r in tr.records),
        "every_result_in_next_input": all(d["in_next_input"] for d in delivery),
        "max_copies_per_result": max(d["copies"] for d in delivery),
        "results_with_duplicate_copies": sum(d["copies"] > d["distinct"] or d["copies"] > 1 for d in delivery),
        "n_results": len(delivery), "delivery": delivery,
        "labels_taken": labs, "labels_appended": life.labels_appended,
        "core_hash_before": h0, "core_hash_after": h1, "core_hash_changed": h0 != h1,
        "canary_hits_inputs": hits_inputs[:20], "canary_scan_run_dir": scan, "privileged_keys_in_inputs": priv[:20],
        "env_exceptions": [str(e)[:200] for e in getattr(env, "errors", [])],
        "engine_errors_rapid": rp.adapter.engine_errors,
    }
    ow.close()
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ("delivery",)}, default=str)[:3000])


if __name__ == "__main__":
    main()
