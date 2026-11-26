"""The existing CAM 2.0 controller as a baseline on Crucible 4.0 open-economy goals.

Arms, all on the same goal instances and evaluated greedily with no weight updates:

    cam_final     the CAM 2.0 checkpoint trained on MiniWoB (frozen Qwen3-1.7B + 65M trainable controller). It was never
                  trained on Crucible, so this measures what it brings to an unfamiliar world.
    cam_initial   the same architecture at its initial weights (the checkpoint saved before that training).
    idle          never acts.
    builder       the scripted competent control (it knows the reference solvers: a feasibility witness, not a peer).

CAM acts through the ordinary tool interface via `agentrs.cam2.envs.crucible_adapter.CrucibleAdapter`: one tool call
per step, then the world advances by `--dt` seconds. This is a Crucible 4.0 experiment: its pool selects the 4.0
reward contract explicitly (``reward_mode="legacy_shaped_reward"``, Crucible OO.8; the default is observation-only).

Moved from crucible/examples/ow_cam_baseline.py to the AGENTRS side (Crucible DECISION_LOG D-81: Crucible never imports
CAM). From the repository root:

    PYTHONPATH=src:crucible/src python scripts/cam2/crucible_ow_baseline.py --ckpt-dir runs/cam2/main_s0/ckpt --run-dir ...
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "crucible", "src"), os.path.join(REPO, "src")]


def gpu_free_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                         text=True, check=True).stdout
    return int(out.split()[0])


def controls(families, seeds, max_days):
    import crucible
    crucible.load_all_modules()
    from crucible.agents.scripted import Builder, IdlePolicy
    from crucible.evaluation.harness import run_goal
    from crucible.goals.contracts import generate_goal
    out = {}
    for name, make in (("idle", IdlePolicy), ("builder", Builder)):
        rows = []
        for fam in families:
            gf, wf, lvl = fam.split(":")
            for s in seeds:
                g = generate_goal(gf, wf, s, int(lvl), split="dev")
                r, _ = run_goal(g, make(), dt=3600, max_days=max_days)
                rows.append({"task": f"{fam}#{s}", "family": fam, "success": bool(r.success),
                             "score": float(r.progress), "actions": int(r.actions)})
        out[name] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO, "runs", "cam2", "main_s0", "ckpt"))
    ap.add_argument("--families", default="service:ow_tiny:0,revenue:ow_tiny:0")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--dt", type=int, default=43200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--min-gpu-mb", type=int, default=8000)
    ap.add_argument("--out", default=os.path.join(REPO, "crucible", "docs", "experiments", "ow_cam_baseline.json"))
    a = ap.parse_args()
    free = gpu_free_mb()
    if free < a.min_gpu_mb:
        raise SystemExit(f"only {free} MiB of GPU memory free; need {a.min_gpu_mb}")
    from agentrs.cam2.envs.base import TaskRef
    from agentrs.cam2.envs.pool import EnvPool
    from agentrs.cam2.trainer import RunConfig, Trainer
    from crucible.research.stats import proportion_ci
    families, seeds = a.families.split(","), [int(s) for s in a.seeds.split(",")]
    tasks = [TaskRef(env="crucible", family=f, seed=s) for f in families for s in seeds]
    t0 = time.time()
    pool = EnvPool("crucible", a.workers, {"max_steps": a.max_steps, "dt": a.dt,
                                           "reward_mode": "legacy_shaped_reward"})   # the 4.0 experiment, named
    out = {"what": __doc__.split("\n")[0], "families": families, "seeds": seeds, "max_steps": a.max_steps, "dt_s": a.dt,
           "simulated_days_per_episode": a.max_steps * a.dt / 86400, "gpu_free_mb_at_start": free, "arms": {}}
    try:
        rc = RunConfig(run_dir=a.run_dir, env="crucible", llm="Qwen/Qwen3-1.7B", workers=a.workers, max_steps=a.max_steps,
                       train_families=families, eval_every=0, ckpt_every=0, cam={"el_goal_cond": False}, save_replay=False,
                       final_eval=False)
        tr = Trainer(rc, pool=pool)
        out["cam_params"] = tr.cam.n_params()
        fp0 = tr.llm.fingerprint()
        for arm, ck in (("cam_initial", "initial.pt"), ("cam_final", "final.pt")):
            tr.load(os.path.join(a.ckpt_dir, ck), weights_only=True)
            r = tr.evaluate(tasks, greedy=True, label=arm)
            k = sum(int(e["success"]) for e in r["episodes"])
            out["arms"][arm] = {"checkpoint": ck, "success": proportion_ci(k, r["n"]), "mean_progress": r["score"],
                                "invalid_rate": r["invalid_rate"], "per_family": r["per_family"], "env_steps": r["env_steps"],
                                "mean_len": r["mean_len"], "errors": r["errors"], "dropped": r["dropped"],
                                "wall_s": r["eval_s"], "episodes": r["episodes"]}
            print(arm, "success", k, "/", r["n"], "progress", round(r["score"], 4), "invalid", round(r["invalid_rate"], 3), flush=True)
            with open(a.out, "w") as fh:
                json.dump(out, fh, indent=1, default=str)
        out["llm_weights_unchanged"] = tr.llm.fingerprint() == fp0
    finally:
        pool.close()
    for name, rows in controls(families, seeds, a.max_steps * a.dt / 86400).items():
        k = sum(r["success"] for r in rows)
        out["arms"][name] = {"success": proportion_ci(k, len(rows)), "mean_progress": sum(r["score"] for r in rows) / len(rows),
                             "episodes": rows}
        print(name, "success", k, "/", len(rows), flush=True)
    out["wall_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
