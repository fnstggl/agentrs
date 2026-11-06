#!/usr/bin/env python
"""Evaluation-only planner ablations on a trained model-based run (no retraining):
planning on/off, horizon, uncertainty penalty, number of candidates, branches.

    python scripts/planner_sweep.py runs/dev/dev_D_s0 --sets val,test,test_family \\
        --grid horizon=1,2 uncertainty_penalty=0,1,3 num_candidates=4
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from agentrs.environments.miniweb.tasks import task_set  # noqa: E402
from agentrs.evaluation.benchmark import evaluate_actor, make_env  # noqa: E402
from agentrs.models.policy import PolicyActor  # noqa: E402
from agentrs.planning.mpc import MPCPlanner, PlannerConfig  # noqa: E402
from agentrs.training.common import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sets", default="val,test,test_family")
    ap.add_argument("--grid", nargs="*", default=["horizon=1,2", "uncertainty_penalty=0,1,3"])
    ap.add_argument("--n-per-family", type=int, default=0, help="0 = use the run's eval set sizes")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    run = Path(a.run)
    res_cfg = json.loads((run / "results.json").read_text())["config"]
    ck = run / "checkpoints"
    pol, wm, cr = load_model(ck / "policy_final"), load_model(ck / "world_model_final"), load_model(ck / "critic_final")
    for m in (pol, wm, cr):
        m.eval()
    env = make_env("miniweb")
    sizes = res_cfg["eval_sets"]
    sets = {}
    for s in a.sets.split(","):
        fams = res_cfg["heldout_families"] if s == "test_family" else res_cfg["train_families"]
        sets[s] = task_set(s, fams, n_per_family=a.n_per_family or sizes[s])
    keys, values = [], []
    for g in a.grid:
        k, v = g.split("=")
        keys.append(k)
        values.append([type(getattr(PlannerConfig(), k))(x) for x in v.split(",")])
    base = PlannerConfig.model_validate(res_cfg["planner"])
    out = {"run": str(run), "results": []}
    configs = [("policy_only", None)] + [(dict(zip(keys, combo)), combo) for combo in itertools.product(*values)]
    for name, combo in configs:
        rec = {"planner": name}
        for s, tasks in sets.items():
            t0 = time.time()
            if combo is None:
                actor = PolicyActor(pol, "greedy")
            else:
                actor = MPCPlanner(pol, wm, cr, base.model_copy(update=name), seed=0)
            f0 = sum(m.backbone.flops_counter for m in (pol, wm, cr))
            m, _ = evaluate_actor(env, tasks, actor)
            rec[s] = {"success_rate": m["success_rate"], "success_ci95": m["success_ci95"],
                      "invalid_action_rate": m["invalid_action_rate"], "mean_steps": m["mean_steps"],
                      "seconds_per_episode": (time.time() - t0) / len(tasks),
                      "inference_flops_per_episode": (sum(x.backbone.flops_counter for x in (pol, wm, cr)) - f0)
                      / len(tasks), "by_family": m["by_family"]}
            print(json.dumps({"planner": name, "set": s, **{k: rec[s][k] for k in ("success_rate", "mean_steps",
                                                                                    "seconds_per_episode")}}),
                  flush=True)
        out["results"].append(rec)
        Path(a.out or (run / "planner_sweep.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
