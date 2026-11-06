#!/usr/bin/env python
"""Run a trained policy in a REAL browser (LocalWebEnv: Playwright + Chromium) and in the
MiniWeb simulator on the same tasks, and compare verified outcomes step by step.

    python scripts/browser_transfer.py runs/v2/v2_B_s0/checkpoints/policy_final --n-per-family 2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentrs.environments.miniweb.tasks import task_set  # noqa: E402
from agentrs.evaluation.benchmark import evaluate_actor, make_env  # noqa: E402
from agentrs.models.policy import PolicyActor  # noqa: E402
from agentrs.training.common import load_model  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-per-family", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pol = load_model(a.policy)
    tasks = task_set(a.split, n_per_family=a.n_per_family)
    sim = make_env("miniweb")
    web = make_env("localweb", id_prefix="e")  # same id format the policy was trained on
    ms, es = evaluate_actor(sim, tasks, PolicyActor(pol, "greedy"))
    mw, ew = evaluate_actor(web, tasks, PolicyActor(pol, "greedy"))
    web.close()
    same_actions = sum([t.action for t in x.transitions] == [t.action for t in y.transitions] for x, y in zip(es, ew))
    same_outcome = sum(bool(x.success) == bool(y.success) for x, y in zip(es, ew))
    res = {"n": len(tasks), "sim_success": ms["success_rate"], "browser_success": mw["success_rate"],
           "identical_action_sequences": same_actions, "identical_outcomes": same_outcome,
           "browser_seconds_per_episode": mw["mean_wall_time_s"], "sim_seconds_per_episode": ms["mean_wall_time_s"]}
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))
