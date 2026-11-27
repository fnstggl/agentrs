"""Repeated staging tasks in the general-purpose computer environment: does the integrated agent get better at them
through its own learning? (spec 12A.17: repeated tasks, primitive-only vs acquired skills; 12A.10: neural learning
independently of the skill library.)

One process = one arm x one seed: a fresh agent (debug scale on the CPU by default, ``--real`` for Qwen3-4B + 50m
under the GPU lock) receives ``--episodes`` objectives of the same task family one after another, each a new
objective ("increase the count to current + 1") with its own decision budget, in ONE persistent computer environment
(staging sites, Chromium, authorization service, signing verifier reading the sites' own state). The browser is put
on the application's start page before each objective (experimental control).

Arms
  learn_skills      neural learning on, skill executor present (skill.run / skill.induce)
  learn_primitive   neural learning on, no skill executor (primitive-only operation)
  frozen_skills     no weight / belief / reward update at all, skill executor present
  frozen_primitive  neither

Usage (from /workspace/camlive):
  .venv/bin/python cam_live/docs/evidence/validation/learning/run_arm.py --arm learn_skills --seed 0 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

ARMS = {"learn_skills": (True, True), "learn_primitive": (True, False), "frozen_skills": (False, True),
        "frozen_primitive": (False, False)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-decisions", type=int, default=40)
    ap.add_argument("--key", default="store_entry")
    ap.add_argument("--transfer-key", default="", help="after the episodes, objectives of this other family")
    ap.add_argument("--transfer-episodes", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--agent", default="{}", help="JSON overrides of the agent settings")
    a = ap.parse_args(argv)
    import torch
    from cam_live.agent import CamLiveAgent
    from cam_live.validation import OffsetClock, StagingComputer, run_episode, staging_config
    learning, skills = ARMS[a.arm]
    cfg = staging_config(None, a.real, a.seed)
    over = {"learning": learning, "skills": skills, "proposer_per_tool": 2}
    if not a.real:
        # the tiny random backbone writes no valid action (measured: 0 of every proposal) and the planner's imagined
        # values are used by no selection rule: both off at debug scale to keep a decision at ~2-3 s on the CPU
        over.update(use_backbone=False, planner=False)
    over.update(json.loads(a.agent))
    cfg.sections["agent"].update(over)
    work = os.path.abspath(a.work or os.path.splitext(a.out)[0] + "_work")
    os.makedirs(work, exist_ok=True)
    report = {"arm": a.arm, "seed": a.seed, "learning": learning, "skills": skills, "key": a.key,
              "episodes_planned": a.episodes, "max_decisions": a.max_decisions, "real_model": bool(a.real),
              "agent_settings": cfg.section("agent"), "episodes": [], "started": time.time(), "errors": []}
    lock = None
    if a.real:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=18.0)
        lock.__enter__()
    clock = OffsetClock()
    sc = StagingComputer(os.path.join(work, "computer"), clock, exclude=() if skills else ("skill",))
    agent = CamLiveAgent(cfg, os.path.join(work, "agent"), clock=clock)
    try:
        plan = [a.key] * a.episodes + ([a.transfer_key] * a.transfer_episodes if a.transfer_key else [])
        for i, key in enumerate(plan):
            try:
                ep = run_episode(agent, sc, key, clock, os.path.join(work, "runs"), max_decisions=a.max_decisions)
            except Exception as e:  # noqa: BLE001 — recorded, the arm continues
                report["errors"].append({"episode": i, "error": f"{type(e).__name__}: {e}",
                                         "trace": traceback.format_exc(limit=8)})
                continue
            ep["index"] = i
            ep["agent_counters"] = dict(agent.counters)
            ep["model_version"] = agent.learner.model_version
            ep["reward_version"] = agent.meta.reward.version
            report["episodes"].append(ep)
            with open(a.out, "w") as f:
                json.dump(report, f, indent=1, default=str)
            print(f"[{a.arm} s{a.seed}] ep {i} {key}: {ep['status']} achieved={ep['achieved']} "
                  f"decisions={ep['decisions']} ops={ep['primitive_ops']} skill_runs={ep['skill_runs']} "
                  f"{ep['seconds']:.0f}s", flush=True)
        report["status"] = agent.status()
        report["finished"] = time.time()
    finally:
        sc.close()
        agent.close()
        if lock is not None:
            lock.__exit__(None, None, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
