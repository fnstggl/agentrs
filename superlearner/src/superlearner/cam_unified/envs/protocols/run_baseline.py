"""Command line: run a frozen semantic baseline (Part 19 #5-#7) on a protocol task through the OpenWorld adapter
and append the record.

    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock python -m \
        superlearner.cam_unified.envs.protocols.run_baseline open_economy --split dev --model Qwen/Qwen3-1.7B --days 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> int:
    from . import load
    from ..baselines import FrozenSemanticAgent, make_backend
    from ..harness import run_lifetime
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--variant", default="tools", choices=["tools", "memory", "planning"])
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--think-s", type=float, default=600.0)
    ap.add_argument("--record", default=os.path.join(HERE, "records", "baselines.jsonl"))
    a = ap.parse_args(argv)
    p = load(a.protocol)
    fam, seed = p.splits[a.split][a.index]
    t0 = time.time()
    be = make_backend(a.model, "cuda")
    agent = FrozenSemanticAgent(a.variant, backend=be, max_new_tokens=a.max_new_tokens)
    # the baseline is a learner-side agent: worlds, operator and evaluator run in another process (only records
    # cross), with the seat and participants the task needs
    ad, rop, ev = p.open(a.split, a.index, isolated=True, evaluator={"integrity": True}, seed=0)
    try:
        res = run_lifetime(agent, ad, ev, days=a.days, think_s=a.think_s)
    finally:
        ad.close()
    pol = agent.policies[0]
    rec = {"protocol": a.protocol, "world_family": fam, "seed": seed, "split": a.split, "model": a.model,
           "variant": a.variant, "days": a.days, "decisions": res.steps, "achieved": res.final[0]["achieved"],
           "score": res.final[0]["score"], "metrics": res.final[0]["metrics"], "return": res.returns[0],
           "elapsed_days": res.elapsed_days[0], "by_tool": res.accounting["by_tool"],
           "errors": res.accounting["errors"], "usage": res.accounting.get("agent"),
           "wall_s": round(time.time() - t0, 1), "gpu_max_mem_mb": res.accounting.get("gpu_max_mem_mb"),
           "parse_failures": pol.stats.get("parse_failures"), "waits": pol.stats.get("waits"),
           "t_wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(os.path.dirname(a.record), exist_ok=True)
    with open(a.record, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("model", "decisions", "achieved", "metrics", "by_tool", "errors",
                                          "parse_failures", "wall_s")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
