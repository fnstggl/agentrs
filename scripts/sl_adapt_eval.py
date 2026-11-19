"""Paired fast-adaptation evaluation of a saved checkpoint (frozen weights), offline; changes no run.

The in-run milestone evaluations store per-episode aggregates over 48 tasks, so their memory DiD (gain under "full" minus
gain under "episode") has a ±0.24 interval. This tool evaluates many more validation tasks and pairs the modes by task:
  * gain(mode, t) = outcome(episode K) - outcome(episode 1) on task t;
  * DiD = mean_t [gain(full, t) - gain(episode, t)]  (memory carried across episodes);
  * K-diff = mean_t [outcome_K(full, t) - outcome_K(episode, t)]  (episode 1 is the same policy in both modes);
with 90% intervals from the per-task paired differences.

Run it with the code snapshot the run was trained with, e.g.
    PYTHONPATH=/workspace/agentrs-sl-exp/src python scripts/sl_adapt_eval.py --ckpt runs/sl/P_N_s1/ckpt/life_500000 \
        --run runs/sl/P_N_s1 --family cam/negotiation --tasks 400 --out result.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from agentrs.sl import worlds as W
from agentrs.sl.model import SLConfig, SLCore
from agentrs.sl.runner import Runner
from agentrs.sl.world import RewardContract, RewardSpec


def paired(a: np.ndarray) -> dict:
    n = len(a)
    m = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {"mean": m, "ci90": 1.645 * se, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--tasks", type=int, default=400)
    ap.add_argument("--split", default="val")
    ap.add_argument("--modes", default="full,episode,none")
    ap.add_argument("--n-worlds", type=int, default=128)
    ap.add_argument("--metric", default="auto", help="success | score | progress | auto (success if any, else score)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.05)
    a = ap.parse_args()
    st = json.loads((Path(a.run) / "state.json").read_text())
    c = st["config"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    m = SLCore(SLConfig(**c["model"])).to(dev)
    m.load_state_dict(torch.load(Path(a.ckpt) / "weights.pt", map_location=dev))
    m.eval()
    rc = RewardContract(RewardSpec(**c["reward"]))
    K = int(c["trial_episodes"])
    W.resolve(a.family)
    tasks = W.eval_tasks(a.family, a.split, a.tasks)
    res, t0 = {}, time.time()
    for mode in a.modes.split(","):
        r = Runner(m, {a.family: 1.0}, W.make_world, W.sample_task, min(len(tasks), a.n_worlds), trial_episodes=K,
                   split=a.split, seed=12345, mode=mode, contract=rc, device=dev, max_episode_steps=c["max_episode_steps"],
                   fixed_tasks=list(tasks), dt_ref=dict(st.get("dt_ref", {})))
        try:
            with torch.no_grad():
                eps = r.run_trials()
        finally:
            r.close()
        by = {}
        for e in eps:
            by.setdefault(e["task"].seed, {})[e["episode"]] = e
        res[mode] = by
    seeds = sorted(set.intersection(*(set(v) for v in res.values())))
    seeds = [s for s in seeds if all(0 in res[md][s] and K - 1 in res[md][s] for md in res)]
    metric = a.metric
    if metric == "auto":
        metric = "success" if any(res[md][s][k]["success"] for md in res for s in seeds for k in res[md][s]) else "score"

    def val(md, s, k):
        return float(res[md][s][k][metric])

    out = {"ckpt": a.ckpt, "run": a.run, "family": a.family, "split": a.split, "tasks": len(seeds), "K": K, "metric": metric,
           "wall_s": round(time.time() - t0, 1), "modes": {}}
    for md in res:
        first = np.array([val(md, s, 0) for s in seeds])
        last = np.array([val(md, s, K - 1) for s in seeds])
        out["modes"][md] = {"first": float(first.mean()), "last": float(last.mean()), "gain": paired(last - first)}
    if "full" in res and "episode" in res:
        g_f = np.array([val("full", s, K - 1) - val("full", s, 0) for s in seeds])
        g_e = np.array([val("episode", s, K - 1) - val("episode", s, 0) for s in seeds])
        out["did_full_vs_episode"] = paired(g_f - g_e)
        out["lastK_full_minus_episode"] = paired(np.array([val("full", s, K - 1) - val("episode", s, K - 1) for s in seeds]))
    if "full" in res and "none" in res:
        out["lastK_full_minus_none"] = paired(np.array([val("full", s, K - 1) - val("none", s, K - 1) for s in seeds]))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
