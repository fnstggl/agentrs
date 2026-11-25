"""Seed sweep of the probe-bandit meta-RL agent (workstream D evidence for A09 / D01 / D02).

Trains `learn_doubles.ProbeAgent` with the production machinery (`train_probe_agent`) for every (condition, seed)
and evaluates it on held-out seeds (`learn_doubles.evaluate`, deterministic policy). One JSON row per run is appended
to ``--out``. Conditions:

* ``lp``        — learning-progress reward (lp_coef 200 cents / nat), stage explore_meta;
* ``extrinsic`` — verified return only (plain actor-critic, stage actor_critic);
* ``lp_anneal`` — LP decayed linearly to 0 over the first half of training (LP as an initial signal only).

Usage (from the repository root, CPU, <= 4 worker processes on the shared machine):

    cd superlearner/tests/cam_unified/learn && PYTHONPATH=../../../src:. OMP_NUM_THREADS=2 \\
        python sweeps/probe_sweep.py --conditions lp extrinsic --seeds 0 1 2 3 4 5 6 7 --workers 4 \\
        --out sweeps/probe_sweep_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # learn_doubles

CONDITIONS = {"lp": dict(lp_coef=200.0), "extrinsic": dict(lp_coef=0.0),
              "lp_anneal": dict(lp_coef=200.0, lp_anneal=0.5)}


def run_one(condition: str, seed: int, iters: int) -> dict:
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    from learn_doubles import ProbeAgent, competence_curve, evaluate, train_probe_agent
    from superlearner.cam_unified.learn.diagnostics import ProbeBanditAdapter
    bayes = {h: ProbeBanditAdapter(1).bayes_optimal_value(h) for h in range(1, 7)}
    agent = ProbeAgent(6, D=64, seed=seed)
    t0 = time.time()
    out = train_probe_agent(agent, iters=iters, B=128, horizon=(1, 6), seed0=1_000 * seed,
                            **CONDITIONS[condition])
    ev = evaluate(agent)
    comp = competence_curve(agent)
    return {"condition": condition, "seed": seed, "iters": iters, "train_s": out["seconds"],
            "wall_s": time.time() - t0, "eval": {str(h): v for h, v in ev.items()},
            "competence_after_k": {str(k): v for k, v in comp.items()},
            "bayes": {str(h): v for h, v in bayes.items()},
            "bayes_matched_H2_6": all(ev[h]["return"] >= bayes[h] - 15.0 for h in range(2, 7)),
            "probe_first_H2_6": sum(ev[h]["probe_first"] for h in range(2, 7)) / 5,
            "mean_return_H2_6": sum(ev[h]["return"] for h in range(2, 7)) / 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["lp", "extrinsic"], choices=sorted(CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "probe_sweep_results.jsonl"))
    a = ap.parse_args()
    if a.workers > 4:
        raise SystemExit("at most 4 worker processes on the shared machine")
    jobs = [(c, s) for c in a.conditions for s in a.seeds]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, c, s, a.iters) for c, s in jobs]
        for f in futs:
            row = f.result()
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(row["condition"], row["seed"], round(row["mean_return_H2_6"], 1), row["bayes_matched_H2_6"],
                  round(row["probe_first_H2_6"], 2), flush=True)
    rows = [json.loads(x) for x in open(a.out)]
    for c in a.conditions:
        rs = [r for r in rows if r["condition"] == c and r["iters"] == a.iters]
        print(f"{c}: {sum(r['bayes_matched_H2_6'] for r in rs)}/{len(rs)} runs Bayes-optimal for H=2..6; "
              f"mean return H2-6 {sum(r['mean_return_H2_6'] for r in rs) / max(1, len(rs)):.1f}")


if __name__ == "__main__":
    main()
