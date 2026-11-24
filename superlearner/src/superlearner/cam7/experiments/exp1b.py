"""Experiment 1, condition B: each learner chooses its own interventions in unfamiliar worlds; weights frozen.

Every learner starts each task with fresh context (no stored experience, no previous posterior) and frozen weights. The
evaluator records the net income of every tick. Adaptation curve: the normalised score (task.score: 0 = holding the
defaults, 1 = the task's private reference policy) of the mean net over the window of ticks [k, 2k) for k in
0, 1, 2, 4, 8, 16, 32, 64 (k = 0: the first tick), i.e. performance after k observations. Also reported: the
whole-episode score (area under the adaptation curve, the protocol's score), regret (1 - score), success, and compute.

Intervals: two-level bootstrap over training seeds and tasks; paired differences are per task, averaged over seeds.

    python -m superlearner.cam7.experiments.exp1b run --set confirm --out ... name=spec ...
    python -m superlearner.cam7.experiments.exp1b report --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .train import HELDOUT, TRAIN, make_learner

KS = (0, 1, 2, 4, 8, 16, 32, 64)
# evaluation seed ranges inside the validation range (VAL_SEED0 + offset): 0-7 development (the decisive sets),
# 60_000+ development for Experiment 1B, 70_000+ untouched confirmation (declared in the ledger before use)
OFFSETS = {"dev": 60_000, "confirm": 70_000}


def window_scores(task, nets: np.ndarray) -> list:
    out = []
    for k in KS:
        lo, hi = k, max(k + 1, 2 * k)
        seg = nets[lo:hi]
        out.append(None if len(seg) == 0 else task.score(float(seg.mean())))
    return out


def run_one(learner, tasks, spec, n_envs: int = 32) -> list[dict]:
    from crucible.research.envs import MechVecEnv
    recs = []
    by_h: dict = {}
    for t in tasks:
        by_h.setdefault(int(t.horizon), []).append(t)
    for h, group in sorted(by_h.items()):
        for i in range(0, len(group), n_envs):
            batch = group[i:i + n_envs]
            env = MechVecEnv(batch, spec)
            obs = env.reset()
            learner.reset_context(env.n)
            nets = np.zeros((h, env.n))
            t0 = time.perf_counter()
            for t in range(h):
                act = learner.act(obs, explore=False)
                obs, rew, done, info = env.step(act)
                nets[t] = info["net_cents"]
            wall = time.perf_counter() - t0
            fin = {r["task"]: r for r in env.drain()}
            for j, task in enumerate(batch):
                r = fin[task.id]
                recs.append({"task": task.id, "family": task.family, "seed": task.seed, "score": r["score"],
                             "success": r["success"], "curve": window_scores(task, nets[:, j]),
                             "wall_s_batch": wall})
    return recs


def run(args) -> None:
    import torch
    from crucible.research.learner import EnvSpec
    from crucible.research.tasks import eval_tasks
    spec = EnvSpec()
    fams = HELDOUT if args.families == "heldout" else TRAIN
    tasks = eval_tasks(fams, args.n, offset=OFFSETS[args.set], workers=8, skip_failures=True)
    os.makedirs(args.out, exist_ok=True)
    for item in args.items:
        name, rest = item.split("=", 1)
        ckpts = [None]
        if "@" in rest:
            rest, ck = rest.split("@", 1)
            ckpts = ck.split(",")
        learner, over = (rest.split(":", 1) + ["{}"])[:2]
        over = json.loads(over)
        scale = over.pop("scale", "base")
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path):
            print("exists", path)
            continue
        per_seed = {}
        for si, ck in enumerate(ckpts):
            torch.manual_seed(si)
            L = make_learner(learner, spec, si, "cuda", dict(over), scale)
            if ck:
                L.load(ck)
            L.freeze(True)
            t0 = time.time()
            recs = run_one(L, tasks, spec)
            u = L.usage()
            per_seed[str(si)] = {"checkpoint": ck, "records": recs, "usage": u.to_dict(), "wall_s": time.time() - t0}
            print(name, si, "score", round(float(np.mean([r["score"] for r in recs])), 3), flush=True)
        with open(path, "w") as fh:
            json.dump({"name": name, "learner": learner, "overrides": over, "set": args.set, "families": fams,
                       "n_per_family": args.n, "seeds": per_seed}, fh)


def _boot2(arr: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """arr [seeds, tasks] -> mean and 95% interval, resampling seeds then tasks."""
    rng = np.random.default_rng(seed)
    S, T = arr.shape
    b = np.array([np.nanmean(arr[rng.integers(S, size=S)][:, rng.integers(T, size=T)]) for _ in range(n_boot)])
    return float(np.nanmean(arr)), float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))


def load(out: str) -> dict:
    res = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(".json") and not f.startswith("report"):
            with open(os.path.join(out, f)) as fh:
                d = json.load(fh)
            res[d["name"]] = d
    return res


def matrix(d: dict, key: str = "score", k_index: int | None = None) -> np.ndarray:
    rows = []
    for s in d["seeds"].values():
        recs = sorted(s["records"], key=lambda r: r["task"])
        if k_index is None:
            rows.append([np.nan if r[key] is None else float(r[key]) for r in recs])
        else:
            rows.append([np.nan if r["curve"][k_index] is None else float(r["curve"][k_index]) for r in recs])
    return np.array(rows)


def report(args) -> dict:
    res = load(args.out)
    out = {"learners": {}, "paired": {}}
    print(f"{'learner':28s} {'episode score':>24s} " + " ".join(f"k={k:<5d}" for k in KS))
    for name, d in res.items():
        sc = _boot2(matrix(d))
        curve = [float(np.nanmean(matrix(d, k_index=i))) for i in range(len(KS))]
        out["learners"][name] = {"score": sc, "curve": curve, "n_seeds": len(d["seeds"])}
        print(f"{name:28s} {sc[0]:+.3f} [{sc[1]:+.3f},{sc[2]:+.3f}] " + " ".join(f"{c:+.3f}" for c in curve))
    if args.ref:
        for ref in args.ref.split(","):
            if ref not in res:
                continue
            for name, d in res.items():
                if name == ref:
                    continue
                a, b = matrix(d), matrix(res[ref])
                diff = a.mean(0, keepdims=True) - b.mean(0, keepdims=True) if a.shape[0] != b.shape[0] else a - b
                out["paired"][f"{name} - {ref}"] = _boot2(diff)
                m, lo, hi = out["paired"][f"{name} - {ref}"]
                print(f"  {name} - {ref}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="dev", choices=list(OFFSETS))
    ap.add_argument("--families", default="heldout", choices=["heldout", "familiar"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ref", default="")
    ap.add_argument("items", nargs="*")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
