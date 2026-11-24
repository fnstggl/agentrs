"""Train any learner through Crucible's own lifetime protocol (identical treatment for CAM 7.0 and the baselines).

    python -m superlearner.cam7.experiments.train '{"name": "dev_cam7", "learner": "cam7", "seed": 0, ...}'

Development evaluation sets are the ones the CAM 6.1 decisive runs used (8 tasks per family at the standard
validation offsets); they have been inspected repeatedly and are development data only. Confirmations use untouched
seeds through `exp1b.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import time

RUNS = "/workspace/agentrs/runs/cam7"
TRAIN = ["abm_congestion", "abm_utility", "abm_wom", "scm_additive", "scm_delay", "scm_gate"]
HELDOUT = ["abm_trust_gate", "scm_delay_gate", "scm_invu", "tech_pipeline"]


def make_learner(name: str, spec, seed: int, device: str, overrides: dict, scale: str = "base"):
    if name == "cam7":
        from ..agent import Cam7Learner
        return Cam7Learner(spec, seed, device, None, scale, **overrides)
    if name == "cam61":
        from superlearner.unified.experiments.harness import make_learner as mk
        return mk("cam61", spec, seed, device, overrides, scale)
    from crucible.research import learners
    return learners.make(name, spec, seed=seed, device="cpu", **overrides)


def run(a: dict) -> dict:
    import numpy as np
    import torch
    from crucible.research import protocols
    from crucible.research.learner import EnvSpec
    from crucible.research.learners.training import train_pool
    from crucible.research.tasks import eval_tasks

    spec = EnvSpec()
    seed = int(a["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    pool = train_pool(a.get("train", TRAIN), int(a.get("pool", 300)), start=int(a.get("pool_start", 0)),
                      workers=int(a.get("workers", 4)))
    sets = {name: eval_tasks(fams, int(a.get("n_eval", 8)), workers=int(a.get("workers", 4)), skip_failures=True)
            for name, fams in a.get("eval", {"familiar": TRAIN, "heldout": HELDOUT}).items()}
    learner = make_learner(a["learner"], spec, seed, a.get("device", "cuda"), a.get("overrides", {}), a.get("scale", "base"))
    order = np.random.default_rng(seed).permutation(len(pool))
    train = [pool[int(i)] for i in order]
    out_dir = os.path.join(RUNS, a["name"], f"s{seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_args.json"), "w") as fh:
        json.dump(a, fh, indent=1)
    t0 = time.time()

    def point(p: dict) -> None:
        row = {k: round(v["score"]["mean"], 3) for k, v in p.get("eval", {}).items() if isinstance(v, dict) and "score" in v}
        extra = {k: round(v, 3) for k, v in getattr(learner, "stats", {}).items() if isinstance(v, float)}
        print(f"[{a['name']} s{seed}] steps={p.get('env_steps')} t={time.time() - t0:.0f}s {row} {extra}", flush=True)

    res = protocols.lifetime(learner, train, int(a.get("budget", 240_000)), int(a.get("eval_every", 60_000)), sets,
                             spec=spec, n_envs=int(a.get("n_envs", 32)), update_every=int(a.get("update_every", 8)),
                             checkpoint_dir=os.path.join(out_dir, "ckpt"), resume=bool(a.get("resume", True)),
                             require_disjoint=True, label=a["name"], on_point=point)
    res["run_args"] = a
    protocols.save_result(res, os.path.join(out_dir, "lifetime.json"))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="JSON run description")
    run(json.loads(ap.parse_args().spec))
