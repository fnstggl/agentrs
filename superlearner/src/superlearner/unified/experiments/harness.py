"""Runs any learner through Crucible's own lifetime protocol, so CAM 6.1 and every baseline get identical treatment."""
from __future__ import annotations

import argparse
import json
import os
import time

RUNS = "/workspace/agentrs/runs/cam61"


def make_learner(name: str, spec, seed: int, device: str, overrides: dict, scale: str):
    if name == "cam61":
        from ..config import SCALES
        from ..runtime.learner import UnifiedLearner
        base = SCALES[scale]()
        cfg_kw = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
        return UnifiedLearner(spec, seed, device, base.with_(**cfg_kw), scale)
    from crucible.research import learners
    return learners.make(name, spec, seed=seed, device="cpu" if name.startswith("ppo") else device, **overrides)


def run(a: dict) -> dict:
    import numpy as np
    import torch
    from crucible.research import protocols
    from crucible.research.learner import EnvSpec
    from crucible.research.learners.training import pool_stream, train_pool
    from crucible.research.tasks import eval_tasks

    spec = EnvSpec()
    seed = int(a["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    pool = train_pool(a["train"], int(a.get("pool", 200)), start=int(a.get("pool_start", 0)), workers=int(a.get("workers", 4)))
    sets = {name: eval_tasks(fams, int(a.get("n_eval", 8)), offset=int(a.get("eval_offset", 0)), workers=int(a.get("workers", 4)),
                             skip_failures=True)
            for name, fams in a["eval"].items()}
    learner = make_learner(a["learner"], spec, seed, a.get("device", "cuda"), a.get("overrides", {}), a.get("scale", "small"))
    order = np.random.default_rng(seed).permutation(len(pool))
    train = [pool[int(i)] for i in order]
    out_dir = os.path.join(RUNS, a["name"], f"s{seed}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    def point(p: dict) -> None:
        row = {k: {"score": v["score"]["mean"], "success": v["success"]["p"]} for k, v in p.get("eval", {}).items()
               if isinstance(v, dict) and "score" in v}
        extra = {}
        if hasattr(learner, "trainer"):
            extra = {"reward_r2": round(learner.trainer.heldout_r2, 3), "effect_r2": round(learner.trainer.effect_r2, 3), "adapt_r2": round(learner.trainer.adapt_r2, 3)}
        print(f"[{a['name']} s{seed}] steps={p.get('env_steps')} t={time.time() - t0:.0f}s {row} {extra}", flush=True)

    res = protocols.lifetime(learner, train, int(a["budget"]), int(a["eval_every"]), sets, spec=spec,
                             n_envs=int(a.get("n_envs", 32)), update_every=int(a.get("update_every", 8)),
                             checkpoint_dir=os.path.join(out_dir, "ckpt"), resume=bool(a.get("resume", True)),
                             require_disjoint=True, label=a["name"], on_point=point)
    res["run_args"] = a
    protocols.save_result(res, os.path.join(out_dir, "lifetime.json"))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="JSON run description")
    run(json.loads(ap.parse_args().spec))
