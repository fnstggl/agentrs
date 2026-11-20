"""Continual-learning retention / plasticity experiment (task statement section 6).

Phase A trains CAM from scratch on task-family set FA. Phase B continues from A's weights (and A's replay buffer,
which travels with the checkpoint) on a disjoint family set FB, in two arms with identical seeds and budgets:
  * plain  - replay used only for world-model training (distill_coef = 0);
  * anchor - replay also anchors the policy/value on A's states (CLEAR-style behaviour KL, distill_coef > 0).
Evaluation (greedy, no updates) on held-out instances (val seed range) of FA and FB after A and after each B arm:
retention = FA success after B relative to after A; plasticity = FB success after B.

  PYTHONPATH=src python scripts/cam2/retention.py --root runs/cam2/retention --steps-a 6000 --steps-b 6000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentrs.cam2.backbone import load_llm  # noqa: E402
from agentrs.cam2.envs.miniwob import TRAIN_FAMILIES  # noqa: E402
from agentrs.cam2.envs.pool import EnvPool  # noqa: E402
from agentrs.cam2.tasks import val_tasks  # noqa: E402
from agentrs.cam2.trainer import RunConfig, Trainer  # noqa: E402

FA, FB = TRAIN_FAMILIES[:8], TRAIN_FAMILIES[8:]
LEARN = {"episodes_per_update": 64, "mb_episodes": 32, "lr": 5.0e-5, "epochs": 2, "warmup_steps": 10, "wm_coef": 0.5}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/cam2/retention")
    ap.add_argument("--steps-a", type=int, default=6000)
    ap.add_argument("--steps-b", type=int, default=6000)
    ap.add_argument("--distill", type=float, default=1.0)
    ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    res_p = root / "results.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {"FA": FA, "FB": FB, "config": vars(a)}
    dev = "cuda"
    llm = load_llm("Qwen/Qwen3-1.7B", device=dev)
    pool = EnvPool("miniwob", a.workers, {"max_steps": 10})
    tA, tB = val_tasks("miniwob", a.per_family, FA), val_tasks("miniwob", a.per_family, FB)

    def run(name, families, steps, seed, init=None, distill=0.0):
        rc = RunConfig(run_dir=str(root / name), env="miniwob", llm=llm.name, workers=a.workers, seed=seed,
                       env_steps=steps, max_minutes=40, max_steps=10, train_families=families, eval_every=0,
                       ckpt_every=5, cam={"el_goal_cond": False}, learn={**LEARN, "distill_coef": distill},
                       init_from=init, phase=name)
        tr = Trainer(rc, llm=llm, pool=pool, device=dev)
        if name not in res or "train" not in res[name]:
            res.setdefault(name, {})["train"] = tr.train()
            res_p.write_text(json.dumps(res, indent=1, default=float))
        for lab, ts in (("FA", tA), ("FB", tB)):
            if lab not in res[name]:
                ev = tr.evaluate(ts, label=f"{name}:{lab}")
                ev.pop("episodes")
                res[name][lab] = ev
                print(f"[retention] {name} on {lab}: {ev['success']:.3f} ci={ev['ci95']}", flush=True)
                res_p.write_text(json.dumps(res, indent=1, default=float))
        ck = str(root / name / "ckpt" / "final.pt")
        tr.close()
        del tr
        torch.cuda.empty_cache()
        return ck

    try:
        ckA = run("phaseA", FA, a.steps_a, a.seed)
        run("phaseB_plain", FB, a.steps_b, a.seed + 1, init=ckA, distill=0.0)
        run("phaseB_anchor", FB, a.steps_b, a.seed + 1, init=ckA, distill=a.distill)
        A = res["phaseA"]["FA"]["success"]
        for arm in ("phaseB_plain", "phaseB_anchor"):
            res[arm]["retention_FA"] = res[arm]["FA"]["success"] / max(1e-9, A)
            res[arm]["forgetting_FA"] = A - res[arm]["FA"]["success"]
            res[arm]["plasticity_FB_gain"] = res[arm]["FB"]["success"] - res["phaseA"]["FB"]["success"]
        res_p.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("retention_FA", "forgetting_FA", "plasticity_FB_gain")}
                          for k, v in res.items() if k.startswith("phaseB")}, indent=1))
    finally:
        pool.close()


if __name__ == "__main__":
    main()
