"""The decisive comparison: do experience and gradient updates improve outcomes on mechanisms never trained on?

Every condition trains on the same six training families and is evaluated, frozen with fresh context, on (a) new seeds of
those families and (b) four held-out mechanism families. Point 0 of each run is the untrained copy of the same network.
Crucible's sealed test families are not touched.

    python -m superlearner.unified.experiments.decisive run      # resumable
    python -m superlearner.unified.experiments.decisive report
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

from .harness import RUNS

TRAIN = ["abm_congestion", "abm_utility", "abm_wom", "scm_additive", "scm_delay", "scm_gate"]
HELDOUT = ["abm_trust_gate", "scm_delay_gate", "scm_invu", "tech_pipeline"]
SEEDS = (0, 1, 2)
BUDGET = 240_000
TRAINING = {"batch_size": 32, "seq_len": 16, "burn_in": 4, "updates_per_step": 0.06, "entropy_coef": 0.003}
CONDITIONS = {
    # First round (2026-11-23), before the regression layer existed; kept so its results stay reproducible
    "cam61_full": ("cam61", {"plan_beta": 2.0, "use_effect_planner": True, "use_adaptive": False}),
    "cam61_no_planner": ("cam61", {"use_effect_planner": False, "use_adaptive": False}),
    "cam61_no_world_model": ("cam61", {"use_effect_planner": False, "use_world_model_loss": False, "use_adaptive": False,
                                       "shared_policy_gradients": True, "use_imagination": False}),
    # Second round, after adding the within-episode regression layer (2026-11-24)
    "adaptive_full": ("cam61", {"use_adaptive": True}),
    "adaptive_no_planner": ("cam61", {"use_adaptive": True, "use_adaptive_planner": False}),
    "adaptive_no_hand_evidence": ("cam61", {"use_adaptive": True, "use_hand_evidence": False}),
    "adaptive_no_world_model": ("cam61", {"use_adaptive": True, "use_world_model_loss": False,
                                          "shared_policy_gradients": True, "use_imagination": False}),
    "ppo_gru": ("ppo_gru", {}),
    "model_based": ("model_based", {}),
}


def spec(cond: str, seed: int) -> dict:
    learner, over = CONDITIONS[cond]
    s = {"name": f"decisive_{cond}", "learner": learner, "seed": seed, "train": TRAIN,
         "eval": {"familiar": TRAIN, "heldout": HELDOUT}, "pool": 300, "n_eval": 8, "budget": BUDGET,
         "eval_every": 60_000, "n_envs": 32, "update_every": 8, "workers": 1, "device": "cuda" if learner == "cam61" else "cpu"}
    s["overrides"] = {**TRAINING, **over} if learner == "cam61" else {}
    return s


def run(max_jobs: int = 12) -> None:
    from crucible.research.tasks import eval_tasks
    for fams in (TRAIN, HELDOUT):
        eval_tasks(fams, 8, workers=16, skip_failures=True)     # build shared task caches once, before workers race
    todo = [(c, s) for c in CONDITIONS for s in SEEDS
            if not os.path.exists(os.path.join(RUNS, f"decisive_{c}", f"s{s}", "lifetime.json"))]
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    os.makedirs(os.path.join(RUNS, "logs"), exist_ok=True)
    live: list = []
    failed = []
    while todo or live:
        while todo and len(live) < max_jobs:
            c, s = todo.pop(0)
            log = open(os.path.join(RUNS, "logs", f"decisive_{c}_s{s}.log"), "a")
            p = subprocess.Popen([sys.executable, "-u", "-m", "superlearner.unified.experiments.harness",
                                  json.dumps(spec(c, s))], stdout=log, stderr=subprocess.STDOUT, env=env)
            live.append((p, c, s))
        time.sleep(10)
        for item in list(live):
            if item[0].poll() is not None:
                live.remove(item)
                if item[0].returncode != 0:
                    failed.append(item[1:])
                print(f"finished {item[1]} s{item[2]} rc={item[0].returncode}", flush=True)
    print("failed:", failed, flush=True)


def _load(cond: str) -> list[dict]:
    out = []
    for s in SEEDS:
        path = os.path.join(RUNS, f"decisive_{cond}", f"s{s}", "lifetime.json")
        if os.path.exists(path):
            with open(path) as fh:
                out.append(json.load(fh))
    return out


def _vec(point: dict, name: str) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in point["eval"][name]["scores"]], dtype=float)


def _boot(x: np.ndarray, seed: int = 0, n: int = 10_000) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    idx = np.random.default_rng(seed).integers(len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def report() -> dict:
    res: dict = {"budget_env_steps": BUDGET, "seeds": list(SEEDS), "conditions": {}}
    finals: dict = {}
    for cond in CONDITIONS:
        runs = _load(cond)
        if not runs:
            continue
        row: dict = {"n_seeds": len(runs)}
        for name in ("familiar", "heldout"):
            first = np.mean([_vec(r["points"][0], name) for r in runs], 0)
            last = np.mean([_vec(r["points"][-1], name) for r in runs], 0)
            finals[(cond, name)] = last
            row[name] = {"untrained": _boot(first), "trained": _boot(last), "trained_minus_untrained": _boot(last - first),
                         "seed_means": [float(np.nanmean(_vec(r["points"][-1], name))) for r in runs]}
        c = runs[0].get("cost", {}).get("train", {})
        row["train_cost_seed0"] = {k: c.get(k) for k in ("env_steps", "updates", "train_cpu_s", "train_gpu_s", "infer_gpu_s")}
        res["conditions"][cond] = row
    res["paired_vs_baselines"] = {}
    pairs = [(a, b) for a in ("cam61_full", "cam61_no_planner") for b in ("cam61_no_world_model", "ppo_gru", "model_based")]
    pairs += [("adaptive_full", b) for b in ("adaptive_no_planner", "adaptive_no_hand_evidence", "adaptive_no_world_model",
                                            "cam61_full", "ppo_gru", "model_based")]
    for a, b in pairs:
        for name in ("familiar", "heldout"):
            if (a, name) in finals and (b, name) in finals:
                res["paired_vs_baselines"][f"{a} - {b} [{name}]"] = _boot(finals[(a, name)] - finals[(b, name)])
    path = os.path.join(RUNS, "decisive_report.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    for cond, row in res["conditions"].items():
        for name in ("familiar", "heldout"):
            r = row[name]
            print(f"{cond:22s}{name:9s} n={row['n_seeds']} untrained {r['untrained'][0]:+.3f} trained {r['trained'][0]:+.3f} "
                  f"gain {r['trained_minus_untrained'][0]:+.3f} [{r['trained_minus_untrained'][1]:+.3f}, "
                  f"{r['trained_minus_untrained'][2]:+.3f}] seeds {[round(x, 3) for x in r['seed_means']]}")
    for k, v in res["paired_vs_baselines"].items():
        print(f"{k:58s}{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
    return res


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
