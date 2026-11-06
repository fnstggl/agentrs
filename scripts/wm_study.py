#!/usr/bin/env python
"""World-model evaluation on held-out instances and held-out FAMILIES vs explicit baselines.

    python scripts/wm_study.py --world-model runs/cache/wm_<key> --train runs/wm_study/train_demos_s0 \
        --eval val=runs/wm_study/val test=runs/wm_study/test test_family=runs/wm_study/test_family
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from agentrs.data.storage import ExperienceStore  # noqa: E402
from agentrs.evaluation.wm_eval import evaluate_world_model  # noqa: E402
from agentrs.training.common import load_model  # noqa: E402

KEYS = ("event_nll", "event_brier", "event_auroc_mean", "event_ece", "reward_nll", "reward_auroc", "terminal_nll",
        "terminal_auroc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--n-generate", type=int, default=80)
    ap.add_argument("--n-drift", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    wm = load_model(a.world_model)
    train = [t for t in ExperienceStore(a.train).load_transitions() if t.split == "train"]
    out = {"world_model": a.world_model, "n_train": len(train), "sets": {}}
    for spec in a.eval:
        name, path = spec.split("=")
        val = ExperienceStore(path).load_transitions()
        res = evaluate_world_model(wm, train, val, n_generate=a.n_generate, n_drift=a.n_drift)
        for k in ("model", "marginal", "action_type"):
            res[k].pop("per_output", None)
        out["sets"][name] = res
        print(name, json.dumps({m: {k: res[m][k] for k in KEYS} for m in ("model", "action_type", "marginal")}),
              json.dumps(res["generation"]), json.dumps(res["drift"]), flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
