#!/usr/bin/env python
"""Run one experiment config (A/B/C/D or an ablation). Example:

    python scripts/run_experiment.py configs/experiments/dev_D.yaml --set seed=1 name=dev_D_s1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentrs.config import load_config  # noqa: E402
from agentrs.training.pipeline import ExperimentConfig, run_experiment  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. rl.algo=reinforce seed=2")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    cfg = load_config(ExperimentConfig, a.config, a.set)
    res = run_experiment(cfg)
    print(json.dumps({k: v["success_rate"] for k, v in res["final"].items()}, indent=1))
