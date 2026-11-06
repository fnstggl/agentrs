#!/usr/bin/env python
"""Run one content-learning configuration (A-E). Example:

    python scripts/run_content_experiment.py configs/content/base.yaml --set variant=C seed=1 name=content_C_s1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentrs.config import load_config  # noqa: E402
from agentrs.content.experiments import ContentExperimentConfig, run_content_experiment  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    res = run_content_experiment(load_config(ContentExperimentConfig, a.config, a.set))
    print(json.dumps({k: v["success_rate"] for k, v in res["final"].items() if isinstance(v, dict) and "success_rate" in v},
                     indent=1))
