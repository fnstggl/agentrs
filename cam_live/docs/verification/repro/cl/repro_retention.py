"""G-cl: reproduce D's retention numbers (diagnostics.json: +2.085 naive vs +0.150 with retention).

Runs D's own experiment (cam_live.credit.diagnose.exp_retention, seed 0, 1 thread) with the committed defaults and,
with --warmup N, with LearnerSettings.warmup_steps forced to N (the default changed 0 -> 8 in the commit that added
diagnostics.json). Prints the summary block.

  .venv/bin/python cam_live/docs/verification/repro/cl/repro_retention.py [--warmup 0] --work DIR
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, "cam_live/src")
from cam_live.credit import diagnose, learner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--warmup", type=int, default=None)
ap.add_argument("--work", required=True)
a = ap.parse_args()
if a.warmup is not None:
    orig = learner.LearnerSettings.from_cfg.__func__

    def from_cfg(cls, cfg, **over):
        over.setdefault("warmup_steps", a.warmup)
        return orig(cls, cfg, **over)
    learner.LearnerSettings.from_cfg = classmethod(from_cfg)
torch.set_num_threads(1)
torch.manual_seed(0)
res = diagnose.exp_retention(a.work)
print(json.dumps({"warmup_override": a.warmup, **res["summary"]}, indent=1))
