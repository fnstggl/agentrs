"""G-nn probe: is the recorded fast-adaptation verdict robust to a stronger recurrent baseline?

    .venv/bin/python cam_live/docs/verification/repro/nn/probe_fast_compare_gru.py [--hidden 256] [--steps 16000] [--kind real|count]

Same task generator, evaluation tasks (seed 0) and training loop as ``cam_live.nn.fast_compare``; only the GRU is
wider and trained longer than in the recorded run (hidden 64, 8000 steps). Compare the printed numbers with
docs/evidence/A/fast_adaptation.json, per_seed[0].
"""
import json
import sys
import time

import _common  # noqa: F401
from _common import info

import torch

from cam_live.contracts import OutcomeKind
from cam_live.nn import fast_compare as fc

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
hidden, steps, kind_s = arg("--hidden", 256), arg("--steps", 16000), arg("--kind", "real")
kind = OutcomeKind.REAL if kind_s == "real" else OutcomeKind.COUNT
torch.set_num_threads(8)
seed = 0
torch.manual_seed(seed)
ge = torch.Generator().manual_seed(777 + seed)                      # exactly as fast_compare.run
splits = {"in_distribution": fc.sample_tasks(fc.TRAIN_FAMILIES, 600, 32, kind, ge),
          "held_out_families": fc.sample_tasks(fc.HELDOUT_FAMILIES, 600, 32, kind, ge)}
gru = fc.GRUAdapter(kind, hidden=hidden)
t0 = time.time()
curve = fc.train(gru, kind, steps, 48, 32, seed)
res = {s: fc.evaluate(gru, d) for s, d in splits.items()}
out = {"kind": kind_s, "gru_hidden": hidden, "gru_steps": steps, "train_s": round(time.time() - t0, 1), "train_curve": curve}
for s, r in res.items():
    out[s] = {"mean_nll_k_ge_1": r["mean_nll_k_ge_1"], "se": r["mean_nll_k_ge_1_se"],
              "nll_at_k": {k: v["nll"] for k, v in r["per_support_size"].items()}}
rec = json.load(open(_common.os.path.join(_common.SRC, "..", "docs", "evidence", "A", "fast_adaptation.json")))
tname = "gaussian" if kind_s == "real" else "count"
m = rec["targets"][tname]["per_seed"][0]["methods"]
out["recorded_seed0"] = {n: {s: m[n][s]["mean_nll_k_ge_1"] for s in ("in_distribution", "held_out_families")} for n in m}
info("result", json.dumps(out))
lp = out["recorded_seed0"]["learned_prior_bayes"]["held_out_families"]
print(f"\nheld-out families, seed 0: learned-prior Bayes (recorded) {lp:.4f}; recorded GRU(64, 8000 steps) "
      f"{out['recorded_seed0']['gru']['held_out_families']:.4f}; this GRU({hidden}, {steps} steps) {out['held_out_families']['mean_nll_k_ge_1']:.4f}")
print("VERDICT:", "the stronger GRU beats the learned prior on held-out families" if out["held_out_families"]["mean_nll_k_ge_1"] < lp
      else "the learned prior still wins on held-out families")
