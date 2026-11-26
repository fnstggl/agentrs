"""W3 real end-to-end check (plan 2026-11-26 §3 W3), part 2: delayed recall at delay 16 >= 95% through the real
core of build_model(make_preset("dom_dev")) (Qwen3-0.6B semantic model; the recall task uses only the core).

The core object trained here is `model.core` of the real build (moved to the GPU, activation checkpointing on so a
run stays under 2 GB): key in the first context (2 seeds) and key after 4 distractor decisions (the first write cannot
simply persist). Contrasts on the same task and seed: the legacy core (no gate, state embeddings 0.5, the pre-DOM
model), the gated core with the audit's literal out_norm candidate, and the gated core without the unit-RMS state.
Held-out accuracy on 4 fresh batches of 64 (chance 1/16).

Run from superlearner/ with PYTHONPATH=src:../crucible/src:  python docs/evidence/dom/W3/w3_e2e_recall.py
Writes docs/evidence/dom/W3/w3_e2e_recall.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import torch

from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.probes import delayed_recall
from superlearner.cam_unified.model import build_model
from superlearner.cam_unified.semantic.encoder import QwenSemantic

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
OUT: dict = {"script": "superlearner/docs/evidence/dom/W3/w3_e2e_recall.py", "device": torch.cuda.get_device_name(0)}
try:
    OUT["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
except Exception:                                                     # pragma: no cover
    OUT["commit"] = None
seeds = [int(s) for s in (sys.argv[1:] or ["0", "1"])]
PATH = os.path.join(HERE, "w3_e2e_recall.json")
# resumable on a shared GPU: completed runs are kept (an out-of-memory caused by other tenants is retried later)
prev = json.load(open(PATH)) if os.path.exists(PATH) else {}
runs = list(prev.get("runs", []))
done = {(r["condition"], r["seed"]) for r in runs}


def save():
    OUT["runs"] = runs
    OUT["pass"] = all(r["best_acc"] >= 0.95 for r in runs if r["condition"].startswith("dom_dev"))
    OUT["n_dom_dev_runs"] = sum(1 for r in runs if r["condition"].startswith("dom_dev"))
    json.dump(OUT, open(PATH, "w"), indent=1)

cfg = make_preset("dom_dev")
sem = QwenSemantic(cfg, model_name=cfg.sem_model, device="cpu", dtype=torch.float32, decoder_grad_checkpoint=False)
DOM = "dom_dev build_model core (gated unit-RMS state, std 0.02, clamp candidate)"
DOM_K4 = "dom_dev build_model core, key after 4 distractor decisions"
for seed, key_step, name, iters in [(s, 0, DOM, 600) for s in seeds] + [(0, 4, DOM_K4, 800)]:
    if (name, seed) in done:
        continue
    torch.manual_seed(seed)
    model = build_model(cfg, components={"semantic": sem})
    core = model.core
    assert type(core) is UnifiedCore and core.state_gate is not None and core.state_emb_std == 0.02
    assert core.state_unit and core.state_candidate == "clamp"
    core.grad_checkpoint = True
    core.to(dev)
    torch.cuda.reset_peak_memory_stats()
    r = delayed_recall(core, 16, iters=iters, target=0.95, eval_every=25, seed=seed, key_step=key_step)
    r.update(condition=name, seed=seed, peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
             commit=OUT["commit"])
    runs.append(r)
    save()
    print(json.dumps({k: r[k] for k in ("condition", "seed", "best_acc", "reached_target_at", "seconds")}), flush=True)
    del model, core
    torch.cuda.empty_cache()

# contrasts: (name, config overrides, candidate mode, unit-RMS state, key_step, iterations)
contrasts = [("legacy core (no gate, state_emb_std 0.5)", dict(state_gate=False, state_emb_std=0.5), "rms", True, 0,
              300),
             ("gated core, audit-literal out_norm candidate, free-scale state", {}, "rms", False, 0, 300),
             ("gated core, clamp candidate, free-scale state", {}, "clamp", False, 0, 300),
             ("gated core, clamp candidate, free-scale state, key after 4", {}, "clamp", False, 4, 600),
             ("legacy core, key after 4", dict(state_gate=False, state_emb_std=0.5), "rms", True, 4, 300)]
for name, over, cand, unit, key_step, iters in contrasts:
    if (name, 0) in done:
        continue
    torch.manual_seed(0)
    c = UnifiedCore(replace(cfg, grad_checkpoint=True, **over)).to(dev)
    c.state_candidate, c.state_unit = cand, unit
    torch.cuda.reset_peak_memory_stats()
    r = delayed_recall(c, 16, iters=iters, target=0.95, eval_every=25, seed=0, key_step=key_step)
    r.update(condition=name, seed=0, peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
             commit=OUT["commit"])
    runs.append(r)
    save()
    print(json.dumps({k: r[k] for k in ("condition", "seed", "best_acc", "reached_target_at", "seconds")}), flush=True)
    del c
    torch.cuda.empty_cache()

save()
print("PASS", OUT["pass"], flush=True)
