"""Main-scale instantiation of the complete system (spec 1.1 / Part 20 A01: "actual 16-layer main configuration
instantiated and run").

`main_scale_probe` builds ``make_config("main")`` (16 layers x 768 wide x 12 heads, 64 state tokens) with every
component - Qwen3-1.7B semantic model, tokenizer, action generator with its LoRA and bridge, core, entities, reasoner,
beliefs, Kalman layer, outcome head, world model, retriever, value / option / budget heads, built-in planner - plus
the trainer, archive, replay, consolidator and trace recorder, then lives a few decisions in a real ``ow_tiny`` world
through the lifetime harness and performs one neural update. Reported: parameter counts per component (CAM's own
trainable parameters vs the pretrained semantic model, and the spec's "core + heads" figure), wall time and peak GPU
memory of every act and of the update.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Optional

import torch

from .harness_agent import HarnessAgent, _jsonable
from .system import SystemConfig, build_system, describe
from .tasks import make_task, open_task_world

RUNS = "/workspace/agentrs/runs/cam_unified/wave2/S"
SPEC_TARGET_CORE_HEADS = (120_000_000, 180_000_000)


def _rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip()]


def core_and_heads(counts: dict) -> dict:
    """The spec's 'trainable core + heads' figure: every CAM component except the semantic-side adapters
    (tokenizer projections are CAM's own and included; LoRA / pretrained weights are not), and the same without the
    action generator (whose latent-to-language bridge maps into the 2048-wide Qwen embedding space)."""
    comps = counts["components"]
    tot = sum(v["trainable"] for v in comps.values())
    ag = comps.get("action_generator", {}).get("trainable", 0)
    return {"cam_trainable_total": tot, "cam_trainable_without_action_generator": tot - ag,
            "action_generator_trainable": ag, "spec_target": list(SPEC_TARGET_CORE_HEADS),
            "in_target_total": SPEC_TARGET_CORE_HEADS[0] <= tot <= SPEC_TARGET_CORE_HEADS[1]}


def main_scale_probe(sem_model: Optional[str] = None, work_dir: Optional[str] = None, n_decisions: int = 3,
                     imag: tuple = (2, 3, 1), max_arg_tokens: int = 32, keep: bool = False) -> dict:
    os.makedirs(RUNS, exist_ok=True)
    work = work_dir or tempfile.mkdtemp(prefix="main_scale_", dir=RUNS)
    shutil.rmtree(work, ignore_errors=True)
    sc = SystemConfig(scale="main", overrides={"max_arg_tokens": int(max_arg_tokens)}, sem_model=sem_model,
                      imag_candidates=imag[0], imag_horizon=imag[1], imag_samples=imag[2], seq_len=4, n_seqs=2,
                      adapt_seq_len=4, adapt_n_seqs=1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    system = build_system(work, sc)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    build_s = time.time() - t0
    build_peak = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
    desc = describe(system)
    h = HarnessAgent(system)
    task, fam, seed = make_task("open_economy", 0)
    ad, rop, rev = open_task_world(task, store_url=f"sqlite:///{os.path.join(work, 'world.db')}")
    from ..envs.harness import run_lifetime
    try:
        res = run_lifetime(h, ad, rev, max_decisions=n_decisions, think_s=600.0, update_every=n_decisions,
                           model_version=lambda: h.model_version)
    finally:
        ad.close()
    dec = _rows(os.path.join(work, "decisions.jsonl"))
    ups = _rows(os.path.join(work, "learner", "updates.jsonl"))
    counts = desc["parameters"]
    out = {"config": {k: desc["cam_config"][k] for k in ("scale", "n_layers", "d_model", "n_heads", "n_state",
                                                          "n_particles", "d_particle", "max_arg_tokens")},
           "d_ff": system.cfg.d_ff, "semantic_model": desc["semantic_model"], "build_report": desc["build_report"],
           "parameters": counts, "action_generator_parts": desc["action_generator_parts"],
           "core_and_heads": core_and_heads(counts), "trainer_groups": desc["trainer"]["groups"],
           "imagination": desc["imagination"], "build_s": build_s, "build_peak_gpu_mb": build_peak,
           "task": {"name": "open_economy", "world_family": fam, "world_seed": seed},
           "acts": [{"step": r["step"], "tool": r["tool"], "source": r["source"], "ok": r["ok"], "error": r["error"],
                     "act_s": r["act_s"], "peak_gpu_mb_act": r["peak_gpu_mb_act"],
                     "decoded_tokens_total": r["decoded_tokens_total"], "n_tools": r["n_tools"]} for r in dec],
           "update": None if not ups else {k: ups[0].get(k) for k in (
               "version", "losses", "update_s", "peak_gpu_mb_update", "grad_norm", "group_grad_norm",
               "backbone_grad_norm", "core_hash_before", "core_hash_after")}
           | {"core_l2_delta": ((ups[0].get("param_delta") or {}).get("core") or {}).get("l2_delta")},
           "lifetime_peak_gpu_mb": h.peak_gpu_mb, "decisions": res.steps,
           "work_dir": work if keep else None}
    system.recorder.close()
    system.archive.close()
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return _jsonable(out)
