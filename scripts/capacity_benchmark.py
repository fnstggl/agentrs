"""Matched capacity benchmark: fixed LoRA vs expanded-rank LoRA vs bounded expert adapters.

All arms start from the same checkpoint and get the same RL budget on identical task instances
(same factory indices, same sampling seeds per cycle): per cycle ``tasks`` x ``group`` rollouts,
GRPO advantages from the hidden verifiers, one PPO epoch (+ the same BC replay). After every cycle
each arm is evaluated (greedy) on the same validation tasks; at the end on the held-out families.
Reported per arm: validation success per cycle (learning efficiency), held-out-family success
(transfer), per-domain success vs the starting checkpoint (forgetting), trainable / active parameters,
peak VRAM and wall time.

    python scripts/capacity_benchmark.py --ckpt runs/overnight/ckpt/promoted --cycles 4 --out runs/capacity_bench.json

GPU-heavy (each arm ≈ cycles x (collection + update + validation)); run it when the GPU is free.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import torch

from agentrs.capacity import ExpertAdapters, expand_lora_rank, param_report
from agentrs.econ.core import REGISTRY, load_all, make_task
from agentrs.econ.learner import (GenPolicy, bc_examples, group_advantages, pg_samples, policy_backbone_cfg, ppo_update,
                                  summarize)
from agentrs.econ.rollout import rollout_batch, run_expert
from agentrs.training.common import GradientTrainer, OptimConfig, load_weights

ROLES = None


def domain_of(prompt: str) -> str:
    global ROLES
    if ROLES is None:
        ROLES = {}
        for k, fam in REGISTRY.items():
            env = fam.make_env(make_task(k, "val", 0))
            ROLES[getattr(env, "role", k.split("/")[0])] = k.split("/")[0]
    for role, d in ROLES.items():
        if role in prompt[:400]:
            return d
    return "browser"


@contextlib.contextmanager
def routed(bb, experts):
    """Route every generation / scoring call per example by its task domain (parsed from the prompt)."""
    if experts is None:
        yield
        return
    g0, s0 = bb.generate_tokens, bb.token_logprobs_from_ids

    def gen(prompts, *a, **k):
        with experts.route([domain_of(p) for p in prompts]):
            return g0(prompts, *a, **k)

    def score(prompts, *a, **k):
        with experts.route([domain_of(p) for p in prompts]):
            return s0(prompts, *a, **k)
    bb.generate_tokens, bb.token_logprobs_from_ids = gen, score
    try:
        yield
    finally:
        bb.generate_tokens, bb.token_logprobs_from_ids = g0, s0


def run_arm(arm: str, a, train_fams, heldout, val_tasks, ho_tasks) -> dict:
    meta = json.loads((Path(a.ckpt) / "meta.json").read_text()) if (Path(a.ckpt) / "meta.json").exists() else {"models": {}}
    torch.manual_seed(0)
    cfg = policy_backbone_cfg(a.model or meta["models"]["policy"])
    if a.cpu:
        cfg = cfg.model_copy(update={"device": "cpu", "dtype": "float32", "gradient_checkpointing": False})
    pol = GenPolicy(cfg, max_new_tokens=a.max_new_tokens, gen_batch=a.gen_batch)
    if not a.model:
        load_weights(Path(a.ckpt) / "policy", pol.bb)
    experts = None
    if arm == "expand_rank":
        expand_lora_rank(pol.bb, a.expand_to)
    elif arm == "experts":
        experts = ExpertAdapters(pol.bb, pol.bb.model, keys=sorted({k.split("/")[0] for k in REGISTRY}),
                                 n_experts=a.n_experts, r=a.expert_r)
    rep = param_report(pol.bb)
    tr = GradientTrainer(pol.bb, OptimConfig(lr=a.lr, total_steps=10**6, grad_clip=1.0))
    if experts is not None:  # the router is a separate module: give it its own parameter group
        tr.optimizer.add_param_group({"params": [experts.router.logits], "lr": a.lr * 50})
    demos = [ep for k in train_fams[:6] for ep in [run_expert(make_task(k, "train", 500_000 + i)) for i in range(4)]
             if ep.success]
    demo_ex = bc_examples(pol, demos)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    curve, t0 = [], time.time()
    with routed(pol.bb, experts):
        def ev(tasks):
            return summarize(rollout_batch(tasks, lambda r: pol.act(r, greedy=True), "eval", provenance="evaluation"))
        start = ev(val_tasks)
        curve.append({"cycle": 0, "val": start["overall"]["rate"]})
        for c in range(1, a.cycles + 1):
            rng = random.Random(1000 + c)
            tasks = [make_task(rng.choice(train_fams), "train", 600_000 + 100 * c + i) for i in range(a.tasks)]
            group_tasks = [t for t in tasks for _ in range(a.group)]
            groups = [f"c{c}:{t.id}" for t in tasks for _ in range(a.group)]
            pol.gen.manual_seed(1000 + c)
            eps = rollout_batch(group_tasks, lambda r: pol.act(r, greedy=False, temperature=1.0), f"v{c}",
                                provenance="agent_policy", groups=groups)
            st = ppo_update(pol, tr, pg_samples(eps, group_advantages(eps)), f"v{c}", demo_examples=demo_ex,
                            bc_coef=0.05, seed=c)
            s = ev(val_tasks)
            curve.append({"cycle": c, "train_success": sum(e.success for e in eps) / len(eps), "val": s["overall"]["rate"],
                          "updates": st["updates"]})
        end = s
        ho = ev(ho_tasks)
    forgetting = {d: round(end["by_domain"][d]["rate"] - start["by_domain"][d]["rate"], 3)
                  for d in start["by_domain"] if d in end["by_domain"]}
    out = {"arm": arm, "params": rep, "active_params_per_forward": rep["total"], "curve": curve,
           "val_start": start["overall"]["rate"], "val_end": end["overall"]["rate"],
           "heldout_families": ho["overall"]["rate"], "domain_change": forgetting,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
           "wall_s": round(time.time() - t0)}
    del pol, tr
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/overnight/ckpt/promoted")
    ap.add_argument("--model", help="override base model (smoke tests: Qwen/Qwen3-0.6B, no checkpoint weights)")
    ap.add_argument("--arms", default="fixed,expand_rank,experts")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--val-per-family", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--expand-to", type=int, default=32)
    ap.add_argument("--n-experts", type=int, default=3)
    ap.add_argument("--expert-r", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--families", type=int, default=0, help="limit families (smoke tests)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="runs/capacity_bench.json")
    a = ap.parse_args()
    load_all()
    train_fams = sorted(k for k, f in REGISTRY.items() if not f.heldout)
    heldout = sorted(k for k, f in REGISTRY.items() if f.heldout)
    if a.families:
        train_fams, heldout = train_fams[::max(1, len(train_fams) // a.families)][:a.families], heldout[:1]
    val_tasks = [make_task(k, "val", i) for k in train_fams for i in range(a.val_per_family)]
    ho_tasks = [make_task(k, "val", i) for k in heldout for i in range(a.val_per_family)]
    res = [run_arm(arm, a, train_fams, heldout, val_tasks, ho_tasks) for arm in a.arms.split(",")]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"time": time.time(), "args": vars(a), "arms": res}, indent=1, default=str))
    for r in res:
        print(f"{r['arm']:12s} trainable {r['params']['trainable']:>11,d} val {r['val_start']:.3f}->{r['val_end']:.3f} "
              f"heldout {r['heldout_families']:.3f} peak {r['peak_vram_gb']} GB {r['wall_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
