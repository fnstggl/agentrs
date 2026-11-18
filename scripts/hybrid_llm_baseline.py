"""Arm D: the frozen pretrained LM acting directly (no CAM, no learning).

Same tasks and seeds as CAM's evaluations (val: 1_000_000 + i, test: i), same permissible information
(the natural-language goal and the natural-language observation / own-action history), same action
grammar and step limits. Every action slot is chosen by the model's own log-likelihood of the option text
(grammar-constrained, so every executed action is valid). Weights are frozen and fingerprinted.

    PYTHONPATH=src python scripts/hybrid_llm_baseline.py --model Qwen/Qwen3-4B --split val --style train --n 60 \
        --out runs/cam_hybrid/llm_direct_qwen3-4b_val_train.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from agentrs.cam.envs import ENVS
from agentrs.cam.hybrid.semantic import FrozenLLMActor
from agentrs.cam.text.render import goal_text, obs_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--families", default="negotiation,product,bizdev")
    ap.add_argument("--split", default="val")
    ap.add_argument("--style", default="train")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_all = time.time()
    actor = FrozenLLMActor(a.model)
    import hashlib

    def fp():
        h = hashlib.sha256()
        for k, p in sorted(actor.model.state_dict().items()):
            h.update(k.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    fp0 = fp()
    base = 1_000_000 if a.split == "val" else 0
    rep = {"model": a.model, "split": a.split, "style": a.style, "n": a.n, "families": {}}
    for fam in a.families.split(","):
        succ, viol, prog, steps, secs = [], [], [], [], []
        for i in range(a.n):
            env = ENVS[fam]()
            env.reset(base + i, a.split)
            g = goal_text(fam, env.goal_tokens(), a.style, key=(a.split, base + i, "goal"))
            hist = []
            t0 = time.time()
            n = 0
            while not env.done and n < 2 * env.max_steps:
                hist.append("Observation: " + obs_text(fam, env.obs_tokens(), a.style, key=(a.split, base + i, "scratch", n)))
                act = actor.act(fam, g, hist, env.grammar())
                hist.append("Action: " + " ".join(act))
                env.step(act)
                n += 1
            secs.append(time.time() - t0)
            succ.append(bool(env.success))
            viol.append(env.task_status().violated is not None)
            prog.append(env.progress())
            steps.append(n)
        p = sum(succ) / len(succ)
        half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / len(succ))
        rep["families"][fam] = {"n": len(succ), "success": sum(succ), "rate": p, "ci95": [max(0, p - half), min(1, p + half)],
                                "violation_rate": sum(viol) / len(viol), "final_progress": sum(prog) / len(prog),
                                "avg_steps": sum(steps) / len(steps), "sec_per_episode": sum(secs) / len(secs)}
        print(fam, rep["families"][fam], flush=True)
    rep["decisions"] = actor.stats["decisions"]
    rep["forward_s"] = actor.stats["forward_s"]
    rep["wall_s"] = time.time() - t_all
    rep["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    rep["weights_unchanged"] = fp() == fp0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
