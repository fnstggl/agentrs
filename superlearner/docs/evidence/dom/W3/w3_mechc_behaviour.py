"""W3 Mechanism C behavioural probe (audit 2026-11-26 h-core, Mechanism C action (e); synthetic inputs, real core).

Task: each row has a semantic token sequence of 200 states (dim 1024 = Qwen3-0.6B width) of random distractors with
one "fact" token = one of 16 key vectors at a random position. The core's context gets only 4 attention-pool-like
summaries (means of 50-token segments: the fact is diluted 1:50, as the tokenizer's pooled tokens dilute token
detail). A linear readout of Z_1 must name the key (chance 1/16). The real dom_dev UnifiedCore (gated state, std 0.02,
cross-attention after blocks 3 and 7) is trained with the semantic context passed explicitly; then evaluated with
the gates open, with the gates closed (gate bias -30), and without semantic input. A second core of the same config
is trained without semantic input (pooled context only) as the baseline.

Run from superlearner/ with PYTHONPATH=src:../crucible/src:  python docs/evidence/dom/W3/w3_mechc_behaviour.py
Writes docs/evidence/dom/W3/w3_mechc_behaviour.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import torch
import torch.nn as nn

from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.semantic_xattn import SemanticContext
from superlearner.cam_unified.interfaces import TokenKind, TokenSet

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT: dict = {"script": "superlearner/docs/evidence/dom/W3/w3_mechc_behaviour.py", "device": dev}
try:
    OUT["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
except Exception:                                                     # pragma: no cover
    OUT["commit"] = None

cfg = make_preset("dom_dev", grad_checkpoint=True)
D, SD, NS, K, SEG = cfg.d_model, cfg.sem_xattn_dim, 200, 16, 50
g0 = torch.Generator().manual_seed(0)
keys = torch.randn(K, SD, generator=g0)
proj = torch.randn(SD, D, generator=g0) / SD ** 0.5                   # fixed map semantic -> CAM width (pooled path)


def batch(B, g):
    y = torch.randint(0, K, (B,), generator=g)
    h = torch.randn(B, NS, SD, generator=g)
    pos = torch.randint(0, NS, (B,), generator=g)
    h[torch.arange(B), pos] = keys[y] * 1.0
    pooled = torch.stack([h[:, i * SEG:(i + 1) * SEG].mean(1) for i in range(NS // SEG)], 1) @ proj
    ctx = TokenSet.from_dense(pooled.to(dev), int(TokenKind.OBS_SEM))
    sem = SemanticContext(h.to(dev), torch.ones(B, NS, dtype=torch.bool, device=dev))
    return y.to(dev), ctx, sem


def run(core, head, ctx, sem, B):
    Z, _ = core(core.init_state(B, dev), ctx, sem=sem)
    return head(Z.mean(1))


def train(use_sem: bool, iters=400, B=64, seed=0):
    torch.manual_seed(seed)
    core = UnifiedCore(cfg).to(dev)
    head = nn.Linear(D, K).to(dev)
    opt = torch.optim.AdamW(list(core.parameters()) + list(head.parameters()), lr=3e-4)
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    hist = []
    for it in range(1, iters + 1):
        y, ctx, sem = batch(B, g)
        loss = nn.functional.cross_entropy(run(core, head, ctx, sem if use_sem else None, B).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()
        if it % 50 == 0:
            hist.append((it, round(float(loss.detach()), 4)))
    return core, head, hist, round(time.time() - t0, 1)


@torch.no_grad()
def evaluate(core, head, mode, n=8, B=64):
    core.eval()
    g = torch.Generator().manual_seed(999)
    acc = []
    saved = {k: m.gate_out.bias.clone() for k, m in core.xattn.items()}
    if mode == "gates_closed":
        for m in core.xattn.values():
            m.gate_out.bias.fill_(-30.0)
    for _ in range(n):
        y, ctx, sem = batch(B, g)
        p = run(core, head, ctx, None if mode == "no_sem" else sem, B).argmax(-1)
        acc.append(float((p == y).float().mean()))
    for k, m in core.xattn.items():
        m.gate_out.bias.copy_(saved[k])
    core.train()
    return sum(acc) / len(acc)


core, head, hist, secs = train(True)
res = {"trained_with_sem": {"loss_history": hist, "seconds": secs,
                            "acc_gates_open": evaluate(core, head, "open"),
                            "acc_gates_closed": evaluate(core, head, "gates_closed"),
                            "acc_no_sem": evaluate(core, head, "no_sem"),
                            "gate_means_last_eval": core.gate_stats()}}
core2, head2, hist2, secs2 = train(False)
res["trained_without_sem"] = {"loss_history": hist2, "seconds": secs2, "acc": evaluate(core2, head2, "no_sem")}
res["chance"] = 1 / K
OUT.update(res)
t = res["trained_with_sem"]
OUT["pass"] = bool(t["acc_gates_open"] - t["acc_gates_closed"] > 0.30)
print(json.dumps(OUT, indent=1), flush=True)
json.dump(OUT, open(os.path.join(HERE, "w3_mechc_behaviour.json"), "w"), indent=1)
