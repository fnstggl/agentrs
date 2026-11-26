"""Diagnostics of the shared backbone's persistent state (W3): delayed recall through Z and state-at-init measures.

* `delayed_recall(core, delay, ...)`: the audit's BPTT task (2026-11-26, h-core). A key (one of K random vectors) is a
  context token at step 0 only; `delay` - 1 further steps and one query step show only distractor tokens; a linear
  readout of mean-pooled Z_T must name the key. Chance = 1/K. The core is trained end to end (BPTT through every
  step) with AdamW; accuracy is measured on fresh held-out batches with the core in eval mode.
* `state_constancy(core, Z, kind=STATE)`: cosine of each row of Z with the constant state-stream embedding
  (RMS-normalised StatePos + KindEmb(STATE)) -- 0.995 for the pre-DOM core at initialisation.
* `relative_change(a, b)`: ||a - b|| / ||a|| (observation / previous-state dependence).

These run on the real `UnifiedCore` of any config (e.g. `build_model(make_preset("dom_dev")).core`); they are probes,
not part of the runtime.
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..interfaces import TokenKind, TokenSet


def relative_change(a: Tensor, b: Tensor) -> float:
    return float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12))


@torch.no_grad()
def state_constancy(core, Z: Tensor, kind: int = TokenKind.STATE) -> Tensor:
    """cos(Z_b, c) per row b, c = RMS-normalised (state_pos + kind_emb[kind]) flattened [S*D]."""
    c = (core.state_pos + core.kind_emb.weight[int(kind)]).detach().float()
    c = c / c.pow(2).mean().sqrt().clamp_min(1e-12)
    return torch.nn.functional.cosine_similarity(Z.detach().float().flatten(1), c.flatten()[None].to(Z.device), -1)


def _recall_batch(keys: Tensor, B: int, delay: int, n_tok: int, D: int, g: torch.Generator, dev) -> tuple:
    K = keys.shape[0]
    y = torch.randint(0, K, (B,), generator=g, device="cpu").to(dev)
    xs = [torch.randn(B, n_tok, D, generator=g, device="cpu").to(dev) for _ in range(delay + 1)]
    xs[0][:, 0] = keys[y]
    return y, xs


def _recall_forward(core, head: nn.Module, xs: list, B: int, dev) -> Tensor:
    Z = core.init_state(B, dev)
    for x in xs:                                  # delay steps (key at step 0) + 1 query step (distractors only)
        Z, _ = core(Z, TokenSet.from_dense(x, int(TokenKind.OBS_SEM)))
    return head(Z.mean(1))


def delayed_recall(core, delay: int, *, n_keys: int = 16, batch: int = 64, n_tok: int = 8, iters: int = 600,
                   lr: float = 3e-4, eval_every: int = 25, eval_batches: int = 4, target: Optional[float] = None,
                   seed: int = 0, clip: float = 1.0, device=None) -> dict:
    """Train `core` (+ a linear readout) on delayed recall at `delay` and report held-out accuracy.

    Stops early once the held-out accuracy reaches `target` (None = run all `iters`). Returns {"delay", "chance",
    "history": [(iteration, train loss, held-out accuracy)], "final_acc", "best_acc", "iters_run", "seconds",
    "reached_target_at"}. The core is modified in place (it is trained)."""
    dev = torch.device(device) if device is not None else next(core.parameters()).device
    D = core.cfg.d_model
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    keys = torch.randn(n_keys, D, generator=g).to(dev)
    head = nn.Linear(D, n_keys).to(dev)
    params = [p for p in core.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    eval_g = torch.Generator().manual_seed(seed + 2)
    eval_sets = [_recall_batch(keys, batch, delay, n_tok, D, eval_g, dev) for _ in range(eval_batches)]

    def evaluate() -> float:
        core.eval()
        with torch.no_grad():
            acc = [float((_recall_forward(core, head, xs, batch, dev).argmax(-1) == y).float().mean())
                   for y, xs in eval_sets]
        core.train()
        return sum(acc) / len(acc)

    hist, t0, reached, it = [], time.time(), None, 0
    acc0 = evaluate()
    hist.append((0, None, acc0))
    core.train()
    for it in range(1, iters + 1):
        y, xs = _recall_batch(keys, batch, delay, n_tok, D, g, dev)
        logits = _recall_forward(core, head, xs, batch, dev)
        loss = nn.functional.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        if it % eval_every == 0 or it == iters:
            acc = evaluate()
            hist.append((it, round(float(loss.detach()), 4), acc))
            if target is not None and acc >= target:
                reached = it
                break
    accs = [h[2] for h in hist]
    return {"delay": delay, "chance": 1.0 / n_keys, "history": hist, "final_acc": accs[-1], "best_acc": max(accs),
            "iters_run": it, "seconds": round(time.time() - t0, 1), "reached_target_at": reached,
            "batch": batch, "n_keys": n_keys, "n_tok": n_tok, "lr": lr}
