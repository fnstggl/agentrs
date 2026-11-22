"""Instruments for the two ways a lifelong learner can fail: forgetting, and losing the ability to learn.

Nothing here trains the learner. The functions measure, on a fixed probe batch of recorded decisions:

    dormant units        share of feed-forward units whose mean absolute activation is at most `tau` times their
                         layer's mean (Sokar et al. 2023); such units no longer carry gradient
    effective rank       exp(entropy of the normalised singular values) of the controller's state z over the probe
                         decisions (Roy & Vetterli 2007); a collapsing representation loses rank
    weight / gradient    L2 norms per parameter group; the gradient is that of the imitation loss on the probe batch
    probe fit            how well a COPY of the controller can fit new, arbitrary targets (a fixed random valid head
                         per probe decision) in a fixed number of optimiser steps from a fresh optimiser - the direct
                         test of plasticity (Lyle et al. 2022): the same data, the same steps, only the starting
                         weights differ, so a higher final loss means those weights have become harder to train
    parameter change     L2 distance per group between two states of the same network

All of it is deterministic given the probe batch and the seed.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..cam.batch import collate, evaluate


def _groups(model) -> dict:
    from ..algo.continual import param_group
    by: dict = {}
    for n, p in model.named_parameters():
        by.setdefault(param_group(n), []).append((n, p))
    return by


def param_change(model, ref: dict) -> dict:
    """Per-group L2 distance between the controller's weights and a reference state (name -> tensor)."""
    out = {}
    tot_d, tot_n = 0.0, 0.0
    for g, items in _groups(model).items():
        # (a reference restored from a checkpoint may sit on another device than it was saved from)
        d = sum(float(((p.detach().cpu().float() - ref[n].detach().cpu().float()) ** 2).sum()) for n, p in items if n in ref)
        n2 = sum(float((ref[n].detach().cpu().float() ** 2).sum()) for n, p in items if n in ref)
        out[g] = {"l2": d ** 0.5, "relative": (d / n2) ** 0.5 if n2 > 0 else 0.0,
                  "params": int(sum(p.numel() for _, p in items))}
        tot_d += d
        tot_n += n2
    out["all"] = {"l2": tot_d ** 0.5, "relative": (tot_d / tot_n) ** 0.5 if tot_n > 0 else 0.0,
                  "params": int(sum(v["params"] for v in out.values()))}
    return out


def state_of(model) -> dict:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def weights_digest(model) -> str:
    import hashlib
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(n.encode())
        h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


def effective_rank(x: torch.Tensor) -> float:
    x = x.float() - x.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp(min=1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))


def diagnostics(model, episodes: list[dict], store, device="cpu", tau: float = 0.025) -> dict:
    """Dormant units, representation rank, weight and gradient norms on the probe episodes."""
    if not episodes:
        return {}
    acts: dict = {}
    hooks = []
    for i, layer in enumerate(model.encoder.layers):
        hooks.append(layer.linear1.register_forward_hook(
            lambda m, inp, out, i=i: acts.__setitem__(i, F.gelu(out.detach()).abs().reshape(-1, out.shape[-1]).mean(0))))
    was = model.training
    model.train()
    try:
        b = collate([e["decisions"] for e in episodes], store, device)
        model.zero_grad(set_to_none=True)
        out = evaluate(model, b, store)
        w = b.t["valid"].float()
        (-(out["logp"] * w).sum() / w.sum().clamp(min=1)).backward()
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    dormant = []
    for i in sorted(acts):
        a = acts[i]
        dormant.append(float(((a / a.mean().clamp(min=1e-12)) <= tau).float().mean()))
    z = out["z"][b.t["valid"]].detach()
    res = {"dormant_fraction": float(np.mean(dormant)) if dormant else 0.0, "dormant_by_layer": [round(x, 4) for x in dormant],
           "state_effective_rank": effective_rank(z), "state_dim": int(z.shape[-1]), "probe_decisions": int(z.shape[0]),
           "weight_norm": {}, "grad_norm": {}}
    for g, items in _groups(model).items():
        res["weight_norm"][g] = float(sum(float((p.detach() ** 2).sum()) for _, p in items) ** 0.5)
        res["grad_norm"][g] = float(sum(float((p.grad.detach() ** 2).sum()) for _, p in items if p.grad is not None) ** 0.5)
    model.zero_grad(set_to_none=True)
    return res


def probe_fit(model, episodes: list[dict], store, device="cpu", steps: int = 40, lr: float = 3e-4, seed: int = 0) -> dict:
    """Train a copy of the controller to choose a fixed random valid head at every probe decision."""
    if not episodes:
        return {}
    live, table = getattr(model, "live", None), model.table
    model.table = None                         # (a table computed with a graph cannot be copied)
    if live is not None:
        model.live = None                      # the copy trains the controller only, on the current text vectors
    try:
        m = copy.deepcopy(model)
    finally:
        model.table = table
        if live is not None:
            model.live = live
    m.train()
    b = collate([e["decisions"] for e in episodes], store, device)
    t = b.t
    g = torch.Generator().manual_seed(seed)
    n_valid = t["head_mask"].sum(-1).clamp(min=1)                                   # [B, T]
    target = (torch.rand(n_valid.shape, generator=g).to(device) * n_valid).long().clamp(max=t["head_mask"].shape[-1] - 1)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    curve = []
    for _ in range(steps):
        out = evaluate(m, b, store)
        w = t["valid"].float()
        nll = -(out["head_logp_all"].gather(2, target.unsqueeze(-1)).squeeze(-1) * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        nll.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        curve.append(float(nll.detach()))
    del m
    return {"steps": steps, "loss_first": curve[0], "loss_last": curve[-1], "loss_mean": float(np.mean(curve)),
            "chance": float((torch.log(n_valid.float()) * t["valid"].float()).sum() / t["valid"].float().sum())}
