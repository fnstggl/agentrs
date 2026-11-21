"""A learned predictive model of the consequences of proposed actions.  [candidate-editable, optional]

Given the controller's state z_t (which already summarises the observation, history and recurrent memory) and a
proposed action (the embedding of the head and of its chosen arguments), `WorldModel` predicts

    ok          will the world accept the action?                              (binary)
    fclass      if not, which failure class?                                   (11-way)
    reward      the evaluator's reward for the decision                        (regression)
    next        the next decision's context encoding c_{t+1}                   (latent prediction, cosine loss)
    gain        discounted return from this decision on (a Q-like estimate)    (regression)

It is trained on recorded transitions (`fit`) and judged on held-out transitions (`evaluate_model`) against trivial
predictors (the base rate, the majority class, "nothing changes"). Imagined values are never assumed accurate:
`superlearner experiment worldmodel` reports held-out prediction quality and whether using the model to re-rank the
policy's head choices changes real task success.

Planning (`plan_bonus`): for every candidate head the model predicts, from z_t and the head alone (arguments not
chosen yet), the probability that the action will be accepted and the gain; the agent adds
``beta_ok * log P(ok) + beta_gain * gain`` to the policy's head logits - a one-step lookahead in latent space.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grounding.executor import FAILURE_CLASSES
from .batch import collate, evaluate

FC = ("",) + FAILURE_CLASSES


class WorldModel(nn.Module):
    def __init__(self, d: int = 384, hid: int = 512):
        super().__init__()
        self.d = d
        self.body = nn.Sequential(nn.Linear(3 * d, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        self.ok = nn.Linear(hid, 1)
        self.fclass = nn.Linear(hid, len(FC))
        self.reward = nn.Linear(hid, 1)
        self.gain = nn.Linear(hid, 1)
        self.next = nn.Linear(hid, d)

    def forward(self, z, e_head, e_args):
        h = self.body(torch.cat([z, e_head, e_args], -1))
        return {"ok": self.ok(h).squeeze(-1), "fclass": self.fclass(h), "reward": self.reward(h).squeeze(-1),
                "gain": self.gain(h).squeeze(-1), "next": self.next(h)}

    @torch.no_grad()
    def plan_bonus(self, z, head_e, beta_ok: float = 1.0, beta_gain: float = 1.0):
        """z [1, d], head_e [1, Nh, d] -> bonus [Nh] to add to the head logits."""
        n = head_e.shape[1]
        out = self.forward(z.expand(n, -1), head_e[0], torch.zeros_like(head_e[0]))
        return beta_ok * F.logsigmoid(out["ok"]) + beta_gain * out["gain"]


def transitions(model, episodes: list[dict], store, device="cpu", gamma: float = 0.995, chunk: int = 16) -> dict:
    """Recorded decisions -> tensors for the world model (the controller is used frozen, as a feature extractor)."""
    zs, eh, ea, ok, fc, rw, nx, gn, has_next = [], [], [], [], [], [], [], [], []
    model.eval()
    for i in range(0, len(episodes), chunk):
        part = episodes[i:i + chunk]
        b = collate([e["decisions"] for e in part], store, device)
        with torch.no_grad():
            out = evaluate(model, b, store)
        v = b.t["valid"]
        B, T = v.shape
        gain = torch.zeros(B, T, device=device)
        run = torch.zeros(B, device=device)
        for t in range(T - 1, -1, -1):
            run = (b.t["reward"][:, t] + gamma * run) * v[:, t]
            gain[:, t] = run
        nxt = torch.cat([out["c"][:, 1:], torch.zeros_like(out["c"][:, :1])], 1)
        hn = torch.cat([v[:, 1:], torch.zeros_like(v[:, :1])], 1)
        cls = torch.zeros(B, T, dtype=torch.long, device=device)
        for bi, e in enumerate(part):
            for ti, d in enumerate(e["decisions"]):
                cls[bi, ti] = FC.index(d.get("fclass", "")) if d.get("fclass", "") in FC else len(FC) - 1
        for name, src in ((zs, out["z"]), (eh, out["e_head"]), (ea, out["e_args"]), (ok, b.t["ok"]), (fc, cls),
                          (rw, b.t["reward"]), (nx, nxt), (gn, gain), (has_next, hn.float())):
            name.append(src[v].detach().cpu())
    return {"z": torch.cat(zs), "e_head": torch.cat(eh), "e_args": torch.cat(ea), "ok": torch.cat(ok), "fclass": torch.cat(fc),
            "reward": torch.cat(rw), "next": torch.cat(nx), "gain": torch.cat(gn), "has_next": torch.cat(has_next)}


def _loss(wm: WorldModel, d: dict, idx) -> tuple[torch.Tensor, dict]:
    out = wm(d["z"][idx], d["e_head"][idx], d["e_args"][idx])
    l_ok = F.binary_cross_entropy_with_logits(out["ok"], d["ok"][idx])
    l_fc = F.cross_entropy(out["fclass"], d["fclass"][idx])
    l_r = F.mse_loss(out["reward"], d["reward"][idx])
    l_g = F.mse_loss(out["gain"], d["gain"][idx])
    hn = d["has_next"][idx]
    l_n = ((1 - F.cosine_similarity(out["next"], d["next"][idx], dim=-1)) * hn).sum() / hn.sum().clamp(min=1)
    return l_ok + l_fc + 5 * l_r + l_g + l_n, {"ok": float(l_ok), "fclass": float(l_fc), "reward": float(l_r), "gain": float(l_g), "next": float(l_n)}


def fit(wm: WorldModel, data: dict, epochs: int = 30, lr: float = 1e-3, batch: int = 512, seed: int = 0, device="cpu") -> list[dict]:
    wm.to(device).train()
    data = {k: v.to(device) for k, v in data.items()}
    opt = torch.optim.AdamW(wm.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    n = len(data["ok"])
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        tot: dict = {}
        for i in range(0, n, batch):
            loss, info = _loss(wm, data, perm[i:i + batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            opt.step()
            for k, v in info.items():
                tot[k] = tot.get(k, 0.0) + v * len(perm[i:i + batch]) / n
        hist.append(dict(tot, epoch=ep))
    wm.eval()
    return hist


@torch.no_grad()
def evaluate_model(wm: WorldModel, data: dict, train: dict | None = None, device="cpu") -> dict:
    """Held-out prediction quality against trivial predictors fitted on the training transitions."""
    wm.to(device).eval()
    d = {k: v.to(device) for k, v in data.items()}
    out = wm(d["z"], d["e_head"], d["e_args"])
    ref = train if train is not None else data
    p_ok = float(ref["ok"].mean())
    ok_pred = (out["ok"] > 0).float()
    maj = int(torch.bincount(ref["fclass"], minlength=len(FC)).argmax())
    hn = d["has_next"] > 0
    cos_model = float(F.cosine_similarity(out["next"][hn], d["next"][hn], dim=-1).mean()) if hn.any() else 0.0
    prev_c = torch.cat([d["next"][:1] * 0, d["next"][:-1]])       # "nothing changes" baseline needs c_t; use z as proxy
    cos_mean = float(F.cosine_similarity(ref["next"].to(device).mean(0, keepdim=True).expand_as(d["next"][hn]), d["next"][hn], dim=-1).mean()) if hn.any() else 0.0
    eps = 1e-6
    nll_model = float(F.binary_cross_entropy_with_logits(out["ok"], d["ok"]))
    nll_base = float(-(d["ok"] * np.log(p_ok + eps) + (1 - d["ok"]) * np.log(1 - p_ok + eps)).mean())
    return {"n": int(len(d["ok"])),
            "ok_accuracy": float((ok_pred == d["ok"]).float().mean()), "ok_accuracy_base_rate": max(p_ok, 1 - p_ok),
            "ok_nll": nll_model, "ok_nll_base_rate": nll_base,
            "fclass_accuracy": float((out["fclass"].argmax(-1) == d["fclass"]).float().mean()),
            "fclass_accuracy_majority": float((d["fclass"] == maj).float().mean()),
            "reward_mse": float(F.mse_loss(out["reward"], d["reward"])),
            "reward_mse_mean_predictor": float(((d["reward"] - ref["reward"].to(device).mean()) ** 2).mean()),
            "gain_mse": float(F.mse_loss(out["gain"], d["gain"])),
            "gain_mse_mean_predictor": float(((d["gain"] - ref["gain"].to(device).mean()) ** 2).mean()),
            "next_latent_cosine": cos_model, "next_latent_cosine_mean_predictor": cos_mean}
