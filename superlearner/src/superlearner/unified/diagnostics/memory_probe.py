"""Where does decision-relevant evidence get lost? A controlled retention diagnostic for CAM's belief pathway.

A cue (+1 or -1) appears in one reading on the first step only. Three other readings carry noise on every step. After
``delay`` steps a random action is taken, and the reward on the following step is ``cue * action``. Nothing else
depends on the cue, so a model can predict that reward only if it kept the cue for ``delay`` steps.

Each architecture shares CAM's observation encoder and is trained under one of two objectives:

    supervised   classify the cue from the belief at step ``delay`` (can the architecture retain it at all?)
    world        CAM's world-model losses only: reward two-hot, observation prediction, and KL for the RSSM
                 (does prediction alone make the belief keep it?)

Afterwards a linear probe, fitted on half of a fresh batch and scored on the other half, reads the cue from every stage
of the pathway at step 0 and at step ``delay``. The cue label is used only by the supervised objective and the probes;
no RL task receives it.

    python -m superlearner.unified.diagnostics.memory_probe grid --out runs/cam7/memory
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import small
from ..core.rssm import RSSM
from ..dist import balanced_kl, make_bins, twohot_loss, twohot_mean
from ..encoders.observation import ObservationEncoder, _encoder, mlp
from ..interfaces import ObservationBatch

N_IDS = 33
PRESENT = [0, 1, 2, 3, 16, 17, 28, 29, 30, 31, 32]


def make_batch(b: int, delay: int, device, gen: torch.Generator) -> dict:
    T = delay + 2
    cue = torch.randint(0, 2, (b,), device=device, generator=gen).float() * 2 - 1
    values = torch.zeros(b, T, N_IDS, device=device)
    values[:, 0, 0] = cue
    values[:, :, 1:4] = torch.randn(b, T, 3, device=device, generator=gen)
    values[:, :, 32] = torch.arange(T, device=device).float() / T
    deltas = values - torch.cat([values[:, :1], values[:, :-1]], 1)
    present = torch.zeros(b, T, N_IDS, dtype=torch.bool, device=device)
    present[:, :, PRESENT] = True
    action = torch.randint(0, 2, (b, T), device=device, generator=gen).float() * 2 - 1
    reward = torch.zeros(b, T, device=device)           # reward arriving at step t, caused by the action at t-1
    reward[:, delay + 1] = cue * action[:, delay]
    return {"values": values, "deltas": deltas, "present": present, "action": action, "reward": reward, "cue": cue}


class Pathway(nn.Module):
    """Observation encoder -> memory core -> belief ``u``. Records every intermediate representation."""

    def __init__(self, arch: str, d: int = 128):
        super().__init__()
        self.arch = arch
        cfg = small().with_(d_model=d, action_dim=d, goal_dim=d, memory_dim=d, use_slots=arch == "rssm_slots",
                            latent_mode="probs" if arch == "rssm_probs" else "sample")
        self.cfg = cfg
        self.enc = ObservationEncoder(cfg, N_IDS)
        self.act = nn.Embedding(2, d)
        if arch.startswith("rssm"):
            self.core = RSSM(cfg)
        elif arch == "gru":
            self.core = nn.GRUCell(2 * d, cfg.deter_dim)
            self.out = mlp(cfg.deter_dim, d, d, layers=1)
        elif arch == "attention":
            self.pos = nn.Embedding(256, d)
            self.core = _encoder(d, 4, 2)
        else:
            raise ValueError(arch)
        self.reward = mlp(d, d, 255, layers=2)
        self.decode = mlp(d, d, N_IDS, layers=1)
        self.cue_head = nn.Linear(d, 1)
        self.register_buffer("bins", make_bins(255, -20, 20), persistent=False)

    def forward(self, batch: dict) -> dict:
        v = batch["values"]
        B, T, N = v.shape
        ids = torch.arange(N, device=v.device).expand(B * T, -1)
        o = ObservationBatch(v.flatten(0, 1), batch["deltas"].flatten(0, 1), batch["present"].flatten(0, 1), ids)
        tokens, pooled = self.enc(o)
        tokens, pooled = tokens.reshape(B, T, -1), pooled.reshape(B, T, -1)
        prev_a = torch.cat([torch.zeros_like(batch["action"][:, :1]), batch["action"][:, :-1]], 1)
        a_emb = self.act(((prev_a + 1) / 2).long()) * (torch.arange(T, device=v.device) > 0).float().view(1, T, 1)
        rec: dict = {"tokens": tokens, "pooled": pooled}
        if self.arch.startswith("rssm"):
            s = self.core.initial(B, v.device)
            goal = torch.zeros(B, self.cfg.goal_dim, device=v.device)
            outs = []
            for t in range(T):
                s = self.core.observe(s, a_emb[:, t], torch.ones(B, device=v.device) * (t > 0), pooled[:, t], goal)
                outs.append(s)
            for f in ("h", "z", "read", "u", "post_logits", "prior_logits"):
                rec[f] = torch.stack([getattr(x, f) for x in outs], 1)
            rec["z"] = rec["z"].flatten(-2)
            rec["post_logits"] = rec["post_logits"].flatten(-2)
            rec["prior_flat"] = rec["prior_logits"].flatten(-2)
        elif self.arch == "gru":
            h = torch.zeros(B, self.cfg.deter_dim, device=v.device)
            hs = []
            for t in range(T):
                h = self.core(torch.cat([pooled[:, t], a_emb[:, t]], -1), h)
                hs.append(h)
            rec["h"] = torch.stack(hs, 1)
            rec["u"] = self.out(rec["h"])
        else:
            x = pooled + a_emb + self.pos(torch.arange(T, device=v.device))
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=v.device), 1)
            rec["u"] = self.core(x, mask=causal)
        return rec

    def loss(self, batch: dict, objective: str, delay: int) -> tuple[Tensor, dict]:
        rec = self.forward(batch)
        u = rec["u"]
        info = {}
        if objective == "supervised":
            logit = self.cue_head(u[:, delay]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, (batch["cue"] > 0).float())
            info["acc"] = float(((logit > 0) == (batch["cue"] > 0)).float().mean())
        elif objective == "world":
            loss = twohot_loss(self.reward(u), batch["reward"], self.bins).mean()
            pres = batch["present"].float()
            loss = loss + ((self.decode(u) - batch["values"]) ** 2 * pres).sum() / pres.sum()
            if self.arch.startswith("rssm"):
                B, T = u.shape[:2]
                post = rec["post_logits"].reshape(B, T, self.cfg.stoch_vars, self.cfg.stoch_classes)
                prior = rec["prior_logits"]
                dyn, rep = balanced_kl(post, prior, self.cfg.free_nats)
                loss = loss + self.cfg.beta_dyn * dyn.mean() + self.cfg.beta_rep * rep.mean()
            pred = twohot_mean(self.reward(u[:, delay + 1]), self.bins)
            info["reward_sign_acc"] = float(((pred > 0) == (batch["reward"][:, delay + 1] > 0)).float().mean())
        else:
            raise ValueError(objective)
        return loss, info


def linear_probe(x: Tensor, y: Tensor) -> float:
    """Ridge probe fitted on the first half, accuracy on the second half."""
    x = x.reshape(x.shape[0], -1).float()
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    n = x.shape[0] // 2
    xa = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1)
    a = xa[:n]
    w = torch.linalg.solve(a.T @ a + 1.0 * torch.eye(a.shape[1], device=x.device), a.T @ y[:n])
    return float(((xa[n:] @ w > 0) == (y[n:] > 0)).float().mean())


def run_one(arch: str, objective: str, delay: int, seed: int, steps: int, device: str) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    m = Pathway(arch).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    t0 = time.time()
    curve = []
    for i in range(steps):
        loss, info = m.loss(make_batch(64, delay, device, gen), objective, delay)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 100.0)
        opt.step()
        if (i + 1) % max(1, steps // 5) == 0:
            curve.append({"step": i + 1, "loss": float(loss), **info})
    with torch.no_grad():
        test = make_batch(512, delay, device, torch.Generator(device=device).manual_seed(10_000 + seed))
        rec = m.forward(test)
        _, final = m.loss(test, objective, delay)
        probes = {}
        for name, x in rec.items():
            if name in ("prior_logits",):
                continue
            probes[name] = {"step0": linear_probe(x[:, 0], test["cue"]), "step_delay": linear_probe(x[:, delay], test["cue"])}
    return {"arch": arch, "objective": objective, "delay": delay, "seed": seed, "steps": steps,
            "params": sum(p.numel() for p in m.parameters()), "seconds": time.time() - t0, "final": final,
            "curve": curve, "probes": probes}


def grid(out: str, archs, objectives, delays, seeds, steps: int, jobs: int) -> None:
    os.makedirs(out, exist_ok=True)
    todo = [(a, o, d, s) for s in seeds for o in objectives for d in delays for a in archs
            if not os.path.exists(os.path.join(out, f"{a}_{o}_d{d}_s{s}.json"))]
    live: list = []
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    while todo or live:
        while todo and len(live) < jobs:
            a, o, d, s = todo.pop(0)
            path = os.path.join(out, f"{a}_{o}_d{d}_s{s}.json")
            p = subprocess.Popen([sys.executable, "-m", "superlearner.unified.diagnostics.memory_probe", "one", "--arch", a,
                                  "--objective", o, "--delay", str(d), "--seed", str(s), "--steps", str(steps),
                                  "--path", path], env=env, stdout=subprocess.DEVNULL, stderr=open(path + ".log", "w"))
            live.append(p)
        time.sleep(5)
        live = [p for p in live if p.poll() is None]


def summarize(out: str) -> None:
    rows = []
    for f in sorted(os.listdir(out)):
        if f.endswith(".json"):
            with open(os.path.join(out, f)) as fh:
                rows.append(json.load(fh))
    for r in sorted(rows, key=lambda r: (r["objective"], r["arch"], r["delay"], r["seed"])):
        p = r["probes"]
        stages = " ".join(f"{k}:{v['step0']:.2f}/{v['step_delay']:.2f}" for k, v in p.items())
        task = r["final"].get("acc", r["final"].get("reward_sign_acc"))
        print(f"{r['objective']:10s} {r['arch']:10s} d={r['delay']:<3d} s{r['seed']} task={task:.2f} | {stages}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["one", "grid", "summary"])
    ap.add_argument("--arch", default="rssm")
    ap.add_argument("--objective", default="world")
    ap.add_argument("--delay", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--path", default="")
    ap.add_argument("--out", default="/workspace/agentrs/runs/cam7/memory")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--delays", default="0,1,4,16,32,64")
    ap.add_argument("--archs", default="rssm,rssm_slots,gru,attention")
    ap.add_argument("--objectives", default="supervised,world")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.cmd == "one":
        res = run_one(a.arch, a.objective, a.delay, a.seed, a.steps, dev)
        tmp = a.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(res, fh)
        os.replace(tmp, a.path)
    elif a.cmd == "grid":
        grid(a.out, a.archs.split(","), a.objectives.split(","),
             [int(x) for x in a.delays.split(",")], [int(x) for x in a.seeds.split(",")], a.steps, a.jobs)
        summarize(a.out)
    else:
        summarize(a.out)
