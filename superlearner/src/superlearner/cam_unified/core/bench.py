"""Throughput / memory benchmark of the workstream-B stack at the dev, main and large scales (spec 1.1, A01).

One "step" = EntityEncoder over the observation tokens -> UnifiedCore over [obs; entities; relations; belief; memory;
goal; action] context -> belief tokens -> Reasoner (fixed J) -> scalar loss -> backward (+ AdamW step). The
"core" mode times the backbone alone. Run under the shared GPU lock:

    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock python -m superlearner.cam_unified.core.bench \
        --scales dev main large --B 8 --N 384
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import torch

from ..config import make_config
from ..interfaces import TokenKind, TokenSet
from .backbone import UnifiedCore
from .bayes import BayesLastLayer
from .beliefs import BeliefModule
from .blocks import count_parameters
from .entities import EntityEncoder
from .reasoning import Reasoner


def build_stack(scale: str, device="cuda", grad_checkpoint: bool = False, bf16: bool = True, **over) -> dict:
    cfg = make_config(scale, grad_checkpoint=grad_checkpoint, **over)
    core = UnifiedCore(cfg, autocast=bf16).to(device)
    stack = {
        "cfg": cfg, "core": core, "reasoner": Reasoner(cfg, core).to(device),
        "entities": EntityEncoder(cfg, autocast=bf16).to(device), "beliefs": BeliefModule(cfg, autocast=bf16).to(device),
        "bayes": BayesLastLayer(cfg).to(device),
    }
    return stack


def parameter_report(stack: dict) -> dict:
    core = stack["core"]
    rep = {"core": core.count_parameters(), "reasoner_own": count_parameters(stack["reasoner"]),
           "entity_encoder": count_parameters(stack["entities"]), "belief_module": count_parameters(stack["beliefs"]),
           "bayes_last_layer": count_parameters(stack["bayes"])}
    rep["workstream_B_total"] = (rep["core"]["total"] + rep["reasoner_own"] + rep["entity_encoder"]
                                 + rep["belief_module"] + rep["bayes_last_layer"])
    return rep


def synthetic_context(cfg, B: int, N: int, device, seed: int = 0) -> tuple:
    """A realistic per-step working context of N tokens: observation tokens with entity bindings (60%), memory,
    goal and action tokens; the entity encoder adds ENTITY/RELATION tokens and the belief K+1 tokens are appended by
    the step, so the observation share is sized to make the total ~N."""
    g = torch.Generator().manual_seed(seed)
    D = cfg.d_model
    n_belief = cfg.n_particles + 1
    n_goal, n_act, n_mem = 4, 4, 32
    n_ent = min(48, cfg.max_entities)
    n_obs = max(8, N - n_belief - n_goal - n_act - n_mem - n_ent - n_ent)

    def ts(n, kind, ent=False):
        x = torch.randn(B, n, D, generator=g)
        e = torch.randint(0, n_ent, (B, n), generator=g) if ent else torch.full((B, n), -1)
        e2 = torch.where(torch.rand(B, n, generator=g) < 0.15, torch.randint(0, n_ent, (B, n), generator=g),
                         torch.full((B, n), -1)) if ent else None
        return TokenSet(x, torch.ones(B, n, dtype=torch.bool), torch.full((B, n), int(kind)),
                        -torch.rand(B, n, generator=g) * 30, e, None, e2).to(device)
    obs = ts(n_obs, TokenKind.OBS_SEM, ent=True)
    rest = TokenSet.cat(ts(n_mem, TokenKind.MEMORY), ts(n_goal, TokenKind.GOAL), ts(n_act, TokenKind.ACTION))
    rest.entity2 = None
    return obs, rest


def _step(stack, obs, rest, Z0, fast, J):
    ents = stack["entities"](obs)
    btok = stack["beliefs"].tokens(fast)
    obs_flat = TokenSet(obs.x, obs.mask, obs.kind, obs.time, obs.entity, None, obs.entity2)
    ctx = TokenSet.cat(obs_flat, ents, btok, rest)
    Z, ctx_out = stack["core"](Z0, ctx)
    R, info = stack["reasoner"](Z, TokenSet.cat(btok, rest), n_iters=J)
    return R.float().pow(2).mean() + ctx_out.float().pow(2).mean() * 0.1, ctx.N


def benchmark(scale: str, B: int = 8, N: int = 384, steps: int = 10, warmup: int = 3, bf16: bool = True,
              grad_checkpoint: bool = False, J: int = 4, mode: str = "step", optimizer: bool = True,
              device: str = "cuda") -> dict:
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    stack = build_stack(scale, device, grad_checkpoint, bf16)
    cfg, core = stack["cfg"], stack["core"]
    params = parameter_report(stack)
    mods = [stack[k] for k in ("core", "reasoner", "entities", "beliefs")]
    ps = [p for m in mods for p in m.parameters()]
    seen, plist = set(), []
    for p in ps:
        if id(p) not in seen:
            seen.add(id(p))
            plist.append(p)
    opt = torch.optim.AdamW(plist, lr=1e-4, fused=True) if optimizer else None
    obs, rest = synthetic_context(cfg, B, N, device)

    def one(train: bool):
        Z0 = core.init_state(B, device)                      # fresh graph per step (learned initial state)
        fast = stack["beliefs"].init_fast(B, device)
        if mode == "core":
            ctx = TokenSet.cat(TokenSet(obs.x, obs.mask, obs.kind, obs.time, obs.entity, None, obs.entity2), rest)
            Z, co = core(Z0, ctx)
            loss, n_ctx = Z.pow(2).mean() + 0.1 * co.pow(2).mean(), ctx.N
        else:
            loss, n_ctx = _step(stack, obs, rest, Z0, fast, J)
        if train:
            loss.backward()
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)
        return n_ctx

    res = {"scale": scale, "B": B, "N_target": N, "mode": mode, "bf16": bf16, "grad_checkpoint": grad_checkpoint,
           "reason_iters": J, "params": params}
    for train in (False, True):
        for m in mods:
            m.train(train)
        ctxm = torch.enable_grad() if train else torch.no_grad()
        with ctxm:
            for _ in range(warmup):
                n_ctx = one(train)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(steps):
                n_ctx = one(train)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / steps
        key = "train" if train else "forward"
        res[key] = {"ms_per_step": 1000 * dt, "tokens_per_s": B * (cfg.n_state + n_ctx) / dt,
                    "peak_mem_GB": (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 30}
        res["ctx_tokens"] = n_ctx
    del stack, opt
    torch.cuda.empty_cache()
    return res


def main(argv: Optional[list] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["dev", "main", "large"])
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--N", type=int, default=384)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--J", type=int, default=4)
    ap.add_argument("--mode", default="step", choices=["step", "core"])
    ap.add_argument("--checkpoint", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = []
    for s in a.scales:
        ck = {"on": True, "off": False}.get(a.checkpoint, s == "large")
        r = benchmark(s, a.B, a.N, a.steps, bf16=not a.fp32, grad_checkpoint=ck, J=a.J, mode=a.mode)
        print(json.dumps({k: v for k, v in r.items() if k != "params"}), flush=True)
        print(json.dumps({"scale": s, "params": r["params"]}), flush=True)
        out.append(r)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    main()
