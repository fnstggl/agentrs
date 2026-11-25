"""Shared test builders and minimal reference doubles for the CAM Unified contract (master-owned).

Workstreams use these to unit-test their component in isolation with exactly the contract's shapes. The doubles are
deliberately tiny and are **never** imported by production code (a test in workstream H enforces that) — they do not
count as implementations of anything in the completion manifest.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

from .config import CamConfig, make_config
from .interfaces import (Action, EventRecord, ExecutionResult, FastState, Likelihood, Observation, ObsKind,
                         OutcomeBatch, StepBatch, SequenceBatch, TokenKind, TokenSet, ToolSchema, new_id)


def tiny_config(**over) -> CamConfig:
    """A very small config for fast unit tests (same capabilities as dev)."""
    base = dict(n_layers=2, d_model=64, n_heads=4, n_state=8, d_particle=32, n_particles=8, bayes_feat=8,
                stoch_groups=8, stoch_classes=8, mem_key_dim=32, reason_layers=1, max_ctx_tokens=64, n_prefix=4,
                max_entities=32, n_options=4)
    base.update(over)
    return make_config("dev", **base)


def rand_tokens(B: int, N: int, D: int, kind: int = TokenKind.OBS_SEM, device=None, ragged: bool = True,
                seed: int = 0) -> TokenSet:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, D, generator=g)
    mask = torch.ones(B, N, dtype=torch.bool)
    if ragged and N > 1:
        for b in range(B):
            n = int(torch.randint(1, N + 1, (1,), generator=g))
            mask[b, n:] = False
    time = -torch.rand(B, N, generator=g) * 5.0
    ent = torch.randint(-1, 8, (B, N), generator=g)
    ts = TokenSet(x, mask, torch.full((B, N), int(kind), dtype=torch.long), time, ent,
                  [[f"ref{b}_{i}" for i in range(N)] for b in range(B)])
    return ts.to(device) if device is not None else ts


def rand_outcomes(B: int, Q: int, D: int, device=None, seed: int = 0, with_y: bool = True) -> OutcomeBatch:
    """Mixed-likelihood outcome queries: one of every Likelihood type when Q >= 6."""
    g = torch.Generator().manual_seed(seed)
    q = rand_tokens(B, Q, D, TokenKind.QUERY, ragged=False, seed=seed + 1)
    lik = torch.tensor([i % 6 for i in range(Q)]).repeat(B, 1)
    n_classes = torch.where(lik == Likelihood.CATEGORICAL, torch.tensor(5), torch.tensor(0))
    y = torch.randn(B, Q, generator=g)
    y = torch.where(lik == Likelihood.BERNOULLI, (y > 0).float(), y)
    y = torch.where(lik == Likelihood.CATEGORICAL, torch.randint(0, 5, (B, Q), generator=g).float(), y)
    y = torch.where(lik == Likelihood.COUNT, torch.randint(0, 20, (B, Q), generator=g).float(), y)
    y = torch.where(lik == Likelihood.SURVIVAL, torch.rand(B, Q, generator=g) * 10, y)
    censored = (lik == Likelihood.SURVIVAL) & (torch.rand(B, Q, generator=g) < 0.3)
    y_mask = torch.ones(B, Q, dtype=torch.bool) if with_y else torch.zeros(B, Q, dtype=torch.bool)
    channel = torch.arange(Q).repeat(B, 1)
    ob = OutcomeBatch(q, lik, n_classes, y, censored, y_mask, channel, [[f"k{i}" for i in range(Q)] for _ in range(B)])
    if device is not None:
        ob = OutcomeBatch(q.to(device), lik.to(device), n_classes.to(device), y.to(device), censored.to(device),
                          y_mask.to(device), channel.to(device), ob.keys)
    return ob


def zero_fast_state(B: int, cfg: CamConfig, C: int = 16, W: int | None = None, device=None) -> FastState:
    K, Dp, F = cfg.n_particles, cfg.d_particle, cfg.bayes_feat
    W = W or cfg.misspec_window
    dev = device
    return FastState(
        particles=torch.randn(B, K, Dp, device=dev) * 0.1,
        logw=torch.full((B, K), -float(torch.log(torch.tensor(float(K)))), device=dev),
        pscale=torch.ones(B, K, Dp, device=dev) * 0.1,
        bayes_mu=torch.zeros(B, C, F, device=dev, dtype=torch.float64),
        bayes_P=torch.eye(F, device=dev, dtype=torch.float64).expand(B, C, F, F).clone() * cfg.bayes_prior_var_init,
        bayes_noise=torch.full((B, C), cfg.bayes_obs_var_init, device=dev, dtype=torch.float64),
        bayes_n=torch.zeros(B, C, device=dev),
        surprise=torch.zeros(B, W, device=dev),
        steps=torch.zeros(B, dtype=torch.long, device=dev),
        channels=[{} for _ in range(B)])


def sample_tools(n: int = 4, seed: int = 0, randomize_names: bool = False) -> list:
    rng = random.Random(seed)
    base = [
        ToolSchema("search_web", "Search the simulated internet for pages matching a query.",
                   {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
        ToolSchema("set_price", "Set the monthly price of one of your products.",
                   {"type": "object", "properties": {"product": {"type": "string"}, "price": {"type": "number"}},
                    "required": ["product", "price"]}),
        ToolSchema("run_python", "Execute a Python program in the sandbox and return stdout/stderr.",
                   {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}),
        ToolSchema("send_message", "Send a message to another participant.",
                   {"type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["to", "text"]}),
        ToolSchema("hire", "Hire workers for a role.",
                   {"type": "object", "properties": {"role": {"type": "string", "enum": ["eng", "sales", "ops"]},
                                                      "count": {"type": "integer"}}, "required": ["role", "count"]}),
    ]
    out = base[:n]
    if randomize_names:
        out = [ToolSchema(f"tool_{rng.randrange(16**6):06x}", t.description, t.parameters) for t in out]
    return out


def sample_observations(n: int = 6, seed: int = 0, t: float = 0.0) -> list:
    rng = random.Random(seed)
    obs = [Observation(ObsKind.TEXT, "Customers in segment A complain that onboarding takes too long.", t=t,
                       source="forum", entity="seg_A", entity_type="customer_group"),
           Observation(ObsKind.NUMERIC, value=round(rng.uniform(1e3, 1e4), 2), unit="USD/month", entity="prod_1",
                       entity_type="product", t=t, source="dashboard"),
           Observation(ObsKind.CODE, "def handler(req):\n    return {'ok': True}\n", t=t, source="repo:svc",
                       entity="repo_1", entity_type="repository"),
           Observation(ObsKind.ERROR, "TimeoutError: upstream db did not respond in 2000 ms", t=t, source="deploy",
                       entity="svc_1", entity_type="service"),
           Observation(ObsKind.RELATION, "", entity="prod_1", target="seg_A", relation="sold_to", t=t, source="crm"),
           Observation(ObsKind.MESSAGE, "Can you deliver 40 units by Friday?", entity="cust_7", target="me", t=t,
                       source="inbox"),
           Observation(ObsKind.NUMERIC, value=rng.uniform(0, 1), unit="fraction", entity="svc_1", t=t,
                       source="monitor", meta={"metric": "error_rate"})]
    return obs[:n]


def sample_events(n: int = 10, env_id: str = "env0", seed: int = 0) -> list:
    rng = random.Random(seed)
    out, ep = [], new_id("ep_")
    for i in range(n):
        obs = sample_observations(3, seed=seed + i, t=float(i))
        act = Action("set_price", {"product": "prod_1", "price": round(rng.uniform(5, 50), 2)}, logp=-1.2)
        res = ExecutionResult(True, [Observation(ObsKind.TOOL_RESULT, '{"ok": true}', t=float(i), source="set_price")],
                              cost=1.0, duration=1.0, started_t=float(i), finished_t=float(i + 1))
        out.append(EventRecord(event_id=new_id("e_"), env_id=env_id, actor_id="actor0", episode_id=ep, step=i,
                               model_version=0, t_sim=float(i), observations=obs, action=act, result=res,
                               reward=rng.uniform(-1, 1), behavior_logp=-1.2))
    return out


def rand_sequence(B: int, T: int, cfg: CamConfig, N: int = 12, Q: int = 6, device=None, seed: int = 0) -> SequenceBatch:
    steps = []
    for t in range(T):
        steps.append(StepBatch(
            obs=rand_tokens(B, N, cfg.d_model, TokenKind.OBS_SEM, device=device, seed=seed + 7 * t),
            act=rand_tokens(B, 2, cfg.d_model, TokenKind.ACTION, device=device, ragged=False, seed=seed + 7 * t + 1),
            outcomes=rand_outcomes(B, Q, cfg.d_model, device=device, seed=seed + 7 * t + 2),
            dt=torch.ones(B, device=device), reward=torch.randn(B, device=device),
            done=torch.zeros(B, dtype=torch.bool, device=device), behavior_logp=torch.full((B,), -1.0, device=device),
            model_version=torch.zeros(B, dtype=torch.long, device=device),
            goal=rand_tokens(B, 2, cfg.d_model, TokenKind.GOAL, device=device, ragged=False, seed=seed + 7 * t + 3)))
    return SequenceBatch(steps)


class DoubleCore(nn.Module):
    """Reference double of UnifiedCoreP: one transformer encoder layer over [state; ctx]. Tests only."""

    def __init__(self, cfg: CamConfig):
        super().__init__()
        self.cfg = cfg
        self.z0 = nn.Parameter(torch.randn(cfg.n_state, cfg.d_model) * 0.02)
        self.kind = nn.Embedding(TokenKind.N_KINDS, cfg.d_model)
        self.layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, 2 * cfg.d_model, batch_first=True,
                                                norm_first=True, dropout=0.0)

    def init_state(self, B: int, device=None):
        return self.z0.unsqueeze(0).expand(B, -1, -1).to(device or self.z0.device).clone()

    def forward(self, Z_prev, ctx: TokenSet):
        B, S, _ = Z_prev.shape
        h = torch.cat([Z_prev + self.kind.weight[TokenKind.STATE], ctx.x + self.kind(ctx.kind)], 1)
        pad = torch.cat([torch.zeros(B, S, dtype=torch.bool, device=h.device), ~ctx.mask], 1)
        out = self.layer(h, src_key_padding_mask=pad)
        return out[:, :S], out[:, S:]


class DoubleOutcomeHead(nn.Module):
    """Reference double of OutcomeHeadP returning Gaussian distributions for every query. Tests only."""

    def __init__(self, cfg: CamConfig, cond_dim: int | None = None):
        super().__init__()
        D = cfg.d_model
        self.att = nn.MultiheadAttention(D, cfg.n_heads, batch_first=True)
        self.cond = nn.Linear(cond_dim or cfg.d_particle, 2 * D)
        self.out = nn.Linear(D, 2)

    def features(self, Z, outcomes: OutcomeBatch):
        f, _ = self.att(outcomes.q.x, Z, Z)
        return f + outcomes.q.x

    def dist(self, feat, outcomes: OutcomeBatch, cond=None):
        if cond is not None:
            s, b = self.cond(cond).chunk(2, -1)               # [B,K,D]
            feat = feat.unsqueeze(1) * (1 + s.unsqueeze(2)) + b.unsqueeze(2)   # [B,K,Q,D]
        mu, logv = self.out(feat).unbind(-1)
        return torch.distributions.Normal(mu, torch.exp(0.5 * logv.clamp(-8, 8)))
