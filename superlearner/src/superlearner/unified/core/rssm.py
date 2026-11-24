"""The recurrent state-space core: deterministic state, stochastic latents, learned slot memory and belief.

One real step runs in this order, and no line reads information from a later step:

    1 encode the observation (done by the caller)     2 h_t = f(h_{t-1}, z_{t-1}, a_{t-1}, dt_t, m_{t-1})
    3 prior p(z_t | h_t)                              4 posterior q(z_t | h_t, E_t)
    5 sample z_t from the posterior                   6 read memory, 7 form the belief u_t
    8 write this step's event to memory for t+1
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import UnifiedConfig
from ..dist import categorical_kl, sample_categorical, unimix_logits
from ..encoders.observation import mlp
from ..interfaces import BeliefState


class SlotMemory(nn.Module):
    """Bounded differentiable memory ``[B, S, Dm]`` with attention reads and selective gated writes."""

    def __init__(self, cfg: UnifiedConfig, query_dim: int, event_dim: int):
        super().__init__()
        s, d = cfg.memory_slots, cfg.memory_dim
        self.slots, self.d = s, d
        self.init_slots = nn.Parameter(torch.randn(s, d) * 0.02)
        self.w_q = nn.Linear(query_dim, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.event = mlp(event_dim, d, d, layers=1)
        self.write_key = nn.Linear(d, d)
        self.write_k = nn.Linear(d, d)
        self.gate = nn.Linear(d + 1, 1)
        self.update = mlp(2 * d, d, d, layers=1)
        nn.init.constant_(self.gate.bias, -2.0)   # start by mostly preserving memory

    def initial(self, b: int) -> Tensor:
        return self.init_slots.unsqueeze(0).expand(b, -1, -1)

    def read(self, memory: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        q = self.w_q(query).unsqueeze(1)
        scores = (q * self.k(memory)).sum(-1) / math.sqrt(self.d)
        alpha = torch.softmax(scores, -1)
        return (alpha.unsqueeze(-1) * self.v(memory)).sum(1), alpha

    def write(self, memory: Tensor, event_in: Tensor, surprise: Tensor) -> tuple[Tensor, Tensor]:
        """``M' = (1 - s) * M + s * Write(M, event)`` where ``s`` is a per-slot share of a learned gate."""
        e = self.event(event_in)
        scores = (self.write_key(e).unsqueeze(1) * self.write_k(memory)).sum(-1) / math.sqrt(self.d)
        share = torch.softmax(scores, -1)
        gate = torch.sigmoid(self.gate(torch.cat([e, surprise.unsqueeze(-1)], -1)))
        s = (gate * share).unsqueeze(-1)
        new = self.update(torch.cat([memory, e.unsqueeze(1).expand(-1, memory.shape[1], -1)], -1))
        return (1 - s) * memory + s * new, s.squeeze(-1)


class RSSM(nn.Module):
    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.stoch_dim = cfg.stoch_vars * cfg.stoch_classes
        self.dyn_in = mlp(self.stoch_dim + cfg.action_dim + 2 + cfg.memory_dim, cfg.deter_dim, cfg.deter_dim, layers=1)
        self.gru = nn.GRUCell(cfg.deter_dim, cfg.deter_dim)
        self.h_norm = nn.LayerNorm(cfg.deter_dim)
        self.prior_net = mlp(cfg.deter_dim, d, self.stoch_dim, layers=2)
        self.post_net = mlp(cfg.deter_dim + d, d, self.stoch_dim, layers=2)
        self.query = mlp(cfg.deter_dim + self.stoch_dim + cfg.goal_dim, d, d, layers=1)
        self.memory = SlotMemory(cfg, d, cfg.deter_dim + self.stoch_dim)
        self.project = mlp(cfg.deter_dim + self.stoch_dim + cfg.memory_dim + cfg.goal_dim, d, d, layers=2)
        self.h0 = nn.Parameter(torch.zeros(cfg.deter_dim))

    # -------------------------------------------------------------------- pieces
    def _logits(self, net: nn.Module, x: Tensor) -> Tensor:
        raw = net(x).reshape(*x.shape[:-1], self.cfg.stoch_vars, self.cfg.stoch_classes)
        return unimix_logits(raw, self.cfg.unimix)

    def prior(self, h: Tensor) -> Tensor:
        return self._logits(self.prior_net, h)

    def posterior(self, h: Tensor, embed: Tensor) -> Tensor:
        return self._logits(self.post_net, torch.cat([h, embed], -1))

    def _transition(self, prev: BeliefState, action: Tensor, dt: Tensor) -> Tensor:
        if not self.cfg.use_recurrence:
            return self.h0.expand(prev.h.shape[0], -1)
        time_in = torch.stack([torch.log1p(dt), (dt > 0).float()], -1)
        x = torch.cat([prev.z.flatten(-2), action, time_in, prev.read], -1)
        return self.h_norm(self.gru(self.dyn_in(x), prev.h))

    def _finish(self, h: Tensor, z: Tensor, memory: Tensor, goal: Tensor, prior: Tensor, post: Tensor | None,
                surprise: Tensor) -> BeliefState:
        flat = z.flatten(-2)
        if self.cfg.use_slots:
            read, _ = self.memory.read(memory, self.query(torch.cat([h, flat, goal], -1)))
            # Dynamics-side events exclude the goal so that what is stored stays useful for other goals.
            new_memory, _ = self.memory.write(memory, torch.cat([h, flat], -1), surprise)
        else:
            read = torch.zeros(h.shape[0], self.cfg.memory_dim, device=h.device, dtype=h.dtype)
            new_memory = memory
        # No running feature normalisation here: imagined beliefs would be scaled by statistics of observed ones,
        # and held-out forecasting got worse with it (2 seeds, ledger 2026-11-24). The attention core keeps it.
        u = self.project(torch.cat([h, flat, read, goal], -1))
        return BeliefState(h, z, new_memory, read, u, prior, post)

    # -------------------------------------------------------------------- public steps
    def initial(self, b: int, device) -> BeliefState:
        cfg = self.cfg
        h = self.h0.expand(b, -1).to(device)
        z = torch.zeros(b, cfg.stoch_vars, cfg.stoch_classes, device=device)
        z[..., 0] = 1.0
        memory = self.memory.initial(b).to(device)
        read = torch.zeros(b, cfg.memory_dim, device=device)
        u = torch.zeros(b, cfg.d_model, device=device)
        logits = torch.zeros(b, cfg.stoch_vars, cfg.stoch_classes, device=device)
        return BeliefState(h, z, memory, read, u, logits, logits)

    def reset_where(self, state: BeliefState, first: Tensor) -> BeliefState:
        """Replace the state of rows where ``first`` is true with the learned initial state."""
        if not first.any():
            return state
        fresh = self.initial(first.shape[0], first.device)

        def pick(new: Tensor, old: Tensor) -> Tensor:
            m = first.bool().reshape(-1, *([1] * (old.dim() - 1)))
            return torch.where(m, new, old)
        return BeliefState(pick(fresh.h, state.h), pick(fresh.z, state.z), pick(fresh.memory, state.memory),
                           pick(fresh.read, state.read), pick(fresh.u, state.u),
                           pick(fresh.prior_logits, state.prior_logits),
                           None if state.post_logits is None else pick(fresh.post_logits, state.post_logits))

    def observe(self, prev: BeliefState, action: Tensor, dt: Tensor, embed: Tensor, goal: Tensor,
                sample: bool = True) -> BeliefState:
        """A real step: the posterior sees the actual observation embedding."""
        h = self._transition(prev, action, dt)
        prior = self.prior(h)
        post = self.posterior(h, embed)
        # A sampled one-hot can erase evidence that the posterior logits hold (measured by the memory probe);
        # "probs" carries the posterior distribution itself.
        z = torch.softmax(post, -1) if self.cfg.latent_mode == "probs" else sample_categorical(post, sample)
        surprise = categorical_kl(post.detach(), prior.detach())
        return self._finish(h, z, prev.memory, goal, prior, post, torch.log1p(surprise))

    def imagine(self, prev: BeliefState, action: Tensor, dt: Tensor, goal: Tensor, sample: bool = True) -> BeliefState:
        """An imagined step. It has no observation argument, so it cannot read a future state."""
        h = self._transition(prev, action, dt)
        prior = self.prior(h)
        z = torch.softmax(prior, -1) if self.cfg.latent_mode == "probs" else sample_categorical(prior, sample)
        surprise = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        return self._finish(h, z, prev.memory, goal, prior, None, surprise)
