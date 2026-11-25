"""Builders shared by the workstream-E tests (importable from test subprocesses too)."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from superlearner.cam_unified.interfaces import (Action, EventRecord, ExecutionResult, Observation, ObsKind,
                                                 Provenance)
from superlearner.cam_unified.memory.retrieval import Retriever


def light_events(n: int, env: str = "env0", episode_len: int = 100, seed: int = 0, start: int = 0,
                 family: str | None = None, actor: str = "actor0", content: str = "") -> list:
    """Lightweight event records: one numeric observation, one small action, an ok/failed result."""
    rng = random.Random(seed)
    out = []
    for i in range(start, start + n):
        ep, step = divmod(i, episode_len)
        ok = rng.random() > 0.1
        obs = [Observation(ObsKind.NUMERIC, content, value=round(rng.uniform(0, 100), 3), unit="cents",
                           entity=f"ent{i % 17}", t=float(i), source="meter")]
        out.append(EventRecord(
            event_id=f"{env}-{seed}-{i:09d}", env_id=env, actor_id=actor, episode_id=f"{env}-ep{ep}", step=step,
            model_version=i // 1000, t_sim=float(i) / 24.0, observations=obs,
            action=Action("set_lever", {"value": round(rng.uniform(-1, 1), 3)}, action_id=f"a{i}"),
            result=ExecutionResult(ok, [], None if ok else "E_RANGE", cost=1.0, duration=1 / 24),
            reward=rng.uniform(-1, 1), done=step == episode_len - 1, behavior_logp=-1.0,
            extra={"family": family} if family else {}, t_wall=1.7e9 + i))
    return out


class ToyModel(nn.Module):
    """A small model holding a real Retriever (parameters + buffers) and a core-like MLP."""

    def __init__(self, cfg):
        super().__init__()
        D = cfg.d_model
        self.core = nn.Sequential(nn.Linear(D, 2 * D), nn.ReLU(), nn.Linear(2 * D, D))
        self.head = nn.Linear(D, 3)
        self.retriever = Retriever(cfg)
        self.register_buffer("running", torch.zeros(D))

    def forward(self, x):
        h = self.core(x) + self.running
        return self.head(h)


class ToyAgent:
    """Follows the CheckpointOps attribute contract."""

    def __init__(self, cfg, seed: int = 0):
        torch.manual_seed(seed)
        self.config = cfg
        self.model = ToyModel(cfg)
        self.optimizers = self.make_optimizers()
        self.schedulers = self.make_schedulers()
        self.fast_state = self.init_fast_state()
        self.latent_state = self.init_latent_state()
        self.registries = []
        self.archive = None
        self.index = None
        self.replay = None
        self.consolidator = None
        self.rng = {"torch": torch.Generator().manual_seed(seed + 1), "np": np.random.default_rng(seed + 2),
                    "py": random.Random(seed + 3)}
        self.version = 0
        self.counters = {}

    def make_optimizers(self):
        ps = [p for p in self.model.parameters() if p.requires_grad]
        return {"core": torch.optim.AdamW(ps, lr=1e-3)}

    def make_schedulers(self):
        return {"core": torch.optim.lr_scheduler.StepLR(self.optimizers["core"], step_size=3, gamma=0.5)}

    def init_fast_state(self):
        from superlearner.cam_unified.testing import zero_fast_state
        g = torch.random.get_rng_state()
        torch.manual_seed(1234)
        fs = zero_fast_state(2, self.config, C=4)
        torch.random.set_rng_state(g)
        return fs

    def init_latent_state(self):
        return torch.zeros(2, self.config.n_state, self.config.d_model)

    def train_steps(self, n: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        opt = self.optimizers["core"]
        for _ in range(n):
            x = torch.randn(16, self.config.d_model, generator=g)
            y = torch.randn(16, 3, generator=g)
            loss = ((self.model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.schedulers["core"].step()
            self.version += 1
        with torch.no_grad():
            self.model.running.add_(0.5)
            self.fast_state.particles.add_(1.0)
            self.fast_state.steps.add_(n)
            self.fast_state.channels[0]["revenue@p1"] = 0
            self.latent_state.add_(2.0)
        self.counters["steps"] = self.counters.get("steps", 0) + n


def build_agent(cfg_dict: dict, seed: int = 99) -> ToyAgent:
    from superlearner.cam_unified.memory.checkpoint import config_from_dict
    return ToyAgent(config_from_dict(cfg_dict), seed=seed)


def probe_input(cfg, n: int = 5) -> torch.Tensor:
    return torch.randn(n, cfg.d_model, generator=torch.Generator().manual_seed(7))
