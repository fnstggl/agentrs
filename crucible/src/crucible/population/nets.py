"""Neural networks of the population's learning participants (torch; imported only when a learning class exists).

* `AdaptiveNet` (class 2, contract `AdaptivePolicy`): an observation encoder, a GRU cell that turns the
  individual's previous memory and its current observation into its new memory, and a candidate scorer that
  predicts the utility of visiting each candidate resource from (encoded observation, memory, candidate features).
  It is a neural contextual bandit with recurrent personal state; about 0.1-2M parameters depending on `hidden`.
* `StrategicMLP` and `StrategicGRU` (class 3): two architectures of strategic policies (categorical heads over the
  entrepreneur's choices plus a value head), 2-20M parameters at the default sizes.

Determinism: parameters are initialised from a numpy generator (never torch's global RNG); updates run on CPU with
one thread and deterministic algorithms inside `deterministic()`, so a learning step is a pure function of the
weights and the examples - replay rebuilds the same weights (`weights_hash` is checked against the logged one).
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch import nn


@contextlib.contextmanager
def deterministic():
    n = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(det)
        torch.set_num_threads(n)


def init_params(module: nn.Module, rng: np.random.Generator) -> None:
    """Glorot-uniform weights and zero biases from a numpy generator, in sorted parameter-name order."""
    with torch.no_grad():
        for name, p in sorted(module.named_parameters()):
            if p.dim() >= 2:
                fan_in, fan_out = p.shape[1], p.shape[0]
                a = float(np.sqrt(6.0 / (fan_in + fan_out)))
                p.copy_(torch.from_numpy(rng.uniform(-a, a, tuple(p.shape)).astype(np.float32)))
            else:
                p.zero_()


def n_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def state_arrays(module: nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float32).copy() for k, v in sorted(module.state_dict().items())}


def load_arrays(module: nn.Module, arrays: dict[str, np.ndarray]) -> None:
    sd = {k: torch.from_numpy(np.array(arrays[k], np.float32)) for k in module.state_dict()}
    module.load_state_dict(sd)


class AdaptiveNet(nn.Module):
    def __init__(self, obs_dim: int, cand_dim: int, mem_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim, self.cand_dim, self.mem_dim, self.hidden = obs_dim, cand_dim, mem_dim, hidden
        self.enc = nn.Linear(obs_dim, hidden)
        self.gru = nn.GRUCell(hidden, mem_dim)
        self.s1 = nn.Linear(hidden + mem_dim + cand_dim, hidden)
        self.s2 = nn.Linear(hidden, hidden)
        self.s3 = nn.Linear(hidden, 1)

    def memory(self, obs: torch.Tensor, mem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.enc(obs))
        return h, self.gru(h, mem)

    def score(self, h: torch.Tensor, mem: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """h [B, hidden], mem [B, M], cand [B, K, C] -> predicted utility [B, K]."""
        B, K, _ = cand.shape
        x = torch.cat([h[:, None, :].expand(B, K, h.shape[1]), mem[:, None, :].expand(B, K, mem.shape[1]), cand], -1)
        return self.s3(torch.relu(self.s2(torch.relu(self.s1(x))))).squeeze(-1)

    def forward(self, obs, mem, cand):
        h, m2 = self.memory(obs, mem)
        return self.score(h, m2, cand), m2


class StrategicMLP(nn.Module):
    """Feed-forward strategic policy: obs -> 2 hidden layers -> one categorical head per decision + value."""
    arch = "mlp"

    def __init__(self, obs_dim: int, heads: dict[str, int], hidden: int = 1024):
        super().__init__()
        self.heads_spec = dict(heads)
        self.l1 = nn.Linear(obs_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, v) for k, v in sorted(heads.items())})
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs, hist=None):
        z = torch.relu(self.l3(torch.relu(self.l2(torch.relu(self.l1(obs))))))
        return {k: h(z) for k, h in self.heads.items()}, self.value(z).squeeze(-1)


class StrategicGRU(nn.Module):
    """Recurrent strategic policy: a GRU over the participant's sequence of observations (its decision history)."""
    arch = "gru"

    def __init__(self, obs_dim: int, heads: dict[str, int], hidden: int = 768):
        super().__init__()
        self.heads_spec = dict(heads)
        self.inp = nn.Linear(obs_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, hidden)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, v) for k, v in sorted(heads.items())})
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs, hist=None):
        """obs [B, obs_dim]; hist [B, T, obs_dim] (previous observations, oldest first) or None."""
        seq = obs[:, None, :] if hist is None else torch.cat([hist, obs[:, None, :]], 1)
        y, _ = self.gru(torch.relu(self.inp(seq)))
        z = torch.relu(self.out(y[:, -1]))
        return {k: h(z) for k, h in self.heads.items()}, self.value(z).squeeze(-1)
