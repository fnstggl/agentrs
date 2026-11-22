"""The architecture-neutral learner contract.

Crucible compares learning systems; it must not assume what one looks like inside. A learner is anything that
implements the seven methods below. Model-free, model-based, recurrent, memory-augmented, search-based and
language-model learners all fit, because the contract speaks only about *when* each kind of adaptation may happen:

    fast adaptation      inside `act`, carried by whatever context the learner keeps between calls;
                         `reset_context` clears it and nothing else
    parametric learning  inside `update`, from experience delivered by `observe`
    frozen evaluation    after `freeze(True)`: no weight change, no writing to and no reading from stored
                         training experience (a learner with a retrieval memory must disable it here)

The three timescales the benchmark separates fall out of these calls:

* within-episode adaptation: reward late in an episode minus reward early, context intact, weights frozen;
* parametric internalisation: frozen evaluation on fresh worlds after `reset_context`, checkpoint before
  training versus checkpoint after;
* meta-learning: the within-episode adaptation curve of the trained checkpoint versus the untrained one.

Numeric environments (`research.envs`) pad every hidden mechanism to one observation and action size:

    observation (float32, EnvSpec.obs_dim) =
        gauges[max_gauges]            symmetric-log readings, 0 where masked
        gauge_mask[max_gauges]
        last_action[max_levers]       the settings in force, scaled to [-1, 1]
        lever_mask[max_levers]
        last_reward, last_revenue, last_cost    in units of the task's public money scale per tick
        funds                         the operator's balance / its starting balance
        time                          tick / horizon
    action (float32, max_levers) in [-1, 1], mapped linearly onto each lever's public range; masked entries
    are ignored.

Slots are assigned to levers and gauges in the public (seeded, role-free) order of the mechanism, so a position
means nothing across worlds. Language-model learners use the kernel world directly through
`crucible.agents.base.Policy` and the `rig.*` tools; `research.bridge` runs any numeric learner through those
same tool calls, so every architecture has identical permissions and information.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

MAX_LEVERS = 12
MAX_GAUGES = 16
N_SCALARS = 5          # last_reward, last_revenue, last_cost, funds, time


@dataclass(frozen=True)
class EnvSpec:
    max_levers: int = MAX_LEVERS
    max_gauges: int = MAX_GAUGES
    horizon: int = 120

    @property
    def obs_dim(self) -> int:
        return 2 * self.max_gauges + 2 * self.max_levers + N_SCALARS

    @property
    def act_dim(self) -> int:
        return self.max_levers

    # slices into an observation vector
    @property
    def gauges(self) -> slice:
        return slice(0, self.max_gauges)

    @property
    def gauge_mask(self) -> slice:
        return slice(self.max_gauges, 2 * self.max_gauges)

    @property
    def last_action(self) -> slice:
        return slice(2 * self.max_gauges, 2 * self.max_gauges + self.max_levers)

    @property
    def lever_mask(self) -> slice:
        return slice(2 * self.max_gauges + self.max_levers, 2 * self.max_gauges + 2 * self.max_levers)

    @property
    def scalars(self) -> slice:
        return slice(2 * self.max_gauges + 2 * self.max_levers, self.obs_dim)


@dataclass
class Usage:
    """Cumulative resources a learner consumed. Reported separately, never folded into one score."""
    env_steps: int = 0
    updates: int = 0                 # optimiser steps (weight changes)
    train_cpu_s: float = 0.0
    train_gpu_s: float = 0.0
    infer_cpu_s: float = 0.0
    infer_gpu_s: float = 0.0
    infer_tokens: int = 0            # language-model learners: prompt + completion tokens
    search_evals: int = 0            # inference-time search: simulated or model evaluations per decision, summed
    params: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Learner:
    """Base class; subclasses override what they need. All arrays are batched over parallel environments."""
    name = "learner"
    kind = "abstract"     # random | scripted | model_free | model_based | llm | search | ...

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu"):
        self.spec = spec
        self.seed = int(seed)
        self.device = device
        self.frozen = False
        self._usage = Usage()

    # ---- fast adaptation ---------------------------------------------------------------------------
    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        """Start of an episode for `n` parallel environments (or only those where `which` is True). Clears
        recurrent state and episodic memory. Never touches weights, optimiser state or long-term buffers."""

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """`obs` is [n, obs_dim]; returns [n, act_dim] in [-1, 1]."""
        raise NotImplementedError

    # ---- parametric learning -----------------------------------------------------------------------
    def observe(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray,
                next_obs: np.ndarray) -> None:
        """One environment step of experience. Not called while frozen."""

    def update(self) -> dict:
        """Apply pending weight updates from stored experience. Returns counters, at least {"updates": int}."""
        return {"updates": 0}

    def freeze(self, frozen: bool = True) -> None:
        """Frozen: `update` is a no-op and stored training experience is neither written nor read."""
        self.frozen = bool(frozen)

    # ---- checkpoints -------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Write everything needed to restore the learner's long-term state (weights; optionally optimiser).
        Must not include episodic context."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    # ---- accounting --------------------------------------------------------------------------------
    def usage(self) -> Usage:
        return self._usage
