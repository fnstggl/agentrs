"""Reward contract: the training objective derived from an environment's verified task contract.

The environment specifies and verifies the task (``CamEnv.contract()`` / ``task_status()``): final
success, deadline (in its own clock: weeks, rounds, steps), resources, hard constraints whose violation
is an unacceptable terminal state, milestones and failure causes. This module turns that contract into
per-transition training signals without any environment-specific rule:

``mode="contract"`` (semi-Markov):
  * discount per unit of *simulated* time: an action that consumes ``dt`` clock units is discounted
    by ``gamma ** dt`` (so one 4-week wait and four 1-week waits are the same decision problem);
  * time cost accrues per clock unit over the action's duration, discounted inside the action; a
    failed episode additionally pays the cost of the rest of the horizon, so every failure costs the
    same total time cost regardless of when it happens — ending early is never cheaper than trying;
  * potential-based shaping on verified progress with the same semi-Markov discount and
    Φ(terminal) = 0 (policy-invariant; it only speeds up credit assignment);
  * +``success`` for the verified final success;
  * a separate *constraint cost* channel (1 when the episode ends by violating a hard constraint)
    that a Lagrangian multiplier converts into reward (``lam``; see ``train.py``).

``mode="legacy"`` reproduces the original CAM reward (per-action discount and step cost) and
``mode="legacy_failcost"`` the first product fix, so that the reward audit can be run on them.
"""
from __future__ import annotations

from dataclasses import dataclass

from agentrs.cam.envs.base import Contract, Status

MODES = ("legacy", "legacy_failcost", "contract")


@dataclass
class RewardSpec:
    mode: str = "contract"
    gamma: float = 0.99        # per clock unit (contract) / per action (legacy modes)
    shaping: float = 0.5
    time_cost: float = 0.005   # per clock unit (contract) / per action (legacy modes)
    success: float = 1.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"reward mode must be one of {MODES}, got {self.mode!r}")


@dataclass
class Transition:
    reward: float
    discount: float            # multiplies the next state's value (gamma ** dt in contract mode)
    dt: float                  # clock units consumed by the action
    violation: float           # 1.0 if this transition ended the episode by violating a hard constraint


class RewardContract:
    def __init__(self, spec: RewardSpec | None = None):
        self.spec = spec or RewardSpec()

    def _ann(self, dt: float) -> float:
        """Discounted sum of a unit cost rate over ``dt`` clock units: sum_{k<dt} gamma^k (continuous)."""
        g = self.spec.gamma
        return dt if g >= 1.0 else (1 - g ** dt) / (1 - g)

    def transition(self, c: Contract, s0: Status, s1: Status, done: bool, success: bool, lam: float = 0.0,
                   n_steps: int = 0, max_steps: int = 0) -> Transition:
        sp = self.spec
        viol = 1.0 if (done and s1.violated) else 0.0
        if sp.mode != "contract":
            r = sp.shaping * (sp.gamma * (0.0 if done else s1.progress) - s0.progress) - sp.time_cost
            if done:
                r += sp.success if success else 0.0
                if sp.mode == "legacy_failcost" and not success:
                    r -= sp.time_cost * max(0, max_steps - n_steps)
            return Transition(reward=r - lam * viol, discount=sp.gamma, dt=1.0, violation=viol)
        dt = max(0.0, s1.clock - s0.clock)
        disc = sp.gamma ** dt
        r = sp.shaping * (disc * (0.0 if done else s1.progress) - s0.progress) - sp.time_cost * self._ann(dt)
        if done:
            if success:
                r += sp.success
            else:  # the rest of the horizon is still paid: failing early is never cheaper than failing late
                r -= sp.time_cost * disc * self._ann(max(0.0, c.horizon - s1.clock))
        return Transition(reward=r - lam * viol, discount=disc, dt=dt, violation=viol)
