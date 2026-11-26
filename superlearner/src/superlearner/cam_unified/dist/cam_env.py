"""Real Crucible OpenWorld environments for distributed actors (spec 14.1 "independent environments"; Part 15).

`make_openworld_env` builds, inside an actor process, one `OpenWorldAdapter` over workstream G's process-isolated
operator (`envs.openworld.open_world_isolated`: the WorldServer, the worlds and the `OpenWorldEvaluator` live in a
separate spawned process; only contract records and scalar verified rewards cross the pipe — the hidden-state
firewall) and wraps it in `OpenWorldActorEnv`, which

* draws the world seed of every episode from the CAM Unified **development** range (experiment register:
  ``seed_base`` = 100,000 by default, confirmation seeds 200,000+ are never used here): episode k of an actor whose
  environment seed is s uses world seed ``seed_base + (s * 7919 + k) % seed_span`` (``seed_span`` = 100,000 keeps every
  seed below 200,000), so actors face independent worlds and every episode is a new world;
* exposes `take_rewards` (verified return increments of the operator-side evaluator, the learner's only
  evaluation-derived input) and `evaluate_goal` (the independent result; its ``details`` stay in the actor process'
  harness code and are never handed to the policy);
* returns the initial event cursors (the largest delivered event ``seq`` of the reset observations) for lossless
  `collect_events`.

The task is the dev open-economy objective `envs.protocols.ow1.open_economy_task` (ow_tiny, 800 consumers, 10,000
cents over a 3-day window + 100 requests, sustained 2 daily checks, 3,000,000 cents endowment, ``days`` horizon).
"""
from __future__ import annotations

from typing import Optional, Sequence

DEV_SEED_BASE = 100_000
DEV_SEED_SPAN = 100_000


class OpenWorldActorEnv:
    """EnvAdapter facade with dev-range world seeds and the evaluator's verified rewards (see module docstring)."""

    def __init__(self, seed: int = 0, days: int = 8, seed_base: int = DEV_SEED_BASE, seed_span: int = DEV_SEED_SPAN,
                 max_wait_s: int = 86400, task: str = "open_economy"):
        if seed_base < DEV_SEED_BASE or seed_base + seed_span > 2 * DEV_SEED_BASE:
            raise ValueError("distributed runs use the development seed range [100000, 200000) only")
        from ..envs.openworld import open_world_isolated
        from ..envs.protocols.ow1 import open_economy_task
        if task != "open_economy":
            raise ValueError(f"unknown task {task!r}")
        self.seed = int(seed)
        self.seed_base, self.seed_span = int(seed_base), int(seed_span)
        self.days = int(days)
        self.episode = 0
        self.world_seeds: list = []
        first = self._world_seed(0)
        self.adapter, self.operator, self.evaluator = open_world_isolated(
            [open_economy_task(seed=first, days=self.days)], evaluator={}, max_wait_s=max_wait_s, seed=self.seed)
        self.n = self.adapter.n
        self.B = self.n

    def _world_seed(self, k: int) -> int:
        return self.seed_base + (self.seed * 7919 + k) % self.seed_span

    # ------------------------------------------------------------------ EnvAdapter
    def reset(self, seeds: Optional[Sequence[int]] = None):
        ws = [self._world_seed(self.episode)]
        self.episode += 1
        self.world_seeds.append(ws[0])
        ob = self.adapter.reset(ws)
        self._cursors = [max([o.seq for o in obs if o.seq is not None] or [0]) for obs in ob.observations]
        return ob

    def initial_cursors(self) -> list:
        return list(self._cursors)

    def observe(self):
        return self.adapter.observe()

    def available_tools(self) -> list:
        return self.adapter.available_tools()

    def execute(self, actions):
        return self.adapter.execute(actions)

    def collect_events(self, since):
        return self.adapter.collect_events(since)

    def advance_time(self, duration):
        return self.adapter.advance_time(duration)

    def goals(self) -> list:
        return self.adapter.goals()

    def close(self) -> None:
        self.adapter.close()

    # ------------------------------------------------------------------ evaluator side (harness only)
    def take_rewards(self) -> list:
        return [float(r) for r in self.evaluator.take_rewards()]

    def evaluate_goal(self, b: int):
        return self.evaluator.evaluate_goal(b)


def make_openworld_env(seed: int = 0, **kw) -> tuple:
    """Actor env factory: (env, evaluator) with the evaluator being the env itself (take_rewards/evaluate_goal)."""
    env = OpenWorldActorEnv(seed=seed, **kw)
    return env, env
