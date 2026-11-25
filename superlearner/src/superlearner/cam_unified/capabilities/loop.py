"""Training and instrumented evaluation loops over the assembled CAM (workstream Y).

`train_agent` drives `CamUnifiedAgent` with the reference runtime (`runtime.run_steps`: act -> execute ->
advance_time(duration + think) -> collect_events -> observe(verified returns) -> update), so training uses exactly the
agent's own objectives (world / adapt / memory / policy through D's CombinedTrainer).

`Episode` replays the agent's per-decision computation step by step for evaluation (`CamUnifiedModel.decide` and
`observe_outcomes`, the same calls `CamUnifiedAgent.act` / `observe` make) and exposes the Decision objects and belief
update infos the capability measurements need (per-particle predictions, Kalman correction, imagined scores,
reasoning iterations). Rewards are the evaluator's score increments, as in the runtime.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from ..interfaces import ObservationBatch
from ..runtime import run_steps


def train_agent(stack, env, evaluator, rounds: int, update_every: int = 1, seeds: Optional[Sequence[int]] = None,
                log_every: int = 0, on_step: Optional[Callable] = None) -> dict:
    """`rounds` decision rounds of the agent on `env` (episodes restart when every row is done; each episode gets
    new derived seeds). Returns per-round mean verified return, update metrics and the compute account."""
    B = env.B
    hist: list = []
    t0 = time.time()

    def cb(i, row):
        hist.append(sum(row["rewards"]) / B)
        if log_every and (i + 1) % log_every == 0:
            print(f"  round {i + 1}: mean return/row {sum(hist[-log_every:]) / log_every:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if on_step is not None:
            on_step(i, row)

    rep = run_steps(stack.agent, env, rounds, evaluator=evaluator, update_every=update_every, seeds=seeds,
                    on_step=cb)
    ups = [u for u in rep.updates if u.get("updated")]
    keys = sorted({k for u in ups for k, v in u.get("losses", {}).items()})
    last = {k: [u["losses"][k] for u in ups[-10:] if k in u.get("losses", {})] for k in keys}
    return {"rounds": rounds, "B": B, "reward_per_round": hist, "n_updates": len(ups),
            "final_losses": {k: (sum(v) / len(v) if v else None) for k, v in last.items()},
            "compute": rep.compute.to_dict(), "dropped": rep.dropped}


@dataclass
class StepRecord:
    decision: object
    info: dict
    rewards: list
    results: list
    obs_before: list
    obs_after: list
    tools: list


@dataclass
class Episode:
    """Step-by-step evaluation episode of `model` on `env` with fresh per-row state (same calls as the agent)."""
    model: object
    env: object
    evaluator: object
    seeds: Sequence[int]
    greedy: bool = True
    generator: Optional[torch.Generator] = None
    think_days: Optional[float] = None
    budget_fn: Optional[Callable] = None
    steps: list = field(default_factory=list)

    def __post_init__(self):
        self.ob = self.env.reset(list(self.seeds))
        B = len(self.ob.env_ids)
        self.B = B
        self.state = self.model.init_state(B, seed=int(self.seeds[0]) if len(self.seeds) else 0)
        self.goals = self.env.goals()
        self.scores = [float(self.evaluator.evaluate_goal(b).score) for b in range(B)]
        self.obs_lists = [list(o) for o in self.ob.observations]
        self.now = [float(t) for t in self.ob.t]
        if self.think_days is None:
            self.think_days = self.model.cfg.think_s / 86400.0
        self.done = [False] * B

    def decide(self, tools: Optional[list] = None, **kw):
        """One decision for every row. `tools` restricts the offered tools (default: the environment's list), e.g.
        to force an intervention for a measurement (the pointer is masked to the offered tools)."""
        tools = self.env.available_tools() if tools is None else tools
        budget = self.budget_fn(self) if self.budget_fn is not None else None
        self.model.eval()
        dec = self.model.decide(self.state, self.obs_lists, self.now, tools, [g for g in self.goals],
                                budget=budget, greedy=kw.pop("greedy", self.greedy),
                                generator=kw.pop("generator", self.generator), **kw)
        return dec, tools

    def execute(self, dec, tools) -> StepRecord:
        results = self.env.execute(dec.actions)
        dur = [(r.duration if r is not None else 0.0) + self.think_days for r in results]
        adv = self.env.advance_time(dur)
        new_scores = [float(self.evaluator.evaluate_goal(b).score) for b in range(self.B)]
        rewards = [n - s for n, s in zip(new_scores, self.scores)]
        self.scores = new_scores
        nxt = [list(results[b].observations if results[b] is not None else []) + list(adv.observations[b])
               for b in range(self.B)]
        info = self.model.observe_outcomes(self.state, dec, results, nxt, rewards)
        rec = StepRecord(dec, info, rewards, results, self.obs_lists, nxt, tools)
        self.steps.append(rec)
        self.obs_lists = nxt
        self.now = [float(t) for t in adv.t]
        self.done = list(adv.done)
        return rec

    def step(self, **kw) -> StepRecord:
        dec, tools = self.decide(**kw)
        return self.execute(dec, tools)


def rebatch(agent) -> None:
    """Let the agent take a batch of different environments next (CamUnifiedAgent.reset refuses to re-batch once
    it has a state; interface change request filed): archive pending observations, drop the per-environment state
    and the fresh on-policy buffer (decisions of another batch must not enter the next policy update)."""
    agent.flush("reset")
    agent.state = None
    agent.fresh = []
    agent._last_fresh = {}


def mean(xs) -> float:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def ci95(xs) -> list:
    """Normal-approximation 95% interval of the mean (for reporting; seeds are few)."""
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(xs) < 2:
        return [mean(xs), mean(xs)]
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    h = 1.96 * sd / math.sqrt(len(xs))
    return [m - h, m + h]
