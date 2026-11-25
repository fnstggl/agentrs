"""Minimal Stage-1 decision loop over any EnvAdapter (spec Part 21 Stage 1, Part 22 compute accounting).

    obs -> agent.act -> env.execute -> env.advance_time(duration + think)  (SMDP clock, architecture decision 5)
        -> env.collect_events(cursors) -> agent.observe(results, verified rewards, next obs) -> [agent.update()]

Rewards are verified return increments computed from the GoalEvaluator's IndependentGoalResult.score; only that
scalar reaches the agent (never `details` or metrics). Workstream G's OpenWorld harness replaces this loop for long
horizons (day-boundary goal checks, rate limits, multi-actor worlds); this one is the reference for any adapter.
"""
from __future__ import annotations

import resource
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import torch

from .interfaces import EnvAdapter, GoalEvaluator, Observation, ObservationBatch


@dataclass
class ComputeAccount:
    """Spec Part 22: GPU time, CPU time, inference tokens, environment steps, decisions, simulated time, memory."""
    wall_s: float = 0.0
    cpu_s: float = 0.0
    act_s: float = 0.0
    env_s: float = 0.0
    observe_s: float = 0.0
    update_s: float = 0.0
    gpu_s: float = 0.0                  # CUDA-event time of act/observe/update when the model is on a GPU
    env_steps: int = 0                  # environment-step slots (B per loop iteration)
    decisions: int = 0                  # executed (non-None) actions
    sim_days: float = 0.0               # simulated time advanced, summed over environments
    updates: int = 0
    peak_gpu_mem_bytes: int = 0
    max_rss_bytes: int = 0
    inference_tokens: Optional[int] = None    # semantic tokens encoded + generated, when the components count them
    archive_records: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunReport:
    steps: list = field(default_factory=list)       # per iteration: rewards, ok flags, tools, sim time
    updates: list = field(default_factory=list)     # agent.update() metrics
    returns: list = field(default_factory=list)     # [B] accumulated verified return
    goal_scores: list = field(default_factory=list)  # [B] last evaluator score (scalar only)
    compute: ComputeAccount = field(default_factory=ComputeAccount)


def _merge(seen_seq: int, *lists) -> tuple:
    """Concatenate observation lists, dropping environment events already delivered (seq <= last seen seq)."""
    out, top, here = [], seen_seq, set()
    for lst in lists:
        for o in lst or []:
            if o.seq is not None:
                if o.seq <= seen_seq or o.seq in here:
                    continue
                here.add(o.seq)
                top = max(top, o.seq)
            out.append(o)
    return out, top


def _initial_cursors(env, ob: ObservationBatch) -> list:
    if hasattr(env, "cursors") and callable(env.cursors):
        return list(env.cursors())
    return [max([o.seq for o in obs if o.seq is not None] or [0]) for obs in ob.observations]


def _tokens_used(agent) -> Optional[int]:
    tot, any_ = 0, False
    for comp in (getattr(agent.model, "tokenizer", None), getattr(agent.model, "action_generator", None),
                 getattr(agent.model, "semantic", None)):
        for attr in ("tokens_encoded", "tokens_generated"):
            v = getattr(comp, attr, None)
            if isinstance(v, int):
                tot += v
                any_ = True
    return tot if any_ else None


def run_steps(agent, env: EnvAdapter, n: int, evaluator: Optional[GoalEvaluator] = None,
              update_every: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
              think_s: Optional[float] = None, reset_on_done: bool = True,
              on_step: Optional[Callable[[int, dict], None]] = None) -> RunReport:
    """Run `n` decision rounds of `agent` (AgentP) on `env`. `update_every`: call agent.update() every k rounds
    (None/0 = never). The clock advances by each action's charged duration plus the think time (days)."""
    acc = ComputeAccount()
    rep = RunReport(compute=acc)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    dev = agent.device
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    think_days = (agent.config.think_s if think_s is None else think_s) / 86400.0

    def gpu_timed(fn):
        if not cuda:
            return fn(), 0.0
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize(dev)
        return out, s.elapsed_time(e) / 1000.0

    def start_episode():
        t = time.perf_counter()
        ob = env.reset(seeds)
        agent.reset(ob.env_ids)
        agent.set_goals(env.goals())
        acc.env_s += time.perf_counter() - t
        scores = [float(evaluator.evaluate_goal(b).score) for b in range(len(ob.env_ids))] if evaluator else None
        return ob, _initial_cursors(env, ob), scores

    ob, cursors, scores = start_episode()
    B = len(ob.env_ids)
    seen = [max([o.seq for o in obs if o.seq is not None] or [-1]) for obs in ob.observations]
    rep.returns = [0.0] * B
    for i in range(n):
        t = time.perf_counter()
        tools = env.available_tools()
        acc.env_s += time.perf_counter() - t
        t = time.perf_counter()
        actions, g = gpu_timed(lambda: agent.act(ob, tools))
        acc.act_s += time.perf_counter() - t
        acc.gpu_s += g
        t = time.perf_counter()
        results = env.execute(actions)
        durations = [((r.duration if r is not None else 0.0) + think_days) if a is not None else 0.0
                     for a, r in zip(actions, results)]
        adv = env.advance_time(durations)
        events, cursors = env.collect_events(cursors)
        acc.env_s += time.perf_counter() - t
        nxt = []
        for b in range(B):
            res_obs = results[b].observations if results[b] is not None else []
            merged, seen[b] = _merge(seen[b], res_obs, adv.observations[b], events[b])
            nxt.append(merged)
        if evaluator is not None:
            new_scores = [float(evaluator.evaluate_goal(b).score) for b in range(B)]
            rewards = [ns - s for ns, s in zip(new_scores, scores)]
            scores = new_scores
        else:
            rewards = [0.0] * B
        next_ob = ObservationBatch(list(ob.env_ids), nxt, list(adv.t), list(adv.done))
        t = time.perf_counter()
        _, g = gpu_timed(lambda: agent.observe(results, rewards, next_ob))
        acc.observe_s += time.perf_counter() - t
        acc.gpu_s += g
        acc.env_steps += B
        acc.decisions += sum(a is not None for a in actions)
        acc.sim_days += float(sum(durations))
        for b in range(B):
            rep.returns[b] += rewards[b]
        row = {"step": i, "rewards": rewards, "ok": [None if r is None else bool(r.ok) for r in results],
               "tools": [None if a is None else a.tool for a in actions], "t": list(adv.t),
               "sources": [None if a is None else a.source.value for a in actions]}
        if update_every and (i + 1) % update_every == 0:
            t = time.perf_counter()
            um, g = gpu_timed(agent.update)
            acc.update_s += time.perf_counter() - t
            acc.gpu_s += g
            if um.get("updated"):
                acc.updates += 1
            rep.updates.append(um)
            row["update"] = um.get("update_id")
        rep.steps.append(row)
        if on_step is not None:
            on_step(i, row)
        if all(next_ob.done) and reset_on_done:
            ob, cursors, scores = start_episode()
            seen = [max([o.seq for o in obs if o.seq is not None] or [-1]) for obs in ob.observations]
        else:
            ob = next_ob
    rep.goal_scores = list(scores) if scores is not None else []
    acc.wall_s = time.perf_counter() - wall0
    acc.cpu_s = time.process_time() - cpu0
    acc.max_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if cuda:
        acc.peak_gpu_mem_bytes = int(torch.cuda.max_memory_allocated(dev))
    acc.inference_tokens = _tokens_used(agent)
    acc.archive_records = agent.archive.count() if getattr(agent, "archive", None) is not None else None
    return rep
