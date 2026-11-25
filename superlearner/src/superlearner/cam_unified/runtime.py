"""Minimal Stage-1 decision loop over any EnvAdapter (spec Part 21 Stage 1, Part 22 compute accounting).

    obs -> agent.act -> env.execute -> env.advance_time(duration + think)  (SMDP clock, architecture decision 5)
        -> env.collect_events(cursors) -> agent.observe(results, verified rewards, next obs) -> [agent.update()]

Rewards are verified return increments computed from the GoalEvaluator's IndependentGoalResult.score; only that
scalar reaches the agent (never `details` or metrics). Workstream G's OpenWorld harness replaces this loop for long
horizons (day-boundary goal checks, rate limits, multi-actor worlds); this one is the reference for any adapter.

Event cursors: the sentinel for "from the episode start" is -1 (exclusive `since`, so an event with seq 0 is
delivered). Each episode gets its own seeds (`episode_seeds(e)`, default derived from the run's seeds), so episodes
after the first are not replays of the same world.
"""
from __future__ import annotations

import resource
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import torch

from .interfaces import EnvAdapter, GoalEvaluator, ObservationBatch

FROM_START = -1                        # event cursor sentinel: nothing delivered yet (exclusive since)
SEED_STRIDE = 1_000_003


@dataclass
class ComputeAccount:
    """Spec Part 22: GPU time, CPU time, inference tokens, environment steps, decisions, simulated time, memory.

    gpu_span_s: CUDA-event span of act/observe/update (includes GPU idle time while the CPU launches work: an upper
    bound, close to wall time for small models). gpu_busy_s: summed CUDA kernel time from torch.profiler (only when
    run_steps(profile_gpu=True); None otherwise)."""
    wall_s: float = 0.0
    cpu_s: float = 0.0
    act_s: float = 0.0
    env_s: float = 0.0
    observe_s: float = 0.0
    update_s: float = 0.0
    gpu_span_s: float = 0.0
    gpu_busy_s: Optional[float] = None
    gpu_kernels: Optional[int] = None
    env_steps: int = 0                  # environment-step slots (B per loop iteration)
    decisions: int = 0                  # executed (non-None) actions
    sim_days: float = 0.0               # simulated time advanced, summed over environments
    updates: int = 0
    episodes: int = 0
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
    episode_seeds: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)     # every budget that left data out (agent.drop_report())
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


def _max_seq(obs) -> int:
    return max([o.seq for o in obs if o.seq is not None] or [FROM_START])


def _initial_cursors(env, ob: ObservationBatch) -> list:
    if hasattr(env, "cursors") and callable(env.cursors):
        return list(env.cursors())
    return [_max_seq(obs) for obs in ob.observations]


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


def derived_seeds(base: Optional[Sequence[int]], episode: int, B: int) -> Optional[list]:
    """Seeds of episode `episode`: the given seeds for episode 0, distinct derived seeds afterwards."""
    if episode == 0:
        return None if base is None else list(base)
    base = list(base) if base is not None else list(range(B))
    return [int(s) + SEED_STRIDE * episode for s in base]


def run_steps(agent, env: EnvAdapter, n: int, evaluator: Optional[GoalEvaluator] = None,
              update_every: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
              episode_seeds: Optional[Callable[[int], Optional[Sequence[int]]]] = None,
              think_s: Optional[float] = None, reset_on_done: bool = True, profile_gpu: bool = False,
              on_step: Optional[Callable[[int, dict], None]] = None) -> RunReport:
    """Run `n` decision rounds of `agent` (AgentP) on `env`. `update_every`: call agent.update() every k rounds
    (None/0 = never). The clock advances by each action's charged duration plus the think time (days). At the end
    the agent's undelivered post-action observations are archived (agent.flush)."""
    acc = ComputeAccount()
    rep = RunReport(compute=acc)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    dev = agent.device
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    think_days = (agent.config.think_s if think_s is None else think_s) / 86400.0
    prof = None
    if cuda and profile_gpu:
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CUDA])
        prof.__enter__()

    def gpu_timed(fn):
        if not cuda:
            return fn(), 0.0
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize(dev)
        return out, s.elapsed_time(e) / 1000.0

    def start_episode(e: int):
        t = time.perf_counter()
        sd = episode_seeds(e) if episode_seeds is not None else derived_seeds(seeds, e, getattr(env, "B", 1))
        rep.episode_seeds.append(None if sd is None else list(sd))
        ob = env.reset(sd)
        agent.reset(ob.env_ids)
        agent.set_goals(env.goals())
        acc.env_s += time.perf_counter() - t
        acc.episodes += 1
        scores = [float(evaluator.evaluate_goal(b).score) for b in range(len(ob.env_ids))] if evaluator else None
        return ob, _initial_cursors(env, ob), scores

    episode = 0
    ob, cursors, scores = start_episode(episode)
    B = len(ob.env_ids)
    seen = [_max_seq(obs) for obs in ob.observations]
    rep.returns = [0.0] * B
    for i in range(n):
        t = time.perf_counter()
        tools = env.available_tools()
        acc.env_s += time.perf_counter() - t
        t = time.perf_counter()
        actions, g = gpu_timed(lambda: agent.act(ob, tools))
        acc.act_s += time.perf_counter() - t
        acc.gpu_span_s += g
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
        acc.gpu_span_s += g
        acc.env_steps += B
        acc.decisions += sum(a is not None for a in actions)
        acc.sim_days += float(sum(durations))
        for b in range(B):
            rep.returns[b] += rewards[b]
        row = {"step": i, "rewards": rewards, "ok": [None if r is None else bool(r.ok) for r in results],
               "tools": [None if a is None else a.tool for a in actions], "t": list(adv.t),
               "sources": [None if a is None else a.source.value for a in actions],
               "delivered": [[o.seq for o in obs if o.seq is not None] for obs in nxt]}
        if update_every and (i + 1) % update_every == 0:
            t = time.perf_counter()
            um, g = gpu_timed(agent.update)
            acc.update_s += time.perf_counter() - t
            acc.gpu_span_s += g
            if um.get("updated"):
                acc.updates += 1
            rep.updates.append(um)
            row["update"] = um.get("update_id")
        rep.steps.append(row)
        if on_step is not None:
            on_step(i, row)
        if all(next_ob.done) and reset_on_done and i + 1 < n:
            episode += 1
            ob, cursors, scores = start_episode(episode)
            seen = [_max_seq(obs) for obs in ob.observations]
            rep.steps[-1]["reset_delivered"] = [[o.seq for o in obs if o.seq is not None] for obs in ob.observations]
        else:
            ob = next_ob
    if hasattr(agent, "flush"):
        agent.flush("end_of_run")
    rep.goal_scores = list(scores) if scores is not None else []
    if prof is not None:
        torch.cuda.synchronize(dev)
        prof.__exit__(None, None, None)
        kern = [e for e in prof.events() if getattr(e, "device_type", None) is not None
                and str(e.device_type).endswith("CUDA")]
        acc.gpu_busy_s = sum(float(getattr(e, "device_time", 0.0) or 0.0) for e in kern) / 1e6
        acc.gpu_kernels = len(kern)
    acc.wall_s = time.perf_counter() - wall0
    acc.cpu_s = time.process_time() - cpu0
    acc.max_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if cuda:
        acc.peak_gpu_mem_bytes = int(torch.cuda.max_memory_allocated(dev))
    acc.inference_tokens = _tokens_used(agent)
    acc.archive_records = agent.archive.count() if getattr(agent, "archive", None) is not None else None
    if hasattr(agent, "drop_report"):
        rep.dropped = agent.drop_report()
    return rep
