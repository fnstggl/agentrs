"""The lifetime harness: one semi-Markov decision loop for every environment family and every agent.

    observe -> agent.act -> adapter.execute -> adapter.advance_time(cost + think | wait + think)
            -> verified reward (operator-side evaluator) -> agent.observe(Transition) -> periodic agent.update

`run_lifetime` drives long horizons (90-180 simulated days and beyond) and records, per decision, a contract
`EventRecord` (the agent's view before acting, the action, its result, the verified return increment, the
semi-Markov duration) to an optional JSONL transcript. Checkpoints (operator state + evaluator state + agent
state + the not yet consumed observation batch + loop counters) are written atomically every
`checkpoint_every_days`; ``resume=True`` continues from the latest one, in this or another process (the world
store must be durable, e.g. ``sqlite:///...``). Compute is accounted per phase (agent, environment execution,
simulation + evaluation, learning) plus the agent's own `usage()`.

Separation: the harness is operator-side code. It hands the agent only the adapter's records and scalar rewards;
`IndependentGoalResult.details` and evaluator objects stay here (they go to `LifetimeResult.final`, the
experimenter's record, never to the agent).
"""
from __future__ import annotations

import json
import os
import pickle
import resource
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

from ..interfaces import EvidenceKind, EventRecord, ObsKind, Provenance, new_id
from .agent_protocol import Transition

DAY = 86400.0


@dataclass
class LifetimeResult:
    steps: int
    t_days: list                       # [B] simulated time at the end (days)
    elapsed_days: list                 # [B] simulated days lived in this run
    returns: list                      # [B] total verified return
    final: list                        # [B] IndependentGoalResult dicts (operator record, includes details)
    success_day: list                  # [B] simulated day of verified success (None = not achieved)
    accounting: dict
    updates: list = field(default_factory=list)
    transcript: Optional[str] = None
    checkpoints: list = field(default_factory=list)
    resumed_from: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class _Meter:
    def __init__(self):
        self.wall = {"act": 0.0, "execute": 0.0, "advance": 0.0, "observe": 0.0, "update": 0.0, "tools": 0.0,
                     "checkpoint": 0.0}
        self.counts = {"decisions": 0, "actions": 0, "waits": 0, "rejected": 0, "skipped": 0}
        self.by_tool: dict[str, int] = {}
        self.errors: dict[str, int] = {}
        self.cpu0 = time.process_time()
        self.t0 = time.perf_counter()

    def timed(self, key: str, fn, *a, **kw):
        t = time.perf_counter()
        out = fn(*a, **kw)
        self.wall[key] += time.perf_counter() - t
        return out

    def summary(self, agent) -> dict:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        out = {"wall_s": time.perf_counter() - self.t0, "cpu_s": time.process_time() - self.cpu0,
               "max_rss_mb": ru.ru_maxrss / 1024.0, "phase_wall_s": dict(self.wall), **self.counts,
               "by_tool": dict(sorted(self.by_tool.items(), key=lambda kv: -kv[1])), "errors": dict(self.errors)}
        if hasattr(agent, "usage"):
            try:
                out["agent"] = agent.usage()
            except Exception as e:  # noqa: BLE001 - accounting must not kill a lifetime
                out["agent"] = {"error": repr(e)}
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                out["gpu_max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        except Exception:  # noqa: BLE001
            pass
        return out


def _self_ids(batch) -> list:
    out = []
    for b, olist in enumerate(batch.observations):
        sid = next((o.entity for o in olist if o.kind is ObsKind.TEXT and o.meta.get("self")), None)
        out.append(sid or batch.env_ids[b])
    return out


def _write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_lifetime(agent, adapter, evaluator=None, *, days: Optional[float] = None,
                 max_decisions: Optional[int] = None, think_s: float = 600.0, update_every: int = 0,
                 seeds: Optional[Sequence[int]] = None, stop_on_success: bool = False,
                 transcript: Optional[str] = None, operator=None, checkpoint_dir: Optional[str] = None,
                 checkpoint_every_days: Optional[float] = None, resume: bool = False,
                 model_version: Optional[Callable[[], int]] = None,
                 on_step: Optional[Callable[[int, Transition], None]] = None,
                 stop_after_decisions: Optional[int] = None) -> LifetimeResult:
    """Run `agent` in `adapter` until every environment is done (its task horizon), `days` simulated days have
    passed, `max_decisions` decisions were made, or (``stop_on_success``) every environment reached verified
    success. `evaluator` supplies verified returns (`take_rewards`) and the final independent results.
    `operator` (with ``checkpoint(path)`` / ``restore(path)``) is needed for checkpoints and resume.
    `stop_after_decisions` ends this call early (simulating an interruption) without closing anything."""
    meter = _Meter()
    ckpts: list[str] = []
    resumed_from = None
    if (checkpoint_every_days or resume) and (operator is None or checkpoint_dir is None):
        raise ValueError("checkpoints and resume need `operator` and `checkpoint_dir`")
    latest = os.path.join(checkpoint_dir, "LATEST") if checkpoint_dir else None
    if resume and latest and os.path.exists(latest):
        with open(latest) as f:
            d = os.path.join(checkpoint_dir, f.read().strip())
        operator.restore(os.path.join(d, "operator.pkl"))
        with open(os.path.join(d, "loop.pkl"), "rb") as f:
            st = pickle.load(f)
        if evaluator is not None and os.path.exists(os.path.join(d, "evaluator.pkl")):
            with open(os.path.join(d, "evaluator.pkl"), "rb") as f:
                evaluator.load_state(f.read())
        agent_restored = False
        if hasattr(agent, "load") and os.path.exists(os.path.join(d, "agent")):
            agent.load(os.path.join(d, "agent"))
            agent_restored = True
        elif hasattr(agent, "begin"):
            agent.begin(st["env_ids"], st["goals"])
        batch, step, returns = st["batch"], st["step"], st["returns"]
        t_start, episode_id, actor_ids = st["t_start"], st["episode_id"], st["actor_ids"]
        next_ckpt, success_day, updates = st["next_ckpt"], st["success_day"], st["updates"]
        ckpts = list(st.get("ckpts", []))
        resumed_from = d
        meter.counts["agent_restored"] = int(agent_restored)
    else:
        batch = adapter.reset(seeds)
        goals = adapter.goals()
        if hasattr(agent, "begin"):
            agent.begin(list(batch.env_ids), goals)
        step = 0
        B = len(batch.env_ids)
        returns = [0.0] * B
        t_start = list(batch.t)
        episode_id = new_id("ep_")
        actor_ids = _self_ids(batch)
        next_ckpt = checkpoint_every_days
        success_day = [None] * B
        updates = []
    B = len(batch.env_ids)
    tw = open(transcript, "a") if transcript else None
    if tw and step == 0:
        tw.write(json.dumps({"header": {"format": "cam-unified-lifetime/1", "env_ids": batch.env_ids,
                                        "think_s": think_s, "episode_id": episode_id}}) + "\n")
    calls_this = 0
    try:
        while True:
            elapsed = [batch.t[b] - t_start[b] for b in range(B)]
            done = [bool(batch.done[b]) or (days is not None and elapsed[b] >= days) for b in range(B)]
            if all(done):
                break
            if max_decisions is not None and step >= max_decisions:
                break
            if stop_on_success and evaluator is not None and all(s is not None for s in success_day):
                break
            if stop_after_decisions is not None and calls_this >= stop_after_decisions:
                break
            tools = meter.timed("tools", adapter.available_tools)
            actions = list(meter.timed("act", agent.act, batch, tools))
            actions = [None if done[b] else actions[b] for b in range(B)]
            results = meter.timed("execute", adapter.execute, actions)
            dts = []
            for b in range(B):
                if done[b] or actions[b] is None:
                    dts.append(0.0 if done[b] else think_s / DAY)
                    meter.counts["skipped"] += int(not done[b] and actions[b] is None)
                    continue
                r = results[b]
                dts.append(float(r.duration) + think_s / DAY)
                tool = actions[b].tool
                meter.by_tool[tool] = meter.by_tool.get(tool, 0) + 1
                if r.meta.get("harness") and tool == "wait":
                    meter.counts["waits"] += 1
                else:
                    meter.counts["actions"] += 1
                if not r.ok:
                    meter.counts["rejected"] += 1
                    meter.errors[str(r.error)] = meter.errors.get(str(r.error), 0) + 1
            nxt = meter.timed("advance", adapter.advance_time, dts)
            rewards = evaluator.take_rewards() if evaluator is not None else [0.0] * B
            mv = int(model_version()) if model_version else int(getattr(agent, "model_version", 0) or 0)
            records = []
            for b in range(B):
                a, r = actions[b], results[b]
                returns[b] += rewards[b]
                records.append(EventRecord(
                    event_id=a.action_id if a is not None else new_id("e_"), env_id=batch.env_ids[b],
                    actor_id=actor_ids[b], episode_id=episode_id, step=step, model_version=mv, t_sim=batch.t[b],
                    observations=batch.observations[b], action=a, result=r, reward=float(rewards[b]),
                    cost=float(r.cost) if r is not None else 0.0, done=bool(nxt.done[b]),
                    behavior_logp=None if a is None else a.logp, provenance=Provenance.REAL,
                    evidence=EvidenceKind.CONTROLLED, extra={"dt_days": dts[b], "skipped": a is None}))
                if evaluator is not None and success_day[b] is None:
                    st = getattr(evaluator, "success_t", None)
                    if st is not None and st[b] is not None:
                        success_day[b] = float(st[b]) / DAY - t_start[b]   # days since the episode began
            tr = Transition(records, nxt, list(rewards), list(nxt.done), dts)
            meter.timed("observe", agent.observe, tr)
            if tw:
                for rec in records:
                    tw.write(rec.to_json() + "\n")
            meter.counts["decisions"] += 1
            step += 1
            calls_this += 1
            if update_every and step % update_every == 0:
                updates.append(meter.timed("update", agent.update))
            if on_step is not None:
                on_step(step, tr)
            batch = nxt
            if checkpoint_every_days and min(batch.t[b] - t_start[b] for b in range(B)) >= next_ckpt:
                t0 = time.perf_counter()
                d = os.path.join(checkpoint_dir, f"ckpt_{len(ckpts):04d}")
                os.makedirs(d, exist_ok=True)
                operator.checkpoint(os.path.join(d, "operator.pkl"))
                if evaluator is not None and hasattr(evaluator, "state_dict"):
                    _write_atomic(os.path.join(d, "evaluator.pkl"), evaluator.state_dict())
                if hasattr(agent, "save"):
                    agent.save(os.path.join(d, "agent"))
                ckpts.append(d)
                next_ckpt += checkpoint_every_days
                if tw:
                    tw.flush()
                _write_atomic(os.path.join(d, "loop.pkl"), pickle.dumps({
                    "batch": batch, "step": step, "returns": returns, "t_start": t_start, "episode_id": episode_id,
                    "actor_ids": actor_ids, "next_ckpt": next_ckpt, "success_day": success_day,
                    "updates": updates, "env_ids": list(batch.env_ids), "goals": adapter.goals(), "ckpts": ckpts}))
                _write_atomic(latest, os.path.basename(d).encode())
                meter.wall["checkpoint"] += time.perf_counter() - t0
    finally:
        if tw:
            tw.close()
    final = []
    for b in range(B):
        if evaluator is not None:
            final.append(evaluator.evaluate_goal(b).to_dict())
        else:
            final.append({})
    return LifetimeResult(steps=step, t_days=list(batch.t), elapsed_days=[batch.t[b] - t_start[b] for b in range(B)],
                          returns=returns, final=final, success_day=success_day, accounting=meter.summary(agent),
                          updates=updates, transcript=transcript, checkpoints=ckpts, resumed_from=resumed_from)
