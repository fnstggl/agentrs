"""cam_live.eval.runner — drive one condition through a sequence of lifetimes with the real persistent runtime.

    run_sequence(agent, family, world_seeds, root, clock, ...) -> {"lifetimes": [...], "compute": {...}}

Each lifetime is one objective of ``family`` (``eval.envs.make_world``) run by ``runtime.loop.LiveRuntime`` with
the diagnostic ``AllowAllPolicy`` and the family's independent verifier, the environment wrapped in
:class:`eval.envs.TruthTap`. After the runtime has finished the lifetime (objective verified, decision budget or
limits), the clock is advanced to the objective's deadline so every consequence of the agent's actions has
arrived, and the verifier's final signed verdict is recorded: *verified progress* = min(1, value / target).

Per lifetime: decisions, verified status / value / progress, ground-truth efficiency (LabWorld) or mean quality
(CodecWorld), money, simulated and wall seconds, inference and training compute (``conditions.Timed``), the
runtime counters, and the truth rows of every executed action.
"""
from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..contracts import ResultStatus
from ..diagnostics.world import AllowAllPolicy, LabWorld
from ..runtime.loop import LiveRuntime, RuntimeConfig
from .conditions import Timed
from .envs import LAB_DECISIONS, LAB_K, CodecWorld, TruthTap, lab_efficiency, make_world

DAY = 86400.0


def runtime_config(k: int = 16) -> RuntimeConfig:
    # checkpoints only when a lifetime finishes (clock seconds between commits far beyond a lifetime)
    return RuntimeConfig(fsync=False, candidates_k=int(k), keep_checkpoints=1, checkpoint_every_s=60 * DAY,
                         journal_every=10 ** 9, journal_min_interval_s=10 ** 9, learn_budget_s=30.0,
                         meta_budget_s=30.0)


class DecisionTap:
    """Proxy around the agent that records, at every ``select``, the ground truth of *every* candidate (not only the
    executed one) next to what the agent predicted for it — the material of the action-quality test (16.3)."""

    def __init__(self, agent: Any, tap: TruthTap):
        object.__setattr__(self, "_a", agent)
        object.__setattr__(self, "_tap", tap)
        object.__setattr__(self, "rows", [])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._a, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._a, name, value)

    def select(self, evaluations: Any) -> Any:
        d = self._a.select(evaluations)
        env = self._tap._env
        cands = []
        for ev in evaluations:
            a = ev.action
            truth = candidate_truth(env, a)
            gr = [p for p in ev.predictions if p.query.goal_relevant]
            cands.append({"operation": a.operation, "arguments": dict(a.arguments), "truth": truth,
                          "p_success": ev.p_success, "value": ev.value, "voi": ev.voi,
                          "goal_mean": gr[0].dist.mean if gr else None, "goal_std": gr[0].dist.std if gr else None})
        self.rows.append({"decision_id": d.decision_id, "chosen_index": d.chosen_index,
                          "probabilities": list(d.probabilities), "exploration": d.exploration,
                          "experiment_id": d.experiment_id, "t": float(env.clock.now()), "candidates": cands})
        return d


def candidate_truth(env: Any, a: Any) -> float:
    """Ground-truth value of a candidate at this moment: expected adopters (LabWorld ``lab.act``: hidden rate x
    effort; 0 for every other operation), or the setting's ratio relative to the best (CodecWorld; 0 otherwise)."""
    if isinstance(env, LabWorld):
        if a.operation != "lab.act":
            return 0.0
        try:
            ch, e = int(a.arguments["channel"]), int(a.arguments["effort"])
            return float(env._rate(ch) * e) if 0 <= ch < env.cfg.n_channels else 0.0
        except (KeyError, TypeError, ValueError):
            return 0.0
    if isinstance(env, CodecWorld):
        r = env.ratio_of(a)
        return 0.0 if r is None else float(r / env.best_ratio())
    return 0.0


def close_out(agent: Any, world: Any, verifier: Any, task: Any, *, learn_budget_s: float = 30.0) -> Dict[str, Any]:
    """After the runtime has stopped at the decision budget with the objective still pending: advance to the
    objective's deadline and deliver what a runtime that kept running without new decisions would have delivered —
    the verifier's signed measurements of every due outcome query and its signed final verdict — through the
    agent's own steps 4, 5, 13, 14 and 15. The same for every condition. Returns what was delivered."""
    if isinstance(world, LabWorld):
        deadline = world.t0 + float(task.limits.duration_s or world.cfg.horizon_s)
        now = world.clock.now()
        if now < deadline:
            world.clock.advance(deadline - now + 1.0)
    final = verifier.verify(task)
    store = getattr(agent, "store", None)
    due = []
    if store is not None:
        try:
            due = [q for _, q in store.pending(world.clock.now())]
        except Exception:  # noqa: BLE001
            due = []
    outs = verifier.measure(due) if due else []
    info = {"due_queries": len(due), "measured": sum(1 for o in outs if o.value is not None),
            "final_status": final.status}
    info["reconcile"] = _jsonable(agent.reconcile(outs, final))
    agent.update_beliefs()
    agent.internal_rewards()
    ups = agent.learn(learn_budget_s) or []
    metas = agent.meta_learn(learn_budget_s) or []
    info["learning_updates"] = [getattr(u, "timescale", "") for u in ups]
    info["meta_updates"] = [{"applied": getattr(m, "applied", None), "skipped_reason": getattr(m, "skipped_reason", ""),
                             "meta_grad_norm": getattr(m, "meta_grad_norm", None),
                             "reward_version_before": getattr(m, "reward_version_before", ""),
                             "reward_version_after": getattr(m, "reward_version_after", ""),
                             "n_independent_outcomes": getattr(m, "n_independent_outcomes", None),
                             "bootstrap_mix": getattr(m, "bootstrap_mix", None)} for m in metas]
    return info


def _jsonable(x: Any) -> Any:
    import json
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:  # noqa: BLE001
        return str(x)


def finalize_lab(world: LabWorld, verifier: Any, task: Any) -> Any:
    """Advance to the deadline (all arrivals caused by the agent's actions are in) and take the final verdict."""
    deadline = world.t0 + float(task.limits.duration_s or world.cfg.horizon_s)
    now = world.clock.now()
    if now < deadline:
        world.clock.advance(deadline - now + 1.0)
    return verifier.verify(task)


def run_lifetime(agent: Any, family: str, world_seed: int, run_dir: str, clock: Any, *, max_decisions: int,
                 k: int = 16, step_cap: int = 6, on_step: Optional[Callable[[Any], None]] = None) -> Dict[str, Any]:
    world, verifier, task = make_world(family, world_seed, clock)
    task.limits.max_decisions = int(max_decisions)
    tap = TruthTap(world)
    agent.attach_verifier(verifier)
    timed = agent if isinstance(agent, Timed) else Timed(agent)
    s0 = dict(timed.seconds)
    c0 = dict(timed.cpu_seconds)
    dtap = DecisionTap(timed, tap)
    rt = LiveRuntime(dtap, tap, AllowAllPolicy(), verifier, task, run_dir, clock, runtime_config(k))
    t_wall, t_sim = time.time(), clock.now()
    rt.start()
    steps = 0
    while rt.status == "running" and steps < step_cap * max_decisions:
        rt.step()
        steps += 1
        if on_step is not None:
            on_step(rt)
    if rt.status == "running":
        rt._finish("stopped", "evaluation step cap reached")
    status_rt = rt.status
    counters = dict(rt.counters)
    errors = list(rt._errors[-5:])
    rt.close()
    closing = None
    if (rt._last_verified.get("outcome") or {}).get("status") not in ("achieved", "failed"):
        try:
            closing = close_out(timed, world, verifier, task)
        except Exception as e:  # noqa: BLE001 — recorded; the lifetime's verdict below is still the verifier's
            closing = {"error": f"{type(e).__name__}: {e}"}
    wall = time.time() - t_wall
    extra: Dict[str, Any] = {}
    if isinstance(world, LabWorld):
        final = finalize_lab(world, verifier, task)
        got = sum(r.get("expected_adopters", 0.0) for r in tap.rows)
        quality = lab_efficiency(tap.rows)
        extra = {"expected_adopters": got, "oracle_expected_adopters": oracle_expected(family, world_seed),
                 "target_adopters": world.cfg.target}
    else:
        final = verifier.verify(task)
        qs = [r.get("quality", 0.0) for r in tap.rows]
        quality = sum(qs) / len(qs) if qs else 0.0
    target = float(final.target or 1.0)
    out = {**extra, "family": family, "world_seed": int(world_seed), "objective_id": task.objective_id,
           "runtime_status": status_rt, "decisions": int(counters.get("decisions", 0)), "steps": steps,
           "verified_status": final.status, "verified_value": float(final.value), "target": target,
           "verified_signature_ok": bool(verifier.check_signature(final)),
           "progress": min(1.0, float(final.value) / target) if target > 0 else 0.0,
           "achieved": final.status == "achieved", "quality": float(quality),
           "money": float(counters.get("money", 0.0)), "sim_seconds": float(clock.now() - t_sim),
           "wall_seconds": wall,
           "inference_s": timed.seconds["inference"] - s0["inference"],
           "training_s": timed.seconds["training"] - s0["training"],
           "inference_cpu_s": timed.cpu_seconds["inference"] - c0["inference"],
           "training_cpu_s": timed.cpu_seconds["training"] - c0["training"],
           "counters": counters, "errors": errors, "runtime_dir": run_dir, "close_out": closing,
           "truth_rows": tap.rows, "decision_rows": dtap.rows}
    if isinstance(world, CodecWorld):
        out["decision_quality"] = [r.get("quality", 0.0) for r in tap.rows]
        out["first_achieving_decision"] = next((i + 1 for i, r in enumerate(tap.rows)
                                                if r.get("ratio", 0.0) >= world.target), None)
    return out


def run_sequence(agent: Any, family: str, world_seeds: Sequence[int], root: str, clock: Any, *,
                 max_decisions: int, k: int = 16, between: Optional[Callable[[int, Dict[str, Any]], None]] = None
                 ) -> Dict[str, Any]:
    timed = agent if isinstance(agent, Timed) else Timed(agent)
    lifetimes = []
    for i, ws in enumerate(world_seeds):
        lt = run_lifetime(timed, family, ws, os.path.join(root, f"lt{i:02d}_{family}_{ws}"), clock,
                          max_decisions=max_decisions, k=k)
        lt["index"] = i
        lifetimes.append(lt)
        if between is not None:
            between(i, lt)
    return {"lifetimes": lifetimes, "compute": {"seconds": dict(timed.seconds), "cpu_seconds": dict(timed.cpu_seconds)}}


# ------------------------------------------------------------------------------------------------ anchors
def _spread(cands: List[Any], k: int) -> List[Any]:
    """The agent's ordering rule without backbone candidates (``CamLiveAgent._order``): an even spread over the
    operations, the walker's order within each operation."""
    by: Dict[str, List[Any]] = {}
    for a in cands:
        by.setdefault(a.operation, []).append(a)
    out: List[Any] = []
    while len(out) < k and any(by.values()):
        for op in list(by):
            if by[op] and len(out) < k:
                out.append(by[op].pop(0))
    return out


_ORACLE_CACHE: Dict[tuple, float] = {}


def oracle_expected(family: str, world_seed: int) -> float:
    """Expected adopters of the oracle anchor on this world (calibration budget, same candidate generator)."""
    key = (family, int(world_seed))
    if key not in _ORACLE_CACHE:
        _ORACLE_CACHE[key] = float(run_anchor("oracle", family, world_seed, max_decisions=LAB_DECISIONS, k=LAB_K,
                                              target=1)["expected_adopters"])
    return _ORACLE_CACHE[key]


def run_anchor(kind: str, family: str, world_seed: int, *, max_decisions: int, k: int = 16, seed: int = 0,
               target: Optional[int] = None) -> Dict[str, Any]:
    """Reference policies without any model: ``uniform`` (uniform over the same k candidates) and ``oracle`` (knows
    the hidden mechanism: unlocks the door channel when it is the best, then always the best intervention).
    They anchor the competence scale; they are not baselines of spec 16.1."""
    from ..contracts import AuthorizationDecision, AuthVerdict, SimClock
    from ..predict.candidates import schema_candidates
    from .envs import oracle_rates
    clock = SimClock()
    world, verifier, task = make_world(family, world_seed, clock, target=target)
    tap = TruthTap(world)
    rng = random.Random(seed * 1_000_003 + world_seed)
    allow = AuthorizationDecision(AuthVerdict.ALLOW, "", ["anchor"], grant="anchor")
    for t in range(max_decisions):
        cands = _spread(schema_candidates(world.discover_tools(), per_tool=4, id_prefix=f"anc{t}"), k)
        if kind == "uniform":
            a = rng.choice(cands)
        elif kind == "oracle":
            a = _oracle_choice(world, cands)
        else:
            raise KeyError(kind)
        world.observe()
        res = tap.execute(a, allow)
        if isinstance(world, CodecWorld) and verifier.verify(task).status == "achieved":
            break
        if isinstance(world, LabWorld) and res.status != ResultStatus.OK and res.failure_detail == "budget exhausted":
            break
    got = None
    if isinstance(world, LabWorld):
        final = finalize_lab(world, verifier, task)
        got = sum(r.get("expected_adopters", 0.0) for r in tap.rows)
        quality = lab_efficiency(tap.rows)
    else:
        final = verifier.verify(task)
        qs = [r.get("quality", 0.0) for r in tap.rows]
        quality = sum(qs) / len(qs) if qs else 0.0
    target = float(final.target or 1.0)
    return {"anchor": kind, "expected_adopters": got, "family": family, "world_seed": world_seed, "decisions": len(tap.rows),
            "verified_status": final.status, "progress": min(1.0, float(final.value) / target),
            "achieved": final.status == "achieved", "quality": quality,
            "first_achieving_decision": (next((i + 1 for i, r in enumerate(tap.rows) if r.get("ratio", 0.0)
                                               >= world.target), None) if isinstance(world, CodecWorld) else None)}


def _oracle_choice(world: Any, cands: List[Any]) -> Any:
    from .envs import oracle_rates
    if isinstance(world, CodecWorld):
        scored = [(world.ratio_of(a) or 0.0, i) for i, a in enumerate(cands)]
        return cands[max(scored)[1]]
    c = world.cfg
    rates = oracle_rates(world)                       # door channel counted as unlocked
    acts = [a for a in cands if a.operation == "lab.act"]
    if not acts:
        return cands[0]
    best = max(acts, key=lambda a: rates[int(a.arguments["channel"])] * int(a.arguments["effort"]))
    door = c.key_door and int(best.arguments["channel"]) == c.n_channels - 1
    if door and world._unlocked_at is None:
        u = next((a for a in cands if a.operation == "lab.unlock"), None)
        if u is not None:
            return u
    return max(acts, key=lambda a: world._rate(int(a.arguments["channel"])) * int(a.arguments["effort"]))
