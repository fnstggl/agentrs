"""`run_learner`: the one driver loop for any learner on any `ExperienceEnv`.

    observe -> act -> execute -> advance -> reward -> observe(next) -> learner.observe -> store -> update

Every step is one interaction of one agent. The learner sees only the observation and the action result (plus the
mode's reward, which is 0.0 in observation-only mode); `verify()` is called by the driver at the end and returned
in the summary for the evaluator - it never reaches the learner. World changes are counted on the operator side:
an executed action counts when the environment reports that it changed world state (`last_changed`: in a kernel
world, its transaction recorded more than its own bookkeeping; in Tier 1, the settings the facility will run at
changed or an effectful rapid action was carried out; a joint rapid action counts when its tick runs, through the
environment's `changes` counter), else when the environment's state digest differs before and after it.

`run_learners(env, {agent: learner}, steps)` drives several learners in one world: every agent decides on its own
observation each step, the actions are executed (a joint tick in Tier 1 rapid worlds), then time passes once.
"""
from __future__ import annotations

import time
from typing import Any

from .observation import kinds_present, validate_observation, validate_result
from .store import make_record


def run_learner(env, learner, steps: int, *, agent: str | None = None, think_s: int | None = None,
                store=None, update_every: int = 32, learn: bool = True, explore: bool = True,
                check_schema: bool = True, count_changes: bool = True) -> dict:
    """Run `learner` (anything with ``act(obs, tools) -> action | None`` and ``observe(reward, result, next_obs,
    done)``; ``update()`` is called every `update_every` steps when `learn`) for `steps` interactions of `agent`.
    `think_s` is the simulated time between decisions (default: the facility tick in Tier 1, else 600 s)."""
    agent = agent or env.agents()[0]
    if think_s is None:
        think_s = int(getattr(env, "tick_s", 600))
    summary: dict[str, Any] = {"steps": 0, "actions": 0, "waits": 0, "ok": 0, "rejected": 0, "world_changes": 0,
                               "updates": 0, "reward_sum": 0.0, "nonzero_rewards": 0, "schema_failures": 0,
                               "kinds": set(), "records": 0, "update_log": [], "params_changed_updates": 0}
    t0 = time.perf_counter()
    obs = env.observe(agent)
    for step in range(int(steps)):
        if check_schema:
            errs = validate_observation(obs)
            if errs:
                summary["schema_failures"] += 1
                summary.setdefault("schema_errors", errs[:5])
        summary["kinds"] |= kinds_present(obs)
        action = learner.act(obs, obs["tools"], explore=explore) if _takes_explore(learner) else \
            learner.act(obs, obs["tools"])
        result = None
        c0 = getattr(env, "changes", None)
        changed = False
        if action is not None:
            tracked = hasattr(env, "last_changed")
            d0 = env.digest() if count_changes and not tracked else None
            result = env.execute(agent, action)
            if check_schema and validate_result(result):
                summary["schema_failures"] += 1
            summary["actions"] += 1
            summary["ok" if result.get("ok") else "rejected"] += 1
            changed = bool(env.last_changed if tracked else env.digest() != d0)
        else:
            summary["waits"] += 1
        env.advance(int(think_s))
        if count_changes and action is not None and (changed or (c0 is not None and getattr(env, "changes", c0) > c0)):
            summary["world_changes"] += 1
        r = float(env.reward(agent))
        summary["reward_sum"] += r
        summary["nonzero_rewards"] += int(r != 0.0)
        nxt = env.observe(agent)
        done = bool(getattr(env, "done", False))
        if hasattr(learner, "observe"):
            learner.observe(r, result, nxt, done)
        if store is not None:
            info = learner.info() if hasattr(learner, "info") else {"name": type(learner).__name__}
            store.append(make_record(env, agent, step, obs, action, result, r, next_cursor=nxt["cursor"],
                                     learner={k: info.get(k) for k in ("name", "version", "params_hash")},
                                     done=done))
            summary["records"] += 1
        summary["steps"] += 1
        if learn and hasattr(learner, "update") and (step + 1) % int(update_every) == 0:
            u = learner.update()
            summary["updates"] += int(u.get("updates", 0))
            summary["params_changed_updates"] += int(bool(u.get("changed")))
            summary["update_log"].append({k: u.get(k) for k in ("updates", "transitions", "loss", "changed",
                                                                "params_hash_after")})
        obs = nxt
        if done and hasattr(env, "reset"):
            env.reset()
            if hasattr(learner, "reset"):
                learner.reset(agent)
            obs = env.observe(agent)
    if learn and hasattr(learner, "update"):
        u = learner.update()
        summary["updates"] += int(u.get("updates", 0))
        summary["params_changed_updates"] += int(bool(u.get("changed")))
    summary["kinds"] = sorted(summary["kinds"])
    summary["wall_s"] = time.perf_counter() - t0
    summary["verification"] = env.verify()
    return summary


def _takes_explore(learner) -> bool:
    import inspect
    try:
        return "explore" in inspect.signature(learner.act).parameters
    except (TypeError, ValueError):
        return False


def run_learners(env, learners: dict, steps: int, *, think_s: int | None = None, update_every: int = 32,
                 learn: bool = True, store=None, check_schema: bool = True) -> dict:
    """Several learners in one world (one per agent of `env`): each step every learner acts on its own observation,
    the actions are executed in agent order (Tier 1 rapid worlds run them as one joint tick), then time passes once.
    Returns a summary per agent (actions, ok, rejected, world changes, updates, nonzero rewards, schema failures)."""
    agents = [a for a in env.agents() if a in learners]
    if not agents:
        raise ValueError(f"no learner for any agent of the environment ({env.agents()})")
    if think_s is None:
        think_s = int(getattr(env, "tick_s", 600))
    out = {a: {"steps": 0, "actions": 0, "ok": 0, "rejected": 0, "updates": 0, "nonzero_rewards": 0,
               "schema_failures": 0, "records": 0, "kinds": set(), "params_changed_updates": 0} for a in agents}
    obs = {a: env.observe(a) for a in agents}
    c_total0 = getattr(env, "changes", 0)
    for step in range(int(steps)):
        acts, results = {}, {}
        for a in agents:
            if check_schema and validate_observation(obs[a]):
                out[a]["schema_failures"] += 1
            out[a]["kinds"] |= kinds_present(obs[a])
            L = learners[a]
            acts[a] = L.act(obs[a], obs[a]["tools"], explore=True) if _takes_explore(L) else L.act(obs[a], obs[a]["tools"])
        for a in agents:
            if acts[a] is None:
                continue
            results[a] = env.execute(a, acts[a])
            out[a]["actions"] += 1
            out[a]["ok" if results[a].get("ok") else "rejected"] += 1
            if check_schema and validate_result(results[a]):
                out[a]["schema_failures"] += 1
        env.advance(int(think_s))
        done = bool(getattr(env, "done", False))
        for a in agents:
            r = float(env.reward(a))
            out[a]["nonzero_rewards"] += int(r != 0.0)
            nxt = env.observe(a)
            L = learners[a]
            if hasattr(L, "observe"):
                L.observe(r, results.get(a), nxt, done)
            if store is not None:
                info = L.info() if hasattr(L, "info") else {"name": type(L).__name__}
                store.append(make_record(env, a, step, obs[a], acts[a], results.get(a), r, next_cursor=nxt["cursor"],
                                         learner={k: info.get(k) for k in ("name", "version", "params_hash")},
                                         done=done))
                out[a]["records"] += 1
            obs[a] = nxt
            out[a]["steps"] += 1
            if learn and hasattr(L, "update") and (step + 1) % int(update_every) == 0:
                u = L.update()
                out[a]["updates"] += int(u.get("updates", 0))
                out[a]["params_changed_updates"] += int(bool(u.get("changed")))
        if done and hasattr(env, "reset"):
            env.reset()
            for a in agents:
                if hasattr(learners[a], "reset"):
                    learners[a].reset(a)
                obs[a] = env.observe(a)
    for a in agents:
        L = learners[a]
        if learn and hasattr(L, "update"):
            u = L.update()
            out[a]["updates"] += int(u.get("updates", 0))
            out[a]["params_changed_updates"] += int(bool(u.get("changed")))
        out[a]["kinds"] = sorted(out[a]["kinds"])
    return {"agents": out, "world_changes": int(getattr(env, "changes", 0) - c_total0),
            "verification": env.verify()}
