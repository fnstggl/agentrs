"""`run_learner`: the one driver loop for any learner on any `ExperienceEnv`.

    observe -> act -> execute -> advance -> reward -> observe(next) -> learner.observe -> store -> update

Every step is one interaction of one agent. The learner sees only the observation and the action result (plus the
mode's reward, which is 0.0 in observation-only mode); `verify()` is called by the driver at the end and returned
in the summary for the evaluator - it never reaches the learner. World changes are counted on the operator side:
an executed action counts when the environment reports that it changed world state (`last_changed`: in a kernel
world, its transaction recorded more than its own bookkeeping; in Tier 1, the settings the facility will run at
changed), else when the environment's state digest differs before and after it.
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
        if action is not None:
            tracked = hasattr(env, "last_changed")
            d0 = env.digest() if count_changes and not tracked else None
            result = env.execute(agent, action)
            if check_schema and validate_result(result):
                summary["schema_failures"] += 1
            summary["actions"] += 1
            summary["ok" if result.get("ok") else "rejected"] += 1
            if count_changes and (env.last_changed if tracked else env.digest() != d0):
                summary["world_changes"] += 1
        else:
            summary["waits"] += 1
        env.advance(int(think_s))
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
                                     learner={k: info.get(k) for k in ("name", "version", "params_hash")}))
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
