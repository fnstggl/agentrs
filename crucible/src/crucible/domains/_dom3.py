"""Shared parts of the Tier 2 domains of workstream I-3 (MASTER_SPEC Parts 9.11-9.15).

* world composition: the substrate and the ONE population (`crucible.population`, D-15) every domain world starts
  from, in kernel dependency order;
* decision recording with provenance (`decide`): every individual decision of a domain activity is one recorded
  cohort batch carrying the decision model's id, version and fidelity-register id (ONE_WORLD_CONTRACTS section 1);
* `logit_choice`: the multinomial-logit (Gumbel-max) choice the domains' population decision models share;
* measurement from the recorded log (`event_counts`, `mechanics_from_events`): a required mechanic counts as
  exercised only when the events that carry it were recorded in the world's log, never because a domain says so;
* `IdlePolicy` (the null baseline every domain must fail with) and `run_env` (drive a policy through the learner
  interface `crucible.experience`, the same path a learner takes).

Nothing here keeps state: population state lives in the population's and the domains' own column tables.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

DAY = 86400
HOUR = 3600
SUBSTRATE = ("identity", "bank", "admin", "artifacts", "cols")


# ====================================================================== composition
def pop_config(n: int, *, tick_s: int = 1800, profile: str = "development", classes: dict | None = None) -> dict:
    """Config of the one population for a domain world: `n` class-1 individuals of the development profile. The
    neural classes (2-4) are off unless a configuration names them (they need their own runtime modules)."""
    cls = {"adaptive": 0, "strategic": 0, "semantic": 0}
    cls.update(classes or {})
    return {"profile": profile, "n": int(n), "classes": cls, "tick_s": int(tick_s)}


def recipe(modules: list[tuple[str, dict]], params: dict | None = None) -> dict:
    """The 4.0 world-family form {modules, module_list, params} of an ordered module list."""
    names = [m for m, _ in modules]
    if len(names) != len(set(names)):
        raise ValueError(f"module listed twice: {names}")
    return {"modules": {m: dict(c) for m, c in modules}, "module_list": names,
            "params": {"cost_model": "deterministic", **(params or {})}}


def order_modules(mods: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Move every module's kernel dependencies before it (stable otherwise)."""
    import crucible
    from ..api.local import ensure_modules
    from ..core.kernel import MODULES
    crucible.load_all_modules()
    ensure_modules([m for m, _ in mods])
    have = {m: c for m, c in mods}
    out: list[str] = []

    def add(m):
        if m in out or m not in have:
            return
        for d in MODULES[m].depends:
            add(d)
        out.append(m)
    for m, _ in mods:
        add(m)
    missing = sorted({d for m in out for d in MODULES[m].depends if d not in have})
    if missing:
        raise ValueError(f"domain world misses kernel dependencies {missing}")
    return [(m, have[m]) for m in out]


# ====================================================================== population decisions
def decide(ctx, kind: str, idx, values: dict, *, table: str, model_id: str, fidelity_id: str,
           model_version: str = "1", extra: dict | None = None, mode: str = "set") -> str | None:
    """One recorded cohort decision with decision provenance (`crucible.population.record_decisions`)."""
    from .. import population as P
    ix = np.asarray(idx, np.int64)
    if ix.size == 0:
        return None
    return P.record_decisions(ctx, kind, ix, values, model_id=model_id, model_version=model_version,
                              fidelity_id=fidelity_id, mode=mode, table=table, extra=extra or {})


def logit_choice(rng, utils: np.ndarray, mask: np.ndarray | None = None, scale: float = 1.0) -> np.ndarray:
    """Row-wise multinomial-logit choice (Gumbel-max): index of the chosen column per row, -1 where nothing is
    available."""
    u = np.asarray(utils, np.float64)
    if u.ndim == 1:
        u = u[None, :]
    g = u + rng.gumbel(0.0, float(scale), u.shape)
    if mask is not None:
        g = np.where(mask, g, -np.inf)
    out = np.argmax(g, axis=1)
    if mask is not None:
        out = np.where(np.any(mask, axis=1), out, -1)
    return out


def population_index(principal: str) -> int | None:
    """Row of a population principal ("p<i>"), None for every other principal."""
    from ..substrate import ids
    return ids.index(principal)


# ====================================================================== measurement from the log
def event_counts(world, since_seq: int = 0, *, prefixes: Iterable[str] = ()) -> Counter:
    """Counts of recorded event types (and `cols.batch` kinds as `batch:<kind>`) since `since_seq`."""
    pre = tuple(prefixes)
    c: Counter = Counter()
    for ev in world.events_range(since_seq):
        t = ev.type
        if t == "cols.batch":
            k = "batch:" + str(ev.payload.get("kind", ""))
            if not pre or k.startswith(pre) or "cols.batch".startswith(pre):
                c[k] += 1
            continue
        if not pre or t.startswith(pre):
            c[t] += 1
    return c


def mechanics_from_events(counts: Counter, table: dict[str, list[tuple[str, int]]]) -> dict[str, dict]:
    """`table`: mechanic -> [(event type or "batch:<kind>" or a prefix ending in '*', minimum count), ...]; a
    mechanic is exercised when EVERY listed event kind reached its minimum in the recorded log."""
    out = {}
    for mech, reqs in table.items():
        got = {}
        ok = True
        for name, need in reqs:
            if name.endswith("*"):
                n = sum(v for k, v in counts.items() if k.startswith(name[:-1]))
            else:
                n = int(counts.get(name, 0))
            got[name] = n
            ok = ok and n >= int(need)
        out[mech] = {"exercised": bool(ok), "events": got}
    return out


def batch_records(world, kind: str, key: str, since_seq: int = 0) -> list:
    """The `extra[key]` lists of every recorded cohort batch of `kind` (evaluator side: decision records)."""
    out = []
    for ev in world.events_range(since_seq):
        if ev.type == "cols.batch" and ev.payload.get("kind") == kind:
            out.extend((ev.payload.get("extra") or {}).get(key) or [])
    return out


# ====================================================================== policies and the learner path
class IdlePolicy:
    """The null baseline: a participant that never acts (every domain objective must fail for it)."""
    name = "idle"
    kind = "scripted"

    def reset(self, actor, rng):
        self.actor = actor

    def act(self, obs, tools):
        return None

    def observe_result(self, action, result):
        pass


def run_env(env, agent: str, policy, *, days: float, dt: int = 3 * HOUR, max_actions_per_step: int = 40) -> dict:
    """Drive `policy` as `agent` through an ExperienceEnv: at each step the policy acts until it waits (returns
    None) or hits `max_actions_per_step`, then the world advances `dt` seconds. Returns counts of the run."""
    steps = int(round(days * DAY / dt))
    if hasattr(policy, "reset"):
        policy.reset(agent, np.random.default_rng(0))
    n_ok = n_err = 0
    rewards = []
    for _ in range(steps):
        for _ in range(int(max_actions_per_step)):
            obs = env.observe(agent)
            act = policy.act(obs, obs.get("tools") or [])
            if act is None:
                break
            res = env.execute(agent, {"tool": act["tool"], "args": act.get("args") or {}})
            if res.get("ok"):
                n_ok += 1
            else:
                n_err += 1
            if hasattr(policy, "observe_result"):
                policy.observe_result(act, res)
        env.advance(int(dt))
        rewards.append(float(env.reward(agent)))
    return {"steps": steps, "actions_ok": n_ok, "actions_failed": n_err, "reward_sum": float(np.sum(rewards)),
            "reward_nonzero": int(np.count_nonzero(rewards))}


def make_env(domain: str, config: str, seed: int, **kw):
    """The learner path: `crucible.experience.make(2, "<domain>:<config>", seed)` (observation_only by default)."""
    from ..experience import make
    return make(2, f"{domain}:{config}", int(seed), **kw)


def related(world, a: str, b: str) -> bool:
    """Whether two principals are related parties (shared control or money control), fail closed for agents."""
    from ..commerce.integrity import independent_customer
    return not independent_customer(world, a, b)


def quantile(xs, q: float) -> float:
    xs = [float(x) for x in xs]
    return float(np.quantile(xs, q)) if xs else float("nan")


def jsonable(x: Any) -> Any:
    from ..core.canonical import to_jsonable
    return to_jsonable(x)
