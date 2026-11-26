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
    max_actions_per_step = max(int(max_actions_per_step), int(getattr(policy, "burst", 0) or 0))
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
            if hasattr(policy, "observe"):
                policy.observe(0.0, res, env.observe(agent), False)
            elif hasattr(policy, "observe_result"):
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


def session_provenance(world, table: str, kinds: Iterable[str]) -> dict:
    """Evaluator side: every recorded decision batch of `kinds` on a domain column table must come from the
    population's own sessions - its REF listed by a `pop.session` batch and every row an individual of that session's
    cohort - and carry a registered decision model. Returns counts; `forced_rows` must be 0."""
    from ..substrate import artifacts
    kinds = tuple(kinds)
    listed: dict[str, np.ndarray] = {}
    out = {"batches": 0, "rows": 0, "organic_rows": 0, "forced_rows": 0, "missing_provenance": 0}
    models = world.state.tables.get("pop_decision_models", {})
    evs = list(world.events_range(0))
    for ev in evs:
        p = ev.payload
        if ev.type == "cols.batch" and p.get("table") == "pop" and p.get("kind") == "pop.session":
            refs = (p.get("extra") or {}).get("decisions") or []
            if refs:
                coh = artifacts.get_array_blob(world, p["ref"])["idx"].astype(np.int64)
                for r in refs:
                    listed[r] = coh
    for ev in evs:
        p = ev.payload
        if ev.type != "cols.batch" or p.get("table") != table or p.get("kind") not in kinds:
            continue
        v = artifacts.get_array_blob(world, p["ref"])["idx"].astype(np.int64)
        out["batches"] += 1
        out["rows"] += int(v.size)
        prov = (p.get("extra") or {}).get("decision")
        if not isinstance(prov, dict) or f"{prov.get('model_id')}@{prov.get('model_version')}" not in models:
            out["missing_provenance"] += 1
            out["forced_rows"] += int(v.size)
            continue
        if p["ref"] not in listed:
            out["forced_rows"] += int(v.size)
            continue
        inside = np.isin(v, listed[p["ref"]])
        out["organic_rows"] += int(inside.sum())
        out["forced_rows"] += int((~inside).sum())
    return out


# ====================================================================== framework adapters (domains-framework/2)
def seeded_rng(name: str, seed: int):
    """The generator a domain's `build` draws its procedural instance from (seed and configuration name)."""
    from ..core.canonical import digest
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(digest(name)[:8], 16)]))


def measured(report_fn, name: str):
    """`Mechanic.count` from a domain's measured mechanics report `report_fn(world, since_seq) -> {mechanic:
    {"exercised", "events"}}`: the number of recorded events behind the mechanic when it is exercised, else 0.
    The report is computed once per (world, log position, since_seq)."""
    def count(world, since_seq):
        key = ("dom3.mech", id(world.state), int(world.seq), int(since_seq), report_fn.__module__)
        cache = world.runtime.setdefault("dom3.mech", {})
        if key not in cache:
            cache.clear()
            cache[key] = report_fn(world, int(since_seq))
        row = cache[key].get(name) or {}
        if not row.get("exercised"):
            return 0
        ev = row.get("events") or {}
        n = sum(int(v) for v in ev.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
        return max(1, n)
    return count


from .base import ScriptedReference as _ScriptedReference, act_ as _act  # noqa: E402


class ReadingReference(_ScriptedReference):
    """A two-phase tool-only reference on the framework's `ScriptedReference`. Each cycle it first calls the
    read-only tools `reads(t)` asks for (their results arrive as its own tool results, nothing else), then
    `decide(t, got)` returns the cycle's actions; `handle(tag, result, action)` receives those actions' results. It
    holds no World: everything it knows came back from its own tool calls (checked by base.reads_world)."""
    burst = 96
    replan_s = 0
    cycle_s = 6 * HOUR

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.phase = "read"
        self.got: dict = {}
        self.next_cycle = 0

    def reads(self, t: int) -> list:
        return []

    def decide(self, t: int, got: dict) -> list:
        return []

    def handle(self, tag, result, action) -> None:
        pass

    def plan(self, obs):
        t = self.t
        if self.phase == "read":
            if t < self.next_cycle:
                return []
            self.got = {}
            self.phase = "decide"
            reqs = list(self.reads(t) or [])
            if reqs:
                return [_act(tool, f"read:{key}", **(args or {})) for key, tool, args in reqs]
        self.phase = "read"
        self.next_cycle = t + int(self.cycle_s)
        return list(self.decide(t, self.got) or [])

    def on_result(self, tag, result, action):
        if tag and tag.startswith("read:"):
            self.got[tag[5:]] = result
        else:
            self.handle(tag, result, action)


A = _act


def cached_evaluator(fn):
    """Evaluator results cached per (world state, simulated instant, principal, objective): the learner interface
    asks the evaluator after every action (its status and the legacy reward gate), and many actions share one
    simulated instant; a world rebuilt from the log (another state object) is evaluated afresh."""
    import functools
    import json as _json

    @functools.wraps(fn)
    def wrapped(world, principal, objective):
        key = (id(world.state), int(world.now), str(principal), _json.dumps(objective, sort_keys=True, default=str))
        cache = world.runtime.setdefault(f"dom3.eval.{fn.__module__}", {})
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = fn(world, principal, objective)
        return cache[key]
    return wrapped
