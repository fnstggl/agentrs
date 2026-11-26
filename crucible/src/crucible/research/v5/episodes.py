"""Driving a learner through `crucible.experience` and measuring what it did (MASTER_SPEC Parts 11.3, 15.2, 15.3).

The driver gives a lever learner (`learners.LeverLearner`) only what a participant observes: the lever names, ranges
and public defaults from the tool schemas, the world's brief, and every tick the change of the participant's own
cash balance as its observation shows it (its own signal: there is no reward in ``observation_only``). The learner
returns normalised settings; the driver executes them as ordinary tool calls (``rapid.levers`` or ``rig.set``) and
advances the clock one tick per decision.

What the instrument scores comes from the world's verifier (`ExperienceEnv.verify`, evaluator side, read by the
driver after every tick and never shown to the learner): the verified cumulative net income per tick. A world's
*normalised score* is (net - net of leaving the defaults alone) / (upper reference - net of leaving the defaults
alone), where the upper reference is the evaluator-side lever oracle of `crucible.rapid.oracles` (a search over
piecewise-constant schedules on the world's realised noise). Both references run on separately built copies of the
world, evaluator side.

Accounting per episode (Part 15.3): ticks, informative interactions (executed actions that changed the world,
counted by the adapter, operator side), action cost (seconds of simulated time the actions occupied), CPU seconds
of the driver process and wall seconds.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from .distribution import Registry, build_registry, canonical, sha, thaw
from .sampler import build_env, rapid_world

_DEFAULT_RE = re.compile(r"default\s+(-?[0-9.eE+-]+)")


class InterfaceError(ValueError):
    """The world offers no continuous lever interface a lever learner can drive."""


class LeverInterface:
    """The public lever interface of one world for one participant."""

    def __init__(self, env, agent: str):
        self.env, self.agent = env, agent
        obs = env.observe(agent)
        tools = {t["name"]: t for t in env.tools(agent)}
        self.kind = None
        names, lo, hi, dflt, integer = [], [], [], [], []
        if "rapid.levers" in tools:
            props = tools["rapid.levers"]["inputSchema"]["properties"]["values"]["properties"]
            for n, p in props.items():
                names.append(n)
                lo.append(float(p["minimum"]))
                hi.append(float(p["maximum"]))
                m = _DEFAULT_RE.search(p.get("description", ""))
                dflt.append(float(m.group(1)) if m else float(p["minimum"]))
                integer.append(p.get("type") == "integer")
            self.kind = "rapid"
        elif "rig.set" in tools:
            r = env.execute(agent, {"tool": "rig.spec", "args": {}})
            if not r.get("ok"):
                raise InterfaceError(f"rig.spec failed: {r.get('error')}")
            for lv in r["result"]["levers"]:
                names.append(lv["name"])
                lo.append(float(lv["min"]))
                hi.append(float(lv["max"]))
                dflt.append(float(lv["default"]))
                integer.append(bool(lv.get("integer")))
            self.kind = "rig"
            obs = env.observe(agent)
        if not names:
            raise InterfaceError("no continuous lever tool (rapid.levers or rig.set) is offered")
        self.names = names
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.default = np.array(dflt)
        self.integer = np.array(integer, dtype=bool)
        self.span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.tick_s = int(obs["numbers"].get("world.tick_s") or 3600)
        left = obs["numbers"].get("world.ticks_left")
        if left is None:
            left = obs.get("resources", {}).get("ticks_left")
        self.horizon = int(left) if left is not None else 60
        self.brief = " ".join(obs.get("text") or [])[:400]
        self._cash0 = self.cash(obs)
        self.current = self.default.copy()

    @staticmethod
    def cash(obs: Mapping) -> float:
        res = obs.get("resources") or {}
        for k in ("cash_cents", "balance_cents"):
            if isinstance(res.get(k), (int, float)):
                return float(res[k])
        nums = obs.get("numbers") or {}
        bal = [v for k, v in nums.items() if k.startswith("bank.balances.")]
        return float(sum(bal)) if bal else 0.0

    def u_default(self) -> np.ndarray:
        return np.clip((self.default - self.lo) / self.span, 0.0, 1.0)

    def context(self) -> dict:
        """What a learner is told at the start of an episode (public face only)."""
        return {"n": len(self.names), "u0": self.u_default(), "horizon": self.horizon,
                "key": sha([self.names, self.brief])[:16], "brief_digest": sha(self.brief)[:16],
                "brief_vec": brief_vector(self.brief)}

    def to_values(self, u: np.ndarray) -> dict:
        x = self.lo + np.clip(np.asarray(u, dtype=float)[: len(self.names)], 0.0, 1.0) * self.span
        x = np.where(self.integer, np.round(x), x)
        return {n: float(v) for n, v in zip(self.names, x)}

    def apply(self, u: np.ndarray) -> dict | None:
        vals = self.to_values(u)
        new = np.array([vals[n] for n in self.names])
        if np.allclose(new, self.current, atol=1e-9):
            return None
        if self.kind == "rapid":
            act = {"tool": "rapid.levers", "args": {"values": vals}}
        else:
            act = {"tool": "rig.set", "args": {"settings": vals}}
        r = self.env.execute(self.agent, act)
        if r.get("ok"):
            self.current = new
        return r


BRIEF_DIM = 16


def brief_vector(text: str, dim: int = BRIEF_DIM) -> np.ndarray:
    """A hashed bag of the brief's words (public text; lever names, which seeds draw, are not words of it)."""
    v = np.zeros(dim)
    for w in re.findall(r"[a-z]{3,}", text.lower()):
        v[int(sha(w)[:8], 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def verified_value(v: Mapping) -> float:
    """The verified objective quantity in an `ExperienceEnv.verify()` record (rapid outcome net/value, 4.0 task net,
    or a numeric domain/civilisation evaluation)."""
    out = v.get("outcome")
    if isinstance(out, Mapping):
        for k in ("value", "net"):
            if isinstance(out.get(k), (int, float)):
                return float(out[k])
    cur = v.get("current")
    if isinstance(cur, Mapping) and isinstance(cur.get("net_cents"), (int, float)):
        return float(cur["net_cents"])
    eps = v.get("episodes")
    if isinstance(eps, list) and eps and isinstance(eps[-1], Mapping) and \
            isinstance(eps[-1].get("net_cents"), (int, float)):
        return float(eps[-1]["net_cents"])          # a 4.0 task whose episode just ended
    ev = v.get("evaluation") or {}
    for k in ("score", "value", "progress"):
        if isinstance(ev.get(k), (int, float)) and not isinstance(ev.get(k), bool):
            return float(ev[k])
    raise ValueError("verify() record has no numeric outcome")


def run_episode(env, learner, *, learn: bool, rng: np.random.Generator, budget: int | None = None,
                agent: str | None = None, verify: bool = True, blind: bool = False) -> dict:
    """One episode of `learner` in `env`. `learn` says whether the learner may update itself at the end;
    `budget` caps the informative interactions granted (later setting changes are not executed). Returns the
    measured record (verified per-tick cumulative net, counts, compute). With ``verify=False`` (an episode held by
    the evaluator process, `evaluator.run_remote_episode`) the driver never calls the verifier; the evaluator records
    the verified trajectory itself. With ``blind=True`` the learner receives no consequences (its signal is 0 every
    tick): the within-episode-adaptation ablation of the instrument."""
    t0, c0 = time.perf_counter(), time.process_time()
    agent = agent or env.agents()[0]
    iface = LeverInterface(env, agent)
    learner.begin(iface.context(), rng)
    prev = iface.cash(env.observe(agent))
    traj, sig, interactions, cost_s, refused = [], [], 0, 0, 0
    for k in range(iface.horizon):
        u = learner.act(k)
        if u is not None:
            if budget is not None and interactions >= budget:
                refused += 1
            else:
                r = iface.apply(u)
                if r is not None:
                    if r.get("ok"):
                        interactions += 1
                        cost_s += int(r.get("cost_s") or 0)
        env.advance(iface.tick_s)
        obs = env.observe(agent)
        cash = iface.cash(obs)
        s = cash - prev
        prev = cash
        sig.append(s)
        learner.observe(k, 0.0 if blind else s)
        if verify:
            traj.append(verified_value(env.verify()))
    learner.end(learn=learn)
    return {"verified_cum": traj, "value": traj[-1] if traj else 0.0, "ticks": iface.horizon,
            "interactions": interactions, "refused_beyond_budget": refused, "action_cost_s": cost_s,
            "cpu_s": time.process_time() - c0, "wall_s": time.perf_counter() - t0, "observed_signal_sum": float(sum(sig)),
            "n_levers": len(iface.names)}


# ====================================================================== evaluator-side references
@lru_cache(maxsize=4096)
def _refs(entry_json: str, seed: int, diff_json: str, obj_json: str) -> tuple[float, float]:
    import json
    from ...rapid import oracles as OR
    entry, diff, obj = json.loads(entry_json), json.loads(diff_json), json.loads(obj_json)
    env = build_env(entry, seed, diff, obj)
    a = env.agents()[0]
    left = env.observe(a)["numbers"].get("world.ticks_left") or 60
    tick = env.observe(a)["numbers"].get("world.tick_s") or 3600
    for _ in range(int(left)):
        env.advance(int(tick))
    wait = verified_value(env.verify())
    raw = rapid_world(entry, seed, diff, obj)
    raw.reset()
    upper = float(OR.lever_oracle(raw)["value"])
    return wait, max(upper, wait)


def references(entry: Mapping, seed: int, difficulty: Mapping | None = None,
               objective: Mapping | None = None) -> dict:
    """Evaluator side: (net of leaving the defaults alone, upper reference) of one Tier 1 lever world."""
    e = thaw(entry)
    if e["source"] not in ("crucible.rapid", "generated"):
        raise ValueError("normalised scores exist for rapid and generated Tier 1 worlds")
    w, u = _refs(canonical(e), int(seed), canonical(difficulty or {}), canonical(objective or {"mode": "assigned"}))
    return {"wait": w, "upper": u}


SCORE_RANGE = (-1.0, 1.5)


def normalised(value: float, refs: Mapping) -> float:
    """(value - wait) / (upper - wait), clipped to `SCORE_RANGE`: a world whose upper reference barely improves on
    waiting would otherwise turn noise into huge scores (the oracle is a search, so scores above 1 are possible)."""
    span = float(refs["upper"]) - float(refs["wait"])
    if span <= 1e-9:
        return 0.0
    return float(np.clip((float(value) - float(refs["wait"])) / span, *SCORE_RANGE))


def lever_entries(registry: Registry | None = None, family_partition: str | None = None,
                  sources=("crucible.rapid",)) -> list[str]:
    """Tier 1 entries with a continuous lever interface, a net-income objective and the primitive-graph engine
    (where lever learners apply and the evaluator-side lever oracle gives a normalising upper reference)."""
    reg = registry or build_registry()
    return [w["id"] for w in reg.worlds(tier=1, family_partition=family_partition)
            if w["source"] in sources and w.get("lever_view") and w["objective"] == "maximize_net"
            and w["engine"] == "primitive_graph"]


def world(registry: Registry, entry_id: str, seed: int, difficulty: Mapping | None = None) -> tuple[Any, dict]:
    """(ExperienceEnv, references) of one evaluation world (evaluator side: any partition)."""
    e = registry[entry_id]
    env = build_env(e, seed, dict(difficulty or {}), {"mode": "assigned"})
    return env, references(e, seed, difficulty)


def score_episode(registry: Registry, entry_id: str, seed: int, learner, *, learn: bool, rng,
                  difficulty: Mapping | None = None, budget: int | None = None, blind: bool = False) -> dict:
    env, refs = world(registry, entry_id, seed, difficulty)
    rec = run_episode(env, learner, learn=learn, rng=rng, budget=budget, blind=blind)
    H = len(rec["verified_cum"])
    cum = np.array(rec["verified_cum"])
    q = max(1, H // 4)
    early = (cum[q - 1]) / q
    late = (cum[-1] - cum[-q - 1]) / q
    span = max(refs["upper"] - refs["wait"], 1e-9)
    rec.update({"entry": entry_id, "seed": int(seed), "score": normalised(rec["value"], refs),
                "early_rate": float(early) * H / span, "late_rate": float(late) * H / span, "refs": refs})
    rec.pop("verified_cum")
    return rec
