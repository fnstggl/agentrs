"""An external learner adapter written only against the public `crucible.experience` interface (SP-11.6).

This file is what a future learner (a CAM version, or anything else) writes to train in Crucible: it imports nothing
from Crucible except `crucible.experience`'s public names and changes nothing in the environment. It exposes the
batched shape of the CAM Unified contract v2 `EnvAdapter` (reset / observe / available_tools / execute /
collect_events / advance_time / goals / close; executing never moves the clock; durations in days) on top of
`make(tier, name, seed, reward_mode)`, and includes a tiny external learner (an epsilon-greedy value table over tool
names, updated from the learner's own reading of its balance) to show experience flowing out and parameters changing.

    from experience_external_adapter import ExperienceEnvAdapter, Action, TableLearner
    env = ExperienceEnvAdapter([{"tier": 1, "name": "mech:scm_additive"}, {"tier": 0, "name": "tiny"}])
    batch = env.reset(seeds=[0, 1])
    ...

See docs/v5/LEARNER_GUIDE.md.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from crucible.experience import ExperienceStore, make, validate_observation

DAY = 86400.0


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict
    returns: str = ""
    cost_hint: float = 0.0
    duration_hint: float = 0.0          # days


@dataclass
class Action:
    tool: str
    args: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    ok: bool
    observations: list
    error: dict | None
    cost: float                         # not measured here: money shows in the observations (bank.balances.*)
    duration: float                     # days (the action's declared cost_s)
    started_t: float                    # days
    finished_t: float                   # days
    action_seq: int | None = None


@dataclass
class ObservationBatch:
    env_ids: list
    observations: list
    t: list
    done: list


def _balance(obs: dict) -> float:
    return float(sum(v for k, v in obs["numbers"].items() if k.startswith("bank.balances.")))


class ExperienceEnvAdapter:
    """B worlds, one agent each, behind a batched CAM-Unified-style adapter."""

    def __init__(self, specs: list[dict], reward_mode: str = "observation_only"):
        self.specs = [dict(s) for s in specs]
        self.reward_mode = reward_mode
        self.envs: list = []
        self.agents: list[str] = []
        self.cursors: list[int] = []

    def reset(self, seeds: list[int]) -> ObservationBatch:
        self.close()
        self.envs, self.agents, self.cursors = [], [], []
        for spec, seed in zip(self.specs, seeds):
            kw = dict(spec.get("kw") or {})
            env = make(int(spec["tier"]), spec["name"], int(seed), self.reward_mode, **kw)
            self.envs.append(env)
            self.agents.append(env.agents()[0])
        obs = self.observe()
        self.cursors = [o["cursor"] for o in obs]
        return ObservationBatch(list(range(len(self.envs))), obs, [o["t"] / DAY for o in obs],
                                [False] * len(self.envs))

    def observe(self) -> list[dict]:
        out = []
        for env, a in zip(self.envs, self.agents):
            o = env.observe(a)
            errs = validate_observation(o)
            if errs:
                raise ValueError(f"invalid observation: {errs[:3]}")
            out.append(o)
        return out

    def available_tools(self) -> list[list[ToolSchema]]:
        out = []
        for env, a in zip(self.envs, self.agents):
            out.append([ToolSchema(t["name"], t.get("description", ""), t.get("inputSchema", {}),
                                   duration_hint=float((t.get("annotations") or {}).get("costSeconds", 0)) / DAY)
                        for t in env.tools(a)])
        return out

    def execute(self, actions: list[Action | None]) -> list[ExecutionResult | None]:
        out = []
        for env, a, act in zip(self.envs, self.agents, actions):
            if act is None:
                out.append(None)
                continue
            r = env.execute(a, {"tool": act.tool, "args": dict(act.args)})
            t0 = float(r["t"]) / DAY
            out.append(ExecutionResult(bool(r["ok"]), [r.get("result")] if r["ok"] else [],
                                       None if r["ok"] else r.get("error"), 0.0, float(r["cost_s"]) / DAY, t0, t0,
                                       r.get("action_seq")))
        return out

    def collect_events(self, since: list[int]) -> tuple[list[list[dict]], list[int]]:
        evs, curs = [], []
        for env, a, c in zip(self.envs, self.agents, since):
            e, nxt = env.observe_since(a, int(c))
            evs.append(e)
            curs.append(nxt)
        return evs, curs

    def advance_time(self, days: list[float]) -> None:
        for env, d in zip(self.envs, days):
            env.advance(int(round(float(d) * DAY)))

    def goals(self) -> list[str | None]:
        return [env.observe(a)["objective"] for env, a in zip(self.envs, self.agents)]

    def rewards(self) -> list[float]:
        return [float(env.reward(a)) for env, a in zip(self.envs, self.agents)]

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self.envs = []


class TableLearner:
    """A deliberately small external learner: epsilon-greedy over tool names, valued by the learner's own measure of
    progress (the change of its observed balance). Arguments come from the tool's JSON schema (defaults, bounds,
    and names the learner has read in earlier results, e.g. lever names from rig.spec)."""

    def __init__(self, seed: int = 0, epsilon: float = 0.3, lr: float = 0.2):
        self.rng = random.Random(seed)
        self.epsilon, self.lr = epsilon, lr
        self.q: dict[str, float] = {}
        self.updates = 0
        self.names: dict[str, tuple[float, float]] = {}

    def remember(self, result: Any) -> None:
        if isinstance(result, dict):
            if isinstance(result.get("name"), str) and isinstance(result.get("min"), (int, float)) \
                    and isinstance(result.get("max"), (int, float)):
                self.names[result["name"]] = (float(result["min"]), float(result["max"]))
            for v in result.values():
                self.remember(v)
        elif isinstance(result, list):
            for v in result[:200]:
                self.remember(v)

    def _args(self, schema: dict) -> dict:
        out = {}
        props = schema.get("properties") or {}
        for k in schema.get("required") or []:
            p = props.get(k, {})
            if "default" in p:
                out[k] = p["default"]
            elif "enum" in p:
                out[k] = self.rng.choice(p["enum"])
            elif p.get("type") in ("number", "integer") and "minimum" in p and "maximum" in p:
                v = self.rng.uniform(p["minimum"], p["maximum"])
                out[k] = int(v) if p["type"] == "integer" else v
            elif p.get("type") == "object" and isinstance(p.get("additionalProperties"), dict) and self.names:
                out[k] = {n: self.rng.uniform(lo, hi) for n, (lo, hi) in self.names.items()}
            else:
                return {}
        return out

    def act(self, obs: dict, tools: list[ToolSchema]) -> Action | None:
        usable = [t for t in tools if not set(t.parameters.get("required") or []) - {"settings"} or
                  t.name in ("rig.set",)]
        if not usable:
            return None
        if self.rng.random() < self.epsilon or not self.q:
            t = self.rng.choice(usable)
        else:
            t = max(usable, key=lambda x: (self.q.get(x.name, 0.0), x.name))
        args = self._args(t.parameters)
        if set(t.parameters.get("required") or []) - set(args):
            return Action("rig.spec" if any(x.name == "rig.spec" for x in tools) else usable[0].name, {})
        return Action(t.name, args)

    def learn(self, tool: str, before: dict, after: dict) -> None:
        g = math.copysign(math.log1p(abs(_balance(after) - _balance(before))), _balance(after) - _balance(before))
        self.q[tool] = self.q.get(tool, 0.0) + self.lr * (g - self.q.get(tool, 0.0))
        self.updates += 1


def run(specs: list[dict], seeds: list[int], steps: int, store_dir: str | None = None,
        reward_mode: str = "observation_only") -> dict:
    """Drive `TableLearner` through `ExperienceEnvAdapter` for `steps` decisions per world; returns counts."""
    env = ExperienceEnvAdapter(specs, reward_mode)
    batch = env.reset(seeds)
    learners = [TableLearner(seed=s) for s in seeds]
    store = ExperienceStore(store_dir) if store_dir else None
    obs = batch.observations
    cursors = [o["cursor"] for o in obs]
    stats = {"actions": 0, "ok": 0, "events": 0, "records": 0, "rewards": 0.0}
    for step in range(steps):
        tools = env.available_tools()
        acts = [L.act(o, ts) for L, o, ts in zip(learners, obs, tools)]
        res = env.execute(acts)
        env.advance_time([max(1.0 / 24, (r.duration if r else 0.0)) for r in res])
        rew = env.rewards()
        nxt = env.observe()
        evs, cursors = env.collect_events(cursors)
        for i, (L, a, r) in enumerate(zip(learners, acts, res)):
            if a is None:
                continue
            stats["actions"] += 1
            stats["ok"] += int(r.ok)
            if r.ok:
                L.remember(r.observations[0] if r.observations else None)
            L.learn(a.tool, obs[i], nxt[i])
            if store is not None:
                store.append({"format": "external-adapter/1", "env": i, "agent": env.agents[i], "step": step,
                              "observation": obs[i], "action": {"tool": a.tool, "args": a.args},
                              "result": {"ok": r.ok, "error": r.error, "action_seq": r.action_seq},
                              "reward": rew[i], "events": len(evs[i])})
                stats["records"] += 1
        stats["events"] += sum(len(e) for e in evs)
        stats["rewards"] += sum(rew)
        obs = nxt
    stats["updates"] = sum(L.updates for L in learners)
    stats["q_tables"] = [dict(L.q) for L in learners]
    if store is not None:
        store.close()
    env.close()
    return stats
