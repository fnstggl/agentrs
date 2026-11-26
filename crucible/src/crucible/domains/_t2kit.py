"""Shared helpers of the Tier 2 domains of workstream I-1 (startup, software, content, community, research).

* `ordered(names)`: a module list in the civilisation generator's genesis order (`crucible.civ.generator.ORDER`), with
  modules the civilisation does not know appended in the given order - a domain world is a slice of the
  civilisation's composition (D-17), never a fork of it.
* `ScriptPolicy`: a scripted, tool-only reference policy written as a generator. The script yields one action
  (`{"tool", "args"}`) at a time and receives that action's result dict; yielding `None` waits one turn. It sees
  nothing but the results of its own calls (no World object, no spec, no state): references are
  observation-limited by construction. Every instance carries `kind = "scripted"`.
* `drive(env, agent, policy, days, dt)`: runs a policy through a `crucible.experience` environment (observe ->
  act -> execute, advance when the policy waits) for `days` simulated days.
* `event_counts(world, types, ...)`: evaluator-side counting of recorded events (mechanics are measured on the
  log, never asserted).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable

DAY = 86400
SCRIPTED = {"scripted": True, "learner": False, "evidence_of_learning": False, "kind": "scripted_reference"}


def ordered(names: Iterable[str]) -> list[str]:
    from ..civ.generator import ORDER
    names = list(dict.fromkeys(names))
    known = [m for m in ORDER if m in names]
    return known + [m for m in names if m not in known]


def recipe(modules: dict[str, dict], params: dict, extra_order: Iterable[str] = ()) -> dict:
    """The 4.0 world-family form {"modules", "module_list", "params"} of a domain world."""
    names = ordered(list(modules) + [m for m in extra_order if m not in modules])
    return {"modules": {m: dict(modules.get(m) or {}) for m in names}, "module_list": names,
            "params": {"cost_model": "deterministic", **params}}


class ScriptPolicy:
    """Generator-driven scripted reference (see the module doc). `script(call_log)` must be a generator function
    taking this policy; it yields actions and receives results."""

    kind = "scripted"
    name = "scripted-reference"
    version = "1"
    burst = 60          # actions per decision point (the script yields None to wait for the next one)

    def __init__(self, script: Callable[["ScriptPolicy"], Any], label: str = "reference"):
        self.label = label
        self._script = script
        self._gen = None
        self._pending: dict | None = None
        self.log: list[dict] = []
        self.failures: list[dict] = []
        self.done = False
        self.t = 0
        self._last: dict | None = None
        self.actor = ""
        self.tools: list[dict] = []

    # 4.0 Policy protocol
    def reset(self, actor: str, rng=None) -> None:
        self.actor = actor
        self._gen = self._script(self)
        self._pending = None

    def act(self, obs: dict, tools: list[dict] | None = None) -> dict | None:
        if tools is not None or (isinstance(obs, dict) and obs.get("tools") is not None):
            self.tools = tools if tools is not None else obs.get("tools")
        if self._gen is None:
            self.reset(obs.get("agent", ""))
        self.t = int(obs.get("t", self.t)) if isinstance(obs, dict) else self.t
        if self.done:
            return None
        try:
            if self._pending is None:
                act = next(self._gen)
            else:
                res, self._pending = self._pending, None
                act = self._gen.send(res)
        except StopIteration:
            self.done = True
            return None
        self._last = act
        return act

    def observe(self, reward, result, next_obs, done) -> None:
        """The framework's policy protocol (`crucible.domains.base.drive`): the result of the last action."""
        if self._last is not None:
            self.observe_result(self._last, result or {})
            self._last = None

    def observe_result(self, action: dict, result: dict) -> None:
        ok = bool(result.get("ok"))
        self.log.append({"t": self.t, "tool": action.get("tool"), "ok": ok})
        if not ok:
            self.failures.append({"t": self.t, "tool": action.get("tool"), "error": result.get("error")})
        self._pending = result

    def usage(self) -> dict:
        return {}


def drive(env, agent: str, policy: ScriptPolicy, days: float, dt: int = 3600, max_actions_per_turn: int = 40,
          on_turn: Callable[[Any], None] | None = None) -> dict:
    """Run `policy` as `agent` through an ExperienceEnv for `days` simulated days. Each turn the policy acts until it
    waits (at most `max_actions_per_turn` actions), then the world advances `dt` seconds."""
    policy.reset(agent)
    now = (lambda: int(env.world.now)) if hasattr(env, "world") else (lambda: int(env.now))
    end = now() + int(days * DAY)
    actions = 0
    while now() < end:
        for _ in range(int(max_actions_per_turn)):
            obs = env.observe(agent)
            act = policy.act(obs, None)
            if act is None:
                break
            res = env.execute(agent, act)
            policy.observe_result(act, res)
            policy._last = None
            actions += 1
        if on_turn is not None:
            on_turn(env)
        env.advance(min(int(dt), end - now()))
    return {"actions": actions, "failures": list(policy.failures), "t": now()}


def ok(res: dict | None) -> Any:
    """The result of a successful action, else None."""
    if isinstance(res, dict) and res.get("ok"):
        return res.get("result")
    return None


def event_counts(world, types: Iterable[str] | None = None, *, actors: Callable[[str], bool] | None = None,
                 since: int = 0) -> Counter:
    """Counts of recorded events by type (optionally only some types, only events whose actor passes `actors`)."""
    want = set(types) if types is not None else None
    c: Counter = Counter()
    for e in world.events_range(since, world.seq):
        if want is not None and e.type not in want:
            continue
        if actors is not None and not actors(e.actor):
            continue
        c[e.type] += 1
    return c


def is_population(principal: str | None) -> bool:
    """A principal of an individual of the one population (`p<i>`)."""
    return isinstance(principal, str) and principal.startswith("p") and principal[1:].isdigit()


def mechanic(exercised: bool, **evidence) -> dict:
    return {"exercised": bool(exercised), **evidence}


def decision_provenance(world, kind_prefixes: tuple[str, ...], tables: tuple[str, ...] | None = None) -> dict:
    """Evaluator side: decision batches of the given kinds (on any column table, or only `tables`) and how many
    of them are organic population decisions - listed by a population session (`pop.session` batch
    `extra.decisions`, i.e. recorded while the population's own tick ran the activity) with registered decision
    provenance. Everything else counts as forced."""
    models = world.state.tables.get("pop_decision_models", {})
    listed: set[str] = set()
    batches = []
    for e in world.events_range(0, world.seq):
        if e.type != "cols.batch":
            continue
        p = e.payload
        if p["table"] == "pop" and p["kind"] == "pop.session":
            listed.update((p.get("extra") or {}).get("decisions") or [])
        elif p["kind"].startswith(tuple(kind_prefixes)) and (tables is None or p["table"] in tables):
            batches.append(p)
    organic = 0
    for p in batches:
        prov = (p.get("extra") or {}).get("decision") or {}
        key = f"{prov.get('model_id')}@{prov.get('model_version')}"
        if p["ref"] in listed and key in models:
            organic += 1
    return {"batches": len(batches), "organic": organic, "forced": len(batches) - organic,
            "rows": int(sum(int(p.get("n", 0)) for p in batches))}


def observation_leaks(obs: dict, forbidden: Iterable[str]) -> list[str]:
    """Keys of an observation (recursively) or strings in it that name evaluator quantities."""
    bad = []
    words = tuple(forbidden)

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k in words:
                    bad.append(f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x[:200]):
                walk(v, f"{path}[{i}]")
    walk(obs, "obs")
    return bad
