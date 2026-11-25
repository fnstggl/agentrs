"""Generic public-interface policies built from an `ActionSpace` alone: wait, uniformly random valid actions and
constant canonical actions (every lever at its minimum, middle or maximum; always the same option of a decision;
everything on one target or spread evenly). They know nothing about any world; admission uses them as the trivial
strategies a world must not let win, and learners can use them as baselines.

A policy is `policy(obs, space, rng) -> action | None` (None waits)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .actions import ActionSpace

Policy = Callable[[dict, ActionSpace, np.random.Generator], dict | None]


def wait(obs, space, rng):
    return None


def _enum(schema: dict, key: str) -> list:
    return list(schema["properties"][key].get("enum", []))


def random_action(space: ActionSpace, rng: np.random.Generator, kinds: list[str] | None = None) -> dict | None:
    ks = [k for k in sorted(space.kinds) if k not in ("program",) and (kinds is None or k in kinds)]
    if not ks:
        return None
    k = ks[int(rng.integers(len(ks)))]
    sch = space.kinds[k]
    if k == "levers":
        props = sch["properties"]["values"]["properties"]
        names = sorted(props)
        pick = [n for n in names if rng.random() < 0.7] or names[:1]
        return {"kind": "levers", "values": {n: float(rng.uniform(props[n]["minimum"], props[n]["maximum"]))
                                             for n in pick}}
    if k == "discrete":
        decs = space.limits.get("decisions") or {d: _enum(sch, "choice") for d in _enum(sch, "decision")}
        d = sorted(decs)[int(rng.integers(len(decs)))]
        opts = decs[d]
        return {"kind": "discrete", "decision": d, "choice": opts[int(rng.integers(len(opts)))]}
    if k == "allocate":
        names = sorted(sch["properties"]["amounts"]["properties"])
        total = float(space.limits.get("allocate_total", 1.0))
        w = rng.dirichlet(np.ones(len(names)))
        return {"kind": "allocate", "amounts": {n: float(total * x * 0.999) for n, x in zip(names, w)}}
    if k == "graph_edit":
        ops, nodes = _enum(sch, "op"), _enum(sch, "src")
        a, b = (int(x) for x in rng.choice(len(nodes), size=2, replace=False))
        return {"kind": "graph_edit", "op": ops[int(rng.integers(len(ops)))], "src": nodes[a], "dst": nodes[b],
                "weight": 1.0}
    if k == "query":
        ts = [t for t in _enum(sch, "target") if t not in ("tools", "history", "spec")] or ["spec"]
        return {"kind": "query", "target": ts[int(rng.integers(len(ts)))], "args": {}}
    if k == "experiment":
        d = sch["properties"]["design"]["properties"]
        sets = {n: float(rng.uniform(p["minimum"], p["maximum"])) for n, p in d["set"]["properties"].items()}
        return {"kind": "experiment", "design": {"set": sets, "measure": list(d["measure"]["items"]["enum"])[:1],
                                                 "ticks": 1, "replicates": 1}}
    if k == "message":
        to = _enum(sch, "to")
        return {"kind": "message", "to": to[int(rng.integers(len(to)))], "text": "hello"}
    if k == "coordinate":
        parties = list(sch["properties"]["with"]["items"]["enum"])
        plan = {pk: pv["enum"][int(rng.integers(len(pv["enum"])))]
                for pk, pv in sch["properties"]["plan"]["properties"].items()}
        return {"kind": "coordinate", "with": parties, "plan": plan}
    if k == "sequence":
        inner = space.limits.get("sequence_kinds", [x for x in space.kinds if x not in ("sequence", "program")])
        steps = [random_action(space, rng, [x for x in inner if x != "sequence"]) for _ in range(2)]
        steps = [s for s in steps if s is not None]
        return {"kind": "sequence", "steps": steps} if steps else None
    return None


def random_policy(obs, space, rng):
    return random_action(space, rng)


def canonical_actions(space: ActionSpace, max_per_kind: int = 6) -> list[tuple[str, dict]]:
    """Constant actions a trivial strategy would repeat: (label, action)."""
    out: list[tuple[str, dict]] = []
    if "levers" in space.kinds:
        props = space.kinds["levers"]["properties"]["values"]["properties"]
        for lab, f in (("levers_min", 0.0), ("levers_mid", 0.5), ("levers_max", 1.0)):
            out.append((lab, {"kind": "levers", "values": {n: float(p["minimum"] + f * (p["maximum"] - p["minimum"]))
                                                           for n, p in props.items()}}))
    if "discrete" in space.kinds:
        decs = space.limits.get("decisions") or {}
        for d in sorted(decs):
            opts = decs[d]
            pick = opts if len(opts) <= max_per_kind else [opts[int(i)] for i in
                                                            np.linspace(0, len(opts) - 1, max_per_kind)]
            for o in pick:
                out.append((f"always_{d}={o}", {"kind": "discrete", "decision": d, "choice": o}))
    if "allocate" in space.kinds:
        names = sorted(space.kinds["allocate"]["properties"]["amounts"]["properties"])
        total = float(space.limits.get("allocate_total", 1.0))
        out.append(("allocate_even", {"kind": "allocate", "amounts": {n: total / len(names) for n in names}}))
        for n in names[:max_per_kind]:
            out.append((f"allocate_all_{n}", {"kind": "allocate", "amounts": {n: total}}))
    return out


def constant(action: dict) -> Policy:
    def pol(obs, space, rng):
        return action
    return pol


def run(env, policy: Policy, rng: np.random.Generator | None = None, ticks: int | None = None,
        agents: dict | None = None) -> dict:
    """Run `policy` (for the first agent; `agents` maps the others to policies, default wait) until the horizon or
    `ticks`; returns the evaluator's outcome."""
    rng = rng or np.random.default_rng(0)
    others = agents or {}
    n = env.horizon - env.t if ticks is None else ticks
    obs = {a: env.observe(a) for a in env.agents()}
    for _ in range(n):
        if env.done:
            break
        acts = {}
        for a in env.agents():
            pol = policy if a == env.agents()[0] else others.get(a, wait)
            x = pol(obs[a], env.action_space(a), rng)
            if x is not None:
                acts[a] = x
        r = env.step(acts if len(env.agents()) > 1 else (acts.get(env.agents()[0])))
        obs = r.get("observations") or {env.agents()[0]: r["observation"]}
    return env.outcome()
