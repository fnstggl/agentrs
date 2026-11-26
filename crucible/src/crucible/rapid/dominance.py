"""The no-dominant-strategy check of admission (MASTER_SPEC Part 7.4), by black-box search over the full action
space with a fixed, prespecified budget (evaluator side).

A family is admitted only if no *simple* policy comes close to a genuinely adaptive public-information reference.
Simple policies know nothing about the mechanism; they are built from the `ActionSpace` alone and control every
learner seat of the world:

    constant open-loop settings   one composite setting held for the whole episode: lever values, one option per
                                  discrete decision, an allocation, a graph edit, a joint plan - every action kind
                                  that changes the world. Found by a grid of canonical settings, uniform random
                                  settings, and coordinate search started from the best of them
    reactive / threshold rules    "if reading k is above theta use setting A, else setting B" (theta at quantiles of
                                  k under A), and win-stay/lose-shift on observed net income
    repeat last best              explore-then-commit: hold each of the K best constants for m ticks, then repeat
                                  the one with the best observed net income for the rest of the episode

The search runs on copies of the scored world itself (common random numbers: every candidate sees the very noise
the reference saw), so the best constant found is the best *in hindsight* for that instance - the static-tuning
ceiling, stronger than any constant a learner could pick from public information. Every candidate's value is the
evaluator's outcome of a whole episode.

Decision H-D1 (recorded before any run, see the workstream H report): budget per seed = BUDGET (68 episodes);
a family passes when the best simple policy's gain over waiting is at most DOMINANCE_FRACTION = 0.8 of the
adaptive reference's gain, and the reference's gain is positive. Rationale: the reference must beat every simple
policy, including the hindsight-best constant, by at least 25% of its own improvement; that margin is larger than
the seed-to-seed spread of reference shares measured in the recovery audit (about 0.1-0.15), so a pass is not an
accident of one noise draw, and it is strict enough that a world in which tuning one or two constants (or a
threshold rule) is nearly optimal is rejected - the failure mode the audit found in 17 of 31 lever families.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .oracles import fork

DOMINANCE_FRACTION = 0.8
BUDGET = {"grid": 16, "random": 16, "coordinate": 16, "reactive": 16, "repeat_best": 4}
WORLD_KINDS = ("levers", "discrete", "allocate", "graph_edit", "coordinate")     # kinds that change the world


# ====================================================================== settings: one composite constant per seat
def _props(space, kind, field):
    return space.kinds[kind]["properties"][field]["properties"]


def _decisions(space) -> dict[str, list[str]]:
    if "discrete" not in space.kinds:
        return {}
    d = space.limits.get("decisions")
    if d:
        return {k: list(v) for k, v in sorted(d.items())}
    sch = space.kinds["discrete"]["properties"]
    return {k: list(sch["choice"]["enum"]) for k in sch["decision"]["enum"]}


class Coords:
    """The coordinates of a composite constant setting for one seat: one per lever (a fraction of its range), one
    categorical per discrete decision, an allocation (a weight per target and a total fraction), one categorical
    graph edit and one categorical joint plan."""

    def __init__(self, space, agent: str, agents: list[str]):
        self.space, self.agent = space, agent
        k = space.kinds
        self.levers = sorted(_props(space, "levers", "values").items()) if "levers" in k else []
        self.decisions = _decisions(space)
        self.alloc = sorted(_props(space, "allocate", "amounts")) if "allocate" in k else []
        self.alloc_total = float(space.limits.get("allocate_total", 1.0))
        self.edits: list[dict] = []
        if "graph_edit" in k:
            sch = k["graph_edit"]["properties"]
            nodes = list(sch["src"]["enum"])
            for op in sch["op"]["enum"]:
                for a in nodes[:8]:
                    for b in nodes[:8]:
                        if a != b:
                            self.edits.append({"kind": "graph_edit", "op": op, "src": a, "dst": b, "weight": 1.0})
        self.plans: list[dict] = []
        if "coordinate" in k:
            sch = k["coordinate"]["properties"]
            parties = [p for p in sch["with"]["items"]["enum"] if p != agent] or list(sch["with"]["items"]["enum"])
            fields = sorted(sch["plan"]["properties"].items())
            combos = [{}]
            for fk, fv in fields:
                combos = [dict(c, **{fk: o}) for c in combos for o in fv.get("enum", [None])[:12]][:64]
            self.plans = [{"kind": "coordinate", "with": parties, "plan": c} for c in combos]
        self.sequence = "sequence" in k and len([x for x in (self.levers, self.decisions, self.alloc) if x]) > 1

    def empty(self) -> bool:
        return not (self.levers or self.decisions or self.alloc or self.edits or self.plans)

    # a setting: {"u": [lever fractions], "d": {decision: option}, "w": [weights], "f": total fraction,
    #             "e": edit index or None, "p": plan index or None}
    def random(self, rng) -> dict:
        return {"u": [float(rng.random()) for _ in self.levers],
                "d": {d: o[int(rng.integers(len(o)))] for d, o in self.decisions.items()},
                "w": [float(x) for x in rng.dirichlet(np.ones(len(self.alloc)))] if self.alloc else [],
                "f": float(rng.choice([0.25, 0.5, 0.75, 1.0])),
                "e": int(rng.integers(len(self.edits))) if self.edits and rng.random() < 0.5 else None,
                "p": int(rng.integers(len(self.plans))) if self.plans else None}

    def canonical(self) -> list[dict]:
        base = {"u": [0.5] * len(self.levers), "d": {d: o[0] for d, o in self.decisions.items()},
                "w": [1.0 / len(self.alloc)] * len(self.alloc) if self.alloc else [], "f": 1.0, "e": None,
                "p": 0 if self.plans else None}
        out = []
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            out.append(dict(base, u=[f] * len(self.levers)))
        for j in range(len(self.levers)):
            for f in (0.0, 1.0):
                u = [0.5] * len(self.levers)
                u[j] = f
                out.append(dict(base, u=u))
        for d, opts in self.decisions.items():
            for o in opts[:8]:
                out.append(dict(base, d=dict(base["d"], **{d: o})))
        for j in range(len(self.alloc)):
            w = [0.0] * len(self.alloc)
            w[j] = 1.0
            out.append(dict(base, w=w))
        for f in (0.25, 0.5):
            out.append(dict(base, f=f))
        for p in range(min(len(self.plans), 8)):
            out.append(dict(base, p=p))
        for e in range(min(len(self.edits), 4)):
            out.append(dict(base, e=e))
        return out

    def neighbours(self, s: dict, step: float, rng) -> list[dict]:
        out = []
        for j in range(len(self.levers)):
            for sgn in (-1, 1):
                u = list(s["u"])
                u[j] = float(np.clip(u[j] + sgn * step, 0.0, 1.0))
                if u != s["u"]:
                    out.append(dict(s, u=u))
        for d, opts in self.decisions.items():
            o = opts[int(rng.integers(len(opts)))]
            if o != s["d"].get(d):
                out.append(dict(s, d=dict(s["d"], **{d: o})))
        if len(self.alloc) > 1:
            a, b = (int(x) for x in rng.choice(len(self.alloc), 2, replace=False))
            w = list(s["w"])
            m = min(w[a], step)
            w[a] -= m
            w[b] += m
            out.append(dict(s, w=w))
        if self.alloc:
            out.append(dict(s, f=float(np.clip(s["f"] + (step if rng.random() < 0.5 else -step), 0.05, 1.0))))
        if self.plans:
            out.append(dict(s, p=int(rng.integers(len(self.plans)))))
        if self.edits:
            out.append(dict(s, e=None if s["e"] is not None else int(rng.integers(len(self.edits)))))
        return out

    def actions(self, s: dict) -> list[dict]:
        acts = []
        if self.levers:
            acts.append({"kind": "levers", "values": {n: float(p["minimum"] + u * (p["maximum"] - p["minimum"]))
                                                      for (n, p), u in zip(self.levers, s["u"])}})
        for d, o in sorted(s["d"].items()):
            acts.append({"kind": "discrete", "decision": d, "choice": o})
        if self.alloc:
            tot = self.alloc_total * s["f"] * 0.999
            ws = np.asarray(s["w"], dtype=float)
            ws = ws / ws.sum() if ws.sum() > 0 else np.full(len(ws), 1.0 / len(ws))
            acts.append({"kind": "allocate", "amounts": {n: float(tot * x) for n, x in zip(self.alloc, ws)}})
        if s.get("e") is not None and self.edits:
            acts.append(self.edits[s["e"]])
        if s.get("p") is not None and self.plans:
            acts.append(self.plans[s["p"]])
        return acts


class Joint:
    """A setting for every learner seat of a world."""

    def __init__(self, env):
        self.agents = env.agents()
        self.coords = {a: Coords(env.action_space(a), a, self.agents) for a in self.agents}

    def empty(self) -> bool:
        return all(c.empty() for c in self.coords.values())

    def random(self, rng) -> dict:
        return {a: c.random(rng) for a, c in self.coords.items()}

    def canonical(self) -> list[dict]:
        per = {a: c.canonical() for a, c in self.coords.items()}
        n = max(len(v) for v in per.values())
        out = [{a: per[a][min(i, len(per[a]) - 1)] for a in self.agents} for i in range(n)]
        if len(self.agents) > 1:                     # seats spread over the options of each decision by position
            for d in sorted(self.coords[self.agents[0]].decisions):
                spread = {}
                for i, a in enumerate(self.agents):
                    c = self.coords[a]
                    opts = c.decisions.get(d)
                    s = per[a][0]
                    spread[a] = dict(s, d=dict(s["d"], **({d: opts[i % len(opts)]} if opts else {})))
                out.append(spread)
        return out

    def neighbours(self, s: dict, step: float, rng) -> list[dict]:
        out = []
        for a, c in self.coords.items():
            for nb in c.neighbours(s[a], step, rng):
                out.append(dict(s, **{a: nb}))
        return out


# ====================================================================== policies over settings
Policy = Callable[[int, dict], dict]          # (t, observations by agent) -> {agent: action}


def _emit(joint: Joint, setting: dict, t: int) -> dict:
    out = {}
    for a, c in joint.coords.items():
        acts = c.actions(setting[a])
        if not acts:
            continue
        if len(acts) == 1:
            out[a] = acts[0]
        elif "sequence" in c.space.kinds:
            inner = c.space.limits.get("sequence_kinds")
            steps = [x for x in acts if inner is None or x["kind"] in inner]
            out[a] = {"kind": "sequence", "steps": steps} if steps else acts[t % len(acts)]
        else:
            out[a] = acts[t % len(acts)]                 # no sequences here: cycle through the components
    return out


def constant(joint: Joint, setting: dict) -> Policy:
    return lambda t, obs: _emit(joint, setting, t)


def _net(obs: dict) -> float:
    r = obs.get("resources", {})
    return float(r.get("revenue_cents_last_tick", 0.0)) - float(r.get("cost_cents_last_tick", 0.0))


def threshold(joint: Joint, key: str, theta: float, a: dict, b: dict) -> Policy:
    a0 = joint.agents[0]

    def pol(t, obs):
        x = obs[a0]["numbers"].get(key)
        return _emit(joint, a if (x is not None and x > theta) else b, t)
    return pol


def win_stay_lose_shift(joint: Joint, a: dict, b: dict) -> Policy:
    a0 = joint.agents[0]
    state = {"cur": a, "last": None}

    def pol(t, obs):
        net = _net(obs[a0])
        if state["last"] is not None and net < state["last"]:
            state["cur"] = b if state["cur"] is a else a
        state["last"] = net
        return _emit(joint, state["cur"], t)
    return pol


def repeat_last_best(joint: Joint, cands: list[dict], hold: int) -> Policy:
    a0 = joint.agents[0]
    state = {"scores": [[] for _ in cands], "best": None}

    def pol(t, obs):
        k = (t - 1) // hold
        if t >= 1 and k < len(cands) and (t - 1) % hold >= hold // 2:
            state["scores"][k].append(_net(obs[a0]))
        i = t // hold
        if i < len(cands):
            return _emit(joint, cands[i], t)
        if state["best"] is None:
            means = [np.mean(s) if s else -math.inf for s in state["scores"]]
            state["best"] = cands[int(np.argmax(means))]
        return _emit(joint, state["best"], t)
    return pol


def play(env, pol: Policy) -> tuple[float, list[dict]]:
    """Run a joint policy on a fork of `env` from its current point; (evaluator value, first agent's observations)."""
    e = fork(env)
    ags = e.agents()
    obs = {a: e.observe(a) for a in ags}
    trace = []
    while not e.done:
        acts = pol(e.t, obs)
        r = e.step(acts if len(ags) > 1 else acts.get(ags[0]))
        obs = r.get("observations") or {ags[0]: r["observation"]}
        trace.append(obs[ags[0]])
    return float(e.outcome()["value"]), trace


# ====================================================================== the search
def search(env, seed: int = 0, budget: dict | None = None) -> dict:
    """Best simple policy on `env` (a fresh world) within the budget. Returns {"best_value", "best_kind",
    "best_label", "episodes", "by_arm": {arm: best value}, "default": value of waiting}."""
    budget = dict(BUDGET, **(budget or {}))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7919]))
    joint = Joint(env)
    v_wait, _ = play(env, lambda t, obs: {})
    res = {"default": v_wait, "episodes": 1, "by_arm": {"wait": v_wait}}
    if joint.empty():
        return dict(res, best_value=v_wait, best_kind="wait", best_label="wait")
    scored: list[tuple[float, dict, str]] = []

    def try_setting(s, arm, label):
        v, tr = play(env, constant(joint, s))
        res["episodes"] += 1
        scored.append((v, s, f"{arm}:{label}"))
        res["by_arm"][arm] = max(res["by_arm"].get(arm, -math.inf), v)
        return v, tr

    canon = joint.canonical()
    if len(canon) > budget["grid"]:
        idx = sorted(rng.choice(len(canon), budget["grid"], replace=False))
        canon = [canon[0]] + [canon[i] for i in idx if i != 0][:budget["grid"] - 1]
    for i, s in enumerate(canon):
        try_setting(s, "grid", str(i))
    for i in range(budget["random"]):
        try_setting(joint.random(rng), "random", str(i))
    best = max(scored, key=lambda x: x[0])
    step, n = 0.25, 0
    while n < budget["coordinate"]:
        nbs = joint.neighbours(best[1], step, rng)
        rng.shuffle(nbs)
        improved = False
        for nb in nbs:
            if n >= budget["coordinate"]:
                break
            v, _ = try_setting(nb, "coordinate", str(n))
            n += 1
            if v > best[0]:
                best = (v, nb, f"coordinate:{n}")
                improved = True
                break
        if not improved:
            step /= 2
            if step < 0.03:
                break
    ranked = sorted(scored, key=lambda x: -x[0])
    top = [s for _, s, _ in ranked[:6]]
    _, trace = play(env, constant(joint, top[0]))
    keys = sorted({k for o in trace for k in o["numbers"]})
    n = 0
    if keys and len(top) > 1:
        while n < budget["reactive"] - 2:
            k = keys[int(rng.integers(len(keys)))]
            xs = [o["numbers"][k] for o in trace if k in o["numbers"]]
            if not xs:
                n += 1
                continue
            theta = float(np.quantile(xs, float(rng.choice([0.25, 0.5, 0.75]))))
            a, b = top[0], top[1 + int(rng.integers(len(top) - 1))]
            if rng.random() < 0.5:
                a, b = b, a
            v, _ = play(env, threshold(joint, k, theta, a, b))
            res["episodes"] += 1
            n += 1
            res["by_arm"]["reactive"] = max(res["by_arm"].get("reactive", -math.inf), v)
            scored.append((v, top[0], f"reactive:{k}>{theta:.3g}"))
    for j in range(min(2, len(top) - 1)):
        v, _ = play(env, win_stay_lose_shift(joint, top[0], top[1 + j]))
        res["episodes"] += 1
        res["by_arm"]["reactive"] = max(res["by_arm"].get("reactive", -math.inf), v)
        scored.append((v, top[0], f"reactive:win-stay-lose-shift/{j}"))
    H = env.horizon
    for K, m in ((3, 2), (3, 4), (5, 2), (5, 4))[:budget["repeat_best"]]:
        if K * m >= H:
            continue
        v, _ = play(env, repeat_last_best(joint, top[:K], m))
        res["episodes"] += 1
        res["by_arm"]["repeat_best"] = max(res["by_arm"].get("repeat_best", -math.inf), v)
        scored.append((v, top[0], f"repeat_best:K={K},m={m}"))
    bv, bs, bl = max(scored, key=lambda x: x[0])
    if v_wait > bv:
        bv, bl = v_wait, "wait"
    return dict(res, best_value=float(bv), best_kind=bl.split(":")[0], best_label=bl)


def check(v_default: float, v_reference: float, simple: dict, fraction: float = DOMINANCE_FRACTION) -> dict:
    """The per-seed verdict: the reference improves on waiting, and the best simple policy's improvement is at
    most `fraction` of the reference's."""
    g_ref = v_reference - v_default
    g_simple = simple["best_value"] - v_default
    ratio = g_simple / g_ref if g_ref > 0 else math.inf
    return {"ok": bool(g_ref > 1e-9 and ratio <= fraction), "reference_gain": float(g_ref),
            "best_simple_gain": float(g_simple), "simple_over_reference": float(ratio) if g_ref > 0 else None,
            "threshold": fraction, "best_simple": simple["best_label"], "episodes": simple["episodes"],
            "by_arm": {k: float(v - v_default) for k, v in simple["by_arm"].items()}}
