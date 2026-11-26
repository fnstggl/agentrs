"""Evaluator side of generated goals (`crucible.rapid.goals`): scoring an episode against a goal, and validating
that a goal is feasible without guaranteeing success.

    evaluate(env, goal)   -> {"value", "success", ...}    from the episode's public trajectory, its ledger and its
                                                          journal; predict/identify goals are answered against the
                                                          counterfactual (the held-out condition run on a copy of
                                                          the world from its start, same noise)
    validate(make_env, goal) -> {"feasible", "witness", ...}   whether some policy meets the goal: the dominance
                                                          search's simple policies and the family's reference are
                                                          tried on copies; for predict/identify goals, whether the
                                                          answer is determined (the counterfactual is computable and
                                                          the effect is larger than the world's noise, or clearly
                                                          null)

A feasible goal can still be missed by a participant: feasibility says a policy exists, not that one is easy to
find. Nothing here is reachable from a world or a participant.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from . import goals as G


def _series(env, reading: str) -> list[tuple[int, float]]:
    a = env.agents()[0]
    return [(int(r["t"]), float(r[reading])) for r in env._last_numbers[a] if reading in r]


def _journal_answer(env, goal: dict, field: str):
    for m in reversed(env._ledger["journal"]):
        c = m.get("content") or {}
        if isinstance(c, dict) and c.get("goal") == goal["id"] and field in c:
            try:
                return float(c[field])
            except (TypeError, ValueError):
                return None
    return None


def _initial(env):
    """A copy of the world at its start (same seed, same noise): the counterfactual's starting point."""
    from .env import Env
    e = Env(env.family, env.seed, env._config, env.difficulty, world_seed=env.world_seed, env_id=env.env_id)
    return e


def counterfactual(env, actions: list[dict], ticks: int, reading: str) -> float | None:
    """Average of `reading` over `ticks` ticks of a copy of the world from its start with `actions` imposed at
    every tick (as one sequence where the world offers sequences, else in rotation)."""
    e = _initial(env)
    seq = "sequence" in e.action_space().kinds
    vals = []
    for k in range(ticks):
        if e.done:
            break
        if not actions:
            act = None
        elif len(actions) == 1:
            act = actions[0]
        else:
            act = {"kind": "sequence", "steps": actions} if seq else actions[k % len(actions)]
        x = e.step(act)["observation"]["numbers"].get(reading)
        if x is not None:
            vals.append(float(x))
    return float(np.mean(vals)) if vals else None


def _noise_scale(env, reading: str, ticks: int) -> float:
    """Spread of the reading's window average across noise replicates of waiting (the world's own noise)."""
    from .env import Env
    outs = []
    for r in range(3):
        e = Env(env.family, env.seed, env._config, env.difficulty, world_seed=env.world_seed + 7919 * (r + 1))
        vals = []
        for _ in range(ticks):
            if e.done:
                break
            x = e.step(None)["observation"]["numbers"].get(reading)
            if x is not None:
                vals.append(float(x))
        if vals:
            outs.append(float(np.mean(vals)))
    return float(np.std(outs)) + 1e-9 if len(outs) > 1 else 1.0


def truth(env, goal: dict) -> dict:
    s = goal["spec"]
    if goal["type"] == "predict_heldout":
        acts = G.setting_actions(s["setting"])
        return {"value": counterfactual(env, acts, s["window"], s["reading"])}
    if goal["type"] == "identify_effect":
        if "lever" in s:
            hi = [{"kind": "levers", "values": {s["lever"]: s["high"]}}]
            lo = [{"kind": "levers", "values": {s["lever"]: s["low"]}}]
        else:
            hi = [{"kind": "discrete", "decision": s["decision"], "choice": s["high"]}]
            lo = [{"kind": "discrete", "decision": s["decision"], "choice": s["low"]}]
        a = counterfactual(env, hi, s["window"], s["reading"])
        b = counterfactual(env, lo, s["window"], s["reading"])
        sc = _noise_scale(env, s["reading"], s["window"])
        diff = (a or 0.0) - (b or 0.0)
        sign = 0 if abs(diff) <= 2.0 * sc else (1 if diff > 0 else -1)
        return {"value": sign, "difference": diff, "noise_scale": sc}
    return {}


def evaluate(env, goal: dict) -> dict:
    """Score the episode so far against `goal`."""
    s = goal["spec"]
    t = goal["type"]
    a = env.agents()[0]
    if t == "maximize":
        if s["quantity"] == "net_income":
            v = float(env._ledger["revenue"][a] - env._ledger["cost"][a])
        else:
            lo, hi = s["window"]
            xs = [x for tt, x in _series(env, s["reading"]) if lo <= tt < hi]
            v = float(np.mean(xs)) if xs else -math.inf
        return {"value": v, "success": None}
    if t == "reach_target":
        xs = [(tt, x) for tt, x in _series(env, s["reading"]) if tt <= s["deadline"]]
        ok_t = [tt for tt, x in xs if (x >= s["target"] if s["direction"] == "up" else x <= s["target"])]
        run, best = 0, 0
        prev = None
        for tt in ok_t:
            run = run + 1 if prev is not None and tt == prev + 1 else 1
            best = max(best, run)
            prev = tt
        return {"value": float(best), "success": bool(best >= s["hold"])}
    if t == "min_cost_subject_to":
        xs = [x for tt, x in _series(env, s["reading"]) if tt >= s["from"]]
        viol = sum(1 for x in xs if x < s["floor"])
        cost = float(env._ledger["cost"][a])
        return {"value": -cost - 1e6 * viol, "cost": cost, "violations": viol, "success": viol == 0 and bool(xs)}
    if t == "predict_heldout":
        p = _journal_answer(env, goal, "prediction")
        tr = truth(env, goal)["value"]
        if p is None or tr is None:
            return {"value": -math.inf, "success": False, "answered": p is not None}
        sc = _noise_scale(env, s["reading"], s["window"])
        err = abs(p - tr)
        return {"value": -err, "error": err, "truth": tr, "success": bool(err <= max(3.0 * sc, 0.05 * abs(tr)))}
    if t == "identify_effect":
        p = _journal_answer(env, goal, "sign")
        tr = truth(env, goal)
        return {"value": float(p == tr["value"]) if p is not None else 0.0, "success": p is not None and
                int(p) == int(tr["value"]), "truth": tr["value"]}
    raise ValueError(t)


def validate(make_env, goal: dict, budget: dict | None = None) -> dict:
    """Is the goal feasible? `make_env()` builds a fresh copy of the world the goal was drawn for."""
    t = goal["type"]
    if t in ("predict_heldout", "identify_effect"):
        env = make_env()
        tr = truth(env, goal)
        ok = tr.get("value") is not None and (t != "identify_effect" or abs(tr["difference"]) > 2 * tr["noise_scale"]
                                              or abs(tr["difference"]) < 0.5 * tr["noise_scale"])
        return {"feasible": bool(ok), "how": "the answer is determined by the counterfactual", "truth": tr}
    if t == "maximize":
        return {"feasible": True, "how": "maximisation has no threshold"}
    from . import dominance as D
    from .references import play
    env = make_env()
    joint = D.Joint(env)
    rng = np.random.default_rng(0)
    tried = 0
    cands = joint.canonical() + [joint.random(rng) for _ in range(int((budget or {}).get("random", 24)))]
    for s in cands:
        e = copy.deepcopy(env)
        pol = D.constant(joint, s)
        obs = {a: e.observe(a) for a in e.agents()}
        while not e.done:
            acts = pol(e.t, obs)
            r = e.step(acts if len(e.agents()) > 1 else acts.get(e.agents()[0]))
            obs = r.get("observations") or {e.agents()[0]: r["observation"]}
        tried += 1
        if evaluate(e, goal).get("success"):
            return {"feasible": True, "how": "a constant setting meets it", "witness": s, "tried": tried}
    e = make_env()
    play(e)
    if evaluate(e, goal).get("success"):
        return {"feasible": True, "how": "the family's reference meets it", "tried": tried + 1}
    return {"feasible": False, "tried": tried + 1}


def generate_feasible(make_env, difficulty: float = 0.5, seed: int = 0, tries: int = 6) -> dict:
    """IndependentlyGenerateGoal with feasibility validation: draw goals from the world's public face until one is
    feasible (each draw is validated; infeasible draws are reported, not silently fixed)."""
    rejected = []
    for k in range(tries):
        g = G.generate(make_env(), difficulty, seed * 1000 + k)
        v = validate(make_env, g)
        if v["feasible"]:
            return {"goal": g, "validation": v, "rejected": rejected}
        rejected.append({"goal": g, "validation": v})
    return {"goal": None, "validation": None, "rejected": rejected}
