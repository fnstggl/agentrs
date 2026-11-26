"""IndependentlyGenerateGoal(world, difficulty) (MASTER_SPEC Part 7): objectives drawn independently of the world.

A goal is generated from a world's *public* face only - its horizon, its learner seats, the action space it offers
and the names of the readings and resources in its first observation - never from its configuration, its hidden
mechanism or its family. The same generator therefore serves every engine and every family, and a family's
mechanism does not decide what a participant is asked to do. Goal types:

    maximize             maximise a chosen observable: net income, or the average of a reading over the last
                         quarter of the horizon
    reach_target         bring a reading to a target (a stated change from its first observed value, up or down)
                         and hold it there for three consecutive ticks before a deadline
    predict_heldout      predict the average of a reading over a window if a stated setting (one that has not been
                         observed) were held from the start - a held-out condition answered in the journal
    identify_effect      state the direction in which a lever (or an option of a decision) moves a reading
    min_cost_subject_to  keep a reading at or above a floor from a stated tick on while spending as little as
                         possible

`generate(env, difficulty, seed)` returns a JSON goal {id, type, text, spec}; `make(family, seed,
objective=goal)` shows the goal's text to the participant. Whether a goal can be met is checked evaluator side
(`goals_eval.validate`, by searching for a policy that meets it on copies of the world), which never guarantees
that a participant will meet it; `goals_eval.evaluate` scores an episode against it. Answers to predict and
identify goals are journal messages {"goal": id, "prediction": x} or {"goal": id, "sign": -1 | 0 | 1}.
"""
from __future__ import annotations

import numpy as np

from ..core.canonical import digest

GOAL_TYPES = ("maximize", "reach_target", "predict_heldout", "identify_effect", "min_cost_subject_to")


def _public(env) -> dict:
    """What the generator may look at: the public spec, the action space and the first observation."""
    obs = env.observe()
    sp = env.action_space()
    levers = []
    if "levers" in sp.kinds:
        for n, p in sorted(sp.kinds["levers"]["properties"]["values"]["properties"].items()):
            levers.append({"name": n, "min": float(p["minimum"]), "max": float(p["maximum"])})
    decisions = {k: list(v) for k, v in sorted((sp.limits.get("decisions") or {}).items())}
    first = _first_readings(env)
    return {"horizon": int(env.spec["horizon_ticks"]), "readings": sorted(set(obs["numbers"]) | set(first)),
            "money": "cash_cents" in obs["resources"], "levers": levers, "decisions": decisions,
            "journal": "message" in sp.kinds}


def _first_readings(env) -> dict:
    """Readings after one waiting tick on a copy (the generator's view of typical magnitudes, a public observation
    any participant would get by waiting)."""
    import copy
    e = copy.deepcopy(env)
    r = e.step(None)
    return dict(r["observation"]["numbers"])


def generate(env, difficulty: float = 0.5, seed: int = 0, goal_type: str | None = None) -> dict:
    """A goal for `env` drawn from its public face; `difficulty` in [0, 1] sets how far targets sit from the
    starting point, how early deadlines fall and how long prediction windows are."""
    pub = _public(env)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(digest(["goal", pub["readings"],
                                                                                [x["name"] for x in pub["levers"]],
                                                                                sorted(pub["decisions"])])[:8],
                                                                         16)]))
    d = float(np.clip(difficulty, 0.0, 1.0))
    H = pub["horizon"]
    first = _first_readings(env)
    readings = [r for r in pub["readings"] if r in first]
    types = [t for t in GOAL_TYPES if _possible(t, pub, readings)]
    gt = goal_type if goal_type is not None else types[int(rng.integers(len(types)))]
    if gt not in types:
        raise ValueError(f"goal type {gt} is not possible here (possible: {types})")
    gid = f"g{int(rng.integers(10 ** 6)):06d}"
    if gt == "maximize":
        if pub["money"] and (not readings or rng.random() < 0.5):
            spec = {"quantity": "net_income"}
            text = "Maximise your net income over the horizon."
        else:
            r = readings[int(rng.integers(len(readings)))]
            spec = {"quantity": "reading", "reading": r, "window": [H - max(1, H // 4), H]}
            text = f"Maximise the average of the reading {r} over the last {max(1, H // 4)} ticks."
    elif gt == "reach_target":
        r = readings[int(rng.integers(len(readings)))]
        x0 = float(first[r])
        up = bool(rng.random() < 0.6)
        rel = 0.1 + 0.5 * d
        target = x0 + (1 if up else -1) * rel * max(abs(x0), 1.0)
        deadline = int(max(5, round(H * (1.0 - 0.5 * d))))
        spec = {"reading": r, "direction": "up" if up else "down", "target": float(target), "deadline": deadline,
                "hold": 3, "start_value": x0}
        text = (f"Bring the reading {r} {'up to at least' if up else 'down to at most'} {target:.4g} (it was "
                f"{x0:.4g}) and keep it there for 3 consecutive ticks, by tick {deadline}.")
    elif gt == "predict_heldout":
        r = readings[int(rng.integers(len(readings)))]
        setting = _setting(pub, rng)
        k = int(max(2, round(3 + 7 * d)))
        spec = {"reading": r, "setting": setting, "window": k}
        text = (f"Predict the average of the reading {r} over the first {k} ticks if, from the start, the setting "
                f"{_describe(setting)} were held (nobody runs that condition for you). File it with the journal "
                f"as {{\"goal\": \"{gid}\", \"prediction\": <number>}}.")
    elif gt == "identify_effect":
        r = readings[int(rng.integers(len(readings)))]
        if pub["levers"] and (not pub["decisions"] or rng.random() < 0.6):
            lv = pub["levers"][int(rng.integers(len(pub["levers"])))]
            spec = {"reading": r, "lever": lv["name"], "low": lv["min"], "high": lv["max"], "window": 6}
            text = (f"Does raising {lv['name']} from {lv['min']:g} to {lv['max']:g} raise (1), lower (-1) or not "
                    f"move (0) the reading {r} over the following 6 ticks? File {{\"goal\": \"{gid}\", "
                    f"\"sign\": 1 | 0 | -1}} with the journal.")
        else:
            dname = sorted(pub["decisions"])[int(rng.integers(len(pub["decisions"])))]
            opts = pub["decisions"][dname]
            a, b = (opts[int(i)] for i in rng.choice(len(opts), 2, replace=False))
            spec = {"reading": r, "decision": dname, "low": a, "high": b, "window": 6}
            text = (f"Does choosing {b} instead of {a} for {dname} raise (1), lower (-1) or not move (0) the "
                    f"reading {r} over the following 6 ticks? File {{\"goal\": \"{gid}\", \"sign\": 1 | 0 | -1}} "
                    f"with the journal.")
    else:
        r = readings[int(rng.integers(len(readings)))]
        x0 = float(first[r])
        floor = x0 - (0.3 - 0.25 * d) * max(abs(x0), 1.0)
        start = int(max(1, round(H * 0.2)))
        spec = {"reading": r, "floor": float(floor), "from": start}
        text = (f"From tick {start} on, keep the reading {r} at or above {floor:.4g} at every tick while spending "
                f"as little as possible.")
    return {"id": gid, "type": gt, "text": text, "spec": spec, "difficulty": d, "generator": "rapid-goals/1"}


def _possible(t: str, pub: dict, readings: list[str]) -> bool:
    if t == "maximize":
        return bool(pub["money"] or readings)
    if t in ("reach_target",):
        return bool(readings) and bool(pub["levers"] or pub["decisions"])
    if t == "predict_heldout":
        return bool(readings) and bool(pub["levers"] or pub["decisions"]) and pub["journal"]
    if t == "identify_effect":
        return bool(readings) and pub["journal"] and bool(pub["levers"] or any(len(v) > 1
                                                                                for v in pub["decisions"].values()))
    if t == "min_cost_subject_to":
        return bool(readings) and pub["money"] and bool(pub["levers"] or pub["decisions"])
    return False


def _setting(pub: dict, rng) -> dict:
    out: dict = {}
    if pub["levers"]:
        out["levers"] = {lv["name"]: float(np.round(lv["min"] + rng.random() * (lv["max"] - lv["min"]), 3))
                         for lv in pub["levers"]}
    if pub["decisions"]:
        out["choices"] = {d: o[int(rng.integers(len(o)))] for d, o in pub["decisions"].items()}
    return out


def _describe(setting: dict) -> str:
    parts = [f"{k}={v:g}" for k, v in sorted(setting.get("levers", {}).items())]
    parts += [f"{k}={v}" for k, v in sorted(setting.get("choices", {}).items())]
    return ", ".join(parts)


def setting_actions(setting: dict) -> list[dict]:
    """The actions that impose a goal's held-out setting (used by the evaluator's counterfactual)."""
    acts = []
    if setting.get("levers"):
        acts.append({"kind": "levers", "values": dict(setting["levers"])})
    for d, o in sorted(setting.get("choices", {}).items()):
        acts.append({"kind": "discrete", "decision": d, "choice": o})
    return acts
