"""Admission of generated structures (MASTER_SPEC Part 7.4; T.4) - evaluator side.

Generated structures (`compose`) are registered as rapid families for the duration of a run and put through the
complete admission of `crucible.rapid.admission` (`seed_report`: consistency, conservation, controllability,
achievable, informative, identifiability, no_dominant_strategy, difficulty, no_leakage, distinct, mechanism). Their
mechanism claims are intervention tests derived from the structure's own construction (the nodes the composer
placed and their roles), run on forks of the world like every rapid claim:

    key_lever_causes_outcome   holding the key lever at the top versus the bottom of its range changes the outcome
                               node within the horizon (the lever is a cause of what the participant is paid for)
    delay_before_capacity      with a construction delay, a pulse on the key lever reaches the capacity node only
                               after at least one tick
    regime_changes_effect      with a regime shift, the regime node's multiplier differs before and after the hidden
                               switch tick under the same constant action
    rival_adapts_to_price      with adaptive rivals, a high versus a low price moves the rivals' price over time,
                               not on the first tick
    wear_degrades_health       with a wear latch, overdriving the key lever lowers the health node later
    ideal_drifts               with a drifting optimum, the best constant key-lever setting early and late differ by
                               at least a fifth of the lever's range

`admit_generated(structures, seeds, jobs)` returns every world's checks; a world is validated when all eleven pass.
Nothing here is imported by world code.
"""
from __future__ import annotations

import concurrent.futures as cf
from typing import Sequence

import numpy as np

CLAIMS_MOD = "crucible.research.v5.genadmit"


def _role_nodes(env, role: str) -> list[str]:
    return [nd["id"] for nd in env._config["nodes"] if nd.get("role") == role]


def _series(env, values_at, ticks: int, node_id: str) -> np.ndarray:
    from ...rapid.oracles import fork
    e = fork(env)
    i = e._engine.idx[node_id]
    out = []
    for k in range(ticks):
        e.step({"kind": "levers", "values": values_at(k)})
        out.append(float(e._state["h"][0, -1, i]))
    return np.array(out)


def _key(env):
    cl = env._config["claim"]
    eng = env._engine
    nd = eng.config["nodes"][eng.idx[cl["lever"]]]["lever"]
    return nd["name"], float(nd["lo"]), float(nd["hi"]), cl


def key_lever_causes_outcome(env) -> dict:
    name, lo, hi, cl = _key(env)
    T = max(8, env.horizon // 4)
    a = _series(env, lambda k: {name: hi}, T, cl["outcome"])
    b = _series(env, lambda k: {name: lo}, T, cl["outcome"])
    d = float(np.abs(a - b).max())
    return {"ok": d > 1e-6, "max_difference": d}


def delay_before_capacity(env) -> dict:
    name, lo, hi, cl = _key(env)
    nodes = _role_nodes(env, "capacity")
    if not nodes:
        return {"ok": False, "error": "no capacity node"}
    T = 24
    base = _series(env, lambda k: {name: lo}, T, nodes[0])
    pulse = _series(env, lambda k: {name: hi if k == 0 else lo}, T, nodes[0])
    d = np.abs(pulse - base)
    first = int(np.argmax(d > 1e-9)) if np.any(d > 1e-9) else -1
    return {"ok": first >= 1, "first_effect_at": first}


def regime_changes_effect(env) -> dict:
    name, lo, hi, cl = _key(env)
    nodes = _role_nodes(env, "regime")
    if not nodes:
        return {"ok": False, "error": "no regime node"}
    s = _series(env, lambda k: {name: (lo + hi) / 2}, env.horizon, nodes[0])
    sign = np.sign(s)
    return {"ok": bool(len(set(sign[np.abs(s) > 1e-9])) > 1), "values_first_last": [float(s[0]), float(s[-1])]}


def rival_adapts_to_price(env) -> dict:
    nodes = _role_nodes(env, "rival")
    levers = [nd for nd in env._config["nodes"] if nd["type"] == "lever"]
    if not nodes or not levers:
        return {"ok": False, "error": "no rival"}
    price = levers[-1]["lever"]
    hi_ = _series(env, lambda k: {price["name"]: price["hi"] * 0.9}, 30, nodes[0])
    lo_ = _series(env, lambda k: {price["name"]: price["lo"] * 1.5 + 0.5}, 30, nodes[0])
    return {"ok": abs(hi_[-1] - lo_[-1]) > 0.5 and abs(hi_[0] - lo_[0]) < 1e-9,
            "difference_end": float(hi_[-1] - lo_[-1])}


def wear_degrades_health(env) -> dict:
    name, lo, hi, cl = _key(env)
    nodes = _role_nodes(env, "health")
    if not nodes:
        return {"ok": False, "error": "no health node"}
    calm = _series(env, lambda k: {name: lo}, env.horizon, nodes[0])
    over = _series(env, lambda k: {name: hi}, env.horizon, nodes[0])
    return {"ok": float(over[-1]) < float(calm[-1]) - 1e-6, "health_end": [float(calm[-1]), float(over[-1])]}


def ideal_drifts(env) -> dict:
    """With a drifting optimum, the constant key-lever setting that maximises the market-response node early in the
    episode differs from the one late in it by at least a fifth of the lever's range (forks over a grid)."""
    name, lo, hi, cl = _key(env)
    nodes = _role_nodes(env, "ideal")
    if not nodes:
        return {"ok": False, "error": "no ideal node"}
    H = env.horizon
    grid = np.linspace(lo, hi, 11)
    early, late = [], []
    for x in grid:
        s = _series(env, lambda k, x=float(x): {name: x}, H, nodes[0])
        early.append(float(s[1:6].mean()))
        late.append(float(s[H - 5:].mean()))
    be, bl = float(grid[int(np.argmax(early))]), float(grid[int(np.argmax(late))])
    return {"ok": abs(bl - be) >= 0.2 * (hi - lo), "best_early": be, "best_late": bl}


def claims_for(structure: dict) -> tuple[str, ...]:
    d = set(structure.get("drivers") or [])
    pop = (structure.get("population") or {}).get("kind")
    out = [f"{CLAIMS_MOD}:key_lever_causes_outcome"]
    if "delayed_payoff" in d:
        out.append(f"{CLAIMS_MOD}:delay_before_capacity")
    if "regime_shift" in d:
        out.append(f"{CLAIMS_MOD}:regime_changes_effect")
    if pop in ("rivals", "mixed"):
        out.append(f"{CLAIMS_MOD}:rival_adapts_to_price")
    if "wear_latch" in d:
        out.append(f"{CLAIMS_MOD}:wear_degrades_health")
    if "drifting_optimum" in d:
        out.append(f"{CLAIMS_MOD}:ideal_drifts")
    return tuple(out)


def _admit_one(args) -> list[dict]:
    structure, seeds = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ...rapid.admission import CHECKS, distinctness, seed_report
    from . import compose as C
    out = []
    with C.registered(structure) as names:
        try:
            distinct = bool(distinctness(names[0])["ok"])        # per family, not per seed (as in admission.admit)
        except Exception:  # noqa: BLE001 - an unverifiable difference is not distinct
            distinct = False
        for sd in seeds:
            try:
                rep = seed_report(names[0], int(sd))
                checks = {c: bool((rep["checks"].get(c) or {}).get("ok")) for c in CHECKS if c != "distinct"}
                checks["distinct"] = distinct
            except Exception as e:  # noqa: BLE001 - a world that crashes admission fails every check
                checks = {c: False for c in CHECKS}
                checks["error"] = False
                rep = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            out.append({"structure_id": names[0], "seed": int(sd), "checks": checks,
                        "validated": all(checks.values()), "error": rep.get("error")})
    return out


def admit_generated(structures: Sequence[dict], seeds: Sequence[int] = (0, 1, 2), jobs: int = 1) -> dict:
    tasks = [(dict(s), tuple(seeds)) for s in structures]
    if jobs > 1:
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_admit_one, tasks))
    else:
        res = [_admit_one(t) for t in tasks]
    worlds = [w for r in res for w in r]
    return {"worlds": worlds, "validated": sum(1 for w in worlds if w["validated"]),
            "structures_validated": len({w["structure_id"] for w in worlds if w["validated"]})}
