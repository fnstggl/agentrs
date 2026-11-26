"""Gate SP-8.12 - adjustable difficulty along nine axes; the fast simulator and the kernel agree (workstream H).

An axis counts when an admitted family exposes it through `make(family, seed, **difficulty)` and changing it
changes the world in the way the axis names (sizes, noise, horizon, costs, hidden states, opponent learning rate,
budgets, options), measured on built worlds. Agreement is measured tick for tick between the vectorised batch
simulator (many seeds at once) and the same worlds hosted in event-sourced kernel worlds as `rig` facilities with
ledger money, and between the batch and single worlds."""
from __future__ import annotations

import os

import time

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.mechanisms.sim import cents
from crucible.rapid import families as F
from crucible.rapid import report as RP
from crucible.rapid.batch import BatchRapid, batchable
from crucible.rapid.families import DIFFICULTY_AXES, build_config
from crucible.rapid.mechview import kernel_world

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def _measured(axis: str, family: str):
    """(value at default, value raised) of the quantity the axis controls, measured on built worlds."""
    fam = F.FAMILIES[family]
    v = fam.axes[axis]
    hi = v * 2 if v else (1.0 if isinstance(v, float) else 1)
    if axis == "hidden_state_complexity" and family == "srch_tools":
        hi = v + 2
    lo_cfg, hi_cfg = build_config(family, 0), build_config(family, 0, **{axis: hi})
    lo_env, hi_env = F.make(family, 0), F.make(family, 0, **{axis: hi})
    if axis == "horizon":
        return lo_env.horizon, hi_env.horizon
    if axis == "observation_noise":
        return _noise_sd(lo_env), _noise_sd(hi_env)
    if axis == "n_variables":
        return len(lo_env.step(None)["observation"]["numbers"]) + len(lo_cfg.get("features", [[0]])[0]), \
            len(hi_env.step(None)["observation"]["numbers"]) + len(hi_cfg.get("features", [[0]])[0])
    if axis == "graph_size":
        return _size(lo_cfg), _size(hi_cfg)
    if axis == "experiment_cost":
        return _cost(lo_env), _cost(hi_env)
    if axis == "hidden_state_complexity":
        return _hidden(lo_cfg), _hidden(hi_cfg)
    if axis == "opponent_capability":
        if F.FAMILIES[family].engine == "primitive_graph":
            return _rival_speed(lo_env), _rival_speed(hi_env)
        a, b = _changed_number(lo_cfg, hi_cfg, ("eta", "memory", "lr", "rate"))
        if a is None and "forget" in lo_cfg:              # a learner that forgets old regrets adapts faster
            a, b = 1.0 - float(lo_cfg["forget"]), 1.0 - float(hi_cfg["forget"])
        return a, b
    if axis == "resources":
        r = _resources(lo_cfg), _resources(hi_cfg)
        return r if None not in r else _changed_number(lo_cfg, hi_cfg, ("budget", "crew", "capacity", "slots",
                                                                        "umax", "seed_budget"))
    if axis == "action_complexity":
        return _options(lo_env), _options(hi_env)
    raise KeyError(axis)


def _changed_number(lo_cfg, hi_cfg, keys):
    """The value, at the default and at the raised axis, of the first named configuration quantity the axis
    controls in this engine (a learning rate, a budget, a crew size, ...)."""
    for k in keys:
        a, b = lo_cfg.get(k), hi_cfg.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            return float(a), float(b)
    return None, None


def _noise_sd(env):
    """Spread of tick-to-tick changes of the first reading while waiting; None when the world shows no reading
    while one waits (its noise lives in query results and is measured through another family)."""
    e = env
    vals = []
    for _ in range(30):
        n = e.step(None)["observation"]["numbers"]
        if n:
            vals.append(list(n.values())[0])
    return float(np.std(np.diff(vals))) if len(vals) >= 3 else None


def _size(cfg):
    for k in ("n_nodes", "K", "G"):
        if k in cfg:
            return cfg[k]
    for k in ("theta", "cap", "mu", "eta", "route_names", "site_names"):
        if k in cfg and isinstance(cfg[k], list):
            return len(cfg[k])
    return len(cfg.get("nodes", []))


def _cost(env):
    c = env._config
    if "experiment" in c:
        return c["experiment"]["cost"]
    return c.get("query_cost", c.get("probe_cost", (c.get("test_cost") or [0])[-1]))


def _hidden(cfg):
    for k in ("T",):
        if k in cfg:
            return len(cfg[k])
    if "stages" in cfg:
        return cfg["stages"]
    if "bumps" in cfg:
        return len(cfg["bumps"])
    if "job_names" in cfg:
        return len(cfg["job_names"])
    return len(cfg.get("cluster_names", []))


def _rival_speed(env):
    eng = env._engine
    p = eng.config["claim"]["lever"]
    name = eng.config["nodes"][eng.idx[p]]["lever"]["name"]
    opp = eng.idx[eng.config["claim"]["opponent"]]
    vals = []
    for _ in range(10):
        env.step({"kind": "levers", "values": {name: 9.0}})
        vals.append(float(env._state["h"][0, -1, opp]))
    return vals[-1] - vals[0]


def _resources(cfg):
    for k in ("servers", "seed_total", "reinforce_budget"):
        if k in cfg:
            return cfg[k]
    if "experiment" in cfg:
        return cfg["experiment"]["budget"]
    for nd in cfg.get("nodes", []):
        if nd.get("role") == "capacity":
            return nd["params"]["init"]
    return None


def _options(env):
    sp = env.action_space()
    if "discrete" in sp.kinds:
        return sum(len(v) for v in sp.limits["decisions"].values())
    return len(sp.kinds)


def test_SP_8_12_difficulty_axes_and_simulator_kernel_agreement():
    adm = RP.admission_report(seeds=(0, 1, 2), jobs=int(os.environ.get("CRUCIBLE_RAPID_JOBS", "1")))
    admitted = {n for n, r in adm["reports"].items() if r["admitted"]}
    axes_ok, where = {}, {}
    for axis in DIFFICULTY_AXES:
        fams = [n for n in sorted(admitted) if axis in F.FAMILIES[n].axes]
        ok = False
        for fam in fams:
            a, b = _measured(axis, fam)
            if a is not None and b is not None and b > a:
                ok, where[axis] = True, f"{fam}: {a:.4g} -> {b:.4g}"
                break
        axes_ok[axis] = ok
    # fast simulator vs kernel-hosted twin, tick for tick, with ledger money
    mism, compared = 0, 0
    rng = np.random.default_rng(42)
    t0 = time.process_time()
    from crucible.rapid import twin as TW
    twin_fams = [n for n in sorted(admitted) if F.FAMILIES[n].lever_view][:4] or \
        [n for n in ("ch_silent_reversal", "hc_hidden_state_sign", "net_routing", "hc_confounded")
         if F.FAMILIES[n].lever_view][:2]
    for fam in twin_fams:
        for seed in (0, 1):
            w, actor, mech = kernel_world(fam, seed)
            fast = BatchRapid(fam, [seed]) if F.FAMILIES[fam].engine in ("primitive_graph", "graph_dynamical") \
                and batchable(fam) else None
            from crucible.mechanisms.sim import MechSim
            from crucible.rapid.env import world_seed_of
            sim = MechSim(mech, world_seed_of(fam, seed), "r1")
            gauges = [g.name for g in sim.engine.gauges(0)]
            lev = sim.engine.levers(0)
            for t in range(40):
                vals = np.array([[lv.lo + rng.random() * (lv.hi - lv.lo) for lv in lev]])
                settings = {lv.name: float(v) for lv, v in zip(lev, vals[0])}
                res = w.execute(actor, {"tool": "rig.set", "args": {"settings": settings}})
                assert res.ok, res.error
                w.advance_time(w.now + 3600)
                rep = w.execute(actor, {"tool": "rig.read", "args": {}}).result["report"]
                s = sim.step(settings)
                compared += 1
                if fast is not None:
                    out = fast.step(vals)
                    vis = [k for k, n in enumerate(fast.engine.gauge_names) if n in gauges] \
                        if fast.engine.kind == "primitive_graph" else list(range(len(gauges)))
                    fr = {gauges[j]: float(out["readings"][0, k]) for j, k in enumerate(vis)}
                    if fr != {g: rep["readings"][g] for g in gauges} or \
                            cents(out["revenue"][0]) != rep["revenue_cents"] or cents(out["cost"][0]) != rep["cost_cents"]:
                        mism += 1
                if s.readings[0] != rep["readings"] or s.revenue[0] != rep["revenue_cents"] or \
                        s.cost[0] != rep["cost_cents"]:
                    mism += 1
            assert w.check_invariants() == []
    twin_cpu = time.process_time() - t0
    # an independent implementation: the scalar interpreter (plain Python, its own primitive formulas) against the
    # vectorised engine, tick for tick on the same draws, for every registered mechanism it can represent
    scalar = {}
    for fam in sorted(F.FAMILIES):
        if TW.supports(build_config(fam, 0)):
            rs = [TW.compare(fam, sd, ticks=40) for sd in (0, 1)]
            scalar[fam] = sum(r["mismatches"] for r in rs)
            compared += sum(r["ticks"] for r in rs)
    scalar_mism = sum(scalar.values())
    mism += scalar_mism
    assert scalar and twin_fams
    # batch vs single worlds, and the speed of the batch path
    b = BatchRapid("net_routing", list(range(64)))
    t1 = time.perf_counter()
    for t in range(30):
        vals = b.lo + rng.random((64, b.n_levers)) * (b.hi - b.lo)
        out = b.step(vals)
    batch_s = time.perf_counter() - t1
    envs = [F.make("net_routing", s) for s in range(16)]
    t2 = time.perf_counter()
    single_mism = 0
    b2 = BatchRapid("net_routing", list(range(16)))
    for t in range(30):
        vals = b2.lo + rng.random((16, b2.n_levers)) * (b2.hi - b2.lo)
        out = b2.step(vals)
        for i in range(16):
            r = envs[i].step({"kind": "levers", "values": {n: float(v) for n, v in zip(b2.lever_names[i], vals[i])}})
            if list(r["observation"]["numbers"].values()) != list(out["readings"][i]):
                single_mism += 1
    single_s = time.perf_counter() - t2
    record("SP-8.12", {"difficulty_axes": sum(axes_ok.values()), "sim_kernel_mismatches": mism,
                       "sim_kernel_ticks_compared": compared, "batch_single_mismatches": single_mism,
                       "independent_scalar_twin_families": len(scalar), "independent_scalar_twin_mismatches":
                       scalar_mism, "kernel_twin_families": len(twin_fams),
                       "batch_steps_per_s": float(64 * 30 / max(batch_s, 1e-9)),
                       "single_and_batch_steps_per_s": float(16 * 30 / max(single_s, 1e-9)),
                       "twin_cpu_s": float(twin_cpu)},
           notes="Axes measured: " + "; ".join(f"{k}: {v}" for k, v in where.items()) +
                 f". Missing: {[k for k, v in axes_ok.items() if not v]}. Kernel twins (event-sourced kernel worlds "
                 f"hosting a rig facility with ledger money, vs the vectorised batch path): {twin_fams}. Independent "
                 f"scalar interpreter vs vectorised engine: {len(scalar)} mechanisms, mismatching ticks {scalar}. "
                 "The scalar twin found (and the engine now fixes) an off-by-one in lagged gauges.")
