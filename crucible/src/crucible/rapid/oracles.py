"""Evaluator-side oracles and references (MASTER_SPEC Part 7.4). Private: nothing in a world imports this module
(checked by `admission.static_isolation`), no participant can call it, and its results never enter an observation.

Three kinds of reference, all run on copies (forks) of a world, never on the world itself:

* search-based: `schedule_oracle` (cross-entropy search over piecewise-constant lever schedules) and `mpc_oracle`
  (receding-horizon search) evaluate candidate controls on the world's *realised* noise - the exact tick
  generators the world will use - with the vectorised batch path, so a candidate's value is exactly what it would
  earn in this world;
* privileged scripted references: policies that read the hidden state or parameters (the fault a machine is in, the
  values of sites, the code of a stage machine) and act on them;
* policy search by simulation on forks (base-stock levels, service ages, seeding plans).

`oracle(env)` returns the best of the references that apply to the family: {"value", "how", "candidates"}.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..mechanisms.sim import TickRng


# ====================================================================== private access
def inside(env):
    """(engine, state, config) of a world. Evaluator only."""
    return env._engine, env._state, env._config


def fork(env):
    """An independent copy of a world at its current point (a controlled counterfactual)."""
    return copy.deepcopy(env)


def run_fn(env, fn, ticks: int | None = None) -> dict:
    """Run a privileged policy fn(env) -> action | None on `env` until the horizon (or `ticks`)."""
    n = env.horizon - env.t if ticks is None else ticks
    for _ in range(n):
        if env.done:
            break
        env.step(fn(env))
    return env.outcome()


def value(env) -> float:
    return float(env.outcome()["value"])


# ====================================================================== search on realised noise (batched engines)
def _noise(env, H: int):
    eng = env._engine
    rng = TickRng(env.world_seed, env.env_id)
    out = []
    for k in range(H):
        z, u = eng.draw(rng(env.t + k))
        out.append((z, u))
    return out


def _tile(state: dict, P: int) -> dict:
    return {k: np.repeat(v, P, axis=0) if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == 1 else
            np.array(v, copy=True) for k, v in state.items()}


def _levers(env) -> tuple[list[str], np.ndarray, np.ndarray]:
    eng = env._engine
    lv = eng.public_levers()
    return [x["name"] for x in lv], np.array([x["min"] for x in lv]), np.array([x["max"] for x in lv])


def eval_schedules(env, X: np.ndarray, choices: dict | None = None, noise=None) -> np.ndarray:
    """Net value over the rest of the horizon of P lever schedules X (P, H, L) in lever units, on the realised
    noise, vectorised. `choices` fixes discrete decisions (decision name -> option) for every candidate."""
    eng = env._engine
    P, H, _ = X.shape
    noise = noise or _noise(env, H)
    st = _tile(eng.copy_state(env._state), P)
    if choices:
        for d, o in choices.items():
            j = [x["name"] for x in eng.dec].index(d)
            st["choice"][:, j] = eng.dec[j]["choices"].index(o)
    names, lo, hi = _levers(env)
    tot = np.zeros(P)
    for k in range(H):
        if _is_pg(eng):
            st["lev"] = np.clip(X[:, k, :], lo, hi)
            st["levset"] = np.ones_like(st["lev"])
        else:
            st["w"] = np.clip(X[:, k, :], lo, hi)
        z, u = noise[k]
        r = eng.batch_step(st, np.repeat(z, P, axis=0), np.repeat(u, P, axis=0), env.t + k)
        tot += r["revenue"] - r["cost"]
    return tot


def schedule_oracle(env, segments: int = 4, pop: int = 48, iters: int = 6, seed: int = 0) -> dict:
    eng = env._engine
    names, lo, hi = _levers(env)
    H = env.horizon - env.t
    L = len(names)
    noise = _noise(env, H)
    combos = [None]
    if getattr(eng, "dec", None):
        combos = [{d["name"]: c} for d in eng.dec[:1] for c in d["choices"]]
    best = (-math.inf, None, None)
    rng = np.random.default_rng(seed)
    seg_of = np.minimum(segments - 1, (np.arange(H) * segments) // max(H, 1))
    for ch in combos:
        mu = np.full((segments, L), 0.5)
        sd = np.full((segments, L), 0.35)
        for it in range(iters):
            U = np.clip(rng.normal(mu, sd, (pop, segments, L)), 0, 1)
            if it == 0:
                U[0] = 0.5
                U[1] = 0.0
                U[2] = 1.0
                d0 = (np.array([x["default"] for x in eng.public_levers()]) - lo) / np.maximum(hi - lo, 1e-12)
                U[3] = d0
            X = lo + U[:, seg_of, :] * (hi - lo)
            v = eval_schedules(env, X, ch, noise)
            order = np.argsort(-v)
            if v[order[0]] > best[0]:
                best = (float(v[order[0]]), X[order[0]], ch)
            el = U[order[:max(4, pop // 8)]]
            mu, sd = el.mean(axis=0), np.maximum(el.std(axis=0), 0.02)
    return {"value": best[0], "schedule": best[1], "choices": best[2], "how": f"schedule CEM ({segments} segments)"}


def mpc_oracle(env, grid: int = 11, look: int = 6) -> dict:
    """Greedy receding-horizon search: each tick, every grid setting is held for `look` ticks on the realised
    noise; the best first move is taken. Levers <= 2."""
    eng = env._engine
    names, lo, hi = _levers(env)
    if len(names) > 2:
        return {"value": -math.inf, "how": "mpc skipped (too many levers)"}
    e = fork(env)
    H = e.horizon - e.t
    axes = [np.linspace(lo[j], hi[j], grid) for j in range(len(names))]
    G = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(names), -1).T
    sched = []
    for k in range(H):
        hl = min(look, H - k)
        X = np.repeat(G[:, None, :], hl, axis=1)
        v = eval_schedules(e, X)
        g = G[int(np.argmax(v))]
        sched.append(g)
        e.step({"kind": "levers", "values": {n: float(x) for n, x in zip(names, g)}})
    return {"value": value(e), "schedule": np.array(sched), "how": f"receding horizon (grid {grid}, look {look})"}


def lever_oracle(env, mpc: bool = False) -> dict:
    best = schedule_oracle(env)
    cands = [best]
    if mpc:
        cands.append(mpc_oracle(env))
    best = max(cands, key=lambda r: r["value"])
    # verify by actually running the schedule on a fork
    e = fork(env)
    names, _, _ = _levers(env)
    for k, row in enumerate(best["schedule"]):
        act = {"kind": "levers", "values": {n: float(x) for n, x in zip(names, row)}}
        if best.get("choices") and k == 0:
            (d, o), = best["choices"].items()
            act = {"kind": "sequence", "steps": [{"kind": "discrete", "decision": d, "choice": o}, act]}
        e.step(act)
    v = value(e)
    return {"value": v, "how": best["how"], "search_value": best["value"]}


# ====================================================================== simulation search on forks
def best_of(env, policies: dict) -> dict:
    """Run each named privileged policy on a fork; the best one."""
    res = {}
    for k, fn in policies.items():
        res[k] = float(run_fn(fork(env), fn)["value"])
    k = max(res, key=res.get)
    return {"value": res[k], "how": k, "candidates": res}


def _const(action):
    return lambda e: action


def des_oracle(env) -> dict:
    eng, st, c = inside(env)
    if eng.sc == "line":
        mu = np.array(c["mu"])
        R = int(c["servers"])
        pols = {}
        for tilt in (0.8, 1.0, 1.2):
            w = (1.0 / mu) ** tilt
            alloc = np.maximum(1, np.floor(R * w / w.sum())).astype(int)
            while alloc.sum() < R:
                alloc[int(np.argmin(alloc * mu))] += 1
            while alloc.sum() > R:
                alloc[int(np.argmax(alloc * mu))] -= 1
            for drop in (0, 1):
                a = alloc.copy()
                if drop and a.sum() > len(a):
                    a[int(np.argmax(a * mu))] -= 1
                act = {"kind": "allocate", "amounts": {n: float(x) for n, x in zip(c["station_names"], a)}}
                pols[f"balanced^{tilt}-{drop}"] = _first_then_wait(act)
        first = best_of(env, pols)
        # local search on the allocation: move one server at a time while it helps (simulated on forks)
        names = c["station_names"]
        cur = np.maximum(1, np.floor(R * (1.0 / mu) / (1.0 / mu).sum())).astype(int)
        while cur.sum() > R:
            cur[int(np.argmax(cur))] -= 1

        def val(a):
            act = {"kind": "allocate", "amounts": {n: float(x) for n, x in zip(names, a)}}
            return float(run_fn(fork(env), _first_then_wait(act))["value"])
        v = val(cur)
        for _ in range(6):
            moves = [(i, j) for i in range(len(cur)) for j in range(len(cur)) if i != j and cur[i] > 1]
            if cur.sum() < R:
                moves += [(-1, j) for j in range(len(cur))]
            scored = []
            for i, j in moves:
                a = cur.copy()
                if i >= 0:
                    a[i] -= 1
                a[j] += 1
                scored.append((val(a), i, j))
            vb, i, j = max(scored)
            if vb <= v:
                break
            if i >= 0:
                cur[i] -= 1
            cur[j] += 1
            v = vb
        if v > first["value"]:
            return {"value": v, "how": f"local search allocation {cur.tolist()}", "candidates": first["candidates"]}
        return first
    if eng.sc == "maintenance":
        eta = np.array(c["eta"])

        def by_age(alpha):
            def fn(e):
                s = e._state
                due = [j for j in range(len(eta)) if s["mode"][j] == "run" and s["age"][j] >= alpha * eta[j]]
                if not due:
                    return None
                j = max(due, key=lambda j: s["age"][j] / eta[j])
                return {"kind": "discrete", "decision": "service", "choice": c["machine_names"][j]}
            return fn
        return best_of(env, {f"service_at_{a}eta": by_age(a) for a in (0.2, 0.3, 0.4, 0.55, 0.7, 0.9)})
    # inventory: base-stock levels with a seasonal demand forecast and the mean lead time (both hidden)
    lead = c["lead_min"] - 1 + 1.0 / c["lead_p"]

    def base_stock(S, cover):
        def fn(e):
            s = e._state
            pos = sum(q for q, _ in s["lots"]) + sum(q for _, q in s["pipeline"])
            t = float(e.t)
            want = sum(e._engine._lam(t + k) for k in range(int(math.ceil(lead + cover)))) + S
            q = int(max(0, min(c["max_order"], round(want - pos))))
            return {"kind": "levers", "values": {c["order_name"]: float(q)}} if q > 0 else None
        return fn
    pols = {f"base_stock_{S}_{cv}": base_stock(S, cv) for S in (0, 3, 6, 10, 15) for cv in (0.5, 1.0, 2.0)}
    return best_of(env, pols)


def _first_then_wait(action):
    def fn(e):
        return action if e.t == 0 else None
    return fn


def graph_oracle(env) -> dict:
    eng, st, c = inside(env)
    if eng.sc in ("routing",):
        return lever_oracle(env)
    if eng.sc == "diffusion":
        names = c["cluster_names"]
        R = float(c["seed_total"])
        pols = {"even": _first_then_wait({"kind": "allocate", "amounts": {n: float(R // len(names)) for n in names}})}
        for k, n in enumerate(names):
            pols[f"all_in_{n}"] = _first_then_wait({"kind": "allocate", "amounts": {n: R}})
            for k2, n2 in enumerate(names):
                if k2 > k:
                    pols[f"split_{n}_{n2}"] = _first_then_wait({"kind": "allocate",
                                                                "amounts": {n: R // 2, n2: R - R // 2}})
        return best_of(env, pols)
    if eng.sc == "cascade":
        P = len(c["cap"])
        zone = np.asarray(c["zone"])
        risk = np.array([np.mean([b / k for b, k, z in zip(c["base"], c["cap"], zone) if z == q])
                         for q in range(len(c["zone_names"]))])

        def backups(e):
            if e.t != 0:
                return None
            steps = []
            for i, p in enumerate(c["dep"]):
                alt = int(np.argmin([c["base"][j] / c["cap"][j] if j != p else 9 for j in range(P)]))
                steps.append({"kind": "graph_edit", "op": "add_edge", "src": c["node_names"][alt],
                              "dst": c["node_names"][P + i]})
            return {"kind": "sequence", "steps": steps[:16]}

        def reinforce(mask, frac, with_backups=False):
            """Every tick, `frac` of the crew budget on the zones in `mask`, in proportion to their load ratio."""
            share = risk * mask / max(float((risk * mask).sum()), 1e-12)
            tot = float(c["reinforce_budget"]) * frac * 0.999
            act = {"kind": "allocate", "amounts": {n: tot * float(s) for n, s in zip(c["zone_names"], share)}}

            def fn(e):
                if with_backups and e.t == 0:
                    return backups(e)
                return act
            return fn
        def adaptive(frac, with_backups):
            def fn(e):
                if with_backups and e.t == 0:
                    return backups(e)
                alive = e.observe()["arrays"].get("alive_by_zone")
                if not alive:
                    return None
                weak = np.array([1.0 - a for a in alive]) + 0.05
                tot = float(c["reinforce_budget"]) * frac
                return {"kind": "allocate", "amounts": {n: tot * float(x / weak.sum()) for n, x in
                                                        zip(c["zone_names"], weak)}}
            return fn
        pols = {"none": lambda e: None, "backups": backups}
        for frac in (0.2, 0.4, 0.6):
            pols[f"adaptive_{frac}"] = adaptive(frac, False)
            pols[f"backups+adaptive_{frac}"] = adaptive(frac, True)
        for frac in (0.25, 0.5, 1.0):
            for top in (1, 2, len(risk)):
                mask = (risk >= np.sort(risk)[-top]).astype(float)
                pols[f"reinforce_top{top}_{frac}"] = reinforce(mask, frac)
                pols[f"backups+reinforce_top{top}_{frac}"] = reinforce(mask, frac, True)
        return best_of(env, pols)
    # repair: privileged greedy - repair the broken link whose repair restores the most hidden demand
    from .engines.graph_dynamical import components

    def greedy(e):
        s = e._state
        down = [k for k in range(len(c["edges"])) if s["alive"][k] < 0.5]
        if not down:
            return None
        nx = s["extra"].shape[0]
        extra = [tuple(int(v) for v in s["extra"][k]) for k in range(nx) if s["extra_alive"][k] > 0.5]

        def served(alive_set):
            comp = components(c["n_nodes"], [c["edges"][k] for k in alive_set] + extra)
            return sum(d for (a, b), dd in zip(c["pairs"], c["pair_demand"]) for d in [dd] if comp[a] == comp[b])
        up = [k for k in range(len(c["edges"])) if s["alive"][k] > 0.5]
        base = served(up)
        gains = [(served(up + [k]) - base, -c["hazard"][k], k) for k in down]
        g, _, k = max(gains)
        if g * c["price"] * 3 < c["repair_cost"] and g <= 0:
            return None
        return {"kind": "discrete", "decision": "repair", "choice": c["edge_names"][k]}
    return best_of(env, {"greedy_repair": greedy})


def posm_oracle(env) -> dict:
    eng, st, c = inside(env)
    if eng.sc == "sign_switch":
        def fn(e):
            s = int(e._state["s"][0])
            x = 10.0 if c["slope"][s] * c["price"] > c["lever_cost"] else 0.0
            return {"kind": "levers", "values": {c["lever_name"]: x}}
        return best_of(env, {"knows_regime": fn})
    if eng.sc == "fault":
        def fn(e):
            s = int(e._state["s"][0])
            return None if s == 0 else {"kind": "discrete", "decision": "repair", "choice": c["repair_names"][s - 1]}
        return best_of(env, {"knows_fault": fn})

    def unlock(e):
        if e.t == 0:
            return {"kind": "sequence", "steps": [{"kind": "discrete", "decision": c["decision_name"],
                                                   "choice": c["options"][k]} for k in c["code"]]}
        return None
    return best_of(env, {"knows_code": unlock})


def search_oracle(env) -> dict:
    eng, st, c = inside(env)
    if eng.sc == "field":
        from .engines.search import cell_name
        v = np.asarray(c["values"])
        order = np.dstack(np.unravel_index(np.argsort(-v, axis=None), v.shape))[0]

        def fn(e):
            left = e._state["left"]
            for x, y in order:
                if left[x, y] > c["dig_cost"]:
                    return {"kind": "discrete", "decision": "dig", "choice": cell_name(int(x), int(y))}
            return None
        return best_of(env, {"knows_values": fn})
    if eng.sc == "distractors":
        vals = np.asarray(c["values"])

        def fn(e):
            left = e._state["left"]
            for i in np.argsort(-vals):
                if left[i] > 0 and vals[i] * c["scale"] > c["invest_cost"]:
                    return {"kind": "discrete", "decision": "invest", "choice": c["item_names"][int(i)]}
            return None
        return best_of(env, {"knows_values": fn})
    if eng.sc == "tools":
        app = np.asarray(c["app"])
        useful_pos = [k for k, tool in enumerate(c["order"]) if app[tool].sum() > 0]
        need = max(useful_pos) + 1 if useful_pos else 0

        def fn(e):
            s = e._state
            if int(s["next"][0]) < need:
                return {"kind": "query", "target": "catalogue"}
            j = int(s["job"][0])
            ok = [k for k in range(len(c["tool_names"])) if s["found"][k] > 0.5 and app[k, j] > 0]
            if not ok:
                return None
            return {"kind": "discrete", "decision": "use", "choice": c["tool_names"][ok[0]]}
        return best_of(env, {"knows_applicability": fn})
    from .engines.search import cell_name

    def coop(e):
        tx, ty = (int(v) for v in e._state["target"])
        plan = {"cell": cell_name(tx, ty)}
        return {a: {"kind": "coordinate", "with": [b for b in e.agents() if b != a], "plan": plan}
                for a in e.agents()}
    return best_of(env, {"knows_target": coop})


def science_oracle(env) -> dict:
    from .engines.primitive_graph import true_response
    eng, st, c = inside(env)
    obj = c["objective"]
    # the same noise-free steady response the score uses (slow processes are scored where they settle)
    vals = true_response(eng, obj["heldout"], obj["gauge"], ticks=int(obj.get("settle_ticks", 1)))
    preds = [{"set": cond, "value": float(v)} for cond, v in zip(obj["heldout"], vals)]

    def fn(e):
        return {"kind": "message", "to": "journal", "content": {"predictions": preds}} if e.t == 0 else None
    return best_of(env, {"knows_law": fn})


def _is_pg(eng) -> bool:
    from .engines.primitive_graph import PrimitiveGraph
    return isinstance(eng, PrimitiveGraph)


def _family_hook(env_or_family, which: str):
    """A family's own evaluator-side function ("module:function" in its registry entry), or None. Accepts the
    environment or the family name."""
    import importlib
    from .families import FAMILIES
    family = env_or_family if isinstance(env_or_family, str) else env_or_family.family
    fam = FAMILIES.get(family)
    ref = getattr(fam, which, None) if fam is not None else None
    if not ref:
        return None
    mod, fn = ref.split(":")
    return getattr(importlib.import_module(mod), fn)


def oracle(env) -> dict:
    """The best private reference for this world (evaluated on a fork of it from its current point)."""
    from .engines.discrete_event import DiscreteEvent
    from .engines.graph_dynamical import GraphDynamical
    from .engines.state_machine import StateMachine
    from .families import FAMILIES
    fam = FAMILIES.get(env.family)
    if fam is not None and fam.oracle:                  # a family's own evaluator-side oracle ("module:function")
        import importlib
        mod, fn = fam.oracle.split(":")
        return getattr(importlib.import_module(mod), fn)(env)
    eng = env._engine
    hook = _family_hook(env, "oracle")
    if hook is not None:
        return hook(env)
    obj = (env._config.get("objective") or {}).get("type")
    if obj == "predict_heldout":
        return science_oracle(env)
    if _is_pg(eng):
        return lever_oracle(env, mpc=len(eng.public_levers()) <= 1)
    if isinstance(eng, DiscreteEvent):
        return des_oracle(env)
    if isinstance(eng, GraphDynamical):
        return graph_oracle(env)
    if isinstance(eng, StateMachine):
        return posm_oracle(env)
    return search_oracle(env)
