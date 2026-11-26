"""Mechanism-level intervention tests, one or more per family (MASTER_SPEC Part 7.5, Gate M item 8).

Each test makes a controlled intervention on copies of a world and checks that the mechanism the family claims is
really there: an effect appears exactly after its delay, a correlation in the records disappears under
intervention, a pathway is closed below a threshold, a fault yields only to its own repair, and so on. Copies of a
world share its tick generators, so two arms that differ only in the intervention see the same world noise (common
random numbers); event-driven worlds, whose draws depend on the path, are compared over several noise replicates.

Evaluator side: these functions read hidden structure. `run(family, seed)` runs every claim of a family.
Each claim returns {"ok": bool, ...what it measured}.
"""
from __future__ import annotations

import numpy as np

from .oracles import fork, inside


# ====================================================================== helpers
def _lever_name(env, node_id):
    eng = env._engine
    return eng.config["nodes"][eng.idx[node_id]]["lever"]["name"]


def _series(env, act_at, ticks: int, node_id: str) -> np.ndarray:
    """Values of a hidden node over `ticks` ticks of a fork, with action act_at(k) at tick k."""
    e = fork(env)
    i = e._engine.idx[node_id]
    out = []
    for k in range(ticks):
        e.step(act_at(k))
        out.append(float(e._state["h"][0, -1, i]))
    return np.array(out)


def _hold(values: dict):
    return lambda k: {"kind": "levers", "values": values}


def _replicates(env, n: int):
    from .env import Env
    return [Env(env.family, env.seed, env._config, env.difficulty, world_seed=env.world_seed + 7919 * (r + 1))
            for r in range(n)]


def _net_of(e) -> float:
    o = e.outcome()
    return float(o["value"])


# ====================================================================== 8.1 hidden causal systems
def confounded_reversal(env) -> dict:
    """The records show a positive association of the spending lever with sales; setting the lever shows its
    effect on sales is negative."""
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    X = _lever_name(env, cl["lever"])
    sales_gauge = cfg["gauges"][0]["name"]
    hx, hy = np.asarray(eng.history_of(st, X)), np.asarray(eng.history_of(st, sales_gauge))
    corr = float(np.corrcoef(hx, hy)[0, 1])
    slope = float(np.polyfit(hx, hy, 1)[0])
    lo = _series(env, _hold({X: 2.0}), 30, cl["outcome"])
    hi = _series(env, _hold({X: 8.0}), 30, cl["outcome"])
    eff = float((hi - lo).mean() / 6.0)
    return {"ok": corr > 0.3 and slope > 0 and eff < 0, "passive_correlation": corr, "passive_slope": slope,
            "interventional_effect_per_unit": eff}


def conditional_pathway(env) -> dict:
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    A, B = _lever_name(env, cl["lever"]), _lever_name(env, cl["gate_lever"])
    th = cl["theta"]
    out = {}
    for lab, b in (("closed", max(0.0, th - 2.0)), ("open", min(10.0, th + 1.0))):
        lo = _series(env, _hold({A: 1.0, B: b}), 10, cl["outcome"])
        hi = _series(env, _hold({A: 9.0, B: b}), 10, cl["outcome"])
        out[lab] = float((hi - lo).mean())
    return {"ok": out["open"] > 1.0 and abs(out["closed"]) < 0.05 * out["open"], "effect_gate_closed": out["closed"],
            "effect_gate_open": out["open"]}


def feedback_amplifies(env) -> dict:
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    M = _lever_name(env, cl["lever"])
    lo = _series(env, _hold({M: 0.5}), 60, cl["outcome"])
    hi = _series(env, _hold({M: 5.0}), 60, cl["outcome"])
    d = hi - lo
    ratio = float(d[40:].mean() / max(d[0], 1e-9))
    return {"ok": d[0] > 0 and ratio >= 1.4, "immediate_effect": float(d[0]), "long_run_effect": float(d[40:].mean()),
            "amplification": ratio}


# ====================================================================== Part 7.2 look-alike markets
def _units(env, price, ticks=4, cert=None, start=None):
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    P = _lever_name(env, cl["price"])
    e = fork(start or env)
    dec = eng.config["nodes"][eng.idx[cl["compat"]]]["options"]
    if cert is not None:
        e.step({"kind": "discrete", "decision": dec["name"], "choice": dec["choices"][cert]})
    i = e._engine.idx[cl["units"]]
    vals = []
    for _ in range(ticks):
        e.step({"kind": "levers", "values": {P: float(price)}})
        vals.append(float(e._state["h"][0, -1, i]))
    return float(np.mean(vals[1:]))


SIGNATURES = ("elastic", "signal", "gate", "drift")


def market_signature(env) -> dict:
    """Which of the four price-response signatures a market shows under interventions (evaluator)."""
    grid = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
    plain = [_units(env, p) for p in grid]
    cert = [_units(env, p, cert=1) for p in grid]
    rng_plain = (max(plain) - min(plain)) / max(np.mean(plain), 1e-9)
    rng_cert = (max(cert) - min(cert)) / max(np.mean(cert), 1e-9)
    gated = rng_plain < 0.35 * rng_cert and np.mean(cert) > 2 * np.mean(plain)
    interior_peak = 0 < int(np.argmax(plain)) < len(grid) - 1 and plain[0] < 0.8 * max(plain)
    decreasing = all(b < a for a, b in zip(plain, plain[1:]))
    # drift: hold a high price long enough for the rival to grow, then measure the price effect again
    eng, st, cfg = inside(env)
    P = _lever_name(env, cfg["claim"]["price"])
    late = fork(env)
    for _ in range(45):
        late.step({"kind": "levers", "values": {P: 9.0}})
    # the price effect is measured on either side of the conversion curve's hidden midpoint, where it is steepest
    # (two fixed prices can both sit on a flat, saturated stretch and hide a change of sensitivity)
    m = float(cfg["claim"].get("mid", 4.5))
    p_lo, p_hi = max(1.0, m - 0.75), min(10.0, m + 0.75)
    early_eff = _units(env, p_hi) - _units(env, p_lo)
    late_eff = _units(env, p_hi, start=late) - _units(env, p_lo, start=late)
    drift = early_eff < 0 and late_eff < 2.0 * early_eff
    sig = "gate" if gated else ("signal" if interior_peak else ("drift" if drift else
                                                               ("elastic" if decreasing else "none")))
    return {"signature": sig, "units_by_price": plain, "units_by_price_certified": cert,
            "early_price_effect": early_eff, "late_price_effect": late_eff}


def market_rule(env) -> dict:
    m = market_signature(env)
    want = env._config["rule"]
    return {"ok": m["signature"] == want, "claimed_rule": want, **m}


def lookalike_matrix(seed: int = 0) -> dict:
    """Run the same intervention protocol on all four look-alike markets: each must show its own signature (and
    therefore none shows another's), although their public interfaces are identical."""
    from .families import make
    fams = {"elastic": "mk_elastic", "signal": "mk_signal", "gate": "mk_gate", "drift": "mk_drift"}
    out, iface = {}, {}
    for rule, fam in fams.items():
        env = make(fam, seed)
        out[fam] = market_signature(env)["signature"]
        sp = env.action_space()
        iface[fam] = (tuple(sorted(sp.kinds)), len(sp.kinds["levers"]["properties"]["values"]["properties"]),
                      len(env.step(None)["observation"]["numbers"]))
    return {"signatures": out, "all_own": all(out[f] == r for r, f in fams.items()),
            "distinct": len(set(out.values())) == 4, "same_interface": len(set(iface.values())) == 1}


# ====================================================================== 8.2 experimental science
def law_extrapolates(env) -> dict:
    from .engines.primitive_graph import true_response
    eng, st, cfg = inside(env)
    x1, x2 = [_lever_name(env, n) for n in cfg["claim"]["levers"]]
    g = cfg["objective"]["gauge"]
    f = lambda a, b: true_response(eng, [{x1: a, x2: b}], g)[0]  # noqa: E731
    inc1, inc2 = f(3, 3) - f(0, 3), f(6, 3) - f(3, 3)
    d_hi, d_lo = f(4, 5) - f(2, 5), f(4, 1) - f(2, 1)
    held = true_response(eng, cfg["objective"]["heldout"], g)
    lin = np.polyfit([0, 3, 6], [f(0, 3), f(3, 3), f(6, 3)], 1)
    lin_err = float(abs(np.polyval(lin, 9.0) - f(9, 3)))
    return {"ok": inc2 < inc1 and d_hi > d_lo and lin_err > 0.05 * abs(f(9, 3)), "increment_0_3": inc1,
            "increment_3_6": inc2, "slope_at_x2_5": d_hi, "slope_at_x2_1": d_lo,
            "linear_extrapolation_error_at_9": lin_err, "heldout_values": [float(v) for v in held]}


def instrument_precision(env) -> dict:
    eng, st, cfg = inside(env)
    x1, x2 = [_lever_name(env, n) for n in cfg["claim"]["levers"]]
    g = cfg["objective"]["gauge"]
    sd = {}
    for inst in ("field", "lab"):
        e = fork(env)
        e._state["budget"] = np.array([1e9])
        r = e.step({"kind": "experiment", "design": {"set": {x1: 3.0, x2: 3.0}, "measure": [g], "ticks": 3,
                                                     "replicates": 8, "instrument": inst}})
        m = r["result"]["results"]["measured"][g]
        sd[inst] = m["se"] * np.sqrt(m["n"])
    from .engines.primitive_graph import true_response
    th = [n for n in cfg["nodes"] if n.get("prim", "").startswith("threshold")][0]["params"]["theta"]
    jump = true_response(eng, [{x1: 3.0, x2: min(6.0, th + 0.2)}], g)[0] - \
        true_response(eng, [{x1: 3.0, x2: max(0.0, th - 0.2)}], g)[0]
    return {"ok": sd["lab"] < 0.3 * sd["field"] and jump > 1.0, "sd_field": float(sd["field"]),
            "sd_lab": float(sd["lab"]), "threshold_jump": float(jump)}


# ====================================================================== 8.6 delayed consequences, 8.3, 8.8
def delayed_return(env) -> dict:
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    I_ = _lever_name(env, cl["lever"])
    base = _series(env, _hold({I_: 0.0}), cl["lag"] + 8, cl["outcome"])
    pulse = _series(env, lambda k: {"kind": "levers", "values": {I_: 10.0 if k == 0 else 0.0}}, cl["lag"] + 8,
                    cl["outcome"])
    d = pulse - base
    first = int(np.argmax(np.abs(d) > 1e-9)) if np.any(np.abs(d) > 1e-9) else -1
    return {"ok": first == cl["lag"] and d[first] > 0, "lag": cl["lag"], "first_effect_at": first,
            "effect_before_lag": float(np.abs(d[:cl["lag"]]).max())}


def damage_persists(env) -> dict:
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    b = _lever_name(env, cl["lever"])
    calm = _series(env, _hold({b: 2.0}), 60, cl["outcome"])
    over = _series(env, lambda k: {"kind": "levers", "values": {b: 10.0 if k < 15 else 2.0}}, 60, cl["outcome"])
    return {"ok": over[:3].mean() > calm[:3].mean() and over[40:].mean() < 0.8 * calm[40:].mean(),
            "gain_while_overdriving": float(over[:3].mean() - calm[:3].mean()),
            "late_output_ratio": float(over[40:].mean() / calm[40:].mean())}


def controller_delay(env) -> dict:
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    h = _lever_name(env, cl["lever"])
    a = _series(env, _hold({h: 3.0}), cl["lag"] + 6, cl["outcome"])
    b = _series(env, _hold({h: 8.0}), cl["lag"] + 6, cl["outcome"])
    d = b - a
    first = int(np.argmax(np.abs(d) > 1e-9)) if np.any(np.abs(d) > 1e-9) else -1
    return {"ok": first == cl["lag"] and d[first] > 0, "lag": cl["lag"], "first_effect_at": first}


def rival_adapts(env) -> dict:
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    p = _lever_name(env, cl["lever"])
    hi = _series(env, _hold({p: 8.0}), 30, cl["opponent"])
    lo = _series(env, _hold({p: 3.0}), 30, cl["opponent"])
    return {"ok": hi[-1] - lo[-1] > 2.0 and abs(hi[0] - lo[0]) < 1e-9, "rival_after_high": float(hi[-1]),
            "rival_after_low": float(lo[-1]), "first_tick_difference": float(hi[0] - lo[0])}


# ====================================================================== discrete-event worlds
def bottleneck_station(env) -> dict:
    eng, st, cfg = inside(env)
    mu = np.asarray(cfg["mu"])
    b = int(np.argmin(mu))
    other = int(np.argmax(mu))
    names = cfg["station_names"]
    base = list(cfg["default_alloc"])

    def run(alloc):
        tot = []
        for e in _replicates(env, 4):
            e.step({"kind": "allocate", "amounts": {n: float(x) for n, x in zip(names, alloc)}})
            for _ in range(e.horizon - 1):
                e.step(None)
            tot.append(e._state["completed"])
        return float(np.mean(tot))
    plus_b, plus_o = list(base), list(base)
    donor = next(k for k in range(len(base)) if k not in (b, other) and base[k] > 1) if len(base) > 2 else other
    plus_b[b] += 1
    plus_b[donor] -= 1
    plus_o[other] += 1
    plus_o[donor] -= 1
    cb, co = run(plus_b), run(plus_o)
    return {"ok": cb > co * 1.05, "completed_server_to_bottleneck": cb, "completed_server_elsewhere": co,
            "bottleneck": b}


def maintenance_pays(env) -> dict:
    eng, st, cfg = inside(env)
    eta = np.asarray(cfg["eta"])
    j = int(np.argmin(eta))
    name = cfg["machine_names"][j]
    res = {}
    for lab, serve in (("never", False), ("service", True)):
        fails, nets = [], []
        for e in _replicates(env, 4):
            for _ in range(e.horizon):
                s = e._state
                act = None
                if serve and s["mode"][j] == "run" and s["age"][j] >= 0.4 * eta[j]:
                    act = {"kind": "discrete", "decision": "service", "choice": name}
                e.step(act)
            fails.append(e._state["failures"])
            nets.append(_net_of(e))
        res[lab] = (float(np.mean(fails)), float(np.mean(nets)))
    return {"ok": res["service"][0] < res["never"][0] and res["service"][1] > res["never"][1],
            "failures_never": res["never"][0], "failures_serviced": res["service"][0],
            "net_never": res["never"][1], "net_serviced": res["service"][1]}


def lead_time(env) -> dict:
    eng, st, cfg = inside(env)
    e = fork(env)
    e.step({"kind": "levers", "values": {cfg["order_name"]: 30.0}})
    t0, first = 0, None
    for k in range(1, 20):
        before = e._state["delivered"]
        e.step(None)
        if e._state["delivered"] > before and first is None:
            first = k
    return {"ok": first is not None and first >= cfg["lead_min"], "lead_min": cfg["lead_min"],
            "first_delivery_tick": first, "ordered_at": t0}


def perishing(env) -> dict:
    eng, st, cfg = inside(env)
    big, small = fork(env), fork(env)
    for _ in range(3):
        big.step({"kind": "levers", "values": {cfg["order_name"]: 40.0}})
        small.step({"kind": "levers", "values": {cfg["order_name"]: 4.0}})
    for _ in range(25):
        big.step(None)
        small.step(None)
    return {"ok": big._state["expired"] > small._state["expired"] and big._state["expired"] > 0,
            "expired_after_large_order": int(big._state["expired"]),
            "expired_after_small_order": int(small._state["expired"]), "shelf_life": cfg["shelf_life"]}


# ====================================================================== network worlds
def _route_run(env, weights, ticks=30):
    e = fork(env)
    names = e._config["route_names"]
    for _ in range(ticks):
        e.step({"kind": "levers", "values": {n: float(w) for n, w in zip(names, weights)}})
    return float(e._state["delivered"][0])


def shared_bottleneck(env) -> dict:
    eng, st, cfg = inside(env)
    K = cfg["K"]
    a = cfg["assign"]
    c1, c2 = cfg["c1"], cfg["c2"]
    best = None
    for k1 in range(K):
        for k2 in range(K):
            for k3 in range(K):
                if len({k1, k2, k3}) < 3 or a[k2] != a[k1] or a[k3] == a[k1]:
                    continue
                e_share = min(c1[k1] + c1[k2], c2[a[k1]]) - min(c1[k1], c2[a[k1]])
                e_dist = min(c1[k3], c2[a[k3]])
                if best is None or abs(e_dist - e_share) > abs(best[3] - best[4]):
                    best = (k1, k2, k3, e_dist, e_share)
    if best is None:
        return {"ok": False, "note": "no pair of routes shares a second hop"}
    k1, k2, k3, e_dist, e_share = best
    w = lambda ks: [1.0 if k in ks else 0.0 for k in range(K)]  # noqa: E731
    d1, ds, dd = _route_run(env, w([k1])), _route_run(env, w([k1, k2])), _route_run(env, w([k1, k3]))
    pred = np.sign(e_dist - e_share)
    return {"ok": np.sign((dd - d1) - (ds - d1)) == pred and abs(e_dist - e_share) > 0.5,
            "gain_adding_sharing_route": ds - d1, "gain_adding_distinct_route": dd - d1,
            "predicted_from_hidden_capacities": float(e_dist - e_share)}


def topology_shift(env) -> dict:
    """The rewiring changes which splits work: the split that was best before the hidden change loses value
    after it compared with the split that is best afterwards (64 random splits, realised noise, vectorised)."""
    from .oracles import _noise, _tile
    eng, st, cfg = inside(env)
    K, at, H = cfg["K"], cfg["rewire_at"], env.horizon
    W = np.random.default_rng(11).dirichlet(np.ones(K), 64)
    s = _tile(eng.copy_state(env._state), 64)
    s["w"] = W
    noise = _noise(env, H)
    per = np.zeros((64, H))
    for k in range(H):
        z, u = noise[k]
        r = eng.batch_step(s, np.repeat(z, 64, axis=0), np.repeat(u, 64, axis=0), k)
        per[:, k] = r["revenue"] - r["cost"]
    before, after = per[:, 5:at].mean(axis=1), per[:, at + 5:].mean(axis=1)
    b0 = int(np.argmax(before))
    loss = float((after.max() - after[b0]) / max(abs(after.max()), 1e-9))
    rb, ra = np.argsort(np.argsort(before)), np.argsort(np.argsort(after))
    rho = float(np.corrcoef(rb, ra)[0, 1])
    return {"ok": loss > 0.03 and rho < 0.95, "relative_loss_of_old_best_after_change": loss,
            "rank_correlation_before_after": rho, "rewire_at": at}


def concentrated_seeding(env) -> dict:
    """Complex contagion stays inside the community it was seeded in: adoption needs a fraction of one's
    neighbours, and the few bridges between communities never supply it."""
    eng, st, cfg = inside(env)
    cl = np.asarray(cfg["cluster"])
    th = np.asarray(cfg["theta"])
    k = int(np.argmin([th[cl == q].mean() for q in range(len(cfg["cluster_names"]))]))
    e = fork(env)
    e.step({"kind": "allocate", "amounts": {cfg["cluster_names"][k]: float(cfg["seed_total"])}})
    for _ in range(20):
        e.step(None)
    inside_k = float(e._state["adopted"][cl == k].sum())
    outside = float(e._state["adopted"][cl != k].sum())
    return {"ok": inside_k > 1.5 * cfg["seed_total"] and outside <= 0.1 * (cl != k).sum(),
            "adopters_in_seeded_community": inside_k, "adopters_elsewhere": outside,
            "seeds": int(cfg["seed_total"])}


def backup_links_contain_cascades(env) -> dict:
    eng, st, cfg = inside(env)
    P = len(cfg["cap"])
    steps = []
    for i, p in enumerate(cfg["dep"]):
        alt = int(np.argmin([cfg["base"][j] / cfg["cap"][j] if j != p else 9 for j in range(P)]))
        steps.append({"kind": "graph_edit", "op": "add_edge", "src": cfg["node_names"][alt],
                      "dst": cfg["node_names"][P + i]})
    lost = {}
    for lab, first in (("none", None), ("backups", {"kind": "sequence", "steps": steps[:16]})):
        tot = 0.0
        for e in [fork(env)] + _replicates(env, 3):
            r = e.step(first)
            g = cfg["gauge_names"][1]
            tot += r["observation"]["numbers"].get(g, 0.0)
            for _ in range(e.horizon - 1):
                tot += e.step(None)["observation"]["numbers"].get(g, 0.0)
        lost[lab] = tot / 4
    return {"ok": lost["backups"] < lost["none"], "nodes_lost_without_backups": lost["none"],
            "nodes_lost_with_backups": lost["backups"]}


def repair_restores(env) -> dict:
    from .engines.graph_dynamical import components
    eng, st, cfg = inside(env)
    N = cfg["n_nodes"]
    edges = [tuple(e) for e in cfg["edges"]]
    import itertools
    cut, bridge = None, None
    for size in (1, 2, 3):
        for S in itertools.combinations(range(len(edges)), size):
            comp = components(N, [e for k, e in enumerate(edges) if k not in S])
            if any(comp[a] != comp[b] for a, b in cfg["pairs"]):
                for k in S:                                  # a link of the cut whose repair alone reconnects a pair
                    c2 = components(N, [e for j, e in enumerate(edges) if j not in S or j == k])
                    if sum(1 for a, b in cfg["pairs"] if c2[a] == c2[b]) > sum(1 for a, b in cfg["pairs"]
                                                                                if comp[a] == comp[b]):
                        cut, bridge = S, k
                        break
            if cut:
                break
        if cut:
            break
    if cut is None:
        return {"ok": False, "note": "no small cut separates a demand pair"}
    served = {}
    for lab, fix in (("left_broken", False), ("repaired", True)):
        vals = []
        for e in [fork(env)] + _replicates(env, 3):
            e._state["alive"] = e._state["alive"].copy()
            e._state["alive"][list(cut)] = 0.0
            act = {"kind": "discrete", "decision": "repair", "choice": cfg["edge_names"][bridge]} if fix else None
            vals.append(e.step(act)["observation"]["numbers"][cfg["gauge_names"][0]])
        served[lab] = float(np.mean(vals))
    return {"ok": served["repaired"] > served["left_broken"], **served, "cut": [cfg["edge_names"][k] for k in cut],
            "repaired_link": cfg["edge_names"][bridge]}


def built_link_carries_traffic(env) -> dict:
    """A link the participant builds (graph_edit add_edge) between the two sides of a cut that separates a demand
    pair carries that pair's traffic: served demand rises against leaving the cut open (same noise, several
    replicates). Checks the mechanism of graph edits, not only that the new link is listed."""
    from .engines.graph_dynamical import components
    eng, st, cfg = inside(env)
    N = cfg["n_nodes"]
    edges = [tuple(e) for e in cfg["edges"]]
    import itertools
    for size in (1, 2, 3):
        for S in itertools.combinations(range(len(edges)), size):
            comp = components(N, [e for k, e in enumerate(edges) if k not in S])
            split = [(a, b) for a, b in cfg["pairs"] if comp[a] != comp[b]]
            if split:
                a, b = split[0]
                served = {}
                for lab, build in (("left_open", False), ("built_link", True)):
                    vals = []
                    for e in [fork(env)] + _replicates(env, 3):
                        e._state["alive"] = e._state["alive"].copy()
                        e._state["alive"][list(S)] = 0.0
                        names = e._config["node_names"]
                        act = {"kind": "graph_edit", "op": "add_edge", "src": names[a], "dst": names[b],
                               "weight": 1.0} if build else None
                        r = e.step(act)
                        if build and not r["result"]["ok"]:
                            return {"ok": False, "error": r["result"].get("error")}
                        vals.append(e.step(None)["observation"]["numbers"][cfg["gauge_names"][0]])
                    served[lab] = float(np.mean(vals))
                return {"ok": served["built_link"] > served["left_open"], **served,
                        "cut": [cfg["edge_names"][k] for k in S]}
    return {"ok": False, "note": "no small cut separates a demand pair"}


# ====================================================================== state machines
def regime_sign(env) -> dict:
    eng, st, cfg = inside(env)
    eff = []
    for s in range(len(cfg["slope"])):
        r = []
        for x in (0.0, 10.0):
            e = fork(env)
            e._state["s"] = np.array([float(s)])
            e.step({"kind": "levers", "values": {cfg["lever_name"]: x}})
            r.append(e._ledger["revenue"]["a0"])
        eff.append(r[1] - r[0])
    return {"ok": any(x > 0 for x in eff) and any(x < 0 for x in eff), "lever_effect_by_regime": eff}


def matching_repair(env) -> dict:
    eng, st, cfg = inside(env)
    F = len(cfg["repair_names"])
    ok = True
    table = []
    for f in range(1, F + 1):
        row = []
        for g in range(1, F + 1):
            e = fork(env)
            e._state["s"] = np.array([float(f)])
            e._engine.T = e._engine.T.copy()
            e._engine.T[0] = 0.0                             # intervention: no new fault during the test
            e._engine.T[0, 0] = 1.0
            e.step({"kind": "discrete", "decision": "repair", "choice": cfg["repair_names"][g - 1]})
            fixed = int(e._state["s"][0]) == 0
            row.append(bool(fixed))
            ok &= fixed == (f == g)
        table.append(row)
    return {"ok": bool(ok), "fixed_by_repair": table}


def unlock_needs_all(env) -> dict:
    eng, st, cfg = inside(env)
    d, opts, code = cfg["decision_name"], cfg["options"], cfg["code"]
    seq = lambda ks: {"kind": "sequence", "steps": [{"kind": "discrete", "decision": d, "choice": opts[k]}  # noqa
                                                    for k in ks]}
    res = {}
    trap_run = code[:1] + [cfg["trap"][1]] + code if len(code) > 1 else code
    for lab, ks in (("all_but_last", code[:-1]), ("full", code), ("trap_then_full", trap_run)):
        e = fork(env)
        e.step(seq(ks))
        e.step(None)
        res[lab] = float(e._ledger["revenue"]["a0"])
    return {"ok": res["all_but_last"] == 0 and res["full"] > 0 and res["trap_then_full"] == 0, **res}


# ====================================================================== search worlds
def field_structure(env) -> dict:
    eng, st, cfg = inside(env)
    v = np.asarray(cfg["values"])
    a, b = v[:-1, :].ravel(), v[1:, :].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    e = fork(env)
    errs = []
    for k in range(40):
        x, y = k % v.shape[0], (3 * k) % v.shape[1]
        r = e._engine.apply(e._state, "a0", {"kind": "query", "target": "probe", "args": {"cell": f"c{x}_{y}"}},
                            np.random.default_rng(k), 0)
        errs.append(r["value"] - v[x, y])
    bias = float(np.mean(errs))
    se = float(np.std(errs) / np.sqrt(len(errs)))
    return {"ok": corr > 0.5 and abs(bias) < 4 * se + 1e-9, "neighbour_correlation": corr, "probe_bias": bias}


def distractor_features(env) -> dict:
    eng, st, cfg = inside(env)
    F = np.asarray(cfg["features"])
    y = np.asarray(cfg["values"])
    inf = cfg["informative"]

    def r2(cols):
        X = np.c_[np.ones(len(y)), F[:, cols]]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ beta
        return float(1 - res.var() / y.var())
    others = [j for j in range(F.shape[1]) if j not in inf]
    best_d = max(r2([i, j]) for i in others for j in others if i < j) if len(others) > 1 else 0.0
    return {"ok": r2(inf) > 0.7 and best_d < 0.5, "r2_informative": r2(inf), "r2_best_distractor_pair": best_d}


def tool_applicability(env) -> dict:
    eng, st, cfg = inside(env)
    app = np.asarray(cfg["app"])
    k1, j1 = [int(x) for x in np.argwhere(app > 0)[0]]
    zeros = np.argwhere(app == 0)
    rates = {}
    for lab, (k, j) in (("applicable", (k1, j1)), ("not_applicable", tuple(int(x) for x in zeros[0]) if len(zeros)
                                                                       else (k1, j1))):
        e = fork(env)
        e._state["found"] = np.ones(len(cfg["tool_names"]))
        for _ in range(30):
            e._state["job"] = np.array([float(j)])
            e.step({"kind": "discrete", "decision": "use", "choice": cfg["tool_names"][k]})
        rev = e._ledger["revenue"]["a0"]
        rates[lab] = float(rev / (30 * cfg["payoff"]))
    return {"ok": rates["applicable"] > 0.6 and rates["not_applicable"] == 0.0 and len(zeros) > 0, **rates}


def needs_both(env) -> dict:
    from .engines.search import cell_name
    eng, st, cfg = inside(env)
    G = cfg["G"]
    res = {}
    for lab in ("one_signal", "both_signals"):
        e = fork(env)
        rng = np.random.default_rng(5)
        obs = {a: e.observe(a) for a in e.agents()}
        sx, sy = [], []
        for t in range(e.horizon):
            nx = obs["a0"]["numbers"].get(cfg["gauge_names"][0])
            ny = obs["a1"]["numbers"].get(cfg["gauge_names"][1])
            if nx is not None:
                sx.append(nx)
            if ny is not None:
                sy.append(ny)
            if t % 4 == 3 and sx:
                x = int(np.clip(round(np.mean(sx[-4:])), 0, G - 1))
                y = int(np.clip(round(np.mean(sy[-4:])), 0, G - 1)) if lab == "both_signals" else int(rng.integers(G))
                plan = {"cell": cell_name(x, y)}
                r = e.step({"a0": {"kind": "coordinate", "with": ["a1"], "plan": plan},
                            "a1": {"kind": "coordinate", "with": ["a0"], "plan": plan}})
                if r["results"]["a0"].get("result", {}).get("hit"):
                    sx, sy = [], []
            else:
                r = e.step(None)
            obs = r["observations"]
        res[lab] = int(e._state["hits"][0])
    return {"ok": res["both_signals"] > res["one_signal"], "hits_one_signal": res["one_signal"],
            "hits_both_signals": res["both_signals"]}


def run(family: str, seed: int, **difficulty) -> dict[str, dict]:
    """Every intervention test of a family on one world."""
    import importlib
    from .families import FAMILIES, make
    out = {}
    for ref in FAMILIES[family].claims:
        mod, fn = ref.split(":")
        try:
            out[fn] = getattr(importlib.import_module(mod), fn)(make(family, seed, **difficulty))
        except Exception as e:  # noqa: BLE001 - a crashing test is a failing test, reported with its reason
            out[fn] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return out
