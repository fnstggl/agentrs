"""Intervention tests and public identification procedures for the H3 families (evaluator side).

Claims (Part 7.5, Gate M item 8) make controlled interventions on copies of a world - forks share the world's tick
generators, so two arms that differ only in the intervention see the same noise - and check that the structure the
family claims is there: two candidate causes of which one is inert, a passive association that reverses under
intervention, a dose response that peaks, a balancing loop, a latch with memory, a sign reversal at a hidden time,
a constraint that tightens, a drifting optimum, and the network and discovery mechanisms.

Identification procedures (Part 7.4) answer each family's experimental question through a `PublicView` only, and
the evaluator compares the answer with the truth read from the hidden structure.

Nothing in a world imports this module (`admission.static_isolation`).
"""
from __future__ import annotations

import numpy as np

from .identify import PublicView, contrast_estimate, contrast_truth
from .oracles import fork, inside


# ====================================================================== helpers
def _lname(env, nid: str) -> str:
    return env._engine.config["nodes"][env._engine.idx[nid]]["lever"]["name"]


def _series(env, act_at, ticks: int, node_id: str) -> np.ndarray:
    """Values of a node over `ticks` ticks of a fork of `env`, with action act_at(k) at tick k."""
    e = fork(env)
    i = e._engine.idx[node_id]
    out = []
    for k in range(ticks):
        e.step(act_at(k))
        out.append(float(e._state["h"][0, -1, i]))
    return np.array(out)


def _hold(values: dict):
    return lambda k: {"kind": "levers", "values": values}


def _gauge_of(env, nid: str) -> str:
    return next(g["name"] for g in env._config["gauges"] if g["src"] == [nid] and not g.get("on_demand"))


def _rev(obs) -> float:
    return float(obs["resources"].get("revenue_cents_last_tick", 0.0))


def _net(obs) -> float:
    r = obs["resources"]
    return float(r.get("revenue_cents_last_tick", 0.0) - r.get("cost_cents_last_tick", 0.0))


def _lever_props(view):
    return view.action_space().kinds["levers"]["properties"]["values"]["properties"]


def _sign_question(env, lever: str, measure, block: int | None = None, question: str = "") -> dict:
    """Does a higher setting of `lever` (75% vs 25% of its range) raise `measure` per tick, in blocks?"""
    view = PublicView(env)
    p = _lever_props(view)[lever]
    lo, hi = p["minimum"], p["maximum"]
    act = {"A": {"kind": "levers", "values": {lever: lo + 0.75 * (hi - lo)}},
           "B": {"kind": "levers", "values": {lever: lo + 0.25 * (hi - lo)}}}
    block = block or max(4, env.horizon // 8)
    truth = contrast_truth(fork(env), act, measure, block)
    est = contrast_estimate(view, act, measure, block)
    return {"question": question or f"does raising {lever} raise the measure", "truth": float(np.sign(truth)),
            "estimate": float(np.sign(est["difference"])), "z": est["z"], "true_difference": truth,
            "ok": bool(np.sign(truth) == np.sign(est["difference"]) and abs(est["z"]) >= 2.0)}


# ====================================================================== 8.1 hidden causal systems
def competing_causes(env) -> dict:
    """Both levers track sales in the records; under intervention only the causal one moves the outcome."""
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["inert"])
    hs = np.asarray(eng.history_of(st, _gauge_of(env, cl["outcome"])))
    ha, hb = np.asarray(eng.history_of(st, A)), np.asarray(eng.history_of(st, B))
    ca, cb = float(np.corrcoef(ha, hs)[0, 1]), float(np.corrcoef(hb, hs)[0, 1])
    eff_a = float((_series(env, _hold({A: 8.0, B: 5.0}), 20, cl["outcome"])
                   - _series(env, _hold({A: 2.0, B: 5.0}), 20, cl["outcome"])).mean())
    eff_b = float((_series(env, _hold({A: 5.0, B: 8.0}), 20, cl["outcome"])
                   - _series(env, _hold({A: 5.0, B: 2.0}), 20, cl["outcome"])).mean())
    return {"ok": ca > 0.3 and cb > 0.3 and eff_a > 0.5 and abs(eff_b) <= 0.02 * eff_a,
            "records_corr_causal": ca, "records_corr_inert": cb, "effect_causal": eff_a, "effect_inert": eff_b}


def which_lever_causes(env) -> dict:
    """Question: which of the two levers raises sales? Public: block contrasts of each lever with the other held,
    measured on revenue (units sold times the posted price)."""
    view = PublicView(env)
    props = _lever_props(view)
    names = sorted(props)
    block = 5
    zs = {}
    for n in names:
        other = [m for m in names if m != n][0]
        act = {"A": {"kind": "levers", "values": {n: 8.0, other: 3.0}},
               "B": {"kind": "levers", "values": {n: 2.0, other: 3.0}}}
        H = min(28, view.horizon - view.t)
        sub = _Prefix(view, H)
        zs[n] = contrast_estimate(sub, act, _rev, block)["z"]
    est = max(zs, key=zs.get)
    truth = _lname(env, env._config["claim"]["lever"])
    return {"question": "which lever causes sales", "estimate": est, "truth": truth,
            "ok": est == truth and zs[est] >= 2.0, "z": zs}


class _Prefix:
    """A view whose horizon ends `n` ticks from now (to run one contrast in part of an episode)."""

    def __init__(self, view, n):
        self.v, self.end = view, view.t + n

    def step(self, a):
        return self.v.step(a)

    @property
    def horizon(self):
        return self.end

    @property
    def t(self):
        return self.v.t


def reactive_reversal(env) -> dict:
    """In the records more spending goes with lower sales; holding the lever higher raises sales."""
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    A = _lname(env, cl["lever"])
    hs = np.asarray(eng.history_of(st, _gauge_of(env, cl["outcome"])))
    ha = np.asarray(eng.history_of(st, A))
    slope = float(np.polyfit(ha, hs, 1)[0])
    corr = float(np.corrcoef(ha, hs)[0, 1])
    eff = float((_series(env, _hold({A: 8.0}), 30, cl["outcome"]) -
                 _series(env, _hold({A: 2.0}), 30, cl["outcome"]))[5:].mean())
    return {"ok": slope < 0 and corr < -0.1 and eff > 0.5, "passive_slope": slope, "passive_corr": corr,
            "interventional_effect": eff}


def lever_raises_sales(env) -> dict:
    """Question: does holding the family's key lever higher raise revenue (units sold)?"""
    name = _lname(env, env._config["claim"]["lever"])
    return _sign_question(env, name, _rev, question=f"does raising {name} raise sales")


def dose_peak(env) -> dict:
    """The response to the dose rises up to the hidden peak dose and falls beyond it."""
    cfg = env._config
    cl = cfg["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    pk = float(cl["peak_dose"])
    v = {}
    for lab, x in (("low", 0.25 * pk), ("peak", pk), ("high", 10.0)):
        v[lab] = float(_series(env, _hold({A: x, B: 5.0}), 20, cl["outcome"])[8:].mean())
    rise, fall = v["peak"] - v["low"], v["peak"] - v["high"]
    return {"ok": rise > 0.5 and fall > 0.5, "response": v, "peak_dose": pk}


def dose_is_non_monotonic(env) -> dict:
    """Question: at which dose is output highest? Public: a dose sweep, revenue at the end of each step."""
    view = PublicView(env)
    props = _lever_props(view)
    A = _lname(env, env._config["claim"]["lever"])
    B = [n for n in props if n != A][0]
    grid = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0]
    hold = max(4, min(6, (view.horizon - view.t) // len(grid)))
    vals = []
    for x in grid:
        r = []
        for k in range(hold):
            o = view.step({"kind": "levers", "values": {A: x, B: 5.0}})["observation"]
            if k >= hold - 2:
                r.append(_rev(o))
        vals.append(float(np.mean(r)))
    est = grid[int(np.argmax(vals))]
    truth = float(env._config["claim"]["peak_dose"])
    return {"question": "at which dose is output highest", "estimate": est, "truth": truth,
            "ok": abs(est - truth) <= 1.25 and vals[-1] < max(vals), "sweep": vals}


def balancing_loop(env) -> dict:
    """A step in the lever moves the output at once, then the hidden balancing loop pulls it back: the lasting
    effect is below the immediate one."""
    cl = env._config["claim"]
    A = _lname(env, cl["lever"])
    step = _series(env, lambda k: {"kind": "levers", "values": {A: 2.0 if k < 15 else 8.0}}, 45, cl["core"])
    base = _series(env, _hold({A: 2.0}), 45, cl["core"])
    d = step - base
    short, late = float(d[15]), float(d[32:45].mean())
    return {"ok": short > 0 and late > 0 and late < 0.8 * short, "immediate_effect": short,
            "lasting_effect": late}


def latch_memory(env) -> dict:
    """The same setting of the latch lever leaves the pathway open after it was pushed high and closed after it
    was held low (hysteresis): the second lever's output differs although current settings are equal."""
    cfg = env._config
    cl = cfg["claim"]
    A, B = _lname(env, cl["latch_lever"]), _lname(env, cl["lever"])
    p = cfg["nodes"][env._engine.idx[cl["latch"]]]["params"]
    band = 0.5 * (p["lo"] + p["hi"])
    up = lambda k: {"kind": "levers", "values": {A: 10.0 if k < 8 else band, B: 7.0}}  # noqa: E731
    down = lambda k: {"kind": "levers", "values": {A: 0.0 if k < 8 else band, B: 7.0}}  # noqa: E731
    la, lb = _series(env, up, 24, cl["latch"]), _series(env, down, 24, cl["latch"])
    sa, sb = _series(env, up, 24, cl["outcome"]), _series(env, down, 24, cl["outcome"])
    diff = float((sa - sb)[14:].mean())
    return {"ok": bool(la[-1] == p["on"] and lb[-1] == p["off"] and diff > 1.0), "latch_after_high": float(la[-1]),
            "latch_after_low": float(lb[-1]), "output_difference": diff, "band_setting": band}


def latch_threshold(env) -> dict:
    """Question: from which setting of the latch lever does the second lever's pathway open? Public: an ascending
    sweep with the second lever high, watching revenue jump."""
    view = PublicView(env)
    props = _lever_props(view)
    cl = env._config["claim"]
    A, B = _lname(env, cl["latch_lever"]), _lname(env, cl["lever"])
    del props
    grid = [float(x) for x in range(1, 10)]
    hold = 5
    vals = []
    for x in grid:
        r = []
        for k in range(hold):
            if view.done:
                break
            o = view.step({"kind": "levers", "values": {A: x, B: 8.0}})["observation"]
            if k >= hold - 2:
                r.append(_rev(o))
        vals.append(float(np.mean(r)) if r else np.nan)
    v = np.asarray(vals)
    mid = 0.5 * (np.nanmin(v) + np.nanmax(v))
    above = [g for g, x in zip(grid, v) if x > mid]
    est = above[0] if above else None
    hi = float(env._config["nodes"][env._engine.idx[cl["latch"]]]["params"]["hi"])
    return {"question": "from which setting does the pathway open", "estimate": est, "truth": hi,
            "ok": est is not None and -0.25 <= est - hi <= 1.25, "sweep": vals}
