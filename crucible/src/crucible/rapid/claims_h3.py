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


def _sign_question(env, lever: str, measure, block: int | None = None, question: str = "", hi_f: float = 0.9,
                   lo_f: float = 0.1) -> dict:
    """Does a higher setting of `lever` (75% vs 25% of its range) raise `measure` per tick, in blocks?"""
    view = PublicView(env)
    p = _lever_props(view)[lever]
    lo, hi = p["minimum"], p["maximum"]
    act = {"A": {"kind": "levers", "values": {lever: lo + hi_f * (hi - lo)}},
           "B": {"kind": "levers", "values": {lever: lo + lo_f * (hi - lo)}}}
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
    return _sign_question(env, name, _rev, block=8, hi_f=1.0, lo_f=0.0, question=f"does raising {name} raise sales")


def lever_raises_sales_any(env) -> dict:
    """Question: does holding the key lever higher raise sales? Public: blocks of the lever at its top and bottom;
    every reading and revenue is compared between the two arms and the sign of the clearest response is the
    answer (an intermediate gauge free of the demand shocks can show the effect more clearly than sales)."""
    from .oracles import fork
    view = PublicView(env)
    name = _lname(env, env._config["claim"]["lever"])
    p = _lever_props(view)[name]
    act = {"A": {"kind": "levers", "values": {name: p["maximum"]}},
           "B": {"kind": "levers", "values": {name: p["minimum"]}}}
    block = 8
    truth = contrast_truth(fork(env), act, _rev, block)
    rows = {"A": [], "B": []}
    for k in range(view.horizon - view.t):
        arm = "A" if (k // block) % 2 == 0 else "B"
        o = view.step(act[arm])["observation"]
        if k % block >= block // 2:
            rows[arm].append(dict(o["numbers"], revenue=_rev(o)))
    zs = {}
    for key in rows["A"][0]:
        xa = np.asarray([r.get(key, np.nan) for r in rows["A"]], dtype=float)
        xb = np.asarray([r.get(key, np.nan) for r in rows["B"]], dtype=float)
        xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
        if len(xa) > 1 and len(xb) > 1:
            se = np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb)) + 1e-9
            zs[key] = float((xa.mean() - xb.mean()) / se)
    best = max(zs, key=lambda k: abs(zs[k]))
    est = float(np.sign(zs[best]))
    return {"question": f"does raising {name} raise sales", "truth": float(np.sign(truth)), "estimate": est,
            "clearest_reading": best, "z": zs[best], "ok": bool(est == np.sign(truth) and abs(zs[best]) >= 2.0)}


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
    grid = [0.5, 1.5, 2.5, 3.5, 5.0, 8.0]
    hold = max(4, min(8, (view.horizon - view.t) // len(grid)))
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


# ====================================================================== 8.10 changing environments
def _at(env, nid: str) -> float:
    return float(env._config["nodes"][env._engine.idx[nid]]["params"]["at"])


def _split_effect(env, act_hi: dict, act_lo: dict, node: str, at: float, settle: int = 3, ticks: int | None = None):
    """Per-tick effect hi - lo on `node` (forks, common noise), averaged before the switch and after it."""
    T = ticks or env.horizon
    d = _series(env, _hold(act_hi), T, node) - _series(env, _hold(act_lo), T, node)
    t = np.arange(T)
    before = d[(t >= settle) & (t < at)]
    after = d[t >= at + settle]
    return float(before.mean()), float(after.mean())


def _change_point(view, act_a: dict, act_b: dict, measure, run: int = 2) -> tuple[float, np.ndarray]:
    """Public: alternate actions A and B in runs of `run` ticks to the horizon, record measure(A) - measure(B) on the
    last tick of each pair of runs, and place the single change point that best splits the series (least squares).
    Returns (estimated tick of the change, the series)."""
    diffs, times = [], []
    while not view.done:
        vals = {}
        for lab, act in (("A", act_a), ("B", act_b)):
            for k in range(run):
                if view.done:
                    break
                o = view.step({"kind": "levers", "values": act})["observation"]
                if k == run - 1:
                    vals[lab] = measure(o)
        if len(vals) == 2:
            diffs.append(vals["A"] - vals["B"])
            times.append(view.t - 2 * run)
    d = np.asarray(diffs)
    best, k_best = np.inf, 1
    for k in range(2, len(d) - 1):
        sse = ((d[:k] - d[:k].mean()) ** 2).sum() + ((d[k:] - d[k:].mean()) ** 2).sum()
        if sse < best:
            best, k_best = sse, k
    return float(times[k_best]), d


def sign_reversal(env) -> dict:
    """Holding the lever high raises sales before the hidden tick and lowers them after it (forks)."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    at = _at(env, cl["regime"])
    before, after = _split_effect(env, {A: 8.0, B: 2.0}, {A: 2.0, B: 2.0}, cl["outcome"], at)
    return {"ok": before > 0.5 and after < -0.5, "effect_before": before, "effect_after": after, "switch_tick": at}


def reversal_time(env) -> dict:
    """Question: when does the lever's effect on sales reverse? Public: alternate the lever high and low, locate the
    change in the high-low revenue difference."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    est, d = _change_point(PublicView(env), {A: 9.0, B: 2.0}, {A: 1.0, B: 2.0}, _rev)
    truth = _at(env, cl["regime"])
    return {"question": "at which tick does the lever's effect reverse", "estimate": est, "truth": truth,
            "ok": abs(est - truth) <= 6.0 and d[:3].mean() > 0 > d[-3:].mean()}


def constraint_tightens(env) -> dict:
    """The marginal product of materials falls by more than half once the machines lose capacity (forks)."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["constrained"])
    at = _at(env, cl["regime"])
    before, after = _split_effect(env, {A: 8.0, B: 3.0}, {A: 5.0, B: 3.0}, cl["outcome"], at)
    return {"ok": before > 0.3 and after < 0.5 * before, "materials_effect_before": before,
            "materials_effect_after": after, "switch_tick": at}


def constraint_time(env) -> dict:
    """Question: when did extra materials stop paying? Public: alternate materials high and medium."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["constrained"])
    est, d = _change_point(PublicView(env), {A: 9.0, B: 6.0}, {A: 2.0, B: 6.0}, _rev)
    truth = _at(env, cl["regime"])
    return {"question": "at which tick does the constraint tighten", "estimate": est, "truth": truth,
            "ok": abs(est - truth) <= 6.0 and d[:3].mean() > d[-3:].mean()}


def drifting_optimum(env) -> dict:
    """The setting of the design lever that maximises sales moves by at least three units between the start and the
    end of the episode, in the claimed direction (forks over a grid of settings)."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    H = env.horizon
    grid = np.arange(0.0, 10.01, 0.5)
    early, late = [], []
    for x in grid:
        s = _series(env, _hold({A: float(x), B: 3.0}), H, cl["outcome"])
        early.append(s[1:6].mean())
        late.append(s[H - 5:].mean())
    be, bl = float(grid[int(np.argmax(early))]), float(grid[int(np.argmax(late))])
    want = 1.0 if cl["direction"] == "up" else -1.0
    return {"ok": (bl - be) * want >= 3.0, "best_early": be, "best_late": bl, "direction": cl["direction"]}


def drift_direction(env) -> dict:
    """Question: does the best design setting rise or fall during the episode? Public: cycle three settings early
    and late, compare which earns most."""
    view = PublicView(env)
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    grid = [1.5, 5.0, 8.5]
    H = view.horizon

    def window(n):
        tot = {x: [] for x in grid}
        for _ in range(n):
            for x in grid:
                for k in range(2):
                    if view.done:
                        return tot
                    o = view.step({"kind": "levers", "values": {A: x, B: 3.0}})["observation"]
                    if k == 1:
                        tot[x].append(_rev(o))
        return tot
    e = window(3)
    while view.t < H - 18 and not view.done:
        view.step({"kind": "levers", "values": {A: 5.0, B: 3.0}})
    lt = window(3)
    be = max(grid, key=lambda x: np.mean(e[x]) if e[x] else -np.inf)
    bl = max(grid, key=lambda x: np.mean(lt[x]) if lt[x] else -np.inf)
    est = "up" if bl > be else ("down" if bl < be else "none")
    return {"question": "does the best setting rise or fall", "estimate": est, "truth": cl["direction"],
            "ok": est == cl["direction"]}


def demand_shift(env) -> dict:
    """The marginal value of capacity jumps at the hidden tick, in the claimed direction (forks)."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    at = _at(env, cl["regime"])
    x = float(cl["capacity_before"])
    before, after = _split_effect(env, {A: x + 3.0, B: 2.0}, {A: x, B: 2.0}, cl["outcome"], at, settle=4)
    ok = after > 1.3 * before if cl["direction"] == "up" else after < 0.7 * before
    return {"ok": bool(ok and before > 0), "capacity_effect_before": before, "capacity_effect_after": after,
            "direction": cl["direction"], "switch_tick": at}


def shift_direction(env) -> dict:
    """Question: did demand rise or fall? Public: alternate capacity high and low; when demand rises extra capacity
    is worth more than before, when it falls less (compare the high-low revenue difference late and early)."""
    view = PublicView(env)
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    _, d = _change_point(view, {A: 8.0, B: 2.0}, {A: 3.0, B: 2.0}, _rev)
    n = max(2, len(d) // 3)
    early, late = float(d[:n].mean()), float(d[-n:].mean())
    est = "up" if late > 1.2 * early else ("down" if late < 0.8 * early else "none")
    return {"question": "did demand rise or fall", "estimate": est, "truth": cl["direction"],
            "ok": est == cl["direction"], "capacity_value_early": early, "capacity_value_late": late}


def tech_transition(env) -> dict:
    """The new process's yield per unit rises with its own use (learning), and the old process loses most of its
    effect at the hidden tick (forks)."""
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    A, Bn = _lname(env, cl["lever"]), _lname(env, cl["new"])
    prod = cfg["nodes"][eng.idx[cl["outcome"]]]["ports"]["x"][1][0]
    y = _series(env, _hold({A: 0.0, Bn: 8.0}), 30, prod)
    learn = float(y[25] / max(y[2], 1e-9))
    at = _at(env, cl["regime"])
    before, after = _split_effect(env, {A: 8.0, Bn: 0.0}, {A: 2.0, Bn: 0.0}, cl["outcome"], at)
    return {"ok": learn > 2.0 and before > 0.5 and after < 0.5 * before, "yield_growth": learn,
            "old_effect_before": before, "old_effect_after": after}


def obsolescence_time(env) -> dict:
    """Question: when did the established process lose its value? Public: alternate it high and low."""
    cl = env._config["claim"]
    A, Bn = _lname(env, cl["lever"]), _lname(env, cl["new"])
    est, d = _change_point(PublicView(env), {A: 9.0, Bn: 0.0}, {A: 1.0, Bn: 0.0}, _rev)
    truth = _at(env, cl["regime"])
    return {"question": "at which tick does the old process become obsolete", "estimate": est, "truth": truth,
            "ok": abs(est - truth) <= 6.0 and d[:3].mean() > d[-3:].mean()}


def rule_change(env) -> dict:
    """Before the hidden tick the first lever pays and the second barely does; afterwards the reverse (forks)."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    at = _at(env, cl["regime_a"])
    ab, aa = _split_effect(env, {A: 8.0, B: 0.0}, {A: 2.0, B: 0.0}, cl["outcome"], at, settle=4)
    bb, ba = _split_effect(env, {A: 0.0, B: 8.0}, {A: 0.0, B: 2.0}, cl["outcome"], at, settle=4)
    return {"ok": ab > 3 * max(aa, 0.0) and ba > 3 * max(bb, 0.0) and ab > 0.5 and ba > 0.5,
            "first_before": ab, "first_after": aa, "second_before": bb, "second_after": ba, "switch_tick": at}


def rule_change_time(env) -> dict:
    """Question: when did the platform change what it rewards? Public: alternate between the two levers."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    est, d = _change_point(PublicView(env), {A: 9.0, B: 1.0}, {A: 1.0, B: 9.0}, _rev, run=4)
    truth = _at(env, cl["regime_a"])
    return {"question": "at which tick does the rule change", "estimate": est, "truth": truth,
            "ok": abs(est - truth) <= 7.0 and d[:2].mean() > 0 > d[-2:].mean()}


# ====================================================================== evaluation families
def latched_reversal(env) -> dict:
    """With the latch held open, the gated lever raises output before the hidden tick and lowers it afterwards; and
    the latch remembers: after a high push, a setting inside its band keeps it open, after a low one closed."""
    cfg = env._config
    cl = cfg["claim"]
    A, B = _lname(env, cl["latch_lever"]), _lname(env, cl["lever"])
    p = cfg["nodes"][env._engine.idx[cl["latch"]]]["params"]
    band = 0.5 * (p["lo"] + p["hi"])
    up = lambda k: {"kind": "levers", "values": {A: 10.0 if k < 8 else band, B: 7.0}}  # noqa: E731
    down = lambda k: {"kind": "levers", "values": {A: 0.0 if k < 8 else band, B: 7.0}}  # noqa: E731
    la, lb = _series(env, up, 16, cl["latch"]), _series(env, down, 16, cl["latch"])
    at = _at(env, cl["regime"])
    before, after = _split_effect(env, {A: 10.0, B: 8.0}, {A: 10.0, B: 1.0}, cl["outcome"], at, settle=4)
    return {"ok": bool(la[-1] == p["on"] and lb[-1] == p["off"] and before > 0.5 and after < -0.5),
            "latch_after_high": float(la[-1]), "latch_after_low": float(lb[-1]), "gated_effect_before": before,
            "gated_effect_after": after, "switch_tick": at}


def latched_reversal_time(env) -> dict:
    """Question: when does the open pathway start to lose money? Public: hold the latch lever high (pathway open),
    alternate the gated lever high and low, locate the change."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["latch_lever"]), _lname(env, cl["lever"])
    est, d = _change_point(PublicView(env), {A: 10.0, B: 9.0}, {A: 10.0, B: 1.0}, _rev, run=4)
    truth = _at(env, cl["regime"])
    return {"question": "at which tick does the gated pathway reverse", "estimate": est, "truth": truth,
            "ok": abs(est - truth) <= 7.0 and d[1:3].mean() > 0 > d[-2:].mean()}


def balancing_constraint(env) -> dict:
    """The balancing loop (immediate effect of materials on the hidden-loop output above the lasting one) and the
    constraint (materials' effect on output falls by more than half after the capacity cut)."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["constrained"])
    step = _series(env, lambda k: {"kind": "levers", "values": {A: 2.0 if k < 15 else 8.0, B: 3.0}}, 45, cl["core"])
    base = _series(env, _hold({A: 2.0, B: 3.0}), 45, cl["core"])
    dd = step - base
    short, late = float(dd[15]), float(dd[32:45].mean())
    at = _at(env, cl["regime"])
    before, after = _split_effect(env, {A: 8.0, B: 3.0}, {A: 4.0, B: 3.0}, cl["outcome"], at)
    return {"ok": short > 0 and 0 < late < 0.8 * short and before > 0.3 and after < 0.5 * before,
            "immediate_effect": short, "lasting_effect": late, "materials_effect_before": before,
            "materials_effect_after": after}


def competing_rule(env) -> dict:
    """Both levers track sales in the records; under intervention only the first moves sales before the hidden
    tick and only the second after it."""
    eng, st, cfg = inside(env)
    cl = cfg["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    hs = np.asarray(eng.history_of(st, _gauge_of(env, cl["outcome"])))
    ca = float(np.corrcoef(np.asarray(eng.history_of(st, A)), hs)[0, 1])
    cb = float(np.corrcoef(np.asarray(eng.history_of(st, B)), hs)[0, 1])
    at = _at(env, cl["regime_a"])
    ab, aa = _split_effect(env, {A: 8.0, B: 0.0}, {A: 2.0, B: 0.0}, cl["outcome"], at, settle=2)
    bb, ba = _split_effect(env, {A: 0.0, B: 8.0}, {A: 0.0, B: 2.0}, cl["outcome"], at, settle=2)
    return {"ok": ca > 0.3 and cb > 0.3 and ab > 0.5 and abs(bb) <= 0.02 * ab and ba > 0.5 and aa < 0.4 * ab,
            "records_corr_first": ca, "records_corr_second": cb, "first_before": ab, "first_after": aa,
            "second_before": bb, "second_after": ba}


def selection_flattens(env) -> dict:
    """The selected gauge responds to the lever less than the real output does, and reads above it (forks with
    common noise)."""
    cl = env._config["claim"]
    eng = env._engine
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    g = _gauge_of(env, cl["outcome"])
    k = eng.gauge_names.index(g)

    def run(x):
        e = fork(env)
        true, read = [], []
        for _ in range(25):
            e.step({"kind": "levers", "values": {A: x, B: 2.0}})
            true.append(float(e._state["h"][0, -1, eng.idx[cl["outcome"]]]))
            read.append(float(e._state["read"][0, k]))
        return np.asarray(true[5:]), np.asarray(read[5:])
    t_hi, r_hi = run(8.0)
    t_lo, r_lo = run(2.0)
    dt = float(t_hi.mean() - t_lo.mean())
    dr = float(np.nanmean(r_hi) - np.nanmean(r_lo))
    above = float(np.nanmean(np.concatenate([r_hi, r_lo])) - np.mean(np.concatenate([t_hi, t_lo])))
    missing = float(np.mean(np.isnan(np.concatenate([r_hi, r_lo]))))
    return {"ok": abs(dt) > 0.5 and abs(dr) < 0.7 * abs(dt) and above > 0 and missing > 0,
            "true_response": dt, "reported_response": dr, "reading_bias": above, "missing_share": missing}


def lock_in(env) -> dict:
    """Increasing returns with two attractors: after the same final settings (no push), a world pushed early ends
    locked high and a world never pushed ends low."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    push = lambda k: {"kind": "levers", "values": {A: 10.0 if k < 15 else 0.0, B: 0.0}}  # noqa: E731
    none = _hold({A: 0.0, B: 0.0})
    sp, sn = _series(env, push, 45, cl["share"]), _series(env, none, 45, cl["share"])
    late = _series(env, lambda k: {"kind": "levers", "values": {A: 10.0 if 30 <= k < 33 else 0.0, B: 0.0}}, 45,
                   cl["share"])
    return {"ok": sp[-1] > 0.7 and sn[-1] < 0.3 and late[-1] < 0.3, "share_after_early_push": float(sp[-1]),
            "share_never_pushed": float(sn[-1]), "share_after_short_late_push": float(late[-1])}


def push_locks_in(env) -> dict:
    """Question: does a temporary push lock the market in? Public: push at full strength for 12 ticks, then stop,
    and compare late revenue with revenue before the push. Truth: the same schedule on a fork, the hidden share at
    the end."""
    cl = env._config["claim"]
    A, B = _lname(env, cl["lever"]), _lname(env, cl["second"])
    sched = lambda k: {"kind": "levers", "values": {A: 10.0 if 3 <= k < 15 else 0.0, B: 0.0}}  # noqa: E731
    share = _series(env, sched, env.horizon, cl["share"])
    view = PublicView(env)
    revs = [_rev(view.step(sched(k))["observation"]) for k in range(view.horizon)]
    est = bool(np.mean(revs[-10:]) > 2.0 * max(np.mean(revs[:3]), 1.0))
    truth = bool(share[-1] > 0.5)
    return {"question": "does a temporary push lock the market in", "estimate": est, "truth": truth,
            "ok": est == truth}
