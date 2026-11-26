"""Evaluator side of the `program_exec` families (workstream H1): oracles, scripted public-information references,
identification procedures and intervention tests. Nothing in a world imports this module
(`admission.static_isolation` checks it); its functions read hidden structure and run only on copies of worlds.

Oracles are privileged programs: the hidden parameters are written into the source of a controller, scheduler or
solver, which then runs through the same sandbox and the same world code as anybody's program (or, where the
hidden state itself is what a program could never see, a privileged policy issuing ordinary actions). References
see only a `PublicView` and the public part of the configuration (names, ranges, costs, rules stated in the brief).
"""
from __future__ import annotations

import json

import numpy as np

from .identify import PublicView
from .oracles import best_of, fork, inside, run_fn


# ====================================================================== helpers
def deploy(source: str, ticks: int | None = None):
    """A privileged policy: deploy `source` at the first tick it sees (for the rest of the episode), then wait."""
    state = {"t0": None}

    def fn(e):
        if state["t0"] is None:
            state["t0"] = e.t
            return {"kind": "program", "source": source, "ticks": int(ticks or (e.horizon - e.t))}
        return None
    return fn


def submit(source: str):
    state = {"done": False}

    def fn(e):
        if not state["done"]:
            state["done"] = True
            return {"kind": "program", "source": source}
        return None
    return fn


def finish(view) -> None:
    while not view.done:
        view.step(None)


def lever_of(view) -> tuple[str, float, float]:
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    n = sorted(props)[0]
    return n, float(props[n]["minimum"]), float(props[n]["maximum"])


def consts(**kw) -> str:
    """Python assignments of JSON-native constants (JSON text is not Python: true/false/null)."""
    return "".join(f"{k} = {json.dumps(v)}\n" for k, v in kw.items()).replace("true", "True") \
        .replace("false", "False").replace("null", "None")


# ====================================================================== fast_control
FC_SRC = """
def act(obs, caller):
    y = obs["reading"]
    d = 0.0
    for start, val in D:
        if obs["tick"] >= start:
            d = val
    u = -((1.0 + A) * G * y + d) / B
    return {"u": max(-UMAX, min(UMAX, u))}
"""

FC_PI_SRC = """
def act(obs, caller):
    m = caller.get("memory") or {"i": D0}
    y = obs["reading"]
    i = m["i"] + KI * y
    u = -((1.0 + A) * G * y + i) / B
    return {"u": max(-UMAX, min(UMAX, u)), "memory": {"i": max(-1.0, min(1.0, i))}}
"""


def fc_source(a: float, b: float, schedule, g: float, umax: float) -> str:
    """Deadbeat-type feedback with the load disturbance schedule [[tick, value], ...] compensated."""
    return consts(A=a, B=b, D=[[int(s), float(v)] for s, v in schedule], G=g, UMAX=umax) + FC_SRC


def oracle_fast_control(env) -> dict:
    eng, st, c = inside(env)
    h = c["hidden"]
    pols = {f"deadbeat_gain_{g}": deploy(fc_source(h["a"], h["b"], h["d"], g, c["umax"])) for g in (0.6, 0.85)}
    return best_of(env, pols)


EXCITE_SRC = """
def act(obs, caller):
    return {"u": AMP if (obs["step"] + obs["tick"]) % 2 == 0 else -AMP}
"""


def fc_identify(view, amp: float = 0.6, ticks: int = 1) -> dict:
    """A system-identification experiment run as a program: an excitation alternating the actuator between +amp
    and -amp every inner step (known to the experimenter); then regress each reading on the previous one and the
    actuator value (y' = (1 + a) y + b u + d). Ticks with a trip or a halt are left out."""
    name, lo, hi = lever_of(view)
    amp_u = amp * hi
    rows, ys = [], []
    tries = got = 0
    while got < ticks and not view.done and tries < 3 * ticks:
        tries += 1
        t = view.t
        r = view.step({"kind": "program", "source": consts(AMP=amp_u) + EXCITE_SRC, "ticks": 1})
        obs = r["observation"]
        tr = obs["arrays"].get("trace") or []
        if any("halted" in s or "tripped" in s for s in obs["text"]) or len(tr) < 2:
            for _ in range(3):
                if not view.done:
                    view.step(None)
            continue
        for j in range(len(tr) - 1):
            rows.append([tr[j], amp_u if (j + t) % 2 == 0 else -amp_u, 1.0])
            ys.append(tr[j + 1])
        got += 1
    if len(rows) < 6:
        return {"ok": False}
    X, y = np.asarray(rows), np.asarray(ys)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"ok": True, "a": float(beta[0] - 1.0), "b": float(beta[1]), "d": float(beta[2]), "umax": hi}


def reference_fast_control(view, env) -> None:
    """Identify by excitation, then deploy the deadbeat feedback with the bias estimated at the start (a
    competent but static design: it does not know the load will change)."""
    est = fc_identify(view)
    if est["ok"] and abs(est["b"]) > 1e-3:
        a = float(np.clip(est["a"], 0.0, 0.5))
        view.step({"kind": "program", "source": fc_source(a, est["b"], [[0, est["d"]]], 0.7, est["umax"]),
                   "ticks": view.horizon - view.t})
    finish(view)


def identify_fast_control(env) -> dict:
    est = fc_identify(PublicView(env))
    b = env._config["hidden"]["b"]
    s = float(np.sign(est.get("b", 0.0))) if est["ok"] else 0.0
    return {"question": "the sign of the actuator's effect on the state", "estimate": s,
            "truth": float(np.sign(b)), "ok": bool(s == np.sign(b))}


def unstable_open_loop(env) -> dict:
    """Two copies that differ only in the initial state, same noise, actuator off: the difference grows by exactly
    (1 + a)^K per tick (an unstable linear plant); and no constant actuator value keeps it in band."""
    c, h = env._config, env._config["hidden"]
    e1, e2 = fork(env), fork(env)
    e1._state["x"], e2._state["x"] = 0.05, 0.10
    e1.step(None)
    e2.step(None)
    ratio = (e2._state["x"] - e1._state["x"]) / 0.05
    want = (1.0 + h["a"]) ** int(c["inner"])
    name = c["u_name"]
    fracs = []
    for u in np.linspace(-c["umax"], c["umax"], 9):
        e = fork(env)
        i0, s0 = e._state["inband"], e._state["steps"]
        for _ in range(6):
            e.step({"kind": "levers", "values": {name: float(u)}})
        fracs.append((e._state["inband"] - i0) / max(1, e._state["steps"] - s0))
    return {"ok": bool(abs(ratio - want) < 1e-6 * want and want > 1.5 and max(fracs) < 0.6),
            "growth_per_tick": float(ratio), "expected": float(want), "best_constant_inband": float(max(fracs))}


def inner_rate_control(env) -> dict:
    """A controller executed at the inner rate (true gains) keeps the plant in band far more often than the best
    per-tick manual policy that knows the exact state and the load and aims each tick's held actuator at zero; and
    after a load step, a static design is off-centre while the same design with integral action (state kept in
    the program's memory) is not."""
    c, h = env._config, env._config["hidden"]
    K, a, b = int(c["inner"]), h["a"], h["b"]
    sc = env._engine.sc
    e = fork(env)
    e.step({"kind": "program", "source": fc_source(a, b, h["d"], 0.7, c["umax"]), "ticks": 8})
    for _ in range(7):
        e.step(None)
    prog = e._state["inband"] / max(1, e._state["steps"])
    g = (1.0 + a) ** K
    geo = (g - 1.0) / a

    def manual(e2):
        x = e2._state["x"]
        uu = -(g * x / geo + sc.bias(e2.t)) / b
        return {"kind": "levers", "values": {c["u_name"]: float(np.clip(uu, -c["umax"], c["umax"]))}}
    m = fork(env)
    run_fn(m, manual, ticks=8)
    man = m._state["inband"] / max(1, m._state["steps"])
    # integral action: after the last load step, a static bias leaves an offset that the integrator removes
    last = h["d"][-1][0]
    off = {}
    for lab, src in (("static", fc_source(a, b, [[0, h["d"][0][1]]], 0.7, c["umax"])),
                     ("integral", consts(A=a, B=b, D0=h["d"][0][1], G=0.7, KI=0.15, UMAX=c["umax"]) + FC_PI_SRC)):
        f = fork(env)
        f.step({"kind": "program", "source": src, "ticks": f.horizon})
        xs = []
        while not f.done:
            f.step(None)
            if f.t > last + 2:
                xs.append(float(np.mean(f._state["trace"])))
        off[lab] = abs(float(np.mean(xs))) if xs else 0.0
    return {"ok": bool(prog >= 0.85 and prog > man + 0.15 and off["integral"] < off["static"]),
            "program_inband": float(prog), "manual_inband": float(man), "offset_after_load_step": off}


# ====================================================================== rule_induction
RULE_SRC = """
def act(r, caller):
    k = CATS.index(r[FC])
    a, b = float(r[FA]), float(r[FB])
    return {"value": OFF[k] + SL[k] * b + K1 * a + K2 * max(0.0, min(a, CUT) - TH) ** P + LIN * max(0.0, a - CUT)}
"""


def rule_source(c: dict, off, sl, k1, k2, th, p, lin=0.0, cut=1e9) -> str:
    fa, fb, fc = c["fields"]
    return consts(FA=fa, FB=fb, FC=fc, CATS=list(c["categories"]), OFF=[float(x) for x in off],
                  SL=[float(x) for x in sl], K1=float(k1), K2=float(k2), TH=float(th), P=float(p), LIN=float(lin),
                  CUT=float(cut)) + RULE_SRC


def oracle_rule_induction(env) -> dict:
    c = env._config
    h = c["hidden"]
    return best_of(env, {"knows_rule": submit(rule_source(c, h["off"], h["slope"], h["k1"], h["k2"], h["theta"],
                                                          h["p"]))})


def probe_many(view, recs: list[dict]) -> list:
    """Probe records, up to 16 per tick (one sequence); None where a probe was refused."""
    out = []
    for i in range(0, len(recs), 16):
        chunk = recs[i:i + 16]
        if view.done:
            out += [None] * len(chunk)
            continue
        r = view.step({"kind": "sequence", "steps": [{"kind": "query", "target": "probe", "args": {"record": x}}
                                                     for x in chunk]})
        res = r["result"]
        got = res.get("results") if res["ok"] else res.get("done_steps")
        if isinstance(got, dict):
            got = [got]
        got = list(got or [])
        out += [g.get("value") if isinstance(g, dict) else None for g in got] + [None] * (len(chunk) - len(got))
    return out


def ri_fit(view, c: dict, scan: int = 17, extra: int = 2) -> dict:
    """Probe an a-scan (first category, b in the middle) and the b-slope of every category low in the box, then
    fit offset + k1 a + k2 max(0, a - theta)^p over a grid of (theta, p)."""
    fa, fb, fc = c["fields"]
    cats = list(c["categories"])
    (alo, ahi), (blo, bhi) = c["box"][fa], c["box"][fb]
    bm = (blo + bhi) / 2
    a_low = alo + 0.15 * (ahi - alo)
    scan_a = [float(a) for a in list(np.linspace(alo, ahi, scan)) + list(np.linspace(ahi - 2.0, ahi, 5)) * extra]
    recs = [{fa: a, fb: bm, fc: cats[0]} for a in scan_a]
    for ct in cats:
        recs += [{fa: a_low, fb: blo, fc: ct}, {fa: a_low, fb: bhi, fc: ct}]
    vals = probe_many(view, recs)
    keep = [i for i in range(len(scan_a)) if vals[i] is not None]
    A_ = np.asarray([scan_a[i] for i in keep])
    Y_ = np.asarray([vals[i] for i in keep])
    per = {k: (vals[len(scan_a) + 2 * k], vals[len(scan_a) + 2 * k + 1]) for k in range(len(cats))}
    best = None
    for th in np.arange(alo + 0.5, ahi - 0.5, 0.1):
        for p in np.arange(1.0, 3.01, 0.1):
            X = np.c_[np.ones_like(A_), A_, np.maximum(0.0, A_ - th) ** p]
            beta, *_ = np.linalg.lstsq(X, Y_, rcond=None)
            sse = float(((X @ beta - Y_) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, float(th), float(p), beta)
    _, th, p, (c0, k1, k2) = best
    sl, off = [], []
    g_low = k1 * a_low + k2 * max(0.0, a_low - th) ** p
    for k in range(len(cats)):
        v0, v1 = per[k]
        if v0 is None or v1 is None:
            sl.append(0.0)
            off.append(c0 if k == 0 else 0.0)
            continue
        s_ = (v1 - v0) / (bhi - blo)
        sl.append(float(s_))
        off.append(float(v0 - s_ * blo - g_low))
    off[0] = float((off[0] + (c0 - sl[0] * bm)) / 2)
    return {"theta": th, "p": p, "k1": float(k1), "k2": float(k2), "off": off, "slope": sl}


def reference_rule_induction(view, env) -> None:
    c = env._config
    f = ri_fit(view, c)
    view.step({"kind": "program", "source": rule_source(c, f["off"], f["slope"], f["k1"], f["k2"], f["theta"],
                                                        f["p"])})
    finish(view)


def identify_rule_induction(env) -> dict:
    c = env._config
    f = ri_fit(PublicView(env), c)                 # the reference's own probe design (a public procedure)
    th = c["hidden"]["theta"]
    return {"question": "where the response to the first field starts to bend", "estimate": f["theta"],
            "truth": float(th), "ok": bool(abs(f["theta"] - th) <= 0.75)}


def kink_at_threshold(env) -> dict:
    """Noise-free probes on a copy along the first field: second differences vanish below the hidden threshold
    and are positive (the response bends upward) beyond it."""
    c, h = env._config, env._config["hidden"]
    e = fork(env)
    e._config["probe_sd"] = 0.0
    e._state["budget"] = 1e9
    fa, fb, fc = c["fields"]
    hstep = 0.25
    grid = np.arange(0.0, 10.0 + 1e-9, hstep)
    v = np.array(probe_many(PublicView(e), [{fa: float(a), fb: 1.0, fc: c["categories"][0]} for a in grid]),
                 dtype=float)
    d2 = v[2:] - 2 * v[1:-1] + v[:-2]
    mid = grid[1:-1]
    below = np.abs(d2[mid < h["theta"] - 2 * hstep]).max(initial=0.0)
    above = d2[mid > h["theta"] + 2 * hstep]
    return {"ok": bool(below < 5e-6 and above.size and (above > 1e-4).all()),
            "max_d2_below": float(below), "min_d2_above": float(above.min()) if above.size else None}


def holdout_needs_the_law(env) -> dict:
    """The exact rule passes every validation and holdout test; the same rule continued along its tangent beyond
    the probe box (identical inside it) passes every validation test but fails much of the holdout suite: the
    holdout measures conditions only the law itself predicts."""
    c, h = env._config, env._config["hidden"]
    exact = rule_source(c, h["off"], h["slope"], h["k1"], h["k2"], h["theta"], h["p"])
    ahi = c["box"][c["fields"][0]][1]
    lin = h["k2"] * h["p"] * max(0.0, ahi - h["theta"]) ** (h["p"] - 1.0)
    tangent = rule_source(c, h["off"], h["slope"], h["k1"], h["k2"], h["theta"], h["p"], lin=lin, cut=ahi)
    out = {}
    for lab, src in (("exact", exact), ("tangent_beyond_box", tangent)):
        e = fork(env)
        r = e.step({"kind": "program", "source": src})
        val = r["result"]["results"]["validation"]
        out[lab] = (val["passed"] / val["of"], e._state["holdout_pass"])
    ok = out["exact"] == (1.0, 1.0) and out["tangent_beyond_box"][0] == 1.0 and out["tangent_beyond_box"][1] < 0.8
    return {"ok": bool(ok), "validation_holdout": out}


# ====================================================================== deadtime_control
DT_SMITH_SRC = """
def act(obs, caller):
    m = caller.get("memory") or {"h": [0.0] * (L + 1), "zm": 0.0, "zd": 0.0, "i": 0.0}
    y, r = obs["reading"], obs["setpoint"]
    e = r - (y - m["zd"] + m["zm"])
    i = m["i"] + e / TI
    u = KC * (e + i)
    if u > UMAX or u < 0.0:
        i = m["i"]
        u = max(0.0, min(UMAX, u))
    h = (m["h"] + [u])[-(L + 1):]
    zm = m["zm"] + (G * u - m["zm"]) / TAU
    zd = m["zd"] + (G * h[0] - m["zd"]) / TAU
    return {"u": u, "memory": {"h": h, "zm": zm, "zd": zd, "i": i}}
"""

DT_PI_SRC = """
def act(obs, caller):
    m = caller.get("memory") or {"i": I0}
    e = obs["setpoint"] - obs["reading"]
    i = m["i"] + e
    u = KC * (e + i / TI)
    if u > UMAX or u < 0.0:
        i = m["i"]
        u = max(0.0, min(UMAX, u))
    return {"u": u, "memory": {"i": i}}
"""


def smith_source(c: dict, G: float, tau: float, L: int, lam: float) -> str:
    return consts(G=G, TAU=tau, L=int(L), KC=tau / (G * lam), TI=tau, UMAX=c["umax"]) + DT_SMITH_SRC


def pi_source(c: dict, G: float, tau: float, L: int, r0: float, umax: float) -> str:
    tc = max(1.0, float(L))
    kc = tau / (G * (tc + L))
    ti = min(tau, 4 * (tc + L))
    return consts(KC=kc, TI=ti, I0=(r0 / G) / kc * ti, UMAX=umax) + DT_PI_SRC


def oracle_deadtime_control(env) -> dict:
    c, h = env._config, env._config["hidden"]
    return best_of(env, {f"smith_predictor_lambda_{lam}": deploy(smith_source(c, h["G"], h["tau"], h["L"], lam))
                         for lam in (2.0, 4.0)})


def dt_identify(view, frac: float = 0.5) -> dict:
    """A step test with the manual lever (two ticks), then a first-order-plus-dead-time fit by grid search."""
    name, lo, hi = lever_of(view)
    U1 = frac * hi
    ys = []
    for _ in range(2):
        if view.done:
            break
        r = view.step({"kind": "levers", "values": {name: U1}})
        ys += list(r["observation"]["arrays"].get("trace") or [])
    ys = np.asarray(ys, dtype=float)
    k = np.arange(len(ys))
    best = None
    for L in range(0, 9):
        for tau in np.arange(1.0, 15.01, 0.25):
            b = np.where(k > L, U1 * (1.0 - (1.0 - 1.0 / tau) ** np.maximum(0, k - L)), 0.0)
            den = float(b @ b)
            if den <= 0:
                continue
            G = float(b @ ys) / den
            sse = float(((G * b - ys) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, L, float(tau), G)
    _, L, tau, G = best
    return {"L": int(L), "tau": tau, "G": max(0.05, G), "umax": hi}


def reference_deadtime_control(view, env) -> None:
    """Public reference: identify a first-order-plus-dead-time model with a step test, then deploy a Smith
    predictor (dead-time compensating PI) built on the identified model."""
    est = dt_identify(view)
    c = dict(env._config, umax=est["umax"])
    view.step({"kind": "program", "source": smith_source(c, est["G"], est["tau"], est["L"], 3.0),
               "ticks": view.horizon - view.t})
    finish(view)


def identify_deadtime_control(env) -> dict:
    est = dt_identify(PublicView(env))
    L = env._config["hidden"]["L"]
    return {"question": "the dead time of the process (inner steps)", "estimate": est["L"], "truth": int(L),
            "ok": bool(abs(est["L"] - L) <= 1)}


def dead_time_exact(env) -> dict:
    """Two copies, same noise, one with the valve opened at step 0: their readings agree exactly for L + 1 steps
    (the input has not arrived) and differ from step L + 1 on."""
    c, h = env._config, env._config["hidden"]
    e1, e2 = fork(env), fork(env)
    r1 = e1.step({"kind": "levers", "values": {c["u_name"]: 0.0}})
    r2 = e2.step({"kind": "levers", "values": {c["u_name"]: 0.5 * c["umax"]}})
    d = np.abs(np.asarray(r2["observation"]["arrays"]["trace"]) - np.asarray(r1["observation"]["arrays"]["trace"]))
    L = int(h["L"])
    first = int(np.argmax(d > 1e-9)) if (d > 1e-9).any() else None
    return {"ok": bool(first == L + 1 and np.all(d[:L + 1] == 0.0)), "first_differing_step": first,
            "dead_time": L}


def _ultimate_gain(G: float, tau: float, L: int) -> float:
    """The proportional gain at which the linearised loop e[k+1] = e[k] + (-G K e[k-L] - e[k]) / tau stops being
    stable (bisection on the growth of a unit deviation over 600 inner steps; evaluator side, hidden plant)."""
    def grows(K):
        e = [1.0] * (L + 1)
        for _ in range(600):
            e.append(e[-1] + (-G * K * e[-1 - L] - e[-1]) / tau)
            if abs(e[-1]) > 1e6:
                return True
        return max(abs(x) for x in e[-60:]) > 0.5
    lo, hi = 0.0, 1.0 / G
    while not grows(hi) and hi < 1e4:
        lo, hi = hi, 2 * hi
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if grows(mid) else (mid, hi)
    return hi


def aggressive_gain_oscillates(env) -> dict:
    """The dead time is what makes high-gain feedback fail: a proportional gain a quarter above the delayed loop's
    ultimate gain (where it stops being stable) - a gain the same plant without its dead time tolerates - tracks
    the setpoint far worse with the plant's dead time in the loop than on a counterfactual copy of the plant whose
    dead time is removed, on the same noise (evaluator side)."""
    c, h = env._config, env._config["hidden"]
    ku = _ultimate_gain(h["G"], h["tau"], int(h["L"]))
    kp = 1.25 * ku
    ku0 = _ultimate_gain(h["G"], h["tau"], 0)
    src = consts(G=h["G"], KP=kp, UMAX=c["umax"]) + """
def act(obs, caller):
    r = obs["setpoint"]
    return {"u": max(0.0, min(UMAX, r / G + KP * (r - obs["reading"])))}
"""
    err = {}
    for lab, L in (("with_dead_time", int(h["L"])), ("dead_time_removed", 0)):
        e = fork(env)
        e._engine.sc.h["L"] = L
        e.step({"kind": "program", "source": src, "ticks": 6})
        for _ in range(5):
            e.step(None)
        err[lab] = e._state["err"] / max(1, e._state["steps"])
    return {"ok": bool(kp < ku0 and err["with_dead_time"] > 1.5 * err["dead_time_removed"]), "mean_abs_error": err,
            "gain": kp, "ultimate_gain_with_dead_time": ku, "ultimate_gain_without": ku0, "dead_time": int(h["L"])}


# ====================================================================== contention
CONT_SRC = """
def act(obs, caller):
    run = {}
    for j in obs["running"]:
        k = CLS[j["type"]]
        run[k] = run.get(k, 0) + 1
    start = []
    for j in obs["queue"]:
        if len(start) >= obs["free"]:
            break
        k = CLS[j["type"]]
        if run.get(k, 0) < PER:
            start.append(j["id"])
            run[k] = run.get(k, 0) + 1
    return {"start": start}
"""


def cont_source(types, cls, per: int) -> str:
    return consts(CLS={t: int(k) for t, k in zip(types, cls)}, PER=int(per)) + CONT_SRC


def oracle_contention(env) -> dict:
    c, h = env._config, env._config["hidden"]
    return best_of(env, {f"one_class_mix_{p}": deploy(cont_source(c["types"], h["cls"], p)) for p in (1, 2)})


def partitions(types: list) -> list[list[int]]:
    """Two equal classes, the first type in class 0: every way to split."""
    import itertools
    J = len(types)
    out = []
    for rest in itertools.combinations(range(1, J), J // 2 - 1):
        cls = [1] * J
        for i in (0,) + rest:
            cls[i] = 0
        out.append(cls)
    return out


def work_rate(prev: list, cur: list, K: int) -> list[float]:
    """Speeds of jobs running at the end of two consecutive ticks (work per inner step)."""
    p = {e["id"]: e["left"] for e in prev if e.get("kind") == "running"}
    return [(p[e["id"]] - e["left"]) / K for e in cur if e.get("kind") == "running" and e["id"] in p]


def cont_test(view, c: dict, ticks: int = 3) -> tuple[list[int], dict]:
    """Deploy the one-job-per-class scheduler for each candidate split in turn and keep the split under which
    running jobs progress fastest in total."""
    K = int(c["inner"])
    score = {}
    for cls in partitions(c["types"]):
        if view.done:
            break
        view.step({"kind": "program", "source": cont_source(c["types"], cls, 1), "ticks": ticks})
        prev = view.observe()["entities"]
        rates = []
        for _ in range(ticks - 1):
            if view.done:
                break
            r = view.step(None)
            cur = r["observation"]["entities"]
            rates.append(sum(work_rate(prev, cur, K)))
            prev = cur
        score[tuple(cls)] = float(np.mean(rates)) if rates else 0.0
    best = max(score, key=score.get) if score else tuple(partitions(c["types"])[0])
    return list(best), score


def cont_infer(view, c: dict, ticks: int = 12) -> tuple[list[int], dict]:
    """Watch the node under the built-in rule capped at two jobs (pairs of jobs share the node for several ticks,
    so speeds are measured cleanly over whole ticks) and explain
    each running job's progress per inner step by the jobs it shared the node with: under each candidate split
    of the types into two classes, a job's speed is v / (1 + kappa (n_same - 1)); v and kappa are fitted by least
    squares (a grid over kappa), and the split that explains the observed speeds best is kept. Returns (split,
    squared error per split)."""
    K = int(c["inner"])
    obs = []                                          # (job type, types running with it, observed speed)
    prev = view.observe()["entities"]
    for k in range(ticks):
        if view.done:
            break
        # first a job alone (its speed is the unhindered speed), then pairs
        cap = {"kind": "levers", "values": {c["cap_name"]: 1.0 if k < 2 else 2.0}}
        cur = view.step(cap)["observation"]["entities"]
        p = {e["id"]: e for e in prev if e.get("kind") == "running"}
        both = [e for e in cur if e.get("kind") == "running" and e["id"] in p]
        if len(both) == len([e for e in cur if e.get("kind") == "running"]) == len(p):
            # the same jobs ran through the whole tick: their speeds are clean
            types = [e["type"] for e in both]
            for e in both:
                obs.append((e["type"], types, (p[e["id"]]["left"] - e["left"]) / K))
        prev = cur
    alone = [v for _, ts, v in obs if len(ts) == 1]
    v0 = float(np.median(alone)) if alone else 1.0           # the brief: a lone job does one work unit per step
    err = {}
    for cls in partitions(c["types"]):
        of = dict(zip(c["types"], cls))
        best = np.inf
        y = np.asarray([v for *_, v in obs])
        for kap in np.linspace(0.0, 4.0, 41):
            x = np.asarray([1.0 / (1.0 + kap * (sum(1 for t in ts if of[t] == of[ty]) - 1)) for ty, ts, _ in obs])
            if len(y):
                best = min(best, float(np.mean((v0 * x - y) ** 2)))
        err[tuple(cls)] = best
    pick = min(err, key=err.get) if err and np.isfinite(min(err.values())) else tuple(partitions(c["types"])[0])
    return list(pick), err


def reference_contention(view, env) -> None:
    c = env._config
    cls, _ = cont_infer(view, c)
    if not view.done:
        view.step({"kind": "program", "source": cont_source(c["types"], cls, 1), "ticks": view.horizon - view.t})
    finish(view)


def identify_contention(env) -> dict:
    c = env._config
    cls, score = cont_infer(PublicView(env), c)
    truth = c["hidden"]["cls"]
    same = cls == truth or cls == [1 - x for x in truth]
    return {"question": "which job types interfere with each other (the hidden classes)", "estimate": cls,
            "truth": truth, "ok": bool(same)}


def same_class_interferes(env) -> dict:
    """Two jobs of one class running together each progress 1 / (1 + kappa) as fast as two jobs of different
    classes (the cap keeps arrivals from joining them)."""
    c, h = env._config, env._config["hidden"]
    t0 = [c["types"][i] for i in range(len(c["types"])) if h["cls"][i] == 0][:2]
    t1 = [c["types"][i] for i in range(len(c["types"])) if h["cls"][i] == 1][:1]
    out = {}
    for lab, pair in (("same_class", t0), ("different_class", [t0[0], t1[0]])):
        e = fork(env)
        e._state["queue"], e._state["running"] = [], [{"id": 900000 + k, "type": ty, "left": 50.0}
                                                      for k, ty in enumerate(pair)]
        e.step({"kind": "levers", "values": {c["cap_name"]: 2}})
        out[lab] = float(np.mean([50.0 - j["left"] for j in e._state["running"] if j["id"] >= 900000]))
    ratio = out["different_class"] / out["same_class"]
    # progress is accounted in rounded work units, so the ratio matches 1 + kappa to about six digits
    return {"ok": bool(abs(ratio - (1.0 + h["kappa"])) < 1e-4 * (1.0 + h["kappa"])), "progress_per_tick": out, "ratio": ratio,
            "expected_ratio": 1.0 + h["kappa"]}


def class_aware_beats_caps(env) -> dict:
    """The scheduler that never runs two jobs of one class together earns more over ten ticks than the built-in
    rule at any concurrency cap."""
    c, h = env._config, env._config["hidden"]
    vals = {}
    for cap in range(1, int(c["slots"]) + 1):
        e = fork(env)
        run_fn(e, lambda x, cap=cap: {"kind": "levers", "values": {c["cap_name"]: cap}}, ticks=10)
        vals[f"cap_{cap}"] = float(e.outcome()["net"])
    e = fork(env)
    e.step({"kind": "program", "source": cont_source(c["types"], h["cls"], 1), "ticks": 10})
    run_fn(e, lambda x: None, ticks=9)
    vals["class_aware"] = float(e.outcome()["net"])
    best_cap = max(v for k, v in vals.items() if k.startswith("cap_"))
    return {"ok": bool(vals["class_aware"] > best_cap), "net_after_10_ticks": vals}


# ====================================================================== stream
ST_EXACT_SRC = """
def act(obs, caller):
    vals = [(r - O[i]) / G[i] for i, r in enumerate(obs["readings"]) if not STUCK[i]]
    if not vals:
        vals = [sum(obs["readings"]) / len(obs["readings"])]
    return {"estimate": sum(vals) / len(vals)}
"""

ST_CAL_SRC = """
def act(obs, caller):
    m = caller.get("memory") or {"res": [], "prev": None}
    rd = obs["readings"]
    n = len(rd)
    a = obs.get("audit")
    if a:
        m["res"] = (m["res"] + [[r - a["value"] for r in a["readings"]]])[-W:]
    off = [sum(x[i] for x in m["res"]) / len(m["res"]) for i in range(n)] if m["res"] else [0.0] * n
    prev = m["prev"]
    vals = [rd[i] - off[i] for i in range(n) if prev is None or rd[i] != prev[i]]
    m["prev"] = rd
    if not vals:
        vals = [sum(rd) / n]
    vals.sort()
    k = len(vals)
    est = vals[k // 2] if k % 2 else 0.5 * (vals[k // 2 - 1] + vals[k // 2])
    return {"estimate": est, "memory": m}
"""


# the same estimator written to do little work per call (compute is billed by the line executed): offsets are
# kept as a running average updated only when an audit arrives, and the median is taken by the built-in sort
ST_LEAN_SRC = """
def act(obs, caller):
    m = caller.get("memory") or {"off": None, "prev": None}
    rd = obs["readings"]
    a = obs.get("audit")
    if a:
        res = [r - a["value"] for r in a["readings"]]
        m["off"] = res if m["off"] is None else [K * o + (1 - K) * x for o, x in zip(m["off"], res)]
    off = m["off"] or [0.0] * len(rd)
    vals = sorted([r - o for r, o, p in zip(rd, off, m["prev"] or [None] * len(rd)) if r != p])
    m["prev"] = rd
    k = len(vals)
    if not k:
        return {"estimate": sum(rd) / len(rd), "memory": m}
    return {"estimate": vals[k // 2] if k % 2 else 0.5 * (vals[k // 2 - 1] + vals[k // 2]), "memory": m}
"""


def oracle_stream(env) -> dict:
    def exact(e):
        s = e._state
        return {"kind": "program", "ticks": 1, "source": consts(G=[float(x) for x in s["g"]],
                                                                O=[float(x) for x in s["o"]],
                                                                STUCK=[bool(x) for x in s["stuck"]]) + ST_EXACT_SRC}
    return best_of(env, {"knows_calibration": exact})


def reference_stream(view, env) -> None:
    view.step({"kind": "program", "source": consts(K=0.5) + ST_LEAN_SRC, "ticks": view.horizon - view.t})
    finish(view)


def identify_stream(env) -> dict:
    """Which sensor drifts fastest: from the daily audits (the true value of one sample and every sensor's reading
    of it), each sensor's residual (reading minus true value) is followed over the episode; a sensor whose reading
    stops changing has stuck, and only its ticks before that count. The drift speed is estimated from how much
    faster the residual spreads over five ticks than over one (a random walk's increments add up, measurement noise
    does not), so noise does not masquerade as drift."""
    view = PublicView(env)
    res, raw = [], []
    while not view.done and view.t < view.horizon - 1:
        r = view.step(None)
        n = r["observation"]["numbers"]
        a = np.asarray(r["observation"]["arrays"]["audit_readings"], dtype=float)
        raw.append(a)
        res.append(a - n["audit_value"])
    R, A = np.asarray(res), np.asarray(raw)
    speed_est = []
    for i in range(R.shape[1]):
        stuck = np.nonzero(np.abs(np.diff(A[:, i])) < 1e-9)[0]
        end = int(stuck[0]) + 1 if len(stuck) else len(A)
        r = R[:end, i]
        if len(r) < 8:
            speed_est.append(-np.inf)
            continue
        v1 = float(np.var(np.diff(r)))
        v5 = float(np.var(r[5:] - r[:-5]))
        speed_est.append(max(v5 - v1, 0.0) / 4.0)
    est = int(np.argmax(speed_est))
    h = env._config["hidden"]
    speed = np.asarray(h["o_sd"]) + env._config["mean"] * np.asarray(h["g_sd"])
    top2 = [int(i) for i in np.argsort(-speed)[:2]]
    return {"question": "which sensor drifts fastest", "estimate": est, "truth": top2, "ok": est in top2}


def sensors_drift(env) -> dict:
    """With no maintenance, the sensors' errors at the audit grow over the episode; recalibrating a sensor resets
    its error to measurement noise."""
    c = env._config
    e = fork(env)
    errs = []
    while not e.done:
        r = e.step(None)
        n = r["observation"]["numbers"]
        errs.append(float(np.sqrt(np.mean((np.asarray(r["observation"]["arrays"]["audit_readings"]) -
                                           n["audit_value"]) ** 2))))
    early, late = float(np.mean(errs[:3])), float(np.mean(errs[-3:]))
    f = fork(env)
    for _ in range(15):
        f.step(None)
    i = int(np.argmax(np.abs(np.asarray(f._state["g"]) * c["mean"] + np.asarray(f._state["o"]) - c["mean"])))
    before = abs(f._state["g"][i] * c["mean"] + f._state["o"][i] - c["mean"])
    f.step({"kind": "discrete", "decision": "recalibrate", "choice": c["sensor_names"][i]})
    return {"ok": bool(late > 2.0 * early and before > 3 * c["sensor_sd"]), "audit_rms_early": early,
            "audit_rms_late": late, "worst_sensor_bias_before_recalibration": float(before)}


def calibration_beats_averaging(env) -> dict:
    """On the same noise, a processor that inverts each sensor's true calibration estimates the true value more
    accurately over 15 ticks than the built-in average, and so does the public audit-calibrating processor (the
    mechanism: drifting gains and offsets and stuck sensors bias the plain average). Accuracy is the mean absolute
    estimation error per sample, which the world's own accounting accumulates (money also includes compute)."""
    out, net = {}, {}

    def err(e):
        return float(e._state["err"]) / max(1, int(e._state["samples"]) if not isinstance(e._state["samples"], list)
                                            else len(e._state["samples"]))
    e = fork(env)
    run_fn(e, lambda x: None, ticks=15)
    out["built_in_mean"], net["built_in_mean"] = err(e), float(e.outcome()["net"])
    e = fork(env)
    run_fn(e, lambda x: {"kind": "program", "ticks": 1, "source": consts(
        G=[float(v) for v in x._state["g"]], O=[float(v) for v in x._state["o"]],
        STUCK=[bool(v) for v in x._state["stuck"]]) + ST_EXACT_SRC}, ticks=15)
    out["exact_calibration"], net["exact_calibration"] = err(e), float(e.outcome()["net"])
    e = fork(env)
    e.step({"kind": "program", "source": consts(W=3) + ST_CAL_SRC, "ticks": 15})
    run_fn(e, lambda x: None, ticks=14)
    out["audit_calibration"], net["audit_calibration"] = err(e), float(e.outcome()["net"])
    # the mechanism claim: miscalibration biases the plain average (inverting the true calibration removes it);
    # whether the public audit method also beats it depends on how fast this world's sensors drift (recorded)
    return {"ok": bool(out["exact_calibration"] < 0.8 * out["built_in_mean"]), "estimation_error": out, "net": net,
            "audit_beats_average": bool(out["audit_calibration"] < out["built_in_mean"])}


# ====================================================================== router
RT_SRC = """
def svc(m, s, q):
    t = A[m] + B[m] * s ** C[m]
    if m == CONG and q > QSTAR:
        t *= 1.0 + CF * (q - QSTAR)
    return t


def act(obs, caller):
    s = obs["size"]
    best, arg = None, 0
    for m in range(len(A)):
        q = len(obs["sizes"][m])
        wait = sum(svc(m, x, q) for x in obs["sizes"][m])
        tot = wait + svc(m, s, q + 1)
        if best is None or tot < best:
            best, arg = tot, m
    return {"line": arg}
"""


def router_source(alpha, beta, gamma, cong=-1, qstar=99, cf=0.0) -> str:
    return consts(A=[float(x) for x in alpha], B=[float(x) for x in beta], C=[float(x) for x in gamma],
                  CONG=int(cong), QSTAR=int(qstar), CF=float(cf)) + RT_SRC


def oracle_router(env) -> dict:
    h = env._config["hidden"]
    return best_of(env, {"knows_service_laws": deploy(router_source(h["alpha"], h["beta"], h["gamma"],
                                                                    h["congested"], h["qstar"], h["cong"]))})


def rt_fit(view, c: dict, ticks: int = 6) -> dict:
    data: dict = {n: [] for n in c["line_names"]}
    for _ in range(ticks):
        if view.done:
            break
        r = view.step(None)
        for e in r["observation"]["entities"]:
            if e.get("kind") == "completed":
                data[e["line"]].append((e["size"], e["service_steps"]))
    al, be = [], []
    for n in c["line_names"]:
        d = np.asarray(data[n], dtype=float).reshape(-1, 2)
        if len(d) >= 3 and np.ptp(d[:, 0]) > 0.5:
            X = np.c_[np.ones(len(d)), d[:, 0]]
            (a, b), *_ = np.linalg.lstsq(X, d[:, 1], rcond=None)
        elif len(d):
            a, b = float(d[:, 1].mean()), 0.0
        else:
            a, b = 2.0, 1.0
        al.append(max(0.1, float(a)))
        be.append(max(0.0, float(b)))
    return {"alpha": al, "beta": be}


def reference_router(view, env) -> None:
    c = env._config
    f = rt_fit(view, c)
    if not view.done:
        view.step({"kind": "program", "source": router_source(f["alpha"], f["beta"], [1.0] * len(f["alpha"])),
                   "ticks": view.horizon - view.t})
    finish(view)


def identify_router(env) -> dict:
    c = env._config
    f = rt_fit(PublicView(env), c, ticks=8)
    est = int(np.argmax([a + b * 4.0 for a, b in zip(f["alpha"], f["beta"])]))
    sc = env._engine.sc
    truth = int(np.argmax([sc.service_time(m, 4.0, 1, 0.0) for m in range(len(c["line_names"]))]))
    return {"question": "which line is slowest for the largest jobs", "estimate": est, "truth": truth,
            "ok": est == truth}


def size_dependent_speed(env) -> dict:
    """Service-time laws cross: the line fastest for the smallest jobs is not the one fastest for the largest."""
    sc = env._engine.sc
    M = len(env._config["line_names"])
    small = [sc.service_time(m, 0.5, 1, 0.0) for m in range(M)]
    large = [sc.service_time(m, 4.0, 1, 0.0) for m in range(M)]
    return {"ok": bool(int(np.argmin(small)) != int(np.argmin(large))), "service_small": small,
            "service_large": large}


def congested_line_slows(env) -> dict:
    """Only the congestible line serves the same job more slowly behind a long queue."""
    h = env._config["hidden"]
    sc = env._engine.sc
    M = len(env._config["line_names"])
    q = h["qstar"] + 3
    ratio = [sc.service_time(m, 2.0, q, 0.0) / sc.service_time(m, 2.0, 1, 0.0) for m in range(M)]
    others = [r for m, r in enumerate(ratio) if m != h["congested"]]
    return {"ok": bool(ratio[h["congested"]] > 1.5 and all(abs(r - 1.0) < 1e-12 for r in others)),
            "slowdown_behind_queue": ratio}


# ====================================================================== fleet
FL_SRC = """
def act(obs, caller):
    cand = [(v, i) for i, v in enumerate(obs["vibration"]) if obs["down"][i] == 0 and v > VSTAR]
    cand.sort(reverse=True)
    return {"service": [i for _, i in cand[:obs["crew"]]]}
"""


def oracle_fleet(env) -> dict:
    c, h = env._config, env._config["hidden"]

    def by_wear(theta):
        def fn(e):
            s = e._state
            cand = sorted(((s["w"][i], i) for i in range(len(s["w"])) if s["down"][i] == 0
                           and s["w"][i] > theta * h["w0"]), reverse=True)[:int(c["crew"])]
            if not cand:
                return None
            return {"kind": "sequence", "steps": [{"kind": "discrete", "decision": "service",
                                                   "choice": c["machine_names"][i]} for _, i in cand]}
        return fn
    return best_of(env, {f"service_above_{th}_w0": by_wear(th) for th in (0.4, 0.6, 0.8, 1.0)})


def reference_fleet(view, env) -> None:
    """Run to failure for a while, learn the vibration seen just before failures, then deploy a policy that
    services the most vibrating machines above a cautious fraction of it."""
    c = env._config
    names = list(c["machine_names"])
    pre, last = [], None
    while not view.done and view.t < 10:
        r = view.step(None)
        obs = r["observation"]
        if last is not None:
            for s in obs["text"]:
                n = s.split(" ")[0]
                if n in names:
                    pre.append(last[names.index(n)])
        last = [obs["numbers"][f"{n}_vibration"] for n in names]
    if pre:
        vstar = 1.0 + 0.6 * (float(np.median(pre)) - 1.0)
    else:
        vstar = 1.0 + 0.6 * (max(last or [1.5]) - 1.0)
    if not view.done:
        view.step({"kind": "program", "source": consts(VSTAR=float(vstar)) + FL_SRC, "ticks": view.horizon - view.t})
    finish(view)


def identify_fleet(env) -> dict:
    view = PublicView(env)
    c = env._config
    for _ in range(5):
        r = view.step(None)
    v = [r["observation"]["numbers"][f"{n}_vibration"] for n in c["machine_names"]]
    est = int(np.argmax(v))
    w = np.asarray([max(0.0, x) for x in env._state["w"]])
    top2 = [int(i) for i in np.argsort(-w)[:2]]
    return {"question": "which machine is most worn now", "estimate": est, "truth": top2, "ok": est in top2}


def service_pays_later(env) -> dict:
    """Same noise (common random numbers), one copy services the three most worn working machines now, the other
    does not: servicing costs output at once (the mean immediate revenue change is negative), and the serviced
    machines fail less often afterwards (their failures over the next 15 ticks, summed over six noise replicates
    of the world). Failures of the other machines are not counted: they see the same draws in both copies."""
    from .env import Env
    c = env._config
    names = list(c["machine_names"])
    imm, fail_a, fail_b = [], 0, 0

    def failed(r, who):
        return sum(1 for x in r["observation"]["text"] if x.endswith(" failed") and x[:-7] in who)
    for r in range(6):
        base = env if r == 0 else Env(env.family, env.seed, env._config, env.difficulty,
                                      world_seed=env.world_seed + 104729 * r)
        e0 = fork(base)
        for _ in range(4):
            e0.step(None)
        worn = sorted(range(len(names)), key=lambda i: -e0._state["w"][i])
        worn = [i for i in worn if e0._state["down"][i] == 0][:3]
        who = {names[i] for i in worn}
        a, b = fork(e0), fork(e0)
        for i in worn:
            ra = a.step({"kind": "discrete", "decision": "service", "choice": names[i]})
            rb = b.step(None)
            imm.append(ra["observation"]["resources"]["revenue_cents_last_tick"] -
                       rb["observation"]["resources"]["revenue_cents_last_tick"])
            fail_a += failed(ra, who)
            fail_b += failed(rb, who)
        for _ in range(15):
            fail_a += failed(a.step(None), who)
            fail_b += failed(b.step(None), who)
    return {"ok": bool(np.mean(imm) < 0 and fail_a < fail_b), "mean_immediate_revenue_change": float(np.mean(imm)),
            "serviced_machine_failures_after_service": fail_a, "serviced_machine_failures_without": fail_b}


def vibration_tracks_wear(env) -> dict:
    """Vibration readings rank machines by their hidden wear (Spearman correlation over a run)."""
    e = fork(env)
    rng = np.random.default_rng(0)
    vib, wear = [], []
    while not e.done:
        a = None
        if rng.random() < 0.5:
            a = {"kind": "discrete", "decision": "service",
                 "choice": e._config["machine_names"][int(rng.integers(len(e._config["machine_names"])))]}
        e.step(a)
        for i in range(len(e._state["w"])):
            if e._state["down"][i] == 0:
                vib.append(e._state["vib"][i])
                wear.append(max(0.0, e._state["w"][i]))
    rv = np.argsort(np.argsort(vib))
    rw = np.argsort(np.argsort(wear))
    rho = float(np.corrcoef(rv, rw)[0, 1])
    return {"ok": bool(rho > 0.6), "spearman": rho}


# ====================================================================== cascade_guard
CG_SRC = """
def act(obs, caller):
    out = []
    for i, x in enumerate(obs["loads"]):
        cap = CAP[i] * MARGIN
        out.append((x - cap) / x if x > cap else 0.0)
    return {"shed": out}
"""


def guard_source(caps, margin: float) -> str:
    return consts(CAP=[float(x) for x in caps], MARGIN=float(margin)) + CG_SRC


def oracle_cascade_guard(env) -> dict:
    c, h = env._config, env._config["hidden"]
    caps = [r * f for r, f in zip(c["ratings"], h["factor"])]
    return best_of(env, {f"knows_capacity_{m}": deploy(guard_source(caps, m)) for m in (0.93, 0.97)})


def cg_learn(view, c: dict, ticks: int = 5) -> list:
    """Run unprotected for a few ticks; a unit's capacity is just below the load it tripped at (units that did
    not trip keep their rating)."""
    names = list(c["unit_names"])
    trip = {}
    for _ in range(ticks):
        if view.done:
            break
        r = view.step(None)
        for e in r["observation"]["entities"]:
            if e.get("kind") == "trip":
                trip[e["unit"]] = min(trip.get(e["unit"], 1e9), e["load"])
    return [min(c["ratings"][i], trip.get(n, 1e9) - 2.0) for i, n in enumerate(names)]


def reference_cascade_guard(view, env) -> None:
    c = env._config
    caps = cg_learn(view, c)
    if not view.done:
        view.step({"kind": "program", "source": guard_source(caps, 0.95), "ticks": view.horizon - view.t})
    finish(view)


def identify_cascade_guard(env) -> dict:
    c = env._config
    caps = cg_learn(PublicView(env), c, ticks=8)
    ratio = [x / r for x, r in zip(caps, c["ratings"])]
    est = int(np.argmin(ratio))
    f = c["hidden"]["factor"]
    top2 = [int(i) for i in np.argsort(f)[:2]]
    return {"question": "which unit is weakest relative to its rating", "estimate": est, "truth": top2,
            "ok": est in top2}


def trips_cascade(env) -> dict:
    """Controlled, unprotected (no shedding below 1.3 x rating), with the demand held steady at 70% of every unit's
    true capacity (no swing, no noise), so nothing trips by itself: forcing one unit out moves half its demand to
    each working neighbour; every neighbour that this pushes past its true capacity trips (and the cascade may run
    on: a tripped neighbour hands its own demand further along the ring)."""
    c, h = env._config, env._config["hidden"]
    N = len(c["ratings"])
    cap = [c["ratings"][i] * h["factor"][i] for i in range(N)]
    match, expected, base_trips, checked = 0, 0, 0, 0
    for i in range(N):
        a, b = fork(env), fork(env)
        for e in (a, b):
            e._config = dict(e._config, amp=0.0, noise_sd=0.0, base=[0.7 * x for x in cap])
            e._engine.config = e._config
            e._engine.sc.c = e._config
            e._state["margin"] = 1.3
        a._state["out"][i] = int(c["restart_steps"])
        tripped = set()
        for _ in range(2):
            ra, rb = a.step(None), b.step(None)
            tripped |= {c["unit_names"].index(x["unit"]) for x in ra["observation"]["entities"]
                        if x.get("kind") == "trip"}
            base_trips += sum(1 for x in rb["observation"]["entities"] if x.get("kind") == "trip")
        for n in ((i - 1) % N, (i + 1) % N):
            exp = 0.7 * cap[n] + 0.35 * cap[i] > cap[n]
            expected += int(exp)
            match += int(n in tripped) if exp else 0
            checked += int(exp)
    return {"ok": bool(match == expected and expected > 0 and base_trips == 0), "predicted_cascades": expected,
            "predicted_that_tripped": match, "trips_without_outage": base_trips}


def capacity_below_rating(env) -> dict:
    """The weakest unit trips at a steady load below its rating (and holds just below its true capacity)."""
    c, h = env._config, env._config["hidden"]
    i = int(np.argmin(h["factor"]))
    out = {}
    for lab, load in (("just_below_rating", 0.995 * c["ratings"][i]),
                      ("just_below_true_capacity", 0.99 * c["ratings"][i] * h["factor"][i])):
        e = fork(env)
        e._config["amp"], e._config["noise_sd"] = 0.0, 0.0
        e._config["base"] = [0.3 * r for r in c["ratings"]]
        e._config["base"][i] = load
        r = e.step(None)
        out[lab] = any(x.get("kind") == "trip" and x["unit"] == c["unit_names"][i] for x in r["observation"]["entities"])
    return {"ok": bool(out["just_below_rating"] and not out["just_below_true_capacity"] and h["factor"][i] < 0.99),
            "tripped": out, "factor": float(h["factor"][i])}


# ====================================================================== rollout
def _p_adopt(env, i: int, share: list) -> float:
    c, h = env._config, env._config["hidden"]
    u = c["users"][i]
    k = c["group_names"].index(u["group"])
    p = 1.0 / (1.0 + np.exp(-(h["w"][0] * u["f1"] + h["w"][1] * u["f2"] + h["bias"][k])))
    return float(p * (1.0 - h["q"]) if share[k] < h["tau"][k] else p)


def expose_ids(ids: list) -> dict:
    return {"kind": "program", "ticks": 1, "source": consts(IDS=[int(i) for i in ids]) +
            "def act(obs, caller):\n    return {\"expose\": IDS}\n"}


def oracle_rollout(env) -> dict:
    c, h = env._config, env._config["hidden"]

    def greedy(ready_only):
        def fn(e):
            sc = e._engine.sc
            share = sc.shares(e._state)
            left = e.horizon - e.t - 1 - h["delay"]
            ev = []
            for i in range(len(c["users"])):
                if e._state["status"][i] != 0:
                    continue
                k = c["group_names"].index(c["users"][i]["group"])
                if ready_only and share[k] < h["tau"][k]:
                    continue
                v = _p_adopt(e, i, share) * c["fee"] * max(0.0, left) - c["exposure_cost"]
                if v > 0:
                    ev.append((v, i))
            ev.sort(reverse=True)
            ids = [i for _, i in ev[:int(c["capacity"])]]
            return expose_ids(ids) if ids else None
        return fn
    return best_of(env, {"greedy_value": greedy(False), "greedy_ready_groups": greedy(True)})


def ro_fit(outcomes: list) -> np.ndarray:
    """Ridge-regularised logistic regression of adoption on (1, f1, f2) by Newton steps."""
    X = np.asarray([[1.0, f1, f2] for f1, f2, _ in outcomes])
    y = np.asarray([o for _, _, o in outcomes], dtype=float)
    w = np.zeros(3)
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) + 0.5 * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + 0.5 * np.eye(3)
        w -= np.linalg.solve(H, g)
    return w


def reference_rollout(view, env) -> None:
    """Explore across groups first, then expose by a fitted adoption score, avoiding groups whose exposures are
    mostly rejected while their adopter share is still low."""
    c = env._config
    G = c["group_names"]
    cap = int(c["capacity"])
    seen: dict = {}
    while not view.done:
        obs = view.observe()
        users = {e["id"]: e for e in obs["entities"] if e.get("kind") == "user"}
        share = [obs["numbers"].get(f"{g}_share", 0.0) for g in G]
        for i, e in users.items():
            if e["status"] in ("adopted", "rejected") and i in seen:
                seen[i]["outcome"] = 1 if e["status"] == "adopted" else 0
        status = {i: users[i]["status"] if i in users else "unexposed" for i in range(len(c["users"]))}
        fresh = [i for i in range(len(c["users"])) if status[i] == "unexposed" and i not in seen]
        done = [(c["users"][i]["f1"], c["users"][i]["f2"], s["outcome"]) for i, s in seen.items()
                if s.get("outcome") is not None]
        if view.t < 2 or len(done) < 6:
            per = {}
            pick = []
            for i in fresh:
                g = c["users"][i]["group"]
                if per.get(g, 0) < 1 and len(pick) < cap:
                    per[g] = per.get(g, 0) + 1
                    pick.append(i)
        else:
            w = ro_fit(done)
            rej = {g: [s["outcome"] for i, s in seen.items() if s.get("outcome") is not None
                       and c["users"][i]["group"] == g] for g in G}
            blocked = {g for k, g in enumerate(G) if len(rej[g]) >= 2 and np.mean(rej[g]) < 0.25 and share[k] < 0.2}
            sc = [(w[0] + w[1] * c["users"][i]["f1"] + w[2] * c["users"][i]["f2"], i) for i in fresh
                  if c["users"][i]["group"] not in blocked]
            sc.sort(reverse=True)
            pick = [i for s, i in sc[:cap] if s > -1.5]
        for i in pick:
            seen[i] = {"t": view.t}
        view.step(expose_ids(pick) if pick else None)


def identify_rollout(env) -> dict:
    """The direction in which the first feature moves adoption, from the outcomes of exposures across groups."""
    view = PublicView(env)
    c = env._config
    cap = int(c["capacity"])
    order = sorted(range(len(c["users"])), key=lambda i: (i % 7, i))
    exposed = []
    while not view.done and view.t < 6:
        obs = view.observe()
        st = {e["id"]: e["status"] for e in obs["entities"] if e.get("kind") == "user"}
        pick = [i for i in order if st.get(i, "unexposed") == "unexposed" and i not in exposed][:cap]
        exposed += pick
        view.step(expose_ids(pick))
    while not view.done and view.t < 14:
        view.step(None)
    st = {e["id"]: e["status"] for e in view.observe()["entities"] if e.get("kind") == "user"}
    done = [(c["users"][i]["f1"], c["users"][i]["f2"], 1 if st.get(i) == "adopted" else 0) for i in exposed
            if st.get(i) in ("adopted", "rejected")]
    w = ro_fit(done) if len(done) >= 4 else np.zeros(3)
    truth = float(np.sign(c["hidden"]["w"][0]))
    return {"question": "whether the first feature raises or lowers adoption", "estimate": float(np.sign(w[1])),
            "truth": truth, "ok": bool(np.sign(w[1]) == truth)}


def adoption_is_delayed(env) -> dict:
    """Exposed users of a ready group (its adopter share set above its readiness level, as the controlled
    condition) pay nothing in the tick they are exposed (same noise, with and without the exposures: revenue is no
    higher than without them) and are all pending; those who adopt do so ticks later, at different delays (four noise replicates of the world). Revenue
    is not compared beyond the exposure tick: unexposed users can adopt by word of mouth in the other copy."""
    from .env import Env
    c, h = env._config, env._config["hidden"]
    k = 0
    g = c["group_names"][k]
    idx = [i for i, u in enumerate(c["users"]) if u["group"] == g]
    want = int(np.ceil(h["tau"][k] * len(idx))) + 1
    same_all, pending_all, delays, exposed = True, True, [], 0
    for r in range(4):
        base = env if r == 0 else Env(env.family, env.seed, env._config, env.difficulty,
                                      world_seed=env.world_seed + 104729 * r)
        a, b = fork(base), fork(base)
        for e in (a, b):
            for j, i in enumerate(idx):
                e._state["status"][i] = 2 if j < want else 0
        ids = [i for i in idx[want:] if a._state["status"][i] == 0][:int(c["capacity"])]
        exposed += len(ids)
        ra = a.step(expose_ids(ids))
        rb = b.step(None)
        # no exposed user pays in the exposure tick (in the other copy some of them may adopt by word of mouth)
        same_all &= ra["observation"]["resources"]["revenue_cents_last_tick"] <= \
            rb["observation"]["resources"]["revenue_cents_last_tick"]
        pending_all &= all(a._state["status"][i] == 1 for i in ids)
        seen = set()
        for d in range(1, 13):
            a.step(None)
            for i in ids:
                if i not in seen and a._state["status"][i] == 2:
                    seen.add(i)
                    delays.append(d)
    return {"ok": bool(exposed and same_all and pending_all and len(delays) >= 2 and len(set(delays)) >= 2),
            "exposed": exposed, "adopted_later": len(delays), "adoption_delays": sorted(delays)}


def unready_groups_reject(env) -> dict:
    """The same users exposed on the same noise are rejected more often when their group's adopter share is set
    below its readiness level than when it is set above it."""
    c, h = env._config, env._config["hidden"]
    G = c["group_names"]
    rej = {"unready": 0, "ready": 0}
    n = 0
    for k, g in enumerate(G):
        idx = [i for i, u in enumerate(c["users"]) if u["group"] == g]
        for lab in ("unready", "ready"):
            e = fork(env)
            st = e._state["status"]
            want = int(np.ceil(h["tau"][k] * len(idx))) + 1 if lab == "ready" else 0
            for j, i in enumerate(idx):
                st[i] = 2 if j < want else 0
            test = idx[-int(c["capacity"]):]

            e.step(expose_ids(test))
            rej[lab] += sum(1 for i in test if e._state["fate"][i] == 3)
            if lab == "ready":
                n += len(test)
    return {"ok": bool(rej["unready"] > rej["ready"]), "rejections": rej, "exposed_per_arm": n}
