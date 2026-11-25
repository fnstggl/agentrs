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
    return "".join(f"{k} = {json.dumps(v)}\n" for k, v in kw.items())


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
    f = ri_fit(PublicView(env), c, scan=13, extra=1)
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
