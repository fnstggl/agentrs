"""Identification procedures and intervention tests of the families redesigned in wave 3b (evaluator side).

Identification answers a family's experimental question from public actions and observations only (`PublicView`);
the truth comes from the hidden configuration. Nothing in a world imports this module.
"""
from __future__ import annotations

import numpy as np

from .identify import PublicView


def _lever(env, nid: str) -> str:
    cfg = env._config
    return cfg["nodes"][env._engine.idx[nid]]["lever"]["name"]


def identify_gate_threshold(env) -> dict:
    """Question: above which setting of the second lever does the first lever start to move sales? Public: with
    the first lever alternating between its ends every tick, sweep the second lever upward in steps and measure
    the sales difference between the high and low ticks at each step; the threshold is where that difference
    jumps."""
    c = env._config["claim"]
    A, B = _lever(env, c["lever"]), _lever(env, c["gate_lever"])
    view = PublicView(env)
    levels = np.linspace(0.0, 10.0, 21)
    diff = []
    for b in levels:
        hi_, lo_ = [], []
        for k in range(max(1, min(3, (view.horizon - 2) // (2 * len(levels))))):
            if view.done:
                break
            r_hi = view.step({"kind": "levers", "values": {A: 10.0, B: float(b)}})
            r_lo = view.step({"kind": "levers", "values": {A: 0.0, B: float(b)}})
            hi_.append(list(r_hi["observation"]["numbers"].values())[0])
            lo_.append(list(r_lo["observation"]["numbers"].values())[0])
        diff.append(float(np.mean(hi_) - np.mean(lo_)) if hi_ else 0.0)
    d = np.asarray(diff)
    k = int(np.argmax(np.diff(d)))
    est = float(0.5 * (levels[k] + levels[k + 1]))
    truth = float(c["theta"])
    return {"question": "the hidden threshold of the second lever that opens the pathway", "estimate": est,
            "truth": truth, "ok": abs(est - truth) <= 0.6}


# ====================================================================== des_bottleneck (shifting bottleneck)
def bottleneck_moves(env) -> dict:
    """After the hidden shift, an extra server at the new slow station completes more jobs than one at the station
    that was slow before (same noise draws in four replicates, the allocation set at the shift)."""
    from .claims import _replicates
    cfg = env._config
    names = cfg["station_names"]
    b0, b1 = int(np.argmin(cfg["mu"])), int(np.argmin(cfg["mu2"]))
    t0 = int(cfg["shift_at"])
    base = list(cfg["default_alloc"])
    donor = next(k for k in range(len(base)) if k not in (b0, b1) and base[k] > 1)

    def run(target):
        alloc = list(base)
        alloc[target] += 1
        alloc[donor] -= 1
        tot = []
        for e in _replicates(env, 4):
            for _ in range(t0):
                e.step(None)
            done0 = e._state["completed"]
            e.step({"kind": "allocate", "amounts": {n: float(x) for n, x in zip(names, alloc)}})
            while not e.done:
                e.step(None)
            tot.append(e._state["completed"] - done0)
        return float(np.mean(tot))
    new, old = run(b1), run(b0)
    return {"ok": new > 1.05 * old, "completed_after_shift_server_to_new_bottleneck": new,
            "completed_after_shift_server_to_old_bottleneck": old, "shift_at": t0}


def queue_tracker(view, env=None, every: int = 6) -> None:
    """Scripted public reference for a line whose slow station can change: every `every` ticks query each
    station's queue (paid) and reallocate the servers - one per station, the rest in proportion to the queues
    (a long queue marks the station that cannot keep up). The allocation starts even."""
    sp = view.action_space()
    stations = sorted(t.split(":", 1)[1] for t in sp.kinds["query"]["properties"]["target"]["enum"]
                      if t.startswith("station:"))
    S = len(stations)
    total = int(sp.limits.get("allocate_total", S))
    alloc = {n: total // S for n in stations}
    view.step({"kind": "allocate", "amounts": {n: float(x) for n, x in alloc.items()}})
    last = view.t
    while not view.done:
        if view.t - last < every:
            view.step(None)
            continue
        last = view.t
        q = {}
        for n in stations:
            if view.done:
                return
            r = view.step({"kind": "query", "target": f"station:{n}"})
            if r["result"]["ok"]:
                q[n] = float(r["result"]["results"]["queue"])
        if len(q) < S or view.done:
            continue
        w = {n: q[n] + 0.5 for n in stations}
        extra = total - S
        share = {n: extra * w[n] / sum(w.values()) for n in stations}
        new = {n: 1 + int(share[n]) for n in stations}
        for n in sorted(stations, key=lambda n: -(share[n] - int(share[n]))):
            if sum(new.values()) >= total:
                break
            new[n] += 1
        if new != alloc:
            alloc = new
            view.step({"kind": "allocate", "amounts": {n: float(x) for n, x in alloc.items()}})


def oracle_shifting_line(env) -> dict:
    """Evaluator: the best of the static line oracles and of privileged schedules that know when and where the
    bottleneck moves: servers in proportion to 1/mu (tilted) before the shift and to 1/mu2 from the shift on."""
    from .oracles import best_of, des_oracle, fork, run_fn
    c = env._config
    names = c["station_names"]
    R = int(c["servers"])
    t0 = int(c["shift_at"])

    def alloc_for(mu, tilt):
        mu = np.asarray(mu, dtype=float)
        w = (1.0 / mu) ** tilt
        a = np.maximum(1, np.floor(R * w / w.sum())).astype(int)
        while a.sum() < R:
            a[int(np.argmin(a * mu))] += 1
        while a.sum() > R:
            a[int(np.argmax(a * mu))] -= 1
        return {"kind": "allocate", "amounts": {n: float(x) for n, x in zip(names, a)}}

    pols = {}
    for tilt in (0.8, 1.0, 1.2):
        for lead in (0, 2):
            a0, a1 = alloc_for(c["mu"], tilt), alloc_for(c["mu2"], tilt)

            def fn(e, a0=a0, a1=a1, lead=lead):
                if e.t == 0:
                    return a0
                if e.t == max(1, t0 - lead):
                    return a1
                return None
            pols[f"knows the shift: 1/mu^{tilt}, switch {lead} ticks early"] = fn
    r = best_of(env, pols)
    s = des_oracle(env)
    return r if r["value"] >= s["value"] else s


# ====================================================================== pe_router (retooled lines)
def router_laws_swap(env) -> dict:
    """After the hidden retooling, the line fastest for the smallest jobs and the line fastest for the largest
    jobs have changed places (service-time laws at a queue of one, no noise)."""
    sc = env._engine.sc
    h = env._config["hidden"]
    M = len(env._config["line_names"])
    t1 = int(h["shift_at"])
    before = [int(np.argmin([sc.service_time(m, s, 1, 0.0, 0) for m in range(M)])) for s in (0.5, 4.0)]
    after = [int(np.argmin([sc.service_time(m, s, 1, 0.0, t1) for m in range(M)])) for s in (0.5, 4.0)]
    return {"ok": bool(before[0] != before[1] and after == [before[1], before[0]]), "fastest_before": before,
            "fastest_after": after, "shift_at": t1}


ROUTER_SHIFT_SRC = """
def svc(A, B, C, m, s, q):
    t = A[m] + B[m] * s ** C[m]
    if m == CONG and q > QSTAR:
        t *= 1.0 + CF * (q - QSTAR)
    return t


def act(obs, caller):
    A, B, C = (A1, B1, C1) if obs["tick"] < SHIFT else (A2, B2, C2)
    s = obs["size"]
    best, arg = None, 0
    for m in range(len(A)):
        q = len(obs["sizes"][m])
        tot = sum(svc(A, B, C, m, x, q) for x in obs["sizes"][m]) + svc(A, B, C, m, s, q + 1)
        if best is None or tot < best:
            best, arg = tot, m
    return {"line": arg}
"""


def oracle_router_shift(env) -> dict:
    """Evaluator: a router that knows every line's service law before and after the retooling and when it
    happens (it routes each job to the line where it would finish first), and the static law-aware router."""
    from .eval_program_exec import consts, deploy, oracle_router
    from .oracles import best_of
    h = env._config["hidden"]
    src = consts(A1=h["alpha"], B1=h["beta"], C1=h["gamma"], A2=h["alpha2"], B2=h["beta2"], C2=h["gamma2"],
                 SHIFT=int(h["shift_at"]), CONG=int(h["congested"]), QSTAR=int(h["qstar"]),
                 CF=float(h["cong"])) + ROUTER_SHIFT_SRC
    r = best_of(env, {"knows the laws and the retooling": deploy(src)})
    s = oracle_router(env)
    return r if r["value"] >= s["value"] else s


def router_refit(view, env, first: int = 5, every: int = 5, window: int = 10) -> None:
    """Scripted public reference: watch completions under the built-in split for a few ticks, fit each line's
    service time as a straight line in job size (least squares on the reported service steps), deploy a router
    that sends each job where it would finish first under the fitted laws, and refit and redeploy every few ticks
    from the completions of the last `window` ticks, so a change of the lines' laws is followed."""
    from .eval_program_exec import RT_SRC, consts
    c = env._config                                   # public: line names are in the brief and the schemas
    names = c["line_names"]
    seen: list = []                                   # (tick, line, size, service steps)

    def collect(r):
        for e in r["observation"]["entities"]:
            if e.get("kind") == "completed":
                seen.append((view.t, e["line"], e["size"], e["service_steps"]))

    def fit():
        al, be = [], []
        for n in names:
            d = np.asarray([(s, v) for t, ln, s, v in seen if ln == n and t > view.t - window],
                           dtype=float).reshape(-1, 2)
            if len(d) >= 3 and np.ptp(d[:, 0]) > 0.5:
                (a, b), *_ = np.linalg.lstsq(np.c_[np.ones(len(d)), d[:, 0]], d[:, 1], rcond=None)
            elif len(d):
                a, b = float(d[:, 1].mean()), 0.0
            else:
                a, b = 2.0, 1.0
            al.append(max(0.1, float(a)))
            be.append(max(0.0, float(b)))
        return al, be
    for _ in range(first):
        if view.done:
            return
        collect(view.step(None))
    while not view.done:
        al, be = fit()
        src = consts(A=al, B=be, C=[1.0] * len(al), CONG=-1, QSTAR=99, CF=0.0) + RT_SRC
        r = view.step({"kind": "program", "source": src, "ticks": min(every, view.horizon - view.t)})
        collect(r)
        for _ in range(every - 1):
            if view.done:
                return
            collect(view.step(None))
