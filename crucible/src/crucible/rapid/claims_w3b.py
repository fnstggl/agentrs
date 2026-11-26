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


# ====================================================================== srch_leads (8.5.1)
def _open_leads(env) -> list[tuple[str, dict]]:
    """(slot, lead with its hidden return under "ret") for every lead open now (evaluator side)."""
    c = env._config
    st = env._state
    out = []
    for i, sl in enumerate(c["slot_names"]):
        j = int(st["cur"][i])
        if j >= 0:
            out.append((sl, dict(c["leads"][j], ret=float(c["returns"][j]))))
    return out


def leads_pay_their_return(env) -> dict:
    """Investing in a lead changes net income by exactly its hidden return minus the price, against not investing
    on the same noise (forks), for every lead open now."""
    from .oracles import fork
    c = env._config
    diffs = []
    for sl, ld in _open_leads(env):
        a, b = fork(env), fork(env)
        a.step({"kind": "discrete", "decision": "invest", "choice": sl})
        b.step(None)
        diffs.append((float(a.outcome()["net"]) - float(b.outcome()["net"]), ld["ret"] - c["price"]))
    ok = bool(diffs) and all(abs(x - y) < 1e-6 for x, y in diffs)
    return {"ok": ok, "net_change_vs_return_minus_price": diffs}


def audit_more_precise(env) -> dict:
    """Repeated screens of one lead scatter far more around its hidden return than repeated audits do (each query
    is an independent draw)."""
    from .oracles import fork
    opened = _open_leads(env)
    if not opened:
        e = fork(env)
        while not opened and not e.done:
            e.step(None)
            opened = _open_leads(e)
        env = e
    sl, ld = opened[0]
    errs = {}
    for kind in ("screen", "audit"):
        e = fork(env)
        # twelve queries of the same lead within one tick (independent draws; the lead cannot expire meanwhile)
        r = e.step({"kind": "sequence", "steps": [{"kind": "query", "target": kind, "args": {"slot": sl}}] * 12})
        vals = [x["estimate"] - ld["ret"] for x in (r["result"].get("results") or [])
                if isinstance(x, dict) and x.get("estimate") is not None]
        errs[kind] = float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
    return {"ok": bool(errs["audit"] < 0.3 * errs["screen"]), "rms_error": errs}


def identify_good_leads(env) -> dict:
    """Question: which of the leads open now are worth more than their price? Public: audit each open lead."""
    view = PublicView(env)
    sp = view.action_space()
    slots = list(sp.limits["decisions"]["invest"])
    price = float(env._config["price"])                # public: stated in the brief
    est = {}
    for sl in slots:
        r = view.step({"kind": "query", "target": "audit", "args": {"slot": sl}})
        res = r["result"].get("results") or {}
        if res.get("lead") is not None:
            est[res["lead"]] = bool(res["estimate"] > price)
    truth = {ld["id"]: bool(r > price) for ld, r in zip(env._config["leads"], env._config["returns"])
             if ld["id"] in est}
    return {"question": "which open leads return more than their price", "estimate": est, "truth": truth,
            "ok": bool(est) and est == truth}


def oracle_leads(env) -> dict:
    """Evaluator: invest in every lead whose hidden return exceeds the price, as soon as it opens, without
    investigating."""
    from .oracles import best_of
    c = env._config

    def fn(e):
        steps = [{"kind": "discrete", "decision": "invest", "choice": sl}
                 for sl, ld in _open_leads(e) if ld["ret"] > c["price"]]
        if not steps:
            return None
        return steps[0] if len(steps) == 1 else {"kind": "sequence", "steps": steps}
    return best_of(env, {"invest in every lead that returns more than its price": fn})


def screen_audit_invest(view, env=None, screen_bar: float = 1.0, audit_bar: float = 1.05) -> None:
    """Scripted public reference: screen every lead once when it opens; audit a lead whose screen comes out above
    `screen_bar` x the price (or whose pitch is strong); invest when the audit is above `audit_bar` x the price.
    The price and the costs are stated in the brief; leads are told apart by the ids the slots show."""
    import re
    brief = " ".join(view.spec["brief"].values())
    m = re.search(r"costs ([0-9.]+) and pays", brief)
    price = float(m.group(1)) if m else 1000.0
    screened: dict = {}
    audited: dict = {}
    obs = view.observe()
    while not view.done:
        open_ = {e["slot"]: e for e in obs["entities"] if e.get("kind") == "lead"}
        steps = []
        for sl, e in open_.items():
            lid = e["id"]
            if lid in audited:
                if audited[lid] > audit_bar * price:
                    steps.append({"kind": "discrete", "decision": "invest", "choice": sl})
            elif lid in screened:
                if screened[lid] > screen_bar * price or (e["pitch"] > 0.8 and screened[lid] > 0.6 * price):
                    steps.append({"kind": "query", "target": "audit", "args": {"slot": sl}})
            else:
                steps.append({"kind": "query", "target": "screen", "args": {"slot": sl}})
        if not steps:
            obs = view.step(None)["observation"]
            continue
        r = view.step(steps[0] if len(steps) == 1 else {"kind": "sequence", "steps": steps})
        obs = r["observation"]
        res = r["result"].get("results")
        res = res if isinstance(res, list) else [res]
        for st, x in zip(steps, res):
            if not isinstance(x, dict) or x.get("lead") is None or "estimate" not in x:
                continue
            (audited if st.get("target") == "audit" else screened)[x["lead"]] = x["estimate"]


# ====================================================================== srch_ventures (8.5.4)
def ventures_return_after_delay(env) -> dict:
    """A unit funded in a sector comes back exactly after that sector's hidden delay (forks: fund one sector now
    versus nothing, on the same noise), as that sector's multiplier times the unit up to the noise."""
    from .oracles import fork
    c = env._config
    firsts = []
    for i, n in enumerate(c["sector_names"]):
        a, b = fork(env), fork(env)
        a.step({"kind": "allocate", "amounts": {n: 1.0}})
        b.step(None)
        f = None
        for k in range(1, 10):
            ra, rb = a.step(None), b.step(None)
            if ra["observation"]["numbers"][f"{n}.returned"] != rb["observation"]["numbers"][f"{n}.returned"]:
                f = k
                break
        firsts.append(f)
    want = [int(x) for x in c["delay"]]
    return {"ok": firsts == want, "first_return_after": firsts, "delays": want}


def venture_prospects_shift(env) -> dict:
    """The return per unit of at least one sector differs by more than the noise between two times (its hidden
    multiplier shifted): mean return of 200 noise draws of one unit funded at the start and at the last switch."""
    eng = env._engine
    c = env._config
    out = []
    for i, segs in enumerate(c["mult_sched"]):
        if len(segs) < 2:
            continue
        m0, m1 = eng.multiplier(i, 0), eng.multiplier(i, int(segs[-1][0]))
        out.append(abs(np.log(m1 / m0)))
    return {"ok": bool(out) and max(out) > 0.3, "largest_log_shift": max(out) if out else None,
            "sectors_with_shifts": len(out)}


def identify_best_sector(env) -> dict:
    """Question: which sector pays most per unit right now? Public: fund every sector equally for ten ticks and
    compare the multiples reported by the investments that have matured from the last five ticks of funding."""
    view = PublicView(env)
    sp = view.action_space()
    names = list(sp.kinds["allocate"]["properties"]["amounts"]["properties"])
    B = float(sp.limits["allocate_total"])
    mult = {n: [] for n in names}
    t_stop = 10
    while not view.done and view.t < t_stop + 6:
        act = {"kind": "allocate", "amounts": {n: B / len(names) for n in names}} if view.t < t_stop else None
        r = view.step(act)
        for e in r["observation"]["entities"]:
            if e.get("kind") == "matured" and e["funded_tick"] >= t_stop - 5 and e["units"] > 0:
                mult[e["sector"]].append(e["returned"] / (100.0 * e["units"]))
    est = max(names, key=lambda n: np.mean(mult[n]) if mult[n] else -1.0)
    eng = env._engine
    truth = env._config["sector_names"][int(np.argmax([eng.multiplier(i, t_stop - 3)
                                                     for i in range(len(env._config["sector_names"]))]))]
    return {"question": "the sector that pays most per unit now", "estimate": est, "truth": truth,
            "ok": est == truth}


def oracle_ventures(env) -> dict:
    """Evaluator: every tick, the whole budget into the sector whose hidden multiplier is highest, when it is above
    one (nothing otherwise)."""
    from .oracles import best_of
    c = env._config
    eng = env._engine

    def pol(share):
        def fn(e):
            m = [eng.multiplier(i, e.t) for i in range(len(c["sector_names"]))]
            i = int(np.argmax(m))
            if m[i] <= 1.0:
                return None
            return {"kind": "allocate", "amounts": {c["sector_names"][i]: float(c["budget"]) * share}}
        return fn
    return best_of(env, {f"knows the multipliers (budget x{s})": pol(s) for s in (1.0, 0.8)})


def venture_tracker(view, env=None, probe: float = 0.2, decay: float = 0.75, seed: int = 0) -> None:
    """Scripted public reference: keep, per sector, a discounted estimate of the multiple its recent investments
    returned (from the statement of matured investments, dated by when they were made) and how uncertain it is;
    every tick put a small probe into every sector (so changes are seen) and the rest of the budget into the
    sector with the highest sampled estimate (Thompson sampling) if it promises more than it costs."""
    rng = np.random.default_rng(seed)
    sp = view.action_space()
    names = list(sp.kinds["allocate"]["properties"]["amounts"]["properties"])
    B = float(sp.limits["allocate_total"])
    w = {n: 0.0 for n in names}
    s1 = {n: 0.0 for n in names}
    s2 = {n: 0.0 for n in names}
    obs = view.observe()
    while not view.done:
        for m in names:                                   # forget a little every tick
            w[m] *= decay
            s1[m] *= decay
            s2[m] *= decay
        for e in obs["entities"]:
            if e.get("kind") == "matured" and e["units"] > 0:
                n = e["sector"]
                x = e["returned"] / (100.0 * e["units"])
                w[n] += e["units"]
                s1[n] += e["units"] * x
                s2[n] += e["units"] * x * x
        draw = {}
        for n in names:
            if w[n] < 1e-6:
                draw[n] = rng.normal(1.0, 0.5)
                continue
            mu = s1[n] / w[n]
            var = max(s2[n] / w[n] - mu * mu, 0.01) / max(w[n], 0.5)
            draw[n] = rng.normal(mu, np.sqrt(var + 0.01))
        best = max(draw, key=draw.get)
        amounts = {n: probe for n in names}
        if draw[best] > 1.0:
            amounts[best] += max(0.0, B - probe * len(names))
        obs = view.step({"kind": "allocate", "amounts": amounts})["observation"]
