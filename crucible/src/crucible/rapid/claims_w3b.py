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


# ====================================================================== plan_commit (8.6.5)
def _open_sites(env) -> list[tuple[str, int]]:
    c, st = env._config, env._state
    return [(sl, j) for sl in c["slot_names"] for j in [env._engine.open_site(st, sl, env.t)] if j is not None]


def commitment_is_irreversible(env) -> dict:
    """Committing to a site uses up a commitment for good and moves its yield into income every period while it
    works (forks: commit versus not, same noise): the income difference in the next periods is the site's yield
    stream, and committing to the same site again is refused."""
    from .oracles import fork
    c = env._config
    e0 = fork(env)
    opened = _open_sites(e0)
    while not opened and not e0.done:
        e0.step(None)
        opened = _open_sites(e0)
    sl, j = opened[0]
    a, b = fork(e0), fork(e0)
    a.step({"kind": "discrete", "decision": "commit", "choice": sl})
    b.step(None)
    again = a.step({"kind": "discrete", "decision": "commit", "choice": sl})["result"]["ok"]
    b.step(None)
    inc = []
    for _ in range(4):
        ra, rb = a.step(None), b.step(None)
        inc.append(ra["observation"]["numbers"]["yield_income"] - rb["observation"]["numbers"]["yield_income"])
    y = float(c["yields"][j])
    working = all(x > 0 for x in inc)
    return {"ok": (not again) and (not working or abs(float(np.mean(inc)) - y) < 4 * float(c["yield_sd"])),
            "recommit_refused": not again, "income_difference": inc, "yield": y}


def fragile_type_fails_more(env) -> dict:
    """Sites of the type with the highest hidden hazard stop working sooner than sites of the type with the lowest:
    for one site of each of those two types, commit to it when it opens and count the periods it works (at most
    25) over 40 noise replicates."""
    from .env import Env
    c = env._config
    hz = c["hazard"]
    lo_t, hi_t = int(np.argmin(hz)), int(np.argmax(hz))
    surv = {}
    for ti in (lo_t, hi_t):
        j = next((j for j, sd in enumerate(c["sites"]) if sd["type_index"] == ti), None)
        if j is None:
            continue
        sd = c["sites"][j]
        runs = []
        for r in range(40):
            e = Env(env.family, env.seed, c, env.difficulty, world_seed=env.world_seed + 9173 * (r + 1))
            while e.t < sd["start"]:
                e.step(None)
            e.step({"kind": "discrete", "decision": "commit", "choice": sd["slot"]})
            k = 0
            while not e.done and k < 25 and e._state["active"][0][1]:
                e.step(None)
                k += int(bool(e._state["active"][0][1]))
            runs.append(k)
        surv[ti] = float(np.mean(runs))
    ok = len(surv) == 2 and surv[hi_t] < surv[lo_t]
    return {"ok": bool(ok), "mean_periods_working": {c["type_names"][t]: v for t, v in surv.items()},
            "hazard": hz}


def identify_fragile_type(env) -> dict:
    """Question: which site type fails fastest? Public: commit to the first open site of each type (the
    commitments allowed), then watch which commitments stop working first (the brief says which types exist)."""
    view = PublicView(env)
    c = env._config
    types = list(c["type_names"])                      # public: named in the brief
    chosen: dict = {}
    obs = view.observe()
    failed_at: dict = {}
    while not view.done:
        act = None
        for e in obs["entities"]:
            if e.get("kind") == "site" and e["type"] not in chosen and len(chosen) < int(c["capacity"]):
                chosen[e["type"]] = e["id"]
                act = {"kind": "discrete", "decision": "commit", "choice": e["slot"]}
                break
        for e in obs["entities"]:
            if e.get("kind") == "commitment" and not e["working"] and e["type"] not in failed_at:
                failed_at[e["type"]] = view.t
        obs = view.step(act)["observation"]
    est = min(chosen, key=lambda t: failed_at.get(t, 10 ** 6)) if chosen else None
    truth = types[int(np.argmax(c["hazard"]))]
    return {"question": "the site type that fails fastest", "estimate": est, "truth": truth,
            "ok": est == truth, "failed_at": failed_at}


def oracle_commit(env) -> dict:
    """Evaluator: commit to a site when its expected remaining value (yield x expected working periods left
    minus upkeep after failure minus the fixed cost) exceeds a reservation level, searched on forks."""
    from .oracles import best_of
    c = env._config
    F, H = float(c["fixed_cost"]), int(c["horizon"])

    def value(j, t):
        h = float(c["hazard"][c["sites"][j]["type_index"]])
        T = H - t - 1
        alive = sum((1 - h) ** k for k in range(1, T + 1))
        return float(c["yields"][j]) * alive - float(c["upkeep"]) * (T - alive) - F

    def pol(res):
        def fn(e):
            best = None
            for sl, j in _open_sites(e):
                v = value(j, e.t)
                if v > res * F and (best is None or v > best[0]):
                    best = (v, sl)
            if best is None or e._state["commits"] >= int(c["capacity"]):
                return None
            return {"kind": "discrete", "decision": "commit", "choice": best[1]}
        return fn
    return best_of(env, {f"knows yields and hazards, reservation {r}xF": pol(r) for r in (0.0, 0.25, 0.5, 1.0)})


def pilot_and_commit(view, env=None, reserve: float = 0.35, prior: tuple = (0.5, 20.0)) -> None:
    """Scripted public reference: pilot every new site whose prospectus is not hopeless; keep, per type, a count of
    periods worked and failures of its own commitments (a type's failure rate, starting from a mild prior); commit
    to a site when its piloted yield times the expected working periods left (given its type's estimated failure
    rate) minus the fixed cost beats a reservation level, which falls as the horizon runs out with commitments
    unused."""
    import re
    brief = " ".join(view.spec["brief"].values())
    F = float(re.search(r"costs ([0-9.]+), cannot be undone", brief).group(1))
    cap = int(re.search(r"at most ([0-9]+) commitments", brief).group(1))
    H = view.horizon
    piloted: dict = {}
    worked = {}
    fails = {}
    prev_working: dict = {}
    used = 0
    obs = view.observe()
    while not view.done:
        for e in obs["entities"]:
            if e.get("kind") == "commitment":
                ty = e["type"]
                if e["working"]:
                    worked[ty] = worked.get(ty, 0) + 1
                elif prev_working.get(e["id"], False):
                    fails[ty] = fails.get(ty, 0) + 1
                prev_working[e["id"]] = e["working"]
        sites = [e for e in obs["entities"] if e.get("kind") == "site"]
        steps = []
        best = None
        for e in sites:
            if e["id"] not in piloted:
                if e["prospectus"] > 0.02 * F and used < cap:
                    steps.append({"kind": "query", "target": "pilot", "args": {"slot": e["slot"]}})
                continue
            ty = e["type"]
            h = (fails.get(ty, 0) + prior[0]) / (worked.get(ty, 0) + fails.get(ty, 0) + prior[1])
            T = H - view.t - 1
            alive = sum((1 - h) ** k for k in range(1, T + 1))
            v = piloted[e["id"]] * alive - 20.0 * (T - alive) - F
            left = cap - used
            bar = reserve * F * min(1.0, (T / H) * 2.0) / max(1, left)
            if v > bar and (best is None or v > best[0]):
                best = (v, e["slot"])
        if best is not None and used < cap:
            steps.insert(0, {"kind": "discrete", "decision": "commit", "choice": best[1]})
        if not steps:
            obs = view.step(None)["observation"]
            continue
        r = view.step(steps[0] if len(steps) == 1 else {"kind": "sequence", "steps": steps})
        obs = r["observation"]
        res = r["result"].get("results")
        res = res if isinstance(res, list) else [res]
        for st, x in zip(steps, res):
            if isinstance(x, dict) and x.get("site") and "yield_estimate" in x:
                piloted[x["site"]] = x["yield_estimate"]
            if st["kind"] == "discrete" and r["result"]["ok"]:
                used += 1


# ====================================================================== net_balance (8.7.5)
def dispatch(demand, c_lin, c_quad, links, loss: float, gmax: float, short_cost: float, step: float = 0.5):
    """Economic dispatch on the grid by increments: repeatedly cover the next `step` of the largest remaining
    shortfall from the source (the node itself or a neighbour over a link with spare capacity) whose marginal
    delivered cost is lowest, while that cost is below the cost of the shortfall. Returns (generation, exports
    matrix)."""
    N = len(demand)
    g = np.zeros(N)
    E = np.zeros((N, N))
    capm = np.zeros((N, N))
    for a, b, cap in links:
        capm[a, b] = capm[b, a] = cap
    left = np.asarray(demand, dtype=float).copy()
    for _ in range(int(sum(demand) / step) + 3 * N):
        j = int(np.argmax(left))
        if left[j] <= 1e-9:
            break
        best = None
        for i in range(N):
            if g[i] + step > gmax + 1e-9:
                continue
            if i != j and E[i, j] + step > capm[i, j] + 1e-9:
                continue
            mc = c_lin[i] + c_quad[i] * (2 * g[i] + step)
            deliv = 1.0 if i == j else (1.0 - loss)
            per = mc / deliv
            if best is None or per < best[0]:
                best = (per, i, deliv)
        if best is None or best[0] > short_cost + 100.0:
            break
        _, i, deliv = best
        g[i] += step
        if i != j:
            E[i, j] += step
        left[j] -= step * deliv
    return g, E


def oracle_balance(env) -> dict:
    """Evaluator: a central dispatcher that sees every node's demand of the coming tick and every cost."""
    from .oracles import best_of
    c = env._config
    eng = env._engine
    ags = c["agents"]

    def pol(scale):
        def fn(e):
            dem = [eng.demand(e._state, i) * scale for i in range(len(ags))]
            g, E = dispatch(dem, c["c_lin"], c["c_quad"], c["links"], c["loss"], c["gen_max"], c["shortage_cost"])
            acts = {}
            for i, a in enumerate(ags):
                steps = [{"kind": "levers", "values": {c["gen_names"][i]: float(g[i])}}]
                nb = eng.neighbours(i)
                steps.append({"kind": "allocate", "amounts": {c["node_names"][j]: float(E[i, j]) for j in nb}})
                acts[a] = {"kind": "sequence", "steps": steps}
            return acts
        return fn
    return best_of(env, {f"central dispatch on the true demands (x{s})": pol(s) for s in (1.0, 1.05)})


def share_and_dispatch(view, env=None, margin: float = 0.85, persist: float = 0.8) -> None:
    """Scripted public reference, per participant: each operator reads its own demand and its own average cost at
    half and at full output (so its linear and quadratic cost terms), messages them to the other operators every
    tick, and every operator runs the same dispatch rule on the latest figures it has (its own demand now, the
    others' as of their last message; link capacities and losses from the brief), then sets its own generation and
    exports as the plan says."""
    import re
    ags = view.agents()
    N = len(ags)
    brief = " ".join(view.spec["brief"].values())
    cap = float(re.search(r"carry at most ([0-9.]+) units", brief).group(1))
    long_cap = float(re.search(r"units, ([0-9.]+) on the long link", brief).group(1))
    links = [[0, 1, cap], [1, 2, cap], [0, 2, long_cap]]         # public: the brief describes the grid
    loss, short = 0.08, 100.0                          # an unserved unit loses its price (the brief)
    know = {a: {} for a in ags}                       # what each operator knows: node -> (demand, c_lin, c_quad)
    hist = {a: {} for a in ags}
    obs = {a: view.observe(a) for a in ags}
    gmax = {a: float(view.action_space(a).kinds["levers"]["properties"]["values"]["properties"][
        next(iter(view.action_space(a).kinds["levers"]["properties"]["values"]["properties"]))]["maximum"])
        for a in ags}
    while not view.done:
        acts = {}
        for i, a in enumerate(ags):
            x = obs[a]["numbers"]
            for m in obs[a]["entities"]:
                if m.get("kind") == "message" and isinstance(m.get("content"), dict) and "node" in m["content"]:
                    k = m["content"]
                    for node, rec in (k.get("all") or {str(k["node"]): [k["demand"], k["c_lin"], k["c_quad"]]}).items():
                        nd = int(node)
                        if nd == i:
                            continue
                        hist[a].setdefault(nd, []).append(float(rec[0]))
                        # the other node's demand now: its last report pulled toward its average (demand is a
                        # persistent process; the brief says it fluctuates)
                        mu = float(np.mean(hist[a][nd]))
                        know[a][nd] = (mu + persist * (float(rec[0]) - mu), float(rec[1]), float(rec[2]))
            if "my_demand" in x:
                G = gmax[a]
                half, full = x["my_cost_at_half"], x["my_cost_at_full"]
                cq = (full - half) / (0.5 * G)
                cl = half - cq * 0.5 * G
                know[a][i] = (float(x["my_demand"]), float(cl), float(cq))
            if len(know[a]) < N:
                # until every node has reported, cover your own demand yourself
                d_own = know[a].get(i, (20.0, 0.0, 0.0))[0]
                acts[a] = {"kind": "sequence", "steps": [
                    {"kind": "levers", "values": {_lever_of(view, a): float(min(gmax[a], d_own * margin))}}] +
                    (_msg_all(ags, a, i, know[a][i]) if i in know[a] else [])}
                continue
            dem = [know[a][k][0] * margin for k in range(N)]
            cl = [know[a][k][1] for k in range(N)]
            cq = [know[a][k][2] for k in range(N)]
            g, E = dispatch(dem, cl, cq, links, loss, gmax[a], short)
            nb = sorted(set(range(N)) - {i})
            sp = view.action_space(a)
            names = list(sp.kinds["allocate"]["properties"]["amounts"]["properties"])
            node_of = {n: k for n, k in zip(names, [k for k in nb if any(
                (l[0] == i and l[1] == k) or (l[1] == i and l[0] == k) for l in links)])}
            steps = [{"kind": "levers", "values": {_lever_of(view, a): float(g[i])}},
                     {"kind": "allocate", "amounts": {n: float(E[i, k]) for n, k in node_of.items()}}] + \
                _msg_all(ags, a, i, know[a][i])
            acts[a] = {"kind": "sequence", "steps": steps}
        r = view.step(acts)
        obs = r["observations"]


def _lever_of(view, a) -> str:
    return next(iter(view.action_space(a).kinds["levers"]["properties"]["values"]["properties"]))


def _msg_all(ags, a, i, rec) -> list[dict]:
    # every tick, this operator's own figures to every other operator
    return [{"kind": "message", "to": b, "content": {"node": i, "demand": rec[0], "c_lin": rec[1], "c_quad": rec[2]}}
            for b in ags if b != a]


def transfer_relieves_shortfall(env) -> dict:
    """Power one node sends to a short neighbour arrives minus the line loss and reduces that neighbour's
    shortfall by exactly that much (forks with the same noise: the sender generates extra and sends it)."""
    from .oracles import fork
    c = env._config
    ags = c["agents"]
    a, b = fork(env), fork(env)
    base = {ag: {"kind": "levers", "values": {c["gen_names"][i]: 0.0}} for i, ag in enumerate(ags)}
    send = dict(base)
    send[ags[0]] = {"kind": "sequence", "steps": [{"kind": "levers", "values": {c["gen_names"][0]: 10.0}},
                                                   {"kind": "allocate", "amounts": {c["node_names"][1]: 10.0}}]}
    ra, rb = a.step(send), b.step(base)
    got = ra["observations"][ags[1]]["numbers"]["my_served"] - rb["observations"][ags[1]]["numbers"]["my_served"]
    want = 10.0 * (1.0 - float(c["loss"]))
    return {"ok": abs(got - want) < 1e-6, "extra_served_at_neighbour": got, "expected": want}


def costs_differ_between_nodes(env) -> dict:
    """The first unit of output costs very different amounts at different nodes (the cheapest node's marginal cost
    at low output is below half the dearest node's), measured from the cost the engine charges."""
    eng = env._engine
    per = [eng.gen_cost(i, 1.0) for i in range(eng.N)]
    return {"ok": min(per) < 0.5 * max(per), "cost_of_first_unit": per}


def identify_cheapest_node(env) -> dict:
    """Question: which node generates most cheaply? Public: every operator reports its own cost figure to the
    first operator by message; the first operator picks the lowest."""
    view = PublicView(env)
    ags = view.agents()
    obs = {a: view.observe(a) for a in ags}
    r = view.step(None)
    obs = r["observations"]
    acts = {a: {"kind": "message", "to": ags[0], "content": {"node": i, "cost": obs[a]["numbers"]["my_cost_at_half"]}}
            for i, a in enumerate(ags) if a != ags[0]}
    r = view.step(acts)
    costs = {0: r["observations"][ags[0]]["numbers"]["my_cost_at_half"]}
    for m in r["observations"][ags[0]]["entities"]:
        if m.get("kind") == "message" and isinstance(m.get("content"), dict):
            costs[int(m["content"]["node"])] = float(m["content"]["cost"])
    est = min(costs, key=costs.get) if len(costs) == len(ags) else None
    truth = int(np.argmin(env._config["c_lin"]))
    return {"question": "the node with the cheapest generation", "estimate": est, "truth": truth, "ok": est == truth}
