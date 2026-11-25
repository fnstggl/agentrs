"""Evaluator side of the H2 families (planning and strategic engines): private oracles, scripted public-information
references, identification procedures and intervention tests (MASTER_SPEC Part 7.4, 7.5).

Nothing in a world imports this module (`admission.static_isolation` checks it in a fresh interpreter). Every
function here runs on a world or on copies of it:

* `oracle_<x>(env)` - the best of several privileged policies (they read hidden parameters and state: lead times,
  demand rates, recipes, rival types and their internal propensities, the opponent's next allocation, an attacker's
  route values, the draws of the coming period...), each evaluated on a fork of the world with its realised noise.
* `reference_<x>(view, env)` - a scripted participant that sees only a `PublicView` (actions, observations, tool
  schemas, the public spec) and uses only public terms of the world (names, prices and costs stated in the brief).
  In worlds with several participants it keeps each participant's knowledge separate: what one participant
  privately sees reaches another only through a `message`.
* `identify_<x>(env)` - answers the family's experimental question from public actions and observations; the truth
  is computed from the hidden structure.
* `claim_<x>(env)` - controlled interventions on forks (common random numbers) that verify the mechanism the family
  claims to have.
"""
from __future__ import annotations

import math

import numpy as np

from ..mechanisms.base import tick_rng
from .env import Env
from .identify import PublicView
from .oracles import best_of, fork, inside, run_fn


# ====================================================================== helpers
def replicate(env, r: int) -> Env:
    """The same world (same hidden configuration) with different noise: a practice episode."""
    return Env(env.family, env.seed, env._config, env.difficulty, world_seed=env.world_seed + 7907 * (r + 1))


def drive(view, fn) -> None:
    while not view.done:
        view.step(fn(view))


def nums(obs) -> dict:
    return obs["numbers"]


def _levers(sp) -> list[str]:
    return sorted(sp.kinds["levers"]["properties"]["values"]["properties"])


def _targets(sp) -> list[str]:
    return list(sp.kinds["allocate"]["properties"]["amounts"]["properties"])


def peek(env, n: int | None = None) -> np.random.Generator:
    """The generator the world's dynamics will use this period (evaluator only: lets an oracle see the period's
    draws before choosing, which bounds what any participant could achieve)."""
    return tick_rng(env.world_seed, env.env_id, env.t)


def greedy_alloc(gain, total: float, step: float, n: int) -> list[float]:
    """Allocate `total` in increments of `step` to maximise sum_k gain(k, x_k) (concave gains), stopping when no
    increment has a positive marginal gain."""
    x = [0.0] * n
    left = total
    while left >= step - 1e-9:
        best, kb = 0.0, -1
        for k in range(n):
            g = gain(k, x[k] + step) - gain(k, x[k])
            if g > best:
                best, kb = g, k
        if kb < 0:
            break
        x[kb] += step
        left -= step
    return x


def rollout(env, candidates, base, look: int | None = None, label: str = "rollout") -> dict:
    """One-step policy improvement on the realised noise: every period, each candidate action (from
    `candidates(e)`) is tried on a fork and followed by the base policy for `look` periods (to the horizon when
    None); the best first action is taken. Returns the value of the improved policy on a fork of `env`."""
    e = fork(env)
    while not e.done:
        cands = candidates(e)
        if len(cands) <= 1:
            e.step(cands[0] if cands else base(e))
            continue
        vals = []
        for a in cands:
            f = fork(e)
            f.step(a)
            run_fn(f, base, None if look is None else min(look, f.horizon - f.t))
            vals.append(float(f.outcome()["value"]))
        e.step(cands[int(np.argmax(vals))])
    return {"value": float(e.outcome()["value"]), "how": label}


def best_with(env, pols: dict, extra: list) -> dict:
    """The best of named privileged policies (on forks) and of extra precomputed results {"value", "how"}."""
    r = best_of(env, pols)
    for x in extra:
        if x["value"] > r["value"]:
            r = {"value": x["value"], "how": x["how"], "candidates": r.get("candidates")}
    return r


def _ok(est, truth) -> bool:
    return bool(est == truth)


# ====================================================================== 8.4.1 distribution
def _dist_act(lam, L, depot, on_order, stocks, z_d, z_s, cover, names, order_name, cap):
    tot = sum(lam)
    pos = depot + on_order + sum(stocks)
    target = tot * (L + 1 + cover) + z_d * math.sqrt(tot * (L + 1))
    q = int(max(0, round(target - pos)))
    ship = [max(0.0, round(lam[i] * cover + z_s * math.sqrt(lam[i]) - stocks[i])) for i in range(len(lam))]
    s = sum(ship)
    if s > cap:
        ship = [math.floor(x * cap / s) for x in ship]
    steps = [{"kind": "levers", "values": {order_name: float(min(q, 20 * len(lam)))}}]
    if sum(ship) > 0:
        steps.append({"kind": "allocate", "amounts": {n: float(x) for n, x in zip(names, ship)}})
    return {"kind": "sequence", "steps": steps}


def oracle_distribution(env) -> dict:
    eng, st, c = inside(env)
    names = c["site_names"]

    def pol(z_d, z_s, cover):
        def fn(e):
            s = e._state
            return _dist_act(c["lam"], c["lead"], s["depot"], sum(q for _, q in s["pipe"]), s["sites"], z_d, z_s,
                             cover, names, c["order_name"], c["ship_cap"])
        return fn
    return best_of(env, {f"base_stock z={zd},{zs} cover={cv}": pol(zd, zs, cv) for zd in (0.0, 1.0, 2.0)
                         for zs in (0.5, 1.5) for cv in (2.0, 2.5)})


def reference_distribution(view, env):
    """Public: learn each site's demand from censored sales (raised when a site ran out) and the lead time from the
    first delivery to the depot; then a base-stock rule for the depot and order-up-to shipments to sites."""
    sp = view.action_space()
    order = _levers(sp)[0]
    names = _targets(sp)
    cap = float(sp.limits["allocate_total"])
    est = {n: 3.0 for n in names}
    L, first = None, None
    obs = view.observe()
    while not view.done:
        x = nums(obs)
        if x:
            for n in names:
                s = x[f"{n}.sold"]
                d = s * 1.3 if x[f"{n}.stockout"] > 0 else s
                est[n] = 0.8 * est[n] + 0.2 * d
            if L is None and first is not None and x["depot_received"] > 0:
                L = view.t - 1 - first
        if first is None:
            first = view.t
        lam = [est[n] for n in names]
        stocks = [x.get(f"{n}.stock", 0.0) for n in names] if x else [0.0] * len(names)
        act = _dist_act(lam, L if L is not None else 3, x.get("depot_stock", 20.0) if x else 20.0,
                        x.get("on_order", 0.0) if x else 0.0, stocks, 1.0, 1.0, 2.0, names, order, cap)
        obs = view.step(act)["observation"]


def identify_distribution(env) -> dict:
    """Question: how many periods does a supplier order take to reach the depot?"""
    view = PublicView(env)
    order = _levers(view.action_space())[0]
    view.step({"kind": "levers", "values": {order: 10.0}})
    est = None
    for k in range(1, 12):
        r = view.step(None)
        if nums(r["observation"]).get("depot_received", 0) > 0:
            est = k
            break
    truth = int(env._config["lead"])
    return {"question": "replenishment lead time", "estimate": est, "truth": truth, "ok": _ok(est, truth)}


def claim_distribution(env) -> dict:
    """An order arrives at the depot exactly after the hidden lead time; shipping stock to the site with the
    highest hidden demand sells more than shipping it to the lowest (same noise)."""
    eng, st, c = inside(env)
    a, b = fork(env), fork(env)
    a.step({"kind": "levers", "values": {c["order_name"]: 15.0}})
    b.step(None)
    first = None
    for k in range(1, 10):
        ra, rb = a.step(None), b.step(None)
        if nums(ra["observation"])["depot_stock"] != nums(rb["observation"])["depot_stock"]:
            first = k
            break
    hi, lo = int(np.argmax(c["lam"])), int(np.argmin(c["lam"]))
    sold = {}
    for lab, i in (("high", hi), ("low", lo)):
        e = fork(env)
        e.step({"kind": "allocate", "amounts": {c["site_names"][i]: 20.0}})
        tot = 0.0
        for _ in range(12):
            tot += sum(v for k, v in nums(e.step(None)["observation"]).items() if k.endswith(".sold"))
        sold[lab] = tot
    return {"ok": first == int(c["lead"]) and sold["high"] > sold["low"], "lead": int(c["lead"]),
            "first_difference_at": first, "sold_shipping_to_high_demand": sold["high"],
            "sold_shipping_to_low_demand": sold["low"]}


# ====================================================================== 8.4.2 perishable
def _lam_t(c, t):
    return float(c["lam_bar"]) * (1.0 + float(c["amp"]) * math.sin(2 * math.pi * (t + float(c["phase"])) / 7.0))


def oracle_perishable(env) -> dict:
    eng, st, c = inside(env)
    S = int(c["shelf"])

    def pol(k, d, issue):
        def fn(e):
            s = e._state
            lam = _lam_t(c, e.t)
            fresh_need = lam * float(c["fresh_frac"])
            flex = lam * (1.0 - float(c["fresh_frac"])) + float(c["lam_bar"]) * float(c["bargain"]) * d
            usable_old = sum(s["lots"][1:S])
            q = max(0.0, fresh_need + max(0.0, flex - usable_old) + k * math.sqrt(lam))
            return {"kind": "sequence", "steps": [
                {"kind": "discrete", "decision": "issue", "choice": issue},
                {"kind": "levers", "values": {c["bake_name"]: float(min(round(q), c["max_bake"])),
                                              c["markdown_name"]: d}}]}
        return fn
    return best_of(env, {f"forecast k={k} markdown={d} {iss}": pol(k, d, iss) for k in (-0.5, 0.0, 0.7, 1.4)
                         for d in (0.0, 0.2, 0.4) for iss in ("fifo", "lifo")})


def reference_perishable(view, env):
    """Public: make last week's same-weekday sales (raised after a sell-out), sell older units first with a modest
    markdown."""
    sp = view.action_space()
    bake, mark = sp.kinds["levers"]["properties"]["values"]["properties"].keys()
    bake, mark = (bake, mark) if "default 0.0, rounded" in sp.kinds["levers"]["properties"]["values"]["properties"][bake][
        "description"] else (mark, bake)
    mx = sp.kinds["levers"]["properties"]["values"]["properties"][bake]["maximum"]
    hist: dict = {}
    last_q = 10.0
    view.step({"kind": "discrete", "decision": "issue", "choice": "fifo"})
    obs = view.observe()
    while not view.done:
        x = nums(obs)
        day = int(x.get("weekday", view.t % 7))
        if x:
            sold = x["sold_fresh"] + x["sold_older"]
            prev_day = (day - 1) % 7
            hist[prev_day] = sold * (1.25 if x["carried_over"] <= 0.5 else 1.0)
        q = hist.get(day, last_q)
        last_q = q
        obs = view.step({"kind": "levers", "values": {bake: float(min(mx, round(q))), mark: 0.2}})["observation"]


def identify_perishable(env) -> dict:
    """Question: for how many periods can a unit be sold before it must be thrown away?"""
    view = PublicView(env)
    sp = view.action_space()
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    bake = next(n for n, p in props.items() if "rounded" in p["description"])
    view.step({"kind": "levers", "values": {bake: float(props[bake]["maximum"])}})
    est = None
    r = None
    for k in range(0, 8):
        if k > 0:
            r = view.step(None)
        x = nums(r["observation"]) if r is not None else {}
        if x.get("discarded", 0) > 0:
            est = k + 1
            break
    if est is None and r is None:
        est = None
    # the batch was made in period 0; a first discard in period k means it could be sold in periods 0..k
    truth = int(env._config["shelf"])
    return {"question": "shelf life", "estimate": est, "truth": truth, "ok": _ok(est, truth)}


def claim_perishable(env) -> dict:
    """A batch made once is thrown away exactly at the end of its shelf life; with too much stock, selling the
    oldest first (fifo) wastes less than selling the newest first (lifo)."""
    eng, st, c = inside(env)
    a = fork(env)
    a.step({"kind": "levers", "values": {c["bake_name"]: float(c["max_bake"])}})
    first = 0 if a._state["lots"] == [0.0] * 6 and False else None
    disc = [nums(a.observe())["discarded"]]
    for _ in range(7):
        disc.append(nums(a.step(None)["observation"])["discarded"])
    first = next((k for k, v in enumerate(disc) if v > 0), None)
    waste = {}
    for issue in ("fifo", "lifo"):
        e = fork(env)
        e.step({"kind": "discrete", "decision": "issue", "choice": issue})
        w = 0.0
        for _ in range(21):
            r = e.step({"kind": "levers", "values": {c["bake_name"]: float(round(1.6 * float(c["lam_bar"])))}})
            w += nums(r["observation"])["discarded"]
        waste[issue] = w
    return {"ok": first == int(c["shelf"]) - 1 and waste["fifo"] < waste["lifo"], "shelf": int(c["shelf"]),
            "first_discard_period": first, "waste_fifo": waste["fifo"], "waste_lifo": waste["lifo"]}


# ====================================================================== 8.4.3 supply chain
def _sc_orders(K, inv, backlog, intransit, upstream_owed, mu_ahead, lags, z):
    out = []
    for k in range(K):
        L = lags[k]
        mu = mu_ahead(k)
        target = mu * (L + 1) + z * math.sqrt(max(mu, 1e-9) * (L + 1))
        pos = inv[k] + intransit[k] + (upstream_owed[k] if k < K - 1 else 0.0) - backlog[k]
        out.append(max(0.0, round(target - pos + (mu if k > 0 else 0.0) * 0.0)))
    return out


def oracle_supply_chain(env) -> dict:
    eng, st, c = inside(env)
    K = len(c["stage_names"])
    lags = [int(x) for x in c["lags"]]

    H = int(c["horizon"])
    from .engines.planning import pois
    demand = []                                     # the demand the world will draw, period by period
    for tt in range(H + 12):
        mu = float(c["mu0"]) if tt < int(c["t_change"]) else float(c["mu1"])
        demand.append(float(pois(float(tick_rng(env.world_seed, env.env_id, tt).random(1)[0]), mu)) if tt < H else mu)

    def pol(z, look):
        def fn(e):
            s = e._state
            t = e.t
            intr = [sum(q for _, kk, q in s["pipe"] if kk == k) for k in range(K)]
            owed = [s["backlog"][k + 1] for k in range(K - 1)] + [0.0]
            cum = [sum(lags[:k + 1]) for k in range(K)]

            def mu_ahead(k):
                if look == "exact":
                    w = demand[t:t + cum[k] + 1]
                    return float(np.mean(w)) if w else 0.0
                tt = t + (cum[k] if look else 0)
                return float(c["mu0"]) if tt < int(c["t_change"]) else float(c["mu1"])
            q = _sc_orders(K, s["inv"], s["backlog"], intr, owed, mu_ahead, lags, z)
            return {"kind": "levers", "values": {n: float(min(30.0, x)) for n, x in zip(c["order_names"], q)}}
        return fn
    return best_of(env, {f"echelon base stock z={z} lookahead={lk}": pol(z, lk) for z in (0.0, 0.5, 1.0, 1.5)
                         for lk in (False, True, "exact")})


def reference_supply_chain(view, env):
    """Public: per stage, track what is on order from its own orders and receipts, estimate the lag of each link
    from the first receipt after an order, forecast demand by exponential smoothing of customer orders, and order up
    to a lag-covering target."""
    sp = view.action_space()
    stages = sorted({k.split(".")[0] for k in nums(view.step(None)["observation"]) if "." in k},
                    key=lambda s: env._config["stage_names"].index(s))       # stage names (public, in the brief)
    orders = list(env._config["order_names"])                               # public lever names, in stage order
    K = len(stages)
    placed = [0.0] * K
    got = [0.0] * K
    lag = [2] * K
    first_order = [None] * K
    seen = [False] * K
    mu = 6.0
    obs = view.observe()
    while not view.done:
        x = nums(obs)
        mu = 0.7 * mu + 0.3 * x.get("customer_orders", mu)
        for k, s in enumerate(stages):
            rec = x.get(f"{s}.received", 0.0)
            got[k] += rec
            if rec > 0 and not seen[k] and first_order[k] is not None:
                seen[k] = True
                lag[k] = max(1, view.t - 1 - first_order[k])
        q = []
        for k, s in enumerate(stages):
            on_order = placed[k] - got[k]
            pos = x.get(f"{s}.on_hand", 0.0) + on_order - x.get(f"{s}.backlog", 0.0)
            target = mu * (lag[k] + 1) + 1.0 * math.sqrt(mu * (lag[k] + 1))
            v = float(min(30.0, max(0.0, round(target - pos))))
            q.append(v)
            placed[k] += v
            if v > 0 and first_order[k] is None:
                first_order[k] = view.t
        obs = view.step({"kind": "levers", "values": {n: v for n, v in zip(orders, q)}})["observation"]
    del sp


def identify_supply_chain(env) -> dict:
    """Question: how many periods does each link of the chain take (shipments into each stage; production at the
    top)? A pulse order at one stage at a time, with the stage above stocked, shows when it arrives."""
    c = env._config
    K = len(c["stage_names"])
    est = []
    for k in range(K):
        view = PublicView(fork(env))
        view.step({"kind": "levers", "values": {c["order_names"][k]: 5.0}})
        got = None
        for j in range(1, 8):
            x = nums(view.step(None)["observation"])
            base = x[f"{c['stage_names'][k]}.received"]
            if base >= 5.0 - 1e-9 and got is None:
                got = j
                break
        est.append(got)
    truth = [int(x) for x in c["lags"]]
    return {"question": "lag of each link", "estimate": est, "truth": truth, "ok": est == truth}


def claim_supply_chain(env) -> dict:
    """Each link delays shipments by exactly its hidden lag (pulse orders on forks with the same noise), and a stage
    whose upstream has run dry backlogs its orders."""
    c = env._config
    K = len(c["stage_names"])
    firsts = []
    for k in range(K):
        a, b = fork(env), fork(env)
        a.step({"kind": "levers", "values": {c["order_names"][k]: 5.0}})
        b.step(None)
        f = None
        for j in range(1, 8):
            xa, xb = nums(a.step(None)["observation"]), nums(b.step(None)["observation"])
            if xa[f"{c['stage_names'][k]}.received"] != xb[f"{c['stage_names'][k]}.received"]:
                f = j
                break
        firsts.append(f)
    e = fork(env)
    big = {c["order_names"][0]: 30.0}
    for _ in range(3):
        r = e.step({"kind": "levers", "values": big})
    backlog = nums(r["observation"])[f"{c['stage_names'][1]}.backlog"] if K > 1 else 1.0
    return {"ok": firsts == [int(x) for x in c["lags"]] and backlog > 0, "lags": [int(x) for x in c["lags"]],
            "first_arrival_differences": firsts, "upstream_backlog_after_large_orders": backlog}


# ====================================================================== 8.4.4 capacity
def oracle_capacity(env) -> dict:
    eng, st, c = inside(env)
    B = int(c["build_lag"])
    blk = float(c["unit_block"])

    def pol(margin, retire_slack, ahead):
        def fn(e):
            s = e._state
            t = e.t
            fut = eng.demand_path(min(int(c["horizon"]) - 1, t + B + ahead))
            have = s["cap"] + sum(x for _, x in s["pending"])
            gap = fut * (1.0 + margin) - have
            rem = int(c["horizon"]) - t - B
            if gap > 0 and rem > 12:
                return {"kind": "discrete", "decision": "build", "choice": "large" if gap >= 2.5 * blk else "small"}
            if retire_slack is not None:
                peak = max(eng.demand_path(tt) for tt in range(t, int(c["horizon"])))
                if s["cap"] - blk >= peak * (1.0 + retire_slack) and not s["pending"]:
                    return {"kind": "discrete", "decision": "build", "choice": "retire"}
            return None
        return fn
    return best_of(env, {f"plan margin={m} retire={r} ahead={a}": pol(m, r, a) for m in (-0.1, 0.0, 0.15)
                         for r in (None, 0.05, 0.25) for a in (0, 6)})


def reference_capacity(view, env):
    """Public: demand = sales + turned away; fit a linear trend to the last 10 periods, learn the build lag from the
    first build, build a block when the trend forecast at the lag exceeds capacity under construction, retire a
    block when demand has stayed well below capacity and is falling."""
    blk = float(env._config["unit_block"])                    # public (stated in the brief)
    dem: list = []
    lag, first, seen = 3, None, False
    low_run = 0
    obs = view.observe()
    while not view.done:
        x = nums(obs)
        act = None
        if x:
            dem.append(x["sales"] + x["turned_away"])
            if first is not None and not seen and x["capacity_added"] > 0:
                lag, seen = view.t - 1 - first, True
            n = min(10, len(dem))
            ys = np.asarray(dem[-n:])
            slope = float(np.polyfit(np.arange(n), ys, 1)[0]) if n >= 4 else 0.0
            fc = float(ys.mean()) + slope * ((n - 1) / 2.0 + lag + 3)
            have = x["capacity"] + x["under_construction"]
            rem = view.horizon - view.t - lag
            low_run = low_run + 1 if float(ys.mean()) < x["capacity"] - 1.5 * blk and slope < 0 else 0
            if n >= 4 and fc > have + 0.5 * blk and rem > 12:
                act = {"kind": "discrete", "decision": "build", "choice": "large" if fc - have > 2.5 * blk else "small"}
                if first is None:
                    first = view.t
            elif low_run >= 4:
                act = {"kind": "discrete", "decision": "build", "choice": "retire"}
                low_run = 0
        obs = view.step(act)["observation"]


def identify_capacity(env) -> dict:
    """Question: which demand scenario is this world in - steady growth (boom), decline (bust) or a late jump?"""
    view = PublicView(env)
    dem = []
    while not view.done:
        x = nums(view.step(None)["observation"])
        dem.append(x["sales"] + x["turned_away"])
    d = np.asarray(dem)
    n = len(d)
    k = max(2, n // 10)
    jumps = np.asarray([d[i:i + k].mean() - d[max(0, i - k):i].mean() for i in range(k, n - k)])
    lin = np.polyfit(np.arange(n), d, 1)
    resid_lin = float(((d - np.polyval(lin, np.arange(n))) ** 2).mean())
    j = int(np.argmax(jumps)) + k
    step = np.r_[np.full(j, d[:j].mean()), np.full(n - j, d[j:].mean())]
    resid_step = float(((d - step) ** 2).mean())
    if lin[0] < 0:
        est = "bust"
    elif resid_step < 0.8 * resid_lin and d[j:].mean() > 1.5 * d[:j].mean():
        est = "late"
    else:
        est = "boom"
    truth = env._config["path"]
    return {"question": "demand scenario", "estimate": est, "truth": truth, "ok": est == truth}


def claim_capacity(env) -> dict:
    """A block ordered now comes on line exactly after the hidden build lag and from then on costs upkeep whether it
    is used or not; retired capacity is gone at once."""
    eng, st, c = inside(env)
    a, b = fork(env), fork(env)
    a.step({"kind": "discrete", "decision": "build", "choice": "small"})
    b.step(None)
    first = None
    for k in range(1, 8):
        xa, xb = nums(a.step(None)["observation"]), nums(b.step(None)["observation"])
        if xa["capacity"] != xb["capacity"]:
            first = k
            break
    ca, cb = a._ledger["cost"]["a0"], b._ledger["cost"]["a0"]
    r = fork(env)
    cap0 = r._state["cap"]
    r.step({"kind": "discrete", "decision": "build", "choice": "retire"})
    return {"ok": first == int(c["build_lag"]) and ca > cb and r._state["cap"] == cap0 - float(c["unit_block"]),
            "build_lag": int(c["build_lag"]), "capacity_first_differs_after": first, "cost_with_block": ca,
            "cost_without": cb}


# ====================================================================== 8.4.5 complements
def _comp_plan(c, A, dem_est, fg, res, pipe_by_res, cover, fgt):
    R, P = len(c["res_names"]), len(c["prod_names"])
    hours = float(c["hours"])
    unit_cost = [sum(A[i][j] * float(c["res_cost"][i]) for i in range(R)) for j in range(P)]
    margin = [float(c["price"][j]) - unit_cost[j] for j in range(P)]
    want = [max(0.0, dem_est[j] * fgt - fg[j] + dem_est[j]) if margin[j] > 0 else 0.0 for j in range(P)]
    order_j = sorted(range(P), key=lambda j: -margin[j])
    plan = [0.0] * P
    left = hours
    for j in order_j:
        h = min(left, math.ceil(want[j]))
        plan[j] = float(h)
        left -= h
    rate = [dem_est[j] if margin[j] > 0 else 0.0 for j in range(P)]
    orders = {}
    for i in range(R):
        use = sum(A[i][j] * rate[j] for j in range(P))
        L = int(c["lead_time"][i])
        target = use * (L + 1 + cover)
        orders[c["order_names"][i]] = float(min(40.0, max(0.0, round(target - res[i] - pipe_by_res[i]))))
    return plan, orders


def oracle_complements(env) -> dict:
    eng, st, c = inside(env)
    A = c["recipe"]
    R = len(c["res_names"])

    def pol(cover, f, fgt):
        def fn(e):
            s = e._state
            pipe = [sum(q for _, ii, q in s["pipe"] if ii == i) for i in range(R)]
            plan, orders = _comp_plan(c, A, [f * x for x in c["dem"]], s["fg"], s["res"], pipe, cover, fgt)
            return {"kind": "sequence", "steps": [{"kind": "levers", "values": orders},
                                                  {"kind": "allocate", "amounts": dict(zip(c["prod_names"], plan))}]}
        return fn
    return best_of(env, {f"recipe plan cover={cv} demand x{f} buffer={b}": pol(cv, f, b) for cv in (0.5, 1.5)
                         for f in (0.9, 1.1) for b in (0.5, 1.5)})


def reference_complements(view, env):
    """Public: make two units of each product alone in turn and read the inputs used (the recipe column); then plan
    production from smoothed sales, order inputs for it over the stated lead times, and keep a small buffer."""
    c = env._config                                   # public terms: names, prices, input costs, lead times, hours
    R, P = len(c["res_names"]), len(c["prod_names"])
    A = [[0.0] * P for _ in range(R)]
    obs = view.observe()
    for j, pn in enumerate(c["prod_names"]):
        r = view.step({"kind": "allocate", "amounts": {pn: 2.0}})
        x = nums(r["observation"])
        made = x[f"{pn}.made"]
        if made > 0:
            for i, rn in enumerate(c["res_names"]):
                A[i][j] = x[f"{rn}.used"] / made
        obs = r["observation"]
    dem = [3.0] * P
    while not view.done:
        x = nums(obs)
        for j, pn in enumerate(c["prod_names"]):
            s = x.get(f"{pn}.sold", 0.0)
            short = x.get(f"{pn}.in_store", 1.0) <= 0.0
            dem[j] = 0.8 * dem[j] + 0.2 * (s * 1.3 if short else s)
        res = [x.get(f"{rn}.stock", 0.0) for rn in c["res_names"]]
        pipe = [x.get(f"{rn}.on_order", 0.0) for rn in c["res_names"]]
        fg = [x.get(f"{pn}.in_store", 0.0) for pn in c["prod_names"]]
        plan, orders = _comp_plan(c, A, dem, fg, res, pipe, 1.0, 1.0)
        obs = view.step({"kind": "sequence", "steps": [{"kind": "levers", "values": orders},
                                                       {"kind": "allocate",
                                                        "amounts": dict(zip(c["prod_names"], plan))}]})["observation"]


def identify_complements(env) -> dict:
    """Question: which inputs does each product need?"""
    view = PublicView(env)
    c = env._config
    est = []
    for pn in c["prod_names"]:
        x = nums(view.step({"kind": "allocate", "amounts": {pn: 2.0}})["observation"])
        est.append([1 if x[f"{rn}.used"] > 0 else 0 for rn in c["res_names"]])
    est_T = [list(r) for r in zip(*est)]
    truth = [[1 if v > 0 else 0 for v in row] for row in c["recipe"]]
    return {"question": "recipe support (inputs per product)", "estimate": est_T, "truth": truth, "ok": est_T == truth}


def claim_complements(env) -> dict:
    """Complementarity: withholding one input stops exactly the products that need it, however much of the other
    inputs there is."""
    eng, st, c = inside(env)
    A = c["recipe"]
    R, P = len(c["res_names"]), len(c["prod_names"])
    ok = True
    rows = []
    for i in range(R):
        e = fork(env)
        e._state["res"] = [0.0 if k == i else 50.0 for k in range(R)]
        x = nums(e.step({"kind": "allocate", "amounts": {pn: float(c["hours"]) / P for pn in c["prod_names"]}})
                 ["observation"])
        made = [x[f"{pn}.made"] for pn in c["prod_names"]]
        need = [A[i][j] > 0 for j in range(P)]
        good = all((m == 0) if nd else (m > 0) for m, nd in zip(made, need))
        ok = ok and good
        rows.append({"withheld": i, "made": made, "needs": need})
    return {"ok": bool(ok), "interventions": rows}


# ====================================================================== 8.4.6 portfolio
def oracle_portfolio(env) -> dict:
    eng, st, c = inside(env)
    N = len(c["proj_names"])

    def pol(k, deadline_first, spread=False):
        def fn(e):
            s = e._state
            act = [j for j in range(N) if s["status"][j] == 0 and not c["dud"][j]]

            def pays(j):                                   # would finishing it with all staff still pay?
                rem = max(1e-6, float(c["work"][j]) - s["work"][j])
                T = rem / (float(c["prod"][j]) * float(c["staff"]) ** float(c["beta"]))
                late = max(0.0, e.t + T - int(c["deadline"][j]))
                return float(c["value"][j]) * max(0.0, 1.0 - late / float(c["late_window"])) > \
                    T * float(c["staff"]) * float(c["wage"])
            act = [j for j in act if pays(j)]
            if not act:
                return {"kind": "allocate", "amounts": {}}

            def score(j):
                rem = max(1e-6, float(c["work"][j]) - s["work"][j])
                return float(c["value"][j]) * float(c["prod"][j]) / rem
            act.sort(key=(lambda j: (c["deadline"][j], -score(j))) if deadline_first else (lambda j: -score(j)))
            top = act[:k]
            if spread:                                     # staff in proportion to remaining work per productivity
                need = [max(1e-6, float(c["work"][j]) - s["work"][j]) / float(c["prod"][j]) for j in top]
                return {"kind": "allocate", "amounts": {c["proj_names"][j]: float(c["staff"]) * n / sum(need)
                                                        for j, n in zip(top, need)}}
            each = float(c["staff"]) / len(top)
            return {"kind": "allocate", "amounts": {c["proj_names"][j]: each for j in top}}
        return fn
    pols = {f"focus top {k}{' by deadline' if df else ''}": pol(k, df) for k in range(1, N + 1) for df in (False, True)}
    pols.update({f"spread over top {k} by need": pol(k, False, True) for k in range(2, N + 1)})
    return best_of(env, pols)


def reference_portfolio(view, env):
    """Public: staff every project evenly for three periods, audit them all (exact work done) and read the progress
    reports to estimate each project's work content and its team's productivity; then put all staff on the project
    with the best estimated pay per period (if it still pays after wages), re-estimating work content from the
    reports as it goes; abandoned projects are dropped."""
    sp = view.action_space()
    names = _targets(sp)
    S = float(sp.limits["allocate_total"])
    c = env._config                     # public: values, deadlines, wage and the lateness window are stated
    value = dict(zip(c["proj_names"], c["value"]))
    beta = 0.65                         # assumed returns to staffing (not known)
    N = len(names)
    view.step({"kind": "allocate", "amounts": {n: S / N for n in names}})
    view.step(None)
    view.step(None)
    r = view.step({"kind": "sequence", "steps": [{"kind": "query", "target": "audit", "args": {"project": n}}
                                                 for n in names]})
    x = nums(r["observation"])
    work = {d["project"]: d["work_done"] for d in r["result"]["results"]} if r["result"]["ok"] else {n: 1.0 for n in names}
    rate = {n: max(1e-3, work[n] / 4.0) / (S / N) ** beta for n in names}
    W = {n: work[n] / max(x[f"{n}.reported_done"], 0.03) for n in names}
    done = dict(work)
    obs = r["observation"]
    while not view.done:
        x = nums(obs)
        best, bn = 0.0, None
        for n in names:
            if x[f"{n}.status"] != 0:
                continue
            f = x[f"{n}.reported_done"]
            if f > 0.1:
                W[n] = 0.7 * W[n] + 0.3 * done[n] / f
            T = max(0.0, W[n] - done[n]) / (rate[n] * S ** beta)
            late = max(0.0, T - x[f"{n}.periods_to_deadline"])
            pay = value[n] * max(0.0, 1.0 - late / 15.0) - T * S * float(c["wage"])
            if pay > 0 and pay / max(T, 1.0) > best:
                best, bn = pay / max(T, 1.0), n
        if bn is None:
            obs = view.step({"kind": "allocate", "amounts": {}})["observation"]
            continue
        done[bn] += rate[bn] * S ** beta
        obs = view.step({"kind": "allocate", "amounts": {bn: S}})["observation"]


def identify_portfolio(env) -> dict:
    """Question: which project's team does the most work per period at equal staffing? Equal staffing for four
    periods, then an exact audit of each project."""
    view = PublicView(env)
    sp = view.action_space()
    names = _targets(sp)
    total = float(sp.limits["allocate_total"])
    view.step({"kind": "allocate", "amounts": {n: total / len(names) for n in names}})
    for _ in range(3):
        view.step(None)
    r = view.step({"kind": "sequence", "steps": [{"kind": "query", "target": "audit", "args": {"project": n}}
                                                 for n in names]})
    done = {x["project"]: x["work_done"] for x in (r["result"]["results"] or [])} if r["result"]["ok"] else {}
    if not done:
        return {"question": "most productive project team", "estimate": None, "ok": False}
    est = max(done, key=done.get)
    truth = env._config["proj_names"][int(np.argmax(env._config["prod"]))]
    return {"question": "most productive project team", "estimate": est, "truth": truth, "ok": est == truth}


def claim_portfolio(env) -> dict:
    """Diminishing returns: doubling a project's staff less than doubles its work per period; a dud project is
    abandoned at its midpoint and never pays."""
    eng, st, c = inside(env)
    j = 0
    n = c["proj_names"][j]
    w = {}
    for x in (2.0, 4.0):
        e = fork(env)
        e.step({"kind": "allocate", "amounts": {n: x}})
        w[x] = e._state["work"][j]
    d = int(np.argmax(c["dud"]))
    e = fork(env)
    e.step({"kind": "allocate", "amounts": {c["proj_names"][d]: float(c["staff"])}})
    while not e.done and e._state["status"][d] == 0:
        e.step(None)
    return {"ok": w[4.0] < 2 * w[2.0] and w[4.0] > w[2.0] and e._state["status"][d] == 2 and e._state["paid"][d] == 0,
            "work_2_staff": w[2.0], "work_4_staff": w[4.0], "dud_status": int(e._state["status"][d])}


# ====================================================================== 8.9.1 assembly
def _asm_split(cost):
    """Which participant should make each component: (maker of component 0, maker of component 1)."""
    a = 0 if cost[0][0] - cost[0][1] < cost[1][0] - cost[1][1] else 1
    return a, 1 - a


def oracle_assembly(env) -> dict:
    eng, st, c = inside(env)
    mx, my = _asm_split(c["unit_cost"])
    cap = float(c["cap"])

    def pol(cover, scale):
        def fn(e):
            s = e._state
            t = e.t
            mu = float(c["mu_mean"]) * (1.0 + float(c["mu_amp"]) * math.sin(2 * math.pi * (t + float(c["mu_phase"]))
                                                                              / float(c["mu_period"])))
            tgt = mu * scale + cover * math.sqrt(mu)
            rx = min(cap, max(0.0, round(tgt - s["buf"][0] + mu * 0.0)))
            ry = min(cap, max(0.0, round(tgt - s["buf"][1])))
            acts = {}
            for i, a in enumerate(e.agents()):
                v = [rx if i == mx else 0.0, ry if i == my else 0.0]
                acts[a] = {"kind": "levers", "values": dict(zip(c["make_names"], v))}
            return acts
        return fn
    return best_of(env, {f"specialise cover={cv} x{sc}": pol(cv, sc) for cv in (0.0, 0.5, 1.0) for sc in (1.0, 1.2)})


def reference_assembly(view, env):
    """Public, per participant: each tells the other its own unit costs (message), then the one with the cheaper
    relative cost of a component makes it, at a rate that tracks smoothed orders and the component buffers."""
    ags = view.agents()
    sp = view.action_space(ags[0])
    makes = _levers(sp)
    mk = {m: m for m in makes}
    obs = view.step({a: None for a in ags} if False else None)
    obs = obs["observations"]
    my_cost = {}
    for a in ags:
        res = obs[a]["resources"]
        keys = sorted(k for k in res if k.startswith("unit_cost_"))
        my_cost[a] = [res[k] for k in keys]
        comp = [k[len("unit_cost_"):] for k in keys]
    r = view.step({a: {"kind": "message", "to": [b for b in ags if b != a][0], "content": {"unit_costs": my_cost[a]}}
                   for a in ags})
    obs = r["observations"]
    partner = {}
    for a in ags:
        for m in obs[a]["entities"]:
            if m.get("kind") == "message" and (m.get("content") or {}).get("unit_costs"):
                partner[a] = m["content"]["unit_costs"]
    plan = {}
    for a in ags:
        mine, theirs = my_cost[a], partner.get(a, my_cost[a][::-1])
        rel_mine = mine[0] - mine[1]
        rel_theirs = theirs[0] - theirs[1]
        plan[a] = 0 if rel_mine < rel_theirs else 1         # the component this participant makes
    ema = {a: 5.0 for a in ags}
    while not view.done:
        acts = {}
        for a in ags:
            x = nums(obs[a])
            ema[a] = 0.7 * ema[a] + 0.3 * x.get("orders", ema[a])
            k = plan[a]
            buf = x.get(f"{comp[k]}.buffer", 0.0)
            rate = max(0.0, round(ema[a] + 0.5 * (ema[a] - buf)))
            v = [0.0, 0.0]
            v[k] = float(min(sp.kinds["levers"]["properties"]["values"]["properties"][makes[k]]["maximum"], rate))
            acts[a] = {"kind": "levers", "values": {f"make_{comp[j]}": v[j] for j in range(2)}}
        obs = view.step(acts)["observations"]
    del mk


def identify_assembly(env) -> dict:
    """Question: which participant should make the first component (comparative advantage)? Each reads its own
    costs and messages them; the answer is computed from the two messages."""
    view = PublicView(env)
    ags = view.agents()
    obs = view.step(None)["observations"]
    costs = {}
    for a in ags:
        res = obs[a]["resources"]
        costs[a] = [res[k] for k in sorted(k for k in res if k.startswith("unit_cost_"))]
    r = view.step({a: {"kind": "message", "to": [b for b in ags if b != a][0], "content": {"c": costs[a]}} for a in ags})
    got = {}
    for a in ags:
        for m in r["observations"][a]["entities"]:
            if m.get("kind") == "message":
                got[m["from"]] = m["content"]["c"]
    rel = {a: got[a][0] - got[a][1] for a in got}
    est = min(rel, key=rel.get) if len(rel) == 2 else None
    names = sorted(env._config["comp_names"])
    ci = env._config["comp_names"].index(names[0])
    mx, my = _asm_split(env._config["unit_cost"])
    truth = ags[mx] if ci == 0 else ags[my]
    return {"question": "who should make component " + names[0], "estimate": est, "truth": truth, "ok": est == truth}


def claim_assembly(env) -> dict:
    """Complementary capabilities: with the same total capacity, specialisation makes products more cheaply than
    each participant making both components; one participant alone can at most make half its capacity of
    products; components without their complement never sell."""
    eng, st, c = inside(env)
    mx, my = _asm_split(c["unit_cost"])
    cap = float(c["cap"])
    ags = env.agents()

    def run(rates, n=12):
        e = fork(env)
        e.step({a: {"kind": "levers", "values": dict(zip(c["make_names"], rates[i]))} for i, a in enumerate(ags)})
        for _ in range(n - 1):
            e.step(None)
        return e
    spec = [[0.0, 0.0], [0.0, 0.0]]
    spec[mx][0] = cap / 2
    spec[my][1] = cap / 2
    both = [[cap / 4, cap / 4], [cap / 4, cap / 4]]
    es, eb = run(spec), run(both)
    cost_s = sum(es._ledger["cost"].values())
    cost_b = sum(eb._ledger["cost"].values())
    sold_s = sum(es._ledger["revenue"].values())
    sold_b = sum(eb._ledger["revenue"].values())
    alone = run([[cap / 2, cap / 2], [0.0, 0.0]], 6)
    one = run([[cap, 0.0], [0.0, 0.0]], 6)
    return {"ok": abs(sold_s - sold_b) < 1e-6 and cost_s < cost_b and one._ledger["revenue"]["a0"] == 0.0
            and alone._state["out"] <= 2 * (cap / 2) * 6 + 1e-9,
            "cost_specialised": cost_s, "cost_both": cost_b, "revenue_one_component_only": one._ledger["revenue"]["a0"]}


# ====================================================================== 8.9.4 schedule
def _sched_assign(jobs_of, cur_of, rem_of, speed, t, rule, nM):
    """A machine assignment: list over machines of participant index or None."""
    cand = [i for i in range(len(jobs_of)) if cur_of[i] >= 0]
    out: list = [None] * nM
    used = set()
    pairs = []
    for i in cand:
        jb = jobs_of[i][cur_of[i]]
        for m in range(nM):
            rem = max(1e-6, rem_of[i]) / max(1e-6, speed[i][m])
            slack = jb["deadline"] - t - rem
            pay = jb["value"] * max(0.0, 1.0 - max(0.0, t + rem - 1 - jb["deadline"]) / 8.0)
            if pay <= 10.0 * rem:                        # would not pay for its machine time (public terms)
                continue
            if rule == "pay":
                sc = (pay - 10.0 * rem) / rem
            elif rule == "density":
                sc = jb["value"] / rem
            elif rule == "urgency":
                sc = jb["value"] / rem + 400.0 / (1.0 + max(0.0, slack))
            else:
                sc = -slack
            pairs.append((sc, i, m))
    for sc, i, m in sorted(pairs, reverse=True):
        if i in used or out[m] is not None:
            continue
        out[m] = i
        used.add(i)
    return out


def _sched_plan_action(ags, machines, assign):
    plan = {m: (ags[i] if i is not None else "none") for m, i in zip(machines, assign)}
    return {a: {"kind": "coordinate", "with": [b for b in ags if b != a], "plan": plan} for a in ags}


def oracle_schedule(env) -> dict:
    eng, st, c = inside(env)
    machines = c["machine_names"]

    def pol(rule):
        def fn(e):
            s = e._state
            cur = [eng.current_job(s, i, e.t) for i in range(3)]
            rem = [float(c["true_time"][i][cur[i]]) - s["work"][i][cur[i]] if cur[i] >= 0 else 0.0 for i in range(3)]
            assign = _sched_assign(c["jobs"], cur, rem, c["speed"], e.t, rule, len(machines))
            if all(x is None for x in assign):
                return None
            return _sched_plan_action(e.agents(), machines, assign)
        return fn
    pols = {f"central {r}": pol(r) for r in ("pay", "density", "urgency", "slack")}
    first = best_of(env, pols)
    base = pols[first["how"]]

    def cands(e):
        out = [base(e)]
        for r in ("pay", "urgency", "slack"):
            a = pols[f"central {r}"](e)
            if a not in out:
                out.append(a)
        return out
    return best_with(env, pols, [rollout(env, cands, base, 12, f"rollout over {first['how']}")])


def reference_schedule(view, env):
    """Public, per participant: after the first period each participant knows its own jobs (from its own
    observation) and broadcasts them by message; then every period each participant computes the same assignment
    from the shared job lists (estimated work, value density with urgency, machine speeds estimated from work done)
    and submits it as a joint plan."""
    ags = view.agents()
    sp = view.action_space(ags[0])
    machines = [o for o in sp.limits["decisions"]["book"] if o != "idle"]
    obs = view.step(None)["observations"]
    known = {a: {} for a in ags}                      # what each participant knows: owner -> job list
    for a in ags:
        known[a][a] = [dict(e) for e in obs[a]["entities"] if e.get("kind") == "job"]
    for k in range(len(ags) - 1):                      # ring broadcast: everyone learns every list in two periods
        acts = {}
        for i, a in enumerate(ags):
            to = ags[(i + 1) % len(ags)]
            acts[a] = {"kind": "message", "to": to, "content": {"jobs": known[a]}}
        r = view.step(acts)
        obs = r["observations"]
        for a in ags:
            for m in obs[a]["entities"]:
                if m.get("kind") == "message" and (m.get("content") or {}).get("jobs"):
                    for owner, js in m["content"]["jobs"].items():
                        known[a].setdefault(owner, js)
    speed = {a: [[1.0] * len(machines) for _ in ags] for a in ags}
    last_assign = None
    while not view.done:
        acts = {}
        for i, a in enumerate(ags):
            x = nums(obs[a])
            if last_assign is not None:
                for m, who in enumerate(last_assign):
                    if who == i and x["work_done_this_period"] > 0:
                        speed[a][i][m] = 0.5 * speed[a][i][m] + 0.5 * x["work_done_this_period"]
        jobs_of, cur, rem = [], [], []
        a0 = ags[0]
        for i, a in enumerate(ags):
            js = known[a0].get(a, [])
            own = {e["id"]: e for e in obs[a]["entities"] if e.get("kind") == "job"}
            jl = [{"name": j["id"], "value": j["value"], "deadline": j["due"], "release": j["release"],
                   "est": j["estimated_work"], "finished": own.get(j["id"], j).get("finished", False),
                   "worked": own.get(j["id"], j).get("worked", 0.0)} for j in js]
            cand = [k for k, j in enumerate(jl) if not j["finished"] and j["release"] <= view.t]
            ci = min(cand, key=lambda k: (jl[k]["deadline"], k)) if cand else -1
            jobs_of.append(jl)
            cur.append(ci)
            rem.append(max(0.5, jl[ci]["est"] - jl[ci]["worked"]) if ci >= 0 else 0.0)
        assign = _sched_assign(jobs_of, cur, rem, speed[a0], view.t, "pay", len(machines))
        last_assign = assign
        if all(x is None for x in assign):
            obs = view.step(None)["observations"]
            continue
        obs = view.step(_sched_plan_action(ags, machines, assign))["observations"]


def identify_schedule(env) -> dict:
    """Question: on which machine does the first participant's work go faster? Reserve each machine for it in turn
    (joint plans) and compare the work done per period."""
    view = PublicView(env)
    ags = view.agents()
    machines = [o for o in view.action_space(ags[0]).limits["decisions"]["book"] if o != "idle"]
    rate = {m: [] for m in machines}
    t = 0
    while not view.done and t < view.horizon - 1 and min(len(v) for v in rate.values()) < 6:
        m = machines[t % len(machines)]
        plan = {mm: (ags[0] if mm == m else "none") for mm in machines}
        r = view.step({a: {"kind": "coordinate", "with": [b for b in ags if b != a], "plan": plan} for a in ags})
        w = nums(r["observations"][ags[0]])["work_done_this_period"]
        if w > 0:
            rate[m].append(w)
        t += 1
    est = max(machines, key=lambda m: np.mean(rate[m]) if rate[m] else 0.0)
    truth = env._config["machine_names"][int(np.argmax(env._config["speed"][0]))]
    return {"question": "faster machine for the first participant", "estimate": est, "truth": truth,
            "ok": est == truth}


def claim_schedule(env) -> dict:
    """Shared machines: two participants booking the same machine in one period both get nothing done (and pay);
    the same two with a joint plan that gives each a machine both get work done."""
    eng, st, c = inside(env)
    ags = env.agents()
    t0 = max(j["release"] for i in range(2) for j in c["jobs"][i][:1])
    e = fork(env)
    rel = [min(j["release"] for j in c["jobs"][i]) for i in range(3)]
    while e.t < max(rel[0], rel[1]):
        e.step(None)
    a, b = fork(e), fork(e)
    m0, m1 = c["machine_names"]
    ra = a.step({ags[0]: {"kind": "discrete", "decision": "book", "choice": m0},
                 ags[1]: {"kind": "discrete", "decision": "book", "choice": m0}})
    plan = {m0: ags[0], m1: ags[1]}
    rb = b.step({x: {"kind": "coordinate", "with": [y for y in ags if y != x], "plan": plan} for x in ags})
    wa = [nums(ra["observations"][x])["work_done_this_period"] for x in ags[:2]]
    wb = [nums(rb["observations"][x])["work_done_this_period"] for x in ags[:2]]
    conflict = nums(ra["observations"][ags[0]])[f"{m0}.double_booked"]
    del t0
    return {"ok": wa == [0.0, 0.0] and min(wb) > 0 and conflict == 1.0, "work_double_booked": wa,
            "work_joint_plan": wb}
