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
    """An order arrives at the depot exactly after the hidden lead time; stock shipped to the site with the
    highest hidden demand sells at a higher rate than stock shipped to the lowest (same noise)."""
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
        # ship the same 20 units to one site and count what that site sells over the next twelve ticks, relative
        # to not shipping (same noise): the extra sales the shipment makes there
        e, f = fork(env), fork(env)
        e.step({"kind": "allocate", "amounts": {c["site_names"][i]: 20.0}})
        f.step(None)
        key = f"{c['site_names'][i]}.sold"
        per = []
        for k in range(30):
            per.append(nums(e.step(None)["observation"]).get(key, 0.0) -
                       nums(f.step(None)["observation"]).get(key, 0.0))
        # once the shipment has arrived, the site sells the extra stock at the rate its demand allows: the
        # largest per-tick extra sale over the next ticks (averaged over the best three) measures that rate
        sold[lab] = float(np.mean(sorted(per)[-3:]))
    return {"ok": first == int(c["lead"]) and sold["high"] > sold["low"], "lead": int(c["lead"]),
            "first_difference_at": first, "extra_sales_rate_high_demand_site": sold["high"],
            "extra_sales_rate_low_demand_site": sold["low"]}


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


# ====================================================================== 8.8.1 price war
def _pw_price(c, x):
    return {"kind": "levers", "values": {c["price_name"]: float(x)}}


def _pw_rollout(env, base, grid, label):
    c = env._config
    e = fork(env)
    while not e.done:
        vals = []
        for p in grid:
            f = fork(e)
            f.step(_pw_price(c, p))
            run_fn(f, base, min(6, f.horizon - f.t))
            vals.append(float(f.outcome()["value"]))
        e.step(_pw_price(c, grid[int(np.argmax(vals))]))
    return {"value": float(e.outcome()["value"]), "how": f"rollout over {label}"}


def oracle_price_war(env) -> dict:
    eng, st, c = inside(env)
    floor = float(c["trigger"]) + 0.05 if c["rival_kind"][1] == "trigger" else 0.0

    def tft(hi, pun, k):
        mem = {"left": 0}

        def fn(e):
            if e.t > 0 and e._state["modes"][0] == 1:     # privileged: the learner rival undercut last period
                mem["left"] = k
            if mem["left"] > 0:
                mem["left"] -= 1
                return _pw_price(c, max(pun, floor))
            return _pw_price(c, hi)
        return fn
    pols = {f"retaliate hi={h} pun={p} k={k}": tft(h, p, k) for h in (14.0, 15.0, 16.0, 17.0, 18.0)
            for p in (10.0, 11.5, 13.0) for k in (1, 2)}
    pols.update({f"constant {p}": (lambda e, p=p: _pw_price(c, p)) for p in (12.0, 13.0, 14.0, 15.0, 16.0)})
    first = best_of(env, pols)
    grid = sorted({max(10.0, floor), 13.0, 15.0, 17.0})
    return best_with(env, pols, [_pw_rollout(env, pols[first["how"]], grid, first["how"])])


def reference_price_war(view, env):
    """Public tit-for-tat: hold a high price; when a rival's posted price falls sharply although your own price did
    not fall, answer with a low price for one period, then return to the high price."""
    name = _levers(view.action_space())[0]
    hi, pun = 16.0, 13.5
    prev = None
    my_prev = hi
    punish = 0
    obs = view.step({"kind": "levers", "values": {name: hi}})["observation"]
    while not view.done:
        x = nums(obs)
        rivals = {k: v for k, v in x.items() if k.endswith(".price")}
        if prev is not None and x["my_price"] >= my_prev - 1e-9:
            if any(prev[k] - v > 0.5 for k, v in rivals.items()):
                punish = 1
        prev, my_prev = rivals, x["my_price"]
        p = pun if punish > 0 else hi
        punish = max(0, punish - 1)
        obs = view.step({"kind": "levers", "values": {name: p}})["observation"]


def identify_price_war(env) -> dict:
    """Question: which rival learns to undercut, and does the other one follow your price or punish low prices
    (trigger)? Hold a high price for twelve periods (the learner jumps down, a follower drifts toward you, a trigger
    stays put), then a low price for four periods (a trigger drops at once, a follower drifts)."""
    view = PublicView(env)
    name = _levers(view.action_space())[0]
    ph1, ph2 = [], []
    for _ in range(12):
        ph1.append(nums(view.step({"kind": "levers", "values": {name: 18.0}})["observation"]))
    for _ in range(4):
        ph2.append(nums(view.step({"kind": "levers", "values": {name: 10.5}})["observation"]))
    rivals = sorted(k for k in ph1[0] if k.endswith(".price"))
    jumps = {r: max(abs(ph1[i][r] - ph1[i - 1][r]) for i in range(1, len(ph1))) for r in rivals}
    learner = max(jumps, key=jumps.get)
    other = [r for r in rivals if r != learner][0]
    o = [x[other] for x in ph2]
    flat_after_drop = o[1] < o[0] - 1.0 and abs(o[3] - o[2]) < 1e-6 and abs(o[2] - o[1]) < 1e-6
    kind = "trigger" if flat_after_drop else "follower"
    c = env._config
    est = [learner.split(".")[0], kind]
    truth = [c["rival_names"][0], c["rival_kind"][1]]
    return {"question": "which rival is the undercutting learner; is the other a follower or a trigger",
            "estimate": est, "truth": truth, "ok": est == truth}


def claim_price_war(env) -> dict:
    """The learner rival's propensity to undercut grows against a passive high price and shrinks against
    retaliation (same noise); the other rival follows your price down, or punishes a price below its trigger."""
    eng, st, c = inside(env)

    def pu(e):
        d = float(c["temp"]) * (e._state["V"][0][1] - e._state["V"][0][0])
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, d))))
    a, b = fork(env), fork(env)
    for _ in range(30):
        a.step(_pw_price(c, 16.0))
        cut = b.t > 0 and b._state["modes"][0] == 1
        b.step(_pw_price(c, 11.0 if cut else 16.0))
    passive, retaliate = pu(a), pu(b)
    hi, lo = fork(env), fork(env)
    for _ in range(4):
        rh = hi.step(_pw_price(c, 18.0))
        rl = lo.step(_pw_price(c, 10.5))
    other = f"{c['rival_names'][1]}.price"
    resp = nums(rh["observation"])[other] - nums(rl["observation"])[other]
    return {"ok": passive > retaliate + 0.1 and resp > 1.0, "undercut_propensity_vs_passive": passive,
            "undercut_propensity_vs_retaliation": retaliate, "other_rival_price_high_minus_low": resp,
            "other_kind": c["rival_kind"][1]}


# ====================================================================== 8.8.2 market entry
def _enter(c, ms):
    steps = [{"kind": "discrete", "decision": c["market_names"][m], "choice": "enter"} for m in ms]
    return {"kind": "sequence", "steps": steps} if steps else None


def oracle_entry(env) -> dict:
    import itertools
    eng, st, c = inside(env)
    M = len(c["market_names"])
    good = [m for m in range(M) if c["inc_kind"][m] != "tough"]
    pols = {"stay out": lambda e: None}
    for k in range(1, len(good) + 1):
        for sub in itertools.combinations(good, k):
            pols[f"enter {list(sub)} and stay"] = (lambda e, sub=sub: _enter(c, sub) if e.t == 0 else None)
    return best_of(env, pols)


def reference_entry(view, env):
    """Public: watch the incumbents' prices for three periods, enter the half of the markets with the highest prices
    (high prices signal high-cost incumbents), and exit a market after three consecutive losing periods."""
    markets = sorted(view.action_space().limits["decisions"])
    seen = {m: [] for m in markets}
    for _ in range(3):
        x = nums(view.step(None)["observation"])
        for m in markets:
            seen[m].append(x[f"{m}.incumbent_price"])
    ranked = sorted(markets, key=lambda m: -np.mean(seen[m]))
    chosen = ranked[:max(1, len(markets) // 2)]
    obs = view.step({"kind": "sequence", "steps": [{"kind": "discrete", "decision": m, "choice": "enter"}
                                                   for m in chosen]})["observation"]
    losses = {m: 0 for m in chosen}
    out = set()
    while not view.done:
        x = nums(obs)
        act = None
        for m in chosen:
            if m in out:
                continue
            losses[m] = losses[m] + 1 if x[f"{m}.my_profit"] < 0 else 0
            if losses[m] >= 3 and act is None:
                act = {"kind": "discrete", "decision": m, "choice": "exit"}
                out.add(m)
        obs = view.step(act)["observation"]


def identify_entry(env) -> dict:
    """Question: which incumbents fight an entrant who stays? Enter every market and stay fourteen periods; an
    incumbent that still fights (losing periods) in at least three of the last six is classed as fighting."""
    view = PublicView(env)
    markets = sorted(view.action_space().limits["decisions"])
    view.step({"kind": "sequence", "steps": [{"kind": "discrete", "decision": m, "choice": "enter"} for m in markets]})
    rows = [nums(view.step(None)["observation"]) for _ in range(14)]
    est = sorted(m for m in markets if sum(r[f"{m}.my_profit"] < 0 for r in rows[-6:]) >= 3)
    c = env._config
    truth = sorted(c["market_names"][k] for k in range(len(c["market_names"])) if c["inc_kind"][k] == "tough")
    return {"question": "which incumbents fight a persistent entrant", "estimate": est, "truth": truth,
            "ok": est == truth}


def claim_entry(env) -> dict:
    """Incumbent policies differ: entering a tough incumbent's market loses money period after period, a soft one
    accommodates, and a reputation-minded one fights for its first periods and then accommodates. Fight rates are
    measured over three noise replicates."""
    eng, st, c = inside(env)
    M = len(c["market_names"])
    rate: dict = {"tough": [], "soft": [], "reputation_early": [], "reputation_late": []}
    for r in range(3):
        e = replicate(env, r)
        e.step(_enter(c, range(M)))
        rows = [nums(e.step(None)["observation"]) for _ in range(16)]
        for m in range(M):
            f = [row[f"{c['market_names'][m]}.my_profit"] < 0 for row in rows]
            k = c["inc_kind"][m]
            if k == "reputation":
                p = int(c["patience"][m]) - 1
                rate["reputation_early"] += f[:max(1, p)]
                rate["reputation_late"] += f[p + 2:]
            else:
                rate[k] += f
    mean = {k: (float(np.mean(v)) if v else None) for k, v in rate.items()}
    ok = mean["tough"] is not None and mean["soft"] is not None and mean["tough"] > 0.6 and mean["soft"] < 0.3
    if mean["reputation_early"] is not None:
        ok = ok and mean["reputation_early"] > 0.5 and mean["reputation_late"] < 0.35
    return {"ok": bool(ok), "fight_rates": mean}


# ====================================================================== 8.8.3 bargaining
def lp_offer(w, v, a):
    """The terms maximising your utility sum w*x subject to the counterpart's utility sum v*(1-x) >= a."""
    cap = max(0.0, 1.0 - a)
    x = [0.0] * len(w)
    for k in sorted(range(len(w)), key=lambda k: -(w[k] / max(v[k], 1e-9))):
        if cap <= 1e-12:
            break
        take = min(1.0, cap / max(v[k], 1e-9))
        x[k] = take
        cap -= take * v[k]
    return x


def _bg_act(c, x, resp="propose"):
    steps = [{"kind": "levers", "values": {n: float(min(1.0, max(0.0, v))) for n, v in zip(c["issue_names"], x)}},
             {"kind": "discrete", "decision": "respond", "choice": resp}]
    return {"kind": "sequence", "steps": steps}


def oracle_bargain(env) -> dict:
    eng, st, c = inside(env)
    K = int(c["rounds"])
    w = c["my_w"]

    def asp_b(j, r):
        r0 = float(c["cp_r0"][j])
        return r0 + (1.0 - r0) * (1.0 - (min(r, K) / K) ** (1.0 / float(c["cp_e"][j])))

    def pol(delta, mode):
        def fn(e):
            s = e._state
            j, r = s["cp"], s["round"]
            v = c["cp_v"][j]
            if c["cp_kind"][j] == "boulware":
                best = max(range(K), key=lambda rr: float(c["deal_value"]) * sum(
                    wk * xk for wk, xk in zip(w, lp_offer(w, v, asp_b(j, rr) + delta))) - rr * float(c["round_cost"]))
                if mode == "early":
                    best = min(best, K - 2)
                return _bg_act(c, lp_offer(w, v, asp_b(j, best) + delta)) if r >= best else None
            z = float(c["cp_zeta"][j])
            target = max(float(c["cp_r0"][j]) + delta, (float(c["asp0"]) + delta) / (1.0 + z))
            if r == 0:
                return _bg_act(c, [1.0] * len(w))
            return _bg_act(c, lp_offer(w, v, target))
        return fn
    return best_of(env, {f"plan margin={d} {m}": pol(d, m) for d in (0.04, 0.06, 0.1) for m in ("late", "early")})


def reference_bargain(view, env):
    """Public: learn each counterpart's ranking of the issues from the order in which it gives them away in its
    offers; propose terms that keep the issues it values least and give away the one it values most, starting
    demanding and conceding each round; accept its offer when it is at least as good as the next proposal."""
    c = env._config                                   # public: issue names and your own weights are in your brief
    w = c["my_w"]
    nI = len(w)
    first_give: dict = {}
    K = 6
    obs = view.step(None)["observation"]
    while not view.done:
        x = nums(obs)
        ent = [e for e in obs["entities"] if e.get("kind") == "counterpart"]
        cp = ent[0]["id"] if ent else "?"
        r = int(x.get("round", 0))
        offer = [x.get(f"offer.{n}", 0.0) for n in c["issue_names"]]
        seen = first_give.setdefault(cp, [0.0] * nI)
        for k in range(nI):
            if offer[k] > 0 and seen[k] == 0:
                seen[k] = 1.0 / (1 + r) + 0.01 * offer[k]
        order = sorted(range(nI), key=lambda k: -seen[k])       # least valued by the counterpart first
        v_hat = [0.0] * nI
        for rank, k in enumerate(order):
            v_hat[k] = 1.0 + rank
        tot = sum(v_hat)
        v_hat = [v / tot for v in v_hat]
        a_next = max(0.45, 0.9 - 0.1 * (r + 1))
        prop = lp_offer(w, v_hat, a_next)
        u_prop = sum(a * b for a, b in zip(w, prop))
        u_off = sum(a * b for a, b in zip(w, offer))
        if u_off >= u_prop - 0.03 or (r >= K - 1 and u_off > 0.2):
            act = {"kind": "discrete", "decision": "respond", "choice": "accept"}
        else:
            act = _bg_act(c, prop)
        obs = view.step(act)["observation"]


def identify_bargain(env) -> dict:
    """Question: which issue does each counterpart value least? Its opening offer gives that issue away first."""
    view = PublicView(env)
    c = env._config
    est = {}
    obs = view.step(None)["observation"]
    while not view.done and len(est) < len(c["cp_names"]):
        x = nums(obs)
        ent = [e for e in obs["entities"] if e.get("kind") == "counterpart"]
        offer = {n: x[f"offer.{n}"] for n in c["issue_names"]}
        act = None
        if ent and ent[0]["id"] not in est and max(offer.values()) > 0:
            est[ent[0]["id"]] = max(offer, key=offer.get)         # the first issue it gives away
        if ent and ent[0]["id"] in est:
            act = {"kind": "discrete", "decision": "respond", "choice": "walk_away"}
        obs = view.step(act)["observation"]
    truth = {c["cp_names"][j]: c["issue_names"][int(np.argmin(c["cp_v"][j]))] for j in range(len(c["cp_names"]))}
    return {"question": "each counterpart's least valued issue", "estimate": est, "truth": truth, "ok": est == truth}


def claim_bargain(env) -> dict:
    """Private preferences make trade-offs pay: at the last round of a deal with a time-dependent counterpart, terms
    built from its weights are accepted while uniform terms worth the same to you are rejected; a tit-for-tat
    counterpart rejects a moderate offer made cold but accepts it after you have conceded to it."""
    eng, st, c = inside(env)
    w = c["my_w"]
    K = int(c["rounds"])
    out: dict = {}
    e = fork(env)
    guard = 0
    while not (c["cp_kind"][e._state["cp"]] == "boulware" and e._state["round"] == K - 1) and guard < 200 \
            and not e.done:
        e.step(None)
        guard += 1
    if not e.done and guard < 200:
        j = e._state["cp"]
        a = eng.cp_aspiration(e._state)
        x = lp_offer(w, c["cp_v"][j], a + 0.08)
        u = sum(p * q for p, q in zip(w, x))
        f1, f2 = fork(e), fork(e)
        f1.step(_bg_act(c, x))
        f2.step(_bg_act(c, [u] * len(w)))
        out["tradeoff_accepted"] = f1._state["deals_made"] > e._state["deals_made"]
        out["uniform_accepted"] = f2._state["deals_made"] > e._state["deals_made"]
        out["aspiration"] = a
    e = fork(env)
    guard = 0
    while not (c["cp_kind"][e._state["cp"]] == "tit_for_tat" and e._state["round"] == 0) and guard < 200 \
            and not e.done:
        e.step(None)
        guard += 1
    if not e.done and guard < 200:
        j = e._state["cp"]
        z = float(c["cp_zeta"][j])
        vt = min(0.9, max(float(c["cp_r0"][j]), float(c["asp0"]) / (1.0 + z)) + 0.07)
        x = lp_offer(w, c["cp_v"][j], vt)
        f1, f2 = fork(e), fork(e)
        n0 = e._state["deals_made"]
        f1.step(_bg_act(c, x))
        f2.step(_bg_act(c, [1.0] * len(w)))
        f2.step(_bg_act(c, x))
        out["tft_cold_accepted"] = f1._state["deals_made"] > n0
        out["tft_after_concession_accepted"] = f2._state["deals_made"] > n0
    ok = "tradeoff_accepted" in out and "tft_cold_accepted" in out and out["tradeoff_accepted"] \
        and not out["uniform_accepted"] and not out["tft_cold_accepted"] and out["tft_after_concession_accepted"]
    return {"ok": bool(ok), **out}


# ====================================================================== 8.8.4 contests (blotto)
def _blotto_br(prize, y, r, budget, cost, step=0.25):
    """Exact best response on a grid (contest success is S-shaped, so a greedy split would stall): dynamic
    programming over sites of prize * x^r / (x^r + y^r) - cost * x with total effort at most `budget`."""
    n = int(round(budget / step))
    K = len(prize)

    def val(k, i):
        x = i * step
        if x <= 0:
            return 0.0
        return prize[k] * (x ** r / (x ** r + y[k] ** r) if y[k] > 0 else 1.0) - cost * x
    best = [0.0] * (n + 1)
    choice = []
    for k in range(K):
        v = [val(k, i) for i in range(n + 1)]
        nb, ch = [0.0] * (n + 1), [0] * (n + 1)
        for b in range(n + 1):
            m, arg = -1e18, 0
            for i in range(b + 1):
                s = best[b - i] + v[i]
                if s > m:
                    m, arg = s, i
            nb[b], ch[b] = m, arg
        best = nb
        choice.append(ch)
    x = [0.0] * K
    b = int(np.argmax(best))
    for k in range(K - 1, -1, -1):
        i = choice[k][b]
        x[k] = i * step
        b -= i
    return x


def oracle_blotto(env) -> dict:
    eng, st, c = inside(env)

    def pol(step, shade):
        def fn(e):
            zb = float(peek(e).standard_normal())
            y = e._engine.opponent_allocation(e._state, zb)
            x = _blotto_br(c["prize"], y, float(c["decisive"]), float(c["budget"]) * shade, float(c["unit_cost"]),
                           step)
            return {"kind": "allocate", "amounts": dict(zip(c["site_names"], x))}
        return fn
    return best_of(env, {f"best response to the opponent's next move (step {s}, budget x{b})": pol(s, b)
                         for s in (0.25, 0.5) for b in (1.0, 0.8)})


def reference_blotto(view, env):
    """Public: estimate the contest's decisiveness from observed efforts and shares, predict the opponent's next
    efforts by its last ones, and best-respond."""
    sp = view.action_space()
    sites = _targets(sp)
    B = float(sp.limits["allocate_total"])
    prize = dict(zip(env._config["site_names"], env._config["prize"]))      # public prizes
    r_est = []
    r = 1.5
    obs = view.step({"kind": "allocate", "amounts": {s: B / len(sites) for s in sites}})["observation"]
    last_x = {s: B / len(sites) for s in sites}
    while not view.done:
        x = nums(obs)
        y = [x[f"{s}.rival_effort"] for s in sites]
        for s, yy in zip(sites, y):
            sh = x[f"{s}.my_share"]
            if 0.02 < sh < 0.98 and yy > 0 and last_x[s] > 0 and abs(math.log(last_x[s] / yy)) > 0.05:
                r_est.append(math.log(sh / (1 - sh)) / math.log(last_x[s] / yy))
        if len(r_est) >= 3:
            r = float(np.clip(np.median(r_est), 0.5, 4.0))
        alloc = _blotto_br([prize[s] for s in sites], y, r, B, 12.0, 0.5)
        last_x = dict(zip(sites, alloc))
        obs = view.step({"kind": "allocate", "amounts": dict(zip(sites, alloc))})["observation"]


def identify_blotto(env) -> dict:
    """Question: at which site will the opponent put the most effort next period? Play an even allocation for ten
    periods and predict the site where its effort was largest most often over the last four; the truth is its effort
    in the period after (computed from its private state)."""
    view = PublicView(env)
    sites = _targets(view.action_space())
    B = float(view.action_space().limits["allocate_total"])
    even = {"kind": "allocate", "amounts": {s: B / len(sites) for s in sites}}
    tops = []
    for _ in range(10):
        x = nums(view.step(even)["observation"])
        tops.append(max(sites, key=lambda s: x[f"{s}.rival_effort"]))
    est = max(set(tops[-4:]), key=lambda s: (tops[-4:].count(s), tops[-1] == s))
    view.step(even)
    truth = env._config["site_names"][int(np.argmax(env._state["y"]))]
    return {"question": "the opponent's most contested site next period", "estimate": est, "truth": truth,
            "ok": est == truth}


def claim_blotto(env) -> dict:
    """The opponent moves on what it has seen: changing your effort this period leaves its effort this period
    unchanged and changes its effort next period; more effort at a site wins a larger share of it."""
    eng, st, c = inside(env)
    s0 = c["site_names"][0]
    a, b = fork(env), fork(env)
    ra = a.step({"kind": "allocate", "amounts": {s0: float(c["budget"])}})
    rb = b.step({"kind": "allocate", "amounts": {c["site_names"][-1]: float(c["budget"])}})
    same_now = all(nums(ra["observation"])[f"{s}.rival_effort"] == nums(rb["observation"])[f"{s}.rival_effort"]
                   for s in c["site_names"])
    ra2, rb2 = a.step(None), b.step(None)
    differs_next = any(nums(ra2["observation"])[f"{s}.rival_effort"] != nums(rb2["observation"])[f"{s}.rival_effort"]
                       for s in c["site_names"])
    lo, hi = fork(env), fork(env)
    sl = nums(lo.step({"kind": "allocate", "amounts": {s0: 1.0}})["observation"])[f"{s0}.my_share"]
    sh = nums(hi.step({"kind": "allocate", "amounts": {s0: 4.0}})["observation"])[f"{s0}.my_share"]
    return {"ok": same_now and differs_next and sh > sl, "opponent_unchanged_same_period": same_now,
            "opponent_changed_next_period": differs_next, "share_effort_1": sl, "share_effort_4": sh}


# ====================================================================== 8.8.5 interdiction
def oracle_interdiction(env) -> dict:
    from .engines.strategic import _pick
    eng, st, c = inside(env)
    N = len(c["route_names"])
    kap = float(c["kappa"])
    I = float(c["inspectors"])
    val = [float(c["bounty"]) + float(c["damage"][j]) for j in range(N)]

    def clairvoyant(e):
        g = peek(e)
        n = int(c["attempts"])
        ur, uc = g.random(n), g.random(n)
        p = e._engine.attacker_mix(e._state)
        need: dict = {}
        for k in range(n):
            j = _pick(p, float(ur[k]))
            need.setdefault(j, []).append(-math.log(max(1e-12, 1.0 - float(uc[k]))) / kap + 1e-6)
        x = [0.0] * N
        left = I
        while True:                               # buy the catches with the most value per inspector first
            best, bj, bx = 0.0, -1, 0.0
            for j in range(N):
                for q in sorted(need.get(j, [])):
                    if q > x[j] and q - x[j] <= left:
                        n_caught = sum(1 for qq in need[j] if x[j] < qq <= q)
                        g_ = (n_caught * val[j] - float(c["deploy_cost"]) * (q - x[j])) / (q - x[j])
                        if g_ > best:
                            best, bj, bx = g_, j, q - x[j]
            if bj < 0:
                break
            x[bj] += bx
            left -= bx
        return {"kind": "allocate", "amounts": dict(zip(c["route_names"], x))}

    def expected(e):
        p = e._engine.attacker_mix(e._state)
        n = int(c["attempts"])

        def gain(j, x):
            return n * p[j] * (1 - math.exp(-kap * x)) * val[j] - float(c["deploy_cost"]) * x
        return {"kind": "allocate", "amounts": dict(zip(c["route_names"], greedy_alloc(gain, I, 0.5, N)))}
    return best_of(env, {"clairvoyant catches": clairvoyant, "best response to the attacker's mix": expected})


def reference_interdiction(view, env):
    """Public: estimate the attacker's route frequencies from catches and breaches (exponential smoothing), assume a
    moderate catch rate per inspector, and deploy to maximise expected catches weighted by bounty plus damage."""
    sp = view.action_space()
    routes = _targets(sp)
    I = float(sp.limits["allocate_total"])
    dmg = dict(zip(env._config["route_names"], env._config["damage"]))      # public route damages
    freq = {r: 1.0 / len(routes) for r in routes}
    obs = view.step({"kind": "allocate", "amounts": {r: I / len(routes) for r in routes}})["observation"]
    while not view.done:
        x = nums(obs)
        n = sum(x[f"{r}.caught"] + x[f"{r}.got_through"] for r in routes)
        for r in routes:
            f = (x[f"{r}.caught"] + x[f"{r}.got_through"]) / max(n, 1.0)
            freq[r] = 0.75 * freq[r] + 0.25 * f

        def gain(k, xx):
            r = routes[k]
            return 5 * freq[r] * (1 - math.exp(-0.35 * xx)) * (30.0 + dmg[r]) - xx
        alloc = greedy_alloc(gain, I, 0.5, len(routes))
        obs = view.step({"kind": "allocate", "amounts": dict(zip(routes, alloc))})["observation"]


def identify_interdiction(env) -> dict:
    """Question: which route does the attacker prefer when nothing is inspected? Inspect nothing for twelve periods
    and count breaches per route."""
    view = PublicView(env)
    routes = _targets(view.action_space())
    cnt = {r: 0.0 for r in routes}
    for _ in range(12):
        x = nums(view.step(None)["observation"])
        for r in routes:
            cnt[r] += x[f"{r}.got_through"]
    est = max(cnt, key=cnt.get)
    truth = env._config["route_names"][int(np.argmax(env._config["att_util"]))]
    return {"question": "the attacker's preferred route", "estimate": est, "truth": truth, "ok": est == truth}


def claim_interdiction(env) -> dict:
    """Strategic substitution: inspecting the attacker's favourite route heavily moves its attempts to other routes
    (it learns from being caught); without inspection it keeps using it. The number of attempts is unchanged."""
    eng, st, c = inside(env)
    top = int(np.argmax(c["att_util"]))
    a, b = fork(env), fork(env)
    ta = [0.0] * len(c["route_names"])
    tb = [0.0] * len(c["route_names"])
    for k in range(20):
        xa = nums(a.step({"kind": "allocate", "amounts": {c["route_names"][top]: float(c["inspectors"])}})
                  ["observation"])
        xb = nums(b.step(None)["observation"])
        if k >= 10:
            for j, r in enumerate(c["route_names"]):
                ta[j] += xa[f"{r}.caught"] + xa[f"{r}.got_through"]
                tb[j] += xb[f"{r}.caught"] + xb[f"{r}.got_through"]
    sa, sb = ta[top] / max(1, sum(ta)), tb[top] / max(1, sum(tb))
    return {"ok": sa < sb - 0.1 and sum(ta) == sum(tb), "share_on_favourite_inspected": sa,
            "share_on_favourite_uninspected": sb, "attempts": [sum(ta), sum(tb)]}


# ====================================================================== 8.8.6 learning opponent
def oracle_learning(env) -> dict:
    from .engines.strategic import _pick
    eng, st, c = inside(env)
    A = c["A"]

    def clairvoyant(e):
        p = e._engine.opponent_mix(e._state)
        j = _pick(p, float(peek(e).random()))
        i = int(np.argmax([A[i][j] for i in range(3)]))
        return {"kind": "discrete", "decision": "play", "choice": c["option_names"][i]}

    def expected(e):
        p = e._engine.opponent_mix(e._state)
        i = int(np.argmax([sum(p[j] * A[i][j] for j in range(3)) for i in range(3)]))
        return {"kind": "discrete", "decision": "play", "choice": c["option_names"][i]}
    return best_of(env, {"clairvoyant best response": clairvoyant, "best response to its mixed strategy": expected})


def reference_learning(view, env):
    """Public: predict the opponent's next action from what it played after your recent actions (a frequency model
    over the last ten periods, conditioned on your last action), and play your best reply to the prediction."""
    opts = view.action_space().limits["decisions"]["play"]
    c = env._config
    A = c["A"]                                       # your own payoffs are public (in your brief)
    hist = []
    k = 0
    obs = view.step({"kind": "discrete", "decision": "play", "choice": opts[0]})["observation"]
    while not view.done:
        x = nums(obs)
        hist.append((int(x["my_play"]), int(x["opponent_played"])))
        recent = hist[-10:]
        mine = hist[-1][0]
        cond = [j for (i, j) in recent[:-1] if i == mine]
        pool = cond if len(cond) >= 3 else [j for _, j in recent]
        cnt = np.bincount(pool, minlength=3) + 0.1
        pj = cnt / cnt.sum()
        i = int(np.argmax([sum(pj[j] * A[i][j] for j in range(3)) for i in range(3)])) if len(hist) >= 3 else k % 3
        k += 1
        obs = view.step({"kind": "discrete", "decision": "play", "choice": c["option_names"][i]})["observation"]


def identify_learning(env) -> dict:
    """Question: which action does the opponent learn to play against your first option? Play it for fourteen
    periods and take the opponent's most frequent action over the last six."""
    view = PublicView(env)
    c = env._config
    first = c["option_names"][0]
    seen = []
    for _ in range(14):
        seen.append(int(nums(view.step({"kind": "discrete", "decision": "play", "choice": first})
                             ["observation"])["opponent_played"]))
    est = int(np.bincount(seen[-6:], minlength=3).argmax())
    truth = int(np.argmax(c["B"][0]))
    return {"question": "the opponent's learned reply to your first option", "estimate": est, "truth": truth,
            "ok": est == truth}


def claim_learning(env) -> dict:
    """The opponent's strategy is learned: after fifteen periods of one option its mixed strategy concentrates on
    the best reply to that option, and it depends on which option you played (same noise)."""
    eng, st, c = inside(env)
    mixes = []
    for i in range(3):
        e = fork(env)
        for _ in range(15):
            e.step({"kind": "discrete", "decision": "play", "choice": c["option_names"][i]})
        mixes.append(eng.opponent_mix(e._state))
    br = [int(np.argmax(c["B"][i])) for i in range(3)]
    conc = [mixes[i][br[i]] for i in range(3)]
    return {"ok": min(conc) > 0.6 and len(set(int(np.argmax(m)) for m in mixes)) == 3,
            "mass_on_best_reply": conc, "best_replies": br}


# ====================================================================== 8.9.2 hidden profile
def _joint(ags, plan):
    return {a: {"kind": "coordinate", "with": [b for b in ags if b != a], "plan": plan} for a in ags}


def oracle_profile(env) -> dict:
    eng, st, c = inside(env)

    def pol(threshold):
        def fn(e):
            s = e._state
            if not s["open"]:
                return None
            vals = [sum(w * f for w, f in zip(c["w"], F)) + float(c["value_sd"]) * ep for F, ep in zip(s["F"], s["eps"])]
            k = int(np.argmax(vals))
            if vals[k] * float(c["scale"]) <= threshold:
                return None
            return _joint(e.agents(), {"option": c["slot_names"][k]})
        return fn
    return best_of(env, {f"knows the weights (decide above {t})": pol(t) for t in (0.0, 20.0, 60.0)})


def reference_profile(view, env):
    """Public, per participant: when a new set of options appears, each participant passes the attributes it knows
    to the next one round a ring, twice, so that all three know every attribute; then all commit to the option with
    the best value under weights fitted (ridge regression) to the payoffs of earlier joint choices."""
    ags = view.agents()
    n = len(ags)
    slots = view.action_space(ags[0]).kinds["coordinate"]["properties"]["plan"]["properties"]["option"]["enum"]
    X, Y = [], []
    obs = {a: view.observe(a) for a in ags}
    know = {a: {} for a in ags}
    stage = None
    while not view.done:
        x0 = nums(obs[ags[0]])
        if x0.get("new_options") == 1.0:
            for i, a in enumerate(ags):
                know[a] = {i: [nums(obs[a])[f"{s}.my_attribute"] for s in slots]}
            stage = 0
        if stage in (0, 1):
            acts = {a: {"kind": "message", "to": ags[(i + 1) % n], "content": {"cols": know[a]}}
                    for i, a in enumerate(ags)}
            r = view.step(acts)
            obs = r["observations"]
            for a in ags:
                for m in obs[a]["entities"]:
                    if m.get("kind") == "message" and (m.get("content") or {}).get("cols"):
                        for k, col in m["content"]["cols"].items():
                            know[a][int(k)] = col
            stage += 1
            continue
        if stage == 2 and all(len(know[a]) == n for a in ags):
            Fm = np.asarray([[know[ags[0]][i][k] for i in range(n)] for k in range(len(slots))])
            if len(Y) >= 3:
                Xa = np.asarray(X)
                w_hat = np.linalg.solve(Xa.T @ Xa + 0.5 * np.eye(n), Xa.T @ np.asarray(Y))
            else:
                w_hat = np.ones(n) / math.sqrt(n)
            k = int(np.argmax(Fm @ w_hat))
            r = view.step(_joint(ags, {"option": slots[k]}))
            res = r["results"][ags[0]]
            if res.get("ok") and (res.get("result") or {}).get("agreed"):
                X.append(list(Fm[k]))
                Y.append(res["result"]["payoff"] / float(env._config["scale"]))     # the payoff scale is public
            obs = r["observations"]
            stage = None
            continue
        obs = view.step(None)["observations"]


def identify_profile(env) -> dict:
    """Question: whose attribute carries the most weight in the options' payoffs? Pool attributes, choose jointly,
    and regress the payoffs of the choices on the pooled attributes."""
    view = PublicView(env)
    ags = view.agents()
    slots = view.action_space(ags[0]).kinds["coordinate"]["properties"]["plan"]["properties"]["option"]["enum"]
    X, Y = [], []
    rng = np.random.default_rng(0)
    obs = {a: view.observe(a) for a in ags}
    while not view.done:
        if nums(obs[ags[0]]).get("options_open") == 1.0:
            Fm = [[nums(obs[a])[f"{s}.my_attribute"] for a in ags] for s in slots]
            k = int(rng.integers(len(slots)))
            r = view.step(_joint(ags, {"option": slots[k]}))
            res = r["results"][ags[0]]
            if res.get("ok") and (res.get("result") or {}).get("agreed"):
                X.append(Fm[k])
                Y.append(res["result"]["payoff"])
            obs = r["observations"]
        else:
            obs = view.step(None)["observations"]
    beta = np.linalg.lstsq(np.c_[np.ones(len(Y)), np.asarray(X)], np.asarray(Y), rcond=None)[0][1:]
    est = int(np.argmax(beta))
    truth = int(np.argmax(env._config["w"]))
    return {"question": "the participant whose attribute weighs most", "estimate": est, "truth": truth,
            "ok": est == truth}


def claim_profile(env) -> dict:
    """Distributed information: choosing by the pooled, correctly weighted attributes earns more than choosing by
    any one participant's attribute alone (same options, same noise)."""
    eng, st, c = inside(env)

    def by(score):
        e = fork(env)
        while not e.done:
            s = e._state
            if s["open"]:
                k = int(np.argmax([score(F) for F in s["F"]]))
                e.step(_joint(e.agents(), {"option": c["slot_names"][k]}))
            else:
                e.step(None)
        return float(e.outcome()["value"])
    pooled = by(lambda F: sum(w * f for w, f in zip(c["w"], F)))
    single = [by(lambda F, i=i: F[i]) for i in range(3)]
    return {"ok": pooled > max(single), "pooled": pooled, "single_attribute": single}


# ====================================================================== 8.9.3 commons
def _effort_for(S, target, q, emax):
    h = max(0.0, S - target)
    return float(min(emax, h / max(q * S, 1e-9))) if S > 0 else 0.0


def oracle_commons(env) -> dict:
    eng, st, c = inside(env)
    P = len(c["patch_names"])
    q, emax = float(c["catchability"]), float(c["e_max"])

    def pol(frac, liquidate):
        def fn(e):
            s = e._state
            acts = {}
            for i, a in enumerate(e.agents()):
                p = i % P if i < P else None
                if p is None:
                    acts[a] = {"kind": "discrete", "decision": "ground", "choice": "rest"}
                    continue
                S = s["S"][p]
                last = e.horizon - e.t <= liquidate
                tgt = 0.0 if last else max(frac * float(c["Kp"][p]), 1.3 * float(c["allee"][p]))
                E = emax if last else _effort_for(S, tgt, q, emax)
                acts[a] = {"kind": "sequence", "steps": [
                    {"kind": "discrete", "decision": "ground", "choice": c["patch_names"][p]},
                    {"kind": "levers", "values": {"effort": E}}]}
            return acts
        return fn
    return best_of(env, {f"escapement {f} of capacity, liquidate last {L}": pol(f, L)
                         for f in (0.45, 0.55, 0.65, 0.75, 0.85) for L in (3, 6, 9)})


def reference_commons(view, env):
    """Public, per participant: each takes its own ground (by its position in the team, so nobody crowds); it
    surveys before fishing, takes the first survey as roughly the ground's capacity, and fishes the survey down to
    60% of that each period (effort from the stated 2% per unit), liquidating in the last three periods."""
    ags = view.agents()
    grounds = [o for o in view.action_space(ags[0]).limits["decisions"]["ground"] if o != "rest"]
    P = len(grounds)
    q, emax = 0.02, 10.0                                   # public: stated in the brief
    acts = {a: {"kind": "sequence", "steps": [
        {"kind": "discrete", "decision": "ground", "choice": grounds[i % P] if i < P else "rest"},
        {"kind": "levers", "values": {"effort": 0.0}}]} for i, a in enumerate(ags)}
    obs = view.step(acts)["observations"]
    cap = {a: nums(obs[a]).get("my_ground_survey", 0.0) / 0.85 for a in ags}
    while not view.done:
        acts = {}
        for i, a in enumerate(ags):
            if i >= P:
                continue
            S = nums(obs[a]).get("my_ground_survey", 0.0)
            last = view.horizon - view.t <= 3
            E = emax if last else _effort_for(S, 0.6 * cap[a], q, emax)
            acts[a] = {"kind": "levers", "values": {"effort": E}}
        obs = view.step(acts)["observations"]


def identify_commons(env) -> dict:
    """Question: which ground has the largest carrying capacity? Survey without fishing for 26 periods (stocks
    regrow toward capacity) and compare the last ten surveys."""
    view = PublicView(env)
    ags = view.agents()
    grounds = [o for o in view.action_space(ags[0]).limits["decisions"]["ground"] if o != "rest"]
    P = len(grounds)
    sv = {g: [] for g in grounds}
    for t in range(26):
        acts = {}
        for i, a in enumerate(ags):
            g = grounds[(i + t * len(ags)) % P] if P > len(ags) else grounds[i % P]
            acts[a] = {"kind": "sequence", "steps": [{"kind": "discrete", "decision": "ground", "choice": g},
                                                     {"kind": "levers", "values": {"effort": 0.0}}]}
        r = view.step(acts)
        if t >= 16:
            for i, a in enumerate(ags):
                x = nums(r["observations"][a])
                g = grounds[(i + t * len(ags)) % P] if P > len(ags) else grounds[i % P]
                if "my_ground_survey" in x:
                    sv[g].append(x["my_ground_survey"])
    est = max(grounds, key=lambda g: np.mean(sv[g]) if sv[g] else -1)
    truth = env._config["patch_names"][int(np.argmax(env._config["Kp"]))]
    return {"question": "the ground with the largest capacity", "estimate": est, "truth": truth, "ok": est == truth}


def claim_commons(env) -> dict:
    """Shared stocks: two boats on one ground catch less per unit of effort than one boat alone; a stock fished below
    its tipping point keeps shrinking when fishing stops, one left above it recovers."""
    eng, st, c = inside(env)
    g0 = c["patch_names"][0]
    ags = env.agents()
    go = {"kind": "sequence", "steps": [{"kind": "discrete", "decision": "ground", "choice": g0},
                                        {"kind": "levers", "values": {"effort": 5.0}}]}
    a, b = fork(env), fork(env)
    ra = a.step({ags[0]: go, ags[1]: go})
    rb = b.step({ags[0]: go})
    per_a = nums(ra["observations"][ags[0]])["my_catch"] / 5.0
    per_b = nums(rb["observations"][ags[0]])["my_catch"] / 5.0
    A_ = float(c["allee"][0])
    lo, hi = fork(env), fork(env)
    lo._state["S"][0] = 0.8 * A_
    hi._state["S"][0] = 2.0 * A_
    for _ in range(15):
        lo.step(None)
        hi.step(None)
    return {"ok": per_a < per_b and lo._state["S"][0] < 0.8 * A_ and hi._state["S"][0] > 2.0 * A_,
            "catch_per_effort_crowded": per_a, "catch_per_effort_alone": per_b,
            "below_tipping_after_rest": lo._state["S"][0] / A_, "above_tipping_after_rest": hi._state["S"][0] / A_}


# ====================================================================== 8.9.5 collaborative discovery
def _test(ags, a, b):
    return _joint(ags, {"a": a, "b": b})


def oracle_discovery(env) -> dict:
    eng, st, c = inside(env)
    pairs = [(float(c["qualA"][i]) * float(c["qualB"][j]), i, j) for i in range(len(c["ingr_a"]))
             for j in range(len(c["ingr_b"])) if c["typeA"][i] == c["typeB"][j]]
    q, i, j = max(pairs)
    worth = float(c["unit_price"]) * float(c["base_yield"]) * q > float(c["try_cost"])

    def fn(e):
        return _test(e.agents(), c["ingr_a"][i], c["ingr_b"][j]) if worth else None
    return best_of(env, {"knows the best compatible pair": fn})


def _disc_explore(view, env, exploit=True):
    """Shared by the reference and the identification: assays, label exchange, joint tests of one representative
    pair per pair of classes (best purity first), then exploitation of the best yield found."""
    ags = view.agents()
    sp0, sp1 = view.action_space(ags[0]), view.action_space(ags[1])
    plan = sp0.kinds["coordinate"]["properties"]["plan"]["properties"]
    A_names, B_names = plan["a"]["enum"], plan["b"]["enum"]
    obs = {a: view.observe(a) for a in ags}
    labels = [{}, {}]
    pur = [{}, {}]
    for k in range(max(len(A_names), len(B_names))):
        acts = {}
        if k < len(A_names):
            acts[ags[0]] = {"kind": "query", "target": "assay", "args": {"ingredient": A_names[k]}}
        if k < len(B_names):
            acts[ags[1]] = {"kind": "query", "target": "assay", "args": {"ingredient": B_names[k]}}
        r = view.step(acts)
        for i, a in enumerate(ags):
            res = r["results"][a]
            if res.get("ok") and isinstance(res.get("result"), dict) and "class" in res["result"]:
                labels[i][res["result"]["ingredient"]] = res["result"]["class"]
            for kk, v in nums(r["observations"][a]).items():
                if kk.endswith(".purity"):
                    pur[i].setdefault(kk[:-7], []).append(v)
        obs = r["observations"]
    r = view.step({a: {"kind": "message", "to": ags[1 - i], "content": {"labels": labels[i],
                                                                         "purity": {k: float(np.mean(v)) for k, v
                                                                                    in pur[i].items()}}}
                   for i, a in enumerate(ags)})
    obs = r["observations"]
    other = [{}, {}]
    for i, a in enumerate(ags):
        for m in obs[a]["entities"]:
            if m.get("kind") == "message" and (m.get("content") or {}).get("labels"):
                other[i] = m["content"]
    # both participants now hold the same information; each computes the same plan
    la, lb = labels[0], other[0].get("labels", {})
    pa = {k: float(np.mean(v)) for k, v in pur[0].items()}
    pb = other[0].get("purity", {})
    ca, cb = sorted(set(la.values())), sorted(set(lb.values()))
    match: dict = {}
    tried: dict = {}
    for x in ca:
        for y in cb:
            if x in match or y in match.values() or view.done:
                continue
            ia = max((n for n in A_names if la.get(n) == x), key=lambda n: pa.get(n, 1.0))
            ib = max((n for n in B_names if lb.get(n) == y), key=lambda n: pb.get(n, 1.0))
            r = view.step(_test(ags, ia, ib))
            res = r["results"][ags[0]]
            yv = res["result"].get("yield", 0.0) if res.get("ok") and res["result"].get("agreed") else 0.0
            tried[(ia, ib)] = yv
            if yv > 18.0:
                match[x] = y
        if len(match) == len(ca) - 1 and len(ca) == len(cb):
            rest_a = [x for x in ca if x not in match]
            rest_b = [y for y in cb if y not in match.values()]
            if rest_a and rest_b:
                match[rest_a[0]] = rest_b[0]
            break
    if not exploit:
        return match
    cands = sorted(((pa.get(ia, 1.0) * pb.get(ib, 1.0), ia, ib) for ia in A_names for ib in B_names
                    if la.get(ia) in match and match[la[ia]] == lb.get(ib)), reverse=True)
    for _, ia, ib in cands[:3]:
        if view.done or (ia, ib) in tried:
            continue
        r = view.step(_test(ags, ia, ib))
        res = r["results"][ags[0]]
        tried[(ia, ib)] = res["result"].get("yield", 0.0) if res.get("ok") and res["result"].get("agreed") else 0.0
    best = max(tried, key=tried.get) if tried else None
    while not view.done:
        if best is None or tried[best] * 2.0 <= 40.0:            # public price 2 and test cost 40
            view.step(None)
        else:
            view.step(_test(ags, *best))
    return match


def reference_discovery(view, env):
    """Public, per participant: assay one's own ingredients, exchange labels and purity readings by message, test
    one pair for each pair of classes (highest purities first) until the classes are matched, try the best
    candidates within matched classes, then run the best pair found every period."""
    _disc_explore(view, env, exploit=True)


def identify_discovery(env) -> dict:
    """Question: which of the partner's class labels corresponds to each of your own?"""
    view = PublicView(env)
    est = _disc_explore(view, env, exploit=False)
    c = env._config
    truth = {c["labelA"][t]: c["labelB"][t] for t in range(len(c["labelA"]))}
    return {"question": "the matching between the two participants' class labels", "estimate": est, "truth": truth,
            "ok": est == truth}


def claim_discovery(env) -> dict:
    """Only compatible pairs yield: every pair is tested once on a fork; yields above the noise occur exactly for
    pairs of the same hidden type. No action of one participant alone earns anything."""
    eng, st, c = inside(env)
    ags = env.agents()
    wrong = 0
    for i, a in enumerate(c["ingr_a"]):
        for j, b in enumerate(c["ingr_b"]):
            e = fork(env)
            r = e.step(_test(ags, a, b))
            y = r["results"][ags[0]]["result"]["yield"]
            if (y > 18.0) != (c["typeA"][i] == c["typeB"][j]):
                wrong += 1
    solo = fork(env)
    rs = solo.step({ags[0]: _test(ags, c["ingr_a"][0], c["ingr_b"][0])[ags[0]]})
    return {"ok": wrong == 0 and not rs["results"][ags[0]]["result"].get("agreed"), "misclassified_pairs": wrong,
            "pairs": len(c["ingr_a"]) * len(c["ingr_b"])}


# ====================================================================== 8.9.6 team formation
def _cover_prob(team, show, skills, need):
    """P(every needed skill is held by a member who turns up), members independent."""
    import itertools
    tot = 0.0
    for pat in itertools.product((0, 1), repeat=len(team)):
        p = 1.0
        for w, z in zip(team, pat):
            p *= show[w] if z else 1.0 - show[w]
        if p == 0:
            continue
        ok = all(any(z and skills[w][k] for w, z in zip(team, pat)) for k in range(len(need)) if need[k])
        tot += p * ok
    return tot


def oracle_formation(env) -> dict:
    import itertools
    eng, st, c = inside(env)
    W = len(c["worker_names"])
    S = len(c["skill_names"])

    def pol(mult, clair):
        def fn(e):
            s = e._state
            need, val = s["task"][:S], s["task"][S]
            us = peek(e).random(W) if clair else None
            pay = [float(c["reservation"][w]) * mult for w in range(W)]
            best, bt = 0.0, None
            for k in (1, 2, 3):
                for team in itertools.combinations(range(W), k):
                    if sum(pay[w] for w in team) > float(c["budget"]):
                        continue
                    show = {w: float(c["reliability"][w]) * s["trust"][w] for w in team}
                    if clair:
                        show = {w: 1.0 if us[w] < show[w] else 0.0 for w in team}
                    pc = _cover_prob(team, show, c["skills"], need)
                    v = val * pc - sum(pay[w] * show[w] for w in team)
                    if v > best:
                        best, bt = v, team
            if bt is None:
                return None
            return {"kind": "allocate", "amounts": {c["worker_names"][w]: pay[w] for w in bt}}
        return fn
    return best_of(env, {f"best team (pay x{m}{', sees turnout' if cl else ''})": pol(m, cl)
                         for m in (1.0, 1.15, 1.32) for cl in (False, True)})


def reference_formation(view, env):
    """Public: start from the workers' claimed skills and a guess of their pay; raise the offer to a worker who
    declines; estimate turnout from experience; for each task pick the cheapest team whose believed coverage of the
    needed skills is high; revise skill beliefs when tasks fail with everyone present."""
    sp = view.action_space()
    workers = _targets(sp)
    budget = float(sp.limits["allocate_total"])
    obs = view.observe()
    skills = list(env._config["skill_names"])                 # public: named in the brief
    claim = {}
    need = [float(x) for x in obs["arrays"].get("first_task_needs", [1.0] * len(skills))]
    val = float((obs["arrays"].get("first_task_value") or [150.0])[0])
    belief: dict = {}
    res_hat = {w: 26.0 for w in workers}
    turn = {w: [3.0, 4.0] for w in workers}                    # turnout successes, trials (prior)
    import itertools
    team: tuple = ()
    while not view.done:
        if not belief and obs["entities"]:
            for e in obs["entities"]:
                if e.get("kind") == "worker":
                    claim[e["id"]] = set(e["claimed_skills"])
                    belief[e["id"]] = [0.85 if sk in claim[e["id"]] else 0.08 for sk in skills]
        if not belief:
            belief = {w: [0.5] * len(skills) for w in workers}
        best, bt = 0.0, ()
        for k in (1, 2, 3):
            for tm in itertools.combinations(workers, k):
                pay = sum(res_hat[w] * 1.25 for w in tm)
                if pay > budget:
                    continue
                show = {w: turn[w][0] / turn[w][1] for w in tm}
                pc = 1.0
                for j, nd in enumerate(need):
                    if nd:
                        miss = 1.0
                        for w in tm:
                            miss *= 1.0 - show[w] * belief[w][j]
                        pc *= 1.0 - miss
                v = val * pc - sum(res_hat[w] * 1.25 * show[w] for w in tm)
                if v > best:
                    best, bt = v, tm
        team = bt
        act = {"kind": "allocate", "amounts": {w: res_hat[w] * 1.25 for w in team}} if team else None
        r = view.step(act)
        obs = r["observation"]
        x = nums(obs)
        ents = {e["id"]: e for e in obs["entities"] if e.get("kind") == "worker"}
        for w in team:
            e = ents.get(w, {})
            if not e.get("accepted"):
                res_hat[w] += 6.0
                continue
            turn[w][1] += 1
            turn[w][0] += 1 if e.get("turned_up") else 0
        present = [w for w in team if ents.get(w, {}).get("turned_up")]
        for j, nd in enumerate(need):
            if not nd:
                continue
            holders = [w for w in present if belief[w][j] > 0.3]
            if x.get("task_completed") == 1.0 and len(holders) == 1:
                belief[holders[0]][j] = 1.0
            if x.get("task_completed") == 0.0 and len(holders) == 1 and len(present) == len(team):
                belief[holders[0]][j] *= 0.3
        need = [x.get(f"next_needs.{sk}", 0.0) for sk in skills]
        val = x.get("next_task_value", val)


def identify_formation(env) -> dict:
    """Question: which worker turns up least reliably? Invite two workers at a time at a generous pay, in
    rotation, and compare turnout rates."""
    view = PublicView(env)
    workers = _targets(view.action_space())
    budget = float(view.action_space().limits["allocate_total"])
    pay = min(46.0, budget / 2)
    up = {w: [0, 0] for w in workers}
    k = 0
    while not view.done and view.t < 40:
        pair = [workers[(2 * k) % len(workers)], workers[(2 * k + 1) % len(workers)]]
        r = view.step({"kind": "allocate", "amounts": {w: pay for w in pair}})
        for e in r["observation"]["entities"]:
            if e.get("kind") == "worker" and e["id"] in pair and e["accepted"]:
                up[e["id"]][1] += 1
                up[e["id"]][0] += int(e["turned_up"])
        k += 1
    est = min(workers, key=lambda w: up[w][0] / max(1, up[w][1]))
    truth = env._config["worker_names"][int(np.argmin(env._config["reliability"]))]
    return {"question": "the least reliable worker", "estimate": est, "truth": truth, "ok": est == truth}


def claim_formation(env) -> dict:
    """Complementary skills: a team covering every needed skill completes tasks, the same team without the only
    holder of one needed skill never does (tasks needing every skill, twelve noise replicates); paying at the
    reservation level erodes a worker's turnout, generous pay does not."""
    eng, st, c = inside(env)
    W, S = len(c["worker_names"]), len(c["skill_names"])
    team = []
    for k in range(S):
        if not any(c["skills"][w][k] for w in team):
            team.append(next(w for w in range(W) if c["skills"][w][k]))
    gap_skill = 0
    holder = next(w for w in team if c["skills"][w][gap_skill])
    weak = [w for w in team if w != holder and not c["skills"][w][gap_skill]] or \
        [w for w in range(W) if not c["skills"][w][gap_skill]][:1]
    full, cut = 0, 0
    for r in range(12):
        for lab, tm in (("full", team), ("cut", weak)):
            e = replicate(env, r)
            e._state["task"] = [1] * S + [300.0]
            e._state["trust"] = [1.0] * W
            e.step({"kind": "allocate", "amounts": {c["worker_names"][w]: float(c["reservation"][w]) * 1.3
                                                    for w in tm}})
            done = int(e._state["done"])
            if lab == "full":
                full += done
            else:
                cut += done
    lo, hi = fork(env), fork(env)
    w0 = 0
    for _ in range(10):
        lo.step({"kind": "allocate", "amounts": {c["worker_names"][w0]: float(c["reservation"][w0]) * 1.0}})
        hi.step({"kind": "allocate", "amounts": {c["worker_names"][w0]: float(c["reservation"][w0]) * 1.4}})
    return {"ok": full > 0 and cut == 0 and lo._state["trust"][w0] < hi._state["trust"][w0],
            "completed_full_team": full, "completed_without_skill_holder": cut,
            "trust_at_reservation_pay": lo._state["trust"][w0], "trust_at_generous_pay": hi._state["trust"][w0]}
