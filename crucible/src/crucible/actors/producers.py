"""Default operating routines of producer minds (registered into `society` as mind routines).

These are deliberately ordinary, bounded-rational business rules - they make the economy run without any learning
agent and leave real room for improvement (better process parameters, predictive maintenance, smarter
procurement, better products):

* `production`   run each plan's design when output stock is below a target of N days of recent sales; shrink a
                 run that the grid or the staff cannot support.
* `maintenance`  repair broken machines at once; maintain each machine every fixed number of run hours.
* `procurement`  reorder inputs from the cheapest offer when stock falls below a reorder point; if the usual item
                 is unavailable, substitute an equivalent one (same stage and form, measured vector within a
                 tolerance) by writing a variant design - input substitution.
* `b2b_pricing` / `retail_pricing`  cost-plus with adjustments for unsold stock and stockouts.
* `operators`    hire operators when labour binds, let one go after a week of idle capacity.
* `grid` / `carrier`  infrastructure owners expand capacity (with a lead time) when utilisation stays high and
                 adjust prices with utilisation.

Every routine stands down on resources delegated to an installed capability (`caps.is_delegated`).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.errors import CrucibleError

DAY = 86400


def _sales(st, org, sku, days, today):
    """Units of `sku` the org sold (retail + business trades) over the last `days` days."""
    n = 0
    for rid in st.find_ids("retail", "seller", org):
        r = st.tables["retail"][rid]
        if r["sku"] == sku:
            n += r.get("sold_total", 0) if days >= 9999 else r.get("sold_y", 0) * days
    for tid in st.find_ids("trade_orders", "seller", org):
        t = st.tables["trade_orders"][tid]
        if t["sku"] == sku and t["t"] >= (today - days) * DAY:
            n += t["qty"]
    return n


def production_routine(soc, ctx, mid, m, rng):
    st = ctx.state
    ind = ctx.world.modules["industry"]
    caps = ctx.world.modules["caps"]
    today = ctx.now // DAY
    cfg = m["routines"].get("production", {})
    cfg = _invest(soc, ctx, mid, m, cfg, ind, today)
    for plan in cfg.get("plans", []):
        mach = st.get("machines", plan["machine"])
        if mach is None or mach["status"] != "ok" or mach["busy_until"] > ctx.now + 3600:
            continue
        if caps.is_delegated(st, m["org"], plan["machine"], "make.operate"):
            continue
        did = plan["design"]
        d = st.get("designs", did)
        if d is None:
            continue
        out_sku = d["material"] if d["op"] == "extract" else f"i:{did}"
        have = ind.qty(st, mach["facility"], out_sku)
        sold = _sales(st, m["org"], out_sku, 3, today) / 3.0
        missed = sum(sum(v for d_, v in o.get("missed", {}).items() if int(d_) >= today - 2)
                     for o in st.find("offers", "seller", m["org"]) if o["sku"] == out_sku) / 3.0
        consumed = plan.get("downstream_per_day", 0)
        target = max(plan.get("min_stock", 50) if sold + missed == 0 else 20,
                     (sold + missed + consumed) * plan.get("target_days", 3))
        short = int(target - have)
        if short <= 0:
            continue
        mdl = ind.phys["models"][mach["model"]]
        units = min(short, int(ind.throughput(mdl, d["speed"]) * plan.get("hours_per_day", 16)))
        for x in d["inputs"]:
            units = min(units, ind.qty(st, mach["facility"], x["sku"]) // x["qty"])
        if d["op"] == "extract":
            site = st.tables["sites"][d["site"]]
            units = min(units, site["reserve"], site["rate"])
        while units >= 1:
            try:
                ctx.call("make.run", {"machine": plan["machine"], "design": did, "units": int(units)},
                         as_principal=m["owner"])
                break
            except CrucibleError as e:
                if e.code in ("capacity_exceeded", "insufficient_funds") or "capacity" in e.message:
                    units //= 2
                    continue
                ctx.emit("society.failed", {"who": m["owner"], "tool": "make.run", "error": e.as_dict()},
                         vis="internal")
                break


def _invest(soc, ctx, mid, m, cfg, ind, today):
    """Capacity: adopt newly delivered idle machines into the plan of the design they can run; after 5 days in
    which recent sales plus unfilled orders exceeded what the plan's machines can make, order another machine."""
    st = ctx.state
    plans = list(cfg.get("plans", []))
    planned = {p["machine"] for p in plans}
    changed = False
    for mach_id in sorted(st.find_ids("machines", "owner", m["org"])):
        if mach_id in planned:
            continue
        mach = st.tables["machines"][mach_id]
        op = ind.phys["models"][mach["model"]]["op"]
        for p in list(plans):
            d = st.get("designs", p["design"])
            if d and d["op"] == op and st.tables["machines"][p["machine"]]["facility"] == mach["facility"]:
                plans.append(dict(p, machine=mach_id))
                changed = True
                break
    pressure = dict(cfg.get("pressure", {}))
    for p in plans[:1] if plans else []:
        d = st.get("designs", p["design"])
        out_sku = d["material"] if d["op"] == "extract" else f"i:{p['design']}"
        sold = _sales(st, m["org"], out_sku, 3, today) / 3.0
        missed = sum(sum(v for d_, v in o.get("missed", {}).items() if int(d_) >= today - 2)
                     for o in st.find("offers", "seller", m["org"]) if o["sku"] == out_sku) / 3.0
        stockouts = 0
        for k in range(1, 4):
            r = st.get("ops_days", f"{m['org']}|{today - k}")
            stockouts += (r or {}).get("loss_n", {}).get("stockout", 0) if r else 0
        mdl = ind.phys["models"][st.tables["machines"][p["machine"]]["model"]]
        n_m = sum(1 for q in plans if q["design"] == p["design"])
        cap = n_m * ind.throughput(mdl, d["speed"]) * 16 * 0.7
        key = p["design"]
        if sold + missed + stockouts / 3.0 > cap:
            pressure[key] = pressure.get(key, 0) + 1
        else:
            pressure[key] = 0
        bank = ctx.world.modules["bank"]
        acct = bank.primary_account(st, m["org"])
        cash = bank.balance(st, acct) if acct else 0
        pending = [o for o in st.find("machine_orders", "owner", m["org"]) if o["status"] == "ordered"]
        if pressure[key] >= 5 and not pending and cash > 2.5 * mdl["price"] and n_m < 4:
            fac = st.tables["machines"][p["machine"]]["facility"]
            if soc.act(ctx, m["owner"], "make.buy_machine", {"facility": fac, "model": st.tables["machines"][p["machine"]]["model"]}):
                pressure[key] = 0
                soc.remember(ctx, mid, m, "invest", None, {"design": key, "sold": round(sold, 1), "missed": round(missed, 1)})
                m = dict(st.tables["minds"][mid])
    if changed or pressure != cfg.get("pressure", {}):
        cfg = dict(cfg, plans=plans, pressure=pressure)
        soc.set(ctx, mid, routines=dict(m["routines"], production=cfg))
    return cfg


def maintenance_routine(soc, ctx, mid, m, rng):
    st = ctx.state
    caps = ctx.world.modules["caps"]
    every = m["routines"].get("maintenance", {}).get("every_h", 150)
    for mach_id in sorted(st.find_ids("machines", "owner", m["org"])):
        mach = st.tables["machines"][mach_id]
        if caps.is_delegated(st, m["org"], mach_id, "make.maintain"):
            continue
        if mach["status"] == "broken":
            soc.act(ctx, m["owner"], "make.repair", {"machine": mach_id})
        elif mach["status"] == "ok" and mach["busy_until"] <= ctx.now and mach["hours_since_maint"] >= every:
            soc.act(ctx, m["owner"], "make.maintain", {"machine": mach_id})


def _equivalent(ind, st, sku, tol=0.6):
    """Offers of items equivalent to `sku`: same stage and form, measured vector within `tol` (L1)."""
    info = ind.item_info(st, sku)
    ref = None
    for _, s in st.scan("stock"):
        if s["sku"] == sku and s["qty"] > 0:
            ref = np.array(s["v"])
            break
    out = []
    for oid, o in st.scan("offers"):
        if o["status"] != "active" or o["sku"] == sku:
            continue
        oi = ind.item_info(st, o["sku"])
        if oi["stage"] != info["stage"] or oi.get("form") != info.get("form"):
            continue
        s = ind.stock(st, o["fac"], o["sku"])
        if s is None or s["qty"] <= 0:
            continue
        if ref is not None and np.abs(np.array(s["v"]) - ref).sum() > tol:
            continue
        out.append((o["price"], oid, o))
    return sorted(out)


def procurement_routine(soc, ctx, mid, m, rng):
    st = ctx.state
    ind = ctx.world.modules["industry"]
    caps = ctx.world.modules["caps"]
    cfg = m["routines"].get("procurement", {})
    for ent in cfg.get("inputs", []):
        if caps.is_delegated(st, m["org"], ent["fac"], "make.trade"):
            continue
        sku = ent["sku"]
        have = ind.qty(st, ent["fac"], sku)
        transit = sum(s["qty"] for s in st.find("shipments", "payer", m["org"]) if s["status"] == "in_transit"
                      and s["to"] == ent["fac"] and s["sku"] == sku) if "shipments" in st.tables else 0
        transit += sum(s["qty"] for _, s in st.scan("shipments") if s["status"] == "in_transit" and s["to"] == ent["fac"]
                       and s["sku"] == sku and s["payer"] != m["org"]) if "shipments" in st.tables else 0
        # demand-driven reorder point and quantity: what our runs consumed over the last week
        used = 0
        for _, r in st.scan("runs"):
            if r["owner"] == m["org"] and r["t0"] >= ctx.now - 7 * DAY:
                used += sum(x[1] for x in r["inputs"] if x[0] == sku)
        per_day = used / 7.0
        reorder = max(ent.get("min_reorder", 40), min(ent["reorder"], per_day * 3)) if used else ent["reorder"]
        order_qty = int(max(ent.get("min_qty", 60), min(ent["qty"], per_day * 5))) if used else ent["qty"]
        ent = dict(ent, qty=order_qty)
        if have + transit >= reorder:
            continue
        cheaper = [x for x in _equivalent(ind, st, sku, tol=0.3) if x[2]["seller"] != m["org"]]
        cur = sorted(o["price"] for _, o in st.scan("offers") if o["sku"] == sku and o["status"] == "active")
        if cheaper and cur and cheaper[0][0] < 0.85 * cur[0] and cfg.get("substitute", True) and \
                ctx.now - ent.get("substituted_t", -10 * DAY) > 10 * DAY:
            _substitute(soc, ctx, mid, m, dict(ent, substituted_t=ctx.now), ind, replace=True)
            m = dict(st.tables["minds"][mid])
        all_offers = sorted((o["price"], oid, o) for oid, o in st.scan("offers")
                            if o["sku"] == sku and o["status"] == "active" and o["seller"] != m["org"])
        cap_price = _affordable_input_price(st, ind, m, sku)
        offers = [x for x in all_offers if ind.qty(st, x[2]["fac"], x[2]["sku"]) > 0
                  and (cap_price is None or x[0] <= cap_price)]
        trust = m["trust"]
        offers = sorted(offers, key=lambda x: (x[0] * (1.5 - soc.trust(m, x[2]["seller"])), x[1]))
        got = 0
        for price, oid, o in offers:
            q = min(ent["qty"] - got, ind.qty(st, o["fac"], o["sku"]))
            if q <= 0:
                continue
            r = soc.act(ctx, m["owner"], "make.order", {"offer": oid, "qty": int(q), "to_facility": ent["fac"]})
            if r:
                got += q
                m = soc.update_trust(ctx, mid, m, o["seller"], 0.2, 0.0)
                m = soc.add_contact(ctx, mid, m, o["seller"])          # trade partners talk to each other
            if got >= ent["qty"]:
                break
        del trust
        if got < ent["qty"] and all_offers:   # unmet demand is visible to the seller as an order-book signal
            ctx.emit("make.offer_missed", {"offer": all_offers[0][1], "qty": int(ent["qty"] - got),
                                           "day": ctx.now // DAY}, vis=(all_offers[0][2]["seller"],))
        if got == 0:
            ops = ctx.world.hooks.get("ops")
            if ops is not None:
                ops.record_loss(ctx, m["org"], "supply_shortfall", 0, int(ent["qty"]), resource=sku)
            for _, oid, o in offers[:1]:
                m = soc.update_trust(ctx, mid, m, o["seller"], 0.0, 0.5)
            if cfg.get("substitute", True):
                _substitute(soc, ctx, mid, m, ent, ind)


def _substitute(soc, ctx, mid, m, ent, ind, replace=False):
    """Input substitution: rewrite the designs that consume `ent['sku']` to use an equivalent available item
    (`replace`: drop the old input and plan - a switch for cost, not a fallback for a shortage)."""
    st = ctx.state
    eq = _equivalent(ind, st, ent["sku"], tol=0.3 if replace else 0.6)
    if not eq:
        return
    _, oid, o = eq[0]
    plans = list(m["routines"].get("production", {}).get("plans", []))
    changed = False
    for i, plan in enumerate(plans):
        d = st.get("designs", plan["design"])
        if d is None or not any(x["sku"] == ent["sku"] for x in d["inputs"]):
            continue
        new_inputs = [dict(x, sku=o["sku"]) if x["sku"] == ent["sku"] else dict(x) for x in d["inputs"]]
        r = soc.act(ctx, m["owner"], "make.design", {"org": m["org"], "name": d["name"] + " (substitute)", "op": d["op"],
                                                    "inputs": new_inputs, "form": d["form"] or "sheet",
                                                    "speed": d["speed"], "temp": d["temp"]})
        if r:
            plans.append(dict(plan, design=r["design"]))
            changed = True
            # sell the substitute product too (same listing price as the original)
            for rid in st.find_ids("retail", "seller", m["org"]):
                rt = st.tables["retail"][rid]
                if rt["sku"] == f"i:{plan['design']}" and rt["status"] == "active":
                    soc.act(ctx, m["owner"], "make.sell", {"facility": rt["fac"], "sku": r["sku"],
                                                          "price_cents": rt["price"], "title": rt["title"]})
    if changed and replace:
        old_designs = {p["design"] for p in plans if any(x["sku"] == ent["sku"] for x in
                                                         (st.get("designs", p["design"]) or {}).get("inputs", []))}
        plans = [p for p in plans if p["design"] not in old_designs]
    if changed:
        inputs = [x for x in m["routines"]["procurement"]["inputs"] if not (replace and x["sku"] == ent["sku"])] + \
            [dict(ent, sku=o["sku"])]
        routines = dict(m["routines"], production=dict(m["routines"]["production"], plans=plans),
                        procurement=dict(m["routines"]["procurement"], inputs=inputs))
        soc.set(ctx, mid, routines=routines)
        soc.remember(ctx, mid, m, "substitute", o["seller"], {"from": ent["sku"], "to": o["sku"]})


def _affordable_input_price(st, ind, m, sku) -> float | None:
    """The most this firm can pay per unit of an input and still sell its output at a margin: output price x
    measured yield, shared over the design's inputs, x 0.85. None if the firm has no priced output yet."""
    plans = m["routines"].get("production", {}).get("plans", [])
    for plan in plans:
        d = st.get("designs", plan["design"])
        if d is None or not any(x["sku"] == sku for x in d["inputs"]):
            continue
        out_sku = f"i:{plan['design']}"
        prices = [r["price"] for r in st.find("retail", "seller", m["org"]) if r["sku"] == out_sku and r["status"] == "active"]
        prices += [o["price"] for o in st.find("offers", "seller", m["org"]) if o["sku"] == out_sku and o["status"] == "active"]
        if not prices:
            return None
        runs = [r for _, r in st.scan("runs") if r["owner"] == m["org"] and r["design"] == plan["design"]
                and r["status"] == "done"][-8:]
        y = (sum(r["made"] for r in runs) / max(1, sum(r["units"] for r in runs))) if runs else 0.7
        n_units = sum(x["qty"] for x in d["inputs"])
        # the price the firm intends to charge: its current price or its cost-plus target, whichever is higher
        cost = unit_cost_estimate(ind, st, m["org"], plan["design"])
        markup = (m["routines"].get("retail_pricing") or m["routines"].get("b2b_pricing") or {}).get("markup") \
            or 1.4 + 0.4 * m["traits"]["risk"]
        ref = max(max(prices), (cost or 0) * markup)
        return 0.85 * ref * max(0.3, y) / max(1, n_units)
    return None


def unit_cost_estimate(ind, st, org, design_id) -> float | None:
    """What a good unit of a design costs the firm, from its own records: recent purchase prices of its inputs,
    its measured yield and energy per good unit on recent runs (at the current grid price), and labour."""
    d = st.get("designs", design_id)
    if d is None:
        return None
    runs = [r for _, r in st.scan("runs") if r["owner"] == org and r["design"] == design_id and r["status"] == "done"][-8:]
    made = sum(r["made"] for r in runs)
    units = sum(r["units"] for r in runs)
    if not made:
        return None
    y = made / units
    c = 0.0
    for x in d["inputs"]:
        # replacement cost: what the input costs on the market now (else what was last paid for it)
        paid = sorted(o["price"] for _, o in st.scan("offers") if o["sku"] == x["sku"] and o["status"] == "active")[:1]
        if not paid:
            paid = [t["price"] for _, t in st.scan("trade_orders") if t["buyer"] == org and t["sku"] == x["sku"]][-5:]
        if paid:
            c += x["qty"] * (sum(paid) / len(paid)) / y
    fac = st.tables["machines"][runs[-1]["machine"]]["facility"]
    region = st.tables["facilities"][fac]["region"]
    g = st.tables["regions"][region].get("grid")
    kwh = sum(r["kwh"] for r in runs) / made
    c += kwh * (g["base_price"] * 1.5 if g else 10.0)
    c += sum(r["labor_h"] for r in runs) / made * 1300.0
    return c


def unit_cost(ind, st, d, mach_model) -> float:
    mdl = ind.phys["models"][mach_model]
    c = mdl["energy_kwh"] * 10.0 + mdl["labor_h_per_h"] * 1200.0 / ind.throughput(mdl, d["speed"]) + 15
    for x in d["inputs"]:
        prices = [o["price"] for _, o in st.scan("offers") if o["sku"] == x["sku"] and o["status"] == "active"]
        c += x["qty"] * (min(prices) if prices else 100)
    return c


def _price_step(p, floor, cost, markup, have, per_day):
    """One day of cost-plus pricing with inventory feedback: move toward cost x markup; nudge up when stock covers
    less than a day of sales, down when it covers more than six; never below floor or above 2.5x the target."""
    target = cost * markup if cost else p
    cover = have / max(1e-9, per_day) if per_day > 0 else (99.0 if have > 0 else 0.0)
    if cover < 1 and per_day > 0:
        p = p * 1.03
    elif cover > 6:
        p = p * 0.97
    p = p + 0.1 * (target - p)
    return int(max(floor, min(p, 2.5 * target if target else p)))


def pricing_routine_b2b(soc, ctx, mid, m, rng):
    st = ctx.state
    ind = ctx.world.modules["industry"]
    caps = ctx.world.modules["caps"]
    cfg = m["routines"].get("b2b_pricing", {})
    markup = cfg.get("markup") or round(1.25 + 0.3 * m["traits"]["risk"], 3)
    for oid in sorted(st.find_ids("offers", "seller", m["org"])):
        o = st.tables["offers"][oid]
        if o["status"] != "active" or caps.is_delegated(st, m["org"], o["fac"], "make.trade"):
            continue
        have = ind.qty(st, o["fac"], o["sku"])
        sold = sum(t["qty"] for t in st.find("trade_orders", "offer", oid) if t["t"] >= ctx.now - 3 * DAY) / 3.0
        missed = sum(v for d_, v in (o.get("missed") or {}).items() if int(d_) >= ctx.now // DAY - 2) / 3.0
        cost = _unit_cost_any(ind, st, m, o["sku"])
        floor = max(cfg.get("floor", {}).get(o["sku"], 1) if not cost else 1, int(cost * 1.1) if cost else 1)
        p = _price_step(o["price"], floor, cost, markup, have, sold + missed)
        if p != o["price"]:
            soc.act(ctx, m["owner"], "make.offer", {"facility": o["fac"], "sku": o["sku"], "price_cents": p})


def _unit_cost_any(ind, st, m, sku):
    """Unit cost of an item the firm makes: designed items through their recent runs; extracted materials through
    the energy and labour of recent extraction runs."""
    if sku.startswith("i:"):
        return unit_cost_estimate(ind, st, m["org"], sku[2:])
    runs = [r for _, r in st.scan("runs") if r["owner"] == m["org"] and r["status"] == "done"
            and (st.get("designs", r["design"]) or {}).get("material") == sku][-8:]
    made = sum(r["made"] for r in runs)
    if not made:
        return None
    return (sum(r["kwh"] for r in runs) * 15.0 + sum(r["labor_h"] for r in runs) * 1300.0) / made


def pricing_routine_retail(soc, ctx, mid, m, rng):
    st = ctx.state
    ind = ctx.world.modules["industry"]
    caps = ctx.world.modules["caps"]
    cfg = m["routines"].get("retail_pricing", {})
    markup = cfg.get("markup") or round(1.4 + 0.4 * m["traits"]["risk"], 3)
    for rid in sorted(st.find_ids("retail", "seller", m["org"])):
        r = st.tables["retail"][rid]
        if r["status"] != "active" or caps.is_delegated(st, m["org"], rid, "market.manage"):
            continue
        have = ind.qty(st, r["fac"], r["sku"])
        stockouts = sum((st.get("ops_days", f"{m['org']}|{d}") or {"loss_n": {}})["loss_n"].get("stockout", 0)
                        for d in range(ctx.now // DAY - 2, ctx.now // DAY + 1)) / 3.0
        cost = _unit_cost_any(ind, st, m, r["sku"])
        floor = max(cfg.get("floor", 1) if not cost else 1, int(cost * 1.1) if cost else 1)
        p = _price_step(r["price"], floor, cost, markup, have, r.get("sold_y", 0) + stockouts)
        # competition: a close substitute (attributes within 0.3) that is cheaper pulls the price down
        s = ind.stock(st, r["fac"], r["sku"])
        if s is not None and s["v"]:
            for rid2, r2 in st.scan("retail"):
                if r2["seller"] == m["org"] or r2["status"] != "active" or r2["price"] >= p:
                    continue
                s2 = ind.stock(st, r2["fac"], r2["sku"])
                if s2 and s2["v"] and np.abs(np.array(s2["v"]) - np.array(s["v"])).sum() < 0.3 and s2["qty"] > 0:
                    p = max(floor, int(min(p, r2["price"] * 1.02)))
        if p != r["price"]:
            soc.act(ctx, m["owner"], "make.sell", {"facility": r["fac"], "sku": r["sku"], "price_cents": p})


def operators_routine(soc, ctx, mid, m, rng):
    st = ctx.state
    ind = ctx.world.modules["industry"]
    org = m["org"]
    day = ctx.now // DAY
    used = (st.get("labor_days", f"{org}|{day}") or {}).get("used", 0.0)
    cap = 8.0 * ind.operators(st, org)
    cfg = m["routines"].get("operators", {})
    idle = int(cfg.get("idle_days", 0))
    npcs = ctx.world.modules.get("npcs")
    if cap == 0 or used > 0.85 * cap:
        if npcs is not None:
            npcs._hire(ctx, org, m["owner"], "operator", int(cfg.get("wage", 220_000)))
        idle = 0
    elif used < 0.3 * cap:
        idle += 1
        hired = [p for p, w in st.scan("npc_workers") if w["employer"] == org and w["role"] == "operator"] \
            if "npc_workers" in st.tables else []
        if idle >= 7 and hired:
            lay = ctx.world.hooks.get("layoff")
            if lay:
                lay(ctx, org, hired[-1], "idle capacity")
            idle = 0
    else:
        idle = 0
    if idle != cfg.get("idle_days", 0):
        soc.set(ctx, mid, routines=dict(m["routines"], operators=dict(cfg, idle_days=idle)))


def grid_routine(soc, ctx, mid, m, rng):
    """Utility: expand a region's grid by 25% (lead time, capital cost) after 3 days above 85% utilisation."""
    st = ctx.state
    day = ctx.now // DAY
    cfg = m["routines"].get("grid", {})
    for rid, r in sorted(st.scan("regions")):
        g = r.get("grid")
        if not g or g["owner"] != m["org"]:
            continue
        us = [(st.get("grid_days", f"{rid}|{d}") or {}).get("used", 0) / max(1, g["capacity"]) for d in range(day - 2, day + 1)]
        pending = (st.meta.get("grid_pending") or {}).get(rid)
        if min(us) > 0.85 and not pending and cfg.get("expand", True):
            add = int(g["capacity"] * 0.25)
            ctx.emit("make.grid_order", {"region": rid, "add": add, "due": ctx.now + cfg.get("lead_days", 10) * DAY},
                     vis="public")
            ctx.schedule(ctx.now + cfg.get("lead_days", 10) * DAY, "industry.grid_expand", {"region": rid, "add": add},
                         prio=29)
            bank = ctx.world.modules["bank"]
            acct = bank.primary_account(st, m["org"])
            if acct and bank.balance(st, acct) > add * 50:
                bank.move(ctx, acct, "acct_external", add * 50, "capital", f"grid expansion {rid}")


def process_learning_routine(soc, ctx, mid, m, rng):
    """Every few days, look at published process knowledge for the machine models we run. Compare its verified
    evidence (yield, energy per good unit) with our own recent runs of that model; buy and apply it when the
    expected saving over 60 days beats its price (trust-weighted)."""
    st = ctx.state
    ind = ctx.world.modules["industry"]
    if soc.p("ablate_diffusion", False) or rng.random() > 0.35:
        return
    plans = m["routines"].get("production", {}).get("plans", [])
    done = set(m["routines"].get("process_learning", {}).get("applied", []))
    for plan in plans:
        mach = st.get("machines", plan["machine"])
        d = st.get("designs", plan["design"])
        if mach is None or d is None or d["owner"] != m["org"]:
            continue
        own = ind.run_evidence(st, m["org"], mach["model"], d["speed"], d["temp"], tol=0.02)
        if not own["runs"] or own["yield"] is None:
            continue
        for nid, n in sorted(st.scan("process_notes")):
            if n["model"] != mach["model"] or n["owner"] == m["org"] or nid in done:
                continue
            ev = n["evidence"]
            if not ev["yield"] or ev["runs"] < 1:
                continue
            units_day = max(1.0, own["units"] / max(1, own["runs"]))
            kwh_price = 12.0
            unit_val = 100.0
            saving = units_day * ((ev["yield"] - own["yield"]) * unit_val +
                                  max(0.0, (own["kwh_per_good_unit"] or 0) - (ev["kwh_per_good_unit"] or 0))
                                  * kwh_price * own["yield"])
            tr = soc.trust(m, n["owner"])
            if saving * 60 * (0.5 + tr) <= n["price"] or saving <= 0:
                continue
            r = soc.act(ctx, m["owner"], "make.buy_process", {"note": nid, "org": m["org"]})
            if r:
                soc.act(ctx, m["owner"], "make.tune", {"design": plan["design"], "speed": r["speed"], "temp": r["temp"]})
                done.add(nid)
                cfg = dict(m["routines"].get("process_learning", {}), applied=sorted(done))
                soc.set(ctx, mid, routines=dict(m["routines"], process_learning=cfg))
                soc.remember(ctx, mid, m, "learned_process", n["owner"], {"note": nid, "expected_saving_day": round(saving, 1)})
                m = dict(st.tables["minds"][mid])
                break


def register_producer_routines(world) -> None:
    rt = world.hooks.setdefault("mind_routines", {})
    for k, f in [("maintenance", maintenance_routine), ("procurement", procurement_routine),
                 ("production", production_routine), ("b2b_pricing", pricing_routine_b2b),
                 ("retail_pricing", pricing_routine_retail), ("operators", operators_routine),
                 ("grid", grid_routine), ("process_learning", process_learning_routine)]:
        rt.setdefault(k, f)
