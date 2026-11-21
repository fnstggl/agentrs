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
        if caps.is_delegated(st, m["org"], plan["machine"]) and plan.get("respect_delegation", True) and \
                cfg.get("delegated_ops", False):
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
        target = max(plan.get("min_stock", 50), (sold + missed + consumed) * plan.get("target_days", 3))
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
        if caps.is_delegated(st, m["org"], mach_id):
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
        if caps.is_delegated(st, m["org"], ent["fac"]):
            continue
        sku = ent["sku"]
        have = ind.qty(st, ent["fac"], sku)
        transit = sum(s["qty"] for s in st.find("shipments", "payer", m["org"]) if s["status"] == "in_transit"
                      and s["to"] == ent["fac"] and s["sku"] == sku) if "shipments" in st.tables else 0
        transit += sum(s["qty"] for _, s in st.scan("shipments") if s["status"] == "in_transit" and s["to"] == ent["fac"]
                       and s["sku"] == sku and s["payer"] != m["org"]) if "shipments" in st.tables else 0
        if have + transit >= ent["reorder"]:
            continue
        all_offers = sorted((o["price"], oid, o) for oid, o in st.scan("offers")
                            if o["sku"] == sku and o["status"] == "active" and o["seller"] != m["org"])
        offers = [x for x in all_offers if ind.qty(st, x[2]["fac"], x[2]["sku"]) > 0]
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


def _substitute(soc, ctx, mid, m, ent, ind):
    """Input substitution: rewrite the designs that consume `ent['sku']` to use an equivalent available item."""
    st = ctx.state
    eq = _equivalent(ind, st, ent["sku"])
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
    if changed:
        inputs = list(m["routines"]["procurement"]["inputs"]) + [dict(ent, sku=o["sku"])]
        routines = dict(m["routines"], production=dict(m["routines"]["production"], plans=plans),
                        procurement=dict(m["routines"]["procurement"], inputs=inputs))
        soc.set(ctx, mid, routines=routines)
        soc.remember(ctx, mid, m, "substitute", o["seller"], {"from": ent["sku"], "to": o["sku"]})


def unit_cost(ind, st, d, mach_model) -> float:
    mdl = ind.phys["models"][mach_model]
    c = mdl["energy_kwh"] * 10.0 + mdl["labor_h_per_h"] * 1200.0 / ind.throughput(mdl, d["speed"]) + 15
    for x in d["inputs"]:
        prices = [o["price"] for _, o in st.scan("offers") if o["sku"] == x["sku"] and o["status"] == "active"]
        c += x["qty"] * (min(prices) if prices else 100)
    return c


def pricing_routine_b2b(soc, ctx, mid, m, rng):
    st = ctx.state
    ind = ctx.world.modules["industry"]
    caps = ctx.world.modules["caps"]
    cfg = m["routines"].get("b2b_pricing", {})
    for oid in sorted(st.find_ids("offers", "seller", m["org"])):
        o = st.tables["offers"][oid]
        if o["status"] != "active" or caps.is_delegated(st, m["org"], o["fac"]) and cfg.get("delegable"):
            continue
        have = ind.qty(st, o["fac"], o["sku"])
        sold = sum(t["qty"] for t in st.find("trade_orders", "offer", oid) if t["t"] >= ctx.now - 3 * DAY)
        floor = cfg.get("floor", {}).get(o["sku"], 1)
        p = o["price"]
        if have == 0 and sold > 0:
            p = int(math.ceil(p * 1.05))
        elif sold == 0 and have > cfg.get("high_stock", 300):
            p = max(floor, int(p * 0.96))
        if p != o["price"]:
            soc.act(ctx, m["owner"], "make.offer", {"facility": o["fac"], "sku": o["sku"], "price_cents": p})


def pricing_routine_retail(soc, ctx, mid, m, rng):
    st = ctx.state
    ind = ctx.world.modules["industry"]
    caps = ctx.world.modules["caps"]
    cfg = m["routines"].get("retail_pricing", {})
    for rid in sorted(st.find_ids("retail", "seller", m["org"])):
        r = st.tables["retail"][rid]
        if r["status"] != "active" or caps.is_delegated(st, m["org"], rid):
            continue
        have = ind.qty(st, r["fac"], r["sku"])
        floor = cfg.get("floor", 1)
        p = r["price"]
        if have == 0 and r.get("sold_y", 0) > 0:
            p = int(math.ceil(p * 1.04))
        elif r.get("sold_y", 0) == 0 and have > cfg.get("high_stock", 200):
            p = max(floor, int(p * 0.95))
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


def register_producer_routines(world) -> None:
    rt = world.hooks.setdefault("mind_routines", {})
    for k, f in [("maintenance", maintenance_routine), ("procurement", procurement_routine),
                 ("production", production_routine), ("b2b_pricing", pricing_routine_b2b),
                 ("retail_pricing", pricing_routine_retail), ("operators", operators_routine),
                 ("grid", grid_routine)]:
        rt.setdefault(k, f)
