"""Genesis of an industrial open world (module `industrygen`).

Writes an initial production society through ordinary internal APIs and events: a machine vendor, a grid utility,
a freight carrier, extraction firms on finite sites, component makers that form materials into components, and
assemblers that combine components into products sold to households. Every firm is run by a Tier-2 mind with the
default routines of `crucible.actors.producers`; their process designs start at arbitrary (not optimised)
operating points. The supply chain is configured as a dependency graph so that a failure upstream propagates:

    mine_k --material--> forge_j --component--> assem_i --product--> households

Nothing here scripts what happens next.
"""
from __future__ import annotations

import numpy as np

from ..core.kernel import Module, register_module

DAY = 86400


@register_module
class IndustryGen(Module):
    name = "industrygen"
    depends = ("identity", "bank", "business", "industry", "freight", "society")

    def genesis(self, ctx):
        cfg = self.config
        w = ctx.world
        idm, bank, biz, ind, soc = (w.modules[k] for k in ("identity", "bank", "business", "industry", "society"))
        rng = w.rng_for("industrygen")
        ph = ind.phys

        def firm(handle, name, sector, cash):
            owner = idm.create_person(ctx, f"{handle}-ceo", f"{name} CEO", "npc", {"role": "owner"})
            bank.open_account(ctx, owner, "checking")
            oid = idm.create_org(ctx, handle, name, owner, "company", {})
            biz.register_company(ctx, oid, owner, sector, fee=0)
            acct = bank.open_account(ctx, oid, "operating")
            bank.move(ctx, "acct_genesis", acct, int(cash), "endowment", f"initial capital of {oid}")
            return oid, owner, acct

        regions = sorted(ctx.state.tables["regions"])
        # infrastructure
        vend, vowner, _ = firm("machina", "Machina Equipment", "equipment", 2_000_000)
        ctx.emit("make.vendors", {"vendors": {mid: vend for mid in ph["models"]}}, vis="internal")
        grid, gowner, _ = firm("gridco", "GridCo", "energy", 20_000_000)
        caps_kwh = cfg.get("grid_kwh", [6000, 6000, 6000, 6000])
        for i, r in enumerate(regions):
            ind.add_grid(ctx, r, grid, int(caps_kwh[i % len(caps_kwh)]), float(cfg.get("kwh_price", 10.0)))
        soc.add_mind(ctx, grid, "firm", grid, gowner, soc.draw_traits(rng), role="utility",
                     extra={"routines": {"grid": {"lead_days": int(cfg.get("grid_lead_days", 10)),
                                                  "expand": bool(cfg.get("grid_expand", True))}}})
        hauler, howner, _ = firm("haulco", "HaulCo", "freight", 10_000_000)
        for lk in cfg.get("links", []):
            lk["owner"] = hauler
        mats = [m["sku"] for m in ph["materials"]]
        chain = cfg.get("chain") or default_chain(len(regions))
        designs: dict[str, str] = {}
        facs: dict[str, str] = {}
        owners: dict[str, tuple] = {}
        for f in chain:
            oid, owner, acct = firm(f["handle"], f["name"], f["role"], f.get("cash", 3_000_000))
            owners[f["handle"]] = (oid, owner, acct)
            fac = ind.add_facility(ctx, oid, regions[f["region"] % len(regions)], f["name"])
            facs[f["handle"]] = fac
            ctx.emit("make.own_operators", {"org": oid, "n": int(f.get("operators", 3))}, vis="internal")
            sp, tp = (round(float(rng.uniform(0.15, 0.95)), 2) for _ in range(2))
            mdl = f"{f['op']}-{'AB'[int(rng.integers(0, 2))]}"
            mach = ind.add_machine(ctx, oid, fac, mdl, wear=float(rng.uniform(0.0, 0.2)))
            for _ in range(int(f.get("machines", 1)) - 1):
                ind.add_machine(ctx, oid, fac, mdl, wear=float(rng.uniform(0.0, 0.2)))
            routines = {"maintenance": {"every_h": int(f.get("maint_every_h", rng.integers(50, 130)))}, "operators": {},
                        "forum_reader": {}, "process_learning": {}}
            if f["op"] == "extract":
                site = ind.add_site(ctx, oid, regions[f["region"] % len(regions)], mats[f["material"]],
                                    int(f.get("reserve", 400_000)), int(f.get("rate", 600)))
                did = ctx.call("make.design", {"org": oid, "name": f"{f['name']} extraction", "op": "extract",
                                               "site": site, "speed": sp, "temp": tp}, as_principal=owner)["design"]
                out_sku = mats[f["material"]]
            elif f["op"] == "form":
                in_sku = mats[f["material"]]
                did = ctx.call("make.design", {"org": oid, "name": f["product_name"], "op": "form",
                                               "inputs": [{"sku": in_sku, "qty": int(f.get("in_qty", 2))}],
                                               "form": f["form"], "speed": sp, "temp": tp}, as_principal=owner)["design"]
                out_sku = f"i:{did}"
                ind.add_stock(ctx, fac, in_sku, int(f.get("initial_input", 600)), ph["materials"][f["material"]]["props"])
                routines["procurement"] = {"inputs": [{"sku": in_sku, "fac": fac, "reorder": int(f.get("reorder", 300)),
                                                       "qty": int(f.get("order_qty", 600))}]}
            else:   # assemble
                ins = [{"sku": f"i:{designs[h]}", "qty": 1} for h in f["components"]]
                did = ctx.call("make.design", {"org": oid, "name": f["product_name"], "op": "assemble", "inputs": ins,
                                               "speed": sp, "temp": tp}, as_principal=owner)["design"]
                out_sku = f"i:{did}"
                routines["procurement"] = {"inputs": [{"sku": x["sku"], "fac": fac, "reorder": int(f.get("reorder", 120)),
                                                       "qty": int(f.get("order_qty", 240))} for x in ins]}
            designs[f["handle"]] = did
            routines["production"] = {"plans": [{"machine": mach, "design": did, "min_stock": int(f.get("min_stock", 120)),
                                                 "target_days": 3, "hours_per_day": 16}]}
            if f["op"] in ("extract", "form"):
                routines["b2b_pricing"] = {"floor": {out_sku: int(f["price"] * 0.6)}}
            else:
                routines["retail_pricing"] = {"floor": int(f["price"] * 0.5)}
            soc.add_mind(ctx, oid, "firm", oid, owner, soc.draw_traits(rng), role=f["role"],
                         extra={"routines": routines})
            f["_out"] = out_sku
            f["_mach"] = mach
        # bootstrap: initial output stock and the offers/listings (through the firms' own tools)
        for f in chain:
            oid, owner, acct = owners[f["handle"]]
            fac = facs[f["handle"]]
            if f["op"] == "assemble":
                for h in f["components"]:
                    up = next(x for x in chain if x["handle"] == h)
                    stock = ind.stock(ctx.state, facs[h], up["_out"])
                    v = list(stock["v"]) if stock else [0.5] * 4
                    ind.add_stock(ctx, fac, up["_out"], int(f.get("initial_input", 200)), v)
                ctx.call("make.sell", {"facility": fac, "sku": f["_out"], "price_cents": int(f["price"]),
                                       "title": f["product_name"]}, as_principal=owner)
            else:
                d = ctx.state.tables["designs"][designs[f["handle"]]]
                v = _bootstrap_vector(ind, ctx.state, d, rng)
                ind.add_stock(ctx, fac, f["_out"], int(f.get("initial_stock", 300)), v)
                ctx.call("make.offer", {"facility": fac, "sku": f["_out"], "price_cents": int(f["price"])},
                         as_principal=owner)
        ctx.emit("note", {"what": "industry genesis", "designs": designs, "facilities": facs}, vis="internal")


def _bootstrap_vector(ind, st, d, rng):
    """The vector of a firm's initial stock: what its own design would have produced (hidden physics)."""
    if d["op"] == "extract":
        return ind.output_vector(st, d, [], 1.0, rng)
    ph = ind.phys
    inputs = []
    for x in d["inputs"]:
        m = next(mm for mm in ph["materials"] if mm["sku"] == x["sku"])
        inputs.append((x["sku"], x["qty"], m["props"], x["sku"], None))
    return ind.output_vector(st, d, inputs, 0.8, rng)


def default_chain(n_regions: int) -> list[dict]:
    """Two mines, two component makers (one of them supplies both assemblers), two assemblers."""
    return [
        {"handle": "mine1", "name": "North Mining", "role": "mining", "op": "extract", "region": 1, "material": 0,
         "price": 70, "operators": 3, "rate": 700},
        {"handle": "mine2", "name": "South Mining", "role": "mining", "op": "extract", "region": 2, "material": 1,
         "price": 80, "operators": 3, "rate": 700},
        {"handle": "forge1", "name": "Plate Works", "role": "components", "op": "form", "region": 1, "material": 0,
         "form": "sheet", "product_name": "plate", "price": 320, "operators": 6, "machines": 2},
        {"handle": "forge2", "name": "Rod Works", "role": "components", "op": "form", "region": 2, "material": 1,
         "form": "rod", "product_name": "rod", "price": 300, "operators": 4},
        {"handle": "assem1", "name": "Homeware", "role": "products", "op": "assemble", "region": 0,
         "components": ["forge1", "forge2"], "product_name": "household unit", "price": 1600, "operators": 4},
        {"handle": "assem2", "name": "Gearsmith", "role": "products", "op": "assemble", "region": 3,
         "components": ["forge1", "forge1"][:1] + ["forge2"], "product_name": "gear kit", "price": 1500,
         "operators": 4},
    ]
