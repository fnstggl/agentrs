"""Production economy: materials, machines, process designs, production runs, energy and freight
(modules `industry` - host `make` - and `freight` - host `freight`).

Represented physics (finite mechanisms, combinatorial possibilities):

* **Materials** have a hidden property vector (strength, conductivity, lightness, purity) in [0, 1]. They are
  extracted at finite sites (reserves deplete) with extraction machines.
* **Process designs** are compositional specifications written by anyone: one operation applied to inputs with
  process parameters. Operations: `extract` (site -> material), `refine` (material -> purer material), `blend`
  (2-4 materials in ratios -> new material; hidden pairwise interactions), `form` (material -> component of one of
  four form types; a hidden nonlinear map from properties to attributes), `assemble` (2-5 components -> product;
  hidden synergies between form types). Items carry a 4-d attribute vector (durability, performance, efficiency,
  appeal) once formed. There is no recipe list and no technology tree: any design that type-checks can be run;
  what it yields is decided by the hidden physics and only learned by producing and measuring.
* **Machines** (bought from equipment vendors with lead times) have a hidden optimal operating point per model:
  output quality and yield fall off with distance of (speed, temperature) from it; energy per unit grows with
  temperature and speed; throughput grows with speed. Machines wear with use (faster with speed), break down with a
  hazard that rises with wear, and are restored by maintenance. Telemetry is a noisy view of wear.
* **Conservation**: runs consume their inputs when they start; outputs never exceed inputs times yield; defective
  units are destroyed; all quantities are integers. Energy comes from finite regional grids whose price rises with
  utilisation and which refuse runs when full. Labour comes from employed operators (8 h/day each).
* **Freight** moves goods between regions along a network with finite daily capacity per link; congestion delays
  shipments to the next day and raises prices.
* **Trade**: business offers (`make.offer`/`make.order`), standing orders (recurring procurement), and retail
  listings that the household population buys from (`crucible.economy.shoppers`).

Not represented: geometry, chemistry beyond the 4-d property model, thermodynamics, spatial layout inside a
facility, transport modes, perishability (unless a world enables `decay`), environmental externalities.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import CapacityExceeded, Conflict, CrucibleError, Forbidden, InsufficientFunds, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ..services.identity import ident

DAY = 86400
OPS = ("extract", "refine", "blend", "form", "assemble")
FORMS = ("sheet", "rod", "fiber", "chip")
PROPS = ("strength", "conductivity", "lightness", "purity")
ATTRS = ("durability", "performance", "efficiency", "appeal")
SYL = ["ka", "lo", "ri", "ven", "tor", "mi", "sa", "dun", "el", "qua", "zi", "bor", "nek", "ul", "fa", "ret"]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def gen_physics(rng, n_materials: int, n_regions: int, cfg: dict) -> dict:
    """Hidden physics of one world (never exposed through any endpoint)."""
    mats = []
    used = set()
    for i in range(n_materials):
        while True:
            name = "".join(rng.choice(SYL, size=2)) + "ite"
            if name not in used:
                used.add(name)
                break
        mats.append({"sku": f"m:{name}", "props": [round(float(x), 3) for x in rng.beta(2, 2, size=4)],
                     "base_cost": int(rng.integers(20, 80))})
    H = rng.normal(0, 0.12, size=(n_materials, n_materials, 4))
    H = (H + H.transpose(1, 0, 2)) / 2
    forms = {f: {"W": rng.normal(0, 1.2, size=(4, 4)).round(3).tolist(), "b": rng.normal(0, 0.4, size=4).round(3).tolist()}
             for f in FORMS}
    Ssyn = rng.normal(0, 0.08, size=(len(FORMS), len(FORMS), 4))
    Ssyn = (Ssyn + Ssyn.transpose(1, 0, 2)) / 2
    models = {}
    for op in OPS:
        for mdl in ("A", "B"):
            mid = f"{op}-{mdl}"
            models[mid] = {
                "op": op, "cap_per_hour": int(rng.integers(8, 25)) * (2 if op == "extract" else 1),
                "energy_kwh": round(float(rng.uniform(0.5, 2.5)) * (1.6 if op in ("refine", "blend") else 1.0), 3),
                "labor_h_per_h": round(float(rng.uniform(0.3, 1.0)), 2),
                "price": int(rng.integers(150_000, 600_000)), "lead_days": int(rng.integers(3, 12)),
                "wear_rate": round(float(rng.uniform(0.004, 0.012)), 4), "h0": round(float(rng.uniform(0.001, 0.004)), 4),
                "repair_cost": int(rng.integers(20_000, 80_000)), "repair_h": int(rng.integers(12, 48)),
                "maint_cost": int(rng.integers(3_000, 12_000)),
                # hidden operating optimum and tolerance
                "s_opt": round(float(rng.uniform(0.25, 0.85)), 3), "t_opt": round(float(rng.uniform(0.2, 0.8)), 3),
                "width": round(float(rng.uniform(0.22, 0.4)), 3), "y_max": round(float(rng.uniform(0.86, 0.97)), 3)}
    return {"materials": mats, "H": H.round(4).tolist(), "forms": forms, "S": Ssyn.round(4).tolist(),
            "models": models, "noise": float(cfg.get("attr_noise", 0.02))}


@register_module
class Industry(Module):
    name = "industry"
    host = "make"
    depends = ("identity", "bank", "ops")

    def install(self, world):
        st = world.state
        for t, idx in [("regions", ()), ("sites", ("owner", "region")), ("facilities", ("owner", "region")),
                       ("machines", ("owner", "facility", "status")), ("machine_orders", ("owner",)),
                       ("designs", ("owner",)), ("stock", ("fac", "owner", "sku")), ("runs", ("machine", "owner", "status")),
                       ("offers", ("seller", "sku", "status")), ("trade_orders", ("buyer", "seller", "offer")),
                       ("standing", ("buyer", "offer", "status")), ("retail", ("seller", "sku", "status")),
                       ("grid_days", ("region",)), ("labor_days", ("org",)), ("design_rights", ("holder", "design")),
                       ("process_notes", ("owner", "model")), ("note_rights", ("holder", "note"))]:
            st.declare_table(t, index=idx)
        world.hooks["industry"] = self
        from ..actors.producers import register_producer_routines
        register_producer_routines(world)
        ro = world.hooks.setdefault("resource_owner", {})
        ro["facility"] = lambda st_, rid: (st_.get("facilities", rid) or {}).get("owner")
        ro["machine"] = lambda st_, rid: (st_.get("machines", rid) or {}).get("owner")
        ro["design"] = lambda st_, rid: (st_.get("designs", rid) or {}).get("owner")
        prev = world.hooks.get("resource_lister")

        def lister(st_, org):
            out = dict(prev(st_, org)) if prev else {}
            out["facility"] = st_.find_ids("facilities", "owner", org)
            out["machine"] = [m for m in st_.find_ids("machines", "owner", org)
                              if st_.tables["machines"][m]["status"] != "scrapped"]
            return out
        world.hooks["resource_lister"] = lister
        world.hooks["resource_container"] = lambda st_, rid: (st_.get("machines", rid) or {}).get("facility")
        rel = world.hooks.get("resource_relation")

        def relation(st_, kind, cands, bound, params):
            if kind == "machine":
                facs = [v for k, v in bound.items() if params[k]["kind"] == "facility"]
                for f in facs:
                    ms = [m for m in cands if st_.tables["machines"][m]["facility"] == f]
                    if ms:
                        return ms[0]
            return rel(st_, kind, cands, bound, params) if rel else None
        world.hooks["resource_relation"] = relation

    def p(self, key, default):
        return self.config.get(key, self.world.spec.params.get(key, default))

    @property
    def phys(self) -> dict:
        return self.world.state.meta["phys"]

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        cfg = self.config
        rng = ctx.world.rng_for("industry-physics")
        n_reg = int(cfg.get("regions", 4))
        phys = gen_physics(rng, int(cfg.get("materials", 6)), n_reg, cfg)
        ctx.emit("make.physics", {"phys": phys}, vis="internal")
        for r in range(n_reg):
            ctx.emit("make.region", {"id": f"r{r}", "name": f"Region {r}"}, vis="public")
        ctx.schedule(ctx.now + DAY + 60, "industry.daily", {}, prio=30, period=DAY, key="industry.daily")

    def add_site(self, ctx, owner, region, material, reserve, rate):
        sid = ctx.state.new_id("site")
        ctx.emit("make.site", {"id": sid, "owner": owner, "region": region, "sku": material, "reserve": int(reserve),
                               "rate": int(rate)}, vis="public")
        return sid

    def add_grid(self, ctx, region, owner, capacity_kwh_day, base_price_cents):
        ctx.emit("make.grid", {"id": region, "owner": owner, "capacity": int(capacity_kwh_day),
                               "base_price": float(base_price_cents)}, vis="public")

    # ------------------------------------------------------------------ physics helpers
    def item_info(self, st, sku: str) -> dict:
        """Static facts about an item type: stage and (for designed items) the design that makes it."""
        if sku.startswith("m:"):
            return {"stage": "material", "form": None}
        d = st.get("designs", sku[2:]) if sku.startswith("i:") else None
        if d is None:
            raise NotFound("unknown item " + sku)
        return {"stage": d["stage"], "form": d.get("form")}

    def quality(self, mdl: dict, speed: float, temp: float) -> float:
        d2 = (speed - mdl["s_opt"]) ** 2 + (temp - mdl["t_opt"]) ** 2
        return math.exp(-d2 / (2 * mdl["width"] ** 2))

    def energy_per_unit(self, mdl: dict, speed: float, temp: float) -> float:
        return mdl["energy_kwh"] * (0.6 + 0.8 * temp + 0.4 * speed * speed)

    def throughput(self, mdl: dict, speed: float) -> float:
        return mdl["cap_per_hour"] * (0.3 + 0.7 * speed)

    def output_vector(self, st, d: dict, inputs: list[tuple[str, int, list]], q: float, rng) -> list[float]:
        """Deterministic part of the hidden physics plus measurement-free noise."""
        ph = self.phys
        mats = {m["sku"]: i for i, m in enumerate(ph["materials"])}
        noise = rng.normal(0, ph["noise"], 4)
        op = d["op"]
        if op == "extract":
            m = ph["materials"][mats[d["material"]]]
            v = np.array(m["props"]) + noise
        elif op == "refine":
            v = np.array(inputs[0][2], float)
            v = v * 0.97 + 0.03 * v[3]
            v[3] = 1 - (1 - v[3]) * (1 - 0.55 * q)
            v = v + noise * 0.5
        elif op == "blend":
            tot = sum(x[1] for x in inputs)
            r = [x[1] / tot for x in inputs]
            vs = [np.array(x[2], float) for x in inputs]
            v = sum(ri * vi for ri, vi in zip(r, vs))
            for a in range(len(inputs)):
                for b in range(a + 1, len(inputs)):
                    ia, ib = mats.get(inputs[a][3]), mats.get(inputs[b][3])
                    if ia is not None and ib is not None:
                        v = v + r[a] * r[b] * 4 * np.array(ph["H"][ia][ib])
            v = v + noise
        elif op == "form":
            f = ph["forms"][d["form"]]
            x = np.array(inputs[0][2], float) * 2 - 1
            v = _sig(np.array(f["W"]) @ x * 1.5 + np.array(f["b"])) * (0.75 + 0.25 * q) + noise
        else:   # assemble
            tot = sum(x[1] for x in inputs)
            w = [x[1] / tot for x in inputs]
            vs = [np.array(x[2], float) for x in inputs]
            v = sum(wi * vi for wi, vi in zip(w, vs))
            fi = [FORMS.index(x[4]) if x[4] in FORMS else None for x in inputs]
            for a in range(len(inputs)):
                for b in range(a + 1, len(inputs)):
                    if fi[a] is not None and fi[b] is not None:
                        v = v + w[a] * w[b] * 4 * np.array(ph["S"][fi[a]][fi[b]])
            distinct = len({x[0] for x in inputs})
            v = (v + 0.06 * (distinct - 1)) * (0.8 + 0.2 * q) - 0.02 * max(0, len(inputs) - 3) + noise
        return [round(float(x), 4) for x in np.clip(v, 0.0, 1.0)]

    # ------------------------------------------------------------------ stock
    def stock(self, st, fac: str, sku: str) -> dict | None:
        return st.get("stock", f"{fac}|{sku}")

    def qty(self, st, fac, sku) -> int:
        s = self.stock(st, fac, sku)
        return s["qty"] if s else 0

    def add_stock(self, ctx, fac: str, sku: str, qty: int, vec: list[float]) -> None:
        if qty <= 0:
            return
        f = ctx.state.tables["facilities"][fac]
        ctx.emit("make.stock_add", {"fac": fac, "sku": sku, "qty": int(qty), "v": vec, "owner": f["owner"]},
                 vis=(f["owner"],))

    def take_stock(self, ctx, fac: str, sku: str, qty: int) -> list[float]:
        s = self.stock(ctx.state, fac, sku)
        if s is None or s["qty"] < qty:
            raise InsufficientFunds(f"{fac} has {s['qty'] if s else 0} of {sku}, needs {qty}")
        ctx.emit("make.stock_take", {"fac": fac, "sku": sku, "qty": int(qty)}, vis=(s["owner"],))
        return list(s["v"])

    # ------------------------------------------------------------------ authorization helpers
    def _fac(self, ctx, fac, cap="make.operate"):
        f = ctx.state.get("facilities", fac)
        if f is None:
            raise NotFound("no facility " + fac)
        ident(ctx.world).authorize(ctx, f["owner"], cap, scope=fac)
        return f

    def _machine(self, ctx, mid, cap):
        m = ctx.state.get("machines", mid)
        if m is None or m["status"] == "scrapped":
            raise NotFound("no machine " + mid)
        if not ident(ctx.world).allowed(ctx, ctx.actor, m["owner"], cap, scope=mid, record=True):
            ident(ctx.world).authorize(ctx, m["owner"], cap, scope=m["facility"])
        return m

    def _pay(self, ctx, owner, account, dst_owner, amount, kind, memo, dst_acct=None):
        bank = ctx.world.modules["bank"]
        if account is None:
            account = bank.primary_account(ctx.state, owner)
        if account is None:
            raise InvalidAction(f"{owner} has no account")
        if bank.account_owner(ctx.state, account) != owner:
            raise Forbidden("account does not belong to " + owner)
        dst = dst_acct or (bank.primary_account(ctx.state, dst_owner) if dst_owner else "acct_external")
        if dst is None:
            dst = "acct_external"
        if amount > 0:
            bank.move(ctx, account, dst, int(amount), kind, memo)
        return account

    # ------------------------------------------------------------------ public information
    @endpoint("catalog", S.obj(), read_only=True)
    def ep_catalog(self, ctx):
        """Public facts: regions, raw materials and where they are extracted, machine models for sale (price, lead
        time, nominal capacity, nominal energy, labour), and the vocabularies of operations, forms and attributes.
        Item properties are not published: measure them."""
        st = ctx.state
        ph = self.phys
        vend = {mid: (st.meta.get("vendors") or {}).get(mid) for mid in ph["models"]}
        return {"regions": [dict(r, id=k) for k, r in st.scan("regions")],
                "materials": [m["sku"] for m in ph["materials"]],
                "sites": [{"site": k, **{x: s[x] for x in ("owner", "region", "sku", "rate", "reserve")}}
                          for k, s in st.scan("sites")],
                "machine_models": [{"model": k, "op": m["op"], "price_cents": m["price"], "lead_days": m["lead_days"],
                                    "nominal_units_per_hour": m["cap_per_hour"], "nominal_kwh_per_unit": m["energy_kwh"],
                                    "operator_hours_per_hour": m["labor_h_per_h"], "vendor": vend.get(k)}
                                   for k, m in sorted(ph["models"].items())],
                "operations": list(OPS), "forms": list(FORMS), "properties": list(PROPS), "attributes": list(ATTRS),
                "process_parameters": {"speed": "[0,1]", "temp": "[0,1]"}}

    @endpoint("grid", S.obj({"region": S.s_str(maxLength=10)}), read_only=True)
    def ep_grid(self, ctx, region=None):
        """Energy grids: daily capacity, today's use and the current marginal price per kWh."""
        out = []
        for rid, r in sorted(ctx.state.scan("regions")):
            g = r.get("grid")
            if not g or (region and rid != region):
                continue
            used = (ctx.state.get("grid_days", f"{rid}|{ctx.now // DAY}") or {}).get("used", 0)
            out.append({"region": rid, "owner": g["owner"], "capacity_kwh_day": g["capacity"], "used_today": used,
                        "price_cents_per_kwh": round(self.energy_price(ctx.state, rid, 0), 3)})
        return out

    def energy_price(self, st, region, extra) -> float:
        g = st.tables["regions"][region]["grid"]
        used = (st.get("grid_days", f"{region}|{self.world.now // DAY}") or {}).get("used", 0)
        u = min(1.5, (used + extra) / max(1, g["capacity"]))
        return g["base_price"] * (1 + 2 * u * u)

    def draw_energy(self, ctx, region: str, kwh: float, account: str, memo: str, *, consumer: str) -> dict:
        """5.0 (crucible.civ, coupling `compute_energy`): draw `kwh` from a regional grid for a load that is not a
        production run (a datacenter). Same grid capacity and utilisation price as production runs: refused with
        CapacityExceeded when today's capacity is used up; otherwise paid from `account` to the grid owner at the
        marginal price and recorded in today's grid use (`make.energy`), which production runs then see."""
        st = ctx.state
        reg = st.get("regions", region)
        g = (reg or {}).get("grid")
        if not g:
            raise NotFound(f"no grid in region {region}")
        kwh = float(kwh)
        if not math.isfinite(kwh) or kwh <= 0:
            raise InvalidAction("kwh must be positive")
        day = ctx.now // DAY
        used = (st.get("grid_days", f"{region}|{day}") or {}).get("used", 0)
        if used + kwh > g["capacity"]:
            raise CapacityExceeded(f"grid {region} cannot supply {kwh:.0f} kWh today ({g['capacity'] - used:.0f} left)")
        price = self.energy_price(st, region, kwh / 2)
        cost = max(1, int(math.ceil(kwh * price)))
        bank = ctx.world.modules["bank"]
        dst = bank.primary_account(st, g["owner"]) or "acct_external"
        bank.move(ctx, account, dst, cost, "energy", memo, ref=region)
        ctx.emit("make.energy", {"region": region, "kwh": round(kwh, 3), "cost": cost, "consumer": consumer},
                 vis=(consumer, g["owner"]))
        return {"kwh": round(kwh, 3), "cost_cents": cost, "price_cents_per_kwh": round(price, 4)}

    # ------------------------------------------------------------------ facilities & machines
    @endpoint("build_facility", S.obj({"org": S.s_str(maxLength=40), "region": S.s_str(maxLength=10),
                                       "name": S.s_str(maxLength=60), "account": S.s_str(maxLength=40)},
                                      ["org", "region"]), cost_s=600)
    def ep_build_facility(self, ctx, org, region, name="", account=None):
        """Build a facility (workshop/warehouse) in a region. Costs `facility_cents` (default 200,000)."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        if not st.has("regions", region):
            raise NotFound("no region " + region)
        self._pay(ctx, org, account, None, int(self.p("facility_cents", 200_000)), "facility", f"facility in {region}")
        fid = st.new_id("fac")
        ctx.emit("make.facility", {"id": fid, "owner": org, "region": region, "name": name}, vis="public")
        return {"facility": fid}

    def add_facility(self, ctx, owner, region, name=""):
        fid = ctx.state.new_id("fac")
        ctx.emit("make.facility", {"id": fid, "owner": owner, "region": region, "name": name}, vis="public")
        return fid

    @endpoint("buy_machine", S.obj({"facility": S.s_str(maxLength=40), "model": S.s_str(maxLength=20),
                                    "account": S.s_str(maxLength=40)}, ["facility", "model"]), cost_s=300)
    def ep_buy_machine(self, ctx, facility, model, account=None):
        """Order a machine from its vendor; delivered to the facility after the model's lead time."""
        st = ctx.state
        f = self._fac(ctx, facility, "org.admin")
        mdl = self.phys["models"].get(model)
        if mdl is None:
            raise NotFound("no model " + model)
        if len(st.find_ids("machines", "facility", facility)) >= int(self.p("machines_per_facility", 12)):
            raise Conflict("facility full")
        vendor = (st.meta.get("vendors") or {}).get(model)
        acct = self._pay(ctx, f["owner"], account, vendor, mdl["price"], "machine", f"machine {model}")
        oid = st.new_id("morder")
        ctx.emit("make.machine_order", {"id": oid, "owner": f["owner"], "facility": facility, "model": model,
                                        "due": ctx.now + mdl["lead_days"] * DAY}, vis=(f["owner"],))
        ctx.schedule(ctx.now + mdl["lead_days"] * DAY, "industry.deliver_machine", {"order": oid}, prio=31)
        if vendor:   # the vendor pays for parts from outside the economy
            bank = ctx.world.modules["bank"]
            va = bank.primary_account(st, vendor)
            if va:
                bank.move(ctx, va, "acct_external", int(mdl["price"] * 0.6), "materials", f"parts for {model}")
        del acct
        return {"order": oid, "due": ctx.now + mdl["lead_days"] * DAY}

    def add_machine(self, ctx, owner, facility, model, wear=0.0):
        mid = ctx.state.new_id("mach")
        ctx.emit("make.machine", {"id": mid, "owner": owner, "facility": facility, "model": model,
                                  "wear": float(wear)}, vis=(owner,))
        return mid

    @process("deliver_machine")
    def pr_deliver_machine(self, ctx, order):
        o = ctx.state.get("machine_orders", order)
        if o is None or o["status"] != "ordered":
            return
        ctx.emit("make.machine_order_done", {"id": order}, vis=(o["owner"],))
        self.add_machine(ctx, o["owner"], o["facility"], o["model"])

    @endpoint("machines", S.obj({"org": S.s_str(maxLength=40), "facility": S.s_str(maxLength=40)}), read_only=True)
    def ep_machines(self, ctx, org=None, facility=None):
        """Machines of you, an organisation, or one facility, with status, hours run and last maintenance
        (needs make.read; a facility-scoped grant is enough for that facility)."""
        st = ctx.state
        if facility:
            f = st.get("facilities", facility)
            if f is None:
                raise NotFound("no facility " + facility)
            owner = f["owner"]
            ident(ctx.world).authorize(ctx, owner, "make.read", scope=facility)
            ids = st.find_ids("machines", "facility", facility)
        else:
            owner = ident(ctx.world).resolve_handle(st, org) if org else ctx.actor
            ident(ctx.world).authorize(ctx, owner, "make.read")
            ids = st.find_ids("machines", "owner", owner)
        out = []
        for mid in ids:
            m = st.tables["machines"][mid]
            out.append({"machine": mid, **{k: m[k] for k in ("facility", "model", "status", "hours", "busy_until",
                                                               "last_maint", "breakdowns")},
                        "op": self.phys["models"][m["model"]]["op"]})
        return out

    @endpoint("telemetry", S.obj({"machine": S.s_str(maxLength=40)}, ["machine"]), read_only=True)
    def ep_telemetry(self, ctx, machine):
        """Sensor readings of a machine: vibration (a noisy view of its wear), hours since maintenance, status."""
        m = self._machine(ctx, machine, "make.read")
        rng = ctx.world.rng_for("telemetry", machine, ctx.now // 3600)
        return {"machine": machine, "status": m["status"], "hours": m["hours"],
                "hours_since_maintenance": m["hours_since_maint"],
                "vibration": round(float(m["wear"] + rng.normal(0, 0.06)), 4), "breakdowns": m["breakdowns"],
                "busy_until": m["busy_until"]}

    @endpoint("maintain", S.obj({"machine": S.s_str(maxLength=40)}, ["machine"]), cost_s=60)
    def ep_maintain(self, ctx, machine):
        """Maintain an idle machine: 4 hours out of service, costs the model's maintenance fee, resets wear.
        Needs `make.maintain` on the owner (scope: machine or facility)."""
        m = self._machine(ctx, machine, "make.maintain")
        if m["status"] != "ok" or m["busy_until"] > ctx.now:
            raise Conflict(f"machine is {m['status']} / busy until {m['busy_until']}")
        mdl = self.phys["models"][m["model"]]
        self._pay(ctx, m["owner"], None, None, mdl["maint_cost"], "maintenance", f"maintenance {machine}")
        ctx.emit("make.maintained", {"id": machine, "until": ctx.now + 4 * 3600}, vis=(m["owner"],))
        return {"machine": machine, "busy_until": ctx.now + 4 * 3600}

    @endpoint("repair", S.obj({"machine": S.s_str(maxLength=40)}, ["machine"]), cost_s=60)
    def ep_repair(self, ctx, machine):
        """Repair a broken machine (the model's repair cost; back in service after its repair time)."""
        m = self._machine(ctx, machine, "make.maintain")
        if m["status"] != "broken":
            raise Conflict("machine is not broken")
        if (m.get("repair_blocked_until") or 0) > ctx.now:
            raise Conflict(f"spare parts arrive at {m['repair_blocked_until']}")
        mdl = self.phys["models"][m["model"]]
        self._pay(ctx, m["owner"], None, None, mdl["repair_cost"], "maintenance", f"repair {machine}")
        until = ctx.now + mdl["repair_h"] * 3600
        down_h = (until - m["broken_at"]) / 3600.0
        ctx.emit("make.repaired", {"id": machine, "until": until}, vis=(m["owner"],))
        ops = ctx.world.hooks.get("ops")
        if ops is not None:
            ops.record_loss(ctx, m["owner"], "machine_downtime", int(down_h * mdl["price"] / 2000), int(round(down_h)),
                            resource=machine)
        return {"machine": machine, "back_at": until}

    # ------------------------------------------------------------------ designs (process specifications)
    @endpoint("design", S.obj({
        "org": S.s_str(maxLength=40), "name": S.s_str(maxLength=80), "op": S.s_enum(list(OPS)),
        "inputs": S.s_arr(S.obj({"sku": S.s_str(maxLength=40), "qty": S.s_int(minimum=1, maximum=20)}, ["sku", "qty"]),
                          "items consumed per output unit", maxItems=5),
        "site": S.s_str("extract: the site", maxLength=40), "form": S.s_enum(list(FORMS)),
        "speed": S.s_num(minimum=0, maximum=1), "temp": S.s_num(minimum=0, maximum=1),
        "public": S.s_bool("anyone may use this design")}, ["org", "name", "op"]), cost_s=120)
    def ep_design(self, ctx, org, name, op, inputs=None, site=None, form=None, speed=0.5, temp=0.5, public=False):
        """Write a process design: an operation applied to inputs with default process parameters. Its output is
        a new item type `i:<design>`; what it is like is only known by producing and measuring it."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        ident(ctx.world).authorize(ctx, org, "make.operate")
        inputs = inputs or []
        stages = [self.item_info(st, x["sku"])["stage"] for x in inputs]
        material = None
        if op == "extract":
            s = st.get("sites", site or "")
            if s is None or inputs:
                raise InvalidAction("extract needs a site and no inputs")
            material, stage = s["sku"], "material"
        elif op == "refine":
            if len(inputs) != 1 or stages[0] != "material":
                raise InvalidAction("refine takes exactly one material input")
            stage = "material"
        elif op == "blend":
            if not 2 <= len(inputs) <= 4 or any(x != "material" for x in stages):
                raise InvalidAction("blend takes 2-4 material inputs")
            stage = "material"
        elif op == "form":
            if len(inputs) != 1 or stages[0] != "material" or form not in FORMS:
                raise InvalidAction("form takes one material input and a form")
            stage = "component"
        else:
            if not 2 <= len(inputs) <= 5 or any(x != "component" for x in stages):
                raise InvalidAction("assemble takes 2-5 component inputs")
            stage = "product"
        for x in inputs:          # designs built on others' private designs need a right to use them
            if x["sku"].startswith("i:"):
                self.require_right(ctx, org, x["sku"][2:], use_only=True)
        did = st.new_id("d")
        ctx.emit("make.design", {"id": did, "owner": org, "name": name, "op": op, "inputs": inputs, "site": site,
                                 "material": material, "form": form, "speed": float(speed), "temp": float(temp),
                                 "stage": stage, "public": bool(public)}, vis="public" if public else (org,))
        if public:
            from ..services.search import index
            index(ctx, did, "design", name, f"{op} {' '.join(x['sku'] for x in inputs)} {form or ''}", org, ref=did)
        return {"design": did, "sku": f"i:{did}", "stage": stage}

    def has_right(self, st, org, did) -> bool:
        d = st.get("designs", did)
        if d is None:
            return False
        if d["public"] or d["owner"] == org:
            return True
        return st.has("design_rights", f"{org}|{did}")

    def require_right(self, ctx, org, did, use_only=False):
        if not self.has_right(ctx.state, org, did):
            if use_only:   # buying an item made with a private design is fine; designing on it needs nothing more
                return
            raise Forbidden(f"{org} has no right to use design {did}")

    @endpoint("license_design", S.obj({"design": S.s_str(maxLength=40), "to": S.s_str(maxLength=40),
                                       "price_cents": S.s_int(minimum=0, maximum=100_000_000)}, ["design", "to"]),
              cost_s=60)
    def ep_license(self, ctx, design, to, price_cents=0):
        """Grant another organisation the right to run your design (knowledge licensing). If a price is given, the
        licensee pays it now (it must have granted you nothing; the payment is pulled with its consent via
        `make.buy_license`)."""
        st = ctx.state
        d = st.get("designs", design)
        if d is None:
            raise NotFound("no design")
        ident(ctx.world).authorize(ctx, d["owner"], "org.admin")
        to = ident(ctx.world).resolve_handle(st, to)
        ctx.emit("make.right", {"holder": to, "design": design, "price": int(price_cents), "pending": price_cents > 0},
                 vis=(to, d["owner"]))
        return {"ok": True, "pending_payment": price_cents > 0}

    @endpoint("buy_license", S.obj({"design": S.s_str(maxLength=40), "org": S.s_str(maxLength=40),
                                    "account": S.s_str(maxLength=40)}, ["design", "org"]), cost_s=60)
    def ep_buy_license(self, ctx, design, org, account=None):
        """Accept and pay for a design licence offered to your organisation."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        r = st.get("design_rights", f"{org}|{design}")
        if r is None or not r["pending"]:
            raise NotFound("no pending licence")
        d = st.tables["designs"][design]
        self._pay(ctx, org, account, d["owner"], r["price"], "licence", f"licence {design}")
        ctx.emit("make.right", {"holder": org, "design": design, "price": r["price"], "pending": False},
                 vis=(org, d["owner"]))
        return {"ok": True}

    @endpoint("designs", S.obj({"org": S.s_str(maxLength=40), "public": S.s_bool()}), read_only=True)
    def ep_designs(self, ctx, org=None, public=False):
        """Designs you own or may use, or all public designs."""
        st = ctx.state
        mine = ctx.world.principals_of(ctx.actor)
        out = []
        for did, d in st.scan("designs"):
            if (public and d["public"]) or (not public and (d["owner"] in mine or any(
                    st.has("design_rights", f"{p}|{did}") for p in mine))):
                if org and d["owner"] != ident(ctx.world).resolve_handle(st, org):
                    continue
                out.append({"design": did, "sku": f"i:{did}", **{k: d[k] for k in (
                    "owner", "name", "op", "inputs", "site", "form", "speed", "temp", "stage", "public")}})
        return out

    @endpoint("tune", S.obj({"design": S.s_str(maxLength=40), "speed": S.s_num(minimum=0, maximum=1),
                             "temp": S.s_num(minimum=0, maximum=1)}, ["design", "speed", "temp"]), cost_s=30)
    def ep_tune(self, ctx, design, speed, temp):
        """Change a design's default operating point (affects runs started afterwards; the item type is the
        same). Needs make.operate on the design's owner (scope: the design)."""
        d = ctx.state.get("designs", design)
        if d is None:
            raise NotFound("no design " + design)
        ident(ctx.world).authorize(ctx, d["owner"], "make.operate", scope=design)
        ctx.emit("make.tuned", {"id": design, "speed": float(speed), "temp": float(temp), "from": [d["speed"], d["temp"]]},
                 vis=(d["owner"],))
        return {"design": design, "speed": float(speed), "temp": float(temp)}

    # ------------------------------------------------------------------ process knowledge (a tradeable artifact)
    def run_evidence(self, st, owner, model, speed, temp, tol=0.05) -> dict:
        """Verified evidence from the owner's own completed runs on `model` near an operating point."""
        made = units = 0
        kwh = 0.0
        n = 0
        for _, r in st.scan("runs"):
            if r["owner"] != owner or r["status"] != "done":
                continue
            m = st.tables["machines"].get(r["machine"])
            if m is None or m["model"] != model or abs(r["speed"] - speed) > tol or abs(r["temp"] - temp) > tol:
                continue
            made += r["made"]
            units += r["units"]
            kwh += r["kwh"]
            n += 1
        return {"runs": n, "units": units, "yield": round(made / units, 4) if units else None,
                "kwh_per_good_unit": round(kwh / made, 4) if made else None}

    @endpoint("publish_process", S.obj({"org": S.s_str(maxLength=40), "model": S.s_str(maxLength=20),
                                        "speed": S.s_num(minimum=0, maximum=1), "temp": S.s_num(minimum=0, maximum=1),
                                        "price_cents": S.s_int(minimum=0, maximum=100_000_000),
                                        "note": S.s_str(maxLength=2000)}, ["org", "model", "speed", "temp"]),
              cost_s=60)
    def ep_publish_process(self, ctx, org, model, speed, temp, price_cents=0, note=""):
        """Publish (and optionally sell) process knowledge: an operating point for a machine model, with the
        evidence of your own runs at that point attached and verified by the world. Buyers see the evidence,
        not the parameters, until they buy."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        if model not in self.phys["models"]:
            raise NotFound("no model " + model)
        ev = self.run_evidence(st, org, model, float(speed), float(temp))
        if not ev["runs"]:
            raise InvalidAction("publish only what you have run: no completed runs of yours near that point")
        nid = st.new_id("pnote")
        ctx.emit("make.process_note", {"id": nid, "owner": org, "model": model, "speed": float(speed),
                                       "temp": float(temp), "price": int(price_cents), "note": note[:2000],
                                       "evidence": ev}, vis="public")
        from ..services.search import index
        index(ctx, nid, "process", f"Process for {model}", f"{note} yield {ev['yield']} energy {ev['kwh_per_good_unit']}",
              org, ref=nid)
        return {"note": nid, "evidence": ev}

    @endpoint("process_notes", S.obj({"model": S.s_str(maxLength=20)}), read_only=True)
    def ep_process_notes(self, ctx, model=None):
        """Published process knowledge with its verified evidence (parameters only for notes you hold)."""
        st = ctx.state
        mine = ctx.world.principals_of(ctx.actor)
        out = []
        for nid, n in st.scan("process_notes"):
            if model and n["model"] != model:
                continue
            held = n["owner"] in mine or n["price"] == 0 or any(st.has("note_rights", f"{p}|{nid}") for p in mine)
            out.append({"note": nid, "owner": n["owner"], "model": n["model"], "price_cents": n["price"],
                        "evidence": n["evidence"], "note_text": n["note"], "buyers": n["buyers"],
                        **({"speed": n["speed"], "temp": n["temp"]} if held else {})})
        return out

    @endpoint("buy_process", S.obj({"note": S.s_str(maxLength=40), "org": S.s_str(maxLength=40),
                                    "account": S.s_str(maxLength=40)}, ["note", "org"]), cost_s=60)
    def ep_buy_process(self, ctx, note, org, account=None):
        """Buy process knowledge for your organisation (reveals its operating point)."""
        st = ctx.state
        n = st.get("process_notes", note)
        if n is None:
            raise NotFound("no note")
        org = ident(ctx.world).resolve_handle(st, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        if st.has("note_rights", f"{org}|{note}"):
            raise Conflict("already bought")
        if n["owner"] == org:
            raise InvalidAction("it is yours")
        self._pay(ctx, org, account, n["owner"], n["price"], "licence", f"process knowledge {note}")
        ctx.emit("make.note_right", {"holder": org, "note": note, "price": n["price"]}, vis=(org, n["owner"]))
        return {"note": note, "model": n["model"], "speed": n["speed"], "temp": n["temp"]}

    # ------------------------------------------------------------------ production runs
    @endpoint("run", S.obj({"machine": S.s_str(maxLength=40), "design": S.s_str(maxLength=40),
                            "units": S.s_int(minimum=1, maximum=5000),
                            "speed": S.s_num(minimum=0, maximum=1), "temp": S.s_num(minimum=0, maximum=1)},
                           ["machine", "design", "units"]), cost_s=60)
    def ep_run(self, ctx, machine, design, units, speed=None, temp=None):
        """Start a production run on an idle machine: consumes the inputs (or site reserve) now, draws energy from
        the regional grid and operator hours from your staff, and delivers the output into the machine's facility
        when it finishes. Needs `make.operate`."""
        st = ctx.state
        m = self._machine(ctx, machine, "make.operate")
        d = st.get("designs", design)
        if d is None:
            raise NotFound("no design " + design)
        org = m["owner"]
        if not self.has_right(st, org, design):
            raise Forbidden(f"{org} may not run design {design} (licence it)")
        mdl = self.phys["models"][m["model"]]
        if mdl["op"] != d["op"]:
            raise InvalidAction(f"machine model {m['model']} performs {mdl['op']}, design needs {d['op']}")
        if m["status"] != "ok" or m["busy_until"] > ctx.now:
            raise Conflict(f"machine is {m['status']}, busy until {m['busy_until']}")
        fac = st.tables["facilities"][m["facility"]]
        s = float(d["speed"] if speed is None else speed)
        t = float(d["temp"] if temp is None else temp)
        hours = units / self.throughput(mdl, s)
        dur = max(600, int(math.ceil(hours * 3600)))
        # labour
        need_h = hours * mdl["labor_h_per_h"]
        day = ctx.now // DAY
        lab = st.get("labor_days", f"{org}|{day}") or {"used": 0.0}
        avail = 8.0 * self.operators(st, org) - lab["used"]
        if need_h > avail + 1e-9:
            raise CapacityExceeded(f"needs {need_h:.1f} operator hours today, {max(0.0, avail):.1f} available")
        # energy
        kwh = units * self.energy_per_unit(mdl, s, t)
        reg = st.tables["regions"][fac["region"]]
        g = reg.get("grid")
        if g:
            used = (st.get("grid_days", f"{fac['region']}|{day}") or {}).get("used", 0)
            if used + kwh > g["capacity"]:
                ops = ctx.world.hooks.get("ops")
                persist = [ops.loss_event(org, "energy_curtailment", 0, 1, fac["region"])] if ops else []
                raise CapacityExceeded(f"grid {fac['region']} cannot supply {kwh:.0f} kWh today "
                                       f"({g['capacity'] - used:.0f} left)", persist=persist)
            price = self.energy_price(st, fac["region"], kwh / 2)
            cost = int(math.ceil(kwh * price))
            bank = ctx.world.modules["bank"]
            acct = bank.primary_account(st, org)
            if acct is None:
                raise InvalidAction("no account")
            dst = bank.primary_account(st, g["owner"]) or "acct_external"
            bank.move(ctx, acct, dst, max(1, cost), "energy", f"{kwh:.0f} kWh for run on {machine}")
        # inputs
        ins = []
        if d["op"] == "extract":
            site = st.tables["sites"][d["site"]]
            if site["owner"] != org:
                raise Forbidden("you do not own that site")
            if site["region"] != fac["region"]:
                raise InvalidAction("extraction machine must be in the site's region")
            took = min(units, site["reserve"], site["rate"])
            if took < units:
                raise CapacityExceeded(f"site can yield {took} units today")
            ctx.emit("make.site_take", {"id": d["site"], "qty": int(units)}, vis=(org,))
        else:
            for x in d["inputs"]:
                v = self.take_stock(ctx, m["facility"], x["sku"], x["qty"] * units)
                info = self.item_info(st, x["sku"])
                ins.append((x["sku"], x["qty"] * units, v, x["sku"], info.get("form")))
        rid = st.new_id("run")
        ctx.emit("make.run", {"id": rid, "machine": machine, "owner": org, "design": design, "units": int(units),
                              "speed": s, "temp": t, "inputs": [[a, b, c, d_, e] for a, b, c, d_, e in ins],
                              "until": ctx.now + dur, "kwh": round(kwh, 2), "labor_h": round(need_h, 3)},
                 vis=(org,))
        ctx.schedule(ctx.now + dur, "industry.finish", {"run": rid}, prio=32)
        return {"run": rid, "finishes": ctx.now + dur, "kwh": round(kwh, 1), "operator_hours": round(need_h, 2)}

    def operators(self, st, org) -> float:
        """Effective operator staffing: employed operators plus founding staff, times the effort the
        organisation's institutions elicit (1.0 for salaried staff; see `gov.effort`)."""
        n = 0
        if "npc_workers" in st.tables:
            n = sum(1 for _, w in st.scan("npc_workers") if w["employer"] == org and w["role"] == "operator")
        n += int((st.meta.get("own_operators") or {}).get(org, 0))
        gov = self.world.hooks.get("gov")
        if gov is not None and n:
            return n * gov.effort(st, org, max(1, len(st.find_ids("members", "org", org))))
        return float(n)

    @process("finish")
    def pr_finish(self, ctx, run):
        st = ctx.state
        r = st.get("runs", run)
        if r is None or r["status"] != "running":
            return
        m = st.tables["machines"][r["machine"]]
        mdl = self.phys["models"][m["model"]]
        d = st.tables["designs"][r["design"]]
        rng = ctx.rng("run", run)
        hours = (r["until"] - r["t0"]) / 3600.0
        # breakdown: hazard rises with wear during the run
        wear0 = m["wear"]
        wear1 = wear0 + hours * mdl["wear_rate"] * (0.5 + r["speed"])
        hz = mdl["h0"] * math.exp(4.0 * (wear0 + wear1) / 2)
        p_fail = 1 - math.exp(-hz * hours)
        frac = 1.0
        broke = False
        if rng.random() < p_fail:
            broke = True
            frac = float(rng.uniform(0.0, 1.0))
        q = self.quality(mdl, r["speed"], r["temp"])
        y = mdl["y_max"] * (0.6 + 0.4 * q) * max(0.3, 1 - 0.5 * wear1)
        made = int(rng.binomial(int(r["units"] * frac), min(1.0, y))) if r["units"] * frac >= 1 else 0
        defects = int(r["units"] * frac) - made
        inputs = [(a, b, c, dd, e) for a, b, c, dd, e in r["inputs"]]
        vec = self.output_vector(st, d, inputs, q, rng)
        ctx.emit("make.run_done", {"id": run, "made": made, "defects": defects, "broke": broke, "wear": round(wear1, 5),
                                   "hours": round(hours * frac, 3), "q": round(q, 4)}, vis=(r["owner"],))
        self.add_stock(ctx, m["facility"], f"i:{r['design']}" if d["op"] != "extract" else d["material"], made, vec)
        ops = ctx.world.hooks.get("ops")
        if ops is not None and defects > 0:
            ops.record_loss(ctx, r["owner"], "defects", 0, defects, resource=r["machine"])

    @endpoint("runs", S.obj({"org": S.s_str(maxLength=40), "limit": S.s_int(minimum=1, maximum=100)}), read_only=True)
    def ep_runs(self, ctx, org=None, limit=30):
        """Your recent production runs with measured outcomes (units made, defects, breakdowns, energy)."""
        st = ctx.state
        owner = ident(ctx.world).resolve_handle(st, org) if org else ctx.actor
        ident(ctx.world).authorize(ctx, owner, "make.read")
        ids = st.find_ids("runs", "owner", owner)[-limit:]
        return [{"run": i, **{k: st.tables["runs"][i][k] for k in ("machine", "design", "units", "speed", "temp", "status",
                                                                  "made", "defects", "broke", "kwh", "t0", "until")}}
                for i in reversed(ids)]

    @endpoint("inventory", S.obj({"facility": S.s_str(maxLength=40)}, ["facility"]), read_only=True)
    def ep_inventory(self, ctx, facility):
        """Stock at a facility with each item type's measured attribute/property vector (quantity-weighted)."""
        f = self._fac(ctx, facility, "make.read")
        out = []
        for sid in ctx.state.find_ids("stock", "fac", facility):
            s = ctx.state.tables["stock"][sid]
            if s["qty"] > 0:
                out.append({"sku": s["sku"], "qty": s["qty"], "measured": dict(zip(
                    PROPS if self.item_info(ctx.state, s["sku"])["stage"] == "material" else ATTRS, s["v"]))})
        del f
        return out

    @endpoint("move", S.obj({"from_facility": S.s_str(maxLength=40), "to_facility": S.s_str(maxLength=40),
                             "sku": S.s_str(maxLength=40), "qty": S.s_int(minimum=1, maximum=1_000_000)},
                            ["from_facility", "to_facility", "sku", "qty"]), cost_s=30)
    def ep_move(self, ctx, from_facility, to_facility, sku, qty):
        """Move your own stock between your facilities (by freight if they are in different regions)."""
        f1 = self._fac(ctx, from_facility, "make.operate")
        f2 = ctx.state.get("facilities", to_facility)
        if f2 is None or f2["owner"] != f1["owner"]:
            raise Forbidden("destination is not your facility")
        v = self.take_stock(ctx, from_facility, sku, qty)
        return self.dispatch(ctx, f1["owner"], from_facility, to_facility, sku, qty, v, f1["owner"])

    def dispatch(self, ctx, payer, src_fac, dst_fac, sku, qty, vec, owner_to, order=None) -> dict:
        st = ctx.state
        r1 = st.tables["facilities"][src_fac]["region"]
        r2 = st.tables["facilities"][dst_fac]["region"]
        fr = ctx.world.modules.get("freight")
        if r1 == r2 or fr is None:
            self.add_stock(ctx, dst_fac, sku, qty, vec)
            return {"arrives": ctx.now}
        return fr.ship(ctx, payer, r1, r2, dst_fac, sku, qty, vec, order)

    # ------------------------------------------------------------------ business-to-business trade
    @endpoint("offer", S.obj({"facility": S.s_str(maxLength=40), "sku": S.s_str(maxLength=40),
                              "price_cents": S.s_int(minimum=1, maximum=10_000_000),
                              "note": S.s_str(maxLength=500)}, ["facility", "sku", "price_cents"]), cost_s=30)
    def ep_offer(self, ctx, facility, sku, price_cents, note=""):
        """Offer an item from a facility's stock to other businesses at a unit price (updates an existing offer)."""
        f = self._fac(ctx, facility, "make.trade")
        st = ctx.state
        for oid in st.find_ids("offers", "seller", f["owner"]):
            o = st.tables["offers"][oid]
            if o["fac"] == facility and o["sku"] == sku and o["status"] == "active":
                ctx.emit("make.offer_update", {"id": oid, "price": int(price_cents), "note": note}, vis="public")
                return {"offer": oid}
        oid = st.new_id("off")
        ctx.emit("make.offer", {"id": oid, "seller": f["owner"], "fac": facility, "sku": sku, "price": int(price_cents),
                                "note": note, "region": f["region"]}, vis="public")
        return {"offer": oid}

    @endpoint("offers", S.obj({"sku": S.s_str(maxLength=40), "stage": S.s_enum(["material", "component", "product"]),
                               "limit": S.s_int(minimum=1, maximum=200)}), read_only=True)
    def ep_offers(self, ctx, sku=None, stage=None, limit=50):
        """Active business offers with seller, region, price, available quantity and the measured vector of the
        stock on offer (buyers see what they would get)."""
        st = ctx.state
        out = []
        ids = st.find_ids("offers", "sku", sku) if sku else st.ids("offers")
        for oid in ids:
            o = st.tables["offers"][oid]
            if o["status"] != "active":
                continue
            info = self.item_info(st, o["sku"])
            if stage and info["stage"] != stage:
                continue
            s = self.stock(st, o["fac"], o["sku"])
            out.append({"offer": oid, "seller": o["seller"], "sku": o["sku"], "stage": info["stage"],
                        "form": info.get("form"), "region": o["region"], "price_cents": o["price"],
                        "available": s["qty"] if s else 0, "measured": list(s["v"]) if s else None, "note": o["note"],
                        "unfilled_last_7d": sum((o.get("missed") or {}).values())})
            if len(out) >= limit:
                break
        return out

    def _order(self, ctx, buyer, offer_id, qty, to_facility, account) -> dict:
        st = ctx.state
        o = st.get("offers", offer_id)
        if o is None or o["status"] != "active":
            raise NotFound("no active offer " + offer_id)
        dst = st.get("facilities", to_facility)
        if dst is None or dst["owner"] != buyer:
            raise Forbidden("deliver to one of your facilities")
        if o["seller"] == buyer:
            raise InvalidAction("cannot buy from yourself")
        avail = self.qty(st, o["fac"], o["sku"])
        if avail < qty:
            raise InsufficientFunds(f"seller has {avail} available")
        total = o["price"] * qty
        self._pay(ctx, buyer, account, o["seller"], total, "purchase", f"{qty} x {o['sku']}")
        v = self.take_stock(ctx, o["fac"], o["sku"], qty)
        tid = st.new_id("tord")
        ctx.emit("make.trade", {"id": tid, "offer": offer_id, "buyer": buyer, "seller": o["seller"], "sku": o["sku"],
                                "qty": int(qty), "price": o["price"], "to": to_facility}, vis=(buyer, o["seller"]))
        sh = self.dispatch(ctx, o["seller"], o["fac"], to_facility, o["sku"], qty, v, buyer, order=tid)
        return {"order": tid, "total_cents": total, **sh}

    @endpoint("order", S.obj({"offer": S.s_str(maxLength=40), "qty": S.s_int(minimum=1, maximum=1_000_000),
                              "to_facility": S.s_str(maxLength=40), "account": S.s_str(maxLength=40)},
                             ["offer", "qty", "to_facility"]), cost_s=60)
    def ep_order(self, ctx, offer, qty, to_facility, account=None):
        """Buy from a business offer: pay now, goods are taken from the seller's stock and shipped (seller pays
        freight) to your facility."""
        dst = ctx.state.get("facilities", to_facility)
        if dst is None:
            raise NotFound("no facility")
        ident(ctx.world).authorize(ctx, dst["owner"], "make.trade", scope=to_facility)
        return self._order(ctx, dst["owner"], offer, qty, to_facility, account)

    @endpoint("standing_order", S.obj({"offer": S.s_str(maxLength=40), "qty": S.s_int(minimum=1, maximum=100_000),
                                       "every_s": S.s_int(minimum=3600, maximum=30 * 86400),
                                       "to_facility": S.s_str(maxLength=40),
                                       "max_price_cents": S.s_int(minimum=1, maximum=10_000_000)},
                                      ["offer", "qty", "every_s", "to_facility", "max_price_cents"]), cost_s=60)
    def ep_standing(self, ctx, offer, qty, every_s, to_facility, max_price_cents):
        """Recurring procurement: buy `qty` from an offer every `every_s` seconds while its price is at most
        `max_price_cents`. Shortfalls are recorded (supply_shortfall) for you."""
        dst = ctx.state.get("facilities", to_facility)
        if dst is None:
            raise NotFound("no facility")
        ident(ctx.world).authorize(ctx, dst["owner"], "make.trade", scope=to_facility)
        sid = ctx.state.new_id("stand")
        ctx.emit("make.standing", {"id": sid, "buyer": dst["owner"], "offer": offer, "qty": int(qty),
                                   "every_s": int(every_s), "to": to_facility, "max_price": int(max_price_cents)},
                 vis=(dst["owner"],))
        ctx.schedule(ctx.now + every_s, "industry.standing", {"id": sid}, prio=33, period=every_s, key=f"stand:{sid}")
        return {"standing_order": sid}

    @endpoint("cancel_standing", S.obj({"standing_order": S.s_str(maxLength=40)}, ["standing_order"]))
    def ep_cancel_standing(self, ctx, standing_order):
        """Cancel a standing order."""
        s = ctx.state.get("standing", standing_order)
        if s is None:
            raise NotFound("no standing order")
        ident(ctx.world).authorize(ctx, s["buyer"], "make.trade")
        ctx.unschedule(f"stand:{standing_order}")
        ctx.emit("make.standing_status", {"id": standing_order, "status": "cancelled"}, vis=(s["buyer"],))
        return {"ok": True}

    @process("standing")
    def pr_standing(self, ctx, id):
        st = ctx.state
        s = st.get("standing", id)
        if s is None or s["status"] != "active":
            return
        o = st.get("offers", s["offer"])
        ops = ctx.world.hooks.get("ops")
        if o is None or o["status"] != "active" or o["price"] > s["max_price"]:
            if ops:
                ops.record_loss(ctx, s["buyer"], "supply_shortfall", 0, s["qty"], resource=s["offer"])
            ctx.emit("make.standing_run", {"id": id, "got": 0}, vis=(s["buyer"],))
            return
        avail = self.qty(st, o["fac"], o["sku"])
        q = min(avail, s["qty"])
        got = 0
        if q > 0:
            try:
                ctx.run_as(s["buyer"], lambda sub: self._order(sub, s["buyer"], s["offer"], q, s["to"], None))
                got = q
            except CrucibleError:
                got = 0
        if got < s["qty"] and ops:
            ops.record_loss(ctx, s["buyer"], "supply_shortfall", (s["qty"] - got) * o["price"], s["qty"] - got,
                            resource=s["offer"])
        ctx.emit("make.standing_run", {"id": id, "got": got}, vis=(s["buyer"],))

    # ------------------------------------------------------------------ retail (household demand)
    @endpoint("sell", S.obj({"facility": S.s_str(maxLength=40), "sku": S.s_str(maxLength=40),
                             "price_cents": S.s_int(minimum=1, maximum=10_000_000), "title": S.s_str(maxLength=80)},
                            ["facility", "sku", "price_cents"]), cost_s=30)
    def ep_sell(self, ctx, facility, sku, price_cents, title=""):
        """List an item for sale to households from a facility's stock (updates an existing listing's price).
        Households in the facility's region and its neighbours may buy it, if they hear of it and like it."""
        f = self._fac(ctx, facility, "market.manage")
        st = ctx.state
        info = self.item_info(st, sku)
        if info["stage"] == "material":
            raise InvalidAction("households buy components and products, not raw materials")
        for rid in st.find_ids("retail", "seller", f["owner"]):
            r = st.tables["retail"][rid]
            if r["fac"] == facility and r["sku"] == sku and r["status"] == "active":
                ctx.emit("make.retail_update", {"id": rid, "price": int(price_cents)}, vis="public")
                return {"retail": rid}
        rid = st.new_id("ret")
        ctx.emit("make.retail", {"id": rid, "seller": f["owner"], "fac": facility, "sku": sku, "price": int(price_cents),
                                 "region": f["region"], "title": title or sku}, vis="public")
        from ..services.search import index
        index(ctx, rid, "product", title or sku, f"{title} {' '.join(ATTRS)} for sale in {f['region']}", f["owner"],
              ref=rid)
        return {"retail": rid}

    @endpoint("retail", S.obj({"region": S.s_str(maxLength=10), "limit": S.s_int(minimum=1, maximum=200)}),
              read_only=True)
    def ep_retail(self, ctx, region=None, limit=100):
        """Retail listings with price, seller, region, availability, and yesterday's units sold."""
        st = ctx.state
        out = []
        for rid, r in st.scan("retail"):
            if r["status"] != "active" or (region and r["region"] != region):
                continue
            out.append({"retail": rid, **{k: r[k] for k in ("seller", "sku", "price", "region", "title")},
                        "available": self.qty(st, r["fac"], r["sku"]), "sold_yesterday": r.get("sold_y", 0)})
            if len(out) >= limit:
                break
        return out

    @endpoint("unlist", S.obj({"retail": S.s_str(maxLength=40)}, ["retail"]))
    def ep_unlist(self, ctx, retail):
        """Stop selling a retail listing."""
        r = ctx.state.get("retail", retail)
        if r is None:
            raise NotFound("no listing")
        ident(ctx.world).authorize(ctx, r["seller"], "market.manage", scope=r["fac"])
        ctx.emit("make.retail_status", {"id": retail, "status": "closed"}, vis="public")
        return {"ok": True}

    def retail_sale(self, ctx, retail: str, qty: int, account: str, *, buyer: str) -> dict:
        """5.0 (crucible.civ.goods): a household purchase from a retail listing by a member of the population.
        Takes the units from the listing's facility stock (refused with InsufficientFunds when it holds fewer),
        moves the listed price from the buyer's own `account` to the seller, and records the units in the listing's
        sales of the day (`make.retail_sold`), which the producers' routines read as `sold_y` the next day."""
        st = ctx.state
        r = st.get("retail", retail)
        if r is None or r["status"] != "active":
            raise NotFound(f"no active retail listing {retail}")
        qty = int(qty)
        if qty <= 0:
            raise InvalidAction("qty must be positive")
        self.take_stock(ctx, r["fac"], r["sku"], qty)
        bank = ctx.world.modules["bank"]
        dst = bank.primary_account(st, r["seller"])
        if dst is None:
            raise InvalidAction("the seller has no account")
        amount = int(qty * r["price"])
        bank.move(ctx, account, dst, amount, "sale", f"{qty} x {r['sku']}", ref=retail)
        ctx.emit("make.retail_sold", {"id": retail, "qty": qty, "cents": amount, "buyer": buyer},
                 vis=(buyer, r["seller"]))
        return {"retail": retail, "qty": qty, "cents": amount}

    def retail_close_day(self, ctx, day: int) -> None:
        """5.0 (crucible.civ.goods): end of a sales day: `sold_y` of every active listing becomes the units it sold
        on `day` (0 when it sold none), as the 4.0 household model did once a day."""
        st = ctx.state
        upd = {}
        for rid, r in st.scan("retail"):
            if r["status"] != "active":
                continue
            v = int(r.get("sold_today", 0)) if r.get("sold_day") == int(day) else 0
            if v != r.get("sold_y", 0):
                upd[rid] = v
        if upd:
            ctx.emit("make.retail_day", {"day": int(day), "sold_y": upd}, vis="internal")

    # ------------------------------------------------------------------ daily upkeep
    @process("daily")
    def pr_daily(self, ctx):
        st = ctx.state
        for sid, s in st.scan("sites"):
            if s.get("regen"):
                ctx.emit("make.site_take", {"id": sid, "qty": -int(s["regen"])}, vis="internal")
        decay = float(self.p("decay", 0.0))
        if decay > 0:
            for k, s in list(st.scan("stock")):
                if s["qty"] > 0 and not s["sku"].startswith("m:"):
                    lost = int(s["qty"] * decay)
                    if lost:
                        ctx.emit("make.stock_take", {"fac": s["fac"], "sku": s["sku"], "qty": lost}, vis=(s["owner"],))
                        ops = ctx.world.hooks.get("ops")
                        if ops is not None:
                            ops.record_loss(ctx, s["owner"], "spoilage", 0, lost, resource=s["fac"])

    # ------------------------------------------------------------------ appliers
    @process("grid_expand")
    def pr_grid_expand(self, ctx, region, add):
        g = ctx.state.tables["regions"][region]["grid"]
        ctx.emit("make.grid", {"id": region, "owner": g["owner"], "capacity": int(g["capacity"] + add),
                               "base_price": g["base_price"]}, vis="public")
        ctx.emit("make.grid_done", {"region": region}, vis="internal")

    @applier("make.grid_order")
    def ap_grid_order(self, ev):
        pend = dict(self.world.state.meta.get("grid_pending") or {})
        pend[ev.payload["region"]] = ev.payload["due"]
        self.world.state.meta_set("grid_pending", pend)

    @applier("make.grid_done")
    def ap_grid_done(self, ev):
        pend = dict(self.world.state.meta.get("grid_pending") or {})
        pend.pop(ev.payload["region"], None)
        self.world.state.meta_set("grid_pending", pend)

    @applier("make.vendors")
    def ap_vendors(self, ev):
        self.world.state.meta_set("vendors", ev.payload["vendors"])

    @applier("make.own_operators")
    def ap_own_ops(self, ev):
        d = dict(self.world.state.meta.get("own_operators") or {})
        d[ev.payload["org"]] = int(ev.payload["n"])
        self.world.state.meta_set("own_operators", d)

    @applier("make.physics")
    def ap_physics(self, ev):
        self.world.state.meta_set("phys", ev.payload["phys"])

    @applier("make.region")
    def ap_region(self, ev):
        self.world.state.insert("regions", ev.payload["id"], {"name": ev.payload["name"], "grid": None})

    @applier("make.grid")
    def ap_grid(self, ev):
        p = ev.payload
        self.world.state.patch("regions", p["id"], grid={"owner": p["owner"], "capacity": p["capacity"],
                                                         "base_price": p["base_price"]})

    @applier("make.site")
    def ap_site(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("sites", p.pop("id"), p)

    @applier("make.site_take")
    def ap_site_take(self, ev):
        s = self.world.state.tables["sites"][ev.payload["id"]]
        self.world.state.patch("sites", ev.payload["id"], reserve=s["reserve"] - ev.payload["qty"])

    @applier("make.facility")
    def ap_facility(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("facilities", p.pop("id"), p)

    @applier("make.machine_order")
    def ap_machine_order(self, ev):
        p = dict(ev.payload)
        p["status"] = "ordered"
        self.world.state.insert("machine_orders", p.pop("id"), p)

    @applier("make.machine_order_done")
    def ap_machine_order_done(self, ev):
        self.world.state.patch("machine_orders", ev.payload["id"], status="delivered")

    @applier("make.machine")
    def ap_machine(self, ev):
        p = ev.payload
        self.world.state.insert("machines", p["id"], {"owner": p["owner"], "facility": p["facility"], "model": p["model"],
                                                     "status": "ok", "wear": p["wear"], "hours": 0.0,
                                                     "hours_since_maint": 0.0, "busy_until": ev.t, "last_maint": ev.t,
                                                     "breakdowns": 0, "broken_at": None, "t": ev.t})

    @applier("make.maintained")
    def ap_maintained(self, ev):
        self.world.state.patch("machines", ev.payload["id"], wear=0.03, hours_since_maint=0.0,
                               busy_until=ev.payload["until"], last_maint=ev.t)

    @applier("make.repaired")
    def ap_repaired(self, ev):
        self.world.state.patch("machines", ev.payload["id"], status="ok", wear=0.05, hours_since_maint=0.0,
                               busy_until=ev.payload["until"], broken_at=None)

    def force_breakdown(self, ctx, machine: str, parts_days: int = 0) -> None:
        """Operator intervention for controlled experiments (a supplier failure): the machine breaks now and,
        with `parts_days`, cannot be repaired before spare parts arrive."""
        ctx.emit("make.broken", {"id": machine, "blocked_until": ctx.now + int(parts_days) * DAY}, vis="internal")

    @applier("make.broken")
    def ap_broken(self, ev):
        m = self.world.state.tables["machines"][ev.payload["id"]]
        self.world.state.patch("machines", ev.payload["id"], status="broken", broken_at=ev.t,
                               breakdowns=m["breakdowns"] + 1, repair_blocked_until=ev.payload.get("blocked_until"))

    @applier("make.tuned")
    def ap_tuned(self, ev):
        self.world.state.patch("designs", ev.payload["id"], speed=ev.payload["speed"], temp=ev.payload["temp"])

    @applier("make.process_note")
    def ap_process_note(self, ev):
        p = dict(ev.payload)
        p.update(t=ev.t, buyers=0)
        self.world.state.insert("process_notes", p.pop("id"), p)

    @applier("make.note_right")
    def ap_note_right(self, ev):
        st = self.world.state
        p = ev.payload
        st.put("note_rights", f"{p['holder']}|{p['note']}", {"holder": p["holder"], "note": p["note"], "t": ev.t})
        n = st.tables["process_notes"][p["note"]]
        st.patch("process_notes", p["note"], buyers=n["buyers"] + 1)

    @applier("make.design")
    def ap_design(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("designs", p.pop("id"), p)

    @applier("make.right")
    def ap_right(self, ev):
        p = ev.payload
        self.world.state.put("design_rights", f"{p['holder']}|{p['design']}",
                             {"holder": p["holder"], "design": p["design"], "price": p["price"], "pending": p["pending"],
                              "t": ev.t})

    @applier("make.stock_add")
    def ap_stock_add(self, ev):
        st = self.world.state
        p = ev.payload
        k = f"{p['fac']}|{p['sku']}"
        s = st.tables["stock"].get(k)
        if s is None or s["qty"] <= 0:
            st.put("stock", k, {"fac": p["fac"], "sku": p["sku"], "owner": p["owner"], "qty": p["qty"], "v": p["v"]})
        else:
            n = s["qty"] + p["qty"]
            v = [round((a * s["qty"] + b * p["qty"]) / n, 4) for a, b in zip(s["v"], p["v"])]
            st.patch("stock", k, qty=n, v=v)

    @applier("make.stock_take")
    def ap_stock_take(self, ev):
        st = self.world.state
        p = ev.payload
        k = f"{p['fac']}|{p['sku']}"
        s = st.tables["stock"][k]
        st.patch("stock", k, qty=s["qty"] - p["qty"])

    @applier("make.run")
    def ap_run(self, ev):
        st = self.world.state
        p = dict(ev.payload)
        p.update(status="running", t0=ev.t, made=None, defects=None, broke=None)
        st.insert("runs", p.pop("id"), p)
        st.patch("machines", ev.payload["machine"], busy_until=ev.payload["until"])
        day = ev.t // 86400
        fac = st.tables["facilities"][st.tables["machines"][ev.payload["machine"]]["facility"]]
        k = f"{fac['region']}|{day}"
        g = st.tables["grid_days"].get(k) or {"region": fac["region"], "day": day, "used": 0.0}
        st.put("grid_days", k, dict(g, used=round(g["used"] + ev.payload["kwh"], 3)))
        k2 = f"{ev.payload['owner']}|{day}"
        lab = st.tables["labor_days"].get(k2) or {"org": ev.payload["owner"], "day": day, "used": 0.0}
        st.put("labor_days", k2, dict(lab, used=round(lab["used"] + ev.payload["labor_h"], 4)))

    @applier("make.energy")
    def ap_energy(self, ev):
        st = self.world.state
        p = ev.payload
        day = ev.t // 86400
        k = f"{p['region']}|{day}"
        g = st.tables["grid_days"].get(k) or {"region": p["region"], "day": day, "used": 0.0}
        st.put("grid_days", k, dict(g, used=round(g["used"] + p["kwh"], 3)))

    @applier("make.run_done")
    def ap_run_done(self, ev):
        st = self.world.state
        p = ev.payload
        r = st.tables["runs"][p["id"]]
        st.patch("runs", p["id"], status="broke" if p["broke"] else "done", made=p["made"], defects=p["defects"],
                 broke=p["broke"])
        m = st.tables["machines"][r["machine"]]
        ch = {"wear": p["wear"], "hours": round(m["hours"] + p["hours"], 3),
              "hours_since_maint": round(m["hours_since_maint"] + p["hours"], 3), "busy_until": ev.t}
        if p["broke"]:
            ch.update(status="broken", broken_at=ev.t, breakdowns=m["breakdowns"] + 1)
        st.patch("machines", r["machine"], **ch)

    @applier("make.offer")
    def ap_offer(self, ev):
        p = dict(ev.payload)
        p.update(status="active", t=ev.t)
        self.world.state.insert("offers", p.pop("id"), p)

    @applier("make.offer_update")
    def ap_offer_update(self, ev):
        self.world.state.patch("offers", ev.payload["id"], price=ev.payload["price"], note=ev.payload["note"])

    @applier("make.offer_missed")
    def ap_offer_missed(self, ev):
        st = self.world.state
        o = st.tables["offers"][ev.payload["offer"]]
        missed = {k: v for k, v in (o.get("missed") or {}).items() if int(k) >= ev.payload["day"] - 6}
        k = str(ev.payload["day"])
        missed[k] = missed.get(k, 0) + ev.payload["qty"]
        st.patch("offers", ev.payload["offer"], missed=missed)

    @applier("make.trade")
    def ap_trade(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("trade_orders", p.pop("id"), p)

    @applier("make.standing")
    def ap_standing(self, ev):
        p = dict(ev.payload)
        p.update(status="active", t=ev.t, runs=0, got=0)
        self.world.state.insert("standing", p.pop("id"), p)

    @applier("make.standing_status")
    def ap_standing_status(self, ev):
        self.world.state.patch("standing", ev.payload["id"], status=ev.payload["status"])

    @applier("make.standing_run")
    def ap_standing_run(self, ev):
        s = self.world.state.tables["standing"][ev.payload["id"]]
        self.world.state.patch("standing", ev.payload["id"], runs=s["runs"] + 1, got=s["got"] + ev.payload["got"])

    @applier("make.retail")
    def ap_retail(self, ev):
        p = dict(ev.payload)
        p.update(status="active", t=ev.t, sold_y=0, sold_total=0)
        self.world.state.insert("retail", p.pop("id"), p)

    @applier("make.retail_update")
    def ap_retail_update(self, ev):
        self.world.state.patch("retail", ev.payload["id"], price=ev.payload["price"])

    @applier("make.retail_status")
    def ap_retail_status(self, ev):
        self.world.state.patch("retail", ev.payload["id"], status=ev.payload["status"])

    @applier("make.retail_sold")
    def ap_retail_sold(self, ev):
        st = self.world.state
        p = ev.payload
        r = st.tables["retail"][p["id"]]
        day = ev.t // DAY
        today = int(r.get("sold_today", 0)) if r.get("sold_day") == day else 0
        st.patch("retail", p["id"], sold_day=day, sold_today=today + int(p["qty"]),
                 sold_total=int(r.get("sold_total", 0)) + int(p["qty"]))

    @applier("make.retail_day")
    def ap_retail_day(self, ev):
        st = self.world.state
        for rid, v in ev.payload["sold_y"].items():
            st.patch("retail", rid, sold_y=int(v))

    def check_invariants(self, world):
        errs = []
        for k, s in world.state.scan("stock"):
            if s["qty"] < 0:
                errs.append(f"negative stock {k}: {s['qty']}")
        for k, s in world.state.scan("sites"):
            if s["reserve"] < 0:
                errs.append(f"site {k} reserve negative")
        return errs


@register_module
class Freight(Module):
    """Freight network (host `freight`): links between regions with transit hours, per-unit cost and daily
    capacity (units). The network is operated by a carrier organisation that earns the freight charges."""
    name = "freight"
    depends = ("identity", "bank", "industry")

    def install(self, world):
        st = world.state
        st.declare_table("links")
        st.declare_table("link_days", index=("link",))
        st.declare_table("shipments", index=("payer", "status"))

    def genesis(self, ctx):
        cfg = self.config
        for i, l in enumerate(cfg.get("links", [])):
            a, b = sorted([l["a"], l["b"]])
            ctx.emit("freight.link", {"id": f"{a}-{b}", "a": a, "b": b, "hours": int(l["hours"]),
                                      "cost": int(l["cost"]), "capacity": int(l["capacity"]),
                                      "owner": l.get("owner")}, vis="public")

    def path(self, st, src, dst) -> list[str]:
        """Shortest path by transit hours (Dijkstra over links)."""
        import heapq
        adj: dict[str, list] = {}
        for lid, l in st.scan("links"):
            adj.setdefault(l["a"], []).append((l["hours"], l["b"], lid))
            adj.setdefault(l["b"], []).append((l["hours"], l["a"], lid))
        dist = {src: 0}
        prev: dict[str, tuple] = {}
        h = [(0, src)]
        while h:
            d, u = heapq.heappop(h)
            if u == dst:
                break
            if d > dist.get(u, 1e18):
                continue
            for w, v, lid in sorted(adj.get(u, [])):
                if d + w < dist.get(v, 1e18):
                    dist[v] = d + w
                    prev[v] = (u, lid)
                    heapq.heappush(h, (d + w, v))
        if dst not in prev:
            raise NotFound(f"no freight route {src} -> {dst}")
        out = []
        cur = dst
        while cur != src:
            u, lid = prev[cur]
            out.append(lid)
            cur = u
        return out[::-1]

    def ship(self, ctx, payer, r1, r2, dst_fac, sku, qty, vec, order=None) -> dict:
        st = ctx.state
        links = self.path(st, r1, r2)
        t = ctx.now
        cost = 0.0
        delayed = False
        for lid in links:
            l = st.tables["links"][lid]
            day = t // DAY
            while True:
                used = (st.get("link_days", f"{lid}|{day}") or {}).get("used", 0)
                if used + qty <= l["capacity"] or used == 0:
                    break
                day += 1
                delayed = True
            depart = max(t, day * DAY)
            u = (used + qty) / max(1, l["capacity"])
            cost += l["cost"] * qty * (1 + u * u)
            ctx.emit("freight.reserve", {"link": lid, "day": day, "qty": int(qty)}, vis="internal")
            t = depart + l["hours"] * 3600
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, payer)
        owner = st.tables["links"][links[0]].get("owner")
        dst = bank.primary_account(st, owner) if owner else "acct_external"
        if src is None:
            raise InvalidAction("payer has no account")
        bank.move(ctx, src, dst or "acct_external", max(1, int(math.ceil(cost))), "freight",
                  f"{qty} x {sku} {r1}->{r2}")
        sid = st.new_id("ship")
        ctx.emit("freight.shipment", {"id": sid, "payer": payer, "to": dst_fac, "sku": sku, "qty": int(qty), "v": vec,
                                      "arrives": int(t), "delayed": delayed, "order": order, "route": links},
                 vis=(payer, st.tables["facilities"][dst_fac]["owner"]))
        ctx.schedule(int(t), "freight.arrive", {"id": sid}, prio=34)
        return {"shipment": sid, "arrives": int(t), "delayed": delayed, "freight_cents": int(math.ceil(cost))}

    @process("arrive")
    def pr_arrive(self, ctx, id):
        s = ctx.state.get("shipments", id)
        if s is None or s["status"] != "in_transit":
            return
        ctx.emit("freight.arrived", {"id": id}, vis=(s["payer"],))
        ind = ctx.world.modules["industry"]
        ind.add_stock(ctx, s["to"], s["sku"], s["qty"], s["v"])
        if s["delayed"]:
            ops = ctx.world.hooks.get("ops")
            owner = ctx.state.tables["facilities"][s["to"]]["owner"]
            if ops is not None:
                ops.record_loss(ctx, owner, "late_delivery", 0, 1, resource=id)

    @endpoint("links", S.obj(), read_only=True)
    def ep_links(self, ctx):
        """The freight network: links, transit hours, base cost per unit and today's use of daily capacity."""
        day = ctx.now // DAY
        return [{"link": lid, **{k: l[k] for k in ("a", "b", "hours", "cost", "capacity", "owner")},
                 "used_today": (ctx.state.get("link_days", f"{lid}|{day}") or {}).get("used", 0)}
                for lid, l in sorted(ctx.state.scan("links"))]

    @endpoint("quote", S.obj({"from_region": S.s_str(maxLength=10), "to_region": S.s_str(maxLength=10),
                              "qty": S.s_int(minimum=1, maximum=1_000_000)}, ["from_region", "to_region", "qty"]),
              read_only=True)
    def ep_quote(self, ctx, from_region, to_region, qty):
        """Route, cost and arrival time a shipment would have if sent now."""
        st = ctx.state
        links = self.path(st, from_region, to_region)
        t, cost = ctx.now, 0.0
        for lid in links:
            l = st.tables["links"][lid]
            used = (st.get("link_days", f"{lid}|{t // DAY}") or {}).get("used", 0)
            cost += l["cost"] * qty * (1 + ((used + qty) / max(1, l["capacity"])) ** 2)
            t += l["hours"] * 3600
        return {"route": links, "cost_cents": int(math.ceil(cost)), "arrives_if_uncongested": int(t)}

    @applier("freight.link")
    def ap_link(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("links", p.pop("id"), p)

    @applier("freight.reserve")
    def ap_reserve(self, ev):
        p = ev.payload
        k = f"{p['link']}|{p['day']}"
        r = self.world.state.tables["link_days"].get(k) or {"link": p["link"], "day": p["day"], "used": 0}
        self.world.state.put("link_days", k, dict(r, used=r["used"] + p["qty"]))

    @applier("freight.shipment")
    def ap_shipment(self, ev):
        p = dict(ev.payload)
        p.update(status="in_transit", t=ev.t)
        self.world.state.insert("shipments", p.pop("id"), p)

    @applier("freight.arrived")
    def ap_arrived(self, ev):
        self.world.state.patch("shipments", ev.payload["id"], status="arrived")
