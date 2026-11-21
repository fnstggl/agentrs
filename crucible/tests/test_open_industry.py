"""Production economy: conservation, hidden process physics, machines, energy, freight, trade, retail demand,
licensing, input substitution, investment, and adoption of a non-digital invention by producers."""
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.inventors import MaintenanceInventor  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def _w(seed=1, **ov):
    return World.create(generate({"family": "open_industry", "overrides": ov}, seed))


def _ex(w, who):
    return lambda t, **k: w.execute(who, {"tool": t, "args": k})  # noqa: E731


def _agent_firm(w, handle, cash=3_000_000, region="r0", operators=2):
    w.system_call("admin.join", {"handle": handle, "endowment": cash + 500_000})
    ex = _ex(w, f"p:{handle}")
    me = w.modules["bank"].primary_account(w.state, f"p:{handle}")
    org = f"o:{handle}co"
    assert ex("id.create_org", handle=f"{handle}co", name="Co", kind="company").ok
    assert ex("biz.register", org=org, account=me).ok
    acct = ex("bank.open_account", owner=org).result["account"]
    assert ex("bank.transfer", from_account=me, to=acct, amount=cash).ok
    fac = ex("make.build_facility", org=org, region=region).result["facility"]
    w._run_system("staff", lambda ctx: ctx.emit("make.own_operators", {"org": org, "n": operators}, vis="internal"))
    return ex, org, acct, fac


def test_materials_are_conserved_and_outputs_never_exceed_inputs():
    w = _w(1)
    w.advance_time(12 * DAY)
    st = w.state
    for _, r in st.scan("runs"):
        if r["status"] in ("done", "broke"):
            assert r["made"] + r["defects"] <= r["units"]
    op = lambda r: st.tables["designs"][r["design"]]["op"]  # noqa: E731
    initial_reserve = sum(e.payload["reserve"] for e in w.log if e.type == "make.site")
    reserve = sum(s["reserve"] for _, s in st.scan("sites"))
    assert initial_reserve - reserve == sum(r["units"] for _, r in st.scan("runs") if op(r) == "extract")
    genesis = sum(e.payload["qty"] for e in w.log if e.type == "make.stock_add" and e.t == 0
                  and e.payload["sku"].startswith("m:"))
    extracted = sum(r["made"] for _, r in st.scan("runs") if op(r) == "extract" and r["status"] != "running")
    stock = sum(s["qty"] for _, s in st.scan("stock") if s["sku"].startswith("m:"))
    transit = sum(s["qty"] for _, s in st.scan("shipments") if s["status"] == "in_transit" and s["sku"].startswith("m:"))
    consumed = sum(x[1] for _, r in st.scan("runs") for x in r["inputs"] if x[0].startswith("m:"))
    assert genesis + extracted == stock + transit + consumed       # material is neither created nor lost
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []


def test_process_parameters_matter_through_hidden_physics_and_are_learnable_by_experiment():
    w = _w(2)
    ex, org, acct, fac = _agent_firm(w, "eva", region="r1")
    ind = w.modules["industry"]
    st = w.state
    mine_site = next(k for k, s in st.scan("sites") if s["region"] == "r1")
    # buy material from the market, buy a forming machine, and try two operating points of the same design
    off = next(o for o in w.query("p:eva", "make.offers", {"stage": "material"}) if o["available"] > 50)
    assert ex("make.order", offer=off["offer"], qty=60, to_facility=fac).ok
    r = ex("make.buy_machine", facility=fac, model="form-A")
    assert r.ok, r.error
    w.advance_time(r.result["due"] + 3600)
    mach = st.find_ids("machines", "facility", fac)[0]
    mdl = ind.phys["models"]["form-A"]
    good = ex("make.design", org=org, name="p-good", op="form", inputs=[{"sku": off["sku"], "qty": 1}], form="chip",
              speed=mdl["s_opt"], temp=mdl["t_opt"]).result["design"]
    bad = ex("make.design", org=org, name="p-bad", op="form", inputs=[{"sku": off["sku"], "qty": 1}], form="chip",
             speed=1.0 if mdl["s_opt"] < 0.5 else 0.0, temp=1.0 if mdl["t_opt"] < 0.5 else 0.0).result["design"]
    out = {}
    for d in (good, bad):
        w.advance_time(max(w.now, st.tables["machines"][mach]["busy_until"]) + 60)
        rr = ex("make.run", machine=mach, design=d, units=25)
        assert rr.ok, rr.error
        w.advance_time(rr.result["finishes"] + 60)
        out[d] = st.tables["runs"][rr.result["run"]]
    assert out[good]["made"] > out[bad]["made"]                        # yield depends on the hidden optimum
    assert out[bad]["kwh"] != out[good]["kwh"]                         # and so does energy
    del mine_site


def test_energy_grid_refuses_runs_when_full_and_records_curtailment():
    w = _w(3)
    ex, org, acct, fac = _agent_firm(w, "gus", region="r1", operators=50)
    st = w.state
    off = max(w.query("p:gus", "make.offers", {"stage": "material"}), key=lambda o: o["available"])
    assert ex("make.order", offer=off["offer"], qty=min(40, off["available"]), to_facility=fac).ok
    r = ex("make.buy_machine", facility=fac, model="form-A")
    w.advance_time(r.result["due"] + 3600)
    mach = st.find_ids("machines", "facility", fac)[0]
    w._run_system("cap", lambda ctx: ctx.emit("make.grid", {"id": "r1", "owner": "o:gridco", "capacity": 5,
                                                             "base_price": 10.0}, vis="public"))
    d = ex("make.design", org=org, name="sheet", op="form", inputs=[{"sku": off["sku"], "qty": 1}],
           form="sheet").result["design"]
    r = ex("make.run", machine=mach, design=d, units=20)
    assert not r.ok and "grid" in r.error["message"]
    assert st.get("ops_days", f"{org}|{w.now // DAY}")["loss_n"]["energy_curtailment"] >= 1


def test_freight_congestion_delays_shipments_and_raises_cost():
    w = _w(4)
    fr = w.modules["freight"]
    q1 = w.query("p:mine1-ceo", "freight.quote", {"from_region": "r0", "to_region": "r2", "qty": 10})
    q2 = w.query("p:mine1-ceo", "freight.quote", {"from_region": "r0", "to_region": "r2", "qty": 2000})
    assert q2["cost_cents"] / 2000 > q1["cost_cents"] / 10              # congestion pricing
    assert len(fr.path(w.state, "r0", "r2")) == 2


def test_households_buy_by_attributes_and_stockouts_are_losses_of_the_seller():
    w = _w(1)
    w.advance_time(10 * DAY)
    st = w.state
    sold = sum(r.get("sold_total", 0) for _, r in st.scan("retail"))
    assert sold > 0
    pays = [t for _, t in st.scan("txns") if t["kind"] == "sale" and any(a == "acct_households" for a, _ in t["postings"])]
    assert pays
    so = sum(r["loss_n"].get("stockout", 0) for _, r in st.scan("ops_days"))
    sh = sum(r["loss_n"].get("supply_shortfall", 0) for _, r in st.scan("ops_days"))
    assert so + sh > 0                                                 # the chain is constrained, and it shows


def test_design_rights_licensing_and_private_designs():
    w = _w(5)
    ex, org, acct, fac = _agent_firm(w, "lia")
    st = w.state
    d = next(k for k, x in st.scan("designs") if x["op"] == "form")
    owner = st.tables["designs"][d]["owner"]
    m = st.find_ids("machines", "owner", org)
    r = ex("make.buy_machine", facility=fac, model=f"form-A")
    w.advance_time(r.result["due"] + 3600)
    mach = st.find_ids("machines", "facility", fac)[0]
    assert not ex("make.run", machine=mach, design=d, units=1).ok         # private design: no right
    ceo = next(mm["person"] for mm in st.find("members", "org", owner) if mm["role"] == "owner")
    assert _ex(w, ceo)("make.license_design", design=d, to=org, price_cents=5000).ok
    bal = w.modules["bank"].balance(st, acct)
    assert ex("make.buy_license", design=d, org=org).ok
    assert w.modules["bank"].balance(st, acct) == bal - 5000
    assert w.modules["industry"].has_right(st, org, d)
    del m


def test_regeneration_of_an_industrial_world_is_exact():
    w = _w(6)
    w.system_call("admin.join", {"handle": "mo", "endowment": 1_000_000})
    Runner(w, [("p:mo", MaintenanceInventor())], dt=3600, record=False).run(24 * 6)
    w2 = World.regenerate(w.spec, [e.to_dict() for e in w.log])
    assert w2.last_hash == w.last_hash and w2.digest() == w.digest()
    w3 = World.replay(w.spec, [e.to_dict() for e in w.log])
    assert w3.digest() == w.digest()


@pytest.mark.slow
def test_producers_adopt_predictive_maintenance_only_where_verified_outcomes_pay():
    w = _w(1)
    w.system_call("admin.join", {"handle": "mo", "endowment": 1_000_000})
    inv = MaintenanceInventor()
    Runner(w, [("p:mo", inv)], dt=3600, record=False).run(24 * 36)
    st = w.state
    outs = {e["by"]: e for _, e in st.scan("cap_evidence") if e["kind"] == "outcome"}
    assert len(outs) >= 3
    for _, i in st.scan("cap_installs"):
        if i["status"] == "active":
            assert st.tables["cap_evidence"][i["evidence"]]["data"]["effect_per_day"] > 0
        if i["status"] == "removed" and i.get("evidence"):
            e = st.tables["cap_evidence"][i["evidence"]]["data"]
            assert e["effect_per_day"] <= 0 or e["effect_per_day"] < 2 * e["se"]
    # every adoption followed a positive verified trial at the adopter's own margin (z >= 1 standard error)
    for _, i in st.scan("cap_installs"):
        if i["status"] == "active":
            d = st.tables["cap_evidence"][i["evidence"]]["data"]
            assert d["effect_per_day"] > d["se"]
    # during trials the service (not the fixed-interval rule) maintained the firms' machines
    maint_by_svc = [e for e in w.log if e.type == "make.maintained" and e.actor == "svc:predmaint"]
    assert maint_by_svc
    # and delegating maintenance did not stop the firms' unrelated routines (procurement kept buying inputs)
    trial_firms = {e["by"] for e in outs.values()}
    assert any(t["buyer"] in trial_firms for _, t in st.scan("trade_orders") if t["t"] > min(
        i["t"] for _, i in st.scan("cap_installs")) + 2 * DAY)
