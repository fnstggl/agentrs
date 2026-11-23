"""The fidelity register is complete and honest, and the measurement functions give sane, deterministic numbers
on small real worlds."""
import json

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.core.kernel import Module, World, process, register_module  # noqa: E402
from crucible.evaluation import fidelity as F  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def build(seed=3, days=7, reprice_day=3):
    """A small real economy: two incumbents, 400 consumers. The incumbents' own daily repricing is switched off
    (price_step 0) so that the one price change of the run - the dearer incumbent raises its price by a cent on
    day 3, an ordinary `market.update` by its owner - has a clean window on both sides."""
    spec = generate({"family": "tiny", "overrides": {"n_consumers": 400, "params": {"cost_model": "deterministic"}}}, seed)
    for f in next(m for m in spec.modules if m["name"] == "civ")["config"]["firms"]:
        f["policy"]["price_step"] = 0.0
    w = World.create(spec, world_id=f"fid-{seed}-{days}")
    info = {}
    for d in range(days):
        if d == reprice_day:
            lid, l = max(((k, dict(v)) for k, v in w.state.scan("listings")), key=lambda kv: (kv[1]["price"], kv[0]))
            owner = w.state.get("npc_firms", l["seller"])["owner"]
            info = {"listing": lid, "host": l["host"], "old": l["price"], "new": l["price"] + 1}
            w.system_call("market.update", {"listing": lid, "price_cents": info["new"]}, as_principal=owner)
        w.advance_time(w.now + DAY)
    return w, info


@pytest.fixture(scope="module")
def econ():
    w, info = build()
    yield w, info
    w.close()


# ---------------------------------------------------------------------------- the register
def test_register_is_valid_and_claims_no_calibration():
    assert F.validate_register() == []
    counts = {s: sum(1 for m in F.REGISTER if m.status == s) for s in F.STATUSES}
    assert counts["calibrated"] == 0 and counts["stylised-fact-checked"] == 0      # nothing is claimed beyond the evidence
    assert counts["plausible-hypothesis"] >= 25 and counts["known-unrealistic"] >= 12
    assert len({m.id for m in F.REGISTER}) == len(F.REGISTER) >= 45
    for k in F.KINDS:                                                               # every kind of realism has entries
        assert sum(1 for m in F.REGISTER if k in m.kinds) >= 3, k
    assert all(m.caveat for m in F.REGISTER if m.status == "known-unrealistic")     # every simplification says what it is


def test_register_refuses_calibration_without_evidence_in_the_repository():
    base = F.REGISTER[0]
    mk = lambda **kw: F.Mechanism(**{**base.__dict__, "id": "x.test", **kw})  # noqa: E731
    assert any("needs evidence" in e for e in F.validate_register([mk(status="calibrated")]))
    assert any("needs evidence" in e for e in F.validate_register([mk(status="stylised-fact-checked")]))
    ev = {"reference": "Some Survey 2020", "comparison": "src/crucible/evaluation/does_not_exist.py::compare"}
    assert any("not in the repository" in e for e in F.validate_register([mk(status="calibrated", evidence=ev)]))
    ev = {"reference": "Some Survey 2020", "comparison": "src/crucible/evaluation/fidelity.py::no_such_function"}
    assert any("has no function" in e for e in F.validate_register([mk(status="stylised-fact-checked", evidence=ev)]))
    ev = {"reference": "Some Survey 2020", "comparison": "src/crucible/evaluation/fidelity.py::adoption_curves"}
    assert any("must name the dataset" in e for e in F.validate_register([mk(status="calibrated", evidence=ev)]))
    assert any("is not in the repository" in e for e in
               F.validate_register([mk(status="calibrated", evidence=dict(ev, dataset="data/none.csv"))]))
    assert F.validate_register([mk(status="stylised-fact-checked", evidence=ev)]) == []   # reference + code that exists
    assert any("evidence given" in e for e in F.validate_register([mk(evidence=ev)]))
    assert any("unknown status" in e for e in F.validate_register([mk(status="validated")]))
    assert any("kinds must be among" in e for e in F.validate_register([mk(kinds=("realistic",))]))


def test_register_covers_every_module_with_behavioural_rules_in_open_digital():
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 100}}, 1))
    mods = F.modules_with_behavioural_rules(w)
    # the modules that act on their own in an open digital economy are all recognised as such ...
    assert {"consumers", "npcs", "society", "cloud", "market", "business", "contracts", "caps", "deal", "net", "shocks",
            "civ", "ops", "wf", "ads", "jobs"} <= set(mods)
    # ... and each has at least one register entry
    assert F.uncovered_modules(w) == []
    for name in mods:
        assert all(m.status in F.STATUSES for m in F.register_for_module(name))
    # every module of the world is covered, behavioural or not (admin is operator-only)
    assert [n for n in w.modules if not F.register_for_module(n) and n not in F.NOT_BEHAVIOURAL] == []
    w.close()

    # a new module with a rule of its own and no entry makes the check fail
    @register_module
    class Unregistered(Module):
        name = "fid_unregistered"

        def genesis(self, ctx):
            ctx.schedule(ctx.now + 3600, "fid_unregistered.tick", {}, period=3600)

        @process("tick")
        def pr_tick(self, ctx):
            pass
    w2 = World.create(generate({"family": "tiny", "extra_modules": ["fid_unregistered"], "drop_modules": ["consumers", "shocks"]}, 1))
    assert F.uncovered_modules(w2) == ["fid_unregistered"]
    assert F.fidelity_report([w2])["worlds"][0]["modules_without_register_entry"] == ["fid_unregistered"]
    w2.close()
    # the other shipped families are covered too
    for fam in ("open_industry", "engineering_lab"):
        w3 = World.create(generate({"family": fam, "overrides": {"n_shoppers": 200}}, 1))
        assert F.uncovered_modules(w3) == [], fam
        w3.close()


# ---------------------------------------------------------------------------- measures on a real small world
def test_adoption_curves_and_concentration(econ):
    w, _ = econ
    a = F.adoption_curves(w)
    assert a["comparison"] == F.NO_REFERENCE and a["n"] == 2
    offers = a["value"]["offerings"]
    n_cons = len(w.state.arrays["c_prov"])
    for host, c in offers.items():
        days = [d for d, _ in c["series"]]
        assert days == list(range(days[0], days[0] + len(days))) and len(days) == 7
        assert 0 < c["peak"] <= n_cons and c["series"][-1][1] == w.state.get("cons_summary", str(_slot(w, host)))["customers"]
        assert 0 <= c["days_to_10pct"] <= c["days_to_50pct"] <= c["days_to_90pct"] <= 6
        assert c["final_share_of_need"] == 1.0                   # one offering per need in this world
    assert a["value"]["median_days_to_half_peak"] is not None
    assert 0.0 <= a["value"]["share_with_interior_inflection"] <= 1.0
    c = F.concentration(w)
    assert set(c["value"]["needs"]) == {"sort.records", "route.shortest"}
    for need, v in c["value"]["needs"].items():
        assert v["last"] == 1.0 and v["active_offerings_last"] == 1 and len(v["series"]) == 7   # monopolies: HHI 1
    assert c["value"]["mean_hhi_over_needs"] == 1.0
    assert F._hhi([1, 1, 1, 1]) == 0.25 and F._hhi([3, 1]) == 0.625 and F._hhi([]) == 0.0


def _slot(w, host):
    return next(o["slot"] for lid, o in w.state.scan("cons_offers") if w.state.get("listings", lid)["host"] == host)


def test_retention_and_churn_are_exact_cohort_statistics(econ):
    w, _ = econ
    r = F.retention_and_churn(w)
    v = r["value"]
    assert r["n"] == 8                                           # replayed to 8 day boundaries
    for h in ("1d", "7d"):
        assert 0.8 < v[h]["retention"] <= 1.0 and v[h]["customer_days"] > 50
        assert abs(v[h]["retention"] + v[h]["switched"] + v[h]["left_unserved"] - 1.0) < 1e-4
        assert v[h]["switched"] == 0.0                           # nobody to switch to: one offering per need
    assert abs(v["churn_per_day"] - (1 - v["1d"]["retention"])) < 1e-9
    assert v["1d"]["customer_days"] > v["7d"]["customer_days"]   # seven one-day pairs against one seven-day pair
    # recomputed independently: replay every prefix from genesis (no chained snapshots) and compare providers
    events = [e.to_dict() for e in w.log]
    snaps = []
    for d in range(1, 8):
        prefix = [e for e in events if e["t"] // DAY < d]
        snaps.append(np.array(World.replay(w.spec, prefix, verify=False).state.arrays["c_prov"], copy=True))
    snaps.append(np.array(w.state.arrays["c_prov"], copy=True))
    kept = sum(int(((a >= 0) & (a == b)).sum()) for a, b in zip(snaps, snaps[1:]))
    total = sum(int((a >= 0).sum()) for a in snaps[:-1])
    assert total == v["1d"]["customer_days"] and abs(kept / total - v["1d"]["retention"]) < 1e-5
    a, b = snaps[0], snaps[7]
    assert abs(float(((a >= 0) & (a == b)).sum() / (a >= 0).sum()) - v["7d"]["retention"]) < 1e-5
    full = World.replay(w.spec, events)
    assert full.digest() == w.digest()                           # the statistic is computed on the real trajectory


def test_price_elasticity_from_an_in_world_price_change(econ):
    w, info = econ
    e = F.price_elasticity(w, window_days=2)
    rows = e["value"]["changes"]
    assert len(rows) == 1 and rows[0]["listing"] == info["listing"] and rows[0]["clean_window"]
    assert rows[0]["price_before"] == info["old"] and rows[0]["price_after"] == info["new"] and rows[0]["day"] == 3
    assert e["comparison"] == F.NO_REFERENCE and e["n"] == 1
    # by hand, from the event log
    slot = _slot(w, info["host"])
    per_day, cust = {}, {}
    for ev in w.log:
        if ev.type == "consumers.hour" and str(slot) in ev.payload["slots"]:
            d = (ev.payload["t"] - 1) // DAY
            per_day[d] = per_day.get(d, 0) + ev.payload["slots"][str(slot)]["requests"]
            cust[d] = ev.payload["slots"][str(slot)]["customers"]
    q0, q1 = (per_day[1] + per_day[2]) / 2, (per_day[4] + per_day[5]) / 2
    p0, p1 = info["old"], info["new"]
    dp = (p1 - p0) / ((p1 + p0) / 2)
    assert rows[0]["requests_per_day_before"] == q0 and rows[0]["requests_per_day_after"] == q1
    assert abs(rows[0]["arc_elasticity"] - ((q1 - q0) / ((q1 + q0) / 2)) / dp) < 1e-3
    pc0, pc1 = (per_day[1] + per_day[2]) / (cust[1] + cust[2]), (per_day[4] + per_day[5]) / (cust[4] + cust[5])
    assert abs(rows[0]["per_customer_elasticity"] - ((pc1 - pc0) / ((pc1 + pc0) / 2)) / dp) < 1e-3
    assert e["value"]["median_arc_elasticity"] == rows[0]["arc_elasticity"] == e["value"]["median_arc_elasticity_clean_windows"]
    assert "observational" in e["note"]
    # the causal figure: two forks of the finished world, one with a dearer offering; demand is lower there
    digest = w.digest()
    cf = F.price_elasticity_counterfactual(w, listing=info["listing"], factor=1.5, days=2)
    v = cf["value"]
    assert v["price_baseline"] == info["new"] and v["price_changed"] > v["price_baseline"]
    assert v["forks"]["changed"]["requests"] < v["forks"]["baseline"]["requests"] and v["arc_elasticity"] < 0
    assert v["forks"]["changed"]["customers_end"] < v["forks"]["baseline"]["customers_end"]
    assert w.digest() == digest                                   # the world itself was not touched
    # a world without price changes reports that, not a number
    w0, _ = build(seed=4, days=2, reprice_day=99)
    e0 = F.price_elasticity(w0)
    assert e0["value"]["median_arc_elasticity"] is None and e0["n"] == 0 and "no in-world price change" in e0["note"]
    w0.close()


def test_entry_exit_reliability_utilisation_timing_and_size(econ):
    w, _ = econ
    x = F.entry_exit(w)["value"]
    assert x["offerings_at_start"] == 2 and x["offering_entries"] == 0 and x["offering_exits"] == 0
    assert x["offering_entry_rate_per_incumbent_month"] == 0.0 and x["companies_at_start"] == 5
    assert x["company_entries"] == 0 and x["company_entry_rate_per_incumbent_month"] == 0.0 and x["simulated_days"] == 7.0
    rel = F.reliability_distribution(w)
    v = rel["value"]
    assert rel["n"] == w.state.count("svc_days") and 0.0 <= v["min"] <= v["p05"] <= v["p50"] <= v["p95"] <= 1.0
    assert 0.0 <= v["share_of_host_days_below_0.95"] <= v["share_of_host_days_below_0.99"] <= 1.0
    u = F.compute_utilisation(w)["value"]
    assert u["rented_share_of_cores_time_avg"] == 0.1 and u["rented_share_of_cores_peak"] == 0.1    # 2 of 20 cores, always
    assert 0.0 < u["instance_utilisation_mean"] <= u["instance_utilisation_p95"] and u["instance_utilisation_mean"] < 2.0
    assert 0.0 <= u["overloaded_share_of_requests"] <= 1.0
    t = F.demand_timing(w)["value"]
    assert abs(sum(t["share_by_hour"]) - 1.0) < 1e-3 and 0 <= t["peak_hour"] < 24 and t["peak_to_trough"] >= 1.0
    s = F.size_distribution(w)["value"]
    assert s["offerings"] == 2 and 0.0 <= s["gini"] <= 0.5 and 0.5 <= s["top_share"] <= 1.0
    # an entrant and an exit are counted
    w2, _ = build(seed=5, days=1, reprice_day=99)
    w2.system_call("admin.join", {"handle": "founder", "endowment": 500_000})
    acct = w2.modules["bank"].primary_account(w2.state, "p:founder")
    w2.execute("p:founder", {"tool": "id.create_org", "args": {"handle": "newco", "name": "NewCo", "kind": "company"}})
    assert w2.execute("p:founder", {"tool": "biz.register", "args": {"org": "o:newco", "account": acct}}).ok
    assert w2.execute("p:founder", {"tool": "biz.dissolve", "args": {"org": "o:newco"}}).ok
    x2 = F.entry_exit(w2)["value"]
    assert x2["company_entries"] == 1 and x2["company_exits"] == {"dissolved": 1}
    assert x2["company_entry_rate_per_incumbent_month"] > 0
    w2.close()


def test_measures_are_deterministic(econ):
    a, _ = build(seed=7, days=3)
    b, _ = build(seed=7, days=3)
    ma, mb = F.measure_world(a), F.measure_world(b)
    assert json.dumps(ma, sort_keys=True) == json.dumps(mb, sort_keys=True)
    assert json.dumps(F.measure_world(a), sort_keys=True) == json.dumps(ma, sort_keys=True)     # and repeatable
    assert a.digest() == b.digest()                                                             # measuring changed nothing
    other = F.measure_world(econ[0])                                                            # a different world differs
    assert json.dumps(other["adoption_curves"], sort_keys=True) != json.dumps(ma["adoption_curves"], sort_keys=True)
    for w in (a, b):
        w.close()


def test_fidelity_report_separates_eight_kinds_and_invents_no_reference(econ):
    w, _ = econ
    rep = F.fidelity_report([w])
    assert list(rep["kinds"]) == list(F.KINDS) and len(rep["kinds"]) == 8
    assert rep["register_problems"] == [] and rep["status_counts"]["calibrated"] == 0
    assert "0 of" in rep["statement"] and "no empirical reference data" in rep["statement"]
    seen = set()
    for kind, sec in rep["kinds"].items():
        assert sec["mechanisms"] and sum(sec["status_counts"].values()) == len(sec["mechanisms"])
        for m in sec["measures"]:
            seen.add(m["measure"])
            assert m["comparison"] == F.NO_REFERENCE and m["how_measured"]
            assert [v["world"] for v in m["values"]] == [w.world_id]
    assert seen == set(F.MEASURES)
    assert rep["kinds"]["transfer"]["measures"] == [] and "real-world" in rep["kinds"]["transfer"]["note"]
    assert rep["kinds"]["behavioural"]["measures"] and rep["kinds"]["economic"]["measures"]
    text = json.dumps(rep)                                       # the report is plain JSON
    assert "no reference data in repository" in text
    assert rep["worlds"][0]["modules_without_register_entry"] == []
    # no numeric "reference", "target" or "empirical" value is attached to any measure
    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                assert not any(t in str(k).lower() for t in ("reference_value", "empirical", "target", "benchmark")), path + "." + str(k)
                walk(v, path + "." + str(k))
        elif isinstance(x, list):
            for v in x:
                walk(v, path)
    walk(rep["kinds"])
    walk(rep["worlds"])
