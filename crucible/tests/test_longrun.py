"""The long-run harness: every field is produced, a broken process is flagged whatever demand says, a collapsed
sector is flagged with its day and proximate cause, shocks are recorded and recovery is reported, runs replay."""
import csv
import json
import os

import pytest

import crucible

crucible.load_all_modules()
from crucible.actors import entry as E  # noqa: E402,F401
from crucible.agents.base import Policy  # noqa: E402
from crucible.core.errors import Forbidden  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation import longrun as L  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
NEEDS = ["sort.records", "text.extract"]
OV = {"n_consumers": 400, "needs": NEEDS, "module_config": {"consumers": {"k_samples": 2}}}
DAY_FIELDS = {"day", "events", "commands", "rejected", "proc_failed", "proc_failures", "violations_new", "active_firms",
              "entries", "exits", "bankruptcies", "price_changes", "new_listings", "adoptions", "founder_entries",
              "founder_exits", "founder_failures", "wall_s", "rss_mb", "log_bytes", "served_total", "requests_total",
              "revenue_total", "households_cents", "collapsed", "new_collapses", "sectors", "top_types"}
SECTOR_FIELDS = {"potential", "requests", "served", "quality", "revenue", "failed", "sellers", "customers", "unserved"}


def _seller(w, need):
    return next(l["seller"] for _, l in w.state.scan("listings") if str(l["need"]) == need and l["status"] == "active")


def test_a_short_run_produces_every_field_passes_the_replay_check_and_writes_its_files(tmp_path):
    rows = []
    s = L.run_long("tiny", 3, 9, overrides=OV, out_dir=str(tmp_path), extra_modules=["entry"], progress=rows.append,
                   label="short")
    assert len(rows) == 9 and [r["day"] for r in rows] == list(range(9))
    for r in rows:
        assert set(r) == DAY_FIELDS
        assert set(r["sectors"]) == set(NEEDS)
        for x in r["sectors"].values():
            assert set(x) == SECTOR_FIELDS
            assert x["customers"] > 0 and x["potential"] > 0 and x["sellers"] >= 1
            assert 0 <= x["served"] <= x["requests"] and x["unserved"] <= x["customers"]
        assert r["events"] > 0 and r["wall_s"] > 0 and r["rss_mb"] > 10 and r["log_bytes"] > r["events"] * 50
        assert r["active_firms"] >= 2 and r["proc_failures"] == 0 and r["violations_new"] == 0
    last = rows[-1]["sectors"]
    assert all(x["served"] > 100 and 0.5 < x["quality"] <= 1.0 and x["revenue"] > 0 for x in last.values())
    assert 0 < s["events"] - sum(r["events"] for r in rows) < 2000        # the rest is genesis
    # the summary
    assert s["household_budget"]["bound_from_day"] is None and s["household_budget"]["balance_end"] > 0
    assert all(r["households_cents"] > 0 for r in rows)
    for k in ("family", "seed", "days", "events", "weeks", "output", "liveness", "flags", "invariants", "economy",
              "cost", "incidents", "replay", "healthy", "wall_s", "entry", "spec_hash", "modules", "files"):
        assert k in s, k
    assert s["replay"]["identical"] is True and s["replay"]["from"] == "genesis" and s["replay"]["events"] == s["events"]
    assert s["healthy"] is True and s["liveness"]["collapses"] == [] and s["liveness"]["unrecovered"] == []
    assert s["flags"]["repeated_process_failures"] == [] and s["invariants"]["violations"] == 0
    assert s["flags"]["underserved_sectors"] == []
    assert s["invariants"]["checks"] >= 9 and s["invariants"]["money"]["private"] > 0
    assert len(s["weeks"]) == 1 and s["weeks"][0]["days"] == [0, 6]
    wk = s["weeks"][0]
    assert wk["served"] == sum(r["served_total"] for r in rows[:7]) and set(wk["sectors"]) == set(NEEDS)
    assert s["cost"]["s_per_sim_day"] > 0 and s["cost"]["log_bytes"] == sum(r["log_bytes"] for r in rows)
    assert s["entry"]["founders"] == E.DEFAULTS["founders"] and "entry" in s["modules"]
    assert s["economy"]["active_firms_end"] >= 2
    # the files
    assert json.load(open(s["files"]["json"]))["replay"]["identical"] is True
    with open(s["files"]["csv"]) as fh:
        table = list(csv.reader(fh))
    assert len(table) == 10 and table[0][:len(L.CSV_BASE)] == L.CSV_BASE
    assert f"{NEEDS[0]}:served" in table[0] and f"{NEEDS[1]}:unserved" in table[0]
    col = table[0].index(f"{NEEDS[0]}:served")
    assert [int(r[col]) for r in table[1:]] == [r["sectors"][NEEDS[0]]["served"] for r in rows]
    assert "wk | " in L.liveness_table(s)
    print("MEASURED short run cost:", s["cost"], "replay", s["replay"])


def test_a_broken_process_is_flagged_although_demand_stays_healthy(tmp_path):
    def before(w, day):
        if day == 3:
            def boom(ctx):
                raise RuntimeError("hidden crash")
            w.processes["market.sweep"] = boom          # an hourly process nobody looks at
    s = L.run_long("tiny", 3, 9, overrides=OV, out_dir=str(tmp_path), before_day=before, replay_check=False)
    flag = s["flags"]["repeated_process_failures"]
    assert [f["proc"] for f in flag] == ["market.sweep"]
    assert flag[0]["failures"] == 6 * 24 and flag[0]["first_day"] == 3 and "hidden crash" in flag[0]["last_error"]
    assert s["flags"]["process_failures"] == {"market.sweep": 144}
    assert s["healthy"] is False
    # demand never noticed: no sector collapsed, the last day serves as much as the days before the break
    assert s["liveness"]["collapses"] == []
    days = list(csv.DictReader(open(s["files"]["csv"])))
    assert int(days[8]["served_total"]) > 0.8 * int(days[2]["served_total"]) > 0
    assert [int(d["proc_failures"]) for d in days] == [0, 0, 0] + [24] * 6
    assert any(v["kind"] == "process.repeated_failure" and v["where"] == "market.sweep" for v in s["invariants"]["first"])


def _overextend(w, org):
    """The 3.0 failure: a firm scales to many cores, the hourly bill exceeds its sales, the cash runs out."""
    st, bank = w.state, w.modules["bank"]
    inst = next(i for i in st.find_ids("instances", "owner", org) if st.tables["instances"][i]["status"] == "running")
    assert w.execute(org.replace("o:", "p:") + "-ceo", {"tool": "cloud.resize", "args": {"instance": inst, "cores": 16}}).ok
    acct = bank.primary_account(st, org)
    w._run_system("test-drain", lambda ctx: bank.move(ctx, acct, "acct_genesis", bank.balance(st, acct), "test", "drain"))


CAUSES = {
    "exit": (lambda w: w.system_call("biz.dissolve", {"org": _seller(w, NEEDS[1])},
                                     as_principal=_seller(w, NEEDS[1]).replace("o:", "p:") + "-ceo"),
             "exit", "exit of a seller", [NEEDS[1]]),
    "unpaid": (lambda w: _overextend(w, _seller(w, NEEDS[1])), "suspension", "suspension for unpaid bills", [NEEDS[1]]),
    "outage": (lambda w: w._run_system("test-outage", lambda ctx: ctx.emit(
        "cloud.outage", {"id": "o:cloud1", "start": ctx.now, "until": ctx.now + 60 * 3600}, vis="public")),
        "outage", "provider outage", NEEDS),
    "process": (lambda w: w.processes.__setitem__("consumers.hour", _raise), "process_failure", "process failure",
                NEEDS),
}


def _raise(ctx):
    raise IndexError("index 7 is out of bounds")


@pytest.mark.parametrize("name", sorted(CAUSES))
def test_a_collapsed_sector_is_flagged_on_the_day_with_its_proximate_cause(name, tmp_path):
    act, kind, label, hit = CAUSES[name]
    s = L.run_long("tiny", 3, 10, overrides=OV, out_dir=str(tmp_path), replay_check=False,
                   before_day=lambda w, day: act(w) if day == 8 else None)
    eps = s["liveness"]["collapses"]
    assert sorted(e["sector"] for e in eps) == sorted(hit), eps
    for e in eps:
        assert e["day"] == 8 and e["served"] < 0.5 * e["reference"] and e["ratio"] < 0.5
        assert e["customers_with_need"] > 0 and e["reference"] > 100
        assert e["cause"]["kind"] == kind and e["cause"]["label"] == label, e["cause"]
        assert e["cause"]["incident"]["day"] == 8 and e["cause"]["incident"]["seq"] > 0
    if name == "exit":
        assert eps[0]["cause"]["incident"]["org"] == "o:extracco1" and eps[0]["cause"]["incident"]["status"] == "dissolved"
        assert s["economy"]["exits"] == 1 and s["liveness"]["unrecovered"] == [NEEDS[1]]
    if name == "unpaid":
        assert "payment failed" in eps[0]["cause"]["incident"]["reason"]
    if name == "outage":
        assert all(e["recovered_day"] is None and e["min_served"] == 0 for e in eps)   # down for 60 hours
    if name == "process":
        assert [f["proc"] for f in s["flags"]["repeated_process_failures"]] == ["consumers.hour"]
        assert "IndexError" in eps[0]["cause"]["incident"]["error"]
    # a sector that was not hit is not flagged, and "served > 0" would have missed nothing here but proves nothing
    assert s["healthy"] is False
    days = list(csv.DictReader(open(s["files"]["csv"])))
    assert days[8]["collapsed"] == ";".join(sorted(hit)) and days[7]["collapsed"] == ""


def test_a_recorded_shock_is_reported_with_recovery_through_entry_and_none_without(tmp_path):
    ov = dict(OV, module_config={**OV["module_config"],
                                 "entry": {"founder_specs": [{"capital": 900_000, "risk": 0.5, "competence": 0.8,
                                                              "informed": 0.9, "patience": 0.5}] * 2,
                                           "first_scan_day": 9.0, "scan_interval_days": 1.0}})
    shock = {"day": 8, "kind": "bankrupt_largest", "sector": NEEDS[1]}
    out = {}
    for arm, extra in (("entry", ["entry"]), ("base", [])):
        s = L.run_long("tiny", 3, 17, overrides=ov, out_dir=str(tmp_path), extra_modules=extra, shock=shock,
                       label=f"shock-{arm}")
        out[arm] = s
        assert "disturb" in s["modules"]                               # the operator tools are part of the spec
        sh = s["shock"]
        assert sh["report"]["done"] and sh["report"]["org"] == "o:extracco1" and sh["report"]["day"] == 8
        assert sh["report"]["how"] == "admin.shock_bankrupt" and sh["report"]["sectors"] == [NEEDS[1]]
        x = sh["sectors"][NEEDS[1]]
        assert x["served_per_day_before"] > 100 and x["lowest_served"] == 0 and x["fell_below_recover_level"]
        assert x["flagged_collapsed_on"] == 8 and x["cause_reported"] == "recorded shock"
        assert s["economy"]["bankruptcies"] == 1 and s["incidents"]["shock"] == 1
        assert s["replay"]["identical"] is True                        # the disturbance is part of the log
        assert s["invariants"]["violations"] == 0
    e, b = out["entry"]["shock"]["sectors"][NEEDS[1]], out["base"]["shock"]["sectors"][NEEDS[1]]
    assert b["recovered_day"] is None and b["served_last_day"] == 0 and b["sellers_end"] == 0
    assert out["base"]["liveness"]["unrecovered"] == [NEEDS[1]] and out["base"]["healthy"] is False
    assert e["recovered_day"] is not None and e["days_to_recover"] <= 8 and e["sellers_end"] >= 1
    assert e["served_last_day"] >= 0.8 * e["served_per_day_before"]
    ep = out["entry"]["liveness"]["collapses"][0]
    assert ep["recovered_day"] == e["recovered_day"] and out["entry"]["liveness"]["unrecovered"] == []
    assert out["entry"]["entry"]["sectors_served_only_by_entrants_at_end"] == [NEEDS[1]]
    assert out["entry"]["healthy"] is True
    print("MEASURED shock recovery:", {"entry": e, "base": b})


def test_operator_shock_tools_are_refused_for_in_world_actors_and_outages_use_the_clouds_own_state():
    w = World.create(generate({"family": "tiny", "overrides": OV, "extra_modules": ["disturb"]}, 3))
    try:
        w.advance_time(DAY)
        owner = "p:extracco1-ceo"
        for tool, args in (("admin.shock_bankrupt", {"org": "o:extracco1"}),
                           ("admin.shock_outage", {"provider": "o:cloud1", "hours": 5})):
            r = w.execute(owner, {"tool": tool, "args": args})
            assert not r.ok and r.error["code"] == Forbidden.code
        assert not any(t["name"].startswith("admin.") for t in w.available_tools(owner))
        rep = L.inject_shock(w, {"kind": "provider_outage", "hours": 5})
        assert rep["done"] and rep["provider"] == "o:cloud1" and rep["sectors"] == sorted(NEEDS)
        assert w.state.tables["providers"]["o:cloud1"]["outage_until"] == w.now + 5 * 3600
        st = w.modules["cloud"].availability(w.state, w.modules["cloud"].live_deployment(w.state, "extrac1"))
        assert st == "provider outage"
        evs = [e.to_dict() for e in w.log]
        assert World.replay(w.spec, evs).digest() == w.digest()
        g = World.regenerate(w.spec, evs)                       # recorded as a command: it also regenerates
        assert [e.hash for e in g.log] == [e.hash for e in w.log]
        g.close()
    finally:
        w.close()


def test_a_whole_sector_can_be_bankrupted_as_one_recorded_disturbance():
    ov = dict(OV, firms_per_need=2)
    w = World.create(generate({"family": "tiny", "overrides": ov, "extra_modules": ["disturb"]}, 3))
    try:
        w.advance_time(2 * DAY)
        rep = L.inject_shock(w, {"kind": "bankrupt_sector", "sector": NEEDS[0]})
        assert rep["done"] and rep["orgs"] == ["o:recordco1", "o:recordco2"] and rep["sectors"] == [NEEDS[0]]
        st = w.state
        assert all(st.tables["companies"][o]["status"] == "bankrupt" for o in rep["orgs"])
        assert L._sector_state(w)[NEEDS[0]]["sellers"] == 0 and L._sector_state(w)[NEEDS[1]]["sellers"] == 2
        from crucible.evaluation.invariants import check_world
        assert check_world(w) == []                      # a real bankruptcy leaves nothing operating
        assert sum(1 for e in w.log if e.type == "cmd" and e.payload.get("tool") == "admin.shock_bankrupt") == 2
    finally:
        w.close()


def test_a_run_resumes_from_its_snapshot_and_the_resumed_part_replays(tmp_path):
    a = L.run_long("tiny", 3, 6, overrides=OV, out_dir=str(tmp_path), label="res", snapshot_every_days=4)
    assert os.path.exists(tmp_path / "res.snap") and a["replay"]["from"] == "genesis"
    first = list(csv.DictReader(open(a["files"]["csv"])))[:4]
    rows = []
    b = L.run_long("tiny", 3, 9, overrides=OV, out_dir=str(tmp_path), label="res", resume=True, progress=rows.append)
    assert [r["day"] for r in rows] == [4, 5, 6, 7, 8]          # continued from the day-4 snapshot
    assert b["replay"]["identical"] is True and b["replay"]["from"] == "snapshot"
    days = list(csv.DictReader(open(b["files"]["csv"])))
    assert [int(d["day"]) for d in days] == list(range(9))
    assert [d["served_total"] for d in days[:4]] == [d["served_total"] for d in first]   # the kept part is kept
    assert len(a["weeks"]) == 0 and len(b["weeks"]) == 1 and b["healthy"] is True
    assert b["invariants"]["violations"] == 0 and b["liveness"]["collapses"] == []


def test_a_sector_that_never_serves_most_of_its_demand_is_flagged_although_it_never_falls():
    def rows(n, requests, served):
        return [{"day": d, "sectors": {"a": {"requests": requests, "served": served},
                                       "b": {"requests": 1000, "served": 990}}} for d in range(n)]
    assert L.underserved_sectors(rows(10, 60_000, 2_600)) == []            # too short a record to call it chronic
    flagged = L.underserved_sectors(rows(40, 60_000, 2_600))
    assert [f["sector"] for f in flagged] == ["a"]
    assert flagged[0]["served_share_last_window"] == round(2600 / 60000, 4) and flagged[0]["window_days"] == 28
    assert flagged[0]["requests_per_day"] == 60_000.0 and flagged[0]["served_per_day"] == 2_600.0
    assert L.underserved_sectors(rows(40, 60_000, 31_000)) == []           # more than half is served
    # a sector with no demand arriving (no seller, no request) is the collapse rule's business, not this one's
    assert L.underserved_sectors(rows(40, 0, 0)) == []
    # the level over the last four weeks is what counts: an old failure that was repaired is not flagged
    healed = rows(20, 60_000, 2_600) + rows(30, 60_000, 58_000)
    assert L.underserved_sectors(healed) == []


class _Clock(Policy):
    name = "clock-watcher"
    kind = "scripted"

    def act(self, obs, tools):
        self.n = getattr(self, "n", 0) + 1
        if self.n % 2:
            return {"tool": "world.clock", "args": {}}
        return {"tool": "bank.transfer", "args": {"from_account": "acct_nope", "to": "nobody", "amount": 1}}


def test_given_agents_act_every_hour_and_their_rejected_commands_are_counted(tmp_path):
    s = L.run_long("tiny", 3, 2, overrides=OV, out_dir=str(tmp_path), agents=[("zed", _Clock(), 50_000)], label="ag")
    assert s["commands"] == 48 and s["rejected_commands"] == 24
    assert s["replay"]["identical"] is True and s["invariants"]["violations"] == 0


def test_the_operator_tools_module_passes_the_admission_harness():
    from crucible.worldgen.admission import admit
    rep = admit(["disturb"], base=["identity", "bank", "admin", "contracts", "business"], steps=200, seed=5)
    assert rep.passed, (rep.checks, rep.errors[:5])
