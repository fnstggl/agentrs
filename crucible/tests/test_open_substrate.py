"""Open-world substrate: operations journal, workflows/triggers, capability registry, delegation, verified trials,
independent adoption by minds, replay and counterfactual forks."""
import json

import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.inventors import CapacityInventor  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.operations import operating_result  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def _world(seed=1, **ov):
    return World.create(generate({"family": "open_digital", "overrides": ov}, seed))


def _agent(w, handle, cash=2_000_000):
    w.system_call("admin.join", {"handle": handle, "endowment": cash})
    ex = lambda t, **k: w.execute(f"p:{handle}", {"tool": t, "args": k})  # noqa: E731
    return ex


def _org(w, ex, handle, cash=500_000):
    me = w.modules["bank"].primary_account(w.state, f"p:{handle}")
    org = f"o:{handle}co"
    assert ex("id.create_org", handle=f"{handle}co", name="Co", kind="company").ok
    assert ex("biz.register", org=org, account=me).ok
    acct = ex("bank.open_account", owner=org).result["account"]
    assert ex("bank.transfer", from_account=me, to=acct, amount=cash).ok
    return org, acct


def _deploy(w, ex, org, acct, host, files):
    repo = ex("code.create_repo", name=host, owner=org).result["repo"]
    assert ex("code.commit", repo=repo, files=files).ok
    inst = ex("cloud.rent", provider="o:cloud1", cores=1, account=acct, owner=org).result["instance"]
    r = ex("cloud.deploy", repo=repo, host=host, instance=inst)
    assert r.ok and r.result["status"] == "live", r.error
    return repo, inst


def _manifest(eps):
    return json.dumps({"entry": "service.py", "endpoints": eps})


def test_operating_journal_is_derived_from_the_ledger_and_replays_exactly():
    w = _world(n_consumers=1500)
    w.advance_time(3 * DAY)
    ops = w.modules["ops"]
    firm = "o:shorteco1"
    rows = [ops.day(w.state, firm, d) for d in range(4)]
    sales = sum(r["flows"].get("sale", 0) for r in rows if r)
    assert sales > 0
    # recompute from the raw ledger
    accts = set(w.state.find_ids("accounts", "owner", firm))
    raw = sum(d for _, t in w.state.scan("txns") if t["kind"] == "sale" for a, d in t["postings"] if a in accts)
    assert raw == sales
    assert operating_result(rows[1]) == sum(v for k, v in rows[1]["flows"].items() if k not in ("endowment",))
    w2 = World.replay(w.spec, [e.to_dict() for e in w.log])
    assert w2.digest() == w.digest()


def test_overload_losses_are_recorded_where_they_happen():
    w = _world(n_consumers=4000)
    w.advance_time(3 * DAY)
    losses = {}
    for _, r in w.state.scan("ops_days"):
        for k, v in r["losses"].items():
            losses[k] = losses.get(k, 0) + v
    assert losses.get("capacity_overload", 0) > 0
    # the loss equals turned-away requests x the listing price at the time (never more than demand that arrived)
    over = sum(r["overload"] for _, r in w.state.scan("svc_days"))
    assert over > 0


def test_triggers_run_as_owner_pay_prices_and_pause_after_failures():
    w = _world(n_consumers=300)
    ex = _agent(w, "tia")
    org, acct = _org(w, ex, "tia")
    files = {"crucible.json": _manifest({"ping": {"handler": "ping", "price_cents": 5},
                                          "boom": {"handler": "boom"}}),
             "service.py": "def ping(args, caller):\n    return {'caller': caller}\n"
                           "def boom(args, caller):\n    raise ValueError('nope')\n"}
    _deploy(w, ex, org, acct, "pinger", files)
    ex2 = _agent(w, "uma")
    org2, acct2 = _org(w, ex2, "uma")
    r = ex2("wf.trigger", tool="pinger.ping", every_s=3600, as_org=org2)
    assert r.ok and r.result["owner"] == org2
    bad = ex2("wf.trigger", tool="pinger.boom", every_s=3600, as_org=org2).result["trigger"]
    bal0 = w.modules["bank"].balance(w.state, acct2)
    w.advance_time(w.now + 12 * 3600)
    t = w.state.get("wf_triggers", r.result["trigger"])
    assert t["runs"] >= 11 and t["fails"] == 0
    assert w.modules["bank"].balance(w.state, acct2) <= bal0 - 5 * 11      # the organisation paid every run
    b = w.state.get("wf_triggers", bad)
    assert b["status"] == "paused" and b["fails_in_row"] >= 10                 # paused after 10 failures
    # someone else cannot run a trigger as an org they do not administer
    assert not ex("wf.trigger", tool="pinger.ping", every_s=3600, as_org=org2).ok


def test_flows_pass_typed_data_iterate_and_are_atomic():
    w = _world(n_consumers=300)
    ex = _agent(w, "fay")
    me = w.modules["bank"].primary_account(w.state, "p:fay")
    _agent(w, "gus")
    steps = [{"id": "who", "tool": "world.whoami", "args": {}},
             {"id": "pay", "tool": "bank.transfer", "foreach": "${input.payees}",
              "args": {"from_account": me, "to": "${item}", "amount": "${input.amount}", "memo": "by ${who.result.principal}"}}]
    f = ex("wf.define", name="payroll", steps=steps, output="${pay.result}").result["flow"]
    b0 = w.modules["bank"].balance(w.state, me)
    r = ex("wf.run", flow=f, input={"payees": ["p:gus", "p:gus"], "amount": 700})
    assert r.ok, r.error
    assert w.modules["bank"].balance(w.state, me) == b0 - 1400
    # a failing step rolls back the earlier ones (atomic)
    r = ex("wf.run", flow=f, input={"payees": ["p:gus", "p:nobody-here"], "amount": 700})
    assert not r.ok
    assert w.modules["bank"].balance(w.state, me) == b0 - 1400


def test_capability_install_creates_scoped_delegation_and_uninstall_revokes_it():
    w = _world(n_consumers=300)
    ex = _agent(w, "ivy")
    org, acct = _org(w, ex, "ivy")
    src = ("import crucible_sdk as sdk\n"
           "def tick(args, caller):\n"
           "    sdk.call('cloud.resize', instance=args['instance'], cores=2)\n"
           "    return {'ok': True}\n"
           "def evil(args, caller):\n"
           "    return sdk.call('cloud.resize', instance=args['other'], cores=3)\n")
    files = {"crucible.json": _manifest({"tick": {"handler": "tick"}, "evil": {"handler": "evil"}}), "service.py": src}
    _deploy(w, ex, org, acct, "sizer", files)
    cap = ex("caps.publish", host="sizer", title="Sizer", description="resizes",
             params={"instance": {"kind": "instance"}}, requires=[{"cap": "cloud.manage", "scope": "$instance"}],
             triggers=[{"endpoint": "tick", "every_s": 3600, "args": {"instance": "$instance"}}])
    assert cap.ok, cap.error
    cid = cap.result["capability"]
    # an independent firm installs it on one of its two instances
    exb = _agent(w, "bea")
    borg, bacct = _org(w, exb, "bea")
    i1 = exb("cloud.rent", provider="o:cloud1", cores=1, account=bacct, owner=borg).result["instance"]
    i2 = exb("cloud.rent", provider="o:cloud1", cores=1, account=bacct, owner=borg).result["instance"]
    assert not exb("caps.install", capability=cid, org=borg, bindings={"instance": "inst_999999"}).ok
    r = exb("caps.install", capability=cid, org=borg, bindings={"instance": i1})
    assert r.ok, r.error
    w.advance_time(w.now + 2 * 3600)
    assert w.state.get("instances", i1)["cores"] == 2                      # delegated scope works
    # the same service principal cannot touch the other instance (scope enforced by the kernel)
    res = exb("sizer.evil", instance=i1, other=i2)
    assert res.ok and res.result["status"] == 500 and w.state.get("instances", i2)["cores"] == 1
    inst_rec = w.state.get("cap_installs", r.result["install"])
    assert inst_rec["independent"] is True
    assert exb("caps.uninstall", install=r.result["install"]).ok
    for gid in r.result["grants"]:
        assert w.state.get("grants", gid)["status"] != "active"
    assert w.check_invariants() == []


def test_related_party_installs_never_count_as_independent_adoption():
    w = _world(n_consumers=300)
    ex = _agent(w, "rex")
    org, acct = _org(w, ex, "rex")
    files = {"crucible.json": _manifest({"tick": {"handler": "tick", "price_cents": 50}}),
             "service.py": "def tick(args, caller):\n    return {}\n"}
    _deploy(w, ex, org, acct, "selfdeal", files)
    cid = ex("caps.publish", host="selfdeal", title="Self", description="nothing",
             triggers=[{"endpoint": "tick", "every_s": 3600, "args": {}}]).result["capability"]
    # the same person creates a second company and installs its own capability there
    assert ex("id.create_org", handle="rexshell", name="Shell", kind="company").ok
    me = w.modules["bank"].primary_account(w.state, "p:rex")
    assert ex("biz.register", org="o:rexshell", account=me).ok
    a2 = ex("bank.open_account", owner="o:rexshell").result["account"]
    ex("bank.transfer", from_account=me, to=a2, amount=100_000)
    r = ex("caps.install", capability=cid, org="o:rexshell")
    assert r.ok
    w.advance_time(w.now + 5 * 3600)
    s = w.modules["caps"].summary(w, cid)
    assert s["independent_adopters"] == 0 and s["independent_payers"] == 0
    assert w.modules["caps"].categories(w, 1) == []


def test_executable_test_suite_records_mode_a_evidence_with_provenance():
    w = _world(n_consumers=300)
    ex = _agent(w, "sol")
    org, acct = _org(w, ex, "sol")
    files = {"crucible.json": _manifest({"add": {"handler": "add", "price_cents": 1}}),
             "service.py": "def add(args, caller):\n    return {'sum': args['a'] + args['b'] + (1 if args['a'] > 50 else 0)}\n"}
    _deploy(w, ex, org, acct, "adder", files)
    cid = ex("caps.publish", host="adder", title="Adder", description="adds two numbers").result["capability"]
    ext = _agent(w, "tess")
    suite = ("def generate(args, caller):\n    s = args['seed']\n    return {'a': s * 13 % 100, 'b': s}\n"
             "def check(args, caller):\n    i, o = args['input'], args['output']\n"
             "    return 1.0 if o and o.get('sum') == i['a'] + i['b'] else 0.0\n")
    repo = ext("code.create_repo", name="suite").result["repo"]
    assert ext("code.commit", repo=repo, files={"suite.py": suite}).ok
    r = ext("caps.test", capability=cid, endpoint="add", suite_repo=repo, cases=8)
    assert r.ok, r.error
    assert 0.0 < r.result["mean_score"] < 1.0                                  # the bug (a > 50) is detected
    e = w.state.get("cap_evidence", r.result["evidence"])
    assert e["kind"] == "test" and e["verified"] and e["independent"] and e["data"]["suite"].startswith(repo)


@pytest.mark.slow
def test_minds_discover_trial_and_adopt_an_invented_capability_on_verified_outcomes():
    w = _world(1)
    w.system_call("admin.join", {"handle": "ada", "endowment": 1_000_000})
    inv = CapacityInventor()
    Runner(w, [("p:ada", inv)], dt=3600, record=False).run(24 * 26)
    st = w.state
    cid = inv.capability
    outcomes = [e for _, e in st.scan("cap_evidence") if e["kind"] == "outcome"]
    assert outcomes, "no trial completed"
    assert all(e["verified"] and e["independent"] for e in outcomes)
    adopted = [i for _, i in st.scan("cap_installs") if i["status"] == "active"]
    assert adopted, [e["data"] for e in outcomes]
    assert w.modules["caps"].summary(w, cid)["independent_adopters"] >= 1
    # every adoption was preceded by a positive verified outcome for that installer
    for i in adopted:
        ev = st.get("cap_evidence", i["evidence"])
        assert ev["data"]["effect_per_day"] > 0
    # the requests were posted by the firms themselves from their own losses, before the capability existed
    reqs = [r for _, r in st.scan("ops_requests")]
    cap_t = st.get("capabilities", cid)["t"]
    assert any(r["t"] < cap_t and r["evidence"] for r in reqs)
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []


def test_regeneration_of_an_open_world_with_triggers_and_sandboxed_services_is_exact():
    w = _world(2, n_consumers=1500)
    w.system_call("admin.join", {"handle": "ada", "endowment": 1_000_000})
    Runner(w, [("p:ada", CapacityInventor(wait_for_demand=False))], dt=3600, record=False).run(24 * 4)
    evs = [e.to_dict() for e in w.log]
    w2 = World.regenerate(w.spec, evs)
    assert w2.seq == w.seq and w2.last_hash == w.last_hash and w2.digest() == w.digest()
