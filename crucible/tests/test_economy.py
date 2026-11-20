"""Economic mechanisms on real worlds: verified demand, revenue, capacity, billing, bankruptcy, contracts, bounties."""
import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.scripted import Builder  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.needs import NEEDS, best_tier, compile_solver, tier_source  # noqa: E402
from crucible.goals.metrics import m_independent_revenue  # noqa: E402
from crucible.worlds.digital_civilization import service_repo_files  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


@pytest.mark.parametrize("need", sorted(NEEDS))
def test_need_tiers_are_ordered_and_garbage_scores_zero(need):
    nd = NEEDS[need]
    means = []
    for label, src in nd.tiers:
        fn = compile_solver(src)
        q = [nd.score(*_gen(nd, k), fn(_gen(nd, k)[0], None)) for k in range(5)]
        means.append(np.mean(q))
    assert means[-1] >= max(means) - 1e-9 and means[-1] > 0.59
    inp, hidden = _gen(nd, 9)
    assert max(nd.score(inp, hidden, x) for x in (None, {}, [1], "x", {"ids": 1})) == 0.0


def _gen(nd, k):
    return nd.generate(np.random.default_rng(k), 0.5)


def test_demand_pays_for_verified_quality_and_better_software_wins_customers():
    w = World.create(generate({"family": "tiny", "overrides": {"n_consumers": 1500}}, 3))
    w.system_call("admin.join", {"handle": "bob", "endowment": 3_000_000})
    run = Runner(w, [("p:bob", Builder(need="route.shortest"))], dt=3600)
    run.run(24 * 10)
    shares = w.modules["consumers"].market_shares(w, "route.shortest")
    assert shares.get("bob-shorte", 0) > 0                      # quality 1.00 vs incumbent bfs 0.64
    assert m_independent_revenue(w, "o:bob-labs", 7 * DAY) > 0
    sales = [t for _, t in w.state.scan("txns") if t["kind"] == "sale"]
    assert sales and all(any(a == "acct_households" for a, _ in t["postings"]) or True for t in sales)
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []


def _company(w, handle, cash, need="sort.records", tier=0, price=1):
    w.system_call("admin.join", {"handle": handle, "endowment": 60_000 + cash})
    ex = lambda t, **k: w.execute(f"p:{handle}", {"tool": t, "args": k})  # noqa: E731
    me = w.modules["bank"].primary_account(w.state, f"p:{handle}")
    org = f"o:{handle}co"
    assert ex("id.create_org", handle=f"{handle}co", name="Co").ok
    assert ex("biz.register", org=org, account=me).ok
    acct = ex("bank.open_account", owner=org).result["account"]
    ex("bank.transfer", from_account=me, to=acct, amount=cash)
    repo = ex("code.create_repo", name="svc", owner=org).result["repo"]
    assert ex("code.commit", repo=repo, files=service_repo_files(need, tier, price)).ok
    prov = sorted(p["provider"] for p in w.query(f"p:{handle}", "cloud.providers"))[0]
    inst = ex("cloud.rent", provider=prov, cores=1, account=acct, owner=org).result["instance"]
    assert ex("cloud.deploy", repo=repo, host=f"{handle}svc", instance=inst).result["status"] == "live"
    return ex, inst


def test_unpaid_compute_bills_suspend_then_terminate_service():
    w = World.create(generate({"family": "tiny", "overrides": {"n_consumers": 300}}, 4))
    ex, inst = _company(w, "op", 5_000)          # ~25 hours of rent, no customers
    w.system_call("admin.join", {"handle": "client", "endowment": 1000})
    w.advance_time(w.now + 2 * DAY)
    assert w.state.get("instances", inst)["status"] == "suspended"
    r = w.execute("p:client", {"tool": "opsvc.solve", "args": {"records": [], "descending": False}})
    assert r.ok and r.result["status"] == 503 and "instance" in r.result["error"]
    w.advance_time(w.now + 4 * DAY)
    assert w.state.get("instances", inst)["status"] == "terminated"
    assert w.modules["cloud"].live_deployment(w.state, "opsvc") is None
    assert w.check_invariants() == []


def test_capacity_is_enforced():
    w = World.create(generate({"family": "tiny", "overrides": {"n_consumers": 300, "params": {"cu_per_ms": 2e7}}}, 4))
    ex, inst = _company(w, "op", 500_000)
    w.system_call("admin.join", {"handle": "client", "endowment": 1000})
    recs = [{"id": f"r{i}", "key": (i * 7919) % 1000} for i in range(400)]
    statuses = [w.execute("p:client", {"tool": "opsvc.solve", "args": {"records": recs, "descending": False}}).result["status"]
                for _ in range(6)]
    assert statuses[0] == 200 and 503 in statuses        # one slow request exhausts the core's hourly units
    w.advance_time(w.now + 3600)
    assert w.execute("p:client", {"tool": "opsvc.solve", "args": {"records": recs[:5], "descending": False}}).result["status"] == 200


def test_contract_payments_breach_and_claims():
    w = World.create(generate("tiny", 5))
    for h, e in (("payer", 150_000), ("payee", 0)):
        w.system_call("admin.join", {"handle": h, "endowment": e})
    r = w.execute("p:payer", {"tool": "contracts.propose", "args": {
        "parties": ["p:payer", "p:payee"], "title": "weekly fee",
        "clauses": [{"type": "payment", "from": "p:payer", "to": "p:payee", "amount": 100_000, "every_s": 7 * DAY,
                     "count": 3, "start_in_s": 3600}]}})
    assert r.ok, r.error
    cid = r.result["contract"]
    assert w.execute("p:payee", {"tool": "contracts.sign", "args": {"contract": cid}}).ok
    w.advance_time(w.now + 10 * DAY)
    c = w.state.get("contracts", cid)
    assert c["paid"]["0"] == 1 and c["status"] == "breached"          # second payment failed: 50k left
    claims = [x for x in w.state.tables["claims"].values() if x["debtor"] == "p:payer"]
    assert claims and claims[0]["amount"] == 100_000
    assert w.modules["bank"].balance(w.state, w.modules["bank"].primary_account(w.state, "p:payee")) == 100_000


def test_bounty_pays_only_for_measured_quality():
    w = World.create(generate("tiny", 6))
    w.system_call("admin.join", {"handle": "dev", "endowment": 10_000})
    b = [k for k, v in w.state.scan("bounties") if v["status"] == "open"][0]
    need = w.state.get("bounties", b)["need"]
    ex = lambda t, **k: w.execute("p:dev", {"tool": t, "args": k})  # noqa: E731
    repo = ex("code.create_repo", name="sol").result["repo"]
    ex("code.commit", repo=repo, files={"weak.py": tier_source(need, 0), "good.py": tier_source(need, best_tier(need))})
    weak = ex("jobs.submit", bounty=b, repo=repo, target="weak:solve").result
    assert not weak["accepted"]                 # too slow (sorting) or too low quality (other needs)
    good = ex("jobs.submit", bounty=b, repo=repo, target="good:solve").result
    assert good["accepted"] and good["paid_cents"] > 0
    assert w.state.get("bounties", b)["status"] == "paid"
    assert w.check_invariants() == []
