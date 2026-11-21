"""Institutions: constitutions enforced at authorisation, proposals decided by simulated members' votes, approvals,
delegated budgets, governed amendment, profit sharing and incentive effects of the sharing rule."""
import crucible

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def _setup(seed=1, n_members=4):
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 300, "workers": 12}}, seed))
    w.system_call("admin.join", {"handle": "gia", "endowment": 3_000_000})
    ex = lambda t, **k: w.execute("p:gia", {"tool": t, "args": k})  # noqa: E731
    me = w.modules["bank"].primary_account(w.state, "p:gia")
    org = "o:giaco"
    assert ex("id.create_org", handle="giaco", name="Gia Coop", kind="company").ok
    assert ex("biz.register", org=org, account=me).ok
    acct = ex("bank.open_account", owner=org).result["account"]
    assert ex("bank.transfer", from_account=me, to=acct, amount=2_000_000).ok
    members = sorted(k for k, m in w.state.scan("minds") if m["kind"] == "person")[:n_members]
    idm = w.modules["identity"]
    w._run_system("setup", lambda ctx: [idm.add_member(ctx, org, p, "member") for p in members])
    return w, ex, org, acct, me, members


RULES = [{"match": {"tool": "bank.transfer", "min_amount": 10_000},
          "require": {"type": "vote", "electorate": "members", "threshold": 0.5, "quorum": 0.4, "window_s": 3 * DAY}},
         {"match": {"tool": "gov.amend"}, "require": {"type": "vote", "electorate": "members", "threshold": 0.66}}]


def test_constitution_blocks_unilateral_spending_and_votes_decide_it():
    w, ex, org, acct, me, members = _setup(1)
    assert ex("gov.adopt", org=org, rules=RULES).ok
    r = ex("bank.transfer", from_account=acct, to="p:gia", amount=50_000, memo="to myself")
    assert not r.ok and "constitution" in r.error["message"]               # even the owner cannot act alone
    assert ex("bank.transfer", from_account=acct, to=members[0], amount=5_000, memo="small").ok   # below threshold
    p = ex("gov.propose", org=org, tool="bank.transfer",
           args={"from_account": acct, "to": members[1], "amount": 40_000, "memo": "bonus"}, reason="good work").result
    t0 = w.now
    w.advance_time(w.now + 3 * DAY + 3600)
    pr = w.state.tables["proposals"][p["proposal"]]
    assert pr["status"] in ("executed", "rejected")
    votes = list(w.state.find("votes", "proposal", p["proposal"]))
    assert votes and all(v["voter"] in members + ["p:gia"] for v in votes)
    if pr["status"] == "executed":
        assert pr["decided_t"] > t0                                         # a vote takes time: decision cost
        assert any(t["memo"] == "bonus" for _, t in w.state.scan("txns"))
    # a self-serving proposal (the proposer pays itself) is voted down by independent members
    q = ex("gov.propose", org=org, tool="bank.transfer",
           args={"from_account": acct, "to": "p:gia", "amount": 900_000, "memo": "founder bonus"}).result
    w.advance_time(w.now + 3 * DAY + 3600)
    assert w.state.tables["proposals"][q["proposal"]]["status"] in ("rejected",)
    assert not any(t["memo"] == "founder bonus" for _, t in w.state.scan("txns"))


def test_amendment_is_itself_governed():
    w, ex, org, acct, me, members = _setup(2)
    assert ex("gov.adopt", org=org, rules=RULES).ok
    r = ex("gov.amend", org=org, rules=[])
    assert not r.ok and "constitution" in r.error["message"]
    assert w.state.tables["constitutions"][org]["rules"] == RULES


def test_owner_approval_is_fast_and_delegation_spends_a_capped_budget_without_votes():
    w, ex, org, acct, me, members = _setup(3, n_members=2)
    rules = [{"match": {"tool": "bank.transfer", "min_amount": 100_000},
              "require": {"type": "approval", "by": "owner"}},
             {"match": {"tool": "bank.transfer"},
              "require": {"type": "delegate", "to": ["p:gia"], "budget_cents_month": 60_000}}]
    assert ex("gov.adopt", org=org, rules=rules).ok
    assert ex("bank.transfer", from_account=acct, to=members[0], amount=30_000).ok
    assert ex("bank.transfer", from_account=acct, to=members[0], amount=30_000).ok
    r = ex("bank.transfer", from_account=acct, to=members[0], amount=30_000)
    assert not r.ok and "budget" in r.error["message"]                     # delegated budget exhausted
    p = ex("gov.propose", org=org, tool="bank.transfer",
           args={"from_account": acct, "to": members[0], "amount": 150_000}).result
    t0 = w.now
    v = ex("gov.vote", proposal=p["proposal"], approve=True)              # the single approver decides at once
    assert v.ok and w.state.tables["proposals"][p["proposal"]]["status"] == "executed"
    assert w.state.tables["proposals"][p["proposal"]]["decided_t"] == t0


def test_profit_sharing_pays_members_by_rule():
    w, ex, org, acct, me, members = _setup(4)
    assert ex("gov.adopt", org=org, rules=[], distribution={"basis": "equal", "share": 0.5}).ok
    # the coop earns money from an outside customer (a real sale)
    bank = w.modules["bank"]
    w.advance_time(5 * DAY)
    w._run_system("sale", lambda ctx: bank.move(ctx, "acct_external", acct, 400_000, "sale", "customers"))
    w.advance_time(31 * DAY)
    paid = [e for e in w.log if e.type == "gov.distributed" and e.payload["org"] == org]
    assert paid and paid[-1].payload["paid"] > 0 and paid[-1].payload["recipients"] == len(members) + 1


def test_equal_sharing_dilutes_incentives_as_the_group_grows():
    w, ex, org, acct, me, members = _setup(5, n_members=2)
    gov = w.modules["gov"]
    assert ex("gov.adopt", org=org, rules=[], distribution={"basis": "equal", "share": 1.0}).ok
    small = gov.effort(w.state, org, 3)
    big = gov.effort(w.state, org, 30)
    assert small > 1.0 > big                                                # ownership motivates; dilution shirks
    w._run_system("amend", lambda ctx: ctx.emit("gov.constitution", {"org": org, "rules": [], "distribution": {
        "basis": "labor", "share": 1.0}}, vis="public"))
    assert gov.effort(w.state, org, 30) > big                               # paying by hours restores effort
