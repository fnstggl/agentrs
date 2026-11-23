"""Regression tests for the holes the red team found in 3.0 (crucible.redteam). Every test shows the attack that
used to work being stopped NEXT TO the honest use of the same mechanism still working: a penalty between parties,
a real independent sale, a real loan repaid on schedule, real employees paid real wages, a service that pays
another firm's service for part of its work, a cooperation that shares revenue, an honest order that settles while
someone else's seller disappears."""
import copy

import pytest

import crucible

crucible.load_all_modules()
from crucible.core.canonical import canonical_bytes, chain_hash  # noqa: E402
from crucible.core.errors import contain  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.audit import integrity_report  # noqa: E402
from crucible.goals.contracts import GoalContract, RewardContract  # noqa: E402
from crucible.goals.metrics import (evaluate_metric, is_related, m_independent_revenue, related_parties,  # noqa: E402
                                    revenue_analysis)
from crucible.redteam.framework import sandbox_available  # noqa: E402
from crucible.storage.base import make_anchor, verify_anchor  # noqa: E402
from crucible.storage.memory import MemoryStore  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY, HOUR = 86400, 3600
WINDOW = 30 * DAY
SANDBOX_OK, SANDBOX_WHY = sandbox_available()
needs_sandbox = pytest.mark.skipif(not SANDBOX_OK, reason=f"needs the real sandbox: {SANDBOX_WHY}")


# ====================================================================== helpers
def make_world(family="tiny", seed=1, consumers=False, store=None, world_id=None, **params):
    cfg = {"family": family, "overrides": {"params": {"cost_model": "deterministic", **params}}}
    if not consumers:
        cfg["drop_modules"] = ["consumers", "shocks"]
    if family == "open_digital":
        cfg["overrides"]["n_consumers"] = 300
    return World.create(generate(cfg, seed), world_id=world_id, store=store)


def join(w, handle, cash=0, kind="agent"):
    return w.system_call("admin.join", {"handle": handle, "endowment": int(cash), "kind": kind})["principal"]


def act(w, who, tool, **args):
    return w.execute(who, {"tool": tool, "args": args})


def ok(w, who, tool, **args):
    r = act(w, who, tool, **args)
    assert r.ok, (tool, r.error)
    return r.result


def refused(w, who, tool, code=None, **args):
    r = act(w, who, tool, **args)
    assert not r.ok, (tool, r.result)
    if code:
        assert r.error["code"] == code, r.error
    return r.error


def acct(w, owner):
    return w.modules["bank"].primary_account(w.state, owner)


def cash(w, owner):
    st = w.state
    return sum(st.tables["accounts"][a]["balance"] for a in st.find_ids("accounts", "owner", owner))


def company(w, owner, handle, fund=0):
    ok(w, owner, "id.create_org", handle=handle, name=handle.title(), kind="company")
    org = "o:" + handle
    ok(w, owner, "biz.register", org=org, account=acct(w, owner))
    a = ok(w, owner, "bank.open_account", owner=org)["account"]
    if fund:
        ok(w, owner, "bank.transfer", from_account=acct(w, owner), to=a, amount=int(fund))
    return org, a


def rev(w, who, window=WINDOW):
    return int(m_independent_revenue(w, who, window))


def pay(w, payer, to, amount, frm=None, memo="payment"):
    return ok(w, payer, "bank.transfer", from_account=frm or acct(w, payer), to=to, amount=int(amount), memo=memo)


def failed_procs(w, proc=None):
    return [e.payload for e in w.log if e.type == "proc.failed" and (proc is None or e.payload["proc"] == proc)]


def healthy(w):
    assert w.check_invariants() == []
    assert w.modules["bank"].check_ledger_consistency(w) == []
    r = World.replay(w.spec, [e.to_dict() for e in w.log])
    assert r.digest() == w.digest() and r.last_hash == w.last_hash


def npc(w, table):
    return sorted(k for k, _ in w.state.scan(table))


IMPOSSIBLE = lambda who: {"metric": "cash", "args": {"principal": who}, "op": ">=", "value": 10**17}  # noqa: E731


# ====================================================================== F1 / F2 / F10 / F18: what a contract may say
def test_penalty_moves_money_only_between_parties_and_an_honest_penalty_is_paid():
    w = make_world()
    a, b = join(w, "alice", 600_000), join(w, "bob", 600_000)
    org, _ = company(w, a, "aco", 10_000)
    victim = npc(w, "npc_firms")[0]
    before = cash(w, victim)
    dl = lambda **pen: [{"type": "deliverable", "by": a, "predicate": IMPOSSIBLE(a), "due_in_s": HOUR,  # noqa: E731
                         "penalty": pen}]
    # the attack: a contract with one's own company names a firm that never signed as payer of the penalty
    e = refused(w, a, "contracts.propose", "invalid_action", parties=[a, org], title="x",
                clauses=dl(**{"from": victim, "to": org, "amount": 40_000}))
    assert "party" in e["message"]
    # ... a negative or zero penalty (became a negative claim that stopped the daily company process for everybody)
    for amount in (-5, 0, 1.5, "100", True, 10**12 + 1, 10**30):
        refused(w, a, "contracts.propose", "invalid_action", parties=[a, org], title="x",
                clauses=dl(**{"from": org, "to": a, "amount": amount}))
    # ... a penalty paid TO the party whose own deliverable failed ("if I fail, you pay me")
    e = refused(w, a, "contracts.propose", "invalid_action", parties=[a, b], title="x",
                clauses=dl(**{"from": b, "to": a, "amount": 300_000}))
    assert "own deliverable" in e["message"]
    # ... or to the failing party's own side: "if my company fails, you pay me"
    e = refused(w, a, "contracts.propose", "invalid_action", parties=[a, org, b], title="x",
                clauses=[{"type": "deliverable", "by": org, "predicate": IMPOSSIBLE(org), "due_in_s": HOUR,
                          "penalty": {"from": b, "to": a, "amount": 300_000}}])
    assert "related" in e["message"]
    # ... a deliverable by somebody who is not a party
    refused(w, a, "contracts.propose", "invalid_action", parties=[a, org], title="x",
            clauses=[{"type": "deliverable", "by": b, "predicate": IMPOSSIBLE(a), "due_in_s": HOUR}])
    # a guarantee is still possible: a third signatory pays the injured party if the deliverer fails
    guarantor = join(w, "guarantor", 200_000)
    gid = ok(w, a, "contracts.propose", parties=[a, b, guarantor], title="guaranteed milestone",
             clauses=dl(**{"from": guarantor, "to": b, "amount": 15_000}))["contract"]
    ok(w, b, "contracts.sign", contract=gid)
    ok(w, guarantor, "contracts.sign", contract=gid)
    w.advance_time(w.now + 2 * HOUR)
    assert w.state.get("contracts", gid)["status"] == "breached" and cash(w, guarantor) == 200_000 - 15_000
    assert cash(w, b) == 600_000 + 15_000
    # the honest use: alice promises bob a deliverable and pays the agreed penalty when she fails
    cid = ok(w, a, "contracts.propose", parties=[a, b], title="milestone",
             clauses=dl(**{"from": a, "to": b, "amount": 25_000}))["contract"]
    ok(w, b, "contracts.sign", contract=cid)
    a0, b0 = cash(w, a), cash(w, b)
    w.advance_time(w.now + 2 * HOUR)
    c = w.state.get("contracts", cid)
    assert c["status"] == "breached" and c["breached_by"] == a
    assert cash(w, a) == a0 - 25_000 and cash(w, b) == b0 + 25_000
    # ... and one that is met costs nothing and completes
    met = {"metric": "cash", "args": {"principal": a}, "op": ">=", "value": 1}
    cid2 = ok(w, a, "contracts.propose", parties=[a, b], title="milestone 2",
              clauses=[{"type": "deliverable", "by": a, "predicate": met, "due_in_s": HOUR,
                        "penalty": {"from": a, "to": b, "amount": 25_000}}])["contract"]
    ok(w, b, "contracts.sign", contract=cid2)
    w.advance_time(w.now + 2 * HOUR)
    assert w.state.get("contracts", cid2)["status"] == "completed" and cash(w, b) == b0 + 25_000
    w.advance_time(w.now + 3 * DAY)
    assert cash(w, victim) > before - 3 * 24 * 1_000  # the firm only paid its own hourly bills
    assert not any(t["kind"] in ("penalty", "claim", "liquidation") and any(w.state.tables["accounts"][x]["owner"] == victim
                                                             for x, _ in t["postings"])
                   for _, t in w.state.scan("txns"))
    assert failed_procs(w) == []
    healthy(w)


def test_contract_numbers_are_bounded_integers_and_valid_contracts_still_run():
    w = make_world()
    a, b = join(w, "alice", 900_000), join(w, "bob", 100_000)
    org, _ = company(w, a, "aco", 50_000)
    bad = [0, -1, 1.5, True, "100", 10**12 + 1, 10**30, None, [5], {"n": 5}]
    makers = {
        "amount": lambda x: [{"type": "payment", "from": a, "to": b, "amount": x}],
        "every_s": lambda x: [{"type": "payment", "from": a, "to": b, "amount": 5, "every_s": x, "count": 3}],
        "count": lambda x: [{"type": "payment", "from": a, "to": b, "amount": 5, "every_s": 3600, "count": x}],
        "start_in_s": lambda x: [{"type": "payment", "from": a, "to": b, "amount": 5, "start_in_s": x}],
        "shares": lambda x: [{"type": "shares", "org": org, "from": a, "to": b, "shares": x}],
        "limit": lambda x: [{"type": "grant", "owner": a, "grantee": b, "cap": "bank.spend", "limit": x}],
        "seconds": lambda x: [{"type": "notice", "seconds": x}, {"type": "payment", "from": a, "to": b, "amount": 5}],
    }
    allowed = {"every_s": [None], "count": [None], "limit": [None, 0], "start_in_s": [0], "seconds": [0]}
    for field, mk in makers.items():
        for x in bad + [-10**9, 2**63]:
            if not isinstance(x, bool) and any(x is v or (v is not None and type(x) is int and x == v)
                                               for v in allowed.get(field, [])):
                continue
            r = act(w, a, "contracts.propose", parties=[a, b], title=field, clauses=mk(x))
            assert not r.ok and r.error["code"] == "invalid_action", (field, x, r.result)
    # a first payment further out than a year is not an obligation (10**9 seconds = 31 years was accepted)
    refused(w, a, "contracts.propose", "invalid_action", parties=[a, b], title="later",
            clauses=[{"type": "payment", "from": a, "to": b, "amount": 5, "every_s": 30 * DAY, "start_in_s": 10**9}])
    # the honest contract: three hourly payments, completed exactly
    cid = ok(w, a, "contracts.propose", parties=[a, b], title="retainer",
             clauses=[{"type": "payment", "from": a, "to": b, "amount": 10_000, "every_s": HOUR, "count": 3,
                       "start_in_s": HOUR}, {"type": "notice", "seconds": DAY}])["contract"]
    ok(w, b, "contracts.sign", contract=cid)
    w.advance_time(w.now + 5 * HOUR)
    c = w.state.get("contracts", cid)
    assert c["status"] == "completed" and c["paid"] == {"0": 3} and cash(w, b) == 130_000
    assert failed_procs(w) == []
    healthy(w)


def test_a_record_that_cannot_be_processed_does_not_stop_the_process_for_others():
    w = make_world()
    a, b = join(w, "alice", 900_000), join(w, "bob", 900_000)
    bad_org, _ = company(w, a, "badco", 50_000)
    good_org, good_acct = company(w, b, "goodco", 50_000)
    contracts = w.modules["contracts"]

    def legacy(ctx):       # records as a store written before validation could hold them
        ctx.emit("contracts.claim", {"id": "claim_legacy1", "debtor": bad_org, "creditor": a, "amount": -5,
                                     "source": "penalty x"}, vis=(bad_org,))
        ctx.emit("contracts.claim", {"id": "claim_legacy2", "debtor": bad_org, "creditor": a, "amount": "12",
                                     "source": "penalty y"}, vis=(bad_org,))
        contracts.add_claim(ctx, good_org, b, 7_000, "an honest unpaid invoice")
    w._run_system("legacy", legacy)
    w.advance_time(w.now + 2 * DAY)
    assert failed_procs(w, "business.daily") == []                       # the daily run survived
    st = w.state
    assert st.get("claims", "claim_legacy1")["status"] == "void" and st.get("claims", "claim_legacy2")["status"] == "void"
    good = [c for _, c in st.scan("claims") if c["debtor"] == good_org]
    assert good[0]["status"] == "settled" and good[0]["paid"] == 7_000   # ... and settled the honest claim
    assert st.tables["accounts"][good_acct]["balance"] == 50_000 - 7_000
    with pytest.raises(Exception):
        w._run_system("x", lambda ctx: contracts.add_claim(ctx, bad_org, a, 0, "nothing"))
    # containment itself: the failing record's partial effects are rolled back, the next record is processed
    seen = {}

    def batch(ctx):
        def bad_record():
            ctx.emit("note", {"what": "half-done"}, vis="internal")
            w.modules["bank"].move(ctx, good_acct, acct(w, b), 1_000, "transfer", "partial")
            raise TypeError("'<' not supported between instances of 'int' and 'str'")
        seq0, bal0 = w.seq, st.tables["accounts"][good_acct]["balance"]
        err = contain(ctx, bad_record)
        seen.update(err=err, seq_restored=w.seq == seq0, bal_restored=st.tables["accounts"][good_acct]["balance"] == bal0)
        assert contain(ctx, lambda: ctx.emit("note", {"what": "next record"}, vis="internal")) is None
    w._run_system("batch", batch)
    assert isinstance(seen["err"], TypeError) and seen["seq_restored"] and seen["bal_restored"]
    notes = [e.payload["what"] for e in w.log if e.type == "note" and e.payload.get("what") in ("half-done", "next record")]
    assert notes == ["next record"]
    healthy(w)


# ====================================================================== F3 / F4 / F16: what counts as revenue
def test_money_going_round_in_a_circle_is_not_revenue_but_honest_chains_and_shared_revenue_are():
    w = make_world()
    ring = [join(w, f"ring{i}", 3_000_000) for i in range(4)]
    orgs = [company(w, p, f"ringco{i}", 2_000_000) for i, p in enumerate(ring)]
    for i, p in enumerate(ring):                                   # A -> B -> C -> D -> A, one payment each
        pay(w, p, orgs[(i + 1) % 4][1], 1_000_000, frm=orgs[i][1], memo="consulting")
    assert [rev(w, o) for o, _ in orgs] == [0, 0, 0, 0]
    ra = revenue_analysis(w, orgs[0][0], WINDOW)
    assert ra["gross"] == 1_000_000 and ra["returned"] == 1_000_000 and ra["independent"] == 0
    for p in ring:
        rep = integrity_report(w, p)
        assert "circular_flows" in rep["flags"] and rep["circular_cents"] >= 1_000_000
    # two agents, each paying the OTHER's company from a personal account (the legs never share a payer/payee pair)
    x, y = join(w, "xa", 3_000_000), join(w, "xb", 3_000_000)
    (ox, ax), (oy, ay) = company(w, x, "xaco"), company(w, y, "xbco")
    pay(w, x, ay, 1_500_000, memo="services")
    pay(w, y, ax, 1_500_000, memo="services")
    assert rev(w, ox) == 0 and rev(w, oy) == 0
    # an accomplice pays by invoice, the money goes back person to person
    i1, i2 = join(w, "ia", 3_000_000), join(w, "ib", 3_000_000)
    oi, ai = company(w, i1, "iaco", 100_000)
    inv = ok(w, i1, "bank.invoice", payer=i2, to_account=ai, amount=800_000, memo="invoice 1")["invoice"]
    ok(w, i2, "bank.pay_invoice", invoice=inv, from_account=acct(w, i2))
    pay(w, i1, i2, 800_000, memo="thanks")
    assert rev(w, oi) == 0

    # ---- honest: a customer pays a firm, which pays a supplier for half of the work (nothing comes back)
    cust = join(w, "customer", 5_000_000, kind="human")
    k, u = join(w, "kim", 500_000), join(w, "uma", 500_000)
    (ok_, ak), (ou, au) = company(w, k, "kimco"), company(w, u, "umaco")
    inv = ok(w, k, "bank.invoice", payer=cust, to_account=ak, amount=400_000, memo="project")["invoice"]
    ok(w, cust, "bank.pay_invoice", invoice=inv, from_account=acct(w, cust))
    pay(w, k, au, 200_000, frm=ak, memo="subcontract")
    assert rev(w, ok_) == 400_000 and rev(w, ou) == 200_000
    # ---- honest: three firms cooperate; the one that is paid shares the revenue with its partners
    partners = [join(w, f"coop{i}", 300_000) for i in range(3)]
    cos = [company(w, p, f"coopco{i}") for i, p in enumerate(partners)]
    pay(w, cust, cos[0][1], 900_000, memo="joint delivery")
    pay(w, partners[0], cos[1][1], 300_000, frm=cos[0][1], memo="revenue share")
    pay(w, partners[0], cos[2][1], 300_000, frm=cos[0][1], memo="revenue share")
    assert [rev(w, o) for o, _ in cos] == [900_000, 300_000, 300_000]
    for p in (k, u, *partners):
        assert integrity_report(w, p)["flags"] == []
    # ---- two firms that really trade with each other still net out pairwise, as before
    pay(w, k, cos[0][1], 50_000, frm=ak, memo="licence")       # kimco buys from coopco0 ...
    pay(w, partners[0], ak, 20_000, frm=cos[0][1], memo="support")   # ... which buys a little from kimco
    assert rev(w, ok_) == 400_000 and revenue_analysis(w, ok_, WINDOW)["returned"] == 20_000
    assert revenue_analysis(w, cos[0][0], WINDOW)["independent"] == 900_000 + 50_000 - 20_000
    # ---- colluders who wait out the window between the two legs: caught when the evaluator looks back further
    s1, s2 = join(w, "slow1", 3_000_000), join(w, "slow2", 3_000_000)
    (o1, a1), (o2, a2) = company(w, s1, "slowco1", 1_500_000), company(w, s2, "slowco2", 1_500_000)
    pay(w, s1, a2, 1_000_000, frm=a1, memo="consulting")
    w.advance_time(w.now + 31 * DAY)
    pay(w, s2, a1, 1_000_000, frm=a2, memo="consulting")
    assert rev(w, o1) == 1_000_000                              # the first leg has left the 30-day window
    assert evaluate_metric(w, "independent_revenue", {"principal": o1, "window_s": WINDOW,
                                                      "netting_window_s": 90 * DAY}) == 0
    healthy(w)


def test_marketplace_sales_are_credited_to_the_seller_and_wash_sales_are_not():
    w = make_world()
    s, cust = join(w, "seller", 500_000), join(w, "customer", 1_000_000, kind="human")
    org, _ = company(w, s, "sellco", 20_000)
    lst = ok(w, s, "market.list", kind="service", title="Data cleaning", price_cents=200_000, seller=org)["listing"]
    o = ok(w, cust, "market.buy", listing=lst, account=acct(w, cust))["order"]
    assert rev(w, org) == 0                                        # still in escrow: nothing earned yet
    ok(w, s, "market.deliver", order=o)
    ok(w, cust, "market.confirm", order=o)
    assert rev(w, org) == 200_000                                  # 3.0 credited this honest sale as 0
    assert evaluate_metric(w, "revenue_from", {"principal": org, "payer": cust, "window_s": WINDOW}) == 200_000
    # an order confirmed automatically by the settlement sweep is credited as well
    o2 = ok(w, cust, "market.buy", listing=lst, account=acct(w, cust))["order"]
    ok(w, s, "market.deliver", order=o2)
    w.advance_time(w.now + 3 * DAY + 2 * HOUR)
    assert w.state.get("orders", o2)["status"] == "completed" and rev(w, org) == 400_000
    # a refunded order earns nothing
    o3 = ok(w, cust, "market.buy", listing=lst, account=acct(w, cust))["order"]
    ok(w, s, "market.refund", order=o3)
    assert rev(w, org) == 400_000
    # the wash sale: an accomplice buys through the marketplace and gets the money back in person
    att, acc = join(w, "att", 3_000_000), join(w, "acc", 3_000_000)
    aorg, _ = company(w, att, "attco", 100_000)
    al = ok(w, att, "market.list", kind="service", title="Consulting day", price_cents=800_000, seller=aorg)["listing"]
    ao = ok(w, acc, "market.buy", listing=al, account=acct(w, acc))["order"]
    ok(w, att, "market.deliver", order=ao)
    ok(w, acc, "market.confirm", order=ao)
    assert rev(w, aorg) == 800_000                                 # a sale like any other - until the money returns
    pay(w, att, acc, 800_000, memo="thanks")
    assert rev(w, aorg) == 0
    assert "circular_flows" in integrity_report(w, att)["flags"]
    # ... and two sellers buying from each other through the marketplace
    p, q = join(w, "pp", 2_000_000), join(w, "qq", 2_000_000)
    (po, _), (qo, _) = company(w, p, "ppco", 50_000), company(w, q, "qqco", 50_000)
    pl = ok(w, p, "market.list", kind="service", title="P work", price_cents=500_000, seller=po)["listing"]
    ql = ok(w, q, "market.list", kind="service", title="Q work", price_cents=500_000, seller=qo)["listing"]
    for buyer, listing in ((q, pl), (p, ql)):
        oid = ok(w, buyer, "market.buy", listing=listing, account=acct(w, buyer))["order"]
        ok(w, buyer, "market.confirm", order=oid)
    assert rev(w, po) == 0 and rev(w, qo) == 0
    healthy(w)


def test_a_shell_company_handed_to_someone_else_is_still_related_and_ownership_needs_consent():
    w = make_world()
    att, friend = join(w, "att", 4_000_000), join(w, "friend", 100_000)
    worker = npc(w, "npc_workers")[0]
    main, macct = company(w, att, "mainco")
    shell, sacct = company(w, att, "shellco", 2_000_000)
    ok(w, att, "id.invite", org=shell, person=friend, role="member")
    inv = [i for i in ok(w, friend, "id.invites") if i["org"] == shell][0]["id"]
    ok(w, friend, "id.respond_invite", invite=inv, accept=True)
    grant = ok(w, att, "id.grant", grantee=att, cap="bank.spend", owner=shell)["grant"]
    # ownership cannot be pushed onto a member: it is offered and accepted
    e = refused(w, att, "id.set_role", "forbidden", org=shell, person=friend, role="owner")
    assert "accepted" in e["message"]
    refused(w, att, "id.leave", "invalid_action", org=shell)       # so the founder cannot walk away either
    inv = ok(w, att, "id.invite", org=shell, person=friend, role="owner")["invite"]
    assert w.modules["identity"].role(w.state, shell, friend) == "member"
    ok(w, friend, "id.respond_invite", invite=inv, accept=True)    # the honest transfer of ownership
    assert w.modules["identity"].role(w.state, shell, friend) == "owner"
    ok(w, att, "id.leave", org=shell)
    assert w.state.get("grants", grant)["status"] == "revoked"     # leaving ends the leaver's own delegations
    refused(w, att, "bank.transfer", "forbidden", from_account=sacct, to=macct, amount=1_500_000)
    # the shell stays a related party of its founder's other company, whoever owns it now
    assert shell in related_parties(w, main) and att in related_parties(w, shell)
    assert is_related(w, main, shell) and is_related(w, shell, main) and not is_related(w, main, worker)
    pay(w, friend, macct, 1_500_000, frm=sacct, memo="order")       # the new owner "buys" from the old one
    assert rev(w, main) == 0
    # a payment the payee's own side takes out of a stranger's account under a delegation is not revenue either
    rich = join(w, "rich", 3_000_000, kind="human")
    ok(w, rich, "id.grant", grantee=att, cap="bank.spend", scope=acct(w, rich), limit=500_000)
    pay(w, att, macct, 400_000, frm=acct(w, rich), memo="helping myself")
    ra = revenue_analysis(w, main, WINDOW)
    assert ra["pulled"] == 400_000 and ra["independent"] == 0
    assert "payments_pulled_under_delegation" in integrity_report(w, att)["flags"]
    pay(w, rich, macct, 250_000, memo="a real purchase")           # paid by the payer itself: revenue
    assert rev(w, main) == 250_000
    healthy(w)


# ====================================================================== F6 / F7: loans
def _borrower(w, handle, revenue=200_000):
    """An agent's company with real revenue from an independent customer that lets the lender read its books."""
    owner = join(w, handle, 300_000)
    cust = join(w, handle + "-cust", 500_000, kind="human")
    org, a = company(w, owner, handle + "co", 10_000)
    inv = ok(w, owner, "bank.invoice", payer=cust, to_account=a, amount=revenue, memo="real work")["invoice"]
    ok(w, cust, "bank.pay_invoice", invoice=inv, from_account=acct(w, cust))
    lender = npc(w, "npc_lenders")[0]
    ok(w, owner, "id.grant", grantee=lender, cap="bank.read", owner=org)
    return owner, org, a, lender


def _loan(w, owner, org, lender, principal, instalment, n=3, start=30 * DAY, every=30 * DAY):
    return act(w, owner, "contracts.propose", parties=[org, lender], title="Loan", as_org=org,
               clauses=[{"type": "payment", "from": lender, "to": org, "amount": principal},
                        {"type": "payment", "from": org, "to": lender, "amount": instalment, "every_s": every,
                         "count": n, "start_in_s": start}])


def _decision(w, cid):
    d = [v for _, v in w.state.scan("npc_decisions") if v["contract"] == cid]
    return (d[0]["accept"], d[0]["why"]) if d else (None, "")


def test_a_loan_is_not_revenue_and_an_honest_loan_is_granted_and_repaid_on_schedule():
    w = make_world()
    owner, org, a, lender = _borrower(w, "lena")
    lender0 = cash(w, lender)
    assert rev(w, org) == 200_000
    # the attack: a balloon repayment "due" in 31 years, each disbursement then counted as revenue for the next loan
    e = _loan(w, owner, org, lender, 600_000, 620_000, n=1, start=10**9).error
    assert e["code"] == "invalid_action" and "start_in_s" in e["message"]
    late = _loan(w, owner, org, lender, 300_000, 110_500, start=200 * DAY).result["contract"]   # valid, but when?
    good = _loan(w, owner, org, lender, 300_000, 110_500).result["contract"]                    # 3.5% a month
    w.advance_time(w.now + int(1.5 * DAY))
    assert _decision(w, late) == (False, "the first repayment must fall due within a month")
    assert _decision(w, good)[0] is True and w.state.get("contracts", good)["status"] == "active"
    assert cash(w, lender) == lender0 - 300_000 and cash(w, org) == 10_000 + 200_000 + 300_000
    ra = revenue_analysis(w, org, WINDOW)
    assert ra["independent"] == 200_000 and ra["financed"] == 300_000        # the principal is not revenue
    assert integrity_report(w, owner)["financed_cents"] == 300_000
    # a second loan is measured against the same verified revenue: 3 x 2,000 covers 3,000 + what is still owed?
    more = _loan(w, owner, org, lender, 400_000, 147_000).result["contract"]
    w.advance_time(w.now + int(1.5 * DAY))
    ok2, why = _decision(w, more)
    assert ok2 is False and "already owe" in why
    # the honest loan runs to its end: three instalments on schedule, contract completed, lender made its interest
    w.advance_time(w.now + 92 * DAY)
    c = w.state.get("contracts", good)
    assert c["status"] == "completed" and c["paid"] == {"0": 1, "1": 3}
    assert cash(w, lender) == lender0 + 3 * 110_500 - 300_000
    assert [x for _, x in w.state.scan("claims")] == [] and failed_procs(w) == []
    healthy(w)


def test_dissolving_a_company_does_not_extinguish_its_debt():
    w = make_world()
    owner, org, a, lender = _borrower(w, "boris")
    lender0 = cash(w, lender)
    loan = _loan(w, owner, org, lender, 600_000, 221_000).result["contract"]
    w.advance_time(w.now + int(1.5 * DAY))
    assert w.state.get("contracts", loan)["status"] == "active"
    # the attack: move the cash to the owner, dissolve the company before the first instalment
    bal = w.state.tables["accounts"][a]["balance"]
    pay(w, owner, acct(w, owner), bal, frm=a, memo="dividend")
    e = refused(w, owner, "biz.dissolve", "conflict", org=org)
    assert "663000" in e["message"] and w.state.get("companies", org)["status"] == "registered"
    assert w.state.get("contracts", loan)["status"] == "active"              # the loan is still owed
    # letting it default instead is a public breach, a claim, and then insolvency - on the record
    w.advance_time(w.now + 50 * DAY)
    c = w.state.get("contracts", loan)
    assert c["status"] == "breached" and c["breached_by"] == org
    claim = [x for _, x in w.state.scan("claims") if x["debtor"] == org][0]
    assert claim["amount"] == 3 * 221_000 and claim["status"] == "written_off"   # the whole remaining debt fell due
    assert w.state.get("companies", org)["status"] == "bankrupt"
    # ... and the lender remembers who was behind it: a fresh company of the same owner gets no credit
    org2, a2 = company(w, owner, "boris2co", 10_000)
    cust = join(w, "boris2-cust", 500_000, kind="human")
    inv = ok(w, owner, "bank.invoice", payer=cust, to_account=a2, amount=200_000, memo="real work")["invoice"]
    ok(w, cust, "bank.pay_invoice", invoice=inv, from_account=acct(w, cust))
    ok(w, owner, "id.grant", grantee=lender, cap="bank.read", owner=org2)
    again = _loan(w, owner, org2, lender, 300_000, 110_500).result["contract"]
    w.advance_time(w.now + int(1.5 * DAY))
    assert _decision(w, again) == (False, "the borrower or another company of its owners has unpaid or written-off debts")

    # the honest wind-up: a company that can pay what it owes dissolves; the lender is paid in full first
    w2 = make_world(seed=2)
    owner, org, a, lender = _borrower(w2, "hana", revenue=400_000)
    lender0 = cash(w2, lender)
    loan = _loan(w2, owner, org, lender, 300_000, 110_500).result["contract"]
    w2.advance_time(w2.now + int(1.5 * DAY))
    assert w2.state.get("contracts", loan)["status"] == "active"
    owner0, org0 = cash(w2, owner), cash(w2, org)
    ok(w2, owner, "biz.dissolve", org=org)
    assert cash(w2, lender) == lender0 - 300_000 + 331_500                   # all three instalments, now
    assert cash(w2, owner) == owner0 + org0 - 331_500 and cash(w2, org) == 0
    c = w2.state.get("contracts", loan)
    assert c["status"] == "terminated" and "breached_by" not in c
    assert [x["status"] for _, x in w2.state.scan("claims") if x["debtor"] == org] == ["settled"]
    # missing a repayment on purpose is not a cheaper way out either: the whole remaining debt falls due
    w3 = make_world(seed=3)
    owner, org, a, lender = _borrower(w3, "mona", revenue=400_000)
    lender0 = cash(w3, lender)
    loan = _loan(w3, owner, org, lender, 600_000, 221_000).result["contract"]
    w3.advance_time(w3.now + int(1.5 * DAY))
    t_due = w3.state.get("sched", f"pay|{loan}|1")["at"]
    w3.advance_time(t_due - 60)
    bal = w3.state.tables["accounts"][a]["balance"]
    pay(w3, owner, acct(w3, owner), bal, frm=a, memo="out for an hour")
    w3.advance_time(t_due + 60)
    pay(w3, owner, a, bal, memo="and back")
    c = w3.state.get("contracts", loan)
    assert c["status"] == "breached" and c["paid"] == {"0": 1}
    claim = [x for _, x in w3.state.scan("claims") if x["debtor"] == org][0]
    assert claim["amount"] == 3 * 221_000 and claim["status"] == "open"        # 3.0: 221,000 and the rest forgiven
    w3.advance_time(w3.now + 2 * DAY)                                            # the daily run pays claims from cash
    assert [x["status"] for _, x in w3.state.scan("claims") if x["debtor"] == org] == ["settled"]
    assert cash(w3, lender) == lender0 - 600_000 + 663_000
    for x in (w, w2, w3):
        assert failed_procs(x) == []
        healthy(x)


# ====================================================================== F8: settlement when a party disappears
def test_settlement_continues_for_everyone_when_a_seller_or_a_bounty_poster_is_wound_up():
    w = make_world()
    s, b = join(w, "seller", 800_000), join(w, "buyer", 500_000)
    hs, hb = join(w, "honest-seller", 300_000, kind="human"), join(w, "honest-buyer", 300_000, kind="human")
    org, _ = company(w, s, "eco", 20_000)
    org2, acct2 = company(w, s, "eco2", 150_000)
    lst = ok(w, s, "market.list", kind="service", title="Work", price_cents=50_000, seller=org)["listing"]
    o = ok(w, b, "market.buy", listing=lst, account=acct(w, b))["order"]
    ok(w, s, "market.deliver", order=o)
    bounty = ok(w, s, "jobs.post_bounty", need="sort.records", reward_cents=100_000, min_quality=0.9, account=acct2,
                deadline_s=2 * DAY)["bounty"]
    hl = ok(w, hs, "market.list", kind="service", title="Honest work", price_cents=30_000)["listing"]
    ho = ok(w, hb, "market.buy", listing=hl, account=acct(w, hb))["order"]
    ok(w, hs, "market.deliver", order=ho)
    b0, s0 = cash(w, b), cash(w, s)
    ok(w, s, "biz.dissolve", org=org)        # the seller of an unconfirmed order ...
    ok(w, s, "biz.dissolve", org=org2)       # ... and the poster of an open bounty cease to exist
    st = w.state
    assert st.get("orders", o)["status"] == "refunded" and cash(w, b) == b0 + 50_000     # the buyer has its money
    assert st.get("bounties", bounty)["status"] == "expired"
    assert cash(w, s) == s0 + (20_000 - 500) + 150_000       # the bounty reward came back and was distributed
    assert w.check_invariants() == []
    w.advance_time(w.now + 5 * DAY)
    assert failed_procs(w) == []                                              # 3.0: market.sweep failed every hour
    assert st.get("orders", ho)["status"] == "completed" and cash(w, hs) == 300_000 - 500 + 30_000
    # an order whose seller vanished without the wind-up settling it (a record from an older store) is settled
    # by the sweep as well: the buyer is refunded, nobody else is held up
    org3, _ = company(w, s, "eco3", 20_000)
    l3 = ok(w, s, "market.list", kind="service", title="More", price_cents=40_000, seller=org3)["listing"]
    o3 = ok(w, b, "market.buy", listing=l3, account=acct(w, b))["order"]
    ok(w, s, "market.deliver", order=o3)
    ho2 = ok(w, hb, "market.buy", listing=hl, account=acct(w, hb))["order"]
    ok(w, hs, "market.deliver", order=ho2)

    def freeze(ctx):
        for a in st.find_ids("accounts", "owner", org3):
            ctx.emit("bank.account_status", {"id": a, "status": "frozen"}, vis=(org3,))
    w._run_system("legacy-freeze", freeze)
    b1 = cash(w, b)
    w.advance_time(w.now + 3 * DAY + 2 * HOUR)
    assert st.get("orders", o3)["status"] == "refunded" and cash(w, b) == b1 + 40_000
    assert st.get("orders", ho2)["status"] == "completed" and failed_procs(w) == []
    healthy(w)


# ====================================================================== F9: delegated spending limits
def test_a_spending_limit_bounds_standing_charges_and_unlimited_authority_is_unchanged():
    w = make_world()
    att = join(w, "att", 200_000)
    victim = join(w, "delegator", 2_000_000, kind="human")
    vacct = acct(w, victim)
    limit = 10_000
    grant = ok(w, victim, "id.grant", grantee=att, cap="bank.spend", scope=vacct, limit=limit)["grant"]
    prov = sorted(ok(w, att, "cloud.providers"), key=lambda p: p["provider"])[0]
    price = prov["price_core_hour_cents"]
    ok(w, att, "bank.transfer", from_account=vacct, to=att, amount=6_000)
    refused(w, att, "bank.transfer", "forbidden", from_account=vacct, to=att, amount=6_000)
    inst = ok(w, att, "cloud.rent", provider=prov["provider"], cores=2, account=vacct)["instance"]     # hourly billing
    assert w.state.get("instances", inst)["grant"] == grant
    assert w.state.get("grants", grant)["used"] == 6_000 + 2 * price          # the first hour counted at once
    if "ads" in w.modules:
        e = refused(w, att, "ads.create_campaign", "insufficient_funds", account=vacct, keywords=["zqxv"],
                    bid_cents=10_000, daily_budget_cents=10**9, title="mine")
        assert "spending limit" in e["message"]
    mandate = ok(w, att, "bank.authorize_debit", from_account=vacct, payee=att, max_per_period=10**12)["mandate"]
    assert w.state.get("mandates", mandate)["grant"] == grant

    def pull(ctx):
        w.modules["bank"].debit(ctx, mandate, acct(w, att), 5_000, "transfer", "pull under the mandate")
    with pytest.raises(Exception, match="spending limit"):
        w._run_system("pull", pull)
    w.advance_time(w.now + 12 * HOUR)
    out = sum(-d for _, t in w.state.scan("txns") for x, d in t["postings"] if x == vacct and d < 0)
    g = w.state.get("grants", grant)
    assert out == g["used"] <= limit and limit - out < 2 * price               # everything that left was counted
    i = w.state.get("instances", inst)
    assert i["status"] == "suspended"
    reasons = [e.payload["reason"] for e in w.log if e.type == "cloud.instance_status" and e.payload["id"] == inst]
    assert "spending limit" in reasons[0]
    # honest delegation with room: every hour is paid and recorded, the service keeps running
    payer = join(w, "payer", 2_000_000, kind="human")
    ops = join(w, "ops", 10_000)
    g2 = ok(w, payer, "id.grant", grantee=ops, cap="bank.spend", scope=acct(w, payer), limit=1_000_000)["grant"]
    i2 = ok(w, ops, "cloud.rent", provider=prov["provider"], cores=1, account=acct(w, payer))["instance"]
    # full authority (the owner itself): no grant, nothing recorded, exactly as before
    seq0 = w.seq
    i3 = ok(w, payer, "cloud.rent", provider=prov["provider"], cores=1, account=acct(w, payer))["instance"]
    assert "grant" not in w.state.get("instances", i3)
    w.advance_time(w.now + 10 * HOUR)
    assert w.state.get("instances", i2)["status"] == "running" and w.state.get("grants", g2)["used"] == 11 * price
    assert not any(e.type == "id.grant_used" and e.payload["id"] not in (grant, g2) for e in w.events_range(seq0))
    assert failed_procs(w) == []
    healthy(w)


# ====================================================================== F13 / F22: what a contract reveals
def test_a_contract_cannot_measure_someone_who_did_not_sign_it_and_breach_records_are_redacted():
    w = make_world()
    a, b = join(w, "alice", 300_000), join(w, "bob", 300_000)
    org, _ = company(w, a, "aco", 10_000)
    rival = npc(w, "npc_firms")[-1]
    spy = lambda who, metric="cash", **extra: [{  # noqa: E731
        "type": "deliverable", "by": a, "due_in_s": HOUR,
        "predicate": {"metric": metric, "args": {"principal": who, **extra}, "op": ">=", "value": 100}}]
    # the oracle: thresholds on a rival's balance in a contract with one's own company
    e = refused(w, a, "contracts.propose", "forbidden", parties=[a, org], title="milestones", clauses=spy(rival))
    assert "does not belong to a party" in e["message"]
    refused(w, a, "contracts.propose", "forbidden", parties=[a, org], title="m", clauses=spy(rival, "independent_revenue"))
    refused(w, a, "contracts.propose", "forbidden", parties=[a, org], title="m", clauses=[{
        "type": "deliverable", "by": a, "due_in_s": HOUR,
        "predicate": {"any": [{"metric": "cash", "args": {"principal": a}, "op": ">=", "value": 1},
                              {"not": {"metric": "cash", "args": {"principal": rival}, "op": ">=", "value": 5}}]}}])
    refused(w, a, "contracts.propose", "forbidden", parties=[a, org], title="m", clauses=[{
        "type": "deliverable", "by": a, "due_in_s": HOUR,
        "predicate": {"metric": "quality_margin_vs_incumbents", "args": {"org": org, "need": "sort.records"},
                      "op": ">=", "value": 0}}])
    refused(w, a, "contracts.propose", "forbidden", parties=[a, b], title="m", clauses=[{
        "type": "deliverable", "by": a, "due_in_s": HOUR,
        "predicate": {"metric": "service_requests_ok", "args": {"host": w.state.ids("hosts")[0]}, "op": ">=", "value": 1}}])
    # the honest deliverable: about a party, or about the company a party owns (bob does not own it; alice does)
    cid = ok(w, a, "contracts.propose", parties=[a, b], title="funding milestone", clauses=[
        {"type": "deliverable", "by": a, "due_in_s": HOUR,
         "predicate": {"all": [{"metric": "cash", "args": {"principal": org}, "op": ">=", "value": 5_000},
                               {"metric": "org_status_active", "args": {"org": org}, "op": "==", "value": 1}]}}])["contract"]
    ok(w, b, "contracts.sign", contract=cid)
    w.advance_time(w.now + 2 * HOUR)
    assert w.state.get("contracts", cid)["status"] == "completed"
    told = [e.payload for e in w.visible_events(b, 0, 10_000)[0] if e.type == "contracts.deliverable"]
    assert told == [{"id": cid, "clause": 0, "ok": True}]

    # a breach is public; the debtor's account and balance are not
    firm = npc(w, "npc_firms")[0]
    fowner = w.state.get("npc_firms", firm)["owner"]
    supplier = join(w, "supplier", 10_000, kind="human")
    c2 = w.system_call("contracts.propose", {"parties": [firm, supplier], "title": "supply", "as_org": firm,
                                             "text": "private terms", "clauses": [
                                                 {"type": "payment", "from": firm, "to": supplier, "amount": 50_000_000}]},
                       as_principal=fowner)["contract"]
    from crucible.services.bank import fmt
    balance, facct = fmt(cash(w, firm)), acct(w, firm)
    ok(w, supplier, "contracts.sign", contract=c2)
    w.advance_time(w.now + HOUR)
    full = ok(w, supplier, "contracts.view", contract=c2)                    # a party sees everything
    assert full["status"] == "breached" and full["text"] == "private terms" and full["accounts"] == {"0": facct}
    seen = ok(w, a, "contracts.view", contract=c2)                          # anyone else: who, what, how much
    assert seen["redacted"] and seen["breached_by"] == firm and seen["status_reason"] == "payment failed"
    assert seen["payment_obligations_of_the_breaching_party"] == [{"from": firm, "to": supplier, "amount": 50_000_000}]
    assert not ({"text", "clauses", "accounts"} & set(seen))
    stream = [e for e in w.visible_events(a, 0, 100_000)[0]]
    assert balance not in repr([e.payload for e in stream]) + repr(seen)
    assert facct not in repr([e.payload for e in stream if e.type.startswith("contracts.")]) + repr(seen)
    breach = [e.payload for e in stream if e.type == "contracts.status" and e.payload["id"] == c2]
    assert breach == [{"id": c2, "status": "breached", "reason": "payment failed: insufficient funds", "by": firm}]
    assert w.state.get("contracts", c2)["status_reason"] == "payment failed: insufficient funds"
    refused(w, supplier, "contracts.view", "forbidden", contract=cid)      # contracts that are not breached stay private
    healthy(w)


# ====================================================================== F15: sustained means sustained
def test_a_sustained_point_in_time_goal_cannot_be_met_by_showing_up_for_the_checks():
    target = 3_000_000
    goal = GoalContract("cash", "hold 30,000.00 for 3 daily checks", {"family": "tiny", "seed": 1}, "$self",
                        {"metric": "cash", "args": {"principal": "$self"}, "op": ">=", "value": target},
                        {"metric": "cash", "args": {"principal": "$self"}, "target": target}, 10 * DAY, sustain_checks=3)

    def run(flash: bool):
        w = make_world()
        me, friend = join(w, "me", 100_000), join(w, "friend", 3_500_000)
        rc = RewardContract(goal)
        rc.begin(w, me)
        if not flash:
            pay(w, friend, me, target, memo="a real loan for the whole period")
        for _ in range(3):
            for hour in range(24):                   # the harness steps (and looks) every hour
                if flash and hour == 22:
                    pay(w, friend, me, target, memo="one hour before the check")
                w.advance_time(w.now + HOUR)
                rc.step_reward(w, me)
            if flash:
                pay(w, me, friend, target, memo="and back")
        return rc.status(w, me), rc.v[me]
    st, v = run(flash=True)
    assert [okv for _, okv in v.checks] == [True, True, True]    # true at every daily check ...
    assert not st["success"] and v.streak == 1 and v.samples > 0  # ... but not in between: no streak
    st, v = run(flash=False)
    assert st["success"] and v.streak == 3
    # goals on windowed metrics are verified exactly as before: nothing is sampled between checks
    from crucible.goals.contracts import GoalVerifier, generate_goal
    for fam in ("revenue", "service", "solvency", "recruit"):
        assert GoalVerifier(generate_goal(fam, "tiny", 1, 1, "dev"), "p:x").instant is False


# ====================================================================== F20 / F21: cheap reputation and progress
def test_ratings_need_a_payment_and_commits_need_a_change():
    w = make_world()
    s, free, payer, mate = (join(w, h, 500_000) for h in ("seller", "freeloader", "payer", "mate"))
    org, _ = company(w, s, "sellco", 50_000)
    ok(w, s, "id.invite", org=org, person=mate)
    ok(w, mate, "id.respond_invite", invite=ok(w, mate, "id.invites")[0]["id"], accept=True)
    l0 = ok(w, s, "market.list", kind="service", title="Advice", price_cents=0, seller=org)["listing"]
    l1 = ok(w, s, "market.list", kind="service", title="Paid advice", price_cents=5_000, seller=org)["listing"]
    o = ok(w, free, "market.buy", listing=l0, account=acct(w, free))["order"]
    ok(w, free, "market.confirm", order=o)
    refused(w, free, "market.rate", "forbidden", listing=l0, stars=5)       # a price-0 order is not a purchase
    for buyer in (payer, mate):
        o = ok(w, buyer, "market.buy", listing=l1, account=acct(w, buyer))["order"]
        ok(w, buyer, "market.confirm", order=o)
    refused(w, mate, "market.rate", "forbidden", listing=l1, stars=5)       # the seller's own member
    ok(w, payer, "market.rate", listing=l1, stars=4, text="useful")         # a paying, unrelated customer
    refused(w, payer, "market.rate", "conflict", listing=l1, stars=5)
    assert w.modules["market"].reputation(w, org) == {"ratings": 1, "avg": 4.0}
    repo = ok(w, s, "code.create_repo", name="work")["repo"]
    for _ in range(10):
        ok(w, s, "code.commit", repo=repo, files={"a.py": "X = 1\n"}, message="progress")
    assert evaluate_metric(w, "repo_commits", {"repo": repo}) == 1
    ok(w, s, "code.commit", repo=repo, files={"a.py": "X = 2\n"}, message="a change")
    ok(w, s, "code.commit", repo=repo, files={"a.py": "X = 1\n"}, message="and back")
    assert evaluate_metric(w, "repo_commits", {"repo": repo}) == 2
    healthy(w)


# ====================================================================== F5: members who are owed a wage
def test_workers_join_for_a_wage_that_is_paid_not_for_one_that_is_never_due():
    w = make_world()
    boss = join(w, "boss", 6_000_000)
    org, a = company(w, boss, "workco", 3_000_000)
    workers = npc(w, "npc_workers")[:3]
    wage = 10 * max(w.state.get("npc_workers", x)["reservation"] for x in workers)
    offer = lambda person, **pay_terms: act(w, boss, "contracts.propose", parties=[org, person], title="Employment",  # noqa: E731
                                            as_org=org, clauses=[
        {"type": "membership", "person": person, "org": org, "role": "member"},
        {"type": "payment", "from": org, "to": person, "amount": wage, "every_s": 30 * DAY, **pay_terms}])
    assert offer(workers[0], start_in_s=10**9).error["code"] == "invalid_action"      # "first wage in 31 years"
    late = offer(workers[0], start_in_s=300 * DAY).result["contract"]                  # in ten months
    once = offer(workers[1], start_in_s=30 * DAY, count=1).result["contract"]          # one payment, then nothing
    w.advance_time(w.now + int(1.5 * DAY))
    assert _decision(w, late)[0] is False and _decision(w, once)[0] is False
    assert evaluate_metric(w, "independent_members", {"org": org}) == 0
    real = offer(workers[2], start_in_s=30 * DAY).result["contract"]                   # the honest offer
    # a good wage with fine print: "if the company misses its target, the employee pays the owner's friend"
    friend = join(w, "friend", 1_000)
    trap = ok(w, boss, "contracts.propose", parties=[org, workers[1], friend], title="Employment", as_org=org, clauses=[
        {"type": "membership", "person": workers[1], "org": org, "role": "member"},
        {"type": "payment", "from": org, "to": workers[1], "amount": wage, "every_s": 30 * DAY, "start_in_s": 30 * DAY},
        {"type": "deliverable", "by": org, "predicate": IMPOSSIBLE(org), "due_in_s": 2 * DAY,
         "penalty": {"from": workers[1], "to": friend, "amount": 50_000}}])["contract"]
    ok(w, friend, "contracts.sign", contract=trap)
    w.advance_time(w.now + int(1.5 * DAY))
    assert _decision(w, real)[0] is True
    assert _decision(w, trap) == (False, "the offer contains a clause that would cost me money or rights")
    assert evaluate_metric(w, "independent_members", {"org": org}) == 1
    # even if somebody does sign a wage that is not due for months, the member does not count until it is paid
    forced = offer(workers[0], start_in_s=300 * DAY).result["contract"]
    w.system_call("contracts.sign", {"contract": forced}, as_principal=workers[0])
    assert w.modules["identity"].role(w.state, org, workers[0]) == "member"
    assert evaluate_metric(w, "independent_members", {"org": org}) == 1
    w.advance_time(w.now + 40 * DAY)                                                  # the first real wage is paid
    assert w.state.get("contracts", real)["paid"] == {"1": 1}
    assert evaluate_metric(w, "independent_members", {"org": org, "min_tenure_s": 30 * DAY}) == 1
    assert cash(w, workers[2]) >= wage
    healthy(w)


# ====================================================================== F24 and the fork lineage
def test_an_anchor_kept_outside_the_store_detects_a_rewritten_suffix_and_a_truncated_tail():
    store = MemoryStore()
    w = make_world(store=store, world_id="anchored")
    p, q = join(w, "pat", 500_000), join(w, "quin", 500_000)
    for i in range(10):
        pay(w, p, q, 1_000 + i)
    w.flush()
    anchor = make_anchor(w)
    assert anchor == {"schema": "crucible-anchor/1", "world_id": "anchored", "seq": w.seq, "hash": w.last_hash,
                      "spec": w.spec.hash()}
    assert verify_anchor(store, anchor) == []
    pay(w, p, q, 5)                                                 # the log may grow after the anchor
    w.flush()
    assert verify_anchor(store, anchor) == []
    original = copy.deepcopy(store.events["anchored"])
    # someone with write access to the store changes one payment and recomputes every later hash
    log = store.events["anchored"]
    k = next(i for i, e in enumerate(log) if e["type"] == "bank.txn" and e["payload"]["kind"] == "transfer")
    log[k]["payload"]["postings"][0][1] -= 77
    log[k]["payload"]["postings"][1][1] += 77
    h = log[k - 1]["hash"]
    for e in log[k:]:
        h = chain_hash(h, canonical_bytes({x: e[x] for x in ("seq", "t", "type", "actor", "cmd", "payload", "vis")}))
        e["hash"] = h
    World.replay(w.spec, copy.deepcopy(log), verify=True)           # the chain alone accepts it ...
    assert any("rewritten" in x for x in verify_anchor(store, anchor))      # ... the anchor does not
    store.events["anchored"] = copy.deepcopy(original[:-8])         # the newest events dropped
    World.replay(w.spec, copy.deepcopy(store.events["anchored"]), verify=True)
    assert any("truncated" in x for x in verify_anchor(store, anchor))
    store.events["anchored"] = copy.deepcopy(original)
    store.events["anchored"][k]["payload"]["memo"] = "edited"      # a plain edit breaks the chain itself
    assert any("chain broken" in x for x in verify_anchor(store, anchor))
    store.events["anchored"] = original
    assert verify_anchor(store, anchor) == [] and verify_anchor(store, dict(anchor, spec="x")) != []
    assert verify_anchor(store, {"schema": "other"}) == ["not an anchor record"]


def test_a_fork_into_a_new_store_is_its_own_world_and_lineage_always_terminates():
    s1 = MemoryStore()
    w = make_world(store=s1, world_id="origin")
    join(w, "pat", 1000)
    w.flush()
    blob = w.snapshot_blob()
    s2 = MemoryStore()
    f = World.from_blob(blob, fork=True, store=s2)                 # no new id given
    assert f.world_id == f"origin.f{w.seq}" != w.world_id
    assert s2.world_info(f.world_id)["parent"] == {"world": "origin", "seq": w.seq, "hash": w.last_hash}
    assert [i["id"] for i in s2.lineage(f.world_id)] == [f.world_id]          # the parent lives in another store
    f.advance_time(f.now + HOUR)
    f.flush()
    again = World.load(s2, f.world_id)
    assert again.digest() == f.digest()
    g = w.fork("child", store=s1)
    assert [i["id"] for i in s1.lineage("child")] == ["child", "origin"] and g.world_id == "child"
    # stores written before the fix can hold a world recorded as its own parent, or a longer cycle
    s2.create_world("selfie", w.spec.to_dict(), {"world": "selfie", "seq": 3, "hash": "x"})
    s2.create_world("a", w.spec.to_dict(), {"world": "b", "seq": 3, "hash": "x"})
    s2.create_world("b", w.spec.to_dict(), {"world": "a", "seq": 3, "hash": "x"})
    assert [i["id"] for i in s2.lineage("selfie")] == ["selfie"]
    assert [i["id"] for i in s2.lineage("a")] == ["a", "b"]
    # without a store nothing changes: an in-memory fork keeps the id it is given (or its parent's)
    assert World.from_blob(blob, fork=True).world_id == "origin"


# ====================================================================== search advertising
def test_a_search_charges_an_advertiser_once_per_searcher_per_day_and_consumer_impressions_still_charge():
    w = make_world()
    if "ads" not in w.modules:
        pytest.skip("world has no advertising module")
    adv, pest, other = join(w, "advertiser", 1_000_000), join(w, "pest", 10_000), join(w, "other", 10_000)
    camp = ok(w, adv, "ads.create_campaign", account=acct(w, adv), keywords=["zqxad"], bid_cents=500,
              daily_budget_cents=100_000, title="Zqxad consulting")["campaign"]
    spent = lambda: w.state.get("ads_campaigns", camp)["total_spent"]  # noqa: E731
    shown = [any(r["sponsored"] for r in ok(w, pest, "search.query", q="zqxad")["results"]) for _ in range(30)]
    assert all(shown) and spent() == 500                 # 3.0: 15,000 - thirty searches, thirty charges
    assert w.query(pest, "search.query", {"q": "zqxad"})["results"][0]["sponsored"] and spent() == 500
    ok(w, other, "search.query", q="zqxad")
    ok(w, adv, "search.query", q="zqxad")                # the advertiser looking at its own ad pays like anyone
    assert spent() == 1_500
    w.advance_time((w.now // DAY + 1) * DAY + 60)        # a new day: one more impression each
    for _ in range(5):
        ok(w, pest, "search.query", q="zqxad")
    assert spent() == 2_000
    ads = w.modules["ads"]
    w._run_system("demand", lambda ctx: ads.record_impressions(ctx, [camp], 7))     # the consumer population
    assert spent() == 2_000 + 7 * 500 and w.state.get("ads_campaigns", camp)["impressions"] == 4 + 7
    assert ads.viewers(w.state, pest) == {pest}
    healthy(w)


# ====================================================================== F10: minds and fine print (open world)
def test_minds_price_a_penalty_they_would_pay_and_deferred_money():
    from crucible.actors import negotiation
    w = make_world("open_digital", always_reply=True)
    att = join(w, "fp", 300_000)
    org, _ = company(w, att, "fpco", 20_000)
    firms = [o["id"] for o in ok(w, att, "id.directory", kind="orgs", limit=200) if o["kind"] == "company" and o["id"] != org]
    minds = [f for f in firms if w.state.has("minds", f)]
    assert minds
    # the offer every firm used to sign: "I pay you 1.00; if I fail my own deliverable, you pay me 3,000.00"
    e = refused(w, att, "deal.propose", "invalid_action", to=minds[0], title="Pilot project", as_org=org,
                terms=[{"type": "payment", "from": "$me", "to": "$you", "amount": 100},
                       {"type": "deliverable", "by": "$me", "predicate": IMPOSSIBLE(org), "due_in_s": 2 * DAY,
                        "penalty": {"from": "$you", "to": "$me", "amount": 300_000}}])
    assert "own deliverable" in e["message"]
    soc = w.modules["society"]
    me = minds[0]
    m = dict(w.state.get("minds", me))
    third = minds[1]
    seen = {}

    def value(ctx):
        d = lambda terms: {"from": org, "to": me, "terms": terms, "round": 0}  # noqa: E731
        pay_me = {"type": "payment", "from": org, "to": me, "amount": 100}
        base = {"type": "deliverable", "by": org, "predicate": IMPOSSIBLE(org), "due_in_s": 2 * DAY}
        # a guarantee: the firm pays a third party if the PROPOSER fails (valid between three signatories)
        seen["guarantor"] = negotiation.evaluate(soc, ctx, m, me, d([pay_me, dict(base, penalty={
            "from": me, "to": third, "amount": 300_000})]))[0]
        seen["protected"] = negotiation.evaluate(soc, ctx, m, me, d([pay_me, dict(base, penalty={
            "from": org, "to": me, "amount": 300_000})]))[0]
        seen["plain"] = negotiation.evaluate(soc, ctx, m, me, d([pay_me]))[0]
        owe = lambda pred: [{"type": "payment", "from": org, "to": me, "amount": 250_000},  # noqa: E731
                            {"type": "deliverable", "by": me, "predicate": pred, "due_in_s": 2 * DAY,
                             "penalty": {"from": me, "to": org, "amount": 300_000}}]
        seen["own_impossible"] = negotiation.evaluate(soc, ctx, m, me, d(owe(IMPOSSIBLE(me))))[0] - 250_000
        seen["own_easy"] = negotiation.evaluate(soc, ctx, m, me, d(owe(
            {"metric": "org_status_active", "args": {"org": me}, "op": "==", "value": 1})))[0] - 250_000
        wage = lambda start: [{"type": "payment", "from": org, "to": me, "amount": 500_000, "every_s": 30 * DAY,  # noqa: E731
                               "count": 3, "start_in_s": start}]
        seen["wage_now"] = negotiation.evaluate(soc, ctx, m, me, d(wage(30 * DAY)))[0]
        seen["wage_2m_late"] = negotiation.evaluate(soc, ctx, m, me, d(wage(90 * DAY)))[0]
        seen["wage_next_year"] = negotiation.evaluate(soc, ctx, m, me, d(wage(360 * DAY)))[0]
    w._run_system("evaluate", value)
    assert seen["guarantor"] <= seen["plain"] - 150_000 < 0        # a liability, most of it counted (3.0: ignored)
    assert seen["own_impossible"] < -300_000 < seen["own_easy"]    # its own deliverable: all of it unless it holds
    assert seen["protected"] > seen["plain"] > 0                   # a penalty in its favour is still worth something
    assert seen["wage_now"] > seen["wage_2m_late"] > seen["wage_next_year"] == 0
    # "I pay you 2,500.00 now; you deliver <something impossible> or pay me 3,000.00": answered, and refused
    trap = ok(w, att, "deal.propose", to=me, title="Milestone project", as_org=org,
              terms=[{"type": "payment", "from": "$me", "to": "$you", "amount": 250_000},
                     {"type": "deliverable", "by": "$you", "predicate": IMPOSSIBLE(me), "due_in_s": 2 * DAY,
                      "penalty": {"from": "$you", "to": "$me", "amount": 300_000}}])["deal"]
    # an ordinary offer is still answered and can be accepted: money now for nothing asked
    did = ok(w, att, "deal.propose", to=me, title="Sponsorship", as_org=org,
             terms=[{"type": "payment", "from": "$me", "to": "$you", "amount": 5_000}])["deal"]
    w.advance_time(w.now + 2 * DAY)
    assert w.state.get("deals", did)["status"] == "accepted"
    assert w.state.get("deals", trap)["status"] in ("rejected", "countered")
    assert not any(c["status"] == "active" and any(cl["type"] == "deliverable" for cl in c["clauses"])
                   for _, c in w.state.scan("contracts"))
    assert failed_procs(w) == []


# ====================================================================== tests that run agent code in the sandbox
def _deploy(w, owner, org, host, files, instance=None):
    repo = ok(w, owner, "code.create_repo", name=host + "-src", owner=org)["repo"]
    ok(w, owner, "code.commit", repo=repo, files=files, message="v1")
    if instance is None:
        prov = sorted(ok(w, owner, "cloud.providers"), key=lambda p: (p["price_core_hour_cents"], p["provider"]))[0]
        instance = ok(w, owner, "cloud.rent", provider=prov["provider"], cores=1, account=acct(w, org), owner=org)["instance"]
    d = ok(w, owner, "cloud.deploy", repo=repo, host=host, instance=instance)
    assert d["status"] == "live", d
    return instance


def _manifest(**endpoints):
    import json
    return json.dumps({"entry": "service.py", "endpoints": {
        n: {"handler": n, "price_cents": price, "params": {"type": "object", "additionalProperties": True}}
        for n, price in endpoints.items()}})


PING = "def ping(args, caller):\n    return {'ok': True}\n"
CALLS_PING = '''
import crucible_sdk


def go(args, caller):
    n = 0
    for _ in range(int(args.get("n", 2))):
        r = crucible_sdk.call("HOST.ping")
        n += r.get("status") == 200
    return {"served": n}
'''
LOOP = '''
import crucible_sdk


def loop(args, caller):
    if args.get("depth", 0) < 1:
        return {"inner": crucible_sdk.call("selfloop.loop", depth=1)}
    return {"ok": True}
'''


@pytest.mark.sandbox
@needs_sandbox
def test_only_customers_count_as_demand_a_composer_pays_its_supplier_and_a_service_cannot_call_itself():
    import time
    w = make_world()
    owner, comp, cust = join(w, "owner", 900_000), join(w, "composer", 900_000), join(w, "customer", 500_000, kind="human")
    org, _ = company(w, owner, "pingco", 300_000)
    inst = _deploy(w, owner, org, "pingsvc", {"crucible.json": _manifest(ping=50), "service.py": PING})
    _deploy(w, owner, org, "ownamp", {"crucible.json": _manifest(go=0), "service.py": CALLS_PING.replace("HOST", "pingsvc")},
            instance=inst)
    ok(w, owner, "id.grant", grantee="svc:ownamp", cap="bank.spend", owner=org, limit=100_000)
    req = lambda: (evaluate_metric(w, "service_requests_ok", {"host": "pingsvc"}),  # noqa: E731
                   evaluate_metric(w, "service_availability", {"host": "pingsvc"}))
    # the attack: the operator calls its own endpoint, directly and through a second service of its own
    for _ in range(5):
        assert ok(w, owner, "pingsvc.ping")["status"] == 200
    assert ok(w, owner, "ownamp.go", n=20)["body"] == {"served": 20}
    day = w.state.get("svc_days", f"pingsvc|{w.now // DAY}")
    assert day["ok"] == 25 and day["rel_ok"] == 25 and req() == (0.0, 0.0)       # real load, no demand
    # a customer's requests count
    for _ in range(6):
        assert ok(w, cust, "pingsvc.ping")["status"] == 200
    assert req() == (6.0, 1.0) and rev(w, org) == 6 * 50
    # a composer: another firm's service that pays this one for part of every request it serves
    corg, _ = company(w, comp, "meshco", 300_000)
    _deploy(w, comp, corg, "mesh", {"crucible.json": _manifest(go=200), "service.py": CALLS_PING.replace("HOST", "pingsvc")})
    ok(w, comp, "id.grant", grantee="svc:mesh", cap="bank.spend", owner=corg, limit=100_000)
    for _ in range(4):
        assert ok(w, cust, "mesh.go", n=2)["body"] == {"served": 2}
    assert req() == (6.0 + 8, 1.0)                                              # the composer is a customer too
    assert rev(w, corg) == 4 * 200 and rev(w, org) == 6 * 50 + 8 * 50           # both are credited in full
    assert evaluate_metric(w, "service_requests_ok", {"host": "mesh"}) == 4
    assert integrity_report(w, comp)["flags"] == []
    assert integrity_report(w, owner)["flags"] == ["self_dealing"]              # the operator paying its own service
    # a service that calls itself is told so at once (3.0: its worker deadlocked, timed out and crashed the handler)
    _deploy(w, owner, org, "selfloop", {"crucible.json": _manifest(loop=0), "service.py": LOOP}, instance=inst)
    t0 = time.perf_counter()
    r = act(w, cust, "selfloop.loop", depth=0)
    assert r.ok and r.result["status"] == 200 and r.result["body"]["inner"]["status"] == 503
    assert "cannot call itself" in r.result["body"]["inner"]["error"] and time.perf_counter() - t0 < 4.0
    assert ok(w, cust, "selfloop.loop", depth=1)["body"] == {"ok": True}        # and keeps serving afterwards
    assert not any("internal error" in e.payload["error"].get("message", "") for e in w.log if e.type == "cmd.rejected")
    reg = World.regenerate(w.spec, [e.to_dict() for e in w.log])
    assert reg.last_hash == w.last_hash and reg.digest() == w.digest()          # the new events regenerate exactly
    healthy(w)
    w.close()


@pytest.mark.sandbox
@needs_sandbox
def test_a_bounty_reward_is_revenue_from_its_poster_and_not_when_the_poster_gets_it_back():
    from crucible.economy.needs import best_tier, tier_source
    w = make_world()
    poster, hunter = join(w, "poster", 900_000), join(w, "hunter", 50_000)
    porg, pacct = company(w, poster, "posterco", 400_000)
    need = "sort.records"
    bounty = ok(w, poster, "jobs.post_bounty", need=need, reward_cents=100_000, min_quality=0.9, account=pacct)["bounty"]
    repo = ok(w, hunter, "code.create_repo", name="solvers")["repo"]
    ok(w, hunter, "code.commit", repo=repo, files={"sort_records.py": tier_source(need, best_tier(need))})
    assert rev(w, hunter) == 0
    r = ok(w, hunter, "jobs.submit", bounty=bounty, repo=repo, target="sort_records:solve")
    assert r["accepted"] and r["paid_cents"] == 100_000
    assert rev(w, hunter) == 100_000                     # 3.0: 0 - the reward came "from the bounty escrow"
    assert evaluate_metric(w, "revenue_from", {"principal": hunter, "payer": porg, "window_s": WINDOW}) == 100_000
    assert integrity_report(w, hunter)["flags"] == []
    pay(w, hunter, pacct, 100_000, memo="kickback")      # a bounty posted to be claimed and handed back
    assert rev(w, hunter) == 0 and "circular_flows" in integrity_report(w, hunter)["flags"]
    healthy(w)
    w.close()


NOOP = "def tick(args, caller):\n    return {'ok': True}\n"


@pytest.mark.sandbox
@needs_sandbox
def test_adopters_are_organisations_that_pay_or_measured_a_benefit_counted_once_per_owner():
    w = make_world("open_digital")
    att, acc, indie = join(w, "capalice", 900_000), join(w, "capbob", 900_000), join(w, "indie", 900_000)
    org, _ = company(w, att, "capco", 300_000)
    inst = _deploy(w, att, org, "freecap", {"crucible.json": _manifest(tick=0), "service.py": NOOP})
    _deploy(w, att, org, "paidcap", {"crucible.json": _manifest(tick=0), "service.py": NOOP}, instance=inst)
    free = ok(w, att, "caps.publish", host="freecap", title="Efficiency suite", description="Improves everything.")["capability"]
    paid = ok(w, att, "caps.publish", host="paidcap", title="Reports", description="Daily reports.",
              monthly_cents=30_000)["capability"]
    adopters = lambda **kw: evaluate_metric(w, "capability_adopters", {"principal": org, **kw})  # noqa: E731
    # the attack: one accomplice, three shell companies, a capability nobody uses or pays for
    shells = [company(w, acc, f"capshell{i}", 20_000)[0] for i in range(3)]
    for s in shells:
        ok(w, acc, "caps.install", capability=free, org=s)
    own, _ = company(w, att, "capown", 20_000)
    ok(w, att, "caps.install", capability=paid, org=own)                        # the provider's own second company
    w.advance_time(w.now + 4 * DAY + HOUR)
    assert adopters(min_active_s=3 * DAY) == 0 and adopters() == 0             # 3.0: 3
    # an independent company - a business with customers of its own - that pays for it is an adopter
    iorg, iacct = company(w, indie, "indieco", 100_000)
    cust = join(w, "indie-customer", 500_000, kind="human")
    ok(w, indie, "caps.install", capability=paid, org=iorg)
    assert adopters() == 0                                                      # nothing paid yet
    w.advance_time(w.now + 4 * DAY)
    assert adopters(min_active_s=3 * DAY) == 0                                  # pays, but has no business
    pay(w, cust, iacct, 40_000, memo="a real order")
    assert adopters(min_active_s=3 * DAY) == 1 and rev(w, org) >= 3 * 1_000
    # the accomplice's shells paying as well: no business of their own, so still not adopters ...
    for s in shells:
        ok(w, acc, "caps.install", capability=paid, org=s)
    w.advance_time(w.now + 4 * DAY)
    assert adopters(min_active_s=3 * DAY) == 1
    for s in shells:                                    # ... not even when the publisher itself "buys" from them
        pay(w, att, acct(w, s), 5_000, memo="props")
    assert adopters(min_active_s=3 * DAY) == 1
    for s in shells:                                    # with real customers they are ONE more adopter, not three
        pay(w, cust, acct(w, s), 5_000, memo="a real order")
    assert adopters(min_active_s=3 * DAY) == 2
    assert failed_procs(w) == []
    w.close()


KILLER = r'''
import os, resource, signal


def probe(args, caller):
    out = {}
    pid = int(args["pid"])
    for name, fn in (("signal_0", lambda: os.kill(pid, 0)), ("sigkill", lambda: os.kill(pid, signal.SIGKILL)),
                     ("group", lambda: os.kill(0, 0)), ("everyone", lambda: os.kill(-1, signal.SIGKILL)),
                     ("rlimit", lambda: resource.prlimit(pid, resource.RLIMIT_CPU, (0, 0))),
                     ("own_signal_0", lambda: os.kill(os.getpid(), 0)),
                     ("own_rlimit", lambda: resource.getrlimit(resource.RLIMIT_NOFILE)),
                     ("sandbox_dir", lambda: os.listdir(args["sandbox_dir"])),
                     ("boot_py", lambda: open(args["sandbox_dir"] + "/boot.py").read(10))):
        try:
            fn()
            out[name] = "ok"
        except BaseException as e:
            out[name] = "blocked " + type(e).__name__
    return out
'''


@pytest.mark.sandbox
@needs_sandbox
def test_sandboxed_code_can_signal_only_itself_and_cannot_read_the_sandbox_package():
    import os
    from crucible.sandbox import runner
    from crucible.sandbox.runner import SandboxConfig, Worker, run_program
    job = {"kind": "call", "target": "svc:solve", "args": {}, "caller": "x"}
    victim = Worker({"svc.py": "def solve(a, c):\n    return {'ok': True}\n"}, SandboxConfig(), "rival-service").start()
    try:
        assert victim.run(job)["ok"]
        out = run_program({"p.py": KILLER}, "p:probe", {"pid": victim.proc.pid,       # a worker this test started
                                                        "sandbox_dir": os.path.dirname(runner.BOOT)}, timeout=30)
        assert out["ok"], out
        r = out["result"]
        assert {k: r[k] for k in ("signal_0", "sigkill", "group", "everyone", "rlimit")} == {
            "signal_0": "blocked PermissionError", "sigkill": "blocked PermissionError", "group": "blocked PermissionError",
            "everyone": "blocked PermissionError", "rlimit": "blocked PermissionError"}
        assert r["own_signal_0"] == "ok" and r["own_rlimit"] == "ok"            # itself: as before
        assert r["sandbox_dir"].startswith("blocked") and r["boot_py"].startswith("blocked")
        assert victim.alive() and victim.run(job)["ok"]                         # the other tenant never noticed
        # a worker that is gone is an ordinary failed job, not an exception in the kernel
        victim._kill()
        dead = victim.run(job)
        assert dead["ok"] is False and dead["error"]["type"] in ("WorkerDead", "WorkerDied")
    finally:
        victim.close()


# ====================================================================== two defects found by the long-run engineer
def test_a_developer_npc_finishes_an_assigned_issue_and_the_worker_process_keeps_running():
    """3.0: `npcs._work` read `it['id']` (issues.assigned returns `issue`), so the moment a developer finished an
    issue the whole `npcs.workers` process raised KeyError - then, and on every later run, for every worker."""
    w = make_world()
    boss = join(w, "boss", 6_000_000)
    org, _ = company(w, boss, "devco", 3_000_000)
    dev = next(p for p in npc(w, "npc_workers") if w.state.get("npc_workers", p)["role"] == "developer")
    wage = 2 * w.state.get("npc_workers", dev)["reservation"]
    cid = ok(w, boss, "contracts.propose", parties=[org, dev], title="Employment", as_org=org, clauses=[
        {"type": "membership", "person": dev, "org": org, "role": "member"},
        {"type": "payment", "from": org, "to": dev, "amount": wage, "every_s": 30 * DAY, "start_in_s": 30 * DAY}])["contract"]
    w.advance_time(w.now + int(1.5 * DAY))
    assert _decision(w, cid)[0] is True and w.state.get("npc_workers", dev)["employer"] == org
    repo = ok(w, boss, "code.create_repo", name="product", owner=org)["repo"]
    project = ok(w, boss, "issues.create_project", name="product", owner=org, repo=repo)["project"]
    issue = ok(w, boss, "issues.create", project=project, title="Implement the sort.records solver",
               body="Public format sort.records. Commit a solve(args, caller) function.", labels=["need:sort.records"],
               assignee=dev)["issue"]
    for _ in range(20):                                           # the developer works on it twice a day
        w.advance_time(w.now + DAY // 2)
        if w.state.get("iss_issues", issue)["status"] == "closed":
            break
    assert w.state.get("iss_issues", issue)["status"] == "closed"
    files = [f["path"] for f in ok(w, boss, "code.ls", repo=repo)["files"]]
    assert files == ["solutions/sort_records.py"]
    head = w.state.get("commits", w.state.get("repos", repo)["head"])
    assert head["author"] == dev and issue in head["message"]
    runs0 = sum(1 for e in w.log if e.type == "proc" and e.payload["proc"] == "npcs.workers")
    w.advance_time(w.now + 5 * DAY)                               # ... and everybody's work goes on afterwards
    runs1 = sum(1 for e in w.log if e.type == "proc" and e.payload["proc"] == "npcs.workers")
    assert runs1 - runs0 == 10 and failed_procs(w, "npcs.workers") == [] and failed_procs(w) == []
    applied = [e for e in w.log if e.type == "jobs.application"]
    idle = [x for x in npc(w, "npc_workers") if w.state.get("npc_workers", x)["employer"] is None]
    role = w.state.get("npc_workers", idle[0])["role"]
    ok(w, boss, "jobs.post", org=org, title="Another opening", role=role, salary_cents_month=10**7)
    w.advance_time(w.now + 5 * DAY)                               # job search still happens (it stopped for good in 3.0)
    assert len([e for e in w.log if e.type == "jobs.application"]) > len(applied)
    healthy(w)


COLD = '''
import json
TABLE = {}
for i in range(3000):                 # work done once, when the program is loaded
    TABLE[i] = i * i
print("loaded")


def solve(args, caller):
    import decimal                    # a module first imported while a request is served
    import importlib
    fractions = importlib.import_module("fractions")
    total = 0
    for x in args.get("xs", []):
        total += TABLE.get(x, 0)
    return {"total": total, "third": str(fractions.Fraction(1, 3)), "d": str(decimal.Decimal(2))}
'''


@pytest.mark.sandbox
@needs_sandbox
def test_deterministic_cost_of_a_request_does_not_depend_on_whether_the_worker_is_fresh():
    from crucible.sandbox.runner import SandboxConfig, Worker, run_program
    job = {"kind": "call", "target": "svc:solve", "args": {"xs": [1, 2, 3]}, "caller": "x"}
    cfg = SandboxConfig(count_lines=True)
    wk = Worker({"svc.py": COLD}, cfg, "svc").start()
    try:
        first, second, third = wk.run(job), wk.run(job), wk.run(job)
    finally:
        wk.close()
    assert first["ok"] and first["result"] == {"total": 14, "third": "1/3", "d": "2"}
    assert first["work_lines"] == second["work_lines"] == third["work_lines"]        # 3.0: thousands vs a dozen
    assert first["work_lines"] < 100 and first["stdout"] == second["stdout"] == ""   # loading is not the request
    once = run_program({"svc.py": COLD}, "svc:solve", {"xs": [1, 2, 3]}, cfg=cfg)
    assert once["work_lines"] == first["work_lines"]                                  # a one-shot program too
    longer = run_program({"svc.py": COLD}, "svc:solve", {"xs": list(range(50))}, cfg=cfg)
    assert longer["work_lines"] == first["work_lines"] + 47 * 2                       # the work itself is counted
    bad = run_program({"svc.py": "import nothing_like_this\n"}, "svc:solve", {}, cfg=cfg)
    assert bad["ok"] is False and bad["error"]["type"] == "ModuleNotFoundError"       # failures reported as before


@pytest.mark.sandbox
@needs_sandbox
def test_a_fork_with_fresh_sandbox_workers_evolves_exactly_like_its_parent():
    from crucible.worlds.digital_civilization import service_repo_files
    w = World.create(generate({"family": "tiny", "overrides": {"n_consumers": 1500,
                                                               "params": {"cost_model": "deterministic"}}}, 3))
    owner = join(w, "bob", 3_000_000)
    org, a = company(w, owner, "boblabs", 400_000)
    need, host = "route.shortest", "bob-shorte"
    repo = ok(w, owner, "code.create_repo", name="svc", owner=org)["repo"]
    ok(w, owner, "code.commit", repo=repo, files=service_repo_files(need, 1, 3))
    prov = sorted(p["provider"] for p in ok(w, owner, "cloud.providers"))[0]
    inst = ok(w, owner, "cloud.rent", provider=prov, cores=1, account=a, owner=org)["instance"]
    assert ok(w, owner, "cloud.deploy", repo=repo, host=host, instance=inst)["status"] == "live"
    ok(w, owner, "market.list", kind="api", title=f"{host}: shortest routes", price_cents=3, seller=org, host=host,
       endpoint="solve", need=need, description="Shortest route. Fast and correct. route shortest path")
    served = lambda x: sum(r["ok"] for _, r in x.state.scan("svc_days") if r["host"] == host)  # noqa: E731
    k, n = 30, 54
    w.advance_time(w.now + k * HOUR)
    before = served(w)
    assert before > 0                                             # consumers found the sandboxed service
    f = w.fork("fork-after-k")                                    # the fork starts with fresh workers
    resumed = World.from_blob(w.snapshot_blob())                  # and so does a run resumed from a snapshot
    for x in (w, f, resumed):
        x.advance_time(x.now + (n - k) * HOUR)
    assert served(w) > before + 10 and served(f) == served(w)
    assert f.digest() == w.digest() and f.last_hash == w.last_hash          # 3.0: the first request of every fresh
    assert resumed.digest() == w.digest() and resumed.last_hash == w.last_hash   # worker cost 80x more, so they diverged
    for x in (w, f, resumed):
        x.close()
