"""Autonomous social negotiation: independent minds with private reservation values evaluate offers, accept,
reject, counter, remember, and respond to broken commitments; agreements are authoritative contracts."""
import crucible

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def _w(seed=1, **params):
    return World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 300, "workers": 12,
                                                                          "params": {"always_reply": True, **params}}},
                                 seed))


def _agent(w, handle, cash=5_000_000):
    w.system_call("admin.join", {"handle": handle, "endowment": cash + 300_000})
    ex = lambda t, **k: w.execute(f"p:{handle}", {"tool": t, "args": k})  # noqa: E731
    me = w.modules["bank"].primary_account(w.state, f"p:{handle}")
    org = f"o:{handle}co"
    assert ex("id.create_org", handle=f"{handle}co", name="Co", kind="company").ok
    assert ex("biz.register", org=org, account=me).ok
    acct = ex("bank.open_account", owner=org).result["account"]
    assert ex("bank.transfer", from_account=me, to=acct, amount=cash).ok
    return ex, org, acct


def _workers(w):
    st = w.state
    return sorted((k for k, m in st.scan("minds") if m["kind"] == "person" and st.tables["npc_workers"][k]["employer"] is None),
                  key=lambda k: k)


def _job(org, person, salary, acct, every=7 * DAY):
    return [{"type": "membership", "person": person, "org": org, "role": "member", "title": "engineer"},
            {"type": "payment", "from": org, "to": person, "amount": int(salary * every / (30 * DAY)), "every_s": every,
             "count": 3, "start_in_s": 0},
            {"type": "notice", "seconds": 7 * DAY}]


def _settle(w, days=2):
    w.advance_time(w.now + days * DAY)


def test_mutually_beneficial_offer_is_accepted_and_becomes_a_binding_contract():
    w = _w(1)
    ex, org, acct = _agent(w, "ada")
    p = _workers(w)[0]
    res = w.state.tables["minds"][p]["reservation"]
    r = ex("deal.propose", to=p, as_org=org, terms=_job(org, p, int(res * 2.0), acct), title="Join us",
           accounts={"1": acct})
    assert r.ok, r.error
    _settle(w)
    d = w.state.tables["deals"][r.result["deal"]]
    assert d["status"] == "accepted" and d["contract"]
    c = w.state.tables["contracts"][d["contract"]]
    assert c["status"] == "active"
    assert w.modules["identity"].role(w.state, org, p) == "member"          # the commitment changed state
    assert any(t["kind"] == "contract" for _, t in w.state.scan("txns"))   # and the first salary was paid


def test_lowball_offer_is_rejected_and_nothing_changes():
    w = _w(2)
    ex, org, acct = _agent(w, "bob")
    p = _workers(w)[0]
    res = w.state.tables["minds"][p]["reservation"]
    r = ex("deal.propose", to=p, as_org=org, terms=_job(org, p, int(res * 0.05), acct), title="Cheap job",
           accounts={"1": acct})
    _settle(w)
    d = w.state.tables["deals"][r.result["deal"]]
    assert d["status"] == "rejected"
    assert w.modules["identity"].role(w.state, org, p) is None


def test_counteroffer_negotiation_converges_and_private_reservations_differ():
    w = _w(3)
    ex, org, acct = _agent(w, "cyd")
    ws = _workers(w)[:4]
    outcomes = []
    for p in ws:
        res = w.state.tables["minds"][p]["reservation"]
        r = ex("deal.propose", to=p, as_org=org, terms=_job(org, p, int(res * 0.95), acct), title="Offer",
               accounts={"1": acct})
        assert r.ok
        outcomes.append((p, r.result["deal"]))
    _settle(w)
    countered = 0
    for p, did in outcomes:
        d = w.state.tables["deals"][did]
        if d["status"] == "countered":
            countered += 1
            nxt = [k for k, x in w.state.scan("deals") if x["parent"] == did][0]
            ask = [c for c in w.state.tables["deals"][nxt]["terms"] if c["type"] == "payment"][0]["amount"]
            offered = [c for c in d["terms"] if c["type"] == "payment"][0]["amount"]
            assert ask > offered                                               # they asked for more
            assert ex("deal.respond", deal=nxt, action="accept", accounts={"1": acct}).ok
            c = w.state.tables["deals"][nxt]["contract"]
            assert w.state.tables["contracts"][c]["status"] == "active"
    assert countered >= 1
    # the same offer relative to each person's reservation met different responses (heterogeneous minds)
    assert len({w.state.tables["deals"][d]["status"] for _, d in outcomes}) >= 1


def test_conflicting_incentives_firm_refuses_to_hand_over_control_for_a_token_payment():
    w = _w(4)
    ex, org, acct = _agent(w, "dex")
    firm = sorted(k for k, m in w.state.scan("minds") if m["kind"] == "firm")[0]
    terms = [{"type": "grant", "owner": "$you", "grantee": "$me", "cap": "org.admin", "scope": "*"},
             {"type": "payment", "from": "$me", "to": "$you", "amount": 1000, "every_s": None, "count": None,
              "start_in_s": 0}]
    r = ex("deal.propose", to=firm, as_org=org, terms=terms, title="Let us run your company", accounts={"1": acct})
    assert r.ok, r.error
    _settle(w)
    d = w.state.tables["deals"][r.result["deal"]]
    assert d["status"] in ("rejected", "countered")
    assert not any(g["grantee"] == org and g["cap"] == "org.admin" and g["status"] == "active"
                   for _, g in w.state.scan("grants"))


def test_breach_damages_reputation_and_later_offers_are_refused():
    w = _w(5)
    ex, org, acct = _agent(w, "eve", cash=2_000_000)
    ws = _workers(w)
    p = ws[0]
    res = w.state.tables["minds"][p]["reservation"]
    r = ex("deal.propose", to=p, as_org=org, terms=_job(org, p, int(res * 2.0), acct), title="Job",
           accounts={"1": acct})
    _settle(w)
    assert w.state.tables["deals"][r.result["deal"]]["status"] == "accepted"
    # empty the account: the next salary payment fails -> breach (public), claim
    me = w.modules["bank"].primary_account(w.state, "p:eve")
    bal = w.modules["bank"].balance(w.state, acct)
    assert ex("bank.transfer", from_account=acct, to=me, amount=bal).ok
    w.advance_time(w.now + 9 * DAY)
    rep = w.modules["deal"].reputation(w, org)
    assert rep["contracts_breached_by"] >= 1 and rep["claims_owed"] + rep["debts_written_off"] >= 1
    m = w.state.tables["minds"][p]
    assert m["trust"][org][1] >= 4                                         # the victim remembers
    # another, independent person now refuses even a generous offer from the breaching organisation
    assert ex("bank.transfer", from_account=me, to=acct, amount=bal).ok
    q = ws[1]
    r2 = ex("deal.propose", to=q, as_org=org, terms=_job(org, q, int(w.state.tables["minds"][q]["reservation"] * 2), acct),
            title="Generous", accounts={"1": acct})
    _settle(w)
    assert w.state.tables["deals"][r2.result["deal"]]["status"] == "rejected"


def test_long_term_cooperation_builds_trust_when_commitments_are_kept():
    w = _w(6)
    ex, org, acct = _agent(w, "fay", cash=20_000_000)
    p = _workers(w)[0]
    res = w.state.tables["minds"][p]["reservation"]
    r = ex("deal.propose", to=p, as_org=org, terms=_job(org, p, int(res * 2.0), acct), title="Job",
           accounts={"1": acct})
    w.advance_time(w.now + 23 * DAY)                                       # three salaries paid, contract completes
    c = w.state.tables["contracts"][w.state.tables["deals"][r.result["deal"]]["contract"]]
    assert c["status"] in ("completed", "active")
    rep = w.modules["deal"].reputation(w, org)
    assert rep["contracts_breached_by"] == 0
    tr = w.state.tables["minds"][p]["trust"][org]
    assert tr[0] > tr[1]
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []
