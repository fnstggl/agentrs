"""ECON wave 3b: mutual-aid associations of the population (economy.assoc) under attack by an agent member.

Regression tests of the verifier's probes p_assoc_takeover and p_assoc_drain (wave-3b repair): members voted yes on
any rule change with probability 0.6 without reading it (an agent made itself treasurer and moved the whole
treasury to itself), and approved any claim of a member who paid dues twice (an agent that never had a job paid
dues twice on one day and drained 48 times what it paid). Members now read what a decision does: only quorum or
window changes of the members' procedure can pass, and claims must qualify under the published terms (membership
age, dues in two months, one benefit per claim gap) and are judged on the evidence of the job loss voters have."""
import pytest

import crucible

crucible.load_all_modules()
from crucible.commerce.worlds import startup_spec  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy import assoc  # noqa: E402

DAY = 86400


@pytest.fixture(scope="module")
def world():
    w = World.create(startup_spec(21, n=450, formats=["schedule.jobs"], extra=("xchg", "inst", "labor")))
    d = 0
    while d < 40:
        d += 1
        w.advance_time(d * DAY)
        if assoc.associations(w.state) and d >= 12:
            break
    assert assoc.associations(w.state), "the population founded an association"
    return w


def _agent(w, handle, endowment):
    pid = w.system_call("admin.join", {"handle": handle, "endowment": endowment})["principal"]
    return pid, (lambda tool, **a: w.execute(pid, {"tool": tool, "args": a}))


def _advance(w, days):
    d0 = w.now // DAY
    for k in range(1, days + 1):
        w.advance_time((d0 + k) * DAY)


def _balance(w, pid):
    return sum(a["balance"] for a in w.state.find("accounts", "owner", pid))


def test_members_read_amendments_and_refuse_a_treasury_takeover(world):
    w = world
    st = w.state
    org, terms = max(assoc.associations(st), key=lambda x: len(st.find_ids("members", "org", x[0])))
    bank = w.modules["bank"]
    oacct = bank.primary_account(st, org)
    usurper, ex = _agent(w, "usurper", 50_000)
    assert ex("inst.join", org=org, note="hello").ok
    cur = st.tables["inst_charters"][org]["charter"]
    grab = assoc.CHARTER(cur["procedures"]["members"]["quorum"], cur["procedures"]["members"]["window_s"] // DAY)
    grab["membership"] = dict(cur["membership"])
    grab["offices"] = {"treasurer": {"holder": usurper, "budget_cents_month": 10 ** 8, "tools": ["bank.*"]}}
    grab["procedures"]["treasurer"] = {"type": "officer", "office": "treasurer"}
    grab["rules"] = [{"match": {"kind": "tool", "tool": "bank.*"}, "procedure": "treasurer"},
                     {"match": {"kind": "amend"}, "procedure": "members"}]
    r = ex("inst.propose", org=org, action={"kind": "amend", "charter": grab}, reason="Streamline payouts.")
    assert r.ok, r.error
    did = r.result["decision"]
    # a quorum change only is operational: the population's own adaptation path classifies as such
    low = assoc.CHARTER(round(max(0.1, cur["procedures"]["members"]["quorum"] - 0.1), 2),
                        cur["procedures"]["members"]["window_s"] // DAY)
    low["membership"] = dict(cur["membership"])
    r2 = ex("inst.propose", org=org, action={"kind": "amend", "charter": low}, reason="Lower the quorum.")
    assert r2.ok, r2.error
    norm = st.tables["inst_decisions"][r2.result["decision"]]["action"]["charter"]
    assert assoc._changed(cur, norm) == {("procedures", "members", "quorum")}
    assert not assoc._changed(cur, st.tables["inst_decisions"][did]["action"]["charter"]) <= assoc.OPERATIONAL
    _advance(w, 4)
    dec = st.tables["inst_decisions"][did]
    ballots = [b["choice"] for b in st.find("inst_ballots", "decision", did)]
    assert dec["status"] in ("rejected", "lapsed"), (dec["status"], ballots.count("yes"), ballots.count("no"))
    assert ballots.count("no") > 5 * ballots.count("yes"), (ballots.count("yes"), ballots.count("no"))
    assert not st.find("inst_offices", "org", org) or all(o["holder"] != usurper
                                                          for o in st.find("inst_offices", "org", org))
    before = _balance(w, usurper)
    amount = bank.balance(st, oacct)
    t = ex("inst.propose", org=org, action={"kind": "tool", "tool": "bank.transfer",
                                            "args": {"from_account": oacct, "to": usurper, "amount": int(amount),
                                                     "memo": "treasury"}}, reason="payout")
    assert t.ok and t.result.get("status") == "open"        # a members' majority decides; nobody executes it alone
    _advance(w, 4)
    assert st.tables["inst_decisions"][t.result["decision"]]["status"] in ("rejected", "lapsed")
    assert _balance(w, usurper) <= before
    assert not w.check_invariants()


def test_claims_must_qualify_and_a_patient_fraud_is_bounded_to_one_benefit_per_gap(world):
    w = world
    st = w.state
    org, terms = max(assoc.associations(st), key=lambda x: len(st.find_ids("members", "org", x[0])))
    bank = w.modules["bank"]
    oacct = bank.primary_account(st, org)
    fr, ex = _agent(w, "freerider", 200_000)
    assert ex("inst.join", org=org, note="insure me").ok
    dues = int(terms["dues_cents_month"])
    assert ex("inst.pay_dues", org=org, amount=dues).ok and ex("inst.pay_dues", org=org, amount=dues).ok
    b0 = _balance(w, fr)

    def claim():
        r = ex("inst.propose", org=org, action={"kind": "tool", "tool": "bank.transfer", "args": {
            "from_account": oacct, "to": fr, "amount": int(terms["benefit_cents"]), "memo": "mutual aid: job loss"}},
            reason="I lost my job; I claim the association's benefit.")
        assert r.ok, r.error
        return r.result["decision"]
    # the probe's drain: eight claims four days apart in the first month of membership - none qualifies
    early = []
    for _ in range(6):
        early.append(claim())
        _advance(w, 4)
    assert all(st.tables["inst_decisions"][d]["status"] in ("rejected", "lapsed") for d in early), \
        [st.tables["inst_decisions"][d]["status"] for d in early]
    assert _balance(w, fr) == b0
    # qualifying (30 days a member, dues in two months): a claim nobody can verify gets the benefit of the doubt ...
    assert ex("inst.pay_dues", org=org, amount=dues).ok
    _advance(w, 8)
    f = assoc.claim_facts(_Ctx(w), org, fr, terms, as_principal=fr)
    assert f["qualifies"] and f["labour_market"] is None, f
    first = claim()
    _advance(w, 4)
    got_first = st.tables["inst_decisions"][first]["status"] == "executed"
    # ... and a second benefit inside the claim gap never qualifies
    again = claim()
    _advance(w, 4)
    assert st.tables["inst_decisions"][again]["status"] in ("rejected", "lapsed")
    gain = _balance(w, fr) - b0 + dues                     # benefits received (one more dues payment since b0)
    assert gain <= int(terms["benefit_cents"]), gain       # at most one benefit (the probe drained 8)
    assert got_first == (gain == int(terms["benefit_cents"]))
    assert not w.check_invariants()


class _Ctx:
    """A read-only view for `claim_facts` outside a session: it calls read-only tools as the given principal."""

    def __init__(self, w):
        self.world, self.state, self.now = w, w.state, w.now

    def call(self, tool, args, as_principal):
        r = self.world.execute(as_principal, {"tool": tool, "args": args})
        assert r.ok, r.error
        return r.result
