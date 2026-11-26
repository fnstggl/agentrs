"""Firms and the population's labour market (domains.firms; SP-9.9, DEF-I-6 firms faucet)."""
import pytest

import crucible
from crucible.core.kernel import World
from crucible.domains import base
from crucible.domains import labor_marketplace as LM

crucible.load_all_modules()


def _world(seed=0, firms=None, days=0):
    sp = base.spec(LM.NAME, "pipeline_open", seed, overrides={"firms": firms} if firms else None)
    w = World.create(sp)
    if days:
        w.advance_time(days * 86400)
    return w


def test_unknown_skills_are_refused_loudly():
    bad = {"firms": [dict(LM._CLIENT), {"handle": "oddco", "goal": "profit", "skills": {"backend": 0.5},
                                         "staff": 0}]}
    with pytest.raises(Exception, match="unknown skills"):
        _world(firms=bad)


def test_population_individuals_enter_the_labour_market_and_are_hired():
    w = _world(days=6)
    st = w.state
    pop_workers = [p for p in st.ids("labor_workers") if p.startswith("p") and not p.startswith("p:")]
    assert len(pop_workers) > 20
    employed = [p for p in pop_workers if st.tables["labor_workers"][p]["employer"]]
    assert employed                                         # initial staff and hires come from the population
    assert any(e.type == "cols.batch" and e.payload["kind"] == "firms.jobsearch" for e in w.log)
    assert not [e for e in w.log if e.type == "proc.failed"]
    assert w.check_invariants() == []


def test_no_external_buyer_revenue_follows_client_budgets():
    """Firms earn only from clients' commissions paid from the clients' own bounded budgets: no money arrives from
    the exogenous economy for work, and a contractor's price above the client's value sells nothing."""
    w = _world(days=20)
    st = w.state
    firms = {o for o, f in st.scan("firms") if f["goal"] != "client"}
    accts = {st.tables["firms"][o]["account"] for o in firms}
    ext = 0
    for lid, e in st.scan("acct_txns"):
        tx = st.tables["txns"][e["txn"]]
        if e["acct"] in accts and any(a == "acct_external" for a, d in tx["postings"]) and \
                sum(d for a, d in tx["postings"] if a == e["acct"]) > 0:
            ext += 1
    assert ext == 0
    assert len(st.tables["firms_commissions"]) >= 2
    pricey = {"firms": [dict(LM._CLIENT), dict(LM.FIRM_SETS["open"][0], margin=60.0),
                        dict(LM.FIRM_SETS["open"][1], margin=60.0)], "commissions": [], "job_search_prior": 0.2}
    w2 = _world(firms=pricey, days=20)
    assert not any(c["status"] == "contracted" for _, c in w2.state.scan("firms_commissions"))


def test_seat_commission_creates_the_participants_project():
    w = _world()
    com = LM._seat_commission(w)
    assert com is not None and com["offer"] and len(com["tasks"]) >= 5
    assert w.state.get("orgs", com["to"]) is not None
