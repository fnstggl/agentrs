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


def test_education_changes_workers_skills_and_their_job_profiles():
    """Education is not inert for employment (INT.json): what individuals learn in the education subsystem raises the
    occupational skills the labour market uses (real work quality, requirement gaps, shortlists) - at entry and again
    in later job-search sessions after more study - and the same individuals without the education subsystem have
    lower skills in the skills education draws on."""
    from crucible.domains import firms as F
    sp = base.spec(LM.NAME, "wide_certified", 0)
    w = World.create(sp)
    w.advance_time(12 * 86400)
    upd = [e for e in w.log if e.type == "labor.worker_update" and "skills" in e.payload["changes"]]
    assert len(upd) >= 5                                    # skills rose after study, in later search sessions
    entry = {e.payload["person"]: e.payload["skills"] for e in w.log if e.type == "labor.worker"}
    rose = [e for e in upd if any(e.payload["changes"]["skills"][k] > entry[e.payload["person"]][k]
                                  for k in F.EDU)]
    assert len(rose) == len(upd)                            # only skills education draws on rise
    assert sum((e.payload.get("extra") or {}).get("education_updates", 0) for e in w.log
               if e.type == "cols.batch" and e.payload["kind"] == "firms.jobsearch") == \
        len([e for e in w.log if e.type == "labor.worker_update" and {"skills", "declared"} & set(e.payload["changes"])])
    # provenance: each education-caused update names the recorded education events it rests on (the individual's own
    # latest study session, the certificates it newly declares), so Tier 3 causal chains cross education -> employment
    from crucible.civ import chains
    byseq = {e.seq: e for e in w.log}
    rp = [e for e in w.log if e.type == "firms.reprofile"]
    assert len(rp) >= 5
    for e in rp:
        i = int(e.payload["person"][1:])
        for r in e.payload["refs"]:
            c = byseq[r]
            assert r < e.seq and c.type in ("learn.sessions", "learn.certified")
            assert (any(s["i"] == i for s in c.payload["sessions"]) if c.type == "learn.sessions"
                    else c.payload["person"] == e.payload["person"])
    pv = chains.build(w, 0)
    cross = sum(1 for b, ps in pv.parents.items() for a in ps
                if pv.subsystem.get(a) == "education" and pv.subsystem.get(b) == "employment")
    assert cross >= len(rp)
    w0 = World.create(base.spec(LM.NAME, "wide_certified", 0, overrides={"drop_modules": ["learn"]}))
    w0.advance_time(12 * 86400)
    both = sorted(set(w.state.ids("labor_workers")) & set(w0.state.ids("labor_workers")))
    both = [p for p in both if p.startswith("p") and not p.startswith("p:")]
    assert len(both) > 100
    edu = lambda x: sum(x.state.tables["labor_workers"][p]["skills"][k] for p in both for k in F.EDU)  # noqa: E731
    non = lambda x: sum(x.state.tables["labor_workers"][p]["skills"][k] for p in both  # noqa: E731
                        for k in F.SKILLS if k not in F.EDU)
    assert edu(w) > edu(w0) + 0.5                            # education raised the skills it draws on
    assert abs(non(w) - non(w0)) < 1e-6                      # and nothing else
    assert not [e for e in w.log if e.type == "proc.failed"] and w.check_invariants() == []
    w.close()
    w0.close()


def test_job_search_leaves_workers_of_another_labour_market_path_alone():
    """One decision path per worker: an individual whose labour-market record another path entered (not a simulated
    worker of `firms`' job-search model; e.g. economy.jobs registers individuals itself and answers their offers) is
    left to that path - `job_search` sessions neither apply for it nor rewrite its profile."""
    import numpy as np
    from crucible.commerce.labor import SKILLS as LSK
    from crucible.substrate import ids
    w = _world()
    st = w.state
    free = [i for i in range(400) if not st.has("labor_workers", ids.pid(i))][:150]
    assert len(free) >= 100

    def register(ctx):
        for i in free:
            p = ids.ensure_principal(ctx, i, account=True)
            # qualified for every posting and cheap: the job-search model would apply for them (139 applications in
            # this run without the one-path rule)
            w.modules["labor"].add_worker(ctx, p, {k: 0.8 for k in LSK}, {k: 1.0 / len(LSK) for k in LSK}, 100_000,
                                          sim=False, hours_week=20, declared=sorted(LSK)[:3])
    w._run_system("test.other_path", register)
    w.advance_time(6 * 86400)
    other = {ids.pid(i) for i in free}
    skipped = sum((e.payload.get("extra") or {}).get("other_path", 0) for e in w.log
                  if e.type == "cols.batch" and e.payload["kind"] == "firms.jobsearch")
    assert skipped > 0                                       # their job-search sessions happened and were left alone
    apps = [r for _, r in st.scan("labor_apps")]
    assert apps and not [r for r in apps if r["worker"] in other]
    assert not [e for e in w.log if e.type == "labor.worker_update" and e.payload["person"] in other]
    assert all(not st.tables["labor_workers"][p]["sim"] for p in other)
    assert not [e for e in w.log if e.type == "proc.failed"] and w.check_invariants() == []
    w.close()
