"""SP-6.1 - businesses and products (workstream F1).

One scenario per feature of MASTER_SPEC Part 6.1, each run on a real world through the ordinary tools and checked
on authoritative state: formation, development, subscriptions, one-time purchases, licensing, marketplaces, SaaS,
educational, media, developer-tool, data, infrastructure and industrial products, contracts, refunds,
cancellations, customer support, competition, bankruptcy - plus the rule that revenue only comes from independent
participants. A feature counts only if its scenario passes; the evidence lists every result.
"""
from __future__ import annotations

import traceback

import numpy as np

from _commerce_kit import (DAY, Actor, balance, failures, healthy, join, make_world, media_business, mint,
                           saas_business)
from crucible.acceptance.v5.evidence import record
from crucible.commerce import revenue
from crucible.economy.needs import NEEDS, tier_source


def _score(need, res, inp, hidden):
    if res.get("status") != 200:
        return 0.0
    return float(NEEDS[need].score(inp, hidden, res.get("body")))


def f_formation():
    w = make_world(101)
    f = join(w, "founder", 500_000)
    fees0 = balance(w, "acct_fees")
    out = f.do("venture.form", handle="newco", name="NewCo", sector="software", capital_cents=200_000)
    org, acct = out["org"], out["account"]
    c = f.do("biz.company", org=org)
    assert c["status"] == "registered" and c["shareholders"][0]["holder"] == f.id
    assert balance(w, acct) == 200_000 and balance(w, f.acct) == 500_000 - 200_000 - 50_000
    assert balance(w, "acct_fees") == fees0 + 50_000
    assert any(r["org"] == org for r in f.do("biz.registry"))
    assert not join(w, "latecomer", 0).attempt("venture.form", handle="newco", name="Again").ok


def f_product_development():
    w = make_world(102)
    f = join(w, "founder", 500_000)
    org = f.do("venture.form", handle="devco", name="DevCo", capital_cents=100_000)["org"]
    p = f.do("prod.create", name="Lib", kind="devtool", owner=org)["product"]
    assert not f.attempt("prod.update", product=p, status="launched").ok, "nothing to deliver yet"
    other = join(w, "other", 1_000)
    orepo = other.do("code.create_repo", name="theirs")["repo"]
    assert not f.attempt("prod.update", product=p, delivery={"type": "repo", "repo": orepo}).ok
    repo = f.do("code.create_repo", name="lib", owner=org)["repo"]
    f.do("prod.update", product=p, delivery={"type": "repo", "repo": repo})
    assert not f.attempt("prod.update", product=p, status="launched").ok, "an empty repository cannot ship"
    c1 = f.do("code.commit", repo=repo, files={"lib.py": "def f(x):\n    return x\n"}, message="v0")["commit"]
    oc = other.do("code.commit", repo=orepo, files={"a.py": "x = 1\n"}, message="x")["commit"]
    assert not f.attempt("prod.release", product=p, version="0.1", commit=oc).ok
    f.do("prod.release", product=p, version="0.1", commit=c1, notes="first")
    c2 = f.do("code.commit", repo=repo, files={"lib.py": "def f(x):\n    return 2 * x\n"}, message="v1")["commit"]
    f.do("prod.release", product=p, version="1.0", commit=c2)
    f.do("prod.update", product=p, status="launched")
    v = f.do("prod.view", product=p)
    assert v["status"] == "launched" and v["version"] == "1.0" and v["releases"] == 2
    assert [r["version"] for r in v["release_notes"]] == ["0.1", "1.0"]


def f_subscriptions():
    w = make_world(103)
    f, org, prod, plans, doc = media_business(w, "subco", monthly=1_500, annual=15_000)
    a, b = join(w, "ca", 40_000), join(w, "cb", 40_000)
    a.do("subs.subscribe", plan=plans["month"])
    b.do("subs.subscribe", plan=plans["year"])
    w.advance_time(61 * DAY)
    assert balance(w, a.acct) == 40_000 - 3 * 1_500 and balance(w, b.acct) == 40_000 - 15_000
    assert revenue.mrr_exact(w, org) == 1_500 + 1_250


def f_one_time_purchases():
    w = make_world(104)
    f, org, prod, plans, doc = media_business(w, "otco", monthly=None)
    off = f.do("prod.offer", product=prod, kind="one_time", price_cents=4_900)["offer"]
    c = join(w, "buyer1", 10_000)
    out = c.do("prod.buy", offer=off)
    assert out["expires"] is None and balance(w, c.acct) == 5_100
    w.advance_time(400 * DAY)
    assert c.attempt("docs.get", doc=doc).ok and c.do("prod.access", product=prod)["delivery"] == "content"
    rep = revenue.revenue_report(w, org, 500 * DAY)
    assert rep["one_time_sales_cents"] == 4_900 and rep["mrr_cents"] == 0.0, "a one-time sale is not MRR"


def f_licensing():
    w = make_world(105)
    f = join(w, "lic-f", 400_000)
    org = f.do("venture.form", handle="licco", name="LicCo", capital_cents=100_000)["org"]
    repo = f.do("code.create_repo", name="engine", owner=org)["repo"]
    f.do("code.commit", repo=repo, files={"engine.py": "def run(x, caller=None):\n    return {'y': x}\n"})
    p = f.do("prod.create", name="Engine", kind="devtool", owner=org, delivery={"type": "repo", "repo": repo})["product"]
    f.do("prod.update", product=p, status="launched")
    off = f.do("prod.offer", product=p, kind="license", price_cents=12_000, seats=5, term_days=30)["offer"]
    c = join(w, "lic-c", 50_000)
    out = c.do("prod.buy", offer=off)
    assert out["expires"] == w.now + 30 * DAY
    assert c.attempt("code.read", repo=repo, path="engine.py").ok
    assert not c.attempt("prod.buy", offer=off).ok, "one active licence at a time"
    ent = [e for e in c.do("prod.entitlements") if e["product"] == p][0]
    assert ent["seats"] == 5 and ent["source"] == "license"
    w.advance_time(31 * DAY)
    assert not c.attempt("code.read", repo=repo, path="engine.py").ok, "the licence expired"
    assert not c.attempt("prod.access", product=p).ok
    c.do("prod.buy", offer=off)
    assert c.attempt("code.read", repo=repo, path="engine.py").ok
    assert w.state.tables["txns"][[x for x in w.state.tables["prod_purchases"].values()][-1]["txn"]]["kind"] == \
        "license"


def f_marketplaces():
    cfg = {"patrons": {"n": 0, "prefix": "mk", "sellers": {"n": 8, "sku": "gear", "stock": [30, 60],
                                                           "unit_cost_cents": [400, 700], "act_every_days": 2},
                       "goods": {"n": 40, "sku": "gear", "units_per_month": [2, 4], "wtp_unit_cents": [1_500, 3_000],
                                 "buy_p": 0.6}, "act_every_days": 1}}
    w = make_world(106, configs=cfg, extra=["patrons"])
    op = join(w, "operator", 300_000)
    oorg = op.do("venture.form", handle="bazaarx", name="BazaarX", capital_cents=100_000)["org"]
    cheap = op.do("xchg.create", name="BazaarX", take_rate_bps=500, operator=oorg)["exchange"]
    rival = join(w, "rival-op", 100_000)
    dear = rival.do("xchg.create", name="DearMart", take_rate_bps=3_000)["exchange"]
    w.advance_time(40 * DAY)
    st = w.state
    sellers = {e: len({x["seller"] for x in st.find("xchg_listings", "exchange", e)}) for e in (cheap, dear)}
    orders = {e: st.find_ids("xchg_orders", "exchange", e) for e in (cheap, dear)}
    rows = [st.tables["xchg_orders"][o] for o in orders[cheap]]
    fees = revenue.revenue_report(w, oorg, 40 * DAY)["marketplace_fee_revenue_cents"]
    assert sellers[cheap] > sellers[dear], sellers
    assert len(rows) >= 20 and len({o["buyer"] for o in rows}) >= 10, len(rows)
    assert fees == sum(o["fee"] for o in rows) > 0
    # trust: a buyer disputes an order within the window and the operator decides (the seller refunds)
    oid = next(o for o in reversed(orders[cheap]) if st.tables["xchg_orders"][o]["dispute_until"] >= w.now)
    o = st.tables["xchg_orders"][oid]
    Actor(w, o["buyer"]).do("xchg.dispute", order=oid, reason="damaged")
    before = balance(w, o["account"])
    op.do("xchg.resolve", order=oid, refund_pct=100)
    assert balance(w, o["account"]) == before + o["amount"]
    held = sum(r["qty"] for r in st.tables["inventory"].values() if r["sku"] == "gear")
    assert held == st.meta_get("goods_minted:gear", 0), "goods are conserved"
    assert not failures(w) and not healthy(w)


def f_saas():
    w = make_world(107, providers=True)
    need = "text.extract"
    f, org, prod, plan, info = saas_business(w, "saasco", need, tier=len(NEEDS[need].tiers) - 1, monthly=2_900)
    c = join(w, "user", 100_000)
    rng = np.random.default_rng(7)
    inp, hidden = NEEDS[need].generate(rng, 0.4)
    assert not c.attempt("prod.use", product=prod, input=inp).ok, "subscribers only"
    c.do("subs.subscribe", plan=plan)
    qs = []
    for _ in range(4):
        inp, hidden = NEEDS[need].generate(rng, 0.4)
        qs.append(_score(need, c.do("prod.use", product=prod, input=inp), inp, hidden))
    assert min(qs) == 1.0, qs
    direct = c.attempt(f"{info['host']}.solve", **inp)
    assert direct.ok and direct.result["status"] == 402 and balance(w, c.acct) == 100_000 - 2_900, \
        "direct calls cost the operator's call price (100,000 cents): the subscription is the way in"
    m = f.do("cloud.metrics", host=info["host"], hours=2)
    assert sum(h["ok"] for h in m["hours"]) >= 4


def f_educational_products():
    w = make_world(108)
    need = "schedule.jobs"
    nd = NEEDS[need]
    f = join(w, "teacher", 400_000)
    org = f.do("venture.form", handle="academy", name="Academy", capital_cents=100_000)["org"]
    course = f.do("code.create_repo", name="scheduling-course", owner=org)["repo"]
    f.do("code.commit", repo=course, message="course", files={
        "README.md": "# Scheduling\nLesson 1: sort jobs by decreasing length, give each to the least loaded machine.",
        "lesson.py": tier_source(need, len(nd.tiers) - 1)})
    p = f.do("prod.create", name="Scheduling course", kind="education", owner=org,
             delivery={"type": "repo", "repo": course, "need": need})["product"]
    f.do("prod.update", product=p, status="launched")
    off = f.do("prod.offer", product=p, kind="one_time", price_cents=9_900)["offer"]
    learner = join(w, "learner", 200_000)
    own = learner.do("code.create_repo", name="mine")["repo"]
    learner.do("code.commit", repo=own, files={"naive.py": tier_source(need, 0)}, message="what I knew")
    rng = np.random.default_rng(3)
    cases = [nd.generate(rng, 0.5) for _ in range(5)]

    def quality(repo, target):
        qs = []
        for inp, hidden in cases:
            r = learner.do("code.run", repo=repo, target=target, input=inp)
            qs.append(float(nd.score(inp, hidden, r["result"])) if r["ok"] else 0.0)
        return float(np.mean(qs))
    before = quality(own, "naive.py:solve")
    assert not learner.attempt("code.run", repo=course, target="lesson.py:solve", input=cases[0][0]).ok
    learner.do("prod.buy", offer=off)
    after = quality(course, "lesson.py:solve")
    assert after - before > 0.05, (before, after)


def f_media_products():
    w = make_world(109)
    f, org, prod, plans, doc = media_business(w, "mediaco", monthly=800)
    fan, other = join(w, "fan", 20_000), join(w, "other", 20_000)
    assert not other.attempt("docs.get", doc=doc).ok
    sid = fan.do("subs.subscribe", plan=plans["month"])["subscription"]
    assert fan.attempt("docs.get", doc=doc).ok
    fan.do("prod.access", product=prod)
    use = w.state.tables["prod_use_holders"][f"{prod}|{fan.id}"]
    assert use["uses"] == 1
    fan.do("subs.cancel", subscription=sid, at_period_end=False)
    assert not fan.attempt("docs.get", doc=doc).ok


def f_developer_tools():
    w = make_world(110)
    f = join(w, "tool-f", 400_000)
    org = f.do("venture.form", handle="toolco", name="ToolCo", capital_cents=100_000)["org"]
    repo = f.do("code.create_repo", name="sdk", owner=org)["repo"]
    f.do("code.commit", repo=repo, files={"sdk.py": "def mean(xs):\n    return sum(xs) / len(xs)\n"})
    p = f.do("prod.create", name="Stats SDK", kind="devtool", owner=org, delivery={"type": "repo", "repo": repo})["product"]
    f.do("prod.update", product=p, status="launched")
    off = f.do("prod.offer", product=p, kind="one_time", price_cents=3_000)["offer"]
    dev = join(w, "dev", 20_000)
    assert not dev.attempt("code.fork", repo=repo, name="mysdk").ok
    dev.do("prod.buy", offer=off)
    mine = dev.do("code.fork", repo=repo, name="mysdk")["repo"]
    dev.do("code.commit", repo=mine, files={"app.py": "from sdk import mean\n\ndef main(x, caller=None):\n"
                                                       "    return mean(x)\n"}, message="use the sdk")
    r = dev.do("code.run", repo=mine, target="app.py:main", input=[2, 4, 9])
    assert r["ok"] and r["result"] == 5.0, r


def f_data_services():
    w = make_world(111, providers=True)
    need = "forecast.series"
    nd = NEEDS[need]
    f, org, prod, plan, info = saas_business(w, "dataco", need, tier=len(nd.tiers) - 1, monthly=4_000, kind="data")
    c = join(w, "analyst", 100_000)
    c.do("subs.subscribe", plan=plan)
    rng = np.random.default_rng(11)
    cases = [nd.generate(rng, 0.5) for _ in range(4)]
    got = float(np.mean([_score(need, c.do("prod.use", product=prod, input=i), i, h) for i, h in cases]))
    from crucible.economy.needs import compile_solver
    base = compile_solver(tier_source(need, 0))
    naive = float(np.mean([nd.score(i, h, base(i, None)) for i, h in cases]))
    assert w.state.tables["products"][prod]["kind"] == "data" and got > naive, (got, naive)


def f_infrastructure_services():
    w = make_world(112, providers=True)
    need = "schedule.jobs"
    nd = NEEDS[need]
    f, org, prod, plan, info = saas_business(w, "infraco", need, tier=len(nd.tiers) - 1, monthly=9_000,
                                             kind="infrastructure")
    boss = join(w, "ops-boss", 300_000)
    cust = boss.do("venture.form", handle="factory", name="Factory", capital_cents=100_000)
    boss.do("subs.subscribe", plan=plan, account=cust["account"])
    rng = np.random.default_rng(5)
    for _ in range(3):
        i, h = nd.generate(rng, 0.5)
        assert _score(need, boss.do("prod.use", product=prod, input=i), i, h) > 0.95
    w.advance_time(2 * DAY)
    pr = revenue.profit(w, org, 30 * DAY)
    assert pr["revenue"] == 9_000 and pr["infrastructure_costs"] > 0
    assert revenue.m_independent_customers(w, org) == 1.0


def f_industrial_products():
    w = make_world(113)
    f = join(w, "maker", 300_000)
    org = f.do("venture.form", handle="foundry", name="Foundry", capital_cents=50_000)["org"]
    mint(w, org, "steel-beam", 50)
    p = f.do("prod.create", name="Steel beam", kind="industrial", owner=org,
             delivery={"type": "goods", "sku": "steel-beam"})["product"]
    f.do("prod.update", product=p, status="launched")
    off = f.do("prod.offer", product=p, kind="one_time", price_cents=1_200)["offer"]
    b = join(w, "builder", 100_000)
    b.do("prod.buy", offer=off, qty=5)
    inv = w.state.tables["inventory"]
    assert inv[f"{b.id}|steel-beam"]["qty"] == 5 and inv[f"{org}|steel-beam"]["qty"] == 45
    assert balance(w, b.acct) == 100_000 - 6_000
    assert not b.attempt("prod.buy", offer=off, qty=46).ok, "out of stock"
    assert not healthy(w)


def f_contracts():
    w = make_world(114)
    f, org, prod, plans, _ = media_business(w, "vendor", monthly=None)
    boss = join(w, "client-boss", 400_000)
    corg = boss.do("venture.form", handle="clientco", name="ClientCo", capital_cents=200_000)["org"]
    cid = boss.do("contracts.propose", parties=[corg, org], title="Support agreement", as_org=corg, clauses=[
        {"type": "payment", "from": corg, "to": org, "amount": 10_000, "every_s": 30 * DAY, "count": 3,
         "start_in_s": 0},
        {"type": "deliverable", "by": org, "due_in_s": 20 * DAY, "penalty": None,
         "predicate": {"metric": "org_status_active", "args": {"org": org}, "op": ">=", "value": 1}}])["contract"]
    f.do("contracts.sign", contract=cid, as_org=org)
    w.advance_time(70 * DAY)
    c = w.state.tables["contracts"][cid]
    assert sum(c["paid"].values()) == 3 and c["status"] in ("active", "completed"), c["status"]
    assert revenue.profit(w, org, 90 * DAY)["revenue"] == 30_000, "contract revenue from an independent client"


def f_refunds():
    w = make_world(115)
    f, org, prod, plans, _ = media_business(w, "refco", monthly=2_000)
    off = f.do("prod.offer", product=prod, kind="one_time", price_cents=6_000)["offer"]
    a, b = join(w, "ra", 20_000), join(w, "rb", 20_000)
    sid = a.do("subs.subscribe", plan=plans["month"])["subscription"]
    pur = b.do("prod.buy", offer=off)["purchase"]
    inv = [i for i in a.do("subs.invoices", subscription=sid) if i["status"] == "paid"][0]
    f.do("subs.refund", invoice=inv["id"], amount_cents=800, reason="late")
    f.do("prod.refund", purchase=pur, amount_cents=6_000, reason="returned")
    assert balance(w, a.acct) == 18_800 and balance(w, b.acct) == 20_000
    assert revenue.revenue_report(w, org)["refunds_cents"] == 6_800


def f_cancellations():
    w = make_world(116)
    f, org, prod, plans, _ = media_business(w, "canco", monthly=2_000)
    c = join(w, "subscriber", 20_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    c.do("subs.cancel", subscription=sid, at_period_end=True)
    w.advance_time(31 * DAY)
    s = c.do("subs.mine")[0]
    assert s["status"] == "cancelled" and balance(w, c.acct) == 18_000


def f_customer_support():
    w = make_world(117)
    f, org, prod, plans, _ = media_business(w, "supco", monthly=1_000)
    c, stranger = join(w, "customer", 20_000), join(w, "stranger", 1_000)
    assert not stranger.attempt("prod.support", product=prod, subject="hello").ok
    c.do("subs.subscribe", plan=plans["month"])
    t = c.do("prod.support", product=prod, subject="cannot open the guide", body="it is blank")["ticket"]
    q = f.do("support.list", owner=org)
    assert [x["ticket"] for x in q] == [t]
    f.do("support.reply", ticket=t, body="fixed - please reload")
    f.do("support.resolve", ticket=t, note="content republished")
    assert w.state.tables["sup_tickets"][t]["status"] == "resolved"
    w.advance_time(2 * DAY)
    ov = f.do("venture.overview", org=org)
    assert ov["support"] is not None and ov["subscriptions"]["active"] == 1


def f_competition():
    need = "text.extract"
    cfg = {"patrons": {"n": 80, "prefix": "cc", "needs": {need: 1.0}, "wtp_month_cents": [2_500, 6_000],
                       "min_quality": [0.86, 0.97], "search_p": 0.4, "act_every_days": 2, "checks": 1}}
    w = make_world(118, configs=cfg, extra=["patrons"], providers=True)
    weak = saas_business(w, "cheapco", need, tier=0, monthly=1_900, trial_days=7)
    strong = saas_business(w, "bestco", need, tier=len(NEEDS[need].tiers) - 1, monthly=2_900, trial_days=7)
    w.advance_time(45 * DAY)
    n_weak = revenue.m_active_subscriptions(w, weak[1])
    n_strong = revenue.m_active_subscriptions(w, strong[1])
    rejected_weak = sum(1 for s in w.state.find("sub_subs", "merchant", weak[1])
                        if s["cancel_reason"] in ("trial_cancelled", "customer"))
    assert n_strong > 2 * max(1.0, n_weak) and rejected_weak > 0, (n_weak, n_strong, rejected_weak)
    assert not failures(w)


def f_bankruptcy():
    # (a) orderly wind-down: customers get their unused prepaid time back, the company dissolves
    w = make_world(119)
    f, org, prod, plans, _ = media_business(w, "windco", capital=50_000, monthly=3_000)
    cs = [join(w, f"w{i}", 20_000) for i in range(3)]
    for c in cs:
        c.do("subs.subscribe", plan=plans["month"])
    w.advance_time(10 * DAY)
    out = f.do("venture.wind_down", org=org, reason="closing")
    assert out["status"] == "dissolved" and out["refunded_cents"] == 3 * (3_000 * 20 // 30)
    assert all(balance(w, c.acct) == 20_000 - 3_000 + 2_000 for c in cs)
    assert w.state.tables["companies"][org]["status"] == "dissolved"
    assert w.state.tables["products"][prod]["status"] == "retired"
    assert all(s["status"] == "cancelled" for s in w.state.find("sub_subs", "merchant", org))
    # (b) insolvency: a debt it cannot pay; customers' prepaid time becomes claims in the bankruptcy
    w = make_world(120)
    f, org, prod, plans, _ = media_business(w, "bustco", capital=5_000, monthly=3_000)
    lender = join(w, "lender", 500_000)
    cs = [join(w, f"b{i}", 20_000) for i in range(3)]
    for c in cs:
        c.do("subs.subscribe", plan=plans["month"])
    cid = f.do("contracts.propose", parties=[org, lender.id], title="Supply", as_org=org, clauses=[
        {"type": "payment", "from": org, "to": lender.id, "amount": 400_000, "every_s": None, "count": None,
         "start_in_s": 3600}])["contract"]
    lender.do("contracts.sign", contract=cid)
    w.advance_time(17 * DAY)
    st = w.state
    assert st.tables["companies"][org]["status"] == "bankrupt", st.tables["companies"][org]
    claims = [c for c in st.tables["claims"].values() if c["debtor"] == org and c["creditor"] in {x.id for x in cs}]
    assert len(claims) == 3 and all(c["status"] in ("settled", "written_off") for c in claims)
    assert sum(int(c.get("paid") or 0) for c in claims) > 0, "customers share the remaining cash"
    assert all(s["status"] == "cancelled" for s in st.find("sub_subs", "merchant", org))
    assert not healthy(w)


def f_revenue_only_from_independent_participants():
    w = make_world(121)
    f, org, prod, plans, _ = media_business(w, "labelco", monthly=5_000, controller="agent-l")
    w.advance_time(31 * DAY)
    assert revenue.mrr_exact(w, org) == 0 and revenue.profit(w, org)["revenue"] == 0, "no customers, no revenue"
    assert not f.attempt("prod.rate", product=prod, stars=5).ok, "sellers cannot rate themselves"
    names = {e["name"] for e in w.available_tools(f.id)}
    assert not any(n.startswith(("prod.set_", "subs.set_")) for n in names)
    from crucible.commerce import integrity
    integrity.SCHEMES["direct_funding"](w, "lbl", f.id, org, plans["month"], 5_000)
    assert revenue.mrr_exact(w, org) == 0, "money the business sends out and gets back is not revenue"


FEATURES = {"formation": f_formation, "product_development": f_product_development, "subscriptions": f_subscriptions,
            "one_time_purchases": f_one_time_purchases, "licensing": f_licensing, "marketplaces": f_marketplaces,
            "saas": f_saas, "educational_products": f_educational_products, "media_products": f_media_products,
            "developer_tools": f_developer_tools, "data_services": f_data_services,
            "infrastructure_services": f_infrastructure_services, "industrial_products": f_industrial_products,
            "contracts": f_contracts, "refunds": f_refunds, "cancellations": f_cancellations,
            "customer_support": f_customer_support, "competition": f_competition, "bankruptcy": f_bankruptcy,
            "independent_revenue_only": f_revenue_only_from_independent_participants}


def test_SP_6_1_businesses_and_products():
    results = {}
    for name, fn in FEATURES.items():
        try:
            fn()
            results[name] = "passed"
        except AssertionError as e:
            results[name] = f"failed: {e}"[:300]
        except Exception as e:
            results[name] = f"error: {type(e).__name__}: {e} {traceback.format_exc()[-400:]}"[:600]
    n = sum(1 for v in results.values() if v == "passed")
    record("SP-6.1", {"business_features": n, "business_features_total": len(FEATURES)},
           notes="; ".join(f"{k}: {v}" for k, v in results.items()))
    assert n == len(FEATURES), {k: v for k, v in results.items() if v != "passed"}
