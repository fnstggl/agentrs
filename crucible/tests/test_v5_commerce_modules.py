"""Crucible 5.0 commerce (workstream F1): module admission, determinism, source-of-funds tracing, public surfaces."""
import os
import sys

import pytest

import crucible

crucible.load_all_modules()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "v5"))
from _commerce_kit import DAY, join, make_world, media_business  # noqa: E402

from crucible.commerce import integrity, revenue  # noqa: E402
from crucible.core.kernel import World, WorldSpec  # noqa: E402
from crucible.goals.metrics import METRIC_SUBJECTS, evaluate_predicate  # noqa: E402
from crucible.worldgen.admission import admit  # noqa: E402

BASE = ["identity", "bank", "admin", "code", "search", "docs", "support", "contracts", "business", "cloud", "market"]


@pytest.mark.parametrize("mods", [["products"], ["products", "subs"], ["products", "xchg"],
                                  ["products", "subs", "ventures"], ["products", "subs", "patrons"],
                                  ["products", "subs", "xchg", "ventures", "patrons"]])
@pytest.mark.parametrize("seed", [0, 1])
def test_commerce_modules_pass_admission(mods, seed):
    cfg = {"patrons": {"n": 5, "sellers": {"n": 2}, "goods": {"n": 3}}}
    rep = admit(mods, base=BASE, steps=300, seed=seed, configs=cfg)
    assert rep.passed, (rep.checks, rep.errors[:5])


def _economy_spec(seed):
    from crucible.commerce.reference import BASE_MODULES
    mods = [{"name": m, "config": {}} for m in BASE_MODULES + ["xchg"]]
    for m in mods:
        if m["name"] == "ventures":
            m["config"] = {"providers": [{"handle": "cloudco", "name": "CloudCo", "cores": 32, "price": 200}],
                           "incumbents": [{"handle": "inc1", "name": "Inc One", "need": "text.extract", "tier": 0,
                                           "price_month": 1_900, "trial_days": 7},
                                          {"handle": "inc2", "name": "Inc Two", "need": "text.extract", "tier": 1,
                                           "price_month": 2_900, "price_year": 29_000, "trial_days": 7}]}
        if m["name"] == "patrons":
            m["config"] = {"n": 40, "needs": {"text.extract": 1.0}, "act_every_days": 2, "search_p": 0.5,
                           "annual_share": 0.5, "sellers": {"n": 3}, "goods": {"n": 10, "buy_p": 0.8}}
    return WorldSpec("commerce-unit", seed, mods, {"cost_model": "deterministic"})


def test_customer_economy_replays_regenerates_and_forks():
    spec = _economy_spec(5)
    w = World.create(spec)
    w.system_call("admin.join", {"handle": "opx", "endowment": 100_000})
    w.execute("p:opx", {"tool": "xchg.create", "args": {"name": "Mart", "take_rate_bps": 600}})
    w.advance_time(70 * DAY)
    st = w.state
    paid = [i for i in st.tables["sub_invoices"].values() if i["status"] == "paid" and i["amount"] > 0]
    assert len({i["customer"] for i in paid}) >= 10, "independent customers paid from their own accounts"
    assert any(i["interval"] == "year" for i in paid) and st.find_ids("xchg_orders", "status", "settled")
    assert not [e for e in w.log if e.type in ("patrons.failed", "subs.bill_failed", "proc.failed")]
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []
    evs = [e.to_dict() for e in w.log]
    assert World.replay(spec, evs).digest() == w.digest()
    g = World.regenerate(spec, evs)
    assert g.last_hash == w.last_hash
    f = w.fork("commerce-fork")
    d0 = w.digest()
    f.advance_time(f.now + 5 * DAY)
    assert w.digest() == d0 and f.digest() != d0
    assert World.from_blob(w.snapshot_blob()).digest() == d0


def _pay_setup(own_cents):
    w = make_world(301)
    f, org, prod, plans, _ = media_business(w, "trace", capital=1_000_000, monthly=1_000)
    x = join(w, "xpayer", own_cents)
    return w, f, org, plans, x


def _goods_sale(w, seller, buyer, sku, price):
    lst = seller.do("market.list", kind="goods", title=f"{sku} x1", price_cents=price, sku=sku)["listing"]
    o = buyer.do("market.buy", listing=lst, account=buyer.acct)["order"]
    seller.do("market.deliver", order=o)
    buyer.do("market.confirm", order=o)


def test_pro_rata_tracing_attributes_only_the_owner_share():
    from _commerce_kit import mint
    w, f, org, plans, x = _pay_setup(9_000)
    # a market price for bolts, set by other pairs of participants
    s1, b1 = join(w, "boltmaker", 2_000), join(w, "boltbuyer", 5_000)
    mint(w, s1.id, "bolt", 5)
    _goods_sale(w, s1, b1, "bolt", 1_000)
    mint(w, x.id, "bolt", 5)          # the merchant buys real goods from x at the market price: not netted
    lst = x.do("market.list", kind="goods", title="Bolts", price_cents=1_000, sku="bolt")["listing"]
    o = f.do("market.buy", listing=lst, account=w.modules["bank"].primary_account(w.state, org))["order"]
    x.do("market.deliver", order=o)
    f.do("market.confirm", order=o)
    x.do("subs.subscribe", plan=plans["month"])
    rec = integrity.related_flows(w, org)["payers"][x.id]
    # the 1,000 cents x received were the market price of the bolts it delivered: its own money, not traced
    assert rec["funded"] == 0 and rec["netted"] == 0 and not rec["flagged"], rec
    assert rec["weight"] == 1


def test_pro_rata_tracing_of_money_given_without_consideration():
    w, f, org, plans, x = _pay_setup(9_000)
    f.do("bank.transfer", from_account=w.modules["bank"].primary_account(w.state, org), to=x.id, amount=1_000,
         memo="gift")
    x.do("bank.transfer", from_account=x.acct, to=join(w, "elsewhere", 0).id, amount=500)
    x.do("subs.subscribe", plan=plans["month"])
    rec = integrity.related_flows(w, org)["payers"][x.id]
    # x held 9,000 of its own and 1,000 of the merchant's, then spent 500 pro rata: 1,000 x (9,500/10,000) left of
    # 9,500, so the 1,000-cent payment carries 1,000 x 950/9,500 = 100 traced cents; netting takes the rest
    assert rec["funded"] == pytest.approx(100.0, abs=0.01), rec
    assert rec["netted"] == 1_000 and rec["weight"] == 0


def test_goods_above_market_price_are_netted():
    from _commerce_kit import mint
    w, f, org, plans, x = _pay_setup(9_000)
    s1, b1 = join(w, "boltmaker", 2_000), join(w, "boltbuyer", 5_000)
    mint(w, s1.id, "bolt", 5)
    _goods_sale(w, s1, b1, "bolt", 100)             # bolts change hands at 1 dollar
    mint(w, x.id, "bolt", 5)
    lst = x.do("market.list", kind="goods", title="Bolts", price_cents=1_000, sku="bolt")["listing"]
    o = f.do("market.buy", listing=lst, account=w.modules["bank"].primary_account(w.state, org))["order"]
    x.do("market.deliver", order=o)
    f.do("market.confirm", order=o)
    x.do("subs.subscribe", plan=plans["month"])
    rec = integrity.related_flows(w, org)["payers"][x.id]
    # 900 of the 1,000 cents are above the market price: they can have come back as x's payment
    assert rec["netted"] == 900, rec


def test_goods_without_any_market_price_are_netted():
    from _commerce_kit import mint
    w, f, org, plans, x = _pay_setup(9_000)
    mint(w, x.id, "bolt", 5)
    lst = x.do("market.list", kind="goods", title="Bolts", price_cents=1_000, sku="bolt")["listing"]
    o = f.do("market.buy", listing=lst, account=w.modules["bank"].primary_account(w.state, org))["order"]
    x.do("market.deliver", order=o)
    f.do("market.confirm", order=o)
    x.do("subs.subscribe", plan=plans["month"])
    rec = integrity.related_flows(w, org)["payers"][x.id]
    assert rec["netted"] == 1_000 and rec["flagged"], rec


def test_direct_transfers_are_netted():
    w, f, org, plans, x = _pay_setup(9_000)
    f.do("bank.transfer", from_account=w.modules["bank"].primary_account(w.state, org), to=x.id, amount=1_000)
    x.do("subs.subscribe", plan=plans["month"])
    rec = integrity.related_flows(w, org)["payers"][x.id]
    assert rec["funded"] == pytest.approx(100.0) and rec["netted"] == 1_000 and rec["flagged"], rec


def test_pass_through_money_is_traced_through_chains_with_mixing():
    w, f, org, plans, x = _pay_setup(0)
    mid = join(w, "middle", 1_000)               # half of what it forwards is its own money
    f.do("bank.transfer", from_account=w.modules["bank"].primary_account(w.state, org), to=mid.id, amount=1_000)
    mid.do("bank.transfer", from_account=mid.acct, to=x.id, amount=2_000)
    x.do("subs.subscribe", plan=plans["month"])
    rec = integrity.related_flows(w, org)["payers"][x.id]
    assert rec["funded"] == pytest.approx(500.0), rec
    assert rec["flagged"], "transfers are netted: the owner's 1,000 can have come back"


def test_refunds_return_the_customers_own_money():
    w, f, org, plans, x = _pay_setup(5_000)
    sid = x.do("subs.subscribe", plan=plans["month"])["subscription"]
    inv = [i for i in x.do("subs.invoices", subscription=sid) if i["status"] == "paid"][0]
    f.do("subs.refund", invoice=inv["id"], reason="sorry")
    x.do("subs.cancel", subscription=sid, at_period_end=False)
    x.do("subs.subscribe", plan=plans["month"])
    rec = integrity.related_flows(w, org)["payers"][x.id]
    assert rec["funded"] == 0 and not rec["flagged"] and rec["paid"] == 1_000, rec
    assert revenue.m_independent_customers(w, org) == 1.0


def test_evaluator_metrics_are_evaluator_only_and_booked_metrics_are_readable():
    from crucible.commerce.observables import EVALUATOR_ONLY, in_world_errors
    for name in ("mrr_cents", "arr_cents", "active_subscriptions", "churn_rate", "independent_customers",
                 "profit_cents", "one_time_revenue_cents"):
        assert name not in METRIC_SUBJECTS and name in EVALUATOR_ONLY       # no in-world predicate can read them
    for name in ("booked_mrr_cents", "booked_subscribers", "booked_revenue_cents"):
        assert METRIC_SUBJECTS[name] == {"principal": "principal"}
    w = make_world(302)
    f, org, prod, plans, _ = media_business(w, "pred", monthly=None, annual=1_250_000)
    for i in range(8):
        join(w, f"pr{i}", 1_300_000).do("subs.subscribe", plan=plans["year"])
    # the evaluator still computes them (outside the world)
    p = {"metric": "arr_cents", "args": {"principal": org}, "op": ">=", "value": 10_000_000}
    assert evaluate_predicate(w, p) and round(revenue.m_mrr(w, org) / 100, 2) == 8_333.33
    assert in_world_errors(w, p, [org])                                   # but not through an agreement
    b = {"metric": "booked_mrr_cents", "args": {"principal": org}, "op": ">=", "value": 1}
    assert in_world_errors(w, b, [org]) == [] and evaluate_predicate(w, b)


def test_public_surfaces_carry_no_private_or_evaluator_data():
    w = make_world(303)
    f, org, prod, plans, _ = media_business(w, "pub", monthly=1_000)
    for i in range(3):
        join(w, f"pb{i}", 10_000).do("subs.subscribe", plan=plans["month"])
    stranger = join(w, "nosy", 1_000)
    entry = stranger.do("prod.catalog")[0]
    assert set(entry) == {"product", "name", "owner", "kind", "need", "delivery", "status", "version", "releases",
                          "description", "rating", "ratings", "offers", "plans"}
    assert not stranger.attempt("venture.overview", org=org).ok
    assert not stranger.attempt("subs.invoices", subscription=w.state.ids("sub_subs")[0]).ok
    assert stranger.do("subs.mine", role="merchant") == []
    tools = {t["name"] for t in w.available_tools(stranger.id)}
    assert not any("flag" in t or "integrity" in t or "related" in t for t in tools)
    for e in w.log:
        if e.type in ("subs.subscription", "subs.invoice") and e.vis != "internal":
            assert stranger.id not in e.vis
