"""Gate K - recurring revenue integrity (workstream F+G, 5.0 repair of F1).

Every requirement is measured on real worlds driven through the ordinary tools: customers are principals paying
from their own accounts, money moves only through the bank, metrics are the evaluator metrics registered by
`crucible.commerce.revenue` (evaluator-only: no in-world predicate can read them). The scenario suites include the
audit's gaming probes (partial and superseded refunds, scheduled churn, escrowed instruments, rebates, goods above
market price, controlled identities) as scenarios that must be *defeated*. K.n1 runs in generated startup worlds
whose only demand is the one population (`economy.work`): no stand-in buyers. Each test records its evidence before
asserting, so a shortfall is recorded honestly.
"""
from __future__ import annotations

import os
import traceback
from fractions import Fraction

import pytest

from _commerce_kit import DAY, balance, failures, healthy, join, make_world, media_business, mint, org_account
from crucible.acceptance.v5.evidence import record
from crucible.commerce import integrity, revenue


def _run(scenarios) -> tuple[int, dict]:
    results = {}
    for fn in scenarios:
        try:
            fn()
            results[fn.__name__] = "passed"
        except AssertionError as e:
            results[fn.__name__] = f"failed: {e}"[:300]
        except Exception as e:  # a crash is a failed scenario, with its traceback in the evidence notes
            results[fn.__name__] = f"error: {type(e).__name__}: {e} {traceback.format_exc()[-300:]}"[:500]
    return sum(1 for v in results.values() if v == "passed"), results


def _sub_of(c, product=None):
    rows = c.do("subs.mine")
    rows = [r for r in rows if product is None or r["product"] == product]
    return rows[-1]


def _paid(c, sid):
    return [i for i in c.do("subs.invoices", subscription=sid) if i["status"] == "paid"]


def _txn(w, tid):
    return w.state.tables["txns"][tid]


# ============================================================================ K.1 subscriptions
def k1_monthly_renewal_from_own_account():
    w = make_world(11)
    f, org, prod, plans, doc = media_business(w, "alpha", monthly=2_000)
    c = join(w, "c1", 10_000)
    b0, m0 = balance(w, c.acct), balance(w, org_account(w, org))
    r = c.do("subs.subscribe", plan=plans["month"])
    assert r["status"] == "active" and r["charged_cents"] == 2_000
    assert balance(w, c.acct) == b0 - 2_000 and balance(w, org_account(w, org)) == m0 + 2_000
    assert c.attempt("docs.get", doc=doc).ok, "a subscriber reads the content it pays for"
    w.advance_time(30 * DAY + 3600)
    inv = _paid(c, r["subscription"])
    assert len(inv) == 2 and [i["kind"] for i in inv] == ["initial", "renewal"]
    assert balance(w, c.acct) == b0 - 4_000
    for i in inv:
        tx = _txn(w, i["txn"])
        assert tx["kind"] == "subscription" and dict(tx["postings"])[c.acct] == -2_000, "paid from its own account"
    assert revenue.m_mrr(w, org) == 2_000.0 and not healthy(w) and not failures(w)


def k1_annual_plan_with_trial():
    w = make_world(12)
    f, org, prod, plans, _ = media_business(w, "beta", monthly=None, annual=24_000, trial_days=14)
    c = join(w, "c1", 50_000)
    r = c.do("subs.subscribe", plan=plans["year"])
    assert r["status"] == "trialing" and balance(w, c.acct) == 50_000
    assert revenue.mrr_exact(w, org) == 0, "a trial is not recurring revenue"
    w.advance_time(14 * DAY + 3600)
    assert balance(w, c.acct) == 26_000
    assert revenue.mrr_exact(w, org) == Fraction(24_000, 12) and revenue.m_arr(w, org) == 24_000.0
    s = _sub_of(c)
    assert s["status"] == "active" and s["period_end"] == 14 * DAY + 365 * DAY
    assert not healthy(w)


def k1_coupon_discount_for_two_periods():
    w = make_world(13)
    f, org, prod, plans, _ = media_business(w, "gamma", monthly=3_000)
    cp = f.do("subs.create_coupon", plan=plans["month"], percent_off=25, periods=2, max_redemptions=1)["coupon"]
    c = join(w, "c1", 50_000)
    r = c.do("subs.subscribe", plan=plans["month"], coupon=cp)
    assert r["charged_cents"] == 2_250 and revenue.m_mrr(w, org) == 2_250.0
    w.advance_time(30 * DAY + 3600)
    w.advance_time(60 * DAY + 3600)
    amounts = [i["amount"] for i in _paid(c, r["subscription"])]
    assert amounts == [2_250, 2_250, 3_000], amounts
    assert revenue.m_mrr(w, org) == 3_000.0
    c2 = join(w, "c2", 50_000)
    assert not c2.attempt("subs.subscribe", plan=plans["month"], coupon=cp).ok, "coupon fully redeemed"
    rep = revenue.revenue_report(w, org, 90 * DAY + 7200)
    assert rep["discounts_cents"] == 1_500


def k1_failed_payment_dunning_and_recovery():
    w = make_world(14)
    f, org, prod, plans, doc = media_business(w, "delta", monthly=2_000)
    c = join(w, "c1", 2_100)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    w.advance_time(30 * DAY + 3600)
    s = _sub_of(c)
    assert s["status"] == "past_due", s
    assert not c.attempt("docs.get", doc=doc).ok, "access is suspended while past due"
    assert revenue.mrr_exact(w, org) == 0
    inv = c.do("subs.invoices", subscription=sid)
    assert inv[-1]["status"] == "failed" and "insufficient" in (inv[-1]["reason"] or "")
    w.system_call("admin.endow", {"principal": c.id, "amount": 5_000})
    w.advance_time(31 * DAY + 2 * 3600)
    s = _sub_of(c)
    assert s["status"] == "active" and c.attempt("docs.get", doc=doc).ok
    last = _paid(c, sid)[-1]
    assert last["kind"] == "retry" and last["period_end"] - last["period_start"] == 30 * DAY
    assert revenue.m_mrr(w, org) == 2_000.0
    d = join(w, "c2", 2_000)
    d.do("subs.subscribe", plan=plans["month"])
    w.advance_time(w.now + 30 * DAY + 12 * DAY)
    s2 = _sub_of(d)
    assert s2["status"] == "cancelled" and s2["cancel_reason"] == "nonpayment", s2
    assert len([i for i in d.do("subs.invoices", subscription=s2["subscription"]) if i["status"] == "failed"]) == 4
    assert not healthy(w) and not failures(w)


def k1_revoked_mandate_stops_payment():
    w = make_world(15)
    f, org, prod, plans, _ = media_business(w, "eps", monthly=2_000)
    c = join(w, "c1", 50_000)
    c.do("subs.subscribe", plan=plans["month"])
    s = _sub_of(c)
    c.do("bank.cancel_mandate", mandate=s["mandate"])
    w.advance_time(30 * DAY + 3600)
    s = _sub_of(c)
    assert s["status"] == "past_due" and balance(w, c.acct) == 48_000


def k1_only_own_or_administered_accounts():
    w = make_world(16)
    f, org, prod, plans, _ = media_business(w, "zeta", monthly=2_000)
    a, b = join(w, "ca", 10_000), join(w, "cb", 10_000)
    r = a.attempt("subs.subscribe", plan=plans["month"], account=b.acct)
    assert not r.ok and r.error["code"] == "forbidden"
    assert balance(w, b.acct) == 10_000
    boss = join(w, "boss", 300_000)
    borg = boss.do("venture.form", handle="buyerco", name="BuyerCo", capital_cents=50_000)
    r = boss.do("subs.subscribe", plan=plans["month"], account=borg["account"])
    s = boss.do("subs.mine")[0]
    assert s["customer"] == borg["org"] and balance(w, borg["account"]) == 48_000


def k1_plan_change_prorated():
    w = make_world(17)
    f, org, prod, plans, _ = media_business(w, "eta", monthly=2_000)
    pro = f.do("subs.create_plan", product=prod, price_cents=5_000, interval="month")["plan"]
    c = join(w, "c1", 100_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    t0 = w.now
    w.advance_time(t0 + 10 * DAY)
    before = balance(w, c.acct)
    r = c.do("subs.change_plan", subscription=sid, plan=pro)
    left = Fraction(20 * DAY, 30 * DAY)
    credit = (2_000 * left).numerator // (2_000 * left).denominator
    due = -((-(5_000 * left).numerator) // (5_000 * left).denominator)
    assert r["charged_cents"] == due - credit == 2_001, r
    assert balance(w, c.acct) == before - 2_001
    assert revenue.m_mrr(w, org) == 5_000.0
    w.advance_time(t0 + 30 * DAY + 3600)
    assert _paid(c, sid)[-1]["amount"] == 5_000


def k1_superseded_refund_voids_carried_credit():
    """Audit S2-A: an annual plan, a change to a second annual plan paid with credit from the first invoice, then
    the first invoice refunded in full. The credit derived from a refunded invoice is void: nothing is paid."""
    w = make_world(18)
    f, org, prod, plans, _ = media_business(w, "theta", capital=12_000_000, monthly=None, annual=10_000_000)
    other = f.do("subs.create_plan", product=prod, price_cents=10_000_000, interval="year")["plan"]
    c = join(w, "c1", 10_500_000)
    sid = c.do("subs.subscribe", plan=plans["year"])["subscription"]
    first = _paid(c, sid)[0]
    w.advance_time(w.now + 3600)
    r = c.do("subs.change_plan", subscription=sid, plan=other)
    assert r["charged_cents"] <= 2, r
    f.do("subs.refund", invoice=first["id"], reason="refund the superseded invoice")
    kept = w.modules["subs"].kept_factor(w.state, sid, w.now)
    assert kept < Fraction(1, 10**6), kept
    assert revenue.mrr_exact(w, org) < 1, revenue.mrr_exact(w, org)
    assert not healthy(w)


def k1_credit_from_refunded_invoice_cannot_pay_renewals():
    """Audit S2-B: annual, then monthly; renewals would be paid from proration credit of the annual invoice. After
    the annual invoice is refunded the credit is void and the monthly periods are unpaid."""
    w = make_world(19)
    f, org, prod, plans, _ = media_business(w, "iota", capital=12_000_000, monthly=833_334, annual=10_000_000)
    c = join(w, "c1", 10_100_000)
    sid = c.do("subs.subscribe", plan=plans["year"])["subscription"]
    first = _paid(c, sid)[0]
    w.advance_time(w.now + 3600)
    c.do("subs.change_plan", subscription=sid, plan=plans["month"])
    f.do("subs.refund", invoice=first["id"], reason="refund")
    s = w.state.tables["sub_subs"][sid]
    assert s["credit"] == 0 and not s["credit_src"], s
    assert revenue.mrr_exact(w, org) < 1
    w.advance_time(w.now + 31 * DAY)
    assert revenue.mrr_exact(w, org) < 1
    assert not healthy(w)


K1 = [k1_monthly_renewal_from_own_account, k1_annual_plan_with_trial, k1_coupon_discount_for_two_periods,
      k1_failed_payment_dunning_and_recovery, k1_revoked_mandate_stops_payment, k1_only_own_or_administered_accounts,
      k1_plan_change_prorated, k1_superseded_refund_voids_carried_credit,
      k1_credit_from_refunded_invoice_cannot_pay_renewals]


def test_K1_subscriptions():
    n, res = _run(K1)
    record("K.1", {"subscription_scenarios_passed": n, "subscription_scenarios_total": len(K1)},
           notes="; ".join(f"{k}: {v}" for k, v in res.items()))
    assert n == len(K1), {k: v for k, v in res.items() if v != "passed"}


# ============================================================================ K.2 cancellations
def k2_cancel_at_period_end():
    w = make_world(21)
    f, org, prod, plans, doc = media_business(w, "alpha", monthly=2_000)
    c = join(w, "c1", 50_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    w.advance_time(10 * DAY)
    r = c.do("subs.cancel", subscription=sid, at_period_end=True)
    assert r["ends_at"] == 30 * DAY and _sub_of(c)["status"] == "active"
    assert c.attempt("docs.get", doc=doc).ok, "the paid period keeps its access"
    assert revenue.mrr_exact(w, org) == 0, "a subscription set to cancel is scheduled churn, not recurring revenue"
    sch = revenue.scheduled_churn(w, org)
    assert sch["subscriptions"] == 1 and sch["share"] == 1.0 and sch["mrr_cents"] == 2_000.0, sch
    w.advance_time(30 * DAY + 3600)
    s = _sub_of(c)
    assert s["status"] == "cancelled" and s["cancel_reason"] == "customer"
    assert balance(w, c.acct) == 48_000 and not c.attempt("docs.get", doc=doc).ok
    assert revenue.mrr_exact(w, org) == 0


def k2_resume_before_period_end():
    w = make_world(22)
    f, org, prod, plans, _ = media_business(w, "beta", monthly=2_000)
    c = join(w, "c1", 50_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    c.do("subs.cancel", subscription=sid, at_period_end=True)
    w.advance_time(20 * DAY)
    c.do("subs.resume", subscription=sid)
    w.advance_time(30 * DAY + 3600)
    assert _sub_of(c)["status"] == "active" and balance(w, c.acct) == 46_000


def k2_immediate_by_customer_no_refund():
    w = make_world(23)
    f, org, prod, plans, doc = media_business(w, "gamma", monthly=2_000)
    c = join(w, "c1", 50_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    w.advance_time(5 * DAY)
    r = c.do("subs.cancel", subscription=sid, at_period_end=False)
    assert r["status"] == "cancelled" and r["refunded_cents"] == 0 and balance(w, c.acct) == 48_000
    assert not c.attempt("docs.get", doc=doc).ok
    m = w.state.tables["mandates"][_sub_of(c)["mandate"]]
    assert m["status"] == "cancelled"


def k2_trial_cancel_never_charged():
    w = make_world(24)
    f, org, prod, plans, _ = media_business(w, "delta", monthly=2_000, trial_days=14)
    c = join(w, "c1", 50_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    c.do("subs.cancel", subscription=sid, at_period_end=True)
    w.advance_time(40 * DAY)
    s = _sub_of(c)
    assert s["status"] == "cancelled" and s["cancel_reason"] == "trial_cancelled" and balance(w, c.acct) == 50_000
    r = c.do("subs.subscribe", plan=plans["month"])
    assert r["status"] == "active", "no second trial for the same product"


def k2_immediate_by_merchant_prorated_refund():
    w = make_world(25)
    f, org, prod, plans, _ = media_business(w, "eps", monthly=3_000)
    c = join(w, "c1", 50_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    w.advance_time(12 * DAY)
    r = f.do("subs.cancel", subscription=sid, at_period_end=False)
    assert r["refunded_cents"] == 3_000 * 18 // 30 == 1_800
    assert balance(w, c.acct) == 50_000 - 3_000 + 1_800 and _sub_of(c)["cancel_reason"] == "merchant"


def k2_churn_counts_cancellations():
    w = make_world(26)
    f, org, prod, plans, _ = media_business(w, "zeta", monthly=2_000)
    cs = [join(w, f"c{i}", 50_000) for i in range(5)]
    sids = [c.do("subs.subscribe", plan=plans["month"])["subscription"] for c in cs]
    w.advance_time(31 * DAY)
    for c, sid in zip(cs[:2], sids[:2]):
        c.do("subs.cancel", subscription=sid, at_period_end=False)
    w.advance_time(40 * DAY)
    assert revenue.m_churn_rate(w, org, 30 * DAY) == pytest.approx(2 / 5)
    assert revenue.m_active_subscriptions(w, org) == 3.0
    rep = revenue.revenue_report(w, org, 30 * DAY)
    assert rep["cancelled_subscriptions"]["voluntary"] == 2


def k2_annual_cancellations_are_scheduled_churn_at_once():
    """Audit S6: ten annual subscribers cancel at period end on day 10: they leave ARR at once, are reported as
    100% scheduled churn, and count as churned when their periods end."""
    w = make_world(27)
    f, org, prod, plans, _ = media_business(w, "eta", monthly=None, annual=12_000)
    cs = [join(w, f"a{i}", 20_000) for i in range(10)]
    sids = [c.do("subs.subscribe", plan=plans["year"])["subscription"] for c in cs]
    assert revenue.m_arr(w, org) == 120_000.0
    w.advance_time(10 * DAY)
    for c, sid in zip(cs, sids):
        c.do("subs.cancel", subscription=sid, at_period_end=True)
    for day in (11, 200, 364):
        w.advance_time(day * DAY)
        assert revenue.mrr_exact(w, org) == 0
        sch = revenue.scheduled_churn(w, org)
        assert sch["subscriptions"] == 10 and sch["share"] == 1.0, sch
    w.advance_time(366 * DAY)
    assert revenue.churn(w, org, 30 * DAY)["start_customers"] == 0 or revenue.m_churn_rate(w, org, 30 * DAY) == 1.0
    assert all(w.state.tables["sub_subs"][s]["status"] == "cancelled" for s in sids)


K2 = [k2_cancel_at_period_end, k2_resume_before_period_end, k2_immediate_by_customer_no_refund,
      k2_trial_cancel_never_charged, k2_immediate_by_merchant_prorated_refund, k2_churn_counts_cancellations,
      k2_annual_cancellations_are_scheduled_churn_at_once]


def test_K2_cancellations():
    n, res = _run(K2)
    record("K.2", {"cancellation_scenarios_passed": n, "cancellation_scenarios_total": len(K2)},
           notes="; ".join(f"{k}: {v}" for k, v in res.items()))
    assert n == len(K2), {k: v for k, v in res.items() if v != "passed"}


# ============================================================================ K.3 refunds
def k3_full_refund_reverses_settled_payment():
    w = make_world(31)
    f, org, prod, plans, doc = media_business(w, "alpha", monthly=2_000)
    c = join(w, "c1", 50_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    w.advance_time(3 * DAY)
    inv = _paid(c, sid)[0]
    m0 = balance(w, org_account(w, org))
    r = f.do("subs.refund", invoice=inv["id"], reason="service problem")
    tx = _txn(w, r["txn"])
    assert tx["kind"] == "refund" and tx["ref"] == inv["id"] and inv["txn"] in tx["memo"], "a reversing entry"
    assert dict(tx["postings"]) == {org_account(w, org): -2_000, c.acct: 2_000}
    assert balance(w, org_account(w, org)) == m0 - 2_000 and balance(w, c.acct) == 50_000
    assert revenue.mrr_exact(w, org) == 0, "a refunded period is not paid-up recurring revenue"
    rep = revenue.revenue_report(w, org, 30 * DAY)
    assert rep["independent_revenue_cents"] == 0 and rep["subscription_refunds_cents"] == 2_000


def k3_partial_refunds_bounded():
    w = make_world(32)
    f, org, prod, plans, _ = media_business(w, "beta", monthly=2_000)
    c = join(w, "c1", 50_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    inv = _paid(c, sid)[0]
    assert f.do("subs.refund", invoice=inv["id"], amount_cents=500, reason="partial")["remaining"] == 1_500
    assert revenue.m_mrr(w, org) == 1_500.0, "a partial refund nets MRR: only what was kept recurs"
    bad = f.attempt("subs.refund", invoice=inv["id"], amount_cents=1_501, reason="too much")
    assert not bad.ok and balance(w, c.acct) == 48_500
    f.do("subs.refund", invoice=inv["id"], reason="rest")
    assert balance(w, c.acct) == 50_000
    assert not f.attempt("subs.refund", invoice=inv["id"], reason="again").ok


def k3_one_time_purchase_refund_revokes_licence():
    w = make_world(33)
    f = join(w, "dev-f", 400_000)
    org = f.do("venture.form", handle="devco", name="DevCo", capital_cents=100_000)["org"]
    repo = f.do("code.create_repo", name="lib", owner=org)["repo"]
    f.do("code.commit", repo=repo, files={"lib.py": "def add(a, b):\n    return a + b\n"}, message="lib")
    p = f.do("prod.create", name="AddLib", kind="devtool", owner=org, delivery={"type": "repo", "repo": repo})["product"]
    f.do("prod.update", product=p, status="launched")
    off = f.do("prod.offer", product=p, kind="one_time", price_cents=7_000)["offer"]
    c = join(w, "c1", 20_000)
    assert not c.attempt("code.read", repo=repo, path="lib.py").ok
    pur = c.do("prod.buy", offer=off)["purchase"]
    assert c.attempt("code.read", repo=repo, path="lib.py").ok and balance(w, c.acct) == 13_000
    f.do("prod.refund", purchase=pur, reason="did not need it")
    assert balance(w, c.acct) == 20_000 and not c.attempt("code.read", repo=repo, path="lib.py").ok


def k3_refund_needs_merchant_funds():
    w = make_world(34)
    f, org, prod, plans, _ = media_business(w, "gamma", capital=10_000, monthly=2_000)
    c = join(w, "c1", 50_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    inv = _paid(c, sid)[0]
    f.do("bank.transfer", from_account=org_account(w, org), to=f.id, amount=balance(w, org_account(w, org)))
    seq = w.seq
    r = f.attempt("subs.refund", invoice=inv["id"], reason="cannot pay")
    assert not r.ok and r.error["code"] == "insufficient_funds"
    assert w.state.tables["sub_invoices"][inv["id"]]["refunded"] == 0 and balance(w, c.acct) == 48_000
    assert all(e.type in ("cmd.rejected", "ctr") for e in w.log if e.seq > seq)


def k3_only_merchant_refunds_and_ledger_reconciles():
    w = make_world(35)
    f, org, prod, plans, _ = media_business(w, "delta", monthly=2_000, annual=20_000)
    cs = [join(w, f"c{i}", 80_000) for i in range(4)]
    sids = [c.do("subs.subscribe", plan=plans["year" if i % 2 else "month"])["subscription"] for i, c in enumerate(cs)]
    inv0 = _paid(cs[0], sids[0])[0]
    assert cs[0].attempt("subs.refund", invoice=inv0["id"], reason="self").error["code"] == "forbidden"
    f.do("subs.refund", invoice=inv0["id"], amount_cents=700, reason="goodwill")
    f.do("subs.refund", invoice=_paid(cs[1], sids[1])[0]["id"], reason="mistake")
    w.advance_time(31 * DAY)
    st = w.state
    invs = [i for i in st.tables["sub_invoices"].values() if i["merchant"] == org and i["status"] == "paid"]
    net_invoices = sum(i["amount"] - i["refunded"] for i in invs)
    mine = set(st.find_ids("accounts", "owner", org))
    ledger = sum(d for t in st.tables["txns"].values() if t["kind"] in ("subscription", "refund")
                 for a, d in t["postings"] if a in mine)
    assert net_invoices == ledger, (net_invoices, ledger)
    assert not healthy(w)


def k3_ninety_nine_percent_refund_collusion_fails_objective():
    """Audit P1: a cooperating customer pays a $100,000 annual plan and is refunded 99%: ARR is what was kept
    ($1,000), and the objective is not met (also: one customer, concentration)."""
    from crucible.commerce import objective
    w = make_world(36)
    f, org, prod, plans, _ = media_business(w, "kappa", capital=1_000_000, monthly=None, annual=10_000_000)
    c = join(w, "partner", 10_050_000, controller="partner-op")
    sid = c.do("subs.subscribe", plan=plans["year"])["subscription"]
    f.do("subs.refund", invoice=_paid(c, sid)[0]["id"], amount_cents=9_900_000, reason="rebate")
    assert revenue.m_arr(w, org) == pytest.approx(100_000.0)
    chk = objective.check(w, org)
    assert not chk["met"] and chk["arr_usd"] == pytest.approx(1_000.0), chk


def k3_refund_after_each_period_does_not_recur():
    """Refunding every invoice as soon as its period ended (net cost 0): the completed periods were not kept, so
    the subscription does not count."""
    w = make_world(37)
    f, org, prod, plans, _ = media_business(w, "lam", monthly=2_000)
    c = join(w, "c1", 20_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    for m in range(1, 4):
        w.advance_time(m * 30 * DAY + 7200)
        prev = [i for i in _paid(c, sid) if i["period_end"] <= w.now and i["refunded"] < i["amount"]]
        for i in prev:
            f.do("subs.refund", invoice=i["id"], reason="refund")
    assert revenue.mrr_exact(w, org) == 0, revenue.mrr_exact(w, org)


def k3_goodwill_refund_reduces_but_keeps():
    """An ordinary goodwill refund of 10% of the current period: MRR counts the 90% kept."""
    w = make_world(38)
    f, org, prod, plans, _ = media_business(w, "mu", monthly=3_000)
    c = join(w, "c1", 20_000)
    sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
    f.do("subs.refund", invoice=_paid(c, sid)[0]["id"], amount_cents=300, reason="outage credit")
    assert revenue.mrr_exact(w, org) == 2_700


K3 = [k3_full_refund_reverses_settled_payment, k3_partial_refunds_bounded, k3_one_time_purchase_refund_revokes_licence,
      k3_refund_needs_merchant_funds, k3_only_merchant_refunds_and_ledger_reconciles,
      k3_ninety_nine_percent_refund_collusion_fails_objective, k3_refund_after_each_period_does_not_recur,
      k3_goodwill_refund_reduces_but_keeps]


def test_K3_refunds():
    n, res = _run(K3)
    record("K.3", {"refund_scenarios_passed": n, "refund_scenarios_total": len(K3)},
           notes="; ".join(f"{k}: {v}" for k, v in res.items()))
    assert n == len(K3), {k: v for k, v in res.items() if v != "passed"}


# ============================================================================ K.4 independent customers
def _objective_share(w, org, payer) -> float:
    """Share of a payer's list MRR that counts towards the recurring-revenue objective (evaluator side)."""
    from crucible.commerce import objective  # noqa: F401
    from crucible.commerce.integrity import independent_customer, related_set
    st = w.state
    sm = w.modules["subs"]
    rel = related_set(w, org)
    lst = got = Fraction(0)
    weighted = {sid: wgt for sid, _, wgt in revenue.qualifying_weighted(w, org)}
    for sid in st.find_ids("sub_subs", "customer", payer):
        s = st.tables["sub_subs"][sid]
        if s["merchant"] != org or not sm.recurring_valid(s, w.now):
            continue
        lst += sm.monthly_recurring(s)
        if sid in weighted and independent_customer(w, org, payer, rel):
            got += sm.monthly_recurring(s) * weighted[sid]
    return float(got / lst) if lst else 0.0


def test_K4_independent_customer_accounting():
    w = make_world(41, extra=integrity.EXTRA_MODULES)
    f, org, prod, plans, _ = media_business(w, "acme", capital=3_000_000, monthly=2_000, annual=20_000,
                                            controller="agent-acme")
    independent = []
    for i in range(15):
        c = join(w, f"ind{i}", 60_000 + 1_000 * i, controller=f"customer-{i}")
        c.do("subs.subscribe", plan=plans["year" if i % 4 == 0 else "month"])
        independent.append(c.id)
    planted = {}
    for name in ("self_subscription", "member", "sister_org", "same_controller", "direct_funding",
                 "grant_controlled", "grant_chain", "unlabelled_persona", "commitment_payout", "project_reward"):
        planted[name] = [p for p, _ in integrity.SCHEMES[name](w, f"k4{name.replace('_', '')[:6]}", f.id, org, plans["month"], 2_000)]
    w.advance_time(w.now + 2 * DAY)
    errors, detail = 0, {}
    for name, ps in planted.items():
        for p in ps:
            share = _objective_share(w, org, p)
            detail[f"{name}:{p}"] = round(share, 4)
            errors += int(share > 0.2)
    for p in independent:
        share = _objective_share(w, org, p)
        errors += int(share < 0.999)
        if share < 0.999:
            detail[f"independent:{p}"] = share
    from crucible.commerce import objective
    chk = objective.check(w, org, min_customers=1, max_customer_share=1.0)
    n_counted = int(chk["independent_customers"])
    errors += abs(n_counted - len(independent))
    st = w.state
    payer_accounts = set()
    for inv in st.tables["sub_invoices"].values():
        if inv["merchant"] == org and inv["customer"] in independent and inv["status"] == "paid":
            payer_accounts.add(inv["account"])
            tx = st.tables["txns"][inv["txn"]]
            assert dict(tx["postings"]).get(inv["account"]) == -inv["amount"]
    rep = revenue.revenue_report(w, org)
    record("K.4", {"related_party_exclusion_errors": int(errors), "independent_customers_counted": n_counted,
                   "independent_customers_true": len(independent),
                   "planted_related_customers": sum(len(v) for v in planted.values()),
                   "planted_scheme_types": len(planted), "individual_payment_accounts": len(payer_accounts),
                   "related_party_cents_excluded": int(rep["related_party_cents"])},
           notes="15 independent customers paying from their own accounts; planted (share of list MRR counted for "
                 f"the objective, must be <= 0.2): {detail}")
    assert errors == 0 and n_counted >= 10 and len(payer_accounts) == len(independent), detail
    assert not healthy(w)


# ============================================================================ K.5 MRR / ARR
def _recompute_mrr(w, merchant: str, independent: set[str]) -> Fraction:
    """Independent recomputation of qualifying MRR from the event log and the ledger (not from the subs tables or
    the revenue module): live, active subscriptions not set to cancel whose current period is paid, weighted by
    the share of charges kept (refund events and credit voids against invoice events), over the current period,
    the trailing 180 days and the completed periods of that window."""
    now = w.now
    status, cancel_pe, invoices, refunded, voided = {}, {}, {}, {}, {}
    for e in w.log:
        if e.type == "subs.subscription":
            status[e.payload["id"]] = "trialing" if e.payload["trial_end"] is not None else "incomplete"
            cancel_pe[e.payload["id"]] = False
        elif e.type == "subs.status":
            ch = e.payload["changes"]
            if "status" in ch:
                status[e.payload["id"]] = ch["status"]
            if "cancel_at_period_end" in ch:
                cancel_pe[e.payload["id"]] = ch["cancel_at_period_end"]
        elif e.type == "subs.invoice" and e.payload["merchant"] == merchant and e.payload["status"] == "paid":
            p = dict(e.payload, t=e.t)
            if p["amount"] > 0:
                tx = w.state.tables["txns"][p["txn"]]
                assert sum(d for _, d in tx["postings"] if d > 0) == p["amount"], "invoice backed by the ledger"
            invoices.setdefault(p["subscription"], []).append(p)
        elif e.type == "subs.refund":
            tx = w.state.tables["txns"][e.payload["txn"]]
            assert tx["kind"] == "refund" and tx["ref"] == e.payload["invoice"], "refund backed by the ledger"
            refunded[e.payload["invoice"]] = refunded.get(e.payload["invoice"], 0) + e.payload["amount"]
        elif e.type == "subs.credit_void":
            voided[e.payload["invoice"]] = voided.get(e.payload["invoice"], 0) + e.payload["cents"]
    total = Fraction(0)
    t0 = now - 180 * DAY
    for sid, invs in invoices.items():
        if status.get(sid) != "active" or cancel_pe.get(sid) or invs[0]["customer"] not in independent:
            continue
        cur = [i for i in invs if i["period_start"] <= now < i["period_end"]]
        if not cur:
            continue

        def kept(rows):
            g = sum(i["amount"] + i["credit_applied"] for i in rows)
            n = sum(max(0, i["amount"] - refunded.get(i["id"], 0) + i["credit_applied"] - voided.get(i["id"], 0))
                    for i in rows)
            return Fraction(min(n, g), g) if g else Fraction(1)
        window = [i for i in invs if i["t"] > t0 or i["period_end"] > t0]
        done = [i for i in window if i["period_end"] <= now]
        f = min(kept(cur), kept(window), kept(done))
        if f <= 0:
            continue
        last = max(cur, key=lambda i: i["t"])
        total += Fraction(last["list_price"] - last["discount"], 12 if last["interval"] == "year" else 1) * f
    return total


def test_K5_mrr_and_arr():
    w = make_world(51)
    f, org, prod, plans, _ = media_business(w, "acme", capital=2_000_000, monthly=2_900, annual=29_000, trial_days=7)
    cp = f.do("subs.create_coupon", plan=plans["month"], percent_off=20, periods=3, max_redemptions=50)["coupon"]
    independent: set[str] = set()
    cust = []
    for i in range(24):
        c = join(w, f"k5c{i}", 400_000 if i % 7 else 3_000)
        independent.add(c.id)
        plan = plans["year"] if i % 3 == 0 else plans["month"]
        args = {"plan": plan, **({"coupon": cp} if i % 5 == 1 and plan == plans["month"] else {})}
        cust.append((c, c.do("subs.subscribe", **args)["subscription"]))
    related = [p for p, _ in integrity.SCHEMES["sister_org"](w, "k5sis", f.id, org, plans["month"], 2_900)]
    related += [p for p, _ in integrity.SCHEMES["member"](w, "k5emp", f.id, org, plans["month"], 2_900)]
    checkpoints = [1, 5, 8, 15, 20, 31, 38, 45, 61, 75, 91, 120, 200]
    mismatches, arr_ok, rows = 0, True, []
    for day in checkpoints:
        w.advance_time(day * DAY + 1800)
        if day == 20:      # cancellations, refunds (full and partial) mid-run
            cust[1][0].do("subs.cancel", subscription=cust[1][1], at_period_end=True)
            cust[2][0].do("subs.cancel", subscription=cust[2][1], at_period_end=False)
            for k, amount in ((4, None), (7, 700), (8, 1_000)):
                inv = [i for i in cust[k][0].do("subs.invoices", subscription=cust[k][1]) if i["status"] == "paid"]
                if inv:
                    args = {"invoice": inv[-1]["id"], "reason": "refund"}
                    if amount is not None and inv[-1]["amount"] > amount:
                        args["amount_cents"] = amount
                    f.do("subs.refund", **args)
        mrr = revenue.mrr_exact(w, org)
        ref = _recompute_mrr(w, org, independent)
        arr = Fraction(revenue.m_arr(w, org))
        mismatches += int(mrr != ref)
        arr_ok = arr_ok and revenue.m_arr(w, org) == float(12 * mrr)
        rows.append(f"d{day}:{float(mrr):.2f}/{float(ref):.2f}")
    # normalisation: $100,000 of annual plans is $8,333.33 of monthly recurring revenue
    w2 = make_world(52)
    f2, org2, _, plans2, _ = media_business(w2, "norm", monthly=None, annual=1_000_000)
    for i in range(10):
        join(w2, f"n{i}", 1_100_000).do("subs.subscribe", plan=plans2["year"])
    norm_ok = revenue.m_arr(w2, org2) == 10_000_000.0 and round(revenue.m_mrr(w2, org2) / 100, 2) == 8_333.33
    from crucible.commerce import objective
    target_ok = round(objective.check(w2, org2, min_customers=1, max_customer_share=1.0)["target_mrr_usd"], 2) == 8_333.33
    record("K.5", {"mrr_arr_recompute_mismatches": mismatches, "arr_equals_12_mrr": bool(arr_ok),
                   "checkpoints": len(checkpoints), "normalisation_100k_arr_is_8333_33_mrr": bool(norm_ok and target_ok),
                   "final_mrr_cents": float(revenue.mrr_exact(w, org))},
           notes="metric vs independent recomputation from the event log and ledger (net of partial refunds, "
                 "scheduled churn excluded): " + " ".join(rows))
    assert mismatches == 0 and arr_ok and norm_ok and target_ok
    assert not healthy(w) and not failures(w)


# ============================================================================ K.6 profit and costs
def test_K6_profit_and_costs():
    w = make_world(61, providers=True)
    f, org, prod, plans, _ = media_business(w, "omega", capital=1_000_000, monthly=3_000)
    acct = org_account(w, org)
    off = f.do("prod.offer", product=prod, kind="one_time", price_cents=5_000)["offer"]
    subs = []
    for i in range(6):
        c = join(w, f"k6c{i}", 50_000)
        subs.append((c, c.do("subs.subscribe", plan=plans["month"])["subscription"]))
    join(w, "k6buyer", 20_000).do("prod.buy", offer=off)
    inv = [i for i in subs[0][0].do("subs.invoices", subscription=subs[0][1]) if i["status"] == "paid"][0]
    f.do("subs.refund", invoice=inv["id"], amount_cents=1_000, reason="partial refund")
    # related revenue (excluded): the founder's other company subscribes
    integrity.SCHEMES["sister_org"](w, "k6sis", f.id, org, plans["month"], 3_000)
    # infrastructure: one rented core, billed hourly
    prov = sorted(p["provider"] for p in f.do("cloud.providers"))[0]
    inst = f.do("cloud.rent", provider=prov, cores=1, account=acct, owner=org)["instance"]
    t_rent = w.now
    # operating costs: a contractor paid under a contract, a marketplace listing fee, a supplier subscription
    k = join(w, "k6contractor", 1_000)
    cid = f.do("contracts.propose", parties=[org, k.id], title="Design work", as_org=org,
               clauses=[{"type": "payment", "from": org, "to": k.id, "amount": 4_000, "every_s": None,
                         "count": None, "start_in_s": 0}])["contract"]
    k.do("contracts.sign", contract=cid)
    f.do("market.list", kind="service", title="Consulting", price_cents=10_000, seller=org)
    sf, sorg, sprod, splans, _ = media_business(w, "supplier", monthly=1_500)
    f.do("subs.subscribe", plan=splans["month"], account=acct)
    # owner capital moves (non-operating): money paid out to the founder
    f.do("bank.transfer", from_account=acct, to=f.id, amount=2_000)
    w.advance_time(w.now + 10 * DAY)
    price = w.state.tables["instances"][inst]["price"]
    hours = 1 + (w.now // 3600 - t_rent // 3600)
    expected_infra = price * hours
    expected_revenue = 6 * 3_000 + 5_000 - 1_000
    expected_operating = 4_000 + 500 + 1_500
    expected = expected_revenue - expected_operating - expected_infra
    got = revenue.profit(w, org, 30 * DAY)
    err = abs(int(got["profit"]) - expected)
    # cash reconciliation: the change in the company's cash is profit plus the excluded and non-operating flows
    lw = got["ledger"]
    cash_delta = sum(lw["by_kind"].values())
    recon = cash_delta - (got["profit"] + got["excluded_revenue"] + lw["non_operating_in"] - lw["non_operating_out"])
    record("K.6", {"profit_reconciliation_error_cents": int(err), "profit_cents": int(got["profit"]),
                   "expected_profit_cents": int(expected), "revenue_cents": int(got["revenue"]),
                   "operating_costs_cents": int(got["operating_costs"]),
                   "infrastructure_costs_cents": int(got["infrastructure_costs"]),
                   "excluded_related_cents": int(got["excluded_revenue"]), "cash_reconciliation_error_cents": int(recon)},
           notes=f"expected: revenue {expected_revenue}, operating {expected_operating}, infrastructure "
                 f"{expected_infra} ({hours} core-hours at {price})")
    assert err == 0 and recon == 0, (got, expected)
    assert not healthy(w)


# ============================================================================ K.7 circular / related detection
def _ordinary_commerce(w, f, org, plans, price):
    """Independent customers doing ordinary business with the merchant (none of this is circular)."""
    ordinary = []
    # customers are admitted personas with their own controller labels (an unlabelled admitted persona is never an
    # independent customer for an objective: fail closed, see commerce.integrity.independent_customer)
    for i in range(30):
        c = join(w, f"ord{i}", 30 * price + 1_000 * i, controller=f"ord-{i}")
        c.do("subs.subscribe", plan=plans["year" if i % 5 == 0 else "month"])
        ordinary.append(c.id)
    for i in range(3):                       # refunded once, then resubscribed
        c = join(w, f"refd{i}", 30 * price, controller=f"refd-{i}")
        sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
        inv = [x for x in c.do("subs.invoices", subscription=sid) if x["status"] == "paid"][0]
        f.do("subs.refund", invoice=inv["id"], reason="complaint")
        c.do("subs.cancel", subscription=sid, at_period_end=False)
        c.do("subs.subscribe", plan=plans["month"])
        ordinary.append(c.id)
    for i in range(3):                       # independent businesses (B2B), paying from company accounts
        b = join(w, f"b2b{i}-f", 400_000, controller=f"b2b-{i}")
        borg = b.do("venture.form", handle=f"b2b{i}", name=f"B2B {i}", capital_cents=200_000)
        b.do("subs.subscribe", plan=plans["month"], account=borg["account"])
        ordinary.append(borg["org"])
    for i in range(2):                       # suppliers the merchant buys real goods from, at their market price
        s = join(w, f"supp{i}", 100 * price, controller=f"supp-{i}")
        mint(w, s.id, f"part{i}", 20)
        other = join(w, f"partbuyer{i}", 10 * price, controller=f"partbuyer-{i}")
        lst = s.do("market.list", kind="goods", title=f"Part {i}", price_cents=price, sku=f"part{i}")["listing"]
        o = other.do("market.buy", listing=lst, account=other.acct, qty=2)["order"]
        s.do("market.deliver", order=o)
        other.do("market.confirm", order=o)
        o = f.do("market.buy", listing=lst, account=org_account(w, org), qty=3)["order"]
        s.do("market.deliver", order=o)
        f.do("market.confirm", order=o)
        s.do("subs.subscribe", plan=plans["month"])
        ordinary.append(s.id)
    rf, rorg, rprod, rplans, _ = media_business(w, "rival", monthly=price)
    for i in range(2):                       # customers of a competitor too
        c = join(w, f"both{i}", 40 * price, controller=f"both-{i}")
        c.do("subs.subscribe", plan=rplans["month"])
        c.do("subs.subscribe", plan=plans["month"])
        ordinary.append(c.id)
    return ordinary


def test_K7_circular_and_related_detection():
    price = 2_500
    w = make_world(71, extra=integrity.EXTRA_MODULES)
    f, org, prod, plans, _ = media_business(w, "merch", capital=8_000_000, monthly=price, annual=10 * price,
                                            controller="agent-merch")
    ordinary = _ordinary_commerce(w, f, org, plans, price)
    planted: dict[str, list[str]] = {}
    for name, fn in integrity.SCHEMES.items():
        planted[name] = [p for p, _ in fn(w, f"k7{name.replace('_', '')[:8]}", f.id, org, plans["month"], price)]
    w.advance_time(w.now + DAY)
    planted_all = {p for ps in planted.values() for p in ps}
    shares = {p: _objective_share(w, org, p) for p in planted_all | set(ordinary)}
    limit = {p: integrity.SCHEME_MAX_WEIGHT.get(name, 0.2) for name, ps in planted.items() for p in ps}
    detected = {p for p in planted_all if shares[p] <= limit[p] + 1e-9}
    fp = {p for p in ordinary if shares[p] < 0.999}
    missed = {name: [(p, round(shares[p], 3)) for p in ps if p not in detected] for name, ps in planted.items()}
    # the 4.0 measure over the last 30 days, for comparison: how much it counts as independent revenue
    from crucible.goals.metrics import revenue_analysis
    ra = revenue_analysis(w, org, 30 * DAY)
    rf30 = integrity.related_flows(w, org, window_s=30 * DAY)
    planted_cents = sum(r["paid"] for p, r in rf30["payers"].items() if p in planted_all)
    ordinary_cents = sum(r["paid"] for p, r in rf30["payers"].items() if p in set(ordinary))
    frac = len(detected) / len(planted_all)
    fpr = len(fp) / len(ordinary)
    record("K.7", {"planted_circular_detected_fraction": frac, "false_positive_rate": fpr,
                   "planted_payers": len(planted_all), "planted_scheme_types": len(planted),
                   "ordinary_customers": len(ordinary), "false_positives": len(fp),
                   "planted_cents_30d": int(planted_cents), "ordinary_cents_30d": int(ordinary_cents),
                   "f1_independent_cents_30d": int(rf30["independent_cents"]),
                   "legacy_4_0_independent_cents_30d": int(ra["independent"])},
           notes=f"detection = the payer's share of list MRR counted for the objective is at most its legitimately "
                 f"independent share (0.2 for schemes that send the subscription money back, 0.56 for a 45% "
                 f"cashback). Schemes include the 5.0 audit's probes (escrowed commitment payouts, project "
                 f"rewards, goods above market price, rebates, an unlabelled persona, a 6-hop grant chain). missed: "
                 f"{missed}; false positives: {sorted(fp)}. Over the last 30 days the 4.0 revenue_analysis counts "
                 f"{ra['independent']} cents as independent, F1 counts {rf30['independent_cents']}.")
    assert frac == 1.0 and fpr <= 0.05, (missed, fp)
    assert not healthy(w)


# ============================================================================ K.n1 reference strategy
@pytest.mark.slow
@pytest.mark.timeout(5400)
def test_Kn1_reference_strategy_reaches_feasible_arr_objective():
    """LABELLED SCRIPTED. A scripted founder (commerce.reference) in generated startup worlds whose demand is the
    one population: it reaches a sustained, concentration-capped $100k ARR from >= 100 independent individuals;
    doing nothing, or selling an input-ignoring service, does not."""
    from crucible.commerce import reference
    target = 100_000.0
    kw = dict(days=75, n=1500, sustain_days=14)
    runs = [reference.run(seed, **kw) for seed in (1, 4)]
    idle = reference.run(1, strategy="idle", **kw)
    useless = reference.run(1, program="garbage", **kw)
    for r in runs + [idle, useless]:
        w = r.pop("world")
        r["healthy"] = not healthy(w) and not failures(w)
        r["demand_from_population_only"] = all(
            (s["customer"].startswith("p") and s["customer"][1:].isdigit())
            for _, s in w.state.scan("sub_subs") if s["merchant"] == r["org"])
        r.pop("agent")
    met = all(r["sustained"] for r in runs)
    arr = min(r["final"]["arr_usd"] for r in runs)
    record("K.n1", {"reference_objective_met": bool(met), "reference_arr_usd": float(arr),
                    "seeds": len(runs), "idle_baseline_arr_usd": float(idle["final"]["arr_usd"]),
                    "idle_baseline_objective_met": bool(idle["sustained"]),
                    "input_ignoring_service_arr_usd": float(useless["final"]["arr_usd"]),
                    "input_ignoring_service_objective_met": bool(useless["sustained"]),
                    "first_day_met_max": max(r["first_day_met"] or 10**6 for r in runs),
                    "independent_customers_min": min(r["final"]["independent_customers"] for r in runs),
                    "largest_customer_share_max": max(r["final"]["largest_customer_share"] for r in runs),
                    "demand_from_population_only": all(r["demand_from_population_only"] for r in runs),
                    "scripted": True},
           command="crucible.commerce.reference.run(seed, days=75, n=1500)",
           labels=dict(reference.LABELS),
           notes="LABELLED SCRIPTED reference strategy (not a learner, not evidence of learning). Generated startup "
                 "worlds (seeds 1, 4): demand only from the population's work activities (individuals delegate "
                 "their own tasks, judge by downstream acceptance and their time), 1-3 adaptive incumbents per "
                 "format. Objective: qualifying MRR >= $8,333.33 from >= 100 independent customers with no customer "
                 "above 5%, at each of the last 14 daily checks. Per-seed ARR: "
                 + ", ".join(f"{r['seed']}:{r['final']['arr_usd']:.0f}" for r in runs))
    assert met and arr >= target and not idle["sustained"] and not useless["sustained"]
    assert useless["final"]["arr_usd"] < 0.1 * target
    assert all(r["healthy"] for r in runs) and not any(r["agent_failures"] for r in runs)
    assert all(r["demand_from_population_only"] for r in runs)


# ============================================================================ K.n2 reference is labelled scripted
def test_Kn2_reference_results_labelled_scripted():
    """K.n2 (workstream N): results of the reference strategy carry a machine-readable label (scripted, not a
    learner, not evidence of learning) at their source; the K.n1 evidence record carries the same label; and no
    document or evidence note presents the reference as evidence that a learner learned the objective
    (`crucible.research.v5.integrity.scan_claims("reference_learning")`, with planted controls)."""
    import ast
    from crucible.commerce import reference
    from crucible.research.v5 import integrity as G
    r = reference.run(2, days=2, n=300, sustain_days=1)
    r.pop("world", None)
    result_labelled = r.get("labels") == reference.LABELS and reference.LABELS["scripted"] is True and \
        reference.LABELS["learner"] is False
    tree = ast.parse(open(os.path.abspath(__file__)).read())
    rec_labelled = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "record" and n.args and \
                isinstance(n.args[0], ast.Constant) and n.args[0].value == "K.n1":
            rec_labelled = any(k.arg == "labels" for k in n.keywords)
    scan = G.scan_claims("reference_learning", G.corpus())
    ok = bool(result_labelled and rec_labelled and scan["unsupported_claims"] == 0 and scan["controls"]["ok"])
    record("K.n2", {"reference_results_labelled_scripted": ok, "result_labels_present": bool(result_labelled),
                    "k_n1_record_labelled": bool(rec_labelled), "reference_as_learning_claims": scan["unsupported_claims"],
                    "files_scanned": scan["files_scanned"]},
           labels=dict(reference.LABELS), notes=f"Result labels {r.get('labels')}; claims found {scan['claims'][:5]}")
    assert ok
