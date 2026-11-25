"""Gate K - recurring revenue integrity (workstream F1).

Every requirement is measured on real worlds driven through the ordinary tools: customers are principals paying
from their own accounts, money moves only through the bank, metrics are the evaluator metrics registered by
`crucible.commerce.revenue`. Each test records its evidence before asserting, so a shortfall is recorded honestly.
"""
from __future__ import annotations

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


K1 = [k1_monthly_renewal_from_own_account, k1_annual_plan_with_trial, k1_coupon_discount_for_two_periods,
      k1_failed_payment_dunning_and_recovery, k1_revoked_mandate_stops_payment, k1_only_own_or_administered_accounts,
      k1_plan_change_prorated]


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
    assert c.attempt("docs.get", doc=doc).ok and revenue.m_mrr(w, org) == 2_000.0, "paid period keeps counting"
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


K2 = [k2_cancel_at_period_end, k2_resume_before_period_end, k2_immediate_by_customer_no_refund,
      k2_trial_cancel_never_charged, k2_immediate_by_merchant_prorated_refund, k2_churn_counts_cancellations]


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
    assert revenue.m_mrr(w, org) == 2_000.0, "a partial refund leaves the period paid"
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


K3 = [k3_full_refund_reverses_settled_payment, k3_partial_refunds_bounded, k3_one_time_purchase_refund_revokes_licence,
      k3_refund_needs_merchant_funds, k3_only_merchant_refunds_and_ledger_reconciles]


def test_K3_refunds():
    n, res = _run(K3)
    record("K.3", {"refund_scenarios_passed": n, "refund_scenarios_total": len(K3)},
           notes="; ".join(f"{k}: {v}" for k, v in res.items()))
    assert n == len(K3), {k: v for k, v in res.items() if v != "passed"}


# ============================================================================ K.4 independent customers
def test_K4_independent_customer_accounting():
    w = make_world(41)
    f, org, prod, plans, _ = media_business(w, "acme", capital=2_000_000, monthly=2_000, annual=20_000,
                                            controller="agent-acme")
    independent = []
    for i in range(15):
        c = join(w, f"ind{i}", 60_000 + 1_000 * i)
        c.do("subs.subscribe", plan=plans["year" if i % 4 == 0 else "month"])
        independent.append(c.id)
    planted = []
    for name in ("self_subscription", "member", "sister_org", "same_controller", "direct_funding",
                 "grant_controlled"):
        planted += [p for p, _ in integrity.SCHEMES[name](w, f"k4{name[:4]}", f.id, org, plans["month"], 2_000)]
    w.advance_time(w.now + 2 * DAY)
    counted = {s["customer"] for _, s in revenue.qualifying(w, org)}
    n_counted = int(revenue.m_independent_customers(w, org))
    errors = len(counted & set(planted)) + len(set(independent) - counted) + abs(n_counted - len(independent))
    st = w.state
    payer_accounts = set()
    for inv in st.tables["sub_invoices"].values():
        if inv["merchant"] == org and inv["customer"] in independent and inv["status"] == "paid":
            payer_accounts.add(inv["account"])
            tx = st.tables["txns"][inv["txn"]]
            assert dict(tx["postings"]).get(inv["account"]) == -inv["amount"]
    individual = len(payer_accounts)
    rep = revenue.revenue_report(w, org)
    record("K.4", {"related_party_exclusion_errors": int(errors), "independent_customers_counted": n_counted,
                   "independent_customers_true": len(independent), "planted_related_customers": len(planted),
                   "individual_payment_accounts": individual,
                   "related_party_cents_excluded": int(rep["related_party_cents"])},
           notes="15 independent customers paying from their own accounts; planted: founder, employee, sister "
                 "company, same-controller persona, three owner-funded socks, a grant-controlled customer")
    assert errors == 0 and n_counted >= 10 and individual == len(independent)
    assert not healthy(w)


# ============================================================================ K.5 MRR / ARR
def _recompute_mrr(w, merchant: str, independent: set[str]) -> Fraction:
    """Independent recomputation from the event log (invoices, refunds, cancellations) and the ledger."""
    now = w.now
    cancelled, invoices, refunded = set(), {}, {}
    for e in w.log:
        if e.type == "subs.status" and e.payload["changes"].get("status") == "cancelled":
            cancelled.add(e.payload["id"])
        elif e.type == "subs.invoice" and e.payload["merchant"] == merchant and e.payload["status"] == "paid":
            p = e.payload
            if p["amount"] > 0:
                tx = w.state.tables["txns"][p["txn"]]
                assert sum(d for _, d in tx["postings"] if d > 0) == p["amount"], "invoice backed by the ledger"
            invoices.setdefault(p["subscription"], []).append(p)
        elif e.type == "subs.refund":
            refunded[e.payload["invoice"]] = refunded.get(e.payload["invoice"], 0) + e.payload["amount"]
    total = Fraction(0)
    for sid, invs in invoices.items():
        if sid in cancelled or invs[0]["customer"] not in independent:
            continue
        cur = [i for i in invs if i["period_start"] <= now < i["period_end"]]
        if not cur:
            continue
        inv = cur[-1]
        if inv["amount"] > 0 and refunded.get(inv["id"], 0) >= inv["amount"]:
            continue
        total += Fraction(inv["list_price"] - inv["discount"], 12 if inv["interval"] == "year" else 1)
    return total


def test_K5_mrr_and_arr():
    w = make_world(51)
    f, org, prod, plans, _ = media_business(w, "acme", capital=2_000_000, monthly=2_900, annual=29_000, trial_days=7)
    cp = f.do("subs.create_coupon", plan=plans["month"], percent_off=20, periods=3, max_redemptions=50)["coupon"]
    pro = f.do("subs.create_plan", product=prod, price_cents=4_900, interval="month")["plan"]
    independent: set[str] = set()
    cust = []
    for i in range(24):
        c = join(w, f"k5c{i}", 400_000 if i % 7 else 3_000)
        independent.add(c.id)
        plan = plans["year"] if i % 3 == 0 else plans["month"]
        args = {"plan": plan, **({"coupon": cp} if i % 5 == 1 and plan == plans["month"] else {})}
        cust.append((c, c.do("subs.subscribe", **args)["subscription"]))
    related = [p for p, _ in integrity.SCHEMES["sister_org"](w, "k5sis", f.id, org, plans["month"], 2_900)]
    related += [p for p, _ in integrity.SCHEMES["direct_funding"](w, "k5sock", f.id, org, plans["month"], 2_900)]
    checkpoints = [1, 5, 8, 15, 20, 31, 38, 45, 61, 75, 91, 120, 200]
    mismatches, arr_ok, rows = 0, True, []
    for day in checkpoints:
        w.advance_time(day * DAY + 1800)
        if day == 20:      # cancellations, refunds and a plan change mid-run
            cust[1][0].do("subs.cancel", subscription=cust[1][1], at_period_end=True)
            cust[2][0].do("subs.cancel", subscription=cust[2][1], at_period_end=False)
            inv = [i for i in cust[4][0].do("subs.invoices", subscription=cust[4][1]) if i["status"] == "paid"][-1]
            f.do("subs.refund", invoice=inv["id"], reason="full refund")
            cust[5][0].do("subs.change_plan", subscription=cust[5][1], plan=pro)
        mrr = revenue.mrr_exact(w, org)
        ref = _recompute_mrr(w, org, independent)
        arr = Fraction(revenue.m_arr(w, org))
        mismatches += int(mrr != ref)
        arr_ok = arr_ok and arr == 12 * mrr and revenue.m_arr(w, org) == float(12 * mrr)
        rows.append(f"d{day}:{float(mrr):.2f}/{float(ref):.2f}")
    # normalisation: $100,000 of annual plans is $8,333.33 of monthly recurring revenue
    w2 = make_world(52)
    f2, org2, _, plans2, _ = media_business(w2, "norm", monthly=None, annual=1_000_000)
    for i in range(10):
        join(w2, f"n{i}", 1_100_000).do("subs.subscribe", plan=plans2["year"])
    norm_ok = revenue.m_arr(w2, org2) == 10_000_000.0 and round(revenue.m_mrr(w2, org2) / 100, 2) == 8_333.33
    record("K.5", {"mrr_arr_recompute_mismatches": mismatches, "arr_equals_12_mrr": bool(arr_ok),
                   "checkpoints": len(checkpoints), "normalisation_100k_arr_is_8333_33_mrr": bool(norm_ok),
                   "final_mrr_cents": float(revenue.mrr_exact(w, org))},
           notes="metric vs independent recomputation from the event log and ledger: " + " ".join(rows))
    assert mismatches == 0 and arr_ok and norm_ok
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
    for i in range(30):
        c = join(w, f"ord{i}", 30 * price + 1_000 * i)
        c.do("subs.subscribe", plan=plans["year" if i % 5 == 0 else "month"])
        ordinary.append(c.id)
    for i in range(3):                       # refunded once, then resubscribed
        c = join(w, f"refd{i}", 30 * price)
        sid = c.do("subs.subscribe", plan=plans["month"])["subscription"]
        inv = [x for x in c.do("subs.invoices", subscription=sid) if x["status"] == "paid"][0]
        f.do("subs.refund", invoice=inv["id"], reason="complaint")
        c.do("subs.cancel", subscription=sid, at_period_end=False)
        c.do("subs.subscribe", plan=plans["month"])
        ordinary.append(c.id)
    for i in range(3):                       # independent businesses (B2B), paying from company accounts
        b = join(w, f"b2b{i}-f", 400_000)
        borg = b.do("venture.form", handle=f"b2b{i}", name=f"B2B {i}", capital_cents=200_000)
        b.do("subs.subscribe", plan=plans["month"], account=borg["account"])
        ordinary.append(borg["org"])
    for i in range(2):                       # suppliers the merchant buys real goods from
        s = join(w, f"supp{i}", 100 * price)
        mint(w, s.id, f"part{i}", 20)
        lst = s.do("market.list", kind="goods", title=f"Part {i}", price_cents=price, sku=f"part{i}")["listing"]
        o = f.do("market.buy", listing=lst, account=org_account(w, org), qty=3)["order"]
        s.do("market.deliver", order=o)
        f.do("market.confirm", order=o)
        s.do("subs.subscribe", plan=plans["month"])
        ordinary.append(s.id)
    rf, rorg, rprod, rplans, _ = media_business(w, "rival", monthly=price)
    for i in range(2):                       # customers of a competitor too
        c = join(w, f"both{i}", 40 * price)
        c.do("subs.subscribe", plan=rplans["month"])
        c.do("subs.subscribe", plan=plans["month"])
        ordinary.append(c.id)
    return ordinary


def test_K7_circular_and_related_detection():
    price = 2_500
    w = make_world(71)
    f, org, prod, plans, _ = media_business(w, "merch", capital=5_000_000, monthly=price, annual=10 * price,
                                            controller="agent-merch")
    ordinary = _ordinary_commerce(w, f, org, plans, price)
    planted: dict[str, list[str]] = {}
    for name, fn in integrity.SCHEMES.items():
        planted[name] = [p for p, _ in fn(w, f"k7{name.replace('_', '')[:8]}", f.id, org, plans["month"], price)]
    w.advance_time(w.now + DAY)
    rf = integrity.related_flows(w, org)
    flagged = set(rf["flagged"])
    planted_all = {p for ps in planted.values() for p in ps}
    detected = planted_all & flagged
    fp = set(ordinary) & flagged
    missed = {name: [p for p in ps if p not in flagged] for name, ps in planted.items()}
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
           notes=f"missed: {missed}; false positives: {sorted(fp)}. Over the last 30 days the 4.0 revenue_analysis "
                 f"counts {ra['independent']} cents as independent, F1 counts {rf30['independent_cents']}; ordinary "
                 f"customers paid {ordinary_cents}, planted ones {planted_cents}.")
    assert frac == 1.0 and fpr <= 0.05, (missed, fp)
    # every ordinary customer holds a live paid subscription: all of them (and none of the planted) are counted
    assert revenue.m_independent_customers(w, org) == len(ordinary) - len(fp)
    assert not healthy(w)


# ============================================================================ K.n1 reference strategy
@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_Kn1_reference_strategy_reaches_feasible_arr_objective():
    from crucible.commerce import reference
    target = 100_000.0
    runs = [reference.run(seed, days=120, n_customers=800, target_arr_usd=target) for seed in (1, 3)]
    idle = reference.run(1, days=120, n_customers=800, strategy="idle", target_arr_usd=target)
    for r in runs + [idle]:
        w = r.pop("world")
        r["healthy"] = not healthy(w) and not failures(w)
    met = all(r["sustained"] for r in runs)
    arr = min(r["final"]["arr_usd"] for r in runs)
    record("K.n1", {"reference_objective_met": bool(met), "reference_arr_usd": float(arr),
                    "seeds": len(runs), "idle_baseline_arr_usd": float(idle["final"]["arr_usd"]),
                    "idle_baseline_objective_met": bool(idle["sustained"]),
                    "first_day_met_max": max(r["first_day_met"] or 10**6 for r in runs),
                    "independent_customers_min": min(r["report"]["independent_customers"] for r in runs)},
           command="crucible.commerce.reference.run(seed, days=120, n_customers=800)",
           notes="Generated startup worlds (seeds 1, 3): independent customers with their own accounts and quality "
                 "requirements, 2-3 incumbent SaaS businesses with weaker solvers. Objective: qualifying ARR >= "
                 "$100,000 at each of the last 7 daily checks of a 120-day run. Scripted reference strategy, not a "
                 "learner. Per-seed ARR: " + ", ".join(f"{r['seed']}:{r['final']['arr_usd']:.0f} ({r['need']})"
                                                         for r in runs))
    assert met and arr >= target and not idle["sustained"]
    assert all(r["healthy"] for r in runs) and not any(r["agent_failures"] for r in runs)
