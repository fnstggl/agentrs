"""Recurring-revenue and profit accounting (evaluator metrics, registered with `goals.metrics.metric`).

Definitions (all integer cents unless noted; `owner` is the merchant principal, usually a company):

* **Qualifying subscription**: a subscription to one of the owner's plans that is `active`, whose current period
  is paid and has not ended, that is **not set to cancel at period end** (that is scheduled churn, reported
  separately: it will not recur), and whose customer is not a related party. Trials and past-due subscriptions
  never qualify.
* **Weight** of a qualifying subscription (exact fraction in [0, 1]) = kept factor x payer weight:
  * kept factor (`subs.kept_factor`): the share of what the subscription was charged that the merchant kept -
    over the current period's invoices (credit from refunded invoices is void), over the trailing 180 days, and over
    the completed periods in that window (the minimum of the three). Partial refunds net MRR;
  * payer weight (`integrity.payer_weights`): 1 - the payer's circular share (money traced from or netted back to
    the owner's side, over all history; rebates, escrowed instruments and goods at above-market prices included).
    Exclusion is proportional: there is no threshold below which a rebate is free.
* **MRR** = sum over qualifying subscriptions of weight x the recurring charge of the current period (list price
  minus a coupon's recurring discount; one-off proration credits excluded), normalised to one month by the plan's
  interval: monthly x 1, annual x 1/12. Computed exactly as a fraction; `mrr_cents` returns it as a float.
* **ARR** = 12 x MRR (exactly).
* Rolling sales are not recurring revenue: one-time sales, licences, marketplace fees and refunds are reported
  separately and never enter MRR.
* **Churn rate** over a window (default 30 days): of the independent customers that had a qualifying
  subscription at the start of the window, the share that has no live (trialing, active or past-due)
  subscription to the owner now. **Scheduled churn**: live subscriptions set to cancel at period end, reported
  as their count, their MRR and their share of customers (`scheduled_churn`).
* **Independent customers**: distinct economic units (an organisation with its owners, as in 4.0 `_Groups`)
  holding at least one qualifying subscription.
* **Evaluator metrics are not readable in-world.** `mrr_cents`, `arr_cents`, `active_subscriptions`,
  `independent_customers`, `churn_rate`, `profit_cents` and `one_time_revenue_cents` depend on the hidden
  related-party and circular-flow analysis, so they are registered without a subject (contract deliverables,
  commitment predicates and charter conditions cannot read them; `EVALUATOR_ONLY` lists them). What a merchant can
  condition on in-world are its **booked** figures - its own subscription and ledger records with no integrity
  analysis (`booked_mrr_cents`, `booked_subscribers`, `booked_revenue_cents`), the same numbers its tools show it.
* **Profit** over a window (cash basis): independent revenue (commerce payments from unflagged payers, net of
  refunds, less circular cents and financing; see `integrity.related_flows`) minus operating costs minus
  infrastructure costs. Infrastructure: compute, hosting, operating expense and hardware/machines/facilities
  (capital expenditure counts when paid). Operating: every other outflow except financing and owner capital
  (transfers, distributions, loans and their repayment, equity, endowments). Refunds received from suppliers
  reduce operating costs.
"""
from __future__ import annotations

from fractions import Fraction

from ..goals.metrics import _Groups, metric
from .integrity import COMMERCE_KINDS, flagged_payers, payer_weights, related_flows

DAY = 86400
INFRA_KINDS = frozenset({"compute", "hosting", "opex", "energy", "hardware", "machine", "facility"})
ACQUISITION_KINDS = frozenset({"ads", "ad_impressions", "marketing"})
# metrics computed with the hidden integrity analysis: never readable through in-world predicates
EVALUATOR_ONLY = frozenset({"mrr_cents", "arr_cents", "active_subscriptions", "independent_customers", "churn_rate",
                            "profit_cents", "one_time_revenue_cents", "qualifying_mrr_cents", "scheduled_churn_share"})
NON_OPERATING = frozenset({"endowment", "transfer", "liquidation", "distribution", "income", "loan",
                           "loan_repayment", "equity", "capital", "dividend", "investment", "savings",
                           "escrow_release"})
RECOVERY_IN = frozenset({"refund"})


def _subs(world):
    return world.modules.get("subs")


def qualifying_weighted(world, owner: str) -> list[tuple[str, dict, Fraction]]:
    """Qualifying subscriptions of `owner` with their weights (see module doc), in id order; weight > 0 only."""
    st = world.state
    sm = _subs(world)
    if sm is None or "sub_subs" not in st.tables:
        return []
    weights, rel = payer_weights(world, owner)
    out = []
    for sid in st.find_ids("sub_subs", "merchant", owner):
        s = st.tables["sub_subs"][sid]
        if not sm.recurring_valid(s, world.now) or s["cancel_at_period_end"] or s["customer"] in rel:
            continue
        w = sm.kept_factor(st, sid, world.now) * weights.get(s["customer"], Fraction(1))
        if w > 0:
            out.append((sid, s, w))
    return out


def qualifying(world, owner: str) -> list[tuple[str, dict]]:
    """Qualifying subscriptions of `owner` (weight > 0), in id order."""
    return [(sid, s) for sid, s, _ in qualifying_weighted(world, owner)]


def mrr_exact(world, owner: str) -> Fraction:
    sm = _subs(world)
    return sum((sm.monthly_recurring(s) * w for _, s, w in qualifying_weighted(world, owner)), Fraction(0))


def mrr_by_customer(world, owner: str) -> dict[str, Fraction]:
    """Qualifying MRR per independent economic unit (customer group)."""
    sm = _subs(world)
    groups = _Groups(world.state)
    out: dict[str, Fraction] = {}
    for _, s, w in qualifying_weighted(world, owner):
        g = groups.of(s["customer"])
        out[g] = out.get(g, Fraction(0)) + sm.monthly_recurring(s) * w
    return out


def scheduled_churn(world, owner: str) -> dict:
    """Live, paid-up subscriptions that are set to cancel at period end (they will not recur)."""
    st = world.state
    sm = _subs(world)
    if sm is None or "sub_subs" not in st.tables:
        return {"subscriptions": 0, "mrr_cents": 0.0, "share": 0.0}
    _, rel = payer_weights(world, owner)
    n = live = 0
    mrr = Fraction(0)
    for sid in st.find_ids("sub_subs", "merchant", owner):
        s = st.tables["sub_subs"][sid]
        if s["customer"] in rel or s["status"] not in ("trialing", "active", "past_due"):
            continue
        live += 1
        if s["cancel_at_period_end"]:
            n += 1
            if sm.recurring_valid(s, world.now):
                mrr += sm.monthly_recurring(s) * sm.kept_factor(st, sid, world.now)
    return {"subscriptions": n, "mrr_cents": float(mrr), "share": n / live if live else 0.0}


def booked(world, owner: str, window_s: int = 30 * DAY) -> dict:
    """The merchant's own books (what its own tools show it, no integrity analysis): live subscriptions, their
    normalised recurring amount and commerce receipts net of refunds over a window."""
    st = world.state
    sm = _subs(world)
    subs_n = 0
    mrr = Fraction(0)
    if sm is not None and "sub_subs" in st.tables:
        for sid in st.find_ids("sub_subs", "merchant", owner):
            s = st.tables["sub_subs"][sid]
            if s["status"] in ("trialing", "active", "past_due"):
                subs_n += 1
                if s["status"] == "active":
                    mrr += sm.monthly_recurring(s)
    lw = ledger_window(world, owner, window_s)
    return {"subscribers": subs_n, "mrr_cents": float(mrr), "revenue_cents": lw["commerce_in"] - lw["refunds_out"]}


def _valid_at(st, sid: str, t: int) -> bool:
    for inv in st.find("sub_invoices", "subscription", sid):
        if inv["status"] == "paid" and inv["period_start"] is not None and inv["period_start"] <= t < inv["period_end"] \
                and (inv["amount"] == 0 or inv["refunded"] < inv["amount"]) and inv["t"] <= t:
            return True
    return False


def churn(world, owner: str, window_s: int = 30 * DAY) -> dict:
    st = world.state
    if "sub_subs" not in st.tables:
        return {"start_customers": 0, "churned": 0, "rate": 0.0}
    _, excluded = payer_weights(world, owner)
    t0 = world.now - int(window_s)
    groups = _Groups(st)
    start: set[str] = set()
    live_now: set[str] = set()
    for sid in st.find_ids("sub_subs", "merchant", owner):
        s = st.tables["sub_subs"][sid]
        if s["customer"] in excluded:
            continue
        g = groups.of(s["customer"])
        if _valid_at(st, sid, t0):
            start.add(g)
        if s["status"] in ("trialing", "active", "past_due"):
            live_now.add(g)
    churned = start - live_now
    return {"start_customers": len(start), "churned": len(churned),
            "rate": len(churned) / len(start) if start else 0.0}


def ledger_window(world, owner: str, window_s: int) -> dict:
    """Owner's own ledger over the window by class: {"commerce_in", "refunds_out", "infra", "operating",
    "recovered", "non_operating_in", "non_operating_out", "by_kind": {kind: net}}."""
    st = world.state
    t0 = world.now - int(window_s)
    mine = set(st.find_ids("accounts", "owner", owner))
    out = {"commerce_in": 0, "refunds_out": 0, "infra": 0, "operating": 0, "recovered": 0, "non_operating_in": 0,
           "non_operating_out": 0, "acquisition": 0, "by_kind": {}}
    done: set[str] = set()
    for acct in sorted(mine):
        for lid in st.indexes[("acct_txns", "acct")].get(acct, ()):
            e = st.tables["acct_txns"][lid]
            if e["t"] <= t0 or e["txn"] in done:
                continue
            done.add(e["txn"])
            tx = st.tables["txns"][e["txn"]]
            net = sum(d for a, d in tx["postings"] if a in mine)
            if net == 0:
                continue
            k = tx["kind"]
            out["by_kind"][k] = out["by_kind"].get(k, 0) + net
            if net > 0:
                if k in COMMERCE_KINDS:
                    out["commerce_in"] += net
                elif k in RECOVERY_IN:
                    out["recovered"] += net
                else:
                    out["non_operating_in"] += net
            else:
                c = -net
                if k == "refund":
                    out["refunds_out"] += c
                elif k in INFRA_KINDS:
                    out["infra"] += c
                elif k in NON_OPERATING:
                    out["non_operating_out"] += c
                else:
                    out["operating"] += c
                    if k in ACQUISITION_KINDS:
                        out["acquisition"] += c
    return out


def profit(world, owner: str, window_s: int = 30 * DAY) -> dict:
    rf = related_flows(world, owner, window_s=window_s)
    lw = ledger_window(world, owner, window_s)
    revenue = int(rf["independent_cents"])
    operating = lw["operating"] - lw["recovered"]
    return {"revenue": revenue, "operating_costs": operating, "infrastructure_costs": lw["infra"],
            "profit": revenue - operating - lw["infra"], "excluded_revenue": rf["excluded_cents"],
            "financed": rf["financed_cents"], "refunds_paid": lw["refunds_out"], "ledger": lw}


def revenue_report(world, owner: str, window_s: int = 30 * DAY) -> dict:
    """Everything Part 6.2 asks to distinguish, for one merchant over one window. Payer weights and flags are
    all-history (the same analysis MRR uses), so the report and MRR never disagree about a payer."""
    st = world.state
    t0 = world.now - int(window_s)
    qw = qualifying_weighted(world, owner)
    mrr = mrr_exact(world, owner)
    weights, rel = payer_weights(world, owner)
    rf_all = related_flows(world, owner)
    flagged_all = sorted(set(rf_all["flagged"]) | rel)
    indep_by_kind: dict[str, Fraction] = {}
    mine = set(st.find_ids("accounts", "owner", owner))
    done: set[str] = set()
    for acct in sorted(mine):
        for lid in st.indexes[("acct_txns", "acct")].get(acct, ()):
            e = st.tables["acct_txns"][lid]
            if e["t"] <= t0 or e["txn"] in done:
                continue
            done.add(e["txn"])
            tx = st.tables["txns"][e["txn"]]
            if tx["kind"] not in COMMERCE_KINDS:
                continue
            payers = [st.tables["accounts"][a]["owner"] for a, d in tx["postings"] if d < 0]
            net = sum(d for a, d in tx["postings"] if a in mine)
            if net <= 0:
                continue
            w = min((Fraction(0) if p in rel else weights.get(p, Fraction(1))) for p in payers) if payers else 1
            indep_by_kind[tx["kind"]] = indep_by_kind.get(tx["kind"], Fraction(0)) + net * w
    ib = {k: int(v) for k, v in indep_by_kind.items()}
    one_time = ib.get("product_sale", 0) + ib.get("license", 0)
    subs_rev = ib.get("subscription", 0)
    fees = ib.get("take_rate", 0)
    other = sum(v for k, v in ib.items() if k not in ("product_sale", "license", "subscription", "take_rate"))
    discounts = refunds_subs = 0
    new_customers: set[str] = set()
    cancelled = {"voluntary": 0, "involuntary": 0, "other": 0}
    trialing = past_due = 0
    groups = _Groups(st)
    if "sub_subs" in st.tables:
        first_paid: dict[str, int] = {}
        for sid in st.find_ids("sub_subs", "merchant", owner):
            s = st.tables["sub_subs"][sid]
            if s["customer"] in rel:
                continue
            trialing += s["status"] == "trialing"
            past_due += s["status"] == "past_due"
            if s["status"] == "cancelled" and s["cancelled_at"] is not None and s["cancelled_at"] > t0:
                r = s["cancel_reason"] or ""
                cancelled["involuntary" if r == "nonpayment" else "voluntary" if r in ("customer", "trial_cancelled")
                          else "other"] += 1
            for inv in st.find("sub_invoices", "subscription", sid):
                if inv["status"] != "paid":
                    continue
                g = groups.of(s["customer"])
                first_paid[g] = min(first_paid.get(g, inv["t"]), inv["t"])
                if inv["t"] > t0:
                    discounts += int(inv["discount"])
                for r in inv["refunds"]:
                    if r["t"] > t0:
                        refunds_subs += int(r["amount"])
        new_customers = {g for g, t in first_paid.items() if t > t0}
    pr = profit(world, owner, window_s)
    ch = churn(world, owner, window_s)
    sch = scheduled_churn(world, owner)
    spend = pr["ledger"]["acquisition"]
    by_cust = mrr_by_customer(world, owner)
    top = max(by_cust.values()) if by_cust else Fraction(0)
    rf = related_flows(world, owner, window_s=window_s)
    return {"owner": owner, "window_s": int(window_s),
            "one_time_sales_cents": one_time, "subscription_revenue_cents": subs_rev,
            "marketplace_fee_revenue_cents": fees, "other_revenue_cents": other,
            "mrr_cents": float(mrr), "arr_cents": float(12 * mrr), "arr_cents_exact": str(12 * mrr),
            "active_subscriptions": len(qw), "independent_customers": len(by_cust),
            "largest_customer_share": float(top / mrr) if mrr else 0.0,
            "trialing_subscriptions": trialing, "past_due_subscriptions": past_due,
            "cancelled_subscriptions": cancelled, "scheduled_churn": sch, "refunds_cents": pr["refunds_paid"],
            "subscription_refunds_cents": refunds_subs, "discounts_cents": discounts,
            "new_customers": len(new_customers), "acquisition_spend_cents": spend,
            "cac_cents": round(spend / len(new_customers), 2) if new_customers else None,
            "churn_rate": ch["rate"], "churn": ch,
            "operating_costs_cents": pr["operating_costs"], "infrastructure_costs_cents": pr["infrastructure_costs"],
            "profit_cents": pr["profit"], "independent_revenue_cents": pr["revenue"],
            "related_party_cents": sum(r["paid"] for r in rf["payers"].values() if r["related"]),
            "circular_cents": sum(r["circular"] for r in rf["payers"].values() if not r["related"]),
            "financed_cents": rf["financed_cents"], "flagged_payers": flagged_all,
            "payer_weights": {p: float(w) for p, w in sorted(weights.items()) if w < 1}}


# ============================================================================ metrics
# Evaluator metrics: computed with the hidden integrity analysis, registered WITHOUT a subject, so no in-world
# predicate (contract deliverable, commitment, charter condition) can read them (goals.metrics.check_predicate_scope).
@metric("mrr_cents", "Evaluator-only. Qualifying normalised monthly recurring revenue of a merchant (cents): active, "
        "paid-up subscriptions not set to cancel, weighted by the share the merchant kept and by the share of each "
        "payer's money that is independent of the merchant; annual plans count one twelfth of their price.",
        instant=True)
def m_mrr(world, principal: str) -> float:
    return float(mrr_exact(world, principal))


@metric("arr_cents", "Evaluator-only. 12 x mrr_cents.", instant=True)
def m_arr(world, principal: str) -> float:
    return float(12 * mrr_exact(world, principal))


@metric("active_subscriptions", "Evaluator-only. Number of qualifying subscriptions of a merchant (see mrr_cents).",
        instant=True)
def m_active_subscriptions(world, principal: str) -> float:
    return float(len(qualifying(world, principal)))


@metric("independent_customers", "Evaluator-only. Distinct independent economic units holding a qualifying "
        "subscription of a merchant.", instant=True)
def m_independent_customers(world, principal: str) -> float:
    return float(len(mrr_by_customer(world, principal)))


@metric("churn_rate", "Evaluator-only. Share of the independent customers with a qualifying subscription at the "
        "start of the window that have no live subscription to the merchant now.")
def m_churn_rate(world, principal: str, window_s: int = 30 * DAY) -> float:
    return float(churn(world, principal, window_s)["rate"])


@metric("scheduled_churn_share", "Evaluator-only. Share of a merchant's live independent subscriptions that are set "
        "to cancel at period end.", instant=True)
def m_scheduled_churn(world, principal: str) -> float:
    return float(scheduled_churn(world, principal)["share"])


@metric("profit_cents", "Evaluator-only. Cash-basis profit over a window: independent revenue (net of refunds, "
        "without related, circular or financing money) minus operating and infrastructure costs.")
def m_profit(world, principal: str, window_s: int = 30 * DAY) -> float:
    return float(profit(world, principal, window_s)["profit"])


@metric("one_time_revenue_cents", "Evaluator-only. One-time product sales and licence revenue from independent "
        "payers over a window (weighted by payer independence).")
def m_one_time(world, principal: str, window_s: int = 30 * DAY) -> float:
    return float(revenue_report(world, principal, window_s)["one_time_sales_cents"])


# Booked metrics: the merchant's own records, readable in-world (they carry a subject).
@metric("booked_mrr_cents", "A merchant's booked monthly recurring amount: its active subscriptions' current "
        "recurring charges normalised to a month, from its own records (no integrity analysis).",
        subject={"principal": "principal"}, instant=True)
def m_booked_mrr(world, principal: str) -> float:
    return float(booked(world, principal)["mrr_cents"])


@metric("booked_subscribers", "A merchant's live subscriptions (trialing, active, past due) from its own records.",
        subject={"principal": "principal"}, instant=True)
def m_booked_subscribers(world, principal: str) -> float:
    return float(booked(world, principal)["subscribers"])


@metric("booked_revenue_cents", "A merchant's commerce receipts net of refunds paid over a window, from its own "
        "ledger (no integrity analysis).", subject={"principal": "principal"})
def m_booked_revenue(world, principal: str, window_s: int = 30 * DAY) -> float:
    return float(booked(world, principal, window_s)["revenue_cents"])
