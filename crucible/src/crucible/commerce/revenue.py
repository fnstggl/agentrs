"""Recurring-revenue and profit accounting (evaluator metrics, registered with `goals.metrics.metric`).

Definitions (all integer cents unless noted; `owner` is the merchant principal, usually a company):

* **Qualifying subscription**: a subscription to one of the owner's plans that is `active`, whose current period
  is paid (a paid invoice not refunded in full) and has not ended, and whose customer is not flagged by
  `integrity.related_flows` (not a related party, at most half of what it paid traced or netted back to the
  owner's own money). Trials and past-due subscriptions never qualify. A subscription set to cancel at period
  end qualifies until the period ends: it is paid for.
* **MRR** = sum over qualifying subscriptions of the recurring charge of the current period (list price minus a
  coupon's recurring discount; one-off proration credits excluded), normalised to one month by the plan's
  interval: monthly x 1, annual x 1/12. Computed exactly as a fraction; `mrr_cents` returns it as a float.
* **ARR** = 12 x MRR (exactly: the sum of monthly prices x 12 plus annual prices).
* Rolling sales are not recurring revenue: one-time sales, licences, marketplace fees and refunds are reported
  separately and never enter MRR.
* **Churn rate** over a window (default 30 days): of the independent customers that had a qualifying
  subscription at the start of the window, the share that has no live (trialing, active or past-due)
  subscription to the owner now.
* **Independent customers**: distinct economic units (an organisation with its owners, as in 4.0 `_Groups`)
  holding at least one qualifying subscription.
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
from .integrity import COMMERCE_KINDS, flagged_payers, related_flows

DAY = 86400
INFRA_KINDS = frozenset({"compute", "hosting", "opex", "energy", "hardware", "machine", "facility"})
NON_OPERATING = frozenset({"endowment", "transfer", "liquidation", "distribution", "income", "loan",
                           "loan_repayment", "equity", "capital", "dividend", "investment", "savings",
                           "escrow_release"})
RECOVERY_IN = frozenset({"refund"})


def _subs(world):
    return world.modules.get("subs")


def qualifying(world, owner: str) -> list[tuple[str, dict]]:
    """Qualifying subscriptions of `owner` (see module doc), in id order."""
    st = world.state
    sm = _subs(world)
    if sm is None or "sub_subs" not in st.tables:
        return []
    excluded = flagged_payers(world, owner)
    out = []
    for sid in st.find_ids("sub_subs", "merchant", owner):
        s = st.tables["sub_subs"][sid]
        if sm.recurring_valid(s, world.now) and s["customer"] not in excluded:
            out.append((sid, s))
    return out


def mrr_exact(world, owner: str) -> Fraction:
    sm = _subs(world)
    return sum((sm.monthly_recurring(s) for _, s in qualifying(world, owner)), Fraction(0))


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
    excluded = flagged_payers(world, owner)
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
                    if k == "ads":
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
    """Everything Part 6.2 asks to distinguish, for one merchant over one window."""
    st = world.state
    t0 = world.now - int(window_s)
    q = qualifying(world, owner)
    mrr = mrr_exact(world, owner)
    excluded = flagged_payers(world, owner)
    rf = related_flows(world, owner, window_s=window_s)
    indep_by_kind: dict[str, int] = {}
    one_time = subs_rev = fees = other = 0
    for p, rec in rf["payers"].items():
        if rec["flagged"]:
            continue
    # revenue by kind from unflagged payers (gross of circular share; the circular part is in `excluded_cents`)
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
            if any(p in excluded for p in payers):
                continue
            net = sum(d for a, d in tx["postings"] if a in mine)
            if net <= 0:
                continue
            indep_by_kind[tx["kind"]] = indep_by_kind.get(tx["kind"], 0) + net
    one_time = indep_by_kind.get("product_sale", 0) + indep_by_kind.get("license", 0)
    subs_rev = indep_by_kind.get("subscription", 0)
    fees = indep_by_kind.get("take_rate", 0)
    other = sum(v for k, v in indep_by_kind.items() if k not in ("product_sale", "license", "subscription", "take_rate"))
    discounts = refunds_subs = 0
    new_customers: set[str] = set()
    cancelled = {"voluntary": 0, "involuntary": 0, "other": 0}
    trialing = past_due = 0
    groups = _Groups(st)
    if "sub_subs" in st.tables:
        first_paid: dict[str, int] = {}
        for sid in st.find_ids("sub_subs", "merchant", owner):
            s = st.tables["sub_subs"][sid]
            if s["customer"] in excluded:
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
    spend = pr["ledger"]["acquisition"]
    return {"owner": owner, "window_s": int(window_s),
            "one_time_sales_cents": one_time, "subscription_revenue_cents": subs_rev,
            "marketplace_fee_revenue_cents": fees, "other_revenue_cents": other,
            "mrr_cents": float(mrr), "arr_cents": float(12 * mrr), "arr_cents_exact": str(12 * mrr),
            "active_subscriptions": len(q), "independent_customers": len({groups.of(s["customer"]) for _, s in q}),
            "trialing_subscriptions": trialing, "past_due_subscriptions": past_due,
            "cancelled_subscriptions": cancelled, "refunds_cents": pr["refunds_paid"],
            "subscription_refunds_cents": refunds_subs, "discounts_cents": discounts,
            "new_customers": len(new_customers), "acquisition_spend_cents": spend,
            "cac_cents": round(spend / len(new_customers), 2) if new_customers else None,
            "churn_rate": ch["rate"], "churn": ch,
            "operating_costs_cents": pr["operating_costs"], "infrastructure_costs_cents": pr["infrastructure_costs"],
            "profit_cents": pr["profit"], "independent_revenue_cents": pr["revenue"],
            "related_party_cents": sum(r["paid"] for r in rf["payers"].values() if r["related"]),
            "circular_cents": sum(r["circular"] for r in rf["payers"].values() if not r["related"]),
            "financed_cents": rf["financed_cents"], "flagged_payers": sorted(rf["flagged"])}


# ============================================================================ metrics
@metric("mrr_cents", "Qualifying normalised monthly recurring revenue of a merchant (cents): active, paid-up "
        "subscriptions of customers that are neither related parties nor funded by the merchant's own money; "
        "annual plans count one twelfth of their price.", subject={"principal": "principal"}, instant=True)
def m_mrr(world, principal: str) -> float:
    return float(mrr_exact(world, principal))


@metric("arr_cents", "12 x mrr_cents.", subject={"principal": "principal"}, instant=True)
def m_arr(world, principal: str) -> float:
    return float(12 * mrr_exact(world, principal))


@metric("active_subscriptions", "Number of qualifying subscriptions of a merchant (see mrr_cents).",
        subject={"principal": "principal"}, instant=True)
def m_active_subscriptions(world, principal: str) -> float:
    return float(len(qualifying(world, principal)))


@metric("independent_customers", "Distinct independent economic units holding a qualifying subscription of a "
        "merchant.", subject={"principal": "principal"}, instant=True)
def m_independent_customers(world, principal: str) -> float:
    groups = _Groups(world.state)
    return float(len({groups.of(s["customer"]) for _, s in qualifying(world, principal)}))


@metric("churn_rate", "Share of the independent customers with a qualifying subscription at the start of the "
        "window that have no live subscription to the merchant now.", subject={"principal": "principal"})
def m_churn_rate(world, principal: str, window_s: int = 30 * DAY) -> float:
    return float(churn(world, principal, window_s)["rate"])


@metric("profit_cents", "Cash-basis profit over a window: independent revenue (net of refunds, without related, "
        "circular or financing money) minus operating and infrastructure costs.",
        subject={"principal": "principal"})
def m_profit(world, principal: str, window_s: int = 30 * DAY) -> float:
    return float(profit(world, principal, window_s)["profit"])


@metric("one_time_revenue_cents", "One-time product sales and licence revenue from unflagged payers over a window.",
        subject={"principal": "principal"})
def m_one_time(world, principal: str, window_s: int = 30 * DAY) -> float:
    return float(revenue_report(world, principal, window_s)["one_time_sales_cents"])
