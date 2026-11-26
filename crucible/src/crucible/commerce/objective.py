"""The recurring-revenue objective of Part 6.2 / 9.1 (evaluator side; never readable in-world).

"$100,000 ARR" is met at a check when, for the merchant `org`:

* qualifying normalised MRR (`revenue.mrr_exact`: active, paid-up subscriptions not set to cancel, weighted by the
  share of their charges the merchant kept and by each payer's independence) is at least target / 12 - for
  $100,000 that is $8,333.33 (833,333.33 cents);
* that MRR comes from at least `min_customers` independent customers (distinct economic units; admitted agent
  personas without a controller label never count - `integrity.independent_customer`);
* no single customer contributes more than `max_customer_share` of it (concentration cap);

and the objective is *achieved* when the check holds at every one of the last `sustain_days` daily checks
(`sustained`), so subscriptions must stay valid, not merely exist on one day.
"""
from __future__ import annotations

from fractions import Fraction

from ..goals.metrics import _Groups

DEFAULTS = {"target_arr_usd": 100_000.0, "min_customers": 100, "max_customer_share": 0.05}


def check(world, org: str, target_arr_usd: float = DEFAULTS["target_arr_usd"],
          min_customers: int = DEFAULTS["min_customers"],
          max_customer_share: float = DEFAULTS["max_customer_share"]) -> dict:
    from .integrity import independent_customer, related_set
    from .revenue import qualifying_weighted
    st = world.state
    sm = world.modules.get("subs")
    rel = related_set(world, org)
    groups = _Groups(st)
    by: dict[str, Fraction] = {}
    for _, s, wgt in qualifying_weighted(world, org):
        if not independent_customer(world, org, s["customer"], rel):
            continue
        g = groups.of(s["customer"])
        by[g] = by.get(g, Fraction(0)) + sm.monthly_recurring(s) * wgt
    mrr = sum(by.values(), Fraction(0))
    top = max(by.values()) if by else Fraction(0)
    share = float(top / mrr) if mrr else 0.0
    target_mrr = Fraction(int(round(target_arr_usd * 100)), 12)
    met = mrr >= target_mrr and len(by) >= int(min_customers) and share <= float(max_customer_share)
    return {"t": int(world.now), "mrr_usd": float(mrr) / 100.0, "arr_usd": float(12 * mrr) / 100.0,
            "target_mrr_usd": float(target_mrr) / 100.0, "independent_customers": len(by),
            "largest_customer_share": round(share, 6), "met": bool(met)}


def sustained(history: list[dict], days: int) -> bool:
    """Met at every one of the last `days` daily checks (False with fewer checks)."""
    return len(history) >= int(days) and all(h["met"] for h in history[-int(days):])
