"""Gate SP-6.1b (MASTER_SPEC Part 6.1): products provide measurable usefulness - revenue from independent population
customers tracks measured value, and useless products earn about nothing.

Generated startup worlds (`commerce.worlds.startup_spec`) whose only demand is the one population (`economy.work`):
individuals delegate steps of their own work to services they subscribe to from their own accounts, judge them by
downstream acceptance of their own work (world physics) and their time, and cancel, complain or ask for refunds from
their own experience. In each world one public format is served by competitors that differ ONLY in their program:
every reference tier of the format's solver ladder, plus two firms whose service ignores its input
(`refsolvers.GARBAGE`), all at the same price with the same trial - nothing in their offers tells them apart.

* measured value (evaluator side, never shown to anyone in the world): the mean exact-verifier score of the program
  each firm actually deployed (read from its repository) on fresh, independently generated inputs of the format;
* revenue: the firm's independent revenue over the run (`commerce.revenue.revenue_report`: subscription and
  one-time payments of independent customers, net of refunds), as a share of all revenue in its world.

Metrics: useless_product_revenue_share (revenue of the input-ignoring firms / all revenue, pooled over worlds),
value_revenue_relation_pvalue (one-sided exact permutation test of Spearman's rank correlation between measured
value and revenue share across all firms of all worlds, 20,000 seeded permutations).
"""
from __future__ import annotations

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5.evidence import record

crucible.load_all_modules()
from crucible.commerce import revenue  # noqa: E402
from crucible.commerce.worlds import startup_spec  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.needs import NEEDS  # noqa: E402

DAY = 86400
WORLDS = [(3, "route.tsp"), (5, "pack.knapsack"), (8, "route.tsp")]
DAYS, N = 35, 800


def _program_value(w, org: str, need: str, k: int = 30) -> float:
    """Mean verifier score of the deployed program of `org` on fresh inputs (evaluator side)."""
    st = w.state
    rid = st.find_ids("repos", "owner", org)[0]
    src = w.modules["code"].files(st, rid)["service.py"]
    ns: dict = {}
    exec(compile(src, f"<{org}>", "exec"), ns)  # noqa: S102 - kernel-authored reference programs, evaluator side
    rng = np.random.default_rng([7151, len(org)])
    scores = []
    for _ in range(k):
        inp, hidden = NEEDS[need].generate(rng, 0.4)
        try:
            out = ns["solve"](inp, "evaluator")
            scores.append(float(NEEDS[need].score(inp, hidden, out)))
        except Exception:  # noqa: BLE001 - a crashing program scores 0
            scores.append(0.0)
    return float(np.mean(scores))


def _world(seed: int, need: str):
    tiers = len(NEEDS[need].tiers)
    incs = [{"handle": f"v{seed}t{t}", "name": f"Vendor {seed}-{t}", "need": need, "tier": t,
             "price_month": 2499, "trial_days": 14, "cores": 2, "cash": 5_000_000} for t in range(tiers)]
    incs += [{"handle": f"v{seed}x{j}", "name": f"Vendor {seed}-x{j}", "need": need, "tier": 0,
              "program": "input_ignoring", "price_month": 2499, "trial_days": 14, "cores": 2, "cash": 5_000_000}
             for j in range(2)]
    w = World.create(startup_spec(seed, n=N, formats=[need], incumbents=incs))
    for d in range(1, DAYS + 1):
        w.advance_time(d * DAY)
    rows = []
    for f in incs:
        org = f"o:{f['handle']}"
        rep = revenue.revenue_report(w, org, window_s=(DAYS + 1) * DAY)
        rows.append({"org": org, "useless": f.get("program") == "input_ignoring", "value": _program_value(w, org, need),
                     "revenue": int(rep["independent_revenue_cents"]), "customers": rep["independent_customers"]})
    tot = sum(r["revenue"] for r in rows)
    for r in rows:
        r["share"] = r["revenue"] / tot if tot else 0.0
    return w, rows, tot


def _spearman(x, y) -> float:
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_SP_6_1b_revenue_tracks_measured_value():
    all_rows, totals, health = [], [], []
    for seed, need in WORLDS:
        w, rows, tot = _world(seed, need)
        for r in rows:
            r["world"] = f"{seed}:{need}"
        all_rows += rows
        totals.append(tot)
        health.append(len(w.check_invariants()) + len(w.modules["bank"].check_ledger_consistency(w)))
    value = np.array([r["value"] for r in all_rows])
    share = np.array([r["share"] for r in all_rows])
    rho = _spearman(value, share)
    rng = np.random.default_rng(6011)
    perm = np.array([_spearman(value, rng.permutation(share)) for _ in range(20_000)])
    p = float((1 + np.sum(perm >= rho - 1e-12)) / (1 + perm.size))
    useless = sum(r["revenue"] for r in all_rows if r["useless"])
    total = sum(r["revenue"] for r in all_rows)
    share_useless = useless / total if total else 1.0
    record("SP-6.1b", {"useless_product_revenue_share": float(share_useless),
                       "value_revenue_relation_pvalue": p, "spearman_rho": rho, "firms": len(all_rows),
                       "worlds": len(WORLDS), "total_independent_revenue_cents": int(total),
                       "useless_revenue_cents": int(useless), "invariant_errors": int(sum(health))},
           command="pytest tests/v5/test_gate_SP_6_1b.py",
           notes="Population startup worlds (n=800, 35 days), one format each, competitors differing only in their "
                 "program (every reference tier + 2 input-ignoring firms, same price and 14-day trial). Value = "
                 "evaluator-side mean verifier score of the deployed program on fresh inputs; revenue = independent "
                 "revenue net of refunds. Firms: " + "; ".join(
                     f"{r['world']} {r['org']} value {r['value']:.3f} rev {r['revenue']} cust {r['customers']}"
                     for r in all_rows))
    assert total > 0, "the population bought nothing (the test would be vacuous)"
    assert share_useless <= 0.05 and p <= 0.01, (share_useless, p, rho)
    assert sum(health) == 0
