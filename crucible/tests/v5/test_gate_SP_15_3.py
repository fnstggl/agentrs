"""SP-15.3 - Learning-efficiency curves at 1-64 interactions and censoring-aware acceleration N_q ratios
(MASTER_SPEC Part 15.3; workstream N).

Learning curves (`instrument.transfer_curves`): untrained and trained transfer controls practise held-out families
(validation items) and are scored at 0, 1, 2, 4, 8, 16, 32, 64, 128 and 256 informative interactions (executed
actions that changed the world). N_q is the first budget whose verified, normalised score reaches the prespecified
threshold q; a run that never reaches it is censored at its largest budget. Censoring is handled when:

* every censored run has no finite N_q and every reached run has one (no run is dropped);
* the Kaplan-Meier median is None exactly when more than half the runs of an arm are censored;
* the acceleration N_q(untrained)/N_q(trained) is reported as a finite ratio only when both medians are finite,
  as a lower bound when only the untrained arm is censored, and as undefined when the trained arm is;
* at an unreachable threshold (q = 1.5, above every score) both arms are censored and no finite speed-up is given.
"""
from __future__ import annotations

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import curves as CV
from crucible.research.v5 import instrument as I

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]


def test_SP15_3_curves_and_censored_acceleration():
    res = I.transfer_curves()
    runs = res["runs"]
    allruns = runs["untrained"] + runs["trained"]
    budgets = sorted({p["budget"] for r in allruns for p in r["curve"]})
    in_1_64 = [b for b in budgets if 1 <= b <= 64]
    consistent = all((r["N_q"] is None) == r["censored"] for r in allruns)
    km = {k: CV.kaplan_meier(v) for k, v in runs.items()}
    km_ok = all((m["median"] is None) == (m["censored"] * 2 > m["runs"]) for m in km.values())
    acc = res["acceleration"]["median_ratio"]
    mu, me = km["untrained"]["median"], km["trained"]["median"]
    kind_ok = (acc["kind"] == "ratio") == (mu is not None and me is not None and me > 0) and \
        (acc["finite"] == (acc["kind"] == "ratio"))
    hi = [dict(r, **CV.n_q(r["curve"], 1.5)) for r in allruns]
    nu = len(runs["untrained"])
    unreach = CV.acceleration(hi[:nu], hi[nu:], n_boot=50)
    no_invented = not unreach["median_ratio"]["finite"] and all(r["censored"] for r in hi)
    handled = bool(consistent and km_ok and kind_ok and no_invented)
    record("SP-15.3", {"censoring_handled": handled, "curve_budgets": len(in_1_64), "budgets_total": len(budgets),
                       "runs": len(allruns), "censored_untrained": km["untrained"]["censored"],
                       "censored_trained": km["trained"]["censored"],
                       "acceleration_finite": bool(acc["finite"]),
                       "rmst_ratio": float(res["acceleration"]["rmst_ratio"] or 0.0)},
           notes=f"q = {res['q']}; KM untrained {km['untrained']['survival']} median {mu}; trained "
                 f"{km['trained']['survival']} median {me}; acceleration {acc}; RMST ratio "
                 f"{res['acceleration']['rmst_ratio']} CI {res['acceleration']['rmst_ratio_ci']}")
    assert handled
    assert in_1_64 == [1, 2, 4, 8, 16, 32, 64] and 0 in budgets and max(budgets) > 64
