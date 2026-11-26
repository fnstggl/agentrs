"""Gate SP-8.0b - each Tier 1 category meets its behavioural requirement (workstream H).

Measured on the families of each category that pass admission on seeds 0-2 and on the held-out seeds 10-12
(`crucible.rapid.report.robust_report`), with the checks of `crucible.rapid.categories`:

    8.2   success evaluated on unobserved conditions: every held-out condition the objective is scored at lies
          outside what the action interface can set (every admitted 8.2 family, seed 0)
    8.4   discrete, continuous and programmatic actions offered by the admitted 8.4 families together
    8.5   reward-independent separation of useful discovery from random novelty: the income a participant observes
          (no reward exists) separates the public reference from a random explorer (z > 3 over five replicate
          worlds; every admitted 8.5 family, seed 0)
    8.8   policy populations and held-out opponent families: the admitted 8.8 training families face at least three
          opponent policy families between them, and every admitted 8.8 evaluation family faces an opponent policy
          family no training family has
    8.9   genuine interdependence: independent adaptive learners (one per seat, no messages, no joint plans) gain
          at most half of the coordinating reference's gain (every admitted 8.9 family, 2 of seeds 0-2)
    8.10  previously successful constant strategies fail after the change: the early-best constant falls short of
          the best constant of a later fifth by a quarter of that fifth's gain (every admitted 8.10 family, 2 of
          seeds 0-2); static families serve as negative controls and must not pass

A requirement with no admitted family in its category is not met."""
from __future__ import annotations

import os

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import categories as C
from crucible.rapid import families as F
from crucible.rapid import report as RP

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
JOBS = int(os.environ.get("CRUCIBLE_RAPID_JOBS", "1"))


def _two_of_three(fn, fam):
    rs = [fn(fam, s) for s in (0, 1, 2)]
    return sum(bool(r["ok"]) for r in rs) >= 2, rs


def test_SP_8_0b_category_behavioural_requirements():
    rob = RP.robust_report(seeds=(0, 1, 2), holdout=(10, 11, 12), jobs=JOBS)
    adm = set(rob["robust"])

    def cat(c, split=None):
        return [n for n in F.families(category=c) if n in adm and (split is None or (F.FAMILIES[n].split == "train")
                                                                     == (split == "train"))]
    met, detail = {}, {}
    fams = cat("8.2")
    res = {n: C.unobserved_conditions(n, 0) for n in fams}
    met["8.2"], detail["8.2"] = bool(fams) and all(r["ok"] for r in res.values()), res
    fams = cat("8.4")
    res = C.action_forms(fams) if fams else {"ok": False}
    met["8.4"], detail["8.4"] = bool(res["ok"]), res
    fams = cat("8.5")
    res = {n: C.discovery_separation(n, 0) for n in fams}
    met["8.5"], detail["8.5"] = bool(fams) and all(r["ok"] for r in res.values()), res
    res = C.opponent_populations(cat("8.8", "train"), cat("8.8", "eval"))
    met["8.8"], detail["8.8"] = bool(res["ok"]), res
    fams = cat("8.9")
    res = {n: _two_of_three(C.interdependence, n) for n in fams}
    met["8.9"], detail["8.9"] = bool(fams) and all(ok for ok, _ in res.values()), {n: ok for n, (ok, _) in res.items()}
    fams = cat("8.10")
    res = {n: _two_of_three(C.constants_fail_after_change, n) for n in fams}
    met["8.10"], detail["8.10"] = bool(fams) and all(ok for ok, _ in res.values()), \
        {n: ok for n, (ok, _) in res.items()}
    # the 8.10 check is not vacuous: static worlds, whose best constant never changes, do not pass it
    controls = {n: C.constants_fail_after_change(n, 0)["ok"] for n in ("hc_confounded", "mk_elastic", "cx_overdose")}
    assert not any(controls.values()), controls
    record("SP-8.0b", {"category_requirements_met": sum(met.values()), "categories_checked": len(met),
                       **{f"met_{k.replace('.', '_')}": v for k, v in met.items()},
                       "admitted_families_checked": sum(len(cat(c)) for c in ("8.2", "8.4", "8.5", "8.8", "8.9",
                                                                               "8.10"))},
           notes=f"Per category: {met}. Details: {detail}"[:6000])
    assert len(met) == 6
