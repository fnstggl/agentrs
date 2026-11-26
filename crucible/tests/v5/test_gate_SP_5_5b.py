"""SP-5.5b (MASTER_SPEC Parts 5.2 / 5.5; owner workstream E): in-world validation of the fitted behaviour - activity
distributions, retention curves, heterogeneity and long-horizon drift of the integrated simulator against held-out
data; in-world randomised experiments (random exposure, advertising holdout) against KuaiRand's random-exposure and
Criteo's randomised estimates; world use inside the model cards' limits.

`crucible.calibration.inworld` measures a generated civilisation run for 21 simulated days: the four reports compare
the activity the world recorded through daily_return@1 / extra_sessions@1 decisions with KuaiRand's held-out users
(a report counts only when the world's activity decisions are organic population decisions under the population's
provenance rules and the runtime evaluated the rows they claim: `inworld.audit`); the experiments analyse counted
decisions labelled with a random-exposure or ad-holdout experiment (the ad holdout per attended display, against
Criteo's shift); the scope check counts world source files that use a fitted checkpoint outside the runtime contract
(syntax-tree scan) and, from the runtime's scope ledger of the world run, scoped decision models evaluated without a
declared use and refused out-of-scope attempts. The model-only driver (the runtime models
forward-simulated from the held-out users' own history) is reported alongside, so model drift and integration drift
are separated. Until the population wires the runtime models and civ.generate builds again, this test FAILS - it
never skips."""
import pytest

from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.acceptance.v5.evidence import record
from crucible.calibration import inworld, runtime

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-5.5b"
DAYS = 21


def _criteria(req):
    return next(e for e in MF.load()["requirements"] if e["id"] == req)["criteria"]


@pytest.mark.slow
@pytest.mark.timeout(10800)
def test_SP_5_5b_in_world_validation_against_heldout_and_randomised_data():
    import torch
    torch.set_num_threads(1)
    model_only = inworld.model_only_reports()
    import crucible
    crucible.load_all_modules()
    from crucible import civ
    runtime.reset_usage()                 # the cross-check and the scope ledger then cover this world run only
    w = civ.generate("balanced", 3, population="development")
    try:
        w.advance_time(DAYS * 86400)
        scope = inworld.out_of_scope_uses()
        au = inworld.audit(w)
        reports, why = {}, {}
        try:
            reports = inworld.in_world_reports(w, au=au)
        except inworld.NotWired as e:
            why["reports"] = str(e)
        experiments = {}
        for name, fn in (("random_exposure", inworld.random_exposure_experiment),
                         ("ad_holdout", inworld.ad_holdout_experiment)):
            try:
                experiments[name] = fn(w, au=au)
            except inworld.NotWired as e:
                why[name] = str(e)
    finally:
        w.close()
    n_reports = sum(1 for k in ("activity_distribution", "retention_curve", "heterogeneity", "long_horizon_drift")
                    if k in reports)
    metrics = {"in_world_validation_reports": n_reports,
               "in_world_reports_within_tolerance": sum(1 for k, v in reports.items()
                                                        if not k.startswith("_") and v.get("within_tolerance")),
               "in_world_randomised_experiments_run": len(experiments),
               "in_world_randomised_experiments_matching": sum(1 for v in experiments.values() if v["matching"]),
               "out_of_card_scope_uses": scope["out_of_card_scope_uses"],
               "model_only_reports_within_tolerance": sum(1 for k, v in model_only.items()
                                                          if not k.startswith("_") and v["within_tolerance"])}
    record("SP-5.5b", metrics, command=CMD,
           notes=f"not wired: {why}; static uses outside the runtime contract: {scope['static_uses']}; scoped "
                 f"models evaluated without a declared use: {scope['scoped_models_evaluated_without_declared_use']}; "
                 f"refused: {scope['refused_out_of_scope_attempts']}; "
                 f"model-only: { {k: v['within_tolerance'] for k, v in model_only.items() if not k.startswith('_')} }")
    assert model_only["_meta"]["individuals"] > 1000          # the reference and the model-only driver really ran
    fails = [msg for c in _criteria("SP-5.5b") for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, (fails, why, scope["static_uses"], scope["scoped_models_evaluated_without_declared_use"])
