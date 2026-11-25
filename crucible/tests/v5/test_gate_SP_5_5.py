"""SP-5.5: prespecified held-out metrics; learned models compared with simple baselines and the better one kept
(owner workstream E).

For every registered model (any workstream) the test checks, from git history and the model card:

* the evaluation plan recorded in the card exists at the recorded commit with the recorded SHA-256, that commit is
  in HEAD's history, it precedes the commit that first added the model card and is contained in the fit commit, and
  the plan names the model's primary metric, secondary metrics and candidate set;
* the kept model follows the prespecified rule (validation argmin, replaced by a simple baseline only when that
  baseline is better on test) and is no worse than any simple baseline on the primary test metric.
"""
import copy

from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.acceptance.v5.evidence import record
from crucible.calibration import audit, registry as R


def test_SP_5_5_prespecified_metrics_and_better_model_kept():
    recs = audit.all_records()
    m = audit.sp55_metrics(recs)
    metrics = {k: v for k, v in m.items() if not k.startswith("_")}
    problems = {k: v["problems"] for k, v in {**m["_plans"]}.items() if v["problems"]}
    sel_problems = {k: v["problems"] for k, v in m["_selection"].items() if v["problems"]}
    record("SP-5.5", metrics, command="PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-5.5",
           artifacts=[f"{R.model_dir(r['name'])}/model_card.json" for r in recs if r.get("card")],
           notes=f"plan problems: {problems}; selection problems: {sel_problems}")
    assert metrics["models_checked"] >= 1
    crit = next(e for e in MF.load()["requirements"] if e["id"] == "SP-5.5")["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, (fails, problems, sel_problems)


def test_SP_5_5_checks_catch_violations():
    card = R.read_card("content_consumption")
    # a model kept although a baseline is better on test
    bad = copy.deepcopy(card)
    sel = bad["selection"]
    sel["test"][sel["kept"]] = max(sel["test"].values()) + 1.0
    assert not audit.selection_check(bad)["ok"]
    # a plan that does not hash to the recorded value
    bad = copy.deepcopy(card)
    bad["provenance"]["evaluation_plan"]["sha256"] = "0" * 64
    assert not audit.plan_check(bad)["ok"]
    # a primary metric other than the prespecified one
    bad = copy.deepcopy(card)
    bad["selection"]["primary_metric"] = "auc"
    assert not audit.plan_check(bad)["ok"]
    assert audit.selection_check(card)["ok"] and audit.plan_check(card)["ok"], audit.plan_check(card)
