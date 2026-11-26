"""SP-5.5: prespecified held-out metrics; learned models compared with simple baselines and the better one kept
(owner workstream E).

For every registered model (any workstream, either card schema; `audit.plan_check`, `audit.selection_check`) the
test checks, from git history and the model card:

* the evaluation plan and every amendment recorded in the card exist at the recorded commit with the recorded
  SHA-256 and are unchanged since, that commit is in HEAD's history, precedes the commit that first added the model
  card, is contained in the fit commit and (when the card records the fit's start) was committed before the fit
  started, and the plan names the model's primary metric, its secondary metrics (fitlib plans) and its candidate set
  (a candidate absent from the plan must be introduced by an amendment of it);
* the kept model follows the prespecified rule (validation argmin, replaced by a simple baseline only when that
  baseline is better on test), is no worse than any simple baseline on the primary test metric, and, where the card
  records paired-bootstrap comparisons, no other candidate is significantly better on test.
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
           notes=f"plan problems: {problems}; selection problems: {sel_problems}; reviews keeping a non-fitted "
                 f"contender: {metrics['reviews_keeping_a_non_fitted_contender']}; protocol deviation for the "
                 f"reviewer (amendment written after an earlier run computed test metrics): "
                 f"{metrics['amendments_after_test_exposure']}")
    assert metrics["models_checked"] >= 1
    # the E3 review of every kept model it applies to: committed after its plan, and its kept contender follows the
    # plan's rule recomputed from the recorded numbers (counted in worse_model_kept otherwise)
    applies = [x for x in audit.reviewed_models() if x.split("@")[0] in {r["name"] for r in recs}]
    assert metrics["reviews_checked"] == len(applies) >= 9, m["_reviews"]
    assert metrics["reviews_ok"] == metrics["reviews_checked"], {k: v["problems"] for k, v in m["_reviews"].items()}
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


def test_SP_5_5_checks_catch_violations_in_loader_schema_cards():
    """The same checks on a card of the other schema (plan + amendment, paired-bootstrap selection record)."""
    card = R.read_card("commercial_response")
    assert audit.selection_check(card)["ok"] and audit.plan_check(card)["ok"], audit.plan_check(card)
    assert [p["role"] for p in audit.plan_check(card)["plans"]] == ["plan", "amendment"]
    # the added candidate is prespecified only by the amendment
    bad = copy.deepcopy(card)
    bad["plan_amendments"] = []
    assert any("t_gbdt_reg" in p for p in audit.plan_check(bad)["problems"])
    # a fit that started before the amendment was committed
    bad = copy.deepcopy(card)
    bad["fit"]["started_at"] = "2026-11-25T07:00:00+00:00"
    assert not audit.plan_check(bad)["ok"]
    # an amendment whose recorded hash is not the committed file
    bad = copy.deepcopy(card)
    bad["plan_amendments"][0]["sha256"] = "0" * 64
    assert not audit.plan_check(bad)["ok"]
    # a kept model other than the validation-best candidate, and a recorded significantly better candidate
    bad = copy.deepcopy(card)
    bad["selection"]["kept"] = "t_logistic"
    assert not audit.selection_check(bad)["ok"]
    bad = copy.deepcopy(card)
    bad["selection"]["worse_model_kept"] = 1
    assert not audit.selection_check(bad)["ok"]
