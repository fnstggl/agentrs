"""SP-5.3 (MASTER_SPEC Part 5.3; owner workstream E): fitted models are compared with the rule-based simulator
behaviour they replace, selected on held-out fit, long-horizon stability and computational efficiency, and the six
prioritised behaviours (attention allocation, repeat consumption, purchase, retention, community participation, social
diffusion) are each fitted or their absence justified.

Measured on the registered models: a rule comparison counts when the model's E3 review scored the simulator's rule
(contender S) on the model's own held-out rows, or its report's `rule_comparison` holds finite held-out values; a
selection counts when it measured both a multi-step long-horizon criterion and the compute cost of the candidates; a
behaviour counts when a qualifying fitted model has its canonical domain."""
from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.acceptance.v5.evidence import record
from crucible.calibration import audit_sp5

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-5.3"


def _criteria(req):
    return next(e for e in MF.load()["requirements"] if e["id"] == req)["criteria"]


def test_SP_5_3_rule_comparison_stability_cost_and_coverage():
    m = audit_sp5.sp53_metrics()
    metrics = {k: v for k, v in m.items() if not k.startswith("_")}
    record("SP-5.3", metrics, command=CMD,
           notes=f"rule comparisons: {sorted(k for k, v in m['_rules'].items() if v)}; without: "
                 f"{sorted(k for k, v in m['_rules'].items() if not v)}; domains: "
                 f"{ {k: v['fitted'] for k, v in m['_domains'].items()} }")
    # every behaviour that counts is backed by a fitted model or a written justification
    for label, v in m["_domains"].items():
        assert v["fitted"] or v["justified_absence"], label
    # each counted rule comparison names its source and holds a finite value
    for name, v in m["_rules"].items():
        if v:
            assert v["source"] and audit_sp5._finite_numbers(v["rule"]), name
    fails = [msg for c in _criteria("SP-5.3") for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, fails
