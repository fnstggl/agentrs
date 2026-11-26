"""Gate SP-9.16 (MASTER_SPEC Part 9.16): feasibility without guaranteed success in every domain - the null baseline
fails, not every learner succeeds, difficulty comes from complexity, scarcity and uncertainty rather than arbitrary
failure probabilities, and every domain objective is probed for degenerate strategies.

Measured by `crucible.domains.base.check` on every registered Part 9 domain (test fixtures excluded); a Part 9
section counts when one of its domains satisfies the criterion, a section without a domain fails. While fewer than
fifteen sections have a domain the test FAILS (it never skips).

* null baseline fails: the domain's null policy (default: never acting) runs the evaluation horizon on every
  reference instance through `crucible.experience.make(2, ...)`, and the evaluator never reports success - while
  the tool-only reference succeeds on a majority of the same instances (feasible, N.5);
* degenerate-strategy probes: the domain declares at least one degenerate strategy for its objective (`Probe`), and
  none of them nor the generic random-tool policy (`base.RandomTools`: not every learner succeeds) achieves it;
* arbitrary failure probabilities: `base.dice_scan` over the domain's source files counts Bernoulli draws whose
  expression or target speaks of failure/success (or whose branch raises or fails) without a `# fidelity: <id>`
  pragma naming a FIDELITY_REGISTER record - must be 0.

Positive controls on the framework fixture (tests/v5/_domain_fixture.py, never counted): its null baseline and
probes fail while its reference succeeds; a planted evaluator that always succeeds makes the null baseline
"succeed" and is detected; a planted failure-dice module is detected."""
import os
import sys
import textwrap
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _domain_fixture as F  # noqa: E402
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.domains import base  # noqa: E402

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_9_16.py"


@pytest.mark.slow
@pytest.mark.timeout(8 * 3600)
def test_SP_9_16_feasible_not_guaranteed_no_dice(tmp_path):
    reps = base.gate_reports()
    m = base.gate_metrics(reps)
    per = {n: base.verdicts(r) for n, r in reps.items()}
    F.register()
    try:
        ctl = base.check(F.DOMAIN, tier3=False)
        lenient = replace(F.DOMAIN, name="fixture_lenient",
                          evaluate=lambda w, p, o: {"success": True, "score": 1.0, "details": {}})
        base.register(lenient)
        lenient_null = base.run("fixture_lenient", "ring_standard", 0, policy="null", days=2)
    finally:
        base.unregister(F.NAME)
        base.unregister("fixture_lenient")
    dice = tmp_path / "planted_dice.py"
    dice.write_text(textwrap.dedent('''
        def deploy(rng, p_fail):
            if rng.random() < p_fail:
                raise RuntimeError("deployment failed")
            return "live"
    '''))
    planted = base.dice_scan([str(dice)])
    metrics = {k: m[k] for k in ("domains_where_null_baseline_fails", "arbitrary_failure_probability_mechanisms",
                                 "domains_with_degenerate_strategy_probe", "domains_registered",
                                 "sections_with_domain", "sections_missing", "domains_with_successful_reference")}
    for k in ("null_fails", "probes_ok", "dice"):
        metrics[f"per_domain_{k}"] = ";".join(f"{n}={v[k]}" for n, v in sorted(per.items())) or "none"
    metrics["probe_results"] = ";".join(
        f"{n}:" + ",".join(f"{p}={x['successes']}/{x['runs']}" for p, x in r["probes"].items())
        for n, r in sorted(reps.items())) or "none"
    metrics["dice_hits"] = ";".join(f"{h['file'].rsplit('/', 1)[-1]}:{h['line']}" for r in reps.values()
                                    for h in r["dice"]["hits"])[:2000] or "none"
    metrics.update({"control_fixture_null_fails": ctl["null_fails"], "control_fixture_probes_fail": ctl["probes_fail"],
                    "control_fixture_feasible": ctl["feasible"], "control_lenient_null_success":
                    lenient_null["success"], "control_planted_dice_detected": planted["mechanisms"],
                    "control_fixture_dice": ctl["dice"]["mechanisms"]})
    evidence.record("SP-9.16", metrics, command=CMD, labels={"scripted": True})
    assert ctl["null_fails"] and ctl["probes_fail"] and ctl["feasible"] and ctl["dice"]["mechanisms"] == 0
    assert lenient_null["success"] is True and planted["mechanisms"] == 1
    assert m["domains_where_null_baseline_fails"] == 15, metrics
    assert m["domains_with_degenerate_strategy_probe"] == 15, metrics
    assert m["sections_with_domain"] == 15 and m["arbitrary_failure_probability_mechanisms"] == 0, metrics
