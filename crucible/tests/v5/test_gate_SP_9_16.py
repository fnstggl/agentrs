"""Gate SP-9.16 (MASTER_SPEC Part 9.16): feasibility without guaranteed success in every domain - the null baseline
fails, not every learner succeeds, difficulty comes from complexity, scarcity and uncertainty rather than arbitrary
failure probabilities, and every domain objective is probed for degenerate strategies.

Measured by `crucible.domains.base.check` on every registered Part 9 domain (test fixtures excluded); a Part 9
section counts when one of its domains satisfies the criterion, a section without a domain fails. While fewer than
fifteen sections have a domain the test FAILS (it never skips).

* null baseline fails: the domain's null policy (default: never acting) runs the evaluation horizon on every
  configuration through `crucible.experience.make(2, ...)`, and the evaluator never reports success - AND the
  tool-only reference succeeds on a majority of the reference instances (feasible, N.5): a null baseline that fails
  where nothing can succeed shows nothing, so an infeasible domain is not counted;
* degenerate-strategy probes: the domain declares at least one degenerate strategy for its objective (`Probe`), and
  none of them nor the generic random-tool policy (`base.RandomTools`: not every learner succeeds) achieves it - in
  a feasible domain;
* arbitrary failure probabilities: `base.dice_scan` over the domain's source files AND the kernel module files of
  its built worlds (where a civilisation-backed domain's mechanisms live) counts random draws that decide a failure
  or success (thresholded draws, binomial draws, weighted True/False choices) without a `# fidelity: <id>` pragma
  naming a FIDELITY_REGISTER record - must be 0.

Positive controls on the framework fixture (tests/v5/_domain_fixture.py, never counted): its null baseline and
probes fail while its reference succeeds; a planted evaluator that always succeeds makes the null baseline
"succeed"; a planted evaluator that never succeeds (an infeasible domain, verifier probe p6) is not counted by
either SP-9.16 count; the six common failure-dice forms of verifier probe p2 (c) are detected; and the scan covers
the kernel module files of the fixture's world (it finds the breakdown draw of economy/industry.py)."""
import os
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _domain_fixture as F  # noqa: E402
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.domains import base  # noqa: E402

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_9_16.py"
CTL_DAYS = 7.0          # the fixture's controls run 7 of its 21 horizon days (its reference succeeds within 3)

DICE_FORMS = {
    "ok_false_return": "def deploy(rng, p):\n    if rng.random() < p:\n        return {'ok': False}\n"
                       "    return {'ok': True}\n",
    "status_down": "def tick(rng, machine):\n    if rng.random() < 0.05:\n        machine['status'] = 'down'\n",
    "choice_with_p": "def deploy(rng):\n    worked = rng.choice([True, False], p=[0.7, 0.3])\n    return worked\n",
    "binomial_n": "def crashes(rng, n, p_crash):\n    return rng.binomial(n, p_crash)\n",
    "integers_roll": "def attempt(rng):\n    if rng.integers(0, 10) < 3:\n        raise RuntimeError('attempt failed')\n",
    "reference_form": "def deploy(rng, p_fail):\n    if rng.random() < p_fail:\n"
                      "        raise RuntimeError('deployment failed')\n",
}


@pytest.mark.slow
@pytest.mark.timeout(8 * 3600)
def test_SP_9_16_feasible_not_guaranteed_no_dice(tmp_path):
    reps = base.gate_reports()
    m = base.gate_metrics(reps)
    per = {n: base.verdicts(r) for n, r in reps.items()}
    F.register()
    try:
        ctl = base.check(F.DOMAIN, tier3=False, days=CTL_DAYS)
        cv = base.verdicts(ctl)
        lenient = replace(F.DOMAIN, name="fixture_lenient", goal=None, objective=F.objective,
                          evaluate=F.always_evaluator)
        base.register(lenient)
        lenient_null = base.run("fixture_lenient", "ring_standard", 0, policy="null", days=2)
        never = replace(F.DOMAIN, name="fixture_infeasible", goal=None, objective=F.objective,
                        evaluate=F.never_evaluator)
        base.register(never)
        irep = base.check(never, tier3=False, seeds=(0,), structure_seeds=(0,), all_configs=False, days=CTL_DAYS)
        iv = base.verdicts(dict(irep, spec_part="9.1"))
        im = base.gate_metrics({"fixture_infeasible": dict(irep, spec_part="9.1")})
    finally:
        for n in (F.NAME, "fixture_lenient", "fixture_infeasible"):
            base.unregister(n)
    forms = {}
    for k, src in DICE_FORMS.items():
        p = tmp_path / f"{k}.py"
        p.write_text(src)
        forms[k] = base.dice_scan([str(p)])["mechanisms"]
    world_files = ctl["structure"]["world_sources"]
    metrics = {k: m[k] for k in ("domains_where_null_baseline_fails", "arbitrary_failure_probability_mechanisms",
                                 "domains_with_degenerate_strategy_probe", "domains_registered",
                                 "sections_with_domain", "sections_missing", "domains_with_successful_reference")}
    for k in ("null_fails", "probes_ok", "dice"):
        metrics[f"per_domain_{k}"] = ";".join(f"{n}={v[k]}" for n, v in sorted(per.items())) or "none"
    metrics["probe_results"] = ";".join(
        f"{n}:" + ",".join(f"{p}={x['successes']}/{x['runs']}" for p, x in r["probes"].items())
        for n, r in sorted(reps.items())) or "none"
    metrics["dice_hits"] = ";".join(f"{h['file'].rsplit('/', 2)[-2]}/{h['file'].rsplit('/', 1)[-1]}:{h['line']}"
                                    for r in reps.values() for h in r["dice"]["hits"])[:3000] or "none"
    metrics.update({"control_fixture_null_fails": cv["null_fails"], "control_fixture_probes_ok": cv["probes_ok"],
                    "control_fixture_feasible": ctl["feasible"], "control_lenient_null_success":
                    lenient_null["success"], "control_infeasible_null_fails_counted": iv["null_fails"],
                    "control_infeasible_probes_counted": iv["probes_ok"],
                    "control_infeasible_section_counts": im["domains_where_null_baseline_fails"] +
                    im["domains_with_degenerate_strategy_probe"],
                    "control_dice_forms_detected": sum(forms.values()),
                    "control_fixture_world_files_scanned": len(world_files),
                    "control_fixture_dice": ctl["dice"]["mechanisms"],
                    "control_fixture_dice_hits": ";".join(f"{h['file'].rsplit('/', 1)[-1]}:{h['line']}"
                                                          for h in ctl["dice"]["hits"]) or "none"})
    evidence.record("SP-9.16", metrics, command=CMD, labels={"scripted": True})
    assert cv["null_fails"] and cv["probes_ok"] and ctl["feasible"]
    assert lenient_null["success"] is True
    assert irep["feasible"] is False and iv["null_fails"] is False and iv["probes_ok"] is False
    assert metrics["control_infeasible_section_counts"] == 0
    assert forms == {k: 1 for k in DICE_FORMS}, forms
    assert any(f.endswith(os.path.join("economy", "industry.py")) for f in world_files)
    assert any(h["file"].endswith("industry.py") for h in ctl["dice"]["hits"] + ctl["dice"]["exempt"])
    assert m["domains_where_null_baseline_fails"] == 15, metrics
    assert m["domains_with_degenerate_strategy_probe"] == 15, metrics
    assert m["domains_with_successful_reference"] == 15, metrics
    assert m["sections_with_domain"] == 15 and m["arbitrary_failure_probability_mechanisms"] == 0, metrics
