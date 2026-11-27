"""SP-6.5b - adoption by execution resists forgery; stars are not adoption; independent autonomous developers adopt;
maintenance continues; agent programs run in the sandbox (workstream G, wave 3b).

* forged_adoptions_counted - the red team's adoption forgeries (tracer counter written by the dependent, shadow copy
  under deps/, existence check, reading the package source, nondeterministic output, inspecting the measurement
  runs), each in a fresh world against the real software ecosystem: evaluator adoptions they obtain. The honest
  control (real calls) must be credited. Adoption is measured behind the attested package boundary
  (`innovation.pkg`);
* star_only_adoptions_counted - a package starred by five unrelated principals and locked but never run by a
  dependent: adoptions the evaluator counts (then a dependent that runs it is counted);
* autonomous_developer_adoptions - in the civilisation's factual arm (`_invention_kit`), adoptions by execution in
  which a population developer's repository runs another independent creator's package (CI across the boundary);
  releases after the first and compatible upgrades are reported (maintenance);
* agent_programs_outside_sandbox - deployments of in-world creators running as trusted in-process code (cloud
  runs every other deployment in the sandbox: Landlock, seccomp, unprivileged user).
"""
import pytest

from _invention_kit import A
from crucible.acceptance.v5.evidence import record
from crucible.innovation import gates as G

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]


def test_SP_6_5b_adoption_integrity_developers_and_sandbox():
    fa = G.forged_adoptions()
    so = G.star_only()
    w = A()["factual"]["world"]
    da = G.developer_adoptions(w)
    sb = G.sandbox_coverage(w)
    m = {k: v for k, v in {**fa, **so, **da, **sb}.items() if isinstance(v, (int, float, bool, str))}
    record("SP-6.5b", m, notes=f"per forgery {fa['per_forgery']}; probe errors {fa['probe_errors']}",
           labels={"autonomous_population_creators": True, "scripted": False,
                   "world": "civilisation (autonomous_developer_adoptions, sandbox coverage)",
                   "scripted_probes": "forged_adoptions_counted and star_only_adoptions_counted run in lean "
                                      "innovation worlds with scripted red-team / reference principals"})
    assert fa["forged_adoptions_counted"] == 0 and fa["honest_control_credited"] and not fa["probe_errors"]
    assert so["star_only_adoptions_counted"] == 0 and so["stars"] >= 5 and so["executed_adoptions_counted"] >= 1
    assert da["autonomous_developer_adoptions"] >= 1
    assert sb["agent_programs_outside_sandbox"] == 0 and sb["creator_deployments"] >= 1 and sb["sandbox_seccomp"]
