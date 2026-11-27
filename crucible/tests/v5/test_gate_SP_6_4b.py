"""SP-6.4b - the example need mechanisms (workflow change, transport or market access, coordination needs, content
creating communities and tools, education changing skill availability) arise from the one reusable relation graph;
an agent-created novel combination of capabilities changes future civilisation state without designer-supplied
demand (workstream G, wave 3b).

Measured in the civilisation arms of `_invention_kit` (autonomous population developers; nobody scripted), with the
one evaluator-side relation graph (`crucible.innovation.relations`: providers reading work activities, cloud
deployments, compute providers' capacity, the freight network, the capability registry and packages; generic
rules). Definitions: `crucible.innovation.gates.need_mechanisms`, `designer_supplied_demand`, `novel_combinations`.

* need_mechanism_examples - of the five listed mechanisms, those for which a need arose after genesis from world
  state (reliability bottleneck and computing demand are reported, never counted). Strict: a workflow change on
  the two-step activity the invention world builder configures (`inventworld.PIPELINE`), and a moderation
  backlog of a community that existed at genesis or whose reported content is of forms published since genesis,
  are reported (`examples_excluded_as_configured`) and not counted;
* designer_supplied_demand_paths - needs derived through relations given as data, measured worlds that declare
  relations as data, configuration naming a combination's parts (workflow, composer, components), and composed
  work activities configured for a combination instead of following the world's generic raw-feed rule
  (`gates.configured_compositions`; negative control in tests/test_v5_innovation_instruments.py);
* novel_combinations_changing_state - verified workflows of population developers combining independent creators'
  services that individuals rely on and that lower the composed activity's hand work against the removal arm.
"""
import pytest

from _invention_kit import A
from crucible.acceptance.v5.evidence import record
from crucible.innovation import gates as G

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]


def test_SP_6_4b_need_mechanisms_and_novel_combination():
    r = G.sp64b_measure(A())
    m = r["metrics"]
    record("SP-6.4b", m, notes=f"found {r['found']}; not demonstrated {r['missing']}; combinations {r['combinations']}; "
                               f"demand {r['demand']}",
           labels={"autonomous_population_creators": True, "scripted": False, "world": "civilisation"})
    print("SP-6.4b", m, r["missing"])
    assert m["designer_supplied_demand_paths"] == 0, m
    assert m["novel_combinations_changing_state"] >= 1, r["combinations"]
    assert m["need_mechanism_examples"] >= 5, r["missing"]
