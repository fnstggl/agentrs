"""Gate J - open-ended digital creation, in the integrated civilisation (workstream G, wave 3b; D-15, D-18).

Measured in `crucible.civ` worlds where technology invention is live (`innovation.inventworld.civ_spec`): the
population's autonomous developers (`innovation.devs`) read public requests for help, write and test programs,
rent compute, deploy services, declare typed capabilities, publish packages and maintain them from their own
metrics and books; the only demand is the one population (`economy.work`): individuals delegate steps of their own
work to providers they discover, judge them by whether their employer accepts the results (world physics; they
never see a verifier) and pay from their own money. No creator is scripted. Counterfactual arms of the same spec and
seed (`innovation.civexp`): developers ship input-ignoring programs (garbage), nobody builds for the generated
categories (ban_novel), nobody builds for the documented formats (ban_published). Definitions:
`crucible.innovation.gates` (module docstring).
"""
import pytest

from _invention_kit import A
from crucible.acceptance.v5.evidence import record
from crucible.innovation import gates as G

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]
LABELS = {"autonomous_population_creators": True, "scripted": False, "world": "civilisation"}


@pytest.fixture(scope="module")
def J():
    return G.j_measure(A())


def _rec(req, J):
    record(req, dict(J[req]), notes=G.j_note(J), labels=LABELS)


def test_J1_agent_creates_executable_products(J):
    _rec("J.1", J)
    assert J["J.1"]["agent_created_executable_products"] >= 2 and J["J.1"]["creators_scripted"] == 0


def test_J2_products_are_published(J):
    _rec("J.2", J)
    assert J["J.2"]["published_products"] >= 2


def test_J3_independent_participants_discover_it(J):
    _rec("J.3", J)
    assert J["J.3"]["independent_discoverers"] >= 10 and J["J.3"]["categories"] >= 2


def test_J4_independent_participants_use_it(J):
    _rec("J.4", J)
    assert J["J.4"]["independent_users"] >= 10


def test_J5_measured_behavior_affects_adoption(J):
    _rec("J.5", J)
    m = J["J.5"]
    assert m["measured_behavior_adoption_pvalue"] <= 0.05
    assert m["kept_good"] / max(1, m["kept_good"] + m["dropped_good"]) > \
        m["kept_garbage"] / max(1, m["kept_garbage"] + m["dropped_garbage"])


def test_J6_genuine_transactions(J):
    _rec("J.6", J)
    assert J["J.6"]["independent_transactions"] >= 1


def test_J7_deployment_changes_world_resource_consumption(J):
    _rec("J.7", J)
    assert J["J.7"]["resource_consumption_delta"] > 0


def test_Jn1_more_than_one_product_category(J):
    _rec("J.n1", J)
    assert J["J.n1"]["product_categories"] >= 2
