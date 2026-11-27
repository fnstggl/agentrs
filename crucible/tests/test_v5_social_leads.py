"""Unit test (workstream C, wave 3b): a member who read about a service for a task of its own work looks into it
(a `shop` session of economy.work) with a probability that depends on what it uses for that task now - its own
adoption records - instead of every relevant reader doing so (social.popinbox.lead_probability)."""
from crucible.social import popinbox as PI


def test_looking_into_a_service_depends_on_what_one_uses_now():
    none = PI.lead_probability([])
    poor = PI.lead_probability([{"uses": 4, "ok": 1}])
    new = PI.lead_probability([{"uses": 1, "ok": 1}])          # too few uses to know it works
    working = PI.lead_probability([{"uses": 6, "ok": 5}])
    mixed = PI.lead_probability([{"uses": 5, "ok": 1}, {"uses": 3, "ok": 3}])
    assert none > poor > working > 0.0
    assert new == poor and mixed == working
    assert none < 1.0                                         # nobody is sent shopping by a read alone
