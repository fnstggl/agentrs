"""Workstream L (wave 3b repair pass): regression tests for the defects an independent verifier found on c5c/w3b-obs
at 507adb24 (R.4 sensitivity, seat controller labels, evaluation-script guard, ...)."""
import pytest

from crucible.core import evaluator_only as EO
from crucible.experience import leakage

pytestmark = pytest.mark.timeout(900)

LEGIT = ("amount", "memo", "title", "body")


class _OverHide(dict):
    """An injected defect: the evaluator-only registry hides legitimate payload fields of every event type."""
    def get(self, k, d=None):
        return tuple(dict.get(self, k, ())) + LEGIT


def _missing(res):
    return {r["test"]: int(r["metrics"].get("legitimate_fields_missing", 0)) for r in res
            if "legitimate_fields_missing" in r["metrics"]}


def test_the_reviewed_registry_covers_every_hidden_field():
    assert leakage.unreviewed_hidden_fields() == []
    for (kind, name), fields in leakage.REVIEWED_EVALUATOR_FIELDS.items():
        assert kind in ("event", "result", "summary") and fields and all(fields.values()), (kind, name)


def test_a_field_hidden_without_review_is_named(monkeypatch):
    monkeypatch.setitem(EO.EVENT_FIELDS, "bank.transfer", ("amount",))
    monkeypatch.setitem(EO.RESULT_FIELDS, "forum.post", ("title",))
    assert leakage.unreviewed_hidden_fields() == ["event:bank.transfer.amount", "result:forum.post.title"]


def test_over_hiding_legitimate_fields_is_counted_missing_by_every_legacy_comparison(monkeypatch):
    """The verifier's probe p6: with the registry wrongly hiding amount / memo / title / body, the Gym, PettingZoo,
    kernel-invariance and WorldEnv comparisons each count missing legitimate fields (before the fix only WorldEnv
    did: the legacy side was stripped of every registry field before the comparison)."""
    tests = (leakage.test_gym, leakage.test_pettingzoo, leakage.test_worldenv, leakage.test_kernel_invariance)
    base = _missing(leakage.run_battery(tests))
    assert all(v == 0 for v in base.values()), base
    monkeypatch.setattr(EO, "EVENT_FIELDS", _OverHide(EO.EVENT_FIELDS))
    over = _missing(leakage.run_battery(tests))
    assert all(over[t.__name__] > 0 for t in tests), over
    # a reviewed field (the controller label) is still normalised: hiding only it leaves nothing missing
    monkeypatch.setattr(EO, "EVENT_FIELDS", dict(EO.EVENT_FIELDS.items()))
    assert all(v == 0 for v in _missing(leakage.run_battery(tests)).values())
