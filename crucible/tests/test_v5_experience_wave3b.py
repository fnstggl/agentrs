"""Workstream L (wave 3b): leak re-probes of the modules merged after wave 3a (social internet, population,
semantic participants, capability registry, domains) and the registry additions they needed."""
import pytest

from crucible.core import evaluator_only as EO
from crucible.experience import make

pytestmark = pytest.mark.timeout(900)


def test_metrics_documented_evaluator_only_are_in_the_registry_union():
    """crucible.social.integrity's organic_* metrics declare themselves EVALUATOR-ONLY (engagement netted by the
    relatedness / funding / sybil / ring analysis); they were kept out of agreements only by having no subject and
    were missing from `evaluator_metrics()`. Every metric documented that way is now evaluator-only."""
    import crucible.social.integrity  # noqa: F401 - registers organic_followers / organic_engagement
    from crucible.goals.metrics import EVALUATOR_ONLY_METRICS, METRIC_DOCS, check_predicate_scope
    documented = {m for m, d in METRIC_DOCS.items() if str(d).startswith("EVALUATOR-ONLY")}
    assert {"organic_followers", "organic_engagement", "quality_margin_vs_incumbents"} <= documented
    assert documented <= EVALUATOR_ONLY_METRICS and documented <= set(EO.evaluator_metrics())
    env = make(0, "tiny", 3, agents=["learner"])                       # observation_only: evaluator data hidden
    a = env.agents()[0]
    leaf = {"metric": "organic_followers", "args": {"principal": a}, "op": ">=", "value": 1}
    errs = check_predicate_scope(env.world, leaf, [a])
    assert any("evaluator" in e for e in errs), errs
    assert EO.metric_errors(env.world, [leaf])
