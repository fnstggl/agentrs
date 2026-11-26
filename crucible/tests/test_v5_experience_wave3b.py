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
    from crucible.goals.metrics import DECLARED_EVALUATOR_ONLY, METRIC_DOCS, check_predicate_scope
    documented = {m for m, d in METRIC_DOCS.items() if str(d).startswith("EVALUATOR-ONLY")}
    assert {"organic_followers", "organic_engagement", "quality_margin_vs_incumbents"} <= documented
    assert documented <= DECLARED_EVALUATOR_ONLY and documented <= set(EO.evaluator_metrics())
    env = make(0, "tiny", 3, agents=["learner"])                       # observation_only: evaluator data hidden
    a = env.agents()[0]
    leaf = {"metric": "organic_followers", "args": {"principal": a}, "op": ">=", "value": 1}
    errs = check_predicate_scope(env.world, leaf, [a])
    assert any("evaluator" in e for e in errs), errs
    assert EO.metric_errors(env.world, [leaf])
    legacy = make(0, "tiny", 3, "legacy_shaped_reward", agents=["learner"])      # the 4.0 contract is unchanged
    assert EO.metric_errors(legacy.world, [leaf]) == []


def test_population_private_state_probe_finds_a_planted_value_and_nothing_else():
    """The Tier 3 / Tier 2 probe of the population's private state (individuals' interests, satisfaction, habits,
    ...): nothing in what a seat reaches in a domain world with the population, and a positive control - one
    individual's private value planted in a public event - is found."""
    from crucible.experience import probes_tiers as PT
    from crucible.experience.registry import tier2_domains
    name = next((n for n, d in sorted(tier2_domains().items())), None)
    assert name is not None, "no Part 9 domain registered: the probe needs a world with the population"
    env = make(2, name, 3)
    a = env.agents()[0]
    assert "pop" in env.world.modules
    env.advance(6 * 3600)
    vals = PT.population_private_values(env.world)
    assert len(vals) >= 50
    reach = [env.observe(a), env.tools(a), env.observe_since(a, 0)]
    P = PT.TierProbe("population private", 2)
    PT.probe_population_private(P, env, a, reach, "clean")
    assert P.result()["n_leaks"] == 0 and P.kinds["exact_value"]["checked"] == len(vals)
    key, v = sorted(vals.items())[len(vals) // 2]
    env.world._run_system("plant", lambda ctx: ctx.emit("note", {"stat": v}, vis="public"))
    reach = [env.observe(a), env.observe_since(a, 0)]
    Q = PT.TierProbe("population private (planted)", 2)
    PT.probe_population_private(Q, env, a, reach, "planted")
    assert Q.kinds["exact_value"]["leaks"] >= 1, (key, v)


def test_fixture_domains_never_count_as_tier2_but_build_by_name():
    """A test fixture registered in the same process (tests/v5/_domain_fixture.py, `fixture=True`) is buildable through
    make(2, ...) but is never one of the Tier 2 domains the gates count; Tier 2 worlds carry the learner-side modules."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v5"))
    import _domain_fixture
    from crucible.domains import base
    from crucible.experience.registry import domains, tier2_domains
    _domain_fixture.register()
    try:
        assert "fixture_retail" in domains() and "fixture_retail" not in tier2_domains()
        env = make(2, "fixture_retail:ring_standard", 0)
        assert env.tier == 2 and {"exp", "goals"} <= set(env.world.modules)
        names = {t["name"] for t in env.tools(env.agents()[0])}
        assert "exp.register" in names and "goals.check_now" not in names
    finally:
        base.unregister("fixture_retail")


def test_hidden_mechanism_probe_finds_a_planted_law_and_nothing_else():
    """A Tier 2 science world's hidden response law (`sci.phenomena[].law`, what a learner must discover by
    experiment) is looked for verbatim and by its distinctive coefficients in everything the seat reaches: nothing
    in an ordinary run; a positive control - the law (or one coefficient) planted in reached material - is found. The
    isolation probe's hidden values include the law too."""
    from crucible.experience import isolation as I
    from crucible.experience import probes_tiers as PT
    from crucible.experience.registry import tier2_domains
    import crucible.domains.base as DB
    found = None
    for n, d in sorted(tier2_domains().items()):
        if d.mode == "B":                                            # cheap standalone worlds first
            env = make(2, n, 3)
            exprs, coefs = PT.hidden_mechanisms(env.world)
            if exprs and coefs:
                found = (n, env, exprs, coefs)
                break
    assert found, "no Part 9 domain world with a hidden response law (research / commercialization) is registered"
    n, env, exprs, coefs = found
    a = env.agents()[0]
    dom = DB.REGISTRY[env.domain]
    DB.drive(env, {a: dom.reference_policy(a, env._objectives[a].objective)}, days=3, dt=int(dom.dt), trace=False)
    reach = [env.observe(a), env.observe_since(a, 0), env.tools(a)]
    P = PT.TierProbe("clean", 2)
    P.scan_channel("observation", reach, values=coefs, canaries=exprs, label=n)
    assert P.result()["n_leaks"] == 0, P.result()["leaks"]
    planted = PT.TierProbe("planted law", 2)
    planted.scan_channel("observation", reach + [{"text": f"the cost follows {exprs[0]}"}], values=coefs,
                         canaries=exprs, label=n)
    assert {l["kind"] for l in planted.result()["leaks"]} >= {"canary"}
    k, v = next(iter(coefs.items()))
    one = PT.TierProbe("planted coefficient", 2)
    one.scan_channel("observation", reach + [{"fit": {"k": v}}], values=coefs, canaries=exprs, label=n)
    assert any(l["kind"] == "exact_value" for l in one.result()["leaks"])
    assert any(str(x).startswith("1.0 + ") or "*" in str(x) for x in I._hidden_of(env).values())
