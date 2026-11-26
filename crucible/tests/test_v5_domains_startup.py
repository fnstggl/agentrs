"""Unit and short integration tests of the startup-economy domain (9.1): built structure, reachability, the
population's demand, the evaluator's independence rules and the degenerate strategies (short horizons)."""
import crucible

crucible.load_all_modules()
from crucible.domains import base  # noqa: E402
from crucible.domains import startup as D  # noqa: E402


def test_configurations_are_structurally_distinct_civ_slices():
    rep = base.structure_report(D.DOMAIN, (0, 1))
    assert rep["train_structures"] == 3 and rep["compositional_eval"] == 1 and rep["structural_eval"] == 1
    assert not rep["errors"] and all(rep["stable_across_seeds"].values())
    mods = {c.name: {m["name"] for m in base.spec("startup", c.name, 0).modules} for c in D.DOMAIN.configs}
    assert "rivals" not in mods["static_market"] and "ads" not in mods["pipeline_monopoly"]
    assert {"pop", "work", "subs", "products", "ventures", "seat"} <= mods["two_formats"]
    assert "patrons" not in set().union(*mods.values())          # no second population (D-15)


def test_reference_launches_and_the_population_tries_it():
    """LABELLED SCRIPTED reference, 16 days: a launched product, trials by population individuals, the evaluator
    sees independent customers; the observation never carries the verdict."""
    r = base.run("startup", "two_formats", 0, policy="reference", days=16, keep_env=True)
    env = r["env"]
    w = env.world
    assert r["proc_failed"] == 0 and r["invariant_violations"] == 0
    assert r["reward_mode"] == "observation_only" and env.reward(r["seats"][0]) == 0.0
    subs = [s for _, s in w.state.scan("sub_subs") if s["merchant"] == "o:founder-co"]
    assert subs and all(s["customer"].startswith("p") and s["customer"][1:].isdigit() for s in subs)
    obs = env.observe(r["seats"][0])
    assert "arr_usd" not in str(obs) and "independent_customers" not in str(obs)
    assert r["mechanics"]["counts"]["Product creation"] > 0 and r["mechanics"]["counts"]["Independent customers"] > 0
    env.close()


def test_self_dealing_and_null_fail():
    for pol in ("self_dealing", "null"):
        r = base.run("startup", "two_formats", 0, policy=pol, days=4)
        assert r["success"] is False, (pol, r["outcome"])
    r = base.run("startup", "two_formats", 0, policy="self_dealing", days=4)
    orgs = r["outcome"]["details"]["orgs"]
    assert orgs and all(o["independent_customers"] == 0 for o in orgs.values())
