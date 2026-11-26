"""Unit and integration tests of the scientific-research domain (9.5): the `sci` repairs (replication statistic with
stated absolute + relative precision and propagated calibration uncertainty, the paper/replication endpoints, the
standards bureau's novel-condition challenges) and the population researchers of `research_domain`."""
import re

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible import experience  # noqa: E402
from crucible.core.kernel import World, WorldSpec  # noqa: E402
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import lawlang  # noqa: E402
from crucible.domains import research_domain as RD  # noqa: E402

DAY = 86400


def _sci_world(seed, fam, *, replicators=4, challenge=True):
    rng = np.random.default_rng(seed)
    xs = lawlang.draw_variables(rng, 2)
    law = lawlang.LAW_FAMILIES[fam](rng, xs)
    ph = {"id": "ph0", "title": "t", "law": law["law"], "noise_rel": 0.02,
          "vars": [{"name": xs[0], "lo": 0.0, "hi": 14.0}, {"name": xs[1], "lo": 0.0, "hi": 7.0}],
          "standards": [{"id": "s1", "at": {xs[0]: 2.0, xs[1]: 1.0}}, {"id": "s2", "at": {xs[0]: 6.0, xs[1]: 3.0}}]}
    rngs = {xs[0]: [0.0, 10.0], xs[1]: [0.0, 5.0]}
    apps = [{"id": "a", "phenomenon": "ph0", "operator": "instr", "range": rngs, "price_cents": 400, "sd": 0.3,
             "capacity_per_day": 60, "gain": 1.03, "offset": 0.2},
            {"id": "b", "phenomenon": "ph0", "operator": "instr2", "range": rngs, "price_cents": 650, "sd": 0.2,
             "capacity_per_day": 40, "gain": 0.98, "offset": -0.1}]
    cfg = {"phenomena": [ph], "apparatus": apps,
           "operators": [{"handle": "instr", "name": "I", "cash": 1000}, {"handle": "instr2", "name": "P", "cash": 1000}],
           "rivals": {"n": 0}, "replicators": {"n": replicators, "points": 16, "calib_reps": 3}}
    if challenge:
        cfg["challenge"] = {"every_days": 10, "start_day": 5}
    mods = [{"name": m, "config": {}} for m in ["identity", "bank", "admin", "artifacts", "docs"]]
    mods.append({"name": "sci", "config": cfg})
    w = World.create(WorldSpec("sci_test", seed, mods, {"cost_model": "deterministic"}))
    me = w.system_call("admin.join", {"handle": "me", "endowment": 5_000_000})["principal"]
    return w, me, law, xs


def _do(w, me, tool, **a):
    r = w.execute(me, {"tool": tool, "args": a})
    assert r.ok, (tool, r.error)
    return r.result


def _publish_variants(w, me, law, xs):
    grid = [{"at": {xs[0]: float(a), xs[1]: float(b)}, "n": 1} for a in np.linspace(0, 10, 6)
            for b in np.linspace(0, 5, 5)]
    r1 = _do(w, me, "sci.run", apparatus="a", design=grid)
    c1 = _do(w, me, "sci.run", apparatus="a", design=[{"standard": "s1", "n": 3}, {"standard": "s2", "n": 3}])
    w.advance_time(w.now + 3 * DAY)
    scaled = re.sub(r"(\d+\.\d+)", lambda m: repr(round(float(m.group(1)) * 1.10, 6)), law["law"])
    out = {}
    for k, m in (("true", law["law"]), ("x1.10", scaled)):
        out[k] = _do(w, me, "sci.publish", phenomenon="ph0", title=k, model=m, data=[r1["run"]],
                     calibration_runs=[c1["run"]])
    return out


def test_paper_and_replication_endpoints_and_calibrated_verdicts():
    """Regression (DEF-I-5): sci.paper and sci.replications work; the true law is consistent in independent
    replications (relative noise and calibration uncertainty are part of the statistic) while a law with every
    constant 10% off is refuted; a reading of the true law's own fit is not a refutation."""
    tally = {"true": [0, 0], "x1.10": [0, 0]}
    own_ok = 0
    for seed, fam in ((0, "sat_lin"), (1, "peak"), (2, "power_exp")):
        w, me, law, xs = _sci_world(seed, fam)
        pubs = _publish_variants(w, me, law, xs)
        own_ok += int(pubs["true"]["fit"]["p"] >= 0.01)
        w.advance_time(w.now + 35 * DAY)
        for k, p in pubs.items():
            paper = _do(w, me, "sci.paper", paper=p["paper"])
            assert "reports" in paper and all("id" in r for r in paper["reports"])
            tally[k][0] += paper["replications"]["consistent"]
            tally[k][1] += paper["replications"]["refuted"]
        reps = _do(w, me, "sci.replications")
        assert reps and all(r["id"].startswith("srep") for r in reps)
        assert not w.check_invariants()
    c, r = tally["true"]
    assert c + r >= 6 and c / (c + r) >= 0.9, tally
    c, r = tally["x1.10"]
    assert c + r >= 6 and r / (c + r) >= 0.8, tally
    assert own_ok == 3


def test_challenges_measure_outside_every_apparatus_range_and_test_papers():
    w, me, law, xs = _sci_world(4, "sat_lin", replicators=0)
    pubs = _publish_variants(w, me, law, xs)
    w.advance_time(w.now + 25 * DAY)
    ch = _do(w, me, "sci.challenges")
    assert len(ch) >= 2
    for c in ch:
        assert len(c["readings"]) == 8
        for row in c["readings"]:
            at = row["at"]
            assert at[xs[0]] > 10.0 or at[xs[1]] > 5.0, at          # beyond every bookable range
            assert at[xs[0]] <= 14.0 and at[xs[1]] <= 7.0
        verdict = {r["paper"]: r["verdict"] for r in c["results"]}
        assert verdict[pubs["true"]["paper"]] == "passed" and verdict[pubs["x1.10"]["paper"]] == "failed"
    # the bureau's readings are a public dataset
    assert all(w.state.tables["artifacts"][c["dataset"]]["published"] for c in ch)


def test_apparatus_datasheet_states_relative_precision():
    w, me, law, xs = _sci_world(5, "peak", replicators=0, challenge=False)
    apps = {a["apparatus"]: a for a in _do(w, me, "sci.apparatus")}
    assert apps["a"]["stated_rel"] == pytest.approx(0.02) and apps["a"]["stated_sd"] == pytest.approx(0.3)


def test_research_world_population_researchers_act_with_provenance_and_no_law_leak():
    env = experience.make(2, "research:bench", 3, semantic=False)
    assert env.reward_mode == "observation_only" and env.agents() == ["p:researcher"]
    w = env.world
    K.drive(env, env.agents()[0], RD.null_policy(), days=16, dt=6 * 3600)
    st = w.state
    assert len(st.tables["scipop_r"]) >= 10
    authors = {p["author"] for _, p in st.scan("sci_papers")}
    assert authors and all(K.is_population(a) for a in authors)
    prov = K.decision_provenance(w, ("scipop.",))
    assert prov["batches"] > 0 and prov["forced"] == 0
    law = w.modules["sci"]._phen_cfg("ph0")["law"]
    assert not any(law in str(e.payload) for e in w.events_range(0, w.seq) if e.vis == "public")
    obs = env.observe("p:researcher")
    assert law not in str(obs)
    # researchers paid for their own measurements
    m = RD.mechanics(w)
    paid = [e for e in w.events_range(0, w.seq) if e.type == "bank.txn" and e.payload["kind"] == "experiment"
            and any(K.is_population(w.state.tables["accounts"][a]["owner"]) for a, amt in e.payload["postings"]
                    if amt < 0)]
    assert len(paid) >= 3 and all(m[k]["exercised"] for k in ("Experimental costs", "Data collection"))
    assert not w.check_invariants()
    ev = RD.DOMAIN.evaluate(w, "p:researcher", RD.DOMAIN.objective(w, "p:researcher", 1))
    assert ev["success"] is False and ev["details"]["papers"] == []          # the null baseline fails


def test_configurations_are_structurally_different_worlds():
    from crucible.domains import base
    seen = {}
    for c in RD.DOMAIN.configs:
        sp = base.spec("research", c.name, 0)
        sci = next(m["config"] for m in sp.modules if m["name"] == "sci")
        sig = (len(sci["phenomena"]), len(sci["phenomena"][0]["vars"]),
               len({tuple(a["range"][sorted(a["range"])[0]]) for a in sci["apparatus"] if a["phenomenon"] == "ph0"}),
               any("drift_per_measure" in a for a in sci["apparatus"]),
               any("stated_rel" in a for a in sci["apparatus"]), len(sci["phenomena"][0]["standards"]))
        seen[c.name] = sig
    assert len(set(seen.values())) == len(seen), seen
    held = base.spec("research", "heldout", 0)
    law = next(m["config"] for m in held.modules if m["name"] == "sci")["phenomena"][0]["law"]
    train_forms = {f for c in ("bench", "drift", "survey") for f in RD.STRUCTURES[c]["law_forms"]}
    assert not set(RD.STRUCTURES["heldout"]["law_forms"]) & train_forms
    assert "sigmoid" in law or "cos" in law or "/ (1 + (" in law


def test_degenerate_strategies_fail():
    """SP-9.16 probes of this domain: a constant model and a copied model never satisfy the evaluator."""
    from crucible.domains import base
    for probe in ("constant_model", "copy_model"):
        r = base.run("research", "bench", 0, policy=probe, days=14)
        assert r["success"] is False, (probe, r["outcome"])
        assert r["drive"]["agents"]["p:researcher"]["ok"] >= 3
    r = base.run("research", "bench", 0, policy="copy_model", days=14)
    assert any(p["copied"] for p in r["outcome"]["details"]["papers"]), r["outcome"]


def test_reference_publishes_an_accurate_law_and_null_fails():
    """LABELLED SCRIPTED reference: feasibility on one seed (its paper predicts novel conditions within the
    difficulty-1 threshold); the evaluator is separate and the observation never carries its verdict."""
    env = experience.make(2, "research:bench", 0, semantic=False)
    a = env.agents()[0]
    obj = RD.DOMAIN.objective(env.world, a, 1)
    pol = RD.reference(a, obj)
    assert pol.kind == "scripted"
    K.drive(env, a, pol, days=6, dt=6 * 3600)
    ev = RD.DOMAIN.evaluate(env.world, a, obj)
    assert ev["details"]["papers"] and ev["details"]["papers"][0]["novel_nrmse"] <= RD.THRESHOLD[1]
    assert not K.observation_leaks(env.observe(a), ("novel_nrmse", "success", "score"))
