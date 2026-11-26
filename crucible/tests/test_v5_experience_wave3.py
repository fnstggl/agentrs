"""Workstream L, wave 3: the nested-call leak fix, the registry additions found by the adversarial probes, Tier 1
self-experiments, the verified-label channel and the per-tier probe helpers (fast unit tests; the gates measure the
same on the real tiers)."""
import json

import pytest

import crucible
from crucible.core import evaluator_only as EO
from crucible.experience import make
from crucible.experience.modes import LEGACY, OBSERVATION_ONLY

pytestmark = pytest.mark.timeout(900)


def _facility(mode):
    from crucible.experience.kernel import KernelExperienceEnv
    from crucible.research.tasks import make_task
    from crucible.worldgen import generate
    from crucible.worldgen import mech_families  # noqa: F401
    crucible.load_all_modules()
    t = make_task("tech_pipeline", 3, validate=False)
    cfg, seed = t.world_config()
    env = KernelExperienceEnv(generate(cfg, seed), agents=["agent"], endowment=t.endowment, reward_mode=mode,
                              extra_modules=["wf"])
    a = env.agents()[0]
    lv = env.execute(a, {"tool": "rig.spec", "args": {}})["result"]["levers"][0]
    env.execute(a, {"tool": "rig.set", "args": {"settings": {lv["name"]: lv["max"]}}})
    env.advance(3 * 3600)
    return env, a


def test_deferred_workflow_jobs_get_participant_views_of_results():
    """Before wave 3 a deferred job (run by the wf process as the participant, origin = system) returned the
    evaluator-only `score` that a direct rig.read strips; the kernel now filters every call made AS a participant."""
    for mode, hidden in ((OBSERVATION_ONLY, True), (LEGACY, False)):
        env, a = _facility(mode)
        ex = lambda tool, **args: env.execute(a, {"tool": tool, "args": args})  # noqa: E731
        fid = ex("wf.define", name="peek", steps=[{"id": "s1", "tool": "rig.read", "args": {}}])["result"]["flow"]
        assert ex("wf.run_later", flow=fid, in_s=60)["ok"]
        env.advance(3600)
        jobs = json.dumps(ex("wf.jobs")["result"])
        evs = json.dumps([e["payload"] for e in env.observe_since(a, 0)[0] if e["type"] == "wf.job_done"])
        assert ('"score"' in jobs) == (not hidden) and ('"score"' in evs) == (not hidden), mode
        # evaluator tools are refused through every nested channel
        for tool in sorted(EO.TOOLS):
            d = ex("wf.define", name=f"x {tool}", steps=[{"id": "s", "tool": tool, "args": {}}])
            if d["ok"] and hidden:
                assert not ex("wf.run", flow=d["result"]["flow"])["ok"]


def test_registry_additions_sci_verdicts_and_metric_union():
    w = make(0, "tiny", 1, agents=["learner"]).world
    rep = {"report": "srep_1", "verdict": "consistent", "p": 0.4, "n": 7, "independent": True}
    paper = {"paper": "paper_1", "replications": {"consistent": 2, "refuted": 0, "independent_consistent": 1},
             "reports": [{"id": "srep_1", "verdict": "consistent", "related_to_author": False, "independent": True,
                          "other_apparatus": True}]}
    assert EO.participant_result(w, "sci.replicate", rep) == {"report": "srep_1", "verdict": "consistent", "p": 0.4,
                                                             "n": 7}
    got = EO.participant_result(w, "sci.paper", paper)
    assert got["replications"] == {"consistent": 2, "refuted": 0}
    assert got["reports"] == [{"id": "srep_1", "verdict": "consistent", "other_apparatus": True}]
    ev = EO.participant_payload(w, "sci.replication", {"id": "s", "verdict": "refuted", "related_to_author": True,
                                                       "independent": False})
    assert ev == {"id": "s", "verdict": "refuted"}
    ms = EO.evaluator_metrics()
    from crucible.commerce.observables import EVALUATOR_ONLY
    from crucible.goals.metrics import EVALUATOR_ONLY_METRICS
    assert set(EVALUATOR_ONLY) <= set(ms) and set(EVALUATOR_ONLY_METRICS) <= set(ms)
    assert EO.metric_errors(w, [{"metric": "scheduled_churn_share"}])
    lw = make(0, "tiny", 1, LEGACY, agents=["learner"]).world
    assert EO.participant_result(lw, "sci.replicate", rep) == rep and not EO.metric_errors(
        lw, [{"metric": "scheduled_churn_share"}])


@pytest.mark.parametrize("name,kw", [("mech:scm_additive", {"validate": False}), ("rapid:net_routing", {}),
                                     ("rapid:team_commons", {})])
def test_tier1_self_experiments_resolve_privately_and_pay_nothing(name, kw):
    env = make(1, name, 3, **kw)
    a = env.agents()[0]
    names = {t["name"] for t in env.tools(a)}
    assert {"exp.register", "exp.result", "exp.mine"} <= names
    bad = env.execute(a, {"tool": "exp.register", "args": {"hypothesis": "h", "measure": {"number": "no.such"},
                                                          "prediction": {"op": "increase"}, "deadline_in_s": 60}})
    assert not bad["ok"] and bad["error"]["code"] == "invalid_action"
    r = env.execute(a, {"tool": "exp.register", "args": {
        "hypothesis": "time passes", "measure": {"number": "world.time_s"},
        "prediction": {"op": "increase", "min_change": 1}, "deadline_in_s": 2 * env.tick_s}})
    assert r["ok"] and isinstance(r["action_seq"], int)
    rewards = []
    for _ in range(3):
        env.advance(env.tick_s)
        rewards += [env.reward(b) for b in env.agents()]
    mine = [e for e in env.observe_since(a, 0)[0] if e["type"] == "exp.resolved"]
    assert len(mine) == 1 and mine[0]["payload"]["held"] is True and mine[0]["cause"]["action_seq"] == r["action_seq"]
    assert mine[0]["cause"]["delay_s"] == 2 * env.tick_s
    for other in env.agents()[1:]:
        assert not [e for e in env.observe_since(other, 0)[0] if e["type"] == "exp.resolved"]
    assert all(x == 0.0 for x in rewards)
    res = env.execute(a, {"tool": "exp.result", "args": {"experiment": r["result"]["experiment"]}})
    assert res["ok"] and res["result"]["status"] == "resolved"


def test_label_channel_needs_authority_and_leaves_the_learner_view_unchanged(tmp_path):
    from crucible.experience.labels import LabelChannel, LabelChannelError
    with pytest.raises(LabelChannelError):
        LabelChannel(make(1, "rapid:net_routing", 1), authorised_by="", purpose="evaluation")
    with pytest.raises(LabelChannelError):
        LabelChannel(make(1, "rapid:net_routing", 1), authorised_by="op", purpose="shaping")
    views = []
    for use in (False, True):
        env = make(1, "rapid:net_routing", 2)
        a = env.agents()[0]
        ch = LabelChannel(env, authorised_by="op", purpose="evaluation", path=str(tmp_path / "l.jsonl")) if use else None
        v = []
        for _ in range(4):
            env.advance(env.tick_s)
            v.append(env.observe(a))
            if ch:
                ch.issue()
        views.append(json.dumps(v, sort_keys=True, default=str))
        assert env.reward(a) == 0.0
    assert views[0] == views[1]
    recs = LabelChannel.read(str(tmp_path / "l.jsonl"))
    assert len(recs) == 4 and LabelChannel.verify_chain(recs)
    recs[1]["outcome"]["success"] = not recs[1]["outcome"]["success"]
    assert not LabelChannel.verify_chain(recs)


def test_probe_helpers():
    from crucible.experience import probes
    from crucible.experience.probes_tiers import harvest, harvest_fill, identity_tokens
    assert probes._key_hits({"objective": {"a0": "maximise", "a1": "maximise"}}) == []
    assert probes._key_hits({"objective": {"success": True, "progress": 0.3}})
    assert probes._key_hits({"x": {"related_to_author": False}})
    ids = harvest([{"post": "fpost_1", "n": 3, "body": "long text with spaces"}], {"principal": ["p:a"]})
    assert ids["post"] == ["fpost_1"] and "body" not in ids
    fill = harvest_fill(ids)
    assert fill({"inputSchema": {"required": ["post"], "properties": {"post": {"type": "string"}}}}) == {"post": "fpost_1"}
    assert fill({"inputSchema": {"required": ["repo"], "properties": {}}}) is None
    env = make(0, "tiny", 918_273, agents=["learner"])
    toks, nums = identity_tokens(env)
    assert env.world.spec.hash()[:12] in toks and nums == {"seed": 918_273.0}


def test_capability_registry_adoption_verdicts_are_evaluator_only():
    """caps.view/list/graph/evidence carried counts and flags computed with the relatedness analysis
    (independent adopters/payers/buyers): a provider could learn which of its own organisations the evaluator links to
    it by having them install its capability. Installs and active installs (world facts) stay."""
    w = make(0, "tiny", 1, agents=["learner"]).world
    summ = {"capability": "cap_1", "installs": 3, "active_installs": 2, "independent_adopters": 1,
            "independent_payers": 0, "public_outcomes": [{"evidence": "e1", "by": "o:x", "independent": False}]}
    got = EO.participant_result(w, "caps.view", summ)
    assert got == {"capability": "cap_1", "installs": 3, "active_installs": 2,
                   "public_outcomes": [{"evidence": "e1", "by": "o:x"}]}
    assert EO.participant_result(w, "caps.list", [summ])[0] == got
    g = EO.participant_result(w, "caps.graph", {"nodes": [], "edges": [], "markets": [
        {"id": "m1", "label": "sort", "sellers": ["o:a"], "buyers": ["o:b", "o:c"]}]})
    assert g["markets"] == [{"id": "m1", "label": "sort", "sellers": ["o:a"]}]
    assert EO.participant_payload(w, "caps.installed", {"id": "i1", "installer": "o:b", "independent": True}) == \
        {"id": "i1", "installer": "o:b"}
    assert EO.participant_payload(w, "caps.market", {"id": "m1", "buyers": ["o:b"]}) == {"id": "m1"}
    assert EO.is_evaluator_tool(w, "caps.categories")
    lw = make(0, "tiny", 1, LEGACY, agents=["learner"]).world
    assert EO.participant_result(lw, "caps.view", summ) == summ
