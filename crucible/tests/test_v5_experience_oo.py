"""Workstream L (recovery): observation-only defaults, kernel-enforced evaluator-only quantities, the multi-agent rapid
adapter, the Tier 2 path against `crucible.domains.base`, Gymnasium/PettingZoo wrappers over `ExperienceEnv`, the
process-isolated evaluator, the experience store hardening and the reference learner's fixes."""
import json
import os

import numpy as np
import pytest

import crucible
from crucible.core import evaluator_only as EO
from crucible.experience import ExperienceStore, make, validate_observation
from crucible.experience.modes import LEGACY, OBSERVATION_ONLY, VERIFIED, check_mode
from crucible.experience.store import StoreCorrupt, StoreLocked

pytestmark = pytest.mark.timeout(900)
DAY = 86400


# ---------------------------------------------------------------------------------------------- defaults
def test_observation_only_is_the_default_and_legacy_needs_naming():
    assert check_mode(None) == OBSERVATION_ONLY
    from crucible.api.gym_env import CrucibleEnv
    from crucible.experience.reference import ReferenceLearner
    from crucible.research.envs import MechVecEnv
    from crucible.research.tasks import make_task
    from crucible.selfplay.arena import SelfPlayEnv
    t = make_task("scm_additive", 3, validate=False)
    assert MechVecEnv([t]).reward_mode == OBSERVATION_ONLY
    assert SelfPlayEnv([make_task("abm_duopoly", 2, validate=False)]).reward_mode == OBSERVATION_ONLY
    assert ReferenceLearner().reward_mode == OBSERVATION_ONLY
    assert make(1, "rapid:net_routing", 0).reward_mode == OBSERVATION_ONLY
    env = make(0, "tiny", 0, agents=["learner"])
    assert env.reward_mode == OBSERVATION_ONLY and EO.hidden(env.world)
    assert not EO.hidden(make(0, "tiny", 0, LEGACY, agents=["learner"]).world)
    g = CrucibleEnv(spec={"family": "tiny", "overrides": {"n_consumers": 50}}, seed=1, endowment=1000)
    assert g.reward_mode == OBSERVATION_ONLY and EO.hidden(g.world)
    g.close()


# ---------------------------------------------------------------------------------------------- evaluator-only
def _org_world(mode):
    env = make(0, "tiny", 4, mode, agents=["learner"], extra_modules=["inst", "commit", "use"],
               overrides={"n_consumers": 50})
    a = env.agents()[0]

    def call(tool, **args):
        r = env.execute(a, {"tool": tool, "args": args})
        assert r["ok"], r
        return r["result"]
    me = [x for x in call("bank.accounts") if x["owner"] == a][0]["account"]
    org = call("id.create_org", handle="probeco", name="Probe Co", kind="company")["org"]
    return env, a, org, me


def test_evaluator_tools_are_hidden_and_refused_in_observation_only_and_kept_in_legacy():
    for mode, hidden in ((OBSERVATION_ONLY, True), (VERIFIED, True), (LEGACY, False)):
        env, a, org, me = _org_world(mode)
        names = {t["name"] for t in env.tools(a)}
        for tool in ("goals.check_now", "use.report", "use.probe"):
            assert (tool not in names) == hidden, (mode, tool)
        r = env.execute(a, {"tool": "goals.check_now", "args": {"goal": "sg_000001"}})
        assert (r["error"]["code"] == "forbidden") == hidden, (mode, r)
        d = env.execute(a, {"tool": "world.describe", "args": {"host": "goals"}})["result"]
        assert ("goals.check_now" in {e["name"] for e in d["endpoints"]}) != hidden
        pred = {"metric": "independent_revenue", "args": {"principal": org, "window_s": DAY}, "op": ">=", "value": 1}
        r = env.execute(a, {"tool": "goals.propose", "args": {"title": "t", "predicate": pred, "due_in_s": 3600}})
        assert r["ok"] != hidden, (mode, r)
        charter = {"procedures": {"vote": {"type": "majority", "electorate": "members"}},
                   "rules": [{"match": {"kind": "tool", "tool": "bank.transfer", "when": pred}, "procedure": "vote"}]}
        r = env.execute(a, {"tool": "inst.adopt", "args": {"org": org, "charter": charter}})
        assert (not r["ok"] and r["error"]["code"] == "forbidden") == hidden, (mode, r)
        # observable metrics stay usable
        own = {"metric": "cash", "args": {"principal": a}, "op": ">=", "value": 1}
        assert env.execute(a, {"tool": "goals.propose", "args": {"title": "c", "predicate": own,
                                                                  "due_in_s": 3600}})["ok"]
        exp = env.execute(a, {"tool": "exp.register", "args": {
            "hypothesis": "h", "measure": {"tool": "goals.check_now", "args": {"goal": "x"}}, "prediction": {
                "op": "==", "value": 1}, "deadline_in_s": 3600}})
        assert (exp["error"]["code"] == "forbidden") == hidden, (mode, exp)


def test_rig_score_and_caps_verdicts_are_filtered_from_results_events_and_summaries():
    from crucible.research.tasks import make_task
    from crucible.worldgen import generate
    from crucible.experience.kernel import KernelExperienceEnv
    t = make_task("tech_pipeline", 3, validate=False)
    cfg, seed = t.world_config()
    for mode, hidden in ((OBSERVATION_ONLY, True), (LEGACY, False)):
        env = KernelExperienceEnv(generate(cfg, seed), agents=["agent"], endowment=t.endowment, reward_mode=mode)
        a = env.agents()[0]
        lv = env.execute(a, {"tool": "rig.spec", "args": {}})["result"]["levers"][0]
        env.execute(a, {"tool": "rig.set", "args": {"settings": {lv["name"]: lv["max"]}}})
        env.advance(3 * 3600)
        rd = env.execute(a, {"tool": "rig.read", "args": {}})["result"]["report"]
        hist = env.execute(a, {"tool": "rig.history", "args": {"n": 2}})["result"]["reports"]
        obs = env.observe(a)
        reports = [e for e in env.observe_since(a, 0)[0] if e["type"] == "rig.report"]
        assert reports and ("score" in reports[0]["payload"]) != hidden
        assert ("score" in rd) != hidden and all(("score" in h) != hidden for h in hist)
        assert any(k.endswith(".score") for k in obs["numbers"]) != hidden
        raw = [e for e in env.world.log if e.type == "rig.report"]
        assert raw and "score" in raw[0].payload                     # the authoritative event keeps it
    p = EO.participant_payload
    w = make(0, "tiny", 0, agents=["learner"]).world
    assert "independent" not in p(w, "caps.adopted", {"independent": True, "note": "x"})
    assert p(w, "caps.adopted", {"independent": True})["independent"] if not EO.hidden(w) else True


# ---------------------------------------------------------------------------------------------- rapid adapter
def test_rapid_multi_agent_results_visibility_units_provenance_and_joint_ticks():
    env = make(1, "rapid:team_commons", 4)
    a0, a1, a2 = env.agents()[:3]
    assert env.multi and env.tick_s == 3600
    r = env.execute(a0, {"tool": "rapid.message", "args": {"to": a2, "text": "secret plan"}})
    assert r["ok"] and r["result"]["queued"] and env.now == 0           # queued for the joint tick
    assert env.execute(a0, {"tool": "rapid.message", "args": {"to": a2, "text": "again"}})["error"]["code"] == \
        "conflict"
    for a in env.agents()[1:]:
        env.execute(a, {"tool": "rapid.query", "args": {"target": "history"}})
    assert env.now == env.tick_s                                         # every participant acted: the tick ran
    res = {a: env.observe(a)["last_result"] for a in env.agents()}
    assert res[a0]["result"] != res[a1]["result"]                       # each agent its own result
    assert res[a0]["t"] == 0 and res[a0]["cost_s"] == env.tick_s
    seen = {a: env.observe_since(a, 0)[0] for a in env.agents()}
    msgs = {a: [e for e in seen[a] if e["type"] == "message"] for a in seen}
    assert msgs[a0] and msgs[a2] and not msgs[a1]                        # the world's own visibility rule
    for a in seen:
        seqs = [e["seq"] for e in seen[a]]
        assert len(seqs) == len(set(seqs)) and seqs == sorted(seqs)       # once, in order
        assert all(e["t"] % env.tick_s == 0 for e in seen[a])             # seconds
    own = [e for e in seen[a0] if e["type"] == "action.result"]
    assert own and own[0]["cause"]["action_seq"] == r["action_seq"] and own[0]["cause"]["via"] == "direct"
    for a in env.agents():
        assert validate_observation(env.observe(a)) == []                 # incl. the receiver's message entities
    d0 = env.digest()
    env.advance(env.tick_s)                                              # the joint tick already used this time
    assert env.digest() == d0 and env.now == env.tick_s
    env.advance(env.tick_s)                                              # a wait: everyone holds, one tick passes
    assert env.digest() != d0 and env.now == 2 * env.tick_s
    single = make(1, "rapid:ch_sign_reversal", 3)
    a = single.agents()[0]
    out = single.execute(a, {"tool": "rapid.query", "args": {"target": "spec"}})
    assert out["ok"] and not ({"family", "seed", "split", "category"} & set(out["result"]["spec"]))
    single.advance(single.tick_s)                                         # the action's tick counts against it
    assert single.now == single.tick_s


# ---------------------------------------------------------------------------------------------- Tier 2 contract
@pytest.fixture
def toy_domain():
    from crucible.domains import base
    from crucible.worldgen.generator import FAMILIES

    def build(cfg, rng, ov):
        return {"modules": {"identity": {}, "bank": {}, "admin": {}, "forum": {"boards": ["general"]}},
                "params": {"toy_shape": cfg.structure["shape"]}, "module_list": ["identity", "bank", "admin", "forum"]}

    def seat(world):
        return world.system_call("admin.join", {"handle": "founder", "endowment": 50_000})["principal"]

    def objective(world, principal, difficulty):
        return {"description": f"Hold at least {40_000 + 1000 * difficulty} cents.", "target": 40_000 + 1000 * difficulty}

    def evaluate(world, principal, obj):
        from crucible.goals.metrics import evaluate_metric
        cash = evaluate_metric(world, "cash", {"principal": principal})
        return {"success": cash >= obj["target"], "score": min(1.0, cash / obj["target"])}

    cfgs = tuple(base.DomainConfig(n, s, {"shape": n}, f"toy {n}") for n, s in
                 (("a", "train"), ("b", "train"), ("c", "train"), ("d", "eval_compositional"), ("e", "eval_structural")))
    d = base.Domain("toydom", "Toy domain (test only)", "9.0", ("posting",), cfgs, build, objective, evaluate,
                    lambda w, p, o: None, lambda w: {"posting": True}, participant_principal=seat)
    base.register(d)
    yield d
    base.REGISTRY.pop("toydom", None)
    for c in cfgs:
        FAMILIES.pop(d.family(c.name), None)


def test_tier2_path_against_domains_base(toy_domain):
    from crucible.experience.registry import domain_and_config
    assert domain_and_config("toydom")[1].name == "a" and domain_and_config("d5_toydom_c")[1].name == "c"
    env = make(2, "toydom:b", 3)
    a = env.agents()[0]
    assert a == "p:founder" and env.tier == 2 and env.reward_mode == OBSERVATION_ONLY and EO.hidden(env.world)
    obs = env.observe(a)
    assert obs["objective"].startswith("Hold at least") and validate_observation(obs) == []
    acct = [x for x in env.execute(a, {"tool": "bank.accounts", "args": {}})["result"]][0]["account"]
    payee = env.world.system_call("admin.join", {"handle": "payee", "endowment": 0})["principal"]
    env.execute(a, {"tool": "bank.transfer", "args": {"from_account": acct, "to": payee, "amount": 20_000,
                                                      "memo": "x"}})
    env.advance(3600)
    v = env.verify()
    assert v["evaluation"][a]["success"] is False and v["agents"][a]["success"] is False
    assert env.reward(a) == 0.0 and "evaluation" not in json.dumps(env.observe(a))
    leg = make(2, "toydom:b", 3, LEGACY)
    assert leg.reward(leg.agents()[0]) == 0.0 and not EO.hidden(leg.world)


# ---------------------------------------------------------------------------------------------- gym / pettingzoo
def test_gym_and_pettingzoo_wrappers_over_every_tier_keep_exact_numbers():
    from crucible.experience.gym import ExperienceGymEnv, ExperienceParallelEnv
    for tier, name in ((1, "mech:scm_additive"), (1, "rapid:net_routing"), (0, "tiny")):
        kw = {"agents": ["learner"]} if tier == 0 else ({"validate": False} if name.startswith("mech") else {})
        g = ExperienceGymEnv(tier, name, 3, max_steps=4, **kw)
        o, info = g.reset()
        assert g.observation_space.contains(o)
        js = json.loads(o["json"])
        k0 = info["numbers_keys"]
        assert all(o["numbers"][i] == js["numbers"][k] for i, k in enumerate(k0) if o["numbers_mask"][i])
        for _ in range(4):
            o, r, term, trunc, info = g.step(None)
            assert r == 0.0 and term is False and not info["reward_is_learning_signal"]
        assert "verifier" in g.verify()
    pz = ExperienceParallelEnv(1, "rapid:team_commons", 4, max_steps=3)
    obs, infos = pz.reset()
    assert set(obs) == set(pz.possible_agents)
    acts = {a: {"tool": "rapid.query", "args": {"target": "history"}} for a in pz.agents}
    obs, rew, term, trunc, infos = pz.step(acts)
    assert set(rew.values()) == {0.0} and not any(term.values())
    assert all(infos[a]["action_result"]["ok"] for a in infos)


# ---------------------------------------------------------------------------------------------- isolation
def test_hosted_world_gives_the_learner_no_way_to_the_verifier():
    from crucible.experience import host
    part, ev = host({"tier": 1, "name": "rapid:net_routing", "seed": 3})
    try:
        a = part.agents()[0]
        assert validate_observation(part.observe(a)) == []
        assert not hasattr(part, "verify") and not hasattr(part, "env")
        status, msg = part._raw("verify")
        assert status == "error" and "not available" in msg
        assert part._raw("env")[0] == "error" and part._raw("__dict__")[0] == "error"
        assert "outcome" in ev.verify()
    finally:
        ev.close()


# ---------------------------------------------------------------------------------------------- store
def test_store_keeps_participant_content_verbatim_locks_writers_and_detects_truncation(tmp_path):
    root = str(tmp_path / "s")
    st = ExperienceStore(root, shard_records=3)
    evil = {"$blob": "f" * 64}
    for i in range(9):
        st.append({"i": i, "observation": {"tools": [{"name": "a.b"}], "entities": [{"content": evil}]}})
    with pytest.raises(StoreLocked):
        ExperienceStore(root)
    st.close()
    recs = list(ExperienceStore(root, readonly=True).iterate())
    assert len(recs) == 9 and recs[0]["observation"]["entities"][0]["content"] == evil
    assert recs[0]["observation"]["tools"] == [{"name": "a.b"}]
    os.remove(os.path.join(root, "shards", "000002.jsonl"))
    with open(os.path.join(root, "manifest.jsonl")) as fh:
        lines = fh.readlines()
    with open(os.path.join(root, "manifest.jsonl"), "w") as fh:
        fh.writelines(lines[:-1])
    with pytest.raises(StoreCorrupt):
        ExperienceStore(root, readonly=True).verify()


def test_store_completes_a_seal_interrupted_between_manifest_and_rename(tmp_path):
    root = str(tmp_path / "s")
    st = ExperienceStore(root, shard_records=100)
    for i in range(5):
        st.append({"i": i})
    st.flush()
    st._fh.close()
    st._fh = None
    real = os.replace
    os.replace = lambda a, b: None if a.endswith(".jsonl.part") else real(a, b)   # die between the two steps
    try:
        st._seal_part()
    finally:
        os.replace = real
    st._unlock()
    st2 = ExperienceStore(root)
    st2.append({"i": 5})
    st2.close()
    assert [r["i"] for r in ExperienceStore(root, readonly=True).iterate()] == list(range(6))


# ---------------------------------------------------------------------------------------------- reference learner
def test_reference_learner_bootstraps_next_state_resets_between_worlds_and_checkpoints_context(tmp_path):
    from crucible.experience import run_learner
    from crucible.experience.reference import ReferenceLearner
    env = make(1, "mech:scm_additive", 3, validate=False)
    L = ReferenceLearner(seed=1)
    run_learner(env, L, 10, update_every=1000, learn=False)
    assert len(L.buf) == 10 and all(b.get("h_after") is not None for b in L.buf)
    boot = L._bootstrap(L.buf[-1])
    assert isinstance(boot, float) and boot != L.buf[-1]["value"]       # V(s_T+1), not V(s_T)
    L.reset("other")
    assert L._boundary and len(L.buf) == 10
    path = str(tmp_path / "ck.pt")
    L.save(path)
    L2 = ReferenceLearner(seed=9)
    L2.load(path)
    assert len(L2.buf) == 10 and L2._boundary and float((L2.h - L.h).abs().sum()) == 0.0
    obs = env.observe(env.agents()[0])
    a1, a2 = L.act(obs, explore=False), L2.act(obs, explore=False)
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)
    assert L.update()["changed"]
    recs = [{"agent": a, "env": {"name": n}, "observation": obs, "action": None, "result": None, "reward": 0.0,
             "done": False, "episode": 0} for a, n in (("x", "w1"), ("x", "w1"), ("y", "w2"), ("y", "w2"))]
    out = ReferenceLearner(seed=2).learn_from_records(recs, batch=64)
    assert out["resets"] == 1 and out["transitions"] == 4


# ---------------------------------------------------------------------------------------------- AGENTRS sl adapter
def test_sl_adapter_pays_nothing_and_never_ends_on_the_verifier_by_default():
    import sys
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    if src not in sys.path:
        sys.path.insert(0, src)
    from agentrs.sl.world import Task
    from crucible.integrations.agentrs_sl import CrucibleSLWorld
    for mode in (None, LEGACY):
        w = CrucibleSLWorld("solvency", "tiny", 0, 3600, max_decisions=4, reward_mode=mode)
        w.initialize_world(Task("crucible/solvency", 1, "train"))
        w.execute("0", ["wait", "END"])
        w.last_invalid = True
        c = w.observe_consequences()
        g = w.goal()
        if mode is None:
            assert c.reward == 0.0 and not any(t.startswith("metric:") for t in g.tokens) and "target" not in g.numbers
            assert EO.hidden(w.world)
        else:
            assert c.reward < 0 and any(t.startswith("metric:") for t in g.tokens)
        w.close()
