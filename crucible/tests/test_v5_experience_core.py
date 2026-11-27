"""Unit and integration tests of the learner interface (crucible.experience, workstream L)."""
import json
import os

import numpy as np
import pytest

from crucible.experience import (ExperienceStore, TierUnavailable, make, validate_action, validate_observation,
                                 validate_result)
from crucible.experience.kernel import KernelExperienceEnv
from crucible.experience.observation import empty, kinds_present
from crucible.experience.reference import ReferenceLearner
from crucible.experience.store import StoreCorrupt
from crucible.research.tasks import make_task
from crucible.worldgen import generate


# ---------------------------------------------------------------------------------------------- observation schema
def test_observation_schema_accepts_valid_and_rejects_invalid():
    o = empty("p:a", t=5, cursor=3)
    o["numbers"]["bank.balances.acct_1"] = 1234
    o["events"].append({"seq": 4, "t": 5, "type": "x", "actor": "sys", "payload": {}, "cause": None})
    assert validate_observation(o) == []
    bad = dict(o, numbers={"x": float("nan")})
    assert any("finite" in e for e in validate_observation(bad))
    assert any("missing field" in e for e in validate_observation({k: v for k, v in o.items() if k != "tools"}))
    assert any("unknown field" in e for e in validate_observation(dict(o, reward=1.0)))
    assert validate_observation(dict(o, events=[{"seq": 1}]))
    assert validate_observation(dict(o, arrays={"a": [1, "x"]}))
    assert validate_action({"tool": "bank.accounts", "args": {}}) == []
    assert validate_action({"tool": "x", "args": {}, "reward": 1})
    assert validate_result({"ok": True, "t": 0, "tool": "x", "cost_s": 1, "result": None}) == []
    assert validate_result({"ok": False, "t": 0, "tool": "x", "cost_s": 1})


def test_invalid_reward_mode_is_refused_everywhere():
    from crucible.research.envs import MechVecEnv
    task = make_task("scm_additive", 1, validate=False)
    with pytest.raises(ValueError):
        MechVecEnv([task], reward_mode="shaped")
    with pytest.raises(ValueError):
        make(1, "mech:scm_additive", 1, "reward_please", validate=False)


# ---------------------------------------------------------------------------------------------- Tier 1 = kernel twin
def test_tier1_adapter_is_the_kernel_facility_tick_for_tick():
    task = make_task("scm_delay", 5, validate=False)
    t1 = make(1, "mech:scm_delay", 5, "observation_only", task=task)
    cfg, seed = task.world_config()
    k = KernelExperienceEnv(generate(cfg, seed), agents=["agent"], endowment=task.endowment,
                            reward_mode="observation_only", experiments=False)
    a1, a2 = t1.agents()[0], k.agents()[0]
    assert a1 == a2 == "p:agent"
    s1 = t1.execute(a1, {"tool": "rig.spec", "args": {}})["result"]
    s2 = k.execute(a2, {"tool": "rig.spec", "args": {}})["result"]
    assert s1 == s2
    # the facility's tools are the kernel facility's own (the Tier 1 adapter adds only its self-experiment tools exp.*,
    # crucible.experience.selfexp; the kernel world here is built without the exp module)
    fac = {t["name"]: t["inputSchema"] for t in t1.tools(a1) if not t["name"].startswith("exp.")}
    assert {t["name"] for t in t1.tools(a1)} - set(fac) == {"exp.register", "exp.result", "exp.mine"}
    assert fac == {t["name"]: t["inputSchema"] for t in k.tools(a2) if t["name"] in fac}
    rng = np.random.default_rng(0)
    for _ in range(25):
        st = {lv["name"]: float(np.round(rng.uniform(lv["min"], lv["max"]), 3)) for lv in s1["levers"]}
        r1 = t1.execute(a1, {"tool": "rig.set", "args": {"settings": st}})
        r2 = k.execute(a2, {"tool": "rig.set", "args": {"settings": st}})
        assert r1["result"] == r2["result"]
        t1.advance(3600)
        k.advance(3600)
        q1 = t1.execute(a1, {"tool": "rig.read", "args": {}})["result"]
        q2 = k.execute(a2, {"tool": "rig.read", "args": {}})["result"]
        assert q1["report"] == q2["report"] and q1["next_settings"] == q2["next_settings"]
    b1 = t1.execute(a1, {"tool": "bank.accounts", "args": {}})["result"][0]["balance"]
    b2 = sum(x["balance"] for x in k.execute(a2, {"tool": "bank.accounts", "args": {}})["result"] if x["owner"] == a2)
    assert b1 == b2
    # the summary paths the kernel facility shows are the Tier 1 adapter's too
    o1, o2 = t1.observe(a1), k.observe(a2)
    shared = [p for p in o2["numbers"] if p.startswith("rig.seats.r1.")]
    assert shared and all(o1["numbers"][p] == o2["numbers"][p] for p in shared)


def test_tier1_unknown_tools_and_bad_args_are_results_not_exceptions():
    env = make(1, "mech:scm_additive", 2, "observation_only", validate=False)
    a = env.agents()[0]
    assert env.execute(a, {"tool": "rig.fly", "args": {}})["error"]["code"] == "unknown_tool"
    assert env.execute(a, {"tool": "rig.set", "args": {"settings": {"nope": 1}}})["error"]["code"] == "invalid_action"
    assert env.execute(a, {"tool": "rig.set", "args": {}})["error"]["code"] == "invalid_action"
    assert env.execute(a, "not an action")["error"]["code"] == "invalid_action"
    assert validate_observation(env.observe(a)) == []


# ---------------------------------------------------------------------------------------------- kernel adapter
def test_kernel_events_are_lossless_visibility_respected_and_causes_only_own():
    env = make(0, "tiny", 2, "observation_only", agents=["ann", "bob", "cid"], max_events=5,
               overrides={"n_consumers": 50})
    a, b, c = env.agents()
    env.observe(a)
    env.observe(b)
    for k in range(12):
        env.execute(a, {"tool": "mail.send", "args": {"to": [b], "subject": f"note {k}", "body": "hello there bob"}})
    env.execute(b, {"tool": "mail.send", "args": {"to": [c], "subject": "secret", "body": "only for cid to read"}})
    env.advance(3600)
    ob = env.observe(b)
    assert len(ob["events"]) == 5 and any(m.startswith("events:") for m in ob["missing"])
    rest, cur = env.observe_since(b, ob["cursor"])
    allb, _ = env.observe_since(b, 0)
    delivered = [e["type"] for e in allb if e["type"] == "mail.delivered"]
    assert len(delivered) == 12 and ob["events"][0]["seq"] < rest[0]["seq"]
    oa = env.observe(a)
    assert not any("only for cid" in json.dumps(e) for e in env.observe_since(a, 0)[0])
    # a's own sends carry a cause, b's view of a's mail does not
    caused = [e for e in env.observe_since(a, 0)[0] if e["cause"]]
    assert caused and all(e["cause"]["tool"] == "mail.send" for e in caused)
    assert not any(e["cause"] for e in allb if e["actor"] == a)
    assert validate_observation(oa) == [] and "artifact_references" in kinds_present(ob)


def test_kernel_action_seq_links_scheduled_consequences():
    env = make(0, "tiny", 3, "observation_only", agents=["ann"], overrides={"n_consumers": 50})
    a = env.agents()[0]
    r = env.execute(a, {"tool": "exp.register", "args": {
        "hypothesis": "my balance does not change in a day", "deadline_in_s": 86400,
        "measure": {"tool": "bank.accounts", "where": {"owner": "$self"}, "field": "balance", "aggregate": "sum"},
        "prediction": {"op": "==", "value": 1_000_000}}})
    assert r["ok"]
    env.advance(2 * 86400)
    evs, _ = env.observe_since(a, 0)
    res = [e for e in evs if e["type"] == "exp.resolved"]
    assert len(res) == 1 and res[0]["cause"]["action_seq"] == r["action_seq"]
    assert res[0]["cause"]["via"] == "scheduled" and res[0]["cause"]["delay_s"] == 86400
    assert res[0]["payload"]["held"] is True


def test_exp_module_admission_and_scope():
    from crucible.worldgen.admission import admit
    rep = admit(["exp"])
    assert rep.passed, rep.errors
    env = make(0, "tiny", 4, "observation_only", agents=["ann", "bob"], overrides={"n_consumers": 50})
    a, b = env.agents()
    acct_b = env.world.modules["bank"].primary_account(env.world.state, b)
    base = {"hypothesis": "h", "prediction": {"op": ">=", "value": 0}, "deadline_in_s": 3600}
    r = env.execute(a, {"tool": "exp.register", "args": dict(base, measure={"tool": "bank.transfer"})})
    assert r["error"]["code"] == "forbidden"                        # not a read-only tool
    r = env.execute(a, {"tool": "exp.register", "args": dict(base, measure={
        "tool": "bank.balance", "args": {"account": acct_b}, "field": "balance"})})
    assert not r["ok"]                                              # someone else's account: the tool refuses
    r = env.execute(a, {"tool": "exp.register", "args": dict(base, measure={"tool": "admin.join"})})
    assert not r["ok"]


# ---------------------------------------------------------------------------------------------- store
def test_store_roundtrip_recovery_and_tamper_detection(tmp_path):
    root = str(tmp_path / "s")
    st = ExperienceStore(root, shard_records=4)
    tools = [{"name": "a.b", "inputSchema": {"type": "object"}}]
    hashes = [st.append({"i": i, "observation": {"tools": tools, "x": i}}) for i in range(10)]
    assert len(set(hashes)) == 10
    st.flush()
    st._fh.close()
    st._fh = None
    st._unlock()                                                     # the writer crashed: the OS drops its lock
    part = os.path.join(root, "shards", "000002.jsonl.part")
    with open(part, "ab") as fh:
        fh.write(b'{"i": 10, "h": "torn')                           # a crashed writer's torn line
    st2 = ExperienceStore(root)
    recs = list(st2.iterate())
    assert [r["i"] for r in recs] == list(range(10)) and recs[3]["observation"]["tools"] == tools
    assert st2.verify()["shards"] == 3 and len(st2) == 10
    st2.append({"i": 10})
    st2.close()
    assert [r["i"] for r in ExperienceStore(root, readonly=True).iterate()] == list(range(11))
    with open(os.path.join(root, "shards", "000000.jsonl"), "r+b") as fh:
        data = fh.read().replace(b'"x":1', b'"x":7')
        fh.seek(0)
        fh.write(data)
    with pytest.raises(StoreCorrupt):
        list(ExperienceStore(root, readonly=True).iterate())
    with pytest.raises(TypeError):
        ExperienceStore(str(tmp_path / "t")).append({"bad": float("inf")})


# ---------------------------------------------------------------------------------------------- reference learner
def test_reference_learner_is_seeded_and_updates():
    env = make(1, "mech:scm_additive", 3, "observation_only", validate=False)
    obs = env.observe(env.agents()[0])
    a1 = ReferenceLearner(seed=5).act(obs)
    a2 = ReferenceLearner(seed=5).act(obs)
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)
    L = ReferenceLearner(seed=1, reward_mode="observation_only")
    from crucible.experience import run_learner
    s = run_learner(env, L, 40, update_every=20)
    assert s["updates"] >= 2 and s["params_changed_updates"] >= 1 and s["schema_failures"] == 0
    L.freeze()
    h = L.params_hash()
    run_learner(env, L, 10, update_every=5)
    assert L.params_hash() == h                                     # frozen: no weight change


def test_reference_learner_draws_id_arguments_from_ids_it_has_seen(tmp_path):
    """An id argument (``listing``, ``order``, ``seller``) draws from the ids the learner saw under that key in its
    own results and visible events; the memory is checkpointed with its state and is not part of its weights."""
    L = ReferenceLearner(seed=0)
    L._harvest([{"listing": "xl_000005", "exchange": "xch_000001", "title": "Fan"}, {"listing": "xl_000011"}])
    L._harvest({"orders": [{"id": "xo_000003", "status": "open"}], "sellers": ["p:trader04", "not an id"]})
    L._harvest_events({"events": [{"type": "x", "payload": {"listing": "xl_000005", "invoice": "inv_000002"}}]})
    assert L.ids["listing"] == ["xl_000011", "xl_000005"]               # most recent last
    assert L.ids["order"] == ["xo_000003"] and L.ids["id"] == ["xo_000003"]
    assert L.ids["seller"] == ["p:trader04"] and L.ids["invoice"] == ["inv_000002"] and "title" not in L.ids
    tool = {"name": "xchg.buy", "inputSchema": {"type": "object", "required": ["listing"],
                                                "properties": {"listing": {"type": "string"}}}}
    obs = {"t": 0, "numbers": {}, "events": [], "tools": [tool], "entities": []}
    picked = set()
    for _ in range(40):
        a = L.act(obs, [tool])
        if a is not None:
            picked.add(a["args"]["listing"])
    assert picked and picked <= {"xl_000005", "xl_000011"}
    p = tmp_path / "ref.pt"
    L.save(str(p))
    L2 = ReferenceLearner(seed=0)
    L2.load(str(p))
    assert L2.ids == L.ids
    L3 = ReferenceLearner(seed=0)
    L3.load_params(L.params_blob())
    assert L3.ids == {}                                             # weights only: no world-specific episodic ids


def test_v5_tiers_are_not_substituted():
    from crucible.experience.registry import civ_available, tier2_domains as domains
    if not domains():
        with pytest.raises(TierUnavailable):
            make(2, "anything", 0)
    else:
        name = sorted(domains())[0]
        assert make(2, name, 0, "observation_only").tier == 2
    if not civ_available():
        with pytest.raises(TierUnavailable):
            make(3, "default", 0)


# ---------------------------------------------------------------------------------------------- an external adapter
def test_external_batched_adapter_follows_the_reward_modes():
    """The batched adapter of the CAM Unified `EnvAdapter` shape that an external learner writes against the public
    interface (examples/experience_external_adapter.py; architecture-neutral - Crucible imports no CAM code, RP.1)
    follows the reward modes: nothing paid and no evaluator key in observation-only, 1.0 once on the verified success
    in verified mode, the 4.0 shaped reward only in explicit legacy mode."""
    from crucible.experience.leakage import find_keys
    from crucible.experience.probes import _cash_goal, _external_adapter
    X = _external_adapter()
    seen = {}
    for mode in ("legacy_shaped_reward", "observation_only", "verified_terminal_outcome"):
        env = X.ExperienceEnvAdapter([{"tier": 0, "name": "tiny", "kw": {
            "agents": ["learner"], "endowment": 1_000_000, "objectives": _cash_goal(1_500_000)}}], reward_mode=mode)
        env.reset([3])
        w, a = env.envs[0].world, env.agents[0]
        payer = w.system_call("admin.join", {"handle": "payer", "endowment": 2_000_000})["principal"]
        w.execute(payer, {"tool": "bank.transfer", "args": {"from_account": w.modules["bank"].primary_account(
            w.state, payer), "to": a, "amount": 700_000, "memo": "invoice"}})
        rs, obs = [], []
        for _ in range(4):
            env.execute([X.Action("bank.transfer", {})])                  # schema-invalid: never penalised
            env.advance_time([1.0])
            rs += env.rewards()
            obs += env.observe()
        seen[mode] = (rs, obs, env.envs[0].verify()["agents"][a]["success"])
        env.close()
    rs, obs, ok = seen["observation_only"]
    assert ok and set(rs) == {0.0} and find_keys(obs) == []
    rs, obs, ok = seen["verified_terminal_outcome"]
    assert ok and sorted(rs) == [0.0, 0.0, 0.0, 1.0] and find_keys(obs) == []
    assert any(r != 0.0 for r in seen["legacy_shaped_reward"][0])
