"""Harness end-to-end tests that need only the self-hosted sandbox (SocialWebLab):
1 account lookup / capabilities, 2 provisioning that needs a human, 7 third-party ingestion,
8 temporal follow-up, 9 verified completion, 10 training batch, 11 optimizer step, 12 promotion,
13 revocation, 14 unauthorised providers, 15 invalid actions, 16 multiple agents on shared inference,
17 restart without losing delayed-observation jobs."""

import asyncio
import time

import pytest
from harness_helpers import PG_URL, REDIS_URL, make_sandbox

from agentrs.harness.db import ActionLog, Experience, Task
from agentrs.harness.tasks import PracticeTaskGenerator
from agentrs.harness.types import ActionRequest, Tier
from agentrs.training.common import param_change, param_snapshot

DAY = 86400.0


def _task(h, tid="t1", allowed=("socialweblab:create_post", "socialweblab:read_feed", "socialweblab:read_user"), tier=1,
          state="running", budget=None):
    h.tasks.create(task_id=tid, source="human", goal="test task", environment="socialweblab",
                   required_capabilities=["socialweblab:create_post"], allowed_actions=list(allowed),
                   verifier={"kind": "socialweblab_help_within", "within_s": DAY}, authorized_tier=tier, budget=budget)
    h.tasks.set_state(tid, state)


def _provision(s, agent="sandbox_agent_001"):
    async def go():
        req = s.h.provisioning.request("socialweblab", ["create_post", "read_content", "reply"], "sandbox practice", agent)
        return await s.h.provisioning.verify(req.id) if req.state == "CREDENTIALS_CONNECTED" else req
    return asyncio.run(go())


def _post(s, body, tid="t1", agent="sandbox_agent_001", **params):
    p = {"space": "linux", "title": body[:40], "body": body, **params}
    return asyncio.run(s.h.gateway.submit(ActionRequest(agent_id=agent, task_id=tid, provider="socialweblab",
                                                        operation="create_post", params=p)))


def test_1_account_lookup_and_capability_checking(sandbox):
    s = sandbox
    assert _provision(s).state == "ACTIVE"
    acc = s.h.accounts.find_for("sandbox_agent_001", "socialweblab", ["create_post"])
    assert acc is not None and acc.status == "active" and acc.bot_disclosure == "labelled"
    assert s.h.accounts.find_for("sandbox_agent_001", "socialweblab", ["direct_message"]) is None
    assert s.h.accounts.find_for("someone_else", "socialweblab", ["create_post"]) is None
    assert s.h.registry.allows("socialweblab", "create_post")[0]
    assert not s.h.registry.allows("bluesky", "create_post")[0]  # shipped disabled
    _task(s.h)
    r = _post(s, "How do I fix ssh when cron hangs? error E1234")
    assert r.status == "executed" and r.provider_object_id
    # an operation the account was not authorised for is refused even though the provider supports it
    s.h.accounts.update(acc.id, capabilities=["read_content"])
    assert _post(s, "Another question about bash 5.1?").status == "denied"


def test_2_provisioning_that_requires_human_setup_and_denial_is_final(sandbox):
    h = sandbox.h
    h.registry.get("bluesky").enabled = True  # even when enabled, accounts are never created automatically
    h.registry.get("bluesky").capabilities["create_post"] = True
    from agentrs.harness.identity import AgentSpec

    h.agents.upsert(AgentSpec(id="comms", display_name="c", role="c", allowed_capabilities=["bluesky:create_post"],
                              max_tier=3))
    req = h.provisioning.request("bluesky", ["create_post"], "publish approved announcements", "comms")
    assert req.state == "PENDING_HUMAN_SETUP"
    assert [x["to"] for x in req.history] == ["REQUESTED", "POLICY_CHECKED", "AUTHORIZATION_REQUIRED",
                                              "PENDING_HUMAN_SETUP"]
    assert any("sign-up" in st or "Create the account yourself" in st for st in req.human_steps)
    assert "human_account_creation" in req.decision_reason
    with h.db.session() as s:
        from agentrs.harness.db import Approval

        ap = s.query(Approval).filter_by(ref_id=req.id).one()
        assert ap.kind == "provisioning" and ap.status == "pending"
    h.provisioning.deny(req.id, "not needed", "operator@example.org")
    again = h.provisioning.request("bluesky", ["create_post"], "please, again", "comms")
    assert again.id == req.id and again.state == "DENIED"  # no repeated requests after a denial
    # an agent may not request capabilities outside its allowance
    bad = h.provisioning.request("bluesky", ["direct_message"], "dm people", "comms")
    assert bad.state == "DENIED" and "may not hold" in bad.decision_reason


def test_8_and_9_followups_verified_completion_and_ledger(sandbox):
    s = sandbox
    gen = PracticeTaskGenerator(s.h.tasks)
    for i in range(3):
        gen.generate("train", i)
    rt = s.rts["sandbox_agent_001"]
    out = asyncio.run(rt.loop(3, environments=["socialweblab"]))
    assert [o["status"] for o in out] == ["executed"] * 3
    with s.h.db.session() as db:
        exps = db.query(Experience).all()
        assert len(exps) == 3 and all(e.reward_status == "pending" for e in exps)
    assert s.h.verification.run()["pending"] == 3  # not failures: the window is still open
    # time passes in the lab; follow-ups run on their schedule
    from agentrs.harness.sandbox import sandbox_fetchers

    for dt in (90, 300, 900, 3600, DAY / 4, DAY / 4, DAY / 4, DAY / 4):  # the outcome worker polls as time passes
        s.lab.advance(dt)
        asyncio.run(s.h.outcomes.run_due(sandbox_fetchers(s.h), {s.lab.time_base: s.lab.now()}))
    L = s.h.outcomes.ledger
    a = L.actions("source_type='self'")[0]
    obs = L.observations(a["action_id"])
    assert len(obs) >= 5 and obs[0]["elapsed"] < 120 and obs[-1]["elapsed"] > DAY / 2  # minutes ... days
    assert a["semantic"]["token_ids"] and a["policy_checkpoint"] == "policy-v0"
    st = s.h.verification.run()
    assert st["matured"] == 3
    with s.h.db.session() as db:
        assert all(e.reward in (0.0, 1.0) for e in db.query(Experience).all())
        assert all(t.state in ("succeeded", "failed") for t in db.query(Task).all())
    assert all(L.rewards(x["action_id"]) for x in L.actions("source_type='self'"))


def test_7_third_party_events_are_ingested_through_authorised_sources(sandbox, tmp_path):
    from agentrs.sensors.observation_network import NetworkConfig
    from agentrs.temporal.system import build_network, run_lab

    s = sandbox
    net = build_network(tmp_path / "sensors", s.lab, None, NetworkConfig(forecasts_enabled=False))
    run_lab(s.lab, net, 6 * 3600, 1800)
    third = net.ledger.actions("source_type='third_party'")
    assert third and all(a["actor_ref"].startswith("actor-") for a in third)  # pseudonymised
    assert net.ledger.stats()["events"] > 0 and net.ledger.stats()["observations"] > 0


def test_10_11_training_batch_and_real_optimizer_step(sandbox, tmp_path):
    from agentrs.checkpoints.registry import CheckpointRegistry
    from agentrs.harness.learning import HarnessLearner
    from agentrs.training.delayed_reward_trainer import DelayedRewardConfig

    s = sandbox
    gen = PracticeTaskGenerator(s.h.tasks)
    for i in range(4):
        for k in range(2):
            gen.generate("train", i, attempt=k)
    asyncio.run(s.rts["sandbox_agent_001"].loop(8, environments=["socialweblab"]))
    s.lab.advance(DAY + 60)
    assert s.h.verification.run()["matured"] == 8
    # rewards that differ within a task group (the real verifier on an untrained model is almost always 0,
    # so a parity verifier stands in to guarantee variance for this optimizer test)
    L = s.h.outcomes.ledger
    for a in L.actions("source_type='self'"):
        L.record_reward(a["action_id"], "delayed", float(int(a["context"]["object_ref"]) % 2), env_time=s.lab.now(),
                        maturity_time=s.lab.now(), checkpoint=a["policy_checkpoint"], objective="test:parity",
                        idem=a["action_id"] + ":parity")
    reg = CheckpointRegistry(tmp_path / "reg.sqlite")
    learner = HarnessLearner(tmp_path / "learn", L, s.cp, reg, DelayedRewardConfig(objective="test:parity"))
    batch = learner.training_batch()
    steps = batch["steps"]
    assert len(steps) == 8 and all(st.token_ids and len(st.token_ids) == len(st.token_logprobs_old) for st in steps)
    assert any(abs(st.advantage) > 0 for st in steps)
    from agentrs.temporal.outcome_labels import build_examples
    from agentrs.sensors.socialweblab_source import lab_temporal_spec

    ex = build_examples(L, steps[0].action_id, lab_temporal_spec())[0]
    assert ex.complete_action and ex.source_type == "self"
    live = param_snapshot(s.cp.policy)
    res = learner.train_candidate()
    assert res["candidate"] is not None and res["param_change"] > 0  # trainable weights changed
    assert param_change(s.cp.policy, live) == 0.0  # the live inference weights were not touched
    rec = reg.get(res["registry_id"])
    assert rec.status == "candidate" and rec.metrics["param_change"] > 0


def test_12_candidate_checkpoint_evaluation_and_promotion(sandbox, tmp_path):
    from agentrs.checkpoints.registry import CheckpointRegistry
    from agentrs.harness.learning import HarnessLearner
    from agentrs.training.delayed_reward_trainer import DelayedRewardConfig

    s = sandbox
    gen = PracticeTaskGenerator(s.h.tasks)
    for i in range(4):
        gen.generate("train", i, attempt=0)
        gen.generate("train", i, attempt=1)
    asyncio.run(s.rts["sandbox_agent_001"].loop(8, environments=["socialweblab"]))
    s.lab.advance(DAY + 60)
    s.h.verification.run()
    L = s.h.outcomes.ledger
    for a in L.actions("source_type='self'"):
        L.record_reward(a["action_id"], "delayed", float(int(a["context"]["object_ref"]) % 2), env_time=s.lab.now(),
                        maturity_time=s.lab.now(), checkpoint=a["policy_checkpoint"], objective="test:parity",
                        idem=a["action_id"] + ":parity")
    reg = CheckpointRegistry(tmp_path / "reg.sqlite")
    prod_id = reg.register("policy", "bc", metrics={})
    reg.promote(prod_id)
    learner = HarnessLearner(tmp_path / "learn", L, s.cp, reg, DelayedRewardConfig(objective="test:parity"))
    cand = learner.train_candidate()
    snap = tmp_path / "eval.db"
    s.srv.control("snapshot", path=str(snap))
    counter = {"n": 0}

    def evaluate(ck):
        s.srv.control("restore", path=str(snap))
        for i in range(3):
            gen.generate("val", i, attempt=1000 + counter["n"])
        counter["n"] += 1
        asyncio.run(s.rts["sandbox_agent_001"].loop(3, environments=["socialweblab"], split="val", checkpoint=ck))
        s.lab.advance(DAY + 60)
        s.h.verification.run()
        with s.h.db.session() as db:
            ex = [e for e in db.query(Experience).all() if e.split == "val" and e.model_checkpoint == ck]
        return {"success_rate": sum(e.reward or 0 for e in ex) / max(1, len(ex)), "n": len(ex)}

    out = learner.promote_if_better(cand, evaluate, s.backend)
    rec = reg.get(cand["registry_id"])
    assert out["candidate"]["n"] == 3 and out["production"]["n"] == 3  # both evaluated on held-out tasks
    assert rec.metrics["heldout"] == out["candidate"]
    if out["promoted"]:
        assert rec.status == "production" or reg.production("policy").id == rec.id
        assert s.backend.production == cand["checkpoint_id"]
    else:
        assert rec.status == "rejected" and s.backend.production == "policy-v0"


def test_13_revocation_immediately_prevents_action(sandbox):
    s = sandbox
    _provision(s)
    _task(s.h)
    assert _post(s, "How do I fix ssh when cron hangs? error E1234").status == "executed"
    acc = s.h.accounts.find_for("sandbox_agent_001", "socialweblab", ["create_post"])
    n_items = len(s.lab.items())
    s.h.accounts.revoke(acc.id, "operator test")
    r = _post(s, "How do I fix apt when kernel hangs? version 6.1.2")
    assert r.status == "denied" and len(s.lab.items()) == n_items
    assert s.h.accounts.get(acc.id).credential_refs == {}
    s.h.policy.emergency_stop(True, "operator", "drill")
    assert "emergency stop" in _post(s, "x y z?").reason
    s.h.policy.emergency_stop(False, "operator")


def test_14_providers_without_authorisation_deny_activity(sandbox):
    h = sandbox.h
    _task(h, "t2", allowed=("reddit:*", "x:*", "bluesky:*"), tier=3)
    for prov, op, params in (("reddit", "create_post", {"subreddit": "test", "title": "t", "text": "hello"}),
                             ("x", "create_post", {"text": "hello"}), ("bluesky", "create_post", {"text": "hello"})):
        if h.gateway.spec(prov, op) is None:
            pytest.skip(f"{prov} connector not installed")
        r = asyncio.run(h.gateway.submit(ActionRequest(agent_id="sandbox_agent_001", task_id="t2", provider=prov,
                                                       operation=op, params=params)))
        assert r.status in ("denied", "rejected"), (prov, r)
    assert not any(h.registry.enabled(p) for p in ("reddit", "x", "bluesky", "linkedin", "instagram", "mastodon"))


def test_15_invalid_or_restricted_actions_are_rejected_before_execution(sandbox):
    s = sandbox
    _provision(s)
    _task(s.h)
    n = len(s.lab.items())
    sub = lambda **kw: asyncio.run(s.h.gateway.submit(ActionRequest(agent_id="sandbox_agent_001", task_id="t1",  # noqa: E731
                                                                    provider="socialweblab", **kw)))
    assert sub(operation="delete_everything", params={}).status == "rejected"
    assert sub(operation="create_post", params={"space": "linux"}).status == "rejected"  # missing params
    assert sub(operation="create_post", params={"space": "linux", "title": "t", "body": "b", "admin": 1}).status == "rejected"
    tok = s.h.secrets.get(s.h.accounts.find_for("sandbox_agent_001", "socialweblab", ["create_post"]).credential_refs["session_token"])
    r = sub(operation="create_post", params={"space": "linux", "title": "t", "body": f"my token is {tok}"})
    assert r.status == "denied" and "credential" in r.reason  # exfiltration blocked
    assert sub(operation="create_post", params={"space": "linux", "title": "t", "body": "free money here"}).status == "denied"
    assert sub(operation="reply", params={"item_id": 1, "body": "hi"}).status == "denied"  # not allowed for the task
    _task(s.h, "t0", tier=0)
    r = asyncio.run(s.h.gateway.submit(ActionRequest(agent_id="sandbox_agent_001", task_id="t0", provider="socialweblab",
                                                     operation="create_post",
                                                     params={"space": "linux", "title": "t", "body": "ok?"})))
    assert r.status == "denied" and "tier" in r.reason  # a tier-0 task cannot mutate
    assert len(s.lab.items()) == n  # nothing reached the application
    with s.h.db.session() as db:
        assert db.query(ActionLog).filter(ActionLog.status.in_(["rejected", "denied"])).count() >= 7  # all audited


def test_16_multiple_agents_share_one_inference_service(tmp_path, scratch_bb):
    agents = tuple(f"sandbox_agent_{i:03d}" for i in range(4))
    s = make_sandbox(tmp_path, scratch_bb, agents=agents)
    try:
        # one shared sandbox account for all four workers (no account per worker)
        _provision(s, agents[0])
        acc = s.h.accounts.find_for(agents[0], "socialweblab", ["create_post"])
        s.h.accounts.update(acc.id, allowed_agents=list(agents))
        gen = PracticeTaskGenerator(s.h.tasks)
        for i in range(8):
            gen.generate("train", i)

        async def run_all():
            return await asyncio.gather(*(s.rts[a].loop(2, environments=["socialweblab"]) for a in agents))

        res = asyncio.run(run_all())
        assert [len(r) for r in res] == [2, 2, 2, 2] and all(o["status"] == "executed" for r in res for o in r)
        assert s.inf.stats["max_batch_seen"] >= 2  # requests from different agents were batched together
        assert len(s.backend.models) == 1  # one model copy
        assert len(s.h.accounts.list("socialweblab")) == 1
        with s.h.db.session() as db:
            assert {t.assigned_agent for t in db.query(Task).all()} == set(agents)
    finally:
        s.srv.stop()
        s.lab.close()


def test_17_restart_keeps_delayed_observation_jobs(tmp_path, scratch_bb):
    from agentrs.harness.sandbox import sandbox_fetchers
    from agentrs.harness.secrets import SecretStore

    key = SecretStore.generate_key()
    s = make_sandbox(tmp_path, scratch_bb, secret_key=key)
    try:
        PracticeTaskGenerator(s.h.tasks).generate("train", 0)
        asyncio.run(s.rts["sandbox_agent_001"].loop(1))
        pending = s.h.outcomes.scheduler.stats().get("horizon:pending", 0)
        assert pending >= 6
        aid = s.h.outcomes.ledger.actions("source_type='self'")[0]["action_id"]
        s.h.outcomes.ledger.close()
        s.h.outcomes.scheduler.close()
        s.h.db.engine.dispose()
        del s.h
        # a new process: rebuilt from the same database, outcome store and secret key
        s2 = make_sandbox(tmp_path, scratch_bb, secret_key=key, lab=s.lab, srv=s.srv)
        assert s2.h.outcomes.scheduler.stats().get("horizon:pending", 0) == pending
        s.lab.advance(2 * DAY)
        st = asyncio.run(s2.h.outcomes.run_due(sandbox_fetchers(s2.h), {s.lab.time_base: s.lab.now()}))
        assert st["done"] >= 5 and st["failed"] == 0
        assert len(s2.h.outcomes.ledger.observations(aid)) >= 5
    finally:
        s.srv.stop()
        s.lab.close()


@pytest.mark.skipif(not (PG_URL and REDIS_URL), reason="set AGENTRS_TEST_PG_URL and AGENTRS_TEST_REDIS_URL")
def test_postgres_and_redis_backends(tmp_path, scratch_bb):
    """The same gateway path against a real PostgreSQL control plane and Redis rate limiter."""
    from sqlalchemy import create_engine, text

    name = f"harness_test_{int(time.time() * 1000)}"
    admin = create_engine(PG_URL.rsplit("/", 1)[0] + "/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    s = make_sandbox(tmp_path, scratch_bb, db_url=PG_URL.rsplit("/", 1)[0] + "/" + name, redis_url=REDIS_URL)
    try:
        assert s.h.db.is_postgres and s.h.policy.limiter.r is not None
        _provision(s)
        _task(s.h, f"pg-{time.time()}")
        with s.h.db.session() as db:
            tid = db.query(Task).order_by(Task.created_at.desc()).first().id
        assert _post(s, f"How do I fix ssh when cron hangs? run {time.time()}", tid=tid).status == "executed"
        from agentrs.harness.jobs import JobQueue

        q = JobQueue(s.h.db)
        q.enqueue("ping", {"x": 1}, idem=f"ping-{time.time()}")
        assert len(q.claim("w1", ["ping"])) >= 1  # FOR UPDATE SKIP LOCKED path
    finally:
        s.srv.stop()
        s.lab.close()
