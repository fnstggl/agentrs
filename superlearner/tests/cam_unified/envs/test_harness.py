"""The SMDP lifetime harness on the real OpenWorld: ExampleInductor (non-privileged control) wrapped as an AgentP
reaches verified success through the adapter; checkpoint / resume in a fresh operator reproduces the
uninterrupted lifetime exactly."""
from __future__ import annotations

import json
import os

import pytest

import crucible
crucible.load_all_modules()
from crucible.research import llm_eval as L                      # noqa: E402
from crucible.usefulness.agents import ExampleInductor          # noqa: E402

from superlearner.cam_unified.interfaces import Action, ActionSource, EventRecord, ObsKind   # noqa: E402
from superlearner.cam_unified.envs.agent_protocol import AgentP, PolicyAgent, Transition     # noqa: E402
from superlearner.cam_unified.envs.harness import run_lifetime   # noqa: E402
from superlearner.cam_unified.envs.openworld import open_world, WAIT_TOOL   # noqa: E402
from superlearner.cam_unified.envs.openworld_eval import OpenWorldEvaluator   # noqa: E402


def tiny_goal(days=8):
    return L.open_economy_goal(world_family="ow_tiny", seed=1, overrides={"n_consumers": 800}, days=days,
                               window_days=3, revenue_cents=10_000, requests=100, sustain_checks=2,
                               endowment=3_000_000)


def test_example_inductor_agent_reaches_verified_success_through_adapter(tmp_path):
    ad, op = open_world([tiny_goal()], seed=0)
    ev = OpenWorldEvaluator(op)
    agent = PolicyAgent(lambda b: ExampleInductor())
    assert isinstance(agent, AgentP)
    seen = []
    path = str(tmp_path / "life.jsonl")
    res = run_lifetime(agent, ad, ev, think_s=600, transcript=path, update_every=24,
                       on_step=lambda k, tr: seen.append(tr))
    ad.close()
    assert res.final[0]["achieved"] is True
    assert res.success_day[0] is not None and res.success_day[0] <= 6.0
    assert res.returns[0] == pytest.approx(2.0, abs=1e-6)         # progress 0 -> 1 plus the success bonus
    assert res.elapsed_days[0] >= 8.0 - 1e-9                       # ran to the task horizon
    assert res.final[0]["metrics"]["independent_revenue"] >= 10_000
    # the agent's rewards are exactly the evaluator's verified increments, delivered per decision
    assert sum(sum(tr.rewards) for tr in seen) == pytest.approx(res.returns[0])
    acc = res.accounting
    assert acc["decisions"] == res.steps and acc["actions"] >= 15 and acc["rejected"] == 0
    assert acc["by_tool"]["cloud.deploy"] == 1 and acc["by_tool"]["code.commit"] == 1
    assert acc["phase_wall_s"]["advance"] > 0 and acc["cpu_s"] > 0
    assert len(res.updates) == res.steps // 24
    # transcript: contract EventRecords; actions SCRIPTED; dt = cost_s + think_s (or wait + think_s)
    lines = open(path).read().splitlines()
    assert json.loads(lines[0])["header"]["format"] == "cam-unified-lifetime/1"
    recs = [EventRecord.from_dict(json.loads(x)) for x in lines[1:]]
    assert len(recs) == res.steps
    acted = [r for r in recs if r.action is not None]
    assert all(r.action.source is ActionSource.SCRIPTED for r in acted)
    for r in acted[:30]:
        expect = r.result.duration + 600 / 86400
        assert r.extra["dt_days"] == pytest.approx(expect) and r.extra["dt_requested_days"] == pytest.approx(expect)
    times = [r.t_sim for r in recs]
    assert times == sorted(times)
    # the deploy's own consequences are linked to its action
    dep = next(r for r in recs if r.action and r.action.tool == "cloud.deploy")
    later = [o for r in recs for o in r.observations if o.caused_by == dep.action.action_id]
    assert any(json.loads(o.content)["type"] == "cloud.deployment" for o in later if o.kind is ObsKind.EVENT)
    # evaluator-only details never reach records
    blob = "\n".join(lines[1:])
    assert "initially_true" not in blob and "ow_tiny" not in blob


class PollAgent:
    """Deterministic, stateful, checkpointable test agent: reads accounts, browses, waits."""

    def __init__(self):
        self.k = 0
        self.model_version = 0
        self.seen_rewards = 0.0

    def act(self, obs, tools):
        self.k += 1
        cycle = [Action("bank.accounts", {}), Action("market.browse", {"limit": 2}),
                 Action(WAIT_TOOL, {"seconds": 1800 + 300 * (self.k % 5)})]
        return [cycle[self.k % 3] for _ in obs.env_ids]

    def observe(self, tr: Transition):
        self.seen_rewards += sum(tr.rewards)

    def update(self):
        self.model_version += 1
        return {"updates": 1}

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"k": self.k, "v": self.model_version, "r": self.seen_rewards}, f)

    def load(self, path):
        d = json.load(open(path))
        self.k, self.model_version, self.seen_rewards = d["k"], d["v"], d["r"]


def _life(store, ckdir, stop=None, resume=False, days=3.0, transcript=None):
    ad, op = open_world([tiny_goal(days=4)], seed=7, store_url=store)
    ev = OpenWorldEvaluator(op)
    agent = PollAgent()
    try:
        res = run_lifetime(agent, ad, ev, days=days, think_s=600, update_every=10, operator=op, checkpoint_dir=ckdir,
                           checkpoint_every_days=1.0, resume=resume, stop_after_decisions=stop, transcript=transcript)
        dig = op.world(0).digest()
        t = op.world(0).now
    finally:
        ad.close()
    return res, dig, t, agent


def _records(path):
    lines = open(path).read().splitlines()
    assert json.loads(lines[0])["header"]["format"] == "cam-unified-lifetime/1"
    assert not any("header" in json.loads(x) for x in lines[1:])         # one lifetime, one header
    return [EventRecord.from_dict(json.loads(x)) for x in lines[1:]]


def _view(r):
    """What a record says, without the random ids (action ids are new per process)."""
    return (r.step, r.t_sim, r.action.tool if r.action else None, json.dumps(r.action.args if r.action else None,
                                                                             sort_keys=True),
            r.result.ok if r.result else None, r.extra["dt_days"], r.reward, len(r.observations))


def test_checkpoint_resume_reproduces_uninterrupted_lifetime(tmp_path):
    full, dig_full, t_full, a_full = _life(f"sqlite:///{tmp_path}/a.db", str(tmp_path / "ck_a"),
                                           transcript=str(tmp_path / "a.jsonl"))
    assert len(full.checkpoints) >= 2
    # interrupted after ~1.5 days of decisions (well after its last checkpoint), then a fresh operator / adapter /
    # agent resumes from disk; the steps the interrupted run wrote after that checkpoint must not survive
    tb = str(tmp_path / "b.jsonl")
    part, _, t_part, _ = _life(f"sqlite:///{tmp_path}/b.db", str(tmp_path / "ck_b"), stop=95, transcript=tb)
    assert 1.0 <= t_part / 86400 < 3.0 and part.checkpoints
    n_interrupted = len(_records(tb))
    rest, dig_rest, t_rest, a_rest = _life(f"sqlite:///{tmp_path}/b.db", str(tmp_path / "ck_b"), resume=True,
                                           transcript=tb)
    assert rest.resumed_from is not None
    assert t_rest == t_full and dig_rest == dig_full
    assert rest.steps == full.steps and a_rest.k == a_full.k and a_rest.model_version == a_full.model_version
    assert rest.returns == pytest.approx(full.returns)
    # transcript: every step exactly once, identical to the uninterrupted lifetime's
    ra, rb = _records(str(tmp_path / "a.jsonl")), _records(tb)
    steps = [r.step for r in rb]
    assert steps == list(range(full.steps)) and len(set(steps)) == len(steps)
    assert n_interrupted == 95 and len(rb) == full.steps
    assert [_view(r) for r in rb] == [_view(r) for r in ra]
    # a resume whose transcript lost data since the checkpoint is refused rather than silently forked
    with open(tb, "r+b") as fh:
        fh.truncate(10)
    with pytest.raises(ValueError):
        _life(f"sqlite:///{tmp_path}/b.db", str(tmp_path / "ck_b"), resume=True, transcript=tb)


def test_shared_world_records_elapsed_semi_markov_duration():
    """Shared world: one synchronous round lasts as long as its longest action; every row's dt is the time that
    actually elapsed for it, the duration it asked for is kept separately."""
    goal = tiny_goal(days=2)
    ad, op = open_world([goal, goal], shared_world=True, seed=1)
    ev = OpenWorldEvaluator(op)

    class Two:
        def act(self, obs, tools):
            return [Action("bank.accounts", {}), Action(WAIT_TOOL, {"seconds": 43200})]

        def observe(self, tr):
            pass

        def update(self):
            return {}
    seen = []
    run_lifetime(Two(), ad, ev, max_decisions=3, think_s=600, on_step=lambda k, tr: seen.append(tr))
    ad.close()
    assert len(seen) == 3
    for tr in seen:
        for b, r in enumerate(tr.records):
            elapsed = tr.next_obs.t[b] - r.t_sim
            assert r.extra["dt_days"] == pytest.approx(elapsed) and tr.dt[b] == pytest.approx(elapsed)
            assert elapsed == pytest.approx((43200 + 600) / 86400)          # the round's longest action
        req0 = tr.records[0].extra["dt_requested_days"]
        assert req0 == pytest.approx(tr.records[0].result.duration + 600 / 86400) and req0 < 0.01
        assert tr.records[1].extra["dt_requested_days"] == pytest.approx((43200 + 600) / 86400)


def test_isolated_operator_checkpoint_resume(tmp_path):
    """The learner-run constructor (worlds + evaluator in another process) checkpoints and resumes through
    RemoteOperator / RemoteEvaluator exactly like the in-process operator."""
    from superlearner.cam_unified.envs.openworld import open_world_isolated

    def life(store, ck, stop=None, resume=False, transcript=None):
        ad, rop, rev = open_world_isolated([tiny_goal(days=4)], evaluator={}, seed=7, store_url=store)
        agent = PollAgent()
        try:
            res = run_lifetime(agent, ad, rev, days=2.0, think_s=600, update_every=10, operator=rop,
                               checkpoint_dir=ck, checkpoint_every_days=1.0, resume=resume,
                               stop_after_decisions=stop, transcript=transcript)
        finally:
            ad.close()
        return res, agent
    full, a_full = life(f"sqlite:///{tmp_path}/a.db", str(tmp_path / "ck_a"), transcript=str(tmp_path / "a.jsonl"))
    tb = str(tmp_path / "b.jsonl")
    part, _ = life(f"sqlite:///{tmp_path}/b.db", str(tmp_path / "ck_b"), stop=90, transcript=tb)
    assert part.checkpoints and 1.0 <= part.t_days[0] < 2.0
    rest, a_rest = life(f"sqlite:///{tmp_path}/b.db", str(tmp_path / "ck_b"), resume=True, transcript=tb)
    assert rest.resumed_from is not None and rest.steps == full.steps and rest.t_days == full.t_days
    assert rest.returns == pytest.approx(full.returns) and a_rest.k == a_full.k
    assert [_view(r) for r in _records(tb)] == [_view(r) for r in _records(str(tmp_path / "a.jsonl"))]


def test_heterogeneous_population_in_one_shared_world():
    """Three independently controlled participants in one world under one operator clock: a competent seller, a
    random actor and an idle one, each driven by its own agent through the shared-world adapter."""
    from crucible.agents.scripted import IdlePolicy, RandomPolicy
    from superlearner.cam_unified.envs.agent_protocol import MixedAgent
    goal = tiny_goal(days=6)
    ad, op = open_world([goal, goal, goal], shared_world=True, seed=5)
    ev = OpenWorldEvaluator(op)
    agent = MixedAgent([(PolicyAgent([ExampleInductor()]), [0]),
                        (PolicyAgent([RandomPolicy()]), [1]),
                        (PolicyAgent([IdlePolicy()]), [2])])
    res = run_lifetime(agent, ad, ev, think_s=600)
    w = op.world(0)
    acts = {k: p.get("actions", 0) for k, p in w.state.scan("principals")}
    ad.close()
    assert len(op.slots) == 1
    assert res.final[0]["achieved"] and not res.final[1]["achieved"] and not res.final[2]["achieved"]
    assert acts[op.principal(0)] >= 15 and acts[op.principal(1)] >= 1 and acts[op.principal(2)] == 0
    assert res.t_days[0] == res.t_days[1] == res.t_days[2]          # one clock
