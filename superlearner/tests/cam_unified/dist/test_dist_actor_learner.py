"""Actors, central learner and launcher (manifest A16, F02, F03): versioned records with delayed outcomes and traces,
latent-state reconstruction at decision-boundary refreshes, cross-actor learning through the shared weights in a real
multiprocess run (with a no-learning control), crash/restart supervision and learner resume."""
from __future__ import annotations

import json
import os
import signal
import time
from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F

from superlearner.cam_unified.dist.actor import ActorProcess, ActorSpec
from superlearner.cam_unified.dist.launcher import Launcher
from superlearner.cam_unified.dist.learner import CentralLearner, LearnerSpec
from superlearner.cam_unified.dist.publish import CheckpointPublisher, read_latest
from superlearner.cam_unified.dist.queue import ExperienceQueue
from superlearner.cam_unified.dist.sinks import JsonlArchive
from superlearner.cam_unified.dist.toyworld import TinyRecurrentPolicy, featurize, make_tiny_learner
from superlearner.cam_unified.observe.trace import load_traces


def _ingest_all(run):
    q = ExperienceQueue(os.path.join(run, "spool"))
    sink = JsonlArchive(os.path.join(run, "archive_test"))
    c = q.consumer(lazy_traces=False)
    events, traces = [], []
    while True:
        r = c.ingest(sink, max_segments=100)
        if r.empty:
            break
        events += r.events
        traces += r.traces
    return events, traces


def test_actor_writes_versioned_records_delayed_outcomes_and_traces(tmp_path):
    run = str(tmp_path / "run")
    CheckpointPublisher(os.path.join(run, "ckpt")).publish(make_tiny_learner(seed=0).state_dict(), 0)
    spec = ActorSpec("actor7", run, env_kwargs={"B": 3, "delay": 2, "episode_len": 10}, seed=7, max_episodes=4,
                     batch_records=50, think_days=0.01)
    a = ActorProcess(spec)
    stats = a.run()
    assert stats["episodes"] == 4 and stats["decisions"] == 4 * 3 * 10
    events, traces = _ingest_all(run)
    dec = [e for e in events if e.action is not None]
    delayed = [e for e in events if e.attributed_to]
    assert len(dec) == 120 and len(delayed) == 120              # every pull pays out exactly once, later
    by_id = {e.event_id: e for e in dec}
    for e in dec:
        assert e.model_version == 0 and e.actor_id == "actor7"
        assert e.behavior_logp == e.action.logp and e.behavior_logp < 0
        assert e.extra["dt"] == pytest.approx(1 / 24 + 0.01)    # tool duration + think time (SMDP)
        assert e.provenance.value == "real"
    for d in delayed:
        cause = by_id[d.attributed_to]                           # attributed to an existing decision ...
        obs = d.observations[0]
        assert obs.caused_by == cause.action.action_id           # ... whose action produced the payout
        assert d.episode_id == cause.episode_id and d.step >= cause.step
        assert d.action is None and d.extra["kind"] == "delayed_outcome"
    # verified reward: per lifetime-episode, the rewards sum to the evaluator's final score
    ep_reward, ep_pay = defaultdict(float), defaultdict(float)
    for e in dec:
        ep_reward[(e.env_id, e.episode_id)] += e.reward
    for d in delayed:
        ep_pay[(d.env_id, d.episode_id)] += d.observations[0].value
    for k in ep_reward:
        assert ep_reward[k] == pytest.approx(ep_pay[k] / 10)
    assert {e.env_id for e in dec} == {"toy7.0", "toy7.1", "toy7.2"}
    # every decision has a trace linked by event id, with candidates, predictions and imagined alternatives
    assert len(traces) == 120
    for t in traces:
        e = by_id[t.event_id]
        assert t.policy_version == 0 and t.selected["tool"] == e.action.tool
        assert t.outcome["reward"] == pytest.approx(e.reward)
        assert len(t.candidates) == 4 and len(t.imagined) == 4 and "payoff" in t.predicted
        assert 0.0 <= t.predicted["payoff"] <= 1.0


class _MidEpisodePublisher(TinyRecurrentPolicy):
    """Test policy: publishes new weights from inside an episode (as the learner would at any time) and checks the
    reconstructed latent state against an independent replay through the acting path."""

    def __init__(self, ckpt, at_call, **kw):
        super().__init__(**kw)
        self.ckpt, self.at_call, self.calls = ckpt, at_call, 0
        self.new_sd = {k: v + 0.3 * torch.randn_like(v) for k, v in self.net.state_dict().items()}
        self.checks = []

    def act(self, obs_lists, tools, goals, active=None):
        self.calls += 1
        if self.calls == self.at_call:
            CheckpointPublisher(self.ckpt).publish(self.new_sd, 1)
        return super().act(obs_lists, tools, goals, active=active)

    def rebuild_state(self, history):
        super().rebuild_state(history)
        ref = TinyRecurrentPolicy(seed=99)
        ref.load_weights(self.new_sd, 1)
        ref.h = ref.net.init_h(len(history))
        ref.a_prev = torch.zeros(len(history), ref.net.n_tools)
        for t in range(len(history[0])):
            ref.act([h[t].observations for h in history], [[None] * 4] * len(history), None)
            k = torch.tensor([int(h[t].action.tool.split("_")[-1]) for h in history])
            ref.a_prev = F.one_hot(k, ref.net.n_tools).float()
        self.checks.append(float((ref.h - self.h).abs().max()))


@pytest.mark.parametrize("threshold,expect_rebuild", [(0.0, True), (1e9, False)])
def test_decision_boundary_refresh_reconstructs_latent_state(tmp_path, threshold, expect_rebuild, monkeypatch):
    run = str(tmp_path / "run")
    ckpt = os.path.join(run, "ckpt")
    CheckpointPublisher(ckpt).publish(make_tiny_learner(seed=0).state_dict(), 0)
    import superlearner.cam_unified.dist.toyworld as tw
    monkeypatch.setattr(tw, "make_tiny_policy", lambda seed=0, **kw: _MidEpisodePublisher(ckpt, 5, seed=seed, **kw),
                        raising=True)
    # the toy policy's act() indexes tools by name: give the reference replay real tool lists
    orig_act = TinyRecurrentPolicy.act

    def act(self, obs_lists, tools, goals, active=None):
        if tools and tools[0] and tools[0][0] is None:
            tools = [[type("T", (), {"name": f"lever_{k}"})() for k in range(4)] for _ in tools]
        return orig_act(self, obs_lists, tools, goals, active=active)
    monkeypatch.setattr(TinyRecurrentPolicy, "act", act)
    spec = ActorSpec("a0", run, env_kwargs={"B": 2, "episode_len": 12}, seed=3, max_episodes=1,
                     refresh_at="decision", rebuild_threshold=threshold)
    a = ActorProcess(spec)
    a.run()
    pol = a.policy
    assert a.stats["refreshes"] == 1 and a.client.version == 1
    events, _ = _ingest_all(run)
    versions = [e.model_version for e in sorted((e for e in events if e.action is not None and e.env_id.endswith(".0")),
                                                key=lambda e: e.step)]
    assert versions == [0] * 5 + [1] * 7          # adopted between decisions 5 and 6 of the same trajectory
    if expect_rebuild:
        assert a.stats["rebuilds"] == 1 and len(pol.checks) == 1
        assert pol.checks[0] < 1e-5               # rebuilt latent == replay of the raw episode under new weights
        assert a.client.drift == 0.0
    else:
        assert a.stats["rebuilds"] == 0 and pol.checks == []
        assert a.client.drift > 0


def _archive_episodes(run, actor):
    sink = JsonlArchive(os.path.join(run, "archive"))
    eps = defaultdict(lambda: {"reward": 0.0, "version": None, "t": None, "n": 0})
    for batch in sink.scan(actor_id=actor):
        for e in batch:
            if e.action is None:
                continue
            ep = eps[e.episode_id]
            ep["reward"] += e.reward
            ep["version"] = e.model_version
            ep["t"] = e.t_wall if ep["t"] is None else min(ep["t"], e.t_wall)
            ep["n"] += 1
    sink.close()
    return sorted((v for v in eps.values() if v["n"] == 16), key=lambda v: v["t"])   # complete env-episodes


def _cross_run(run, train: bool, seconds: float):
    L = LearnerSpec(run_dir=run, model_kwargs={"seed": 0, "lr": 1e-2, "train": train}, min_decisions=128,
                    max_batch_decisions=256, train_actors=["actor1"], max_lag=16)
    A = [ActorSpec(actor_id=a, run_dir=run, env_kwargs={"B": 4, "mechanism_seed": 0}, seed=s, batch_records=192)
         for a, s in (("actor1", 1), ("actor2", 2))]
    la = Launcher(run, L, A)
    return la.run(duration_s=seconds)


def test_one_actors_experience_changes_another_actors_behaviour(tmp_path):
    """A16 mechanism: the learner trains only on actor1's experience; actor2 (a different process, own seed and
    worlds, never trained on) behaves differently after adopting the shared weights. Control: identical run whose
    learner publishes versions without optimising."""
    run = str(tmp_path / "learn")
    s = _cross_run(run, True, 12.0)
    assert s["restarts"] == {} and s["learner_version"] >= 5
    ups = [json.loads(l) for l in open(os.path.join(run, "learner", "updates.jsonl"))]
    trained = [u for u in ups if u["trained"]]
    assert trained and all(set(u["by_actor"]) == {"actor1"} for u in trained)
    sink = JsonlArchive(os.path.join(run, "archive"))
    consumed = [i for u in trained for i in u["event_ids"]]
    assert {e.actor_id for e in sink.get(consumed[:500])} == {"actor1"}   # the updates used actor1's records only
    assert sink.count() > 0 and any(e.actor_id == "actor2" for b in sink.scan(actor_id="actor2") for e in b)
    sink.close()
    eps = _archive_episodes(run, "actor2")
    early = [e["reward"] for e in eps if e["version"] == 0]
    late = [e["reward"] for e in eps[-16:]]
    assert len(early) >= 1 and len(late) == 16
    assert max(e["version"] for e in eps[-16:]) >= 5          # actor2 ran the newer shared weights
    m_early, m_late = sum(early) / len(early), sum(late) / len(late)

    ctrl = str(tmp_path / "control")
    _cross_run(ctrl, False, 12.0)
    ceps = _archive_episodes(ctrl, "actor2")
    c_late = [e["reward"] for e in ceps[-16:]]
    c_m_late = sum(c_late) / len(c_late)
    assert max(e["version"] for e in ceps[-16:]) >= 5          # versions advanced in the control too
    print(f"actor2 score: learning run early {m_early:.3f} -> late {m_late:.3f}; control late {c_m_late:.3f}")
    assert m_late > 0.6 and m_late - m_early > 0.3
    assert m_late - c_m_late > 0.3
    assert abs(c_m_late - 0.25) < 0.15                         # chance level for 4 tools without learning


def test_crashed_actor_is_restarted_and_queue_stays_consistent(tmp_path):
    run = str(tmp_path / "run")
    L = LearnerSpec(run_dir=run, model_kwargs={"seed": 0}, min_decisions=64, max_batch_decisions=256, stale_s=2.0)
    A = [ActorSpec(actor_id=f"a{i}", run_dir=run, env_kwargs={"B": 4}, seed=10 + i, batch_records=48)
         for i in range(2)]
    la = Launcher(run, L, A, max_restarts=2)
    la.start()
    t0 = time.time()
    killed = None
    while time.time() - t0 < 40:
        la.poll()
        st = la._read_status().get("a0", {})
        if killed is None and st.get("decisions", 0) > 300:
            killed = la.kill("a0", signal.SIGKILL)
        if killed and la.restarts.get("a0") == 1:
            st2 = la._read_status().get("a0", {})
            if st2.get("pid") not in (None, killed) and st2.get("decisions", 0) > 300:
                break
        time.sleep(0.1)
    time.sleep(3)                                       # let the learner see the dead incarnation's open episode
    v_before = read_latest(os.path.join(run, "ckpt"))["version"]
    lpid = la.kill("learner", signal.SIGKILL)           # the learner crashes too
    t1 = time.time()
    while time.time() - t1 < 60:
        la.poll()
        lst = la._read_status().get("learner", {})
        if la.restarts.get("learner") == 1 and lst.get("pid") not in (None, lpid) and \
                lst.get("version", -1) > v_before:
            break
        time.sleep(0.1)
    s = la.stop()
    assert s["restarts"] == {"a0": 1, "learner": 1}
    assert s["learner_counts"].get("resumed_from", -1) >= v_before   # resumed from LATEST, versions continue
    assert s["learner_version"] > v_before
    assert killed is not None
    evs = [json.loads(l) for l in open(os.path.join(run, "launcher.jsonl"))]
    assert [e["event"] for e in evs if e["name"] == "a0"][:3] == ["start", "exit", "restart"]
    assert any(e["event"] == "exit" and e["name"] == "a0" and e["exitcode"] == -signal.SIGKILL for e in evs)
    q = s["queue"]
    assert q["gaps"] == {} and q["totals"]["quarantined"] == 0
    sink = JsonlArchive(os.path.join(run, "archive"))
    ids = [e.event_id for b in sink.scan() for e in b]
    assert len(ids) == len(set(ids)) == sink.count() == q["seen_events"]
    a0 = [e for b in sink.scan(actor_id="a0") for e in b if e.action is not None]
    assert len({e.episode_id for e in a0}) >= 2
    sink.close()
    lc = s["learner_counts"]
    assert lc.get("updates", 0) > 0
    st = json.load(open(os.path.join(run, "status", "learner.json")))
    assert st["assembler"].get("stale_truncated", 0) + st["assembler"].get("episode_end", 0) >= 1


def test_learner_restart_resumes_from_latest_and_rebuilds_open_trajectories(tmp_path):
    run = str(tmp_path / "run")
    m = make_tiny_learner(seed=0, lr=1e-2)
    L = CentralLearner(run, m, sink=JsonlArchive(os.path.join(run, "archive")), min_decisions=64)
    a = ActorProcess(ActorSpec("a0", run, env_kwargs={"B": 4}, seed=1, max_episodes=6, batch_records=1000))
    a.run()
    for _ in range(20):
        L.step()
    v = L.version
    assert v >= 2
    # more experience arrives and is ingested, but the learner crashes before training on it
    a2 = ActorProcess(ActorSpec("a1", run, env_kwargs={"B": 4}, seed=2, max_episodes=1, batch_records=1000))
    a2.run()
    L.ingest()
    pending = L.backlog()
    assert pending == 64
    sd = {k: t.clone() for k, t in m.state_dict().items()}
    L.close()
    m2 = make_tiny_learner(seed=123)
    L2 = CentralLearner(run, m2, sink=JsonlArchive(os.path.join(run, "archive")), min_decisions=64)
    assert L2.version == v == read_latest(os.path.join(run, "ckpt"))["version"]
    assert all(torch.equal(m2.state_dict()[k], sd[k]) for k in sd)      # weights restored from the checkpoint
    assert L2.counts["recovered_records"] == 128                       # 64 decisions + 64 delayed outcomes
    r = L2.step()
    assert r is not None and r["by_actor"] == {"a1": 64}                # trained after the restart, not lost
    L2.close()


class _FlakyEnv:
    """EnvAdapter wrapper whose world raises once mid-episode (a crashed simulator)."""

    def __init__(self, inner, fail_at):
        self.inner, self.fail_at, self.n = inner, fail_at, 0
        self.env_ids = inner.env_ids

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def execute(self, actions):
        self.n += 1
        if self.n == self.fail_at:
            raise RuntimeError("simulator crashed")
        return self.inner.execute(actions)


def test_environment_failure_closes_episode_and_actor_continues(tmp_path, monkeypatch):
    import superlearner.cam_unified.dist.toyworld as tw
    made = []
    orig = tw.make_toy_env

    def factory(seed=0, **kw):
        env, ev = orig(seed=seed, **kw)
        made.append(env)
        return (_FlakyEnv(env, 5), ev) if len(made) == 1 else (env, ev)
    monkeypatch.setattr(tw, "make_toy_env", factory)
    run = str(tmp_path / "run")
    m = make_tiny_learner(seed=0)
    L = CentralLearner(run, m, sink=JsonlArchive(os.path.join(run, "archive")), min_decisions=16)
    a = ActorProcess(ActorSpec("a0", run, env_kwargs={"B": 2, "episode_len": 8}, seed=4, max_episodes=2,
                               batch_records=1000))
    st = a.run()
    assert st["env_failures"] == 1 and st["truncated_episodes"] == 2 and st["episodes"] == 2
    assert len(made) == 2 and "simulator crashed" in st["last_error"]
    L.ingest()
    trs = L.assembler.pop_ready()
    trunc = [t for t in trs if t.truncated]
    assert len(trunc) == 2 and all(t.n == 4 for t in trunc)          # 4 decisions made before the crash
    assert len([t for t in trs if not t.truncated]) == 4
    L.close()
