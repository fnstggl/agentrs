"""Central learner details (manifest F02/F03, spec 14.3/14.4): per-decision policy lag inside mixed-version
trajectories, continuous training of long episodes in bootstrapped segments, honest accounting of what the model
trained, durable consumption with rollback on restart, unpublished versions above LATEST, orphan and late outcomes,
and summed delayed outcomes in the traces."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest
import torch

from superlearner.cam_unified.dist.actor import ActorProcess, ActorSpec
from superlearner.cam_unified.dist.learner import CentralLearner, Trajectory, TrajectoryAssembler
from superlearner.cam_unified.dist.offpolicy import LagRouter, vtrace
from superlearner.cam_unified.dist.publish import CheckpointPublisher, read_latest
from superlearner.cam_unified.dist.queue import ExperienceQueue
from superlearner.cam_unified.dist.sinks import JsonlArchive
from superlearner.cam_unified.dist.toyworld import make_tiny_learner
from superlearner.cam_unified.interfaces import Action, EventRecord, Observation, ObsKind, new_id
from superlearner.cam_unified.observe.trace import load_traces

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


class _Recorder:
    """Test double of LearnerModelP that records exactly what each update received."""

    def __init__(self, reply=None):
        self.w = torch.zeros(3)
        self.batches = []
        self.reply = reply

    def state_dict(self):
        return {"w": self.w.clone()}

    def load_state_dict(self, sd):
        self.w = sd["w"].clone()

    def update(self, batch):
        self.batches.append([{"lags": list(t.lags), "mask": list(t.train_mask), "exact": list(t.exact),
                              "steps": [e.step for e in t.events], "bootstrap": t.bootstrap, "terminal": t.terminal,
                              "ids": t.trained_ids(), "route": t.route} for t in batch.trajectories])
        self.w += 1.0
        return dict(self.reply(batch)) if self.reply else {"n": len(batch.trajectories)}


def _dec(actor, ep, step, version, done=False, logp=-1.2, extra=None, tool="lever_1", feat=0.0):
    obs = [Observation(ObsKind.NUMERIC, value=feat + i, meta={"i": i}) for i in range(8)]
    return EventRecord(event_id=new_id("e_"), env_id=f"{actor}.0", actor_id=actor, episode_id=ep, step=step,
                       model_version=version, t_sim=float(step), observations=obs, action=Action(tool, logp=logp),
                       behavior_logp=logp, done=done, extra={"dt": 1 / 24, **(extra or {})})


def _payout(dec, value=1.0):
    return EventRecord(event_id=new_id("e_"), env_id=dec.env_id, actor_id=dec.actor_id, episode_id=dec.episode_id,
                       step=dec.step, model_version=dec.model_version, t_sim=dec.t_sim + 0.1,
                       observations=[Observation(ObsKind.EVENT, f"payout {value}", value=value,
                                                 meta={"key": "payoff"})],
                       attributed_to=dec.event_id, extra={"kind": "delayed_outcome"})


def _learner(run, model, **kw):
    kw.setdefault("max_lag", 4)
    return CentralLearner(run, model, sink=JsonlArchive(os.path.join(run, "archive")), **kw)


def _advance(L, w, n_updates, actor="warm"):
    """Bring the learner to a version with fresh complete on-policy episodes of 4 decisions."""
    for _ in range(n_updates):
        ep = new_id("ep_")
        for i in range(4):
            w.put(_dec(actor, ep, i, L.version, done=(i == 3)))
        w.flush()
        assert L.step() is not None


def test_mixed_version_trajectory_is_routed_per_decision(tmp_path):
    """Decision-boundary refresh inside one trajectory: versions [0]*5 + [7]*3 trained at learner version 7 with
    max_lag 4 -> the 5 stale decisions (lag 7) are not trained (recurrent context only), the 3 fresh ones are; only
    fresh decisions whose latent state was built under version 7 count as exactly on-policy."""
    run = str(tmp_path / "run")
    rec = _Recorder()
    L = _learner(run, rec, min_decisions=4, max_batch_decisions=4)
    w = ExperienceQueue(os.path.join(run, "spool")).writer("a", batch_records=1000)
    _advance(L, w, 7)
    assert L.version == 7
    # A: refreshed to v7 at step 5 and the latent state was rebuilt under v7 -> the fresh decisions are exact
    # B: refreshed to v8 at step 5 without rebuilding (state still built under v0) -> importance ratio at lag 0
    for fresh_v, state_fresh, exact_fresh in ((7, 7, True), (8, 0, False)):
        assert L.version == fresh_v
        ep = new_id("ep_")
        recs = [_dec("mix", ep, i, 0 if i < 5 else fresh_v, done=(i == 7),
                     extra={"state_version": 0 if i < 5 else state_fresh}) for i in range(8)]
        for r in recs:
            w.put(r)
        w.flush()
        out = L.step()
        assert out is not None
        b = rec.batches[-1][0]
        assert b["lags"] == [fresh_v] * 5 + [0] * 3           # true per-decision lags (was max-version: all 0)
        assert b["mask"] == [False] * 5 + [True] * 3           # lag > max_lag: recurrent context only
        assert b["exact"] == [False] * 5 + [exact_fresh] * 3
        assert b["ids"] == [r.event_id for r in recs[5:]] and b["route"] == "on_policy"
        assert out["event_ids"] == [r.event_id for r in recs[5:]] and out["n_decisions"] == 3
        assert out["lag"]["n_discarded"] == 5 and out["lag"]["discard_reasons"] == {"lag_over_max": 5}
        assert out["lag"]["n_exact_on_policy"] == (3 if exact_fresh else 0)
    skipped = [json.loads(l) for l in open(os.path.join(run, "learner", "skipped.jsonl"))]
    assert sum(len(s["event_ids"]) for s in skipped if s["reason"] == "lag_over_max") == 10
    assert L.router.totals()["counts"]["discard:lag_over_max"] == 10
    L.close()


def test_router_exactness_rules():
    R = LagRouter(max_lag=8)
    ep = "ep"
    uniform = [_dec("a", ep, i, 5) for i in range(4)]
    tr = Trajectory("a", ep, "a.0", events=uniform)
    R.route_trajectories([tr], 5)
    assert tr.exact == [True] * 4 and tr.route == "on_policy"
    greedy = [_dec("a", ep, i, 5, logp=0.0, extra={"deterministic": True, "state_version": 5}) for i in range(4)]
    tg = Trajectory("a", ep, "a.0", events=greedy)
    R.route_trajectories([tg], 5)
    assert tg.exact == [False] * 4 and tg.n == 4      # deterministic behaviour: trained with ratio pi/1
    mid = [_dec("a", ep, i + 40, 5, extra={"state_version": 5}) for i in range(4)]
    tm = Trajectory("a", ep, "a.0", events=mid)
    R.route_trajectories([tm], 5)
    assert tm.exact == [False] * 4                    # segment re-scored from a burn-in, not from the episode start
    strict = LagRouter(max_lag=8, allow_deterministic=False)
    tg2 = Trajectory("a", ep, "a.0", events=greedy)
    rep = strict.route_trajectories([tg2], 5)
    assert tg2.route == "discarded" and rep.reasons == {"deterministic_behavior": 4}


def test_tiny_model_uses_importance_ratios_where_not_exact():
    m = make_tiny_learner(seed=0)
    ep = "ep"
    recs = [_dec("a", ep, i, 0 if i < 5 else 7, done=(i == 7), logp=math.log(0.25),
                 extra={"state_version": 0 if i < 5 else 7}, tool=f"lever_{i % 4}") for i in range(8)]
    tr = Trajectory("a", ep, "a.0", events=recs, outcomes={r.event_id: [_payout(r)] for r in recs}, terminal=True)
    rep = LagRouter(max_lag=8).route_trajectories([tr], 7)
    assert tr.lags == [7] * 5 + [0] * 3 and tr.exact == [False] * 5 + [True] * 3
    from superlearner.cam_unified.dist.learner import LearnerBatch
    met = m.update(LearnerBatch(7, [tr], rep.summary()))
    assert met["optimizer_step"] and met["n_steps"] == 8 and met["n_exact_steps"] == 3 and met["n_ratio_steps"] == 5
    assert met["n_offpolicy_steps"] == 5 and met["rho_mean_offpolicy"] is not None
    assert met["rho_mean_offpolicy"] <= 1.0          # clipped importance weights


def test_vtrace_cut_bootstraps_from_the_next_value():
    """A non-trained step (log rho = CUT) after the trained ones makes its own value the bootstrap target."""
    from superlearner.cam_unified.dist.offpolicy import CUT_LOG_RHO
    r = torch.tensor([[1.0, 2.0, 99.0]])
    v = torch.tensor([[0.5, 0.25, 4.0]])
    disc = torch.tensor([[0.9, 0.9, 0.9]])
    lr = torch.tensor([[0.0, 0.0, CUT_LOG_RHO]])
    vs, _, rho = vtrace(r, v, torch.zeros(1), disc, lr)
    assert rho[0, 2] == 0 and vs[0, 2] == pytest.approx(4.0)                   # bootstrap step: its own value
    assert vs[0, 1] == pytest.approx(2.0 + 0.9 * 4.0)                          # bootstrapped, reward 99 unused
    assert vs[0, 0] == pytest.approx(1.0 + 0.9 * (2.0 + 0.9 * 4.0))


def test_long_episode_trains_continuously_in_bootstrapped_segments(tmp_path):
    """Verifier probe P5: one 100-decision episode in 8-record segments with min_decisions 4 used to stall at the
    backlog limit until stale_s. Now segments of the running episode are trained as they settle; every decision is
    trained exactly once; burn-in and bootstrap steps are never trained."""
    run = str(tmp_path / "run")
    rec = _Recorder()
    # max_lag 100: the whole episode is acted by v0 while the learner updates after every segment (lag routing of
    # such data is tested above; here every segment must be trainable)
    L = _learner(run, rec, min_decisions=4, max_batch_decisions=16, segment_len=16, burn_in=4, settle=2,
                 ingest_segments=1, max_lag=100)
    w = ExperienceQueue(os.path.join(run, "spool")).writer("a", batch_records=8)
    ep = new_id("ep_")
    decs = []
    for i in range(100):
        d = _dec("a", ep, i, 0, done=(i == 99))
        decs.append(d)
        w.put(d)
        if i >= 1:
            w.put(_payout(decs[i - 1]))
    w.flush()
    upd_while_running = 0
    for k in range(200):
        L.step()
        if L.queue.pending():                       # the episode's final records are still in the spool
            upd_while_running = L.counts["updates"]
    assert not L.queue.pending()
    assert upd_while_running >= 3                   # trained while the episode was still running
    assert L.counts.get("ingest_paused", 0) == 0
    trained = [i for b in rec.batches for t in b for i in t["ids"]]
    assert sorted(trained) == sorted(d.event_id for d in decs)      # every decision exactly once
    segs = [t for b in rec.batches for t in b]
    mids = [t for t in segs if t["bootstrap"]]
    assert len(mids) == 6 and segs[-1]["terminal"] and not segs[-1]["bootstrap"]
    for t in mids:                      # [burn-in (<= 4, not trained)] + 16 trained + 1 bootstrap (not trained)
        assert t["mask"][-1] is False and sum(t["mask"]) == 16
        nb = len(t["mask"]) - 17
        assert nb == (0 if t["steps"][0] == 0 else 4) and t["mask"][:nb] == [False] * nb
        assert t["steps"] == list(range(t["steps"][0], t["steps"][0] + len(t["steps"])))
    assert L.assembler.held_events() == 0 and L.backlog() == 0
    L.close()


def test_long_actor_episodes_train_with_the_tiny_model(tmp_path):
    run = str(tmp_path / "run")
    m = make_tiny_learner(seed=0, lr=1e-2)
    L = CentralLearner(run, m, sink=JsonlArchive(os.path.join(run, "archive")), min_decisions=64,
                       max_batch_decisions=64, segment_len=32, burn_in=8, settle=2, ingest_segments=1)
    a = ActorProcess(ActorSpec("a0", run, env_kwargs={"B": 2, "episode_len": 200}, seed=5, max_episodes=1,
                               batch_records=40))
    a.run()
    ups = []
    while True:
        r = L.step()
        if r is not None:
            ups.append(r)
        elif not L.queue.pending() and L.backlog() < 64:
            break
    assert sum(u["n_decisions"] for u in ups) == 400 - L.backlog()
    assert any(u["n_segments_mid_episode"] > 0 for u in ups)
    mets = [u["metrics"] for u in ups]
    assert sum(x["n_bootstrap_cuts"] for x in mets) > 0 and sum(x["n_context_steps"] for x in mets) > 0
    assert all(x["optimizer_step"] for x in mets)
    L.close()


def test_model_no_step_and_model_skipped_decisions_are_not_counted_as_trained(tmp_path):
    run = str(tmp_path / "run")
    calls = {"n": 0}

    def reply(batch):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"optimizer_step": False, "reason": "nothing usable"}
        first = batch.trajectories[0].trained_ids()
        return {"loss": 0.0, "skipped_event_ids": first[:2]}
    rec = _Recorder(reply)
    L = _learner(run, rec, min_decisions=4, max_batch_decisions=4)
    w = ExperienceQueue(os.path.join(run, "spool")).writer("a", batch_records=1000)
    for _ in range(2):
        ep = new_id("ep_")
        for i in range(4):
            w.put(_dec("a", ep, i, 0, done=(i == 3)))
        w.flush()
    v0 = L.version
    assert L.step() is None                          # no optimiser step: no version, no publication
    assert L.version == v0 and read_latest(L.paths["ckpt"])["version"] == v0
    r = L.step()
    assert r is not None and L.version == v0 + 1
    assert r["n_decisions"] == 2 and len(r["event_ids"]) == 2
    assert L.counts["trained_decisions"] == 2
    sk = [json.loads(l) for l in open(L.skipped_path)]
    assert {s["reason"]: len(s["event_ids"]) for s in sk} == {"model_no_step": 4, "model_skipped": 2}
    L.close()


def test_crash_before_the_first_update_loses_nothing(tmp_path):
    """Verifier probe P3: records ingested before the first update are re-assembled after a restart."""
    run = str(tmp_path / "run")
    L = _learner(run, _Recorder(), min_decisions=64)
    w = ExperienceQueue(os.path.join(run, "spool")).writer("a", batch_records=1000)
    for _ in range(4):
        ep = new_id("ep_")
        for i in range(4):
            w.put(_dec("a", ep, i, 0, done=(i == 3)))
        w.flush()
    L.step()
    assert L.counts["updates"] == 0 and L.sink.count() == 16
    L.close()
    L2 = _learner(run, _Recorder(), min_decisions=16)
    assert L2.counts["recovered_records"] == 16
    r = L2.step()
    assert r is not None and r["n_decisions"] == 16
    L2.close()


def test_unpublished_updates_are_rolled_back_and_retrained_after_a_crash(tmp_path):
    run = str(tmp_path / "run")
    rec = _Recorder()
    L = _learner(run, rec, min_decisions=4, max_batch_decisions=4, publish_every=2)
    w = ExperienceQueue(os.path.join(run, "spool")).writer("a", batch_records=1000)
    _advance(L, w, 2)                                 # v1 (unpublished), v2 (published)
    ep = new_id("ep_")
    ids = []
    for i in range(4):
        d = _dec("a", ep, i, L.version, done=(i == 3))
        ids.append(d.event_id)
        w.put(d)
    w.flush()
    assert L.step() is not None and L.version == 3    # trained into v3, which is never published
    assert read_latest(L.paths["ckpt"])["version"] == 2
    L.close()                                         # crash before v4 would have been published
    rec2 = _Recorder()
    L2 = _learner(run, rec2, min_decisions=4, max_batch_decisions=4, publish_every=2)
    assert L2.version == 2 and L2.counts["rolled_back_records"] == 4
    r = L2.step()
    assert r is not None and r["event_ids"] == ids and L2.version == 3     # re-trained, not lost
    ups = [json.loads(l) for l in open(L2.updates_path)]
    assert any(u.get("rollback") and u["rolled_back_versions"] == [3] for u in ups)
    L2.close()


def test_restart_after_a_rejected_version_above_latest(tmp_path):
    """Verifier probe P2: LATEST at v3, a rejected v4 on disk; the restarted learner must publish v4."""
    run = str(tmp_path / "run")
    L = _learner(run, _Recorder(), min_decisions=4, max_batch_decisions=4)
    w = ExperienceQueue(os.path.join(run, "spool")).writer("a", batch_records=1000)
    _advance(L, w, 3)
    m = CheckpointPublisher(L.paths["ckpt"], validate_fn=lambda sd, meta: (False, {"probe": "worse"})).publish(
        {"w": torch.full((3,), 9.0)}, 4)
    assert not m["validated"] and read_latest(L.paths["ckpt"])["version"] == 3
    L.close()
    L2 = _learner(run, _Recorder(), min_decisions=4, max_batch_decisions=4)
    assert L2.version == 3 and L2.publisher.last_version == 3 and L2.publisher.unpublished == [4]
    assert any(f.startswith(".unpublished-v00000004-") for f in os.listdir(L2.paths["ckpt"]))
    _advance(L2, w, 1)
    assert read_latest(L2.paths["ckpt"])["version"] == 4
    L2.close()


_CRASH_PUB = r"""
import os, sys
sys.path.insert(0, {src!r})
import torch
from superlearner.cam_unified.dist.publish import CheckpointPublisher
pub = CheckpointPublisher({root!r})
for v in range(3):
    pub.publish({{"w": torch.full((4,), float(v))}}, v)
CheckpointPublisher._validate = lambda *a, **k: os._exit(9)      # die after the rename, before LATEST
pub.publish({{"w": torch.full((4,), 3.0)}}, 3)
"""


def test_publisher_killed_between_rename_and_latest(tmp_path):
    root = str(tmp_path / "ckpt")
    p = subprocess.run([sys.executable, "-c", _CRASH_PUB.format(src=SRC, root=root)])
    assert p.returncode == 9
    assert os.path.isdir(os.path.join(root, "v00000003")) and read_latest(root)["version"] == 2
    pub = CheckpointPublisher(root)
    assert pub.last_version == 2 and pub.unpublished == [3] and pub.versions() == [0, 1, 2]
    m = pub.publish({"w": torch.full((4,), 3.5)}, 3)
    assert m["validated"] and read_latest(root)["version"] == 3


def test_orphan_and_late_outcomes_are_resolved_not_dropped(tmp_path):
    run = str(tmp_path / "run")
    L = _learner(run, _Recorder(), min_decisions=4, max_batch_decisions=4, waiting_ttl_s=-1.0)
    L.assembler.trained_cap = 2                       # forget trained ids quickly: the durable index must answer
    w = ExperienceQueue(os.path.join(run, "spool")).writer("a", batch_records=1000)
    ep = new_id("ep_")
    decs = [_dec("a", ep, i, 0, done=(i == 3)) for i in range(4)]
    for d in decs:
        w.put(d)
    w.flush()
    assert L.step() is not None
    assert decs[0].event_id not in L.assembler.trained      # pruned FIFO: oldest first
    assert decs[3].event_id in L.assembler.trained
    late = _payout(decs[0])                           # arrives after its decision was trained and forgotten
    ghost = _payout(_dec("a", ep, 9, 0))              # attributed to a decision that never arrived
    w.put(late)
    w.put(ghost)
    w.flush()
    L.step()
    assert L.counts["late_outcomes_resolved"] == 1 and L.counts["orphan_outcomes"] == 1
    orph = [json.loads(l) for l in open(L.orphans_path)]
    assert [o["event_id"] for o in orph] == [ghost.event_id]
    assert L.assembler.late and L.assembler.late[0].event_id == late.event_id
    L.close()


def test_several_delayed_outcomes_of_one_decision_are_summed_in_the_trace(tmp_path):
    from superlearner.cam_unified.interfaces import DecisionTrace
    run = str(tmp_path / "run")
    L = _learner(run, _Recorder(), min_decisions=100)
    w = ExperienceQueue(os.path.join(run, "spool")).writer("a", batch_records=1000)
    d = _dec("a", "ep1", 0, 0)
    w.put(d, DecisionTrace(decision_id="d1", event_id=d.event_id, env_id=d.env_id, policy_version=0, t_sim=0.0,
                           objective="", predicted={"payoff": 0.5}, outcome={"reward": 0.0}))
    w.put(_payout(d, 1.0))
    w.put(_payout(d, 2.5))
    w.flush()
    L.step()
    L.traces.flush()
    (t,) = load_traces(L.paths["traces"], env_id=d.env_id)
    assert t.outcome["payoff"] == pytest.approx(3.5) and len(t.outcome["delayed"]) == 2
    L.close()


def test_assembler_memory_is_bounded_for_a_long_lifetime():
    A = TrajectoryAssembler(segment_len=16, burn_in=4, settle=2)
    ep = "ep"
    peak = 0
    for i in range(5000):
        A.add([_dec("a", ep, i, 0)])
        A.pop_ready()
        peak = max(peak, A.held_events())
    assert peak <= 16 + 4 + 2 and A.counts["segments"] == (5000 - 2) // 16
