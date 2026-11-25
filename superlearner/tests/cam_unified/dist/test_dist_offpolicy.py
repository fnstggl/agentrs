"""Policy-lag tracking, routing and V-trace off-policy correction (manifest F03)."""
from __future__ import annotations

import json
import math
import os

import pytest
import torch

from superlearner.cam_unified.dist.learner import CentralLearner
from superlearner.cam_unified.dist.offpolicy import (LagRouter, importance_log_ratio, semi_markov_discount, vtrace)
from superlearner.cam_unified.dist.queue import ExperienceQueue
from superlearner.cam_unified.dist.sinks import JsonlArchive
from superlearner.cam_unified.interfaces import Action, EventRecord, new_id


def test_vtrace_on_policy_lambda1_is_the_discounted_return():
    g = torch.Generator().manual_seed(0)
    B, T = 3, 7
    r = torch.randn(B, T, generator=g)
    v = torch.randn(B, T, generator=g)
    boot = torch.randn(B, generator=g)
    disc = torch.full((B, T), 0.9)
    disc[1, 3] = 0.0                                  # a terminal step
    vs, adv, rho = vtrace(r, v, boot, disc, torch.zeros(B, T))
    G = boot.clone()
    ref = torch.zeros(B, T)
    for t in reversed(range(T)):
        G = r[:, t] + disc[:, t] * G
        ref[:, t] = G
    assert torch.allclose(vs, ref, atol=1e-5)
    assert torch.equal(rho, torch.ones(B, T))


def test_vtrace_matches_the_tested_superlearner_reference():
    from superlearner.unified.learning.returns import vtrace as ref_vtrace
    g = torch.Generator().manual_seed(1)
    B, T = 4, 9
    r, v = torch.randn(B, T, generator=g), torch.randn(B, T + 1, generator=g)
    disc = torch.rand(B, T, generator=g)
    lr = torch.randn(B, T, generator=g) * 0.7
    for lam in (1.0, 0.9):
        ref_vs, ref_rho = ref_vtrace(r, v, disc, torch.ones(B, T), lr, lam, 1.0, 1.0)
        vs, _, rho = vtrace(r, v[:, :-1], v[:, -1], disc, lr, 1.0, 1.0, lam)
        assert torch.allclose(vs, ref_vs[:, :-1], atol=1e-5)
        assert torch.allclose(rho, ref_rho)


def test_vtrace_clips_importance_weights_and_semi_markov_discount():
    lr = torch.tensor([[math.log(5.0), math.log(0.2)]])
    _, _, rho = vtrace(torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1), torch.ones(1, 2), lr, rho_bar=1.0)
    assert torch.allclose(rho, torch.tensor([[1.0, 0.2]]))
    d = semi_markov_discount(torch.tensor([0.0, 1.0, 2.0]), kappa=0.5)
    assert torch.allclose(d, torch.exp(torch.tensor([0.0, -0.5, -1.0])))
    with pytest.raises(ValueError):
        semi_markov_discount(torch.tensor([-1.0]), 0.5)
    lag = torch.tensor([0, 2])
    assert torch.equal(importance_log_ratio(torch.tensor([-1.0, -1.0]), torch.tensor([-1.5, -1.5]), lag),
                       torch.tensor([0.0, 0.5]))       # lag 0 is exactly on-policy; lag 2 keeps the ratio


def _rec(version, logp=-1.0, action=True):
    return EventRecord(event_id=new_id("e_"), env_id="e", actor_id="a", episode_id="ep", step=0,
                       model_version=version, t_sim=0.0, observations=[],
                       action=Action("x", logp=logp) if action else None, behavior_logp=logp if action else None)


def test_lag_router_routes_by_lag_and_counts():
    R = LagRouter(max_lag=3, mode="vtrace")
    recs = [_rec(10), _rec(9), _rec(7), _rec(6), _rec(2), _rec(11), _rec(10, logp=None)]
    recs[-1].behavior_logp = None
    rep = R.route(recs, learner_version=10)
    assert [r.model_version for r in rep.on_policy] == [10]
    assert [r.model_version for r in rep.off_policy] == [9, 7]
    assert sorted(r.model_version for r in rep.discarded) == [2, 6, 10, 11]
    assert rep.reasons == {"lag_over_max": 2, "future_version": 1, "no_behavior_logp": 1}
    s = rep.summary()
    assert s["lag_max"] == 8 and s["lag_hist"]["0"] == 2
    D = LagRouter(max_lag=3, mode="discard")
    rep2 = D.route(recs[:4], learner_version=10)
    assert len(rep2.on_policy) == 1 and not rep2.off_policy and len(rep2.discarded) == 3
    assert D.totals()["counts"]["discard:stale_in_discard_mode"] == 3


class _RecordingModel:
    """Test double of LearnerModelP: one 'optimiser step' per update; records what it was given."""

    def __init__(self):
        self.w = torch.zeros(4)
        self.batches = []

    def state_dict(self):
        return {"w": self.w.clone()}

    def load_state_dict(self, sd):
        self.w = sd["w"].clone()

    def update(self, batch):
        self.batches.append([(t.actor_id, t.lag, t.route, t.n) for t in batch.trajectories])
        self.w += 1.0
        return {"loss": float(len(batch.trajectories))}


def _episode(writer, actor, version, n=4):
    ep = new_id("ep_")
    ids = []
    for i in range(n):
        e = EventRecord(event_id=new_id("e_"), env_id=f"{actor}.0", actor_id=actor, episode_id=ep, step=i,
                        model_version=version, t_sim=float(i), observations=[], action=Action("x", logp=-0.5),
                        behavior_logp=-0.5, done=(i == n - 1), extra={"dt": 0.1})
        writer.put(e)
        ids.append(e.event_id)
    writer.flush()
    return ids


@pytest.mark.parametrize("mode", ["vtrace", "discard"])
def test_learner_policy_lag_accounting_and_routing(tmp_path, mode):
    run = str(tmp_path / "run")
    model = _RecordingModel()
    L = CentralLearner(run, model, sink=JsonlArchive(os.path.join(run, "archive")), max_lag=2, lag_mode=mode,
                       min_decisions=4, max_batch_decisions=4)
    assert L.version == 0 and L.publisher.latest()["version"] == 0
    w = ExperienceQueue(os.path.join(run, "spool")).writer("actor1", batch_records=100)
    # advance the learner to version 3 with fresh on-policy data
    for v in range(3):
        _episode(w, "actor1", L.version)
        assert L.step() is not None
    assert L.version == 3
    # now deliver experience produced by stale policies: lags 0, 1, 2, 3, 5
    for bv in (3, 2, 1, 0):
        _episode(w, "actor1", bv)
    _episode(w, "actor1", 3)
    for _ in range(10):
        L.step()
    ups = [json.loads(l) for l in open(os.path.join(run, "learner", "updates.jsonl"))]
    trained = [u for u in ups if u["trained"]]
    after = [b[0][1:3] for b in model.batches[3:]]          # (lag, route) of each update after version 3
    st = L.stats()
    if mode == "vtrace":
        # consumed in order at learner versions 3, 4, 5, 5, 5: behaviour 3 (lag 0), 2 (lag 2 -> V-trace),
        # 1 (lag 4 -> discarded), 0 (lag 5 -> discarded), 3 (lag 2 -> V-trace)
        assert after == [(0, "on_policy"), (2, "off_policy"), (2, "off_policy")]
        assert st["router"]["lag_hist"] == {"0": 16, "2": 8, "4": 4, "5": 4}
        assert st["router"]["counts"] == {"on_policy": 16, "off_policy": 8, "discarded": 8,
                                          "discard:lag_over_max": 8, "exact_on_policy": 16}
        assert L.version == 6
    else:
        # strictly on-policy: every stale trajectory is discarded, never trained as if it were fresh
        assert after == [(0, "on_policy")]
        assert st["router"]["lag_hist"] == {"0": 16, "1": 4, "2": 4, "3": 4, "4": 4}
        assert st["router"]["counts"]["discarded"] == 16
        assert L.version == 4
    assert st["counts"]["discarded_decisions"] == st["router"]["counts"]["discarded"]
    for u in trained:                                  # every update records its lag summary and consumed events
        assert "lag_hist" in u["lag"] and len(u["event_ids"]) == u["n_decisions"]
    assert any(not u["trained"] for u in ups)          # batches with only stale data are logged, not trained
    skipped = [json.loads(l) for l in open(os.path.join(run, "learner", "skipped.jsonl"))]
    n_skipped = sum(len(x["event_ids"]) for x in skipped)
    assert n_skipped == st["router"]["counts"]["discarded"]       # every untrained decision is logged with a reason
    assert L.consumed([i for x in skipped for i in x["event_ids"]]) == {i for x in skipped for i in x["event_ids"]}
    assert L.publisher.latest()["version"] == L.version   # publish_every=1
    L.close()


def test_learner_drain_consumes_trace_only_and_duplicate_segments(tmp_path):
    from superlearner.cam_unified.interfaces import DecisionTrace
    run = str(tmp_path / "run")
    L = CentralLearner(run, _RecordingModel(), sink=JsonlArchive(os.path.join(run, "archive")), ingest_segments=1)
    w = ExperienceQueue(os.path.join(run, "spool")).writer("a", batch_records=100)
    ids = _episode(w, "a", 0)
    w.put(trace=DecisionTrace(decision_id=new_id("d_"), event_id=ids[0], env_id="a.0", policy_version=0, t_sim=0.0,
                              objective=""))
    w.flush()                                            # a segment holding only a trace
    _episode(w, "a", 0)
    L.run(max_updates=0)                                 # stops immediately, then drains the spool
    assert not L.queue.pending() and L.sink.count() == 8
    assert L.consumer.stats()["totals"]["traces"] == 1
    L.close()
