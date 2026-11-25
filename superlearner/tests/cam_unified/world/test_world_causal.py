"""Causal provenance (C02) recorded per event and attached to EVENT tokens; intervention-sensitive contrastive
representation learning (W03)."""
import math

import torch

from superlearner.cam_unified.interfaces import (Action, ActionSource, EventRecord, EvidenceKind, ExecutionResult,
                                                 Observation, ObsKind, Provenance, TokenKind, TokenSet)
from superlearner.cam_unified.world.causal import (PROV_FEATURES, CausalLedger, CausalProvenance,
                                                   InterventionContrastive, provenance_features)


def _ev(eid, t, action=None, result=None, obs=(), evidence=EvidenceKind.CONTROLLED, logp=None, **kw):
    return EventRecord(event_id=eid, env_id="env0", actor_id="a0", episode_id="ep", step=0, model_version=0, t_sim=t,
                       observations=list(obs), action=action, result=result, behavior_logp=logp, evidence=evidence,
                       **kw)


def _feat(ev):
    return dict(zip(PROV_FEATURES, provenance_features(ev)))


def test_ledger_records_assignment_simultaneity_delays_and_missingness():
    led = CausalLedger(effect_window_days=0.0)
    a = Action("set_price", {"price": 3}, source=ActionSource.CAM_DECODER, logp=math.log(0.25))
    ea = led.record(_ev("eA", 0.0, a, ExecutionResult(True, started_t=0.0, finished_t=0.5, duration=0.5),
                        logp=math.log(0.25)))
    assert abs(ea.randomization_p - 0.25) < 1e-9 and ea.simultaneous == []
    b = Action("hire", {"count": 1}, source=ActionSource.CAM_DECODER, logp=math.log(0.5))
    eb = led.record(_ev("eB", 0.2, b, ExecutionResult(True, started_t=0.2, finished_t=0.3), logp=math.log(0.5)))
    assert eb.simultaneous == [a.action_id]                      # A still running when B was assigned
    assert eb.extra["causal"]["confounding_possible"]
    c = Action("deploy", source=ActionSource.CAM_PLANNER, proposal={"n": 3, "chosen": 1, "deterministic": True})
    ec = led.record(_ev("eC", 2.0, c, ExecutionResult(True, started_t=2.0, finished_t=2.0)))
    assert ec.randomization_p == 1.0 and ec.simultaneous == []    # deterministic planner choice, nothing active
    assert ec.extra["causal"]["confounding_possible"]             # not randomised -> confounding possible
    # a delayed asynchronous outcome caused by A, with one missing measurement
    late = [Observation(ObsKind.NUMERIC, value=None, unit="cents", t=3.0, caused_by="eA", meta={"missing": True}),
            Observation(ObsKind.NUMERIC, value=120.0, unit="cents", t=3.0, caused_by=a.action_id,
                        meta={"reliability": 0.8})]
    ed = led.record(_ev("eD", 3.0, None, None, late, evidence=EvidenceKind.OBSERVATIONAL))
    assert ed.attributed_to == "eA"
    assert ed.extra["causal"]["delays_days"] == [3.0, 3.0]
    assert ed.extra["causal"]["missing_frac"] == 0.5
    f = _feat(ed)
    assert f["attributed"] == 1.0 and f["delay_known"] == 1.0 and abs(f["log_delay_days"] - math.log1p(3.0)) < 1e-9
    assert f["missing_frac"] == 0.5 and f["ev_observational"] == 1.0 and f["confounding_possible"] == 1.0
    assert f["rand_known"] == 0.0 and f["reliability_known"] == 1.0 and abs(f["reliability"] - 0.8) < 1e-9
    # a randomised intervention alone is the clean causal case
    e = Action("set_price", {"price": 5}, source=ActionSource.CAM_DECODER, logp=math.log(0.5))
    ee = led.record(_ev("eE", 10.0, e, ExecutionResult(True, started_t=10.0, finished_t=10.1), logp=math.log(0.5)))
    fe = _feat(ee)
    assert fe["rand_p"] == 0.5 and fe["rand_known"] == 1.0 and fe["log_n_simultaneous"] == 0.0
    assert fe["confounding_possible"] == 0.0 and fe["ev_controlled"] == 1.0 and fe["result_missing"] == 0.0
    # pending result is recorded as missing; counterfactual provenance is flagged
    ef = led.record(_ev("eF", 11.0, Action("noop"), None, provenance=Provenance.COUNTERFACTUAL))
    ff = _feat(ef)
    assert ff["result_missing"] == 1.0 and ff["counterfactual"] == 1.0
    # the recorded fields survive the record JSON round trip
    back = EventRecord.from_dict(EventRecord.from_dict(eb.to_dict()).to_dict())
    assert back.simultaneous == eb.simultaneous and back.extra["causal"] == eb.extra["causal"]


def test_provenance_features_are_appended_to_event_tokens_only():
    D = 32
    torch.manual_seed(0)
    mod = CausalProvenance(D)
    led = CausalLedger()
    r1 = led.record(_ev("e1", 0.0, Action("a", source=ActionSource.CAM_DECODER, logp=math.log(0.5)),
                        ExecutionResult(True), logp=math.log(0.5)))
    r2 = led.record(_ev("e2", 1.0, None, None, [Observation(ObsKind.NUMERIC, value=None, t=1.0, caused_by="e1")],
                        evidence=EvidenceKind.OBSERVATIONAL))
    x = torch.randn(1, 4, D)
    kind = torch.tensor([[TokenKind.EVENT, TokenKind.OBS_NUM, TokenKind.EVENT, TokenKind.EVENT]])
    ts = TokenSet(x, torch.tensor([[True, True, True, False]]), kind, torch.zeros(1, 4), torch.full((1, 4), -1),
                  [["e1", "e2", "e2", "e1"]])
    out = mod.attach(ts, {"e1": r1, "e2": r2})
    d = (out.x - x).abs().sum(-1)[0]
    assert d[0] > 0 and d[2] > 0                     # real EVENT tokens get their provenance features
    assert d[1] == 0 and d[3] == 0                   # non-event token and padded event token unchanged
    assert (out.x[0, 0] - x[0, 0] - (out.x[0, 2] - x[0, 2])).abs().sum() > 1e-4     # different provenance
    feats = mod.features_for(ts, {"e1": r1, "e2": r2})
    assert torch.allclose(feats[0, 0], torch.tensor(provenance_features(r1)))
    assert torch.equal(feats[0, 1], torch.zeros(len(PROV_FEATURES)))
    out.x.sum().backward()
    assert mod.net[0].weight.grad is not None


def _mechanism_data(N, g):
    """Hidden mechanism m flips the sign of the response to the intervention; the observation is dominated by a
    large irrelevant surface and carries only a small cue of m."""
    m = torch.randint(0, 2, (N,), generator=g)
    surface = torch.randn(N, 16, generator=g) * 3.0
    cue = (2 * m - 1).float().unsqueeze(1) * 0.3 + 0.05 * torch.randn(N, 1, generator=g)
    x = torch.cat([surface, cue], 1)
    a = torch.randint(1, 3, (N,), generator=g)                  # interventions of strength 1 or 2
    y = torch.where(m == 0, a.float(), -a.float()) + 0.05 * torch.randn(N, generator=g)
    return x, m, a, y.unsqueeze(1)


def _knn_acc(e, labels, k=5):
    e = torch.nn.functional.normalize(e, dim=-1)
    sim = e @ e.T
    sim.fill_diagonal_(-2)
    nn_idx = sim.topk(k, dim=1).indices
    pred = (labels[nn_idx].float().mean(1) > 0.5).long()
    return float((pred == labels).float().mean())


def test_intervention_contrastive_separates_controllable_dynamics_not_surface():
    torch.manual_seed(0)
    D = 32
    enc = torch.nn.Sequential(torch.nn.Linear(17, 64), torch.nn.SiLU(), torch.nn.Linear(64, D))
    obj = InterventionContrastive(D, d_embed=16, n_heads=4)
    opt = torch.optim.Adam(list(enc.parameters()) + list(obj.parameters()), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    for _ in range(300):
        x, m, a, y = _mechanism_data(128, g)
        xs = torch.nn.functional.normalize(x, dim=-1)
        out = obj.loss(enc(x), a, y, obs_similarity=xs @ xs.T)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    assert out["n_anchors"] > 100
    xt, mt, at, yt = _mechanism_data(400, torch.Generator().manual_seed(99))
    with torch.no_grad():
        emb = obj.embed(enc(xt))
    acc_learned = _knn_acc(emb, mt)
    acc_raw = _knn_acc(xt, mt)
    assert acc_learned > 0.9 and acc_raw < 0.7, (acc_learned, acc_raw)
    # states with the same controllable response embed close, different responses apart
    sim = emb @ emb.T
    same = mt.unsqueeze(0) == mt.unsqueeze(1)
    assert float(sim[same].mean()) > float(sim[~same].mean()) + 0.5


def test_contrastive_targets_follow_response_similarity():
    obj = InterventionContrastive(8, d_embed=4, n_heads=2, hard_weight=0.0)
    resp = torch.tensor([[0.0], [0.1], [5.0], [5.1], [0.0]])
    mask = torch.tensor([[True], [True], [True], [True], [False]])
    d, both = obj.response_distance(resp, mask)
    assert abs(float(d[0, 1]) - 0.01) < 1e-6 and not bool(both[0, 4])
    iv = ["a", "a", "a", "a", "b"]
    Z = torch.randn(5, 8, requires_grad=True)
    out = obj.loss(Z, iv, resp, mask)
    out["loss"].backward()
    assert out["n_anchors"] == 4 and Z.grad[4].abs().sum() == 0      # the unobserved / lone state gives no signal
