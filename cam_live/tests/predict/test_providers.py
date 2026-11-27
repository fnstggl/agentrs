"""Workstream B's loss providers on the real (debug-scale) CamLiveModel with honestly collected LabWorld experience."""
import copy
import dataclasses

import pytest
import torch

from cam_live.contracts import GoalStatus, OutcomeObservation, OutcomeStatus
from cam_live.learning_api import (TIMESCALE_FREQUENT, TIMESCALE_SLOW, LossProvider, encode_decisions,
                                   matured_outcomes)
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_REWARD
from cam_live.predict.evaluator import UnverifiedOutcomeError
from cam_live.predict.heads import HEAD_NAMES, attach_heads
from cam_live.predict.imagination import ImaginedDataError
from cam_live.predict.providers import (EvaluatorAdapterLoss, EvaluatorLoss, FastPriorLoss, WorldModelAdapterLoss,
                                        WorldModelLoss, default_providers, encode_with_adapters, record_rows)


def _loss(rig, provider, records, **kw):
    batch = encode_decisions(rig.model, rig.store, records, **{**provider.batch_kwargs, **kw})
    return provider.loss(rig.model, batch, rig.store)


def test_heads_are_registered_on_the_model_and_reused(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    counts = rig.model.count_parameters()
    assert set(HEAD_NAMES) <= set(counts["heads"])
    assert counts["head_groups"]["predict_world"] == GROUP_HEADS and counts["reward"] == 0
    again = attach_heads(rig.model)
    assert again.world is rig.heads.world and again.policy is rig.heads.policy and again.scaler is rig.heads.scaler
    state = rig.model.trainable_state()
    assert any(k.startswith("heads.predict_world.") for k in state)
    assert any("normaliser" in k for k in state)                      # running kind scales are checkpointed


def test_providers_follow_the_protocol(lab_rig):
    rig = lab_rig(n_lifetimes=1)
    provs = default_providers(rig.heads, rig.check_signature)
    assert [p.name for p in provs] == ["world_model", "fast_prior", "evaluator", "world_model_adapters",
                                       "evaluator_adapters"]
    assert all(isinstance(p, LossProvider) for p in provs)
    assert [p.timescale for p in provs] == [TIMESCALE_FREQUENT] * 3 + [TIMESCALE_SLOW] * 2
    assert len(default_providers(rig.heads, None)) == 3               # no verifier check -> no evaluator providers


def test_world_loss_trains_on_matured_outcomes_only(lab_rig):
    rig = lab_rig()
    wm = WorldModelLoss(rig.heads)
    recs = [r for r in rig.records if matured_outcomes(r)][:8]
    n_matured = sum(len(matured_outcomes(r)) for r in recs)
    loss, m = _loss(rig, wm, recs)
    assert m["world_n_targets"] == n_matured == m["world_n_targets"] and torch.isfinite(loss)
    assert m["world_n_count"] == n_matured                            # two horizons per intervention
    assert "fast_prior_nll_count" in m and "transition_nll" in m

    # the same decisions while their outcomes are still pending or unobservable: no target, whatever value is offered
    pending = []
    for r in recs:
        p = copy.deepcopy(r)
        p.outcomes = [OutcomeObservation(o.query_id, OutcomeStatus.PENDING if i % 2 else OutcomeStatus.UNOBSERVABLE,
                                         value=999.0, observed_at=o.observed_at) for i, o in enumerate(r.outcomes)]
        p.post_item_refs = []
        pending.append(p)
    assert all(not matured_outcomes(p) for p in pending) and not any(wm.eligible(p) for p in pending)
    assert record_rows(pending) == [[] for _ in pending]
    batch = encode_decisions(rig.model, rig.store, pending)
    loss_p, m_p = wm.loss(rig.model, batch, rig.store)
    assert loss_p is None and m_p["world_n_targets"] == 0

    # mixed batch: the pending rows add nothing
    mixed, m_mixed = _loss(rig, WorldModelLoss(rig.heads, transition_weight=0.0, fast_weight=0.0), recs + pending)
    only, m_only = _loss(rig, WorldModelLoss(rig.heads, transition_weight=0.0, fast_weight=0.0), recs)
    assert m_mixed["world_n_targets"] == m_only["world_n_targets"]
    assert abs(m_mixed["world_nll_count"] - m_only["world_nll_count"]) < 1e-5


def test_world_loss_gradient_reaches_core_heads_and_fast_prior_only(lab_rig):
    rig = lab_rig()
    wm = WorldModelLoss(rig.heads)
    assert wm.eligible(rig.records[0])
    loss, _ = _loss(rig, wm, [r for r in rig.records if wm.eligible(r)][:12])
    rig.model.zero_grad()
    loss.backward()
    g = rig.model.grad_norms()
    assert g[GROUP_EXPERIENTIAL] > 0 and g[GROUP_HEADS] > 0
    assert g[GROUP_ADAPTERS] == 0 and g[GROUP_BACKBONE_FROZEN] == 0 and g[GROUP_REWARD] == 0
    prior = rig.model.exp.fast_prior
    assert prior.mu0.grad is not None and float(prior.mu0.grad.abs().sum()) > 0         # fast layer is in the loss
    assert float(sum(p.grad.abs().sum() for p in rig.heads.transition.parameters() if p.grad is not None)) > 0
    assert all(p.grad is None for p in rig.heads.policy.parameters())                  # not this provider's head

    diag = dataclasses.replace(rig.records[0], provenance="unlabelled")
    assert not wm.eligible(diag)                                      # provenance is never mixed silently


def test_fast_prior_loss_is_prequential_within_an_objective(lab_rig):
    rig = lab_rig()
    fp = FastPriorLoss(rig.heads)
    recs = [r for r in rig.lifetimes[0].records if fp.eligible(r)] + [r for r in rig.lifetimes[1].records
                                                                    if fp.eligible(r)]
    loss, m = _loss(rig, fp, recs)
    n = sum(len(matured_outcomes(r)) for r in recs)
    assert m["fast_prior_n"] == n and torch.isfinite(loss)
    assert {"fast_online_nll_count", "fast_online_nll_first_count", "fast_online_nll_later_count"} <= set(m)
    rig.model.zero_grad()
    loss.backward()
    exp = rig.model.exp
    for p in (exp.fast_prior.mu0, exp.fast_prior.chol_diag_raw, exp.feature_map.out.weight):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0   # prior and features both learn through updates
    assert all(p.grad is None for p in rig.heads.world.members.parameters())            # head enters as a detached offset
    none, m0 = _loss(rig, fp, [r for r in rig.records if not matured_outcomes(r)][:4])
    assert none is None and m0["fast_prior_n"] == 0


def test_evaluator_loss_needs_a_signed_resolved_outcome(lab_rig):
    rig = lab_rig()
    ev = EvaluatorLoss(rig.heads, rig.check_signature)
    lt = rig.lifetimes[0]
    deadline = lt.world.t0 + lt.world.cfg.horizon_s
    assert not ev.eligible(lt.records[0])
    loss, m = _loss(rig, ev, lt.records)
    assert loss is None and m["eval_success_n"] == 0                  # nothing verified yet: no target at all

    forged = dataclasses.replace(lt.verified, status=GoalStatus.ACHIEVED, value=1e6)    # signature no longer matches
    with pytest.raises(UnverifiedOutcomeError):
        ev.add_outcome(forged, deadline=deadline)
    with pytest.raises(UnverifiedOutcomeError):
        ev.add_outcome(dataclasses.replace(lt.verified, signature=""), deadline=deadline)
    assert not ev.eligible(lt.records[0])

    from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
    w = LabWorld(rig.L.lab_config(seed=77))
    t = lab_task(w)
    pend = LabVerifier(w).verify(t)
    rig.lifetimes.append(type(lt)(task=t, records=[], verified=pend, world=w))
    assert pend.status == GoalStatus.PENDING
    assert ev.add_outcome(pend, deadline=w.t0 + w.cfg.horizon_s) is False                # remembered as pending

    assert ev.add_outcome(lt.verified, deadline=deadline, trajectory=lt.records) is True
    assert all(ev.eligible(r) for r in lt.records)
    loss, m = _loss(rig, ev, lt.records)
    assert m["eval_success_n"] == len(lt.records)                     # every earlier prefix is a target
    assert m["eval_resources_n"] > 0 and m["eval_magnitude_n"] == len(lt.records)
    assert m["eval_time_n"] == 0                                      # achievement time unknown -> no time target
    rig.model.zero_grad()
    loss.backward()
    g = rig.model.grad_norms()
    assert g[GROUP_HEADS] > 0 and g[GROUP_EXPERIENTIAL] > 0 and g[GROUP_ADAPTERS] == 0 and g[GROUP_REWARD] == 0


def test_overconfident_evaluator_receives_corrective_gradient(lab_rig):
    rig = lab_rig(n_lifetimes=6)
    ev = EvaluatorLoss(rig.heads, rig.check_signature)
    failed = [lt for lt in rig.lifetimes if lt.verified.status == GoalStatus.FAILED]
    done = [lt for lt in rig.lifetimes if lt.verified.status == GoalStatus.ACHIEVED]
    assert failed and done
    lt = failed[0]
    ev.add_outcome(lt.verified, deadline=lt.world.t0 + lt.world.cfg.horizon_s)
    with torch.no_grad():
        rig.heads.evaluator.out.bias[0] = 1.386                       # "80 % success" everywhere

    def p_mean():
        with torch.no_grad():
            b = encode_decisions(rig.model, rig.store, lt.records, grad=False)
            hz = torch.full((len(lt.records),), 86400.0, dtype=torch.float64)
            return float(rig.heads.evaluator(b.b, b.e_g, hz).p_success.mean())

    before = p_mean()
    assert before > 0.75
    opt = torch.optim.SGD(rig.heads.evaluator.parameters(), lr=0.5)
    for _ in range(5):
        loss, _ = _loss(rig, ev, lt.records)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert p_mean() < before - 0.05                                   # reality falsified the forecast


def test_adapter_providers_send_prediction_gradient_into_the_semantic_adapters(lab_rig):
    rig = lab_rig(adapter_grad_last=2, adapter_max_records=3)
    lt = rig.lifetimes[0]
    recs = [r for r in lt.records if matured_outcomes(r)][:6]
    wma = WorldModelAdapterLoss(rig.heads)
    assert wma.timescale == TIMESCALE_SLOW and wma.batch_kwargs["with_grad_semantic"] is True

    plain = encode_decisions(rig.model, rig.store, recs, with_next=True)             # no graph into the adapters
    loss, m = wma.loss(rig.model, plain, rig.store)                   # the provider re-encodes with gradient itself
    assert m["world_n_targets"] == sum(len(matured_outcomes(r)) for r in recs[:3])   # bounded: adapter_max_records
    rig.model.zero_grad()
    loss.backward()
    g = rig.model.grad_norms()
    assert g[GROUP_ADAPTERS] > 0 and g[GROUP_BACKBONE_FROZEN] == 0 and g[GROUP_REWARD] == 0
    assert all(p.grad is None for p in rig.model.param_groups()[GROUP_BACKBONE_FROZEN])

    batch = encode_with_adapters(rig.model, rig.store, recs, grad_last=1)
    assert batch.with_grad_semantic and batch.z.shape[1] == 1 and batch.b_next is not None
    eva = EvaluatorAdapterLoss(rig.heads, rig.check_signature)
    eva.add_outcome(lt.verified, deadline=lt.world.t0 + lt.world.cfg.horizon_s)
    loss, _ = eva.loss(rig.model, batch, rig.store)
    rig.model.zero_grad()
    loss.backward()
    assert rig.model.grad_norms()[GROUP_ADAPTERS] > 0

    loss, _ = _loss(rig, WorldModelLoss(rig.heads), recs)             # the frequent provider leaves adapters alone
    rig.model.zero_grad()
    loss.backward()
    assert rig.model.grad_norms()[GROUP_ADAPTERS] == 0


def test_imagined_values_cannot_become_training_targets(lab_rig):
    rig = lab_rig(n_lifetimes=1)
    ev = EvaluatorLoss(rig.heads, rig.check_signature)
    lt = rig.lifetimes[0]

    class Imagined:
        imagined = True
        objective_id, status, signature = lt.verified.objective_id, GoalStatus.ACHIEVED, "x"

    with pytest.raises(ImaginedDataError):
        ev.add_outcome(Imagined(), deadline=0.0)
    with pytest.raises(UnverifiedOutcomeError):
        ev.add_outcome({"objective_id": "o", "status": GoalStatus.ACHIEVED}, deadline=0.0)
