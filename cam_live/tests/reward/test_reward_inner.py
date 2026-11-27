"""PolicyImprovementLoss on the real (debug) model: one function for the learner and for the meta unroll."""
import dataclasses
import math

import pytest
import torch

from cam_live.contracts import (CandidateEvaluation, Distribution, ItemKind, OutcomeObservation, OutcomeStatus,
                                PredictionRecord, Trust)
from cam_live.learning_api import TIMESCALE_FREQUENT, LossProvider, encode_decisions
from cam_live.reward import diagnose as dg
from cam_live.reward.assemble import STORED_KEY, SIGNAL_SOURCES, record_signals
from cam_live.reward.inner import POLICY_PREFIX, TRUNK_PREFIX, GroupClip
from cam_live.reward.integrity import depends_on, params_hash
from cam_live.reward.meta import MetaGradient, QueryData, flat
from cam_live.reward.signals import SCALAR_NAMES, SIGNAL_NAMES

IDX = {n: i for i, n in enumerate(SIGNAL_NAMES)}


@pytest.fixture(scope="module")
def rig():
    """A debug model, one LabWorld lifetime acted by its own policy and rewarded step by step (never mutated)."""
    ec = dg.ExperimentConfig()
    model, store, clock, ml = dg.make_stack(0, ec, mix=0.5)
    lt, world, _ = dg.lab_lifetime(ml, store, clock, ec, seed=4242, update_every=0)
    return dict(ec=ec, model=model, store=store, clock=clock, ml=ml, lt=lt, world=world)


def test_provider_contract_and_same_function_as_the_unroll(rig):
    ml, model, store, recs = rig["ml"], rig["model"], rig["store"], rig["lt"].records[:8]
    prov = ml.inner
    assert isinstance(prov, LossProvider) and prov.timescale == TIMESCALE_FREQUENT and prov.weight > 0
    assert all(prov.eligible(r) for r in recs)
    batch = encode_decisions(model, store, recs, with_next=True, grad=True)
    loss, metrics = prov.loss(model, batch, store)
    cached = prov.build_batch(store, recs)                       # encode_decisions(grad=False) inside
    assert not cached.b.requires_grad
    theta, psi = prov.subset.live(), ml.reward.psi()
    same = prov.functional(theta, psi, cached, detach_reward=True)
    unroll = prov.inner_loss(theta, psi, cached)                 # what MetaGradient differentiates
    assert abs(float(loss) - float(same)) < 1e-5 and abs(float(loss) - float(unroll)) < 1e-5
    grads = torch.autograd.grad(loss, list(theta.values()), allow_unused=True, retain_graph=True)
    got = {n for n, g in zip(theta, grads) if g is not None and bool((g != 0).any())}
    assert any(n.startswith(TRUNK_PREFIX) for n in got) and any(n.startswith(POLICY_PREFIX) for n in got)
    assert not depends_on(loss, ml.reward)                       # R_psi is detached in the learner's loss ...
    assert depends_on(unroll, ml.reward)                         # ... and differentiated in the unroll
    assert metrics["mix"] == 0.5 and metrics["n"] == 8.0


def test_inner_loss_and_real_policy_update_leave_psi_bit_identical():
    r = dg.psi_isolation_check(seed=1)
    assert r["loss_has_grad_path_to_psi"] is False
    assert r["psi_tensors_with_gradient_after_backward"] == 0
    assert r["psi_bit_identical_after_optimizer_step"] and r["psi_bit_identical_after_policy_update"]
    assert r["theta_changed_by_optimizer_step"]
    assert r["policy_prob_mean_abs_change_policy_update"] > 0


def test_reward_step_labels_every_decision_and_replay_moves_no_state(rig):
    ml, store, recs = rig["ml"], rig["store"], rig["lt"].records
    for r in recs:
        assert r.intrinsic_reward is not None and r.bootstrap_reward is not None
        assert r.reward_version.startswith("mixed:mix=0.500:rpsi1-")       # never labelled as meta-learned
        assert r.exp_id in ml.inner.life_states
    assert int(ml.reward.lifetime.steps) == len(recs)
    entry = ml.mixer.log[-1]
    assert entry["label"] == "mixed" and "from_bootstrap" in entry and "from_learned" in entry
    state, gate = ml.reward.lifetime.state.clone(), ml.inner.bootstrap.gate.state_dict()
    v = ml.reward.version
    for _ in range(3):
        ml.inner.build_batch(store, recs)
    assert torch.equal(state, ml.reward.lifetime.state) and gate == ml.inner.bootstrap.gate.state_dict()
    assert ml.reward.version == v


def test_signal_sources_are_documented_and_causal(rig):
    ml, store, recs = rig["ml"], rig["store"], rig["lt"].records
    assert set(SCALAR_NAMES) <= set(SIGNAL_SOURCES)
    asm = ml.inner.assembler.assemble(recs, store)
    m = asm.mask
    for name in ("cost_money", "cost_seconds", "cost_compute_s", "verified_progress_delta", "status.ok"):
        assert bool((m[:, IDX[name]] == 1).all()), name
    assert bool((m[1:, IDX["elapsed_s"]] == 1).all()) and float(m[0, IDX["elapsed_s"]]) == 0.0
    for name in ("success_before", "success_after", "delta_v", "voi", "info_gain", "epistemic_before"):
        assert float(m[:, IDX[name]].sum()) == 0.0, name          # no source in these records: masked, not invented
    rec = recs[5]
    sig, extras, t_r = record_signals(rec, store)
    ctx = set(rec.context_item_refs)
    new = [i for i in store.items(rec.post_item_refs) if i.ref not in ctx and i.kind == ItemKind.OUTCOME
           and i.trust == Trust.VERIFIER]
    assert sig["verified_progress_delta"] == pytest.approx(sum(q.value for i in new for q in i.quantities))
    assert sig["cost_seconds"] == rec.result.cost.seconds and sig["elapsed_s"] > 0
    assert t_r >= rec.decision.ts
    # values the agent recorded at decision time win; learning progress is a bootstrap-only extra
    ev = CandidateEvaluation(action=rec.decision.chosen, p_success=0.3, voi=0.2, uncertainty=0.8,
                             extras={"info_gain": 0.11, "epistemic_after": 0.6})
    cands = list(rec.decision.candidates)
    cands[rec.decision.chosen_index] = ev
    dec = dataclasses.replace(rec.decision, candidates=cands,
                              diagnostics={STORED_KEY: {"voi": 0.7, "learning_progress": 0.25, "context_novelty": 0.4}})
    sig2, extras2, _ = record_signals(dataclasses.replace(rec, decision=dec), store)
    assert sig2["success_before"] == 0.3 and sig2["voi"] == 0.7 and sig2["epistemic_before"] == 0.8
    assert sig2["info_gain"] == 0.11 and sig2["epistemic_after"] == 0.6 and sig2["context_novelty"] == 0.4
    assert extras2 == {"learning_progress": 0.25}


def test_prediction_error_uses_only_prospective_predictions_seen_by_reward_time(rig):
    store, recs = rig["store"], rig["lt"].records
    rec = next(r for r in recs if r.result.expected_outcomes and r.outcomes)
    q = rec.result.expected_outcomes[0]
    obs = next(o for o in rec.outcomes if o.query_id == q.query_id)
    _, _, t_r = record_signals(rec, store)

    def with_pred(made_at, observed_at):
        p = PredictionRecord(prediction_id="p", query=q, dist=Distribution("normal", mean=1.0, std=2.0), made_at=made_at)
        cands = list(rec.decision.candidates)
        cands[rec.decision.chosen_index] = CandidateEvaluation(action=rec.decision.chosen, predictions=[p])
        o = dataclasses.replace(obs, observed_at=observed_at)
        return dataclasses.replace(rec, decision=dataclasses.replace(rec.decision, candidates=cands), outcomes=[o])
    seen = record_signals(with_pred(rec.decision.ts - 1.0, t_r), store)[0]
    assert seen["prediction_error_std"] == pytest.approx(abs(obs.value - 1.0) / 2.0)
    later = record_signals(with_pred(rec.decision.ts - 1.0, t_r + 10.0), store)[0]
    assert "prediction_error_std" not in later                    # matured after the reward was computed
    hindsight = record_signals(with_pred(t_r + 1.0, t_r), store)[0]
    assert "prediction_error_std" not in hindsight                # prediction registered after the outcome


def test_evaluator_head_fills_success_signals(rig):
    from cam_live.predict.evaluator import EvaluatorConfig, GoalEvaluator
    from cam_live.reward.assemble import SignalAssembler
    ml, model, store, recs = rig["ml"], rig["model"], rig["store"], rig["lt"].records[:6]
    torch.manual_seed(0)
    asm = SignalAssembler(evaluator=GoalEvaluator(EvaluatorConfig(d_model=model.d_model, hidden=32)))
    batch = encode_decisions(model, store, recs, with_next=True, grad=False)
    out = asm.assemble_batch(batch, store)
    s = out.signals
    assert bool((out.mask[:, [IDX["success_before"], IDX["success_after"], IDX["delta_v"]]] == 1).all())
    assert torch.allclose(s.delta_v, s.success_after - s.success_before)
    assert bool(((s.success_before > 0) & (s.success_before < 1)).all())


def test_returns_are_duration_aware_and_stop_at_lifetime_boundaries():
    ec = dg.ExperimentConfig()
    model, store, clock, ml = dg.make_stack(2, ec, mix=1.0)
    recs = []
    for i in range(2):
        ml.new_lifetime()
        recs += dg.lab_lifetime(ml, store, clock, ec, seed=60 + i, update_every=0, n_decisions=6)[0].records
    ib = ml.inner.build_batch(store, list(reversed(recs)))        # order of the input does not matter
    assert ib.exp_ids == [r.exp_id for r in recs]
    assert ib.segments == [(0, 6), (6, 12)]
    half = ml.inner.cfg.half_life_s
    for i in range(11):
        want = 0.0 if i == 5 else 0.5 ** ((recs[i + 1].decision.ts - recs[i].decision.ts) / half)
        assert float(ib.gamma[i]) == pytest.approx(want, rel=1e-5)
    assert float(ib.gamma[11]) == 0.0
    # a long wait discounts what follows more than a short action does
    assert len({round(float(g), 6) for g in ib.gamma[:5]}) > 1
    theta, psi = ml.subset.live(), ml.reward.psi()
    _, aux = ml.inner.functional(theta, psi, ib, detach_reward=True, return_aux=True)
    r, G = aux["reward"], aux["returns"]
    assert float(G[5]) == pytest.approx(float(r[5]), abs=1e-6)    # last step of lifetime 1: no leak from lifetime 2
    assert float(G[4]) == pytest.approx(float(r[4] + ib.gamma[4] * r[5]), abs=1e-5)


def test_meta_gradient_matches_finite_differences_on_the_real_adaptable_subset():
    r = dg.meta_gradient_check(Ks=(1, 2, 4))
    assert r["goal_free_support_rows"] == 2 and r["n_theta"] > 50_000
    for K, v in r["K"].items():
        assert v["rel_error_sampled_coords"] < 1e-4, (K, v)
        assert v["directional_rel_error"] < 1e-5, (K, v)
        assert v["every_tensor_nonzero"], (K, v["grad_norm_per_tensor"])
        assert v["grad_norm_reward_detached"] == 0.0              # detached reward: the pathway is gone, exactly
        assert v["surrogate_has_graph_when_theta_K_detached"] is False
        assert v["policy_prob_mean_abs_change_support"] > 1e-4    # the inner update changes behaviour
    assert r["K"]["4"]["policy_prob_mean_abs_change_support"] > r["K"]["1"]["policy_prob_mean_abs_change_support"]


def _fd_setup(seed=3):
    ec = dg.ExperimentConfig()
    model, store, clock, ml = dg.make_stack(seed, ec, mix=0.0)
    lt, _, _ = dg.lab_lifetime(ml, store, clock, ec, seed=4300 + seed, update_every=0)
    D = torch.float64
    ib = ml.inner.build_batch(store, lt.records[:8]).to(D)
    ibq = ml.inner.build_batch(store, lt.records[8:], policy_only=True).to(D)
    theta = ml.subset.snapshot(D)
    psi = {k: v.detach().clone().to(D).requires_grad_(True) for k, v in ml.reward.psi().items()}
    q = QueryData(batch=ibq, utility=torch.linspace(-1, 1, ibq.n, dtype=D), baseline=0.0)
    return ml, ib, theta, psi, q


def _directional(mg, theta, psi, ib, q, eps=1e-4):
    gen = torch.Generator().manual_seed(0)
    names = list(psi)
    d = {n: torch.randn(psi[n].shape, dtype=torch.float64, generator=gen) for n in names}
    norm = math.sqrt(sum(float((v ** 2).sum()) for v in d.values()))
    d = {n: v / norm for n, v in d.items()}

    def J(sign):
        p = {n: psi[n].detach() + sign * eps * d[n] for n in names}
        return float(mg.outer_surrogate(mg.inner_update(theta, p, [ib], create_graph=False).detached(), q).surrogate)
    return (J(+1) - J(-1)) / (2 * eps), d


def test_value_baseline_must_not_be_plainly_detached_for_K_above_one():
    """Regression: with ``v.detach()`` as the advantage baseline the unroll loses the path
    theta_k(psi) -> V -> advantage and the meta-gradient is biased for K >= 2 (found on the real model)."""
    ml, ib, theta, psi, q = _fd_setup()
    clip = GroupClip({POLICY_PREFIX: 1.0, TRUNK_PREFIX: 1.0})
    aware = MetaGradient(ml.inner.inner_loss, ml.inner.logp, inner_lr=0.1, K=2, grad_transform=clip)
    naive = MetaGradient(lambda th, p, b: ml.inner.functional(th, p, b), ml.inner.logp, inner_lr=0.1, K=2,
                         grad_transform=clip)
    assert aware.stopgrad_aware and not naive.stopgrad_aware
    fd, d = _directional(aware, theta, psi, ib, q)
    dots = {}
    for name, mg in (("aware", aware), ("naive", naive)):
        res = mg.meta_gradient(mg.inner_update(theta, psi, [ib], create_graph=True), q, "on_policy")
        dots[name] = sum(float((res.grads[n] * d[n]).sum()) for n in d)
    assert abs(dots["aware"] - fd) / abs(fd) < 1e-5
    assert abs(dots["naive"] - fd) / abs(fd) > 1e-4               # the bias this fix removed
    # with K = 1 there is no earlier theta to depend on: both agree
    a1 = MetaGradient(ml.inner.inner_loss, ml.inner.logp, inner_lr=0.1, K=1, grad_transform=clip)
    n1 = MetaGradient(lambda th, p, b: ml.inner.functional(th, p, b), ml.inner.logp, inner_lr=0.1, K=1,
                      grad_transform=clip)
    ga = a1.meta_gradient(a1.inner_update(theta, psi, [ib], create_graph=True), q, "on_policy").grads
    gn = n1.meta_gradient(n1.inner_update(theta, psi, [ib], create_graph=True), q, "on_policy").grads
    assert float((flat(ga) - flat(gn)).norm()) < 1e-12 * max(1.0, float(flat(ga).norm()))


def test_checkpointed_unroll_equals_full_unroll_on_the_real_model():
    ml, ib, theta, psi, q = _fd_setup(seed=4)
    clip = GroupClip({POLICY_PREFIX: 1.0, TRUNK_PREFIX: 1.0})
    out = {}
    for mode in ("unroll", "checkpoint"):
        mg = MetaGradient(ml.inner.inner_loss, ml.inner.logp, inner_lr=0.1, K=4, mode=mode, grad_transform=clip)
        out[mode] = flat(mg.meta_gradient(mg.inner_update(theta, psi, [ib], create_graph=True), q, "on_policy").grads)
    assert float(out["unroll"].norm()) > 0
    assert float((out["unroll"] - out["checkpoint"]).norm() / out["unroll"].norm()) < 1e-9


def test_group_clip_is_per_group_and_differentiable():
    x = torch.tensor([3.0, 4.0], requires_grad=True)
    g = [x * 2.0, torch.tensor([0.3]), None]
    out = GroupClip({"policy.": 1.0, "trunk.": 1.0})(["policy.a", "trunk.b", "trunk.c"], g)
    assert float(out[0].norm()) == pytest.approx(1.0) and float(out[1]) == pytest.approx(0.3) and out[2] is None
    assert out[0].requires_grad                                   # the clip is part of the differentiated function
    assert GroupClip({"policy.": 0.0})(["policy.a"], [x])[0] is x
