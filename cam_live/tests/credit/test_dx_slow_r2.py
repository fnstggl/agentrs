"""DX round 2 regression tests for the slow (adapter) step: G2L-9 micro vetoes and multi-provider dominance (CL-03).

Probe losses with a known, linear effect of the adapters: training and held-out loss of a provider are
10 + s * <u, theta> / n (u: a fixed direction per provider; eval_sign -1 turns a training gain into a held-out rise),
so first-order statements are exact."""
import json

import torch

from cam_live.config import CamLiveConfig
from cam_live.credit.learner import LearnerSettings
from cam_live.learning_api import TIMESCALE_SLOW, UpdatePlan
from cam_live.nn.api import GROUP_ADAPTERS


class Linear:
    timescale = TIMESCALE_SLOW
    needs_latents = False

    def __init__(self, name, s, direction=None, eval_sign=1.0):
        self.name, self.s, self.u, self.eval_sign, self.weight = name, s, direction, eval_sign, 1.0

    def eligible(self, record):
        return True

    def loss(self, model, batch, store):
        ps = model.param_groups()[GROUP_ADAPTERS]
        n = sum(p.numel() for p in ps)
        if self.u is None:
            x = sum(p.sum() for p in ps)
        else:
            x = sum((p * u).sum() for p, u in zip(ps, self.u))
        v = 10.0 + self.s * x / n
        return (v if model.training else self.eval_sign * v), {}


def _lab(lab_factory, lr=1e-3, **settings):
    cfg = CamLiveConfig.debug()
    cfg.optim.lr_adapters = lr
    st = dict(slow_guard=True, val_fraction=0.4, val_batch=32, slow_batch_size=4, warmup_steps=0)
    st.update(settings)
    torch.manual_seed(0)
    return lab_factory(n_worlds=8, n_decisions=6, cfg=cfg, world=False, policy=False, credit=False,
                       settings=LearnerSettings.from_cfg(cfg, **st))


def _slow(L):
    rec = L.learn(UpdatePlan(trigger="test", timescales=(TIMESCALE_SLOW,)))[-1]
    return rec, json.loads(rec.notes or "{}")


def _adapters(m):
    return [p.detach().clone() for p in m.param_groups()[GROUP_ADAPTERS]]


def _moved(a, m):
    return any(not torch.equal(x, y) for x, y in zip(a, m.param_groups()[GROUP_ADAPTERS]))


def test_a_consistent_rise_of_1e_minus_5_relative_no_longer_vetoes_a_step_that_helps_another_provider(lab_factory):
    """G2-learn's real-configuration case: evaluator +6.8e-6 relative (all chunks equal: paired SE 0) vetoed a step
    that improved the world model's held-out loss by 14 %."""
    lab = _lab(lab_factory)
    L, m = lab.learner, lab.model
    L.register(Linear("world", s=1400.0))                         # trains and generalises: -14 %
    L.register(Linear("evaluator", s=0.068, eval_sign=-1.0))     # chunks identical; held-out +6.8e-6 relative
    L.refresh_eligibility(full=True)
    w0 = _adapters(m)
    rec, notes = _slow(L)
    g = notes["guard"]
    ev, wo = g["by_provider"]["evaluator"], g["by_provider"]["world"]
    assert 0 < ev["paired_change"] / abs(ev["validation_before"]) < 1e-5 and ev["paired_se"] == 0.0
    assert wo["paired_change"] / abs(wo["validation_before"]) < -0.1
    assert g["accepted"] and _moved(w0, m)                          # old rule: tolerance 1e-6, vetoed
    assert ev["tolerance"] >= 1e-4 * abs(ev["validation_before"]) and ev["min_tolerance"] > 1e-7 * 10


def test_a_single_validation_chunk_gets_the_relative_minimum_not_the_float_floor(lab_factory):
    lab = _lab(lab_factory, val_batch=1)
    L, m = lab.learner, lab.model
    L.register(Linear("world", s=1400.0))
    L.register(Linear("evaluator", s=0.068, eval_sign=-1.0))
    L.refresh_eligibility(full=True)
    rec, notes = _slow(L)
    g = notes["guard"]
    assert all(v["n_chunks"] == 1 for v in g["by_provider"].values())
    assert g["accepted"]


def test_a_real_degradation_above_the_minimum_is_still_refused_and_no_evidence_still_defers(lab_factory):
    lab = _lab(lab_factory)
    L, m = lab.learner, lab.model
    L.register(Linear("world", s=1400.0))
    L.register(Linear("evaluator", s=20.0, eval_sign=-1.0))      # +2e-3 relative: a real rise
    L.refresh_eligibility(full=True)
    w0 = _adapters(m)
    rec, notes = _slow(L)
    g = notes["guard"]
    assert not g["accepted"] and not _moved(w0, m)
    assert g["by_provider"]["evaluator"]["paired_change"] > g["by_provider"]["evaluator"]["tolerance"]
    lab2 = _lab(lab_factory, val_fraction=0.0)
    lab2.learner.register(Linear("world", s=1400.0))
    lab2.learner.refresh_eligibility(full=True)
    rec, notes = _slow(lab2.learner)
    assert notes["deferred"] is True and notes["guard"] is None


def _directions(m, seed, mix=None):
    g = torch.Generator().manual_seed(seed)
    u = [torch.randn(p.shape, generator=g) for p in m.param_groups()[GROUP_ADAPTERS]]
    if mix is None:
        return u
    base, a, b = mix
    return [a * x + b * y for x, y in zip(base, u)]


def test_a_dominant_loss_no_longer_pushes_the_adapters_against_another_providers_loss(lab_factory):
    """The grounding loss dominated the adapter gradient (7.0 against the world model's 0.026) and its step raised the
    world model's held-out loss. With the conflict-aware combination the applied update is first-order
    non-increasing for every provider (exact here: the losses are linear), and the step is kept."""
    for balance in (False, True):
        lab = _lab(lab_factory, slow_balance=balance)
        L, m = lab.learner, lab.model
        u_g = _directions(m, 1)
        u_w = _directions(m, 2, mix=(u_g, -0.6, 0.8))             # partly opposed to the grounding direction
        L.register(Linear("grounding", s=1000.0, direction=u_g))
        L.register(Linear("world", s=10.0, direction=u_w))
        L.refresh_eligibility(full=True)
        w0 = _adapters(m)
        rec, notes = _slow(L)
        g = notes["guard"]
        world = g["by_provider"]["world"]["paired_change"]
        if not balance:                                           # the old combination: world rises, step refused
            assert world > g["by_provider"]["world"]["tolerance"] and not g["accepted"] and not _moved(w0, m)
        else:
            up = notes["update_projection"]
            assert up["first_order_change_after"]["world"] <= 1e-9 and up["first_order_change_after"]["grounding"] < 0
            assert world <= 1e-6 and g["by_provider"]["grounding"]["paired_change"] < 0
            assert g["accepted"] and _moved(w0, m)
            assert notes["slow_balance"]["conflicts"]                # the gradients did conflict


def test_cocoa_reads_the_representation_and_the_update_does_not_raise_its_loss_to_first_order(lab_factory):
    """COCOA's hindsight model trains on detached latents (no adapter gradient), but its held-out loss rose 1-22 %
    with the adapter steps (G2-learn round 2). Its sensitivity gradient (attached latents, its own training records
    only, never a training step of COCOA) now constrains the applied update."""
    cfg = CamLiveConfig.debug()
    cfg.optim.lr_adapters = 1e-3
    torch.manual_seed(0)
    lab = lab_factory(n_worlds=8, n_decisions=6, cfg=cfg, world=False, policy=False, credit=True, bridge=True,
                      settings=LearnerSettings.from_cfg(cfg, slow_guard=True, val_split="record", val_fraction=0.3,
                                                        slow_batch_size=4, warmup_steps=0))
    L, m, cocoa = lab.learner, lab.model, lab.cocoa
    L.refresh_eligibility(full=True)
    from cam_live.learning_api import TIMESCALE_FREQUENT
    for _ in range(4):                      # a trained hindsight model (at initialisation its residual is zero)
        L.learn(UpdatePlan(trigger="test", timescales=(TIMESCALE_FREQUENT,), max_steps=2))
    trained_before = set(cocoa.trained_objectives)
    rng = torch.get_rng_state()
    batch_recs = lab.store.get_many(lab.store.sample_ids(4, loss=cocoa.name, seed=1))
    from cam_live.credit.learner import encode_with_adapter_grad
    batch = encode_with_adapter_grad(m, lab.store, batch_recs, grad_last=4)
    loss = cocoa.adapter_sensitivity_loss(m, batch, lab.store)
    assert loss is not None and loss.requires_grad
    g = torch.autograd.grad(loss, m.param_groups()[GROUP_ADAPTERS], allow_unused=True)
    assert any(x is not None and float(x.abs().sum()) > 0 for x in g)        # it does depend on the adapters
    assert torch.equal(rng, torch.get_rng_state()) and cocoa.trained_objectives == trained_before
    rec, notes = _slow(L)
    assert notes.get("protected_readers", {}).get("cocoa", 0) > 0, {k: notes.get(k) for k in ("skipped", "guard")}
    if True:
        up = notes["update_projection"]
        assert up["first_order_change_after"]["cocoa"] <= 1e-9 * max(1.0, abs(up["first_order_change_before"]["cocoa"]))


def test_a_rejected_step_shrinks_the_next_adapter_step_and_accepted_ones_grow_it_back(lab_factory):
    lab = _lab(lab_factory)
    L, m = lab.learner, lab.model
    probe = L.register(Linear("world", s=1400.0, eval_sign=-1.0))     # every step raises its held-out loss
    L.refresh_eligibility(full=True)
    scales, lrs = [], []
    for _ in range(3):
        rec, notes = _slow(L)
        assert not notes["guard"]["accepted"]
        scales.append(notes["step_scale"])
        lrs.append(notes["lr"]["semantic_adapters"])
    assert scales == [1.0, 0.5, 0.25] and abs(lrs[1] / lrs[0] - 0.5) < 1e-12
    assert L.slow_guard_state["step_scale"] == 0.125
    probe.eval_sign = 1.0                                              # now it helps: accepted, the step grows
    rec, notes = _slow(L)
    assert notes["guard"]["accepted"] and notes["step_scale"] == 0.125
    assert L.slow_guard_state["step_scale"] == 0.125 * 1.5
    assert L.state_dict()["slow_guard"]["step_scale"] == 0.125 * 1.5          # checkpointed with the schedules
    L.slow_guard_state["step_scale"] = 1.0 / 16
    probe.eval_sign = -1.0
    rec, notes = _slow(L)
    assert not notes["guard"]["accepted"] and L.slow_guard_state["step_scale"] == 1.0 / 16    # bounded below
