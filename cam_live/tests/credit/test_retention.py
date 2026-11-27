"""Retention: anchors with reference outputs, functional regularisation, measured forgetting, and promotion /
rollback gated by validation and retention (CL-04, CL-05). Real model at debug scale, LabWorld experience."""
import dataclasses
import json
import math

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.credit.checkpoint import CheckpointManager, PromotionGates, promote_learner, save_learner
from cam_live.credit.labkit import collect_family, lab_eval_fn, lab_functions
from cam_live.credit.learner import LearnerSettings
from cam_live.credit.retention import (AnchorSet, RetentionProvider, evaluate_families, measure_forgetting,
                                       retention_metrics, validation_metrics)
from cam_live.diagnostics.world import LabConfig
from cam_live.learning_api import UpdatePlan
from cam_live.nn.api import GROUP_EXPERIENTIAL

FAM_A = LabConfig(n_channels=4, family="lab/A", target=30)
FAM_B = LabConfig(n_channels=4, family="lab/B", target=5, rate_scale=0.3, w_mean=(-1.0, 0.5, -0.3))


def family_ids(lab, family, held_out=None):
    recs = [r for r in lab.store.all() if r.objective_family == family]
    if held_out is not None:
        recs = [r for r in recs if lab.learner.is_validation(r) == held_out]
    return [r.exp_id for r in recs]


def test_anchor_penalty_is_zero_at_the_reference_and_measures_functional_change(lab_factory):
    lab = lab_factory(n_worlds=6, credit=False, lab_cfg=FAM_A)
    model, store = lab.model, lab.store
    anchors = AnchorSet(lab_functions(lab.world, lab.policy), max_per_family=20)
    recs = store.all()
    assert anchors.add(model, store, recs) == 20 and len(anchors) == 20 and anchors.families() == ["lab/A"]
    assert anchors.add(model, store, recs[:20]) == 0                     # an anchor keeps its first reference
    ids = anchors.ids("lab/A")[:8]
    model.eval()
    pen, m = anchors.penalty(model, store, ids)
    assert float(pen.detach()) < 1e-9 and m["n_anchors"] == 8
    assert set(m) >= {"latent_b", "latent_z", "world_log_mean", "action_probs"}
    g = torch.autograd.grad(pen, [p for p in model.param_groups()[GROUP_EXPERIENTIAL]], allow_unused=True)
    assert any(x is not None for x in g)                                 # differentiable w.r.t. the core
    with torch.no_grad():                                                # the function changes ...
        for p in model.param_groups()[GROUP_EXPERIENTIAL]:
            p.add_(0.02 * torch.randn_like(p))
    pen2, m2 = anchors.penalty(model, store, ids)
    assert float(pen2.detach()) > 1e-4 and m2["latent_b"] > 0 and m2["action_probs"] >= 0     # ... and the penalty sees it
    fresh = AnchorSet(lab_functions(lab.world, lab.policy))
    fresh.load_state_dict(anchors.state_dict())
    assert float(fresh.penalty(model, store, ids)[0].detach()) == pytest.approx(float(pen2.detach()), rel=1e-6)


def test_measure_forgetting_reports_nll_calibration_and_decision_agreement(lab_factory):
    lab = lab_factory(n_worlds=10, credit=False, lab_cfg=FAM_A)
    model, store, learner = lab.model, lab.store, lab.learner
    ev = lab_eval_fn(lab.world, lab.policy)
    ids = family_ids(lab, "lab/A")
    before = evaluate_families(model, store, {"lab/A": ids}, ev)
    s = before["lab/A"].summary()
    assert s["n"] == len(ids) and 0 < s["n_scored"] < s["n"]             # only records with a matured outcome score
    assert math.isfinite(s["nll"]) and math.isfinite(s["mean_sq_z"]) and 0 <= s["coverage90"] <= 1
    same = measure_forgetting(model, store, before, ev)["lab/A"]
    assert same["nll_change"] == 0.0 and same["decision_agreement"] == 1.0 and same["action_tv"] == 0.0
    learner.learn(UpdatePlan(trigger="test", max_steps=6))
    rep = measure_forgetting(model, store, before, ev)["lab/A"]
    assert rep["nll_change"] != 0.0 and rep["nll_after"] == pytest.approx(rep["nll_before"] + rep["nll_change"], rel=1e-4)
    assert rep["action_tv"] > 0 and 0 <= rep["decision_agreement"] <= 1 and math.isfinite(rep["calibration_change"])
    ret = retention_metrics({"lab/A": rep})
    assert ret["lab/A/neg_nll"] == -rep["nll_after"] and ret["lab/A/decision_agreement"] == rep["decision_agreement"]
    assert validation_metrics({"world": 2.5, "bad": float("nan")}) == {"heldout/neg_world": -2.5}


def test_retention_provider_keeps_the_anchored_function_closer_while_a_new_family_is_learned(lab_factory):
    def run(with_retention):
        cfg = CamLiveConfig.debug()
        cfg.optim.lr_experiential = 1e-3
        lab = lab_factory(n_worlds=8, cfg=cfg, credit=False, lab_cfg=FAM_A,
                          settings=LearnerSettings.from_cfg(cfg, strata={"recent": 1.0}))
        anchors = AnchorSet(lab_functions(lab.world, lab.policy))
        anchors.add(lab.model, lab.store, lab.store.all())
        collect_family(lab.store, lab.linker, FAM_B, 8, 6, seed=11, clock=lab.clock)
        if with_retention:
            lab.learner.register(RetentionProvider(anchors, weight=20.0, n_anchors=8))
        out = lab.learner.learn(UpdatePlan(trigger="test", max_steps=8))
        lab.model.eval()
        with torch.no_grad():
            pen, _ = anchors.penalty(lab.model, lab.store, anchors.ids()[:24])
        return float(pen), out

    drift_plain, _ = run(False)
    drift_kept, out = run(True)
    last = [r for r in out if r.timescale == "frequent"][-1]
    assert "retention" in last.losses and "world" in last.losses                # both losses in the same update
    n = json.loads(last.notes)
    assert n["experiential_share"]["retention"] > 0 and max(n["experiential_share_applied"].values()) <= 0.9 + 1e-6
    assert drift_kept < 0.7 * drift_plain, (drift_kept, drift_plain)


def test_promotion_needs_validation_and_retention_and_a_refusal_rolls_the_learner_back(lab_factory, tmp_path):
    cfg = CamLiveConfig.debug()
    lab = lab_factory(n_worlds=10, cfg=cfg, credit=False, lab_cfg=FAM_A,
                      settings=LearnerSettings.from_cfg(cfg, val_fraction=0.3))
    learner, model, store = lab.learner, lab.model, lab.store
    ev = lab_eval_fn(lab.world, lab.policy)
    mgr = CheckpointManager(str(tmp_path / "ckpt"), gates=PromotionGates(max_retention_drop=0.02, max_validation_drop=0.5))
    learner.learn(UpdatePlan(trigger="test", max_steps=3))
    held = family_ids(lab, "lab/A", held_out=True)
    assert held
    base = evaluate_families(model, store, {"lab/A": held}, ev)
    rep0 = measure_forgetting(model, store, base, ev)
    dec0, info0, rb0 = promote_learner(mgr, learner, retention=retention_metrics(rep0))
    assert dec0.promoted and rb0 is None and mgr.promoted().checkpoint_id == info0.checkpoint_id
    assert any(k.startswith("heldout/neg_") for k in dec0.validation)
    good_hashes, good_counters = model.group_hashes(), dict(learner.counters)
    # a damaging stretch of learning: the earlier family's predictions get worse
    learner.provider("world").weight = -3.0        # (-1.0 stopped being damaging after nn/ e00d28f: NLL change -0.085)
    lr = dict(learner.lr)
    learner.lr.update({g: 3e-3 for g in lr})
    learner.learn(UpdatePlan(trigger="test", max_steps=14))
    learner.provider("world").weight = 1.0
    learner.lr.update(lr)
    rep1 = measure_forgetting(model, store, base, ev)
    assert rep1["lab/A"]["nll_change"] > 0.02
    dec1, info1, rb1 = promote_learner(mgr, learner, retention=retention_metrics(rep1))
    assert not dec1.promoted and any("forgetting" in r for r in dec1.reasons)
    assert rb1 is not None and rb1["rolled_back_to"] == info0.checkpoint_id
    assert model.group_hashes() == good_hashes and learner.counters["frequent"] == good_counters["frequent"]
    assert mgr.promoted().checkpoint_id == info0.checkpoint_id
    assert [e["event"] for e in mgr.promotion_log()] == ["promote", "promote", "rollback"]
    # the rejected checkpoint is retained for comparison; the internal reward can never justify a promotion
    assert mgr.get(info1.checkpoint_id) is not None
    info2 = save_learner(mgr, learner)
    dec2 = mgr.promote(info2.checkpoint_id, validation={"intrinsic_reward": 99.0}, retention={"reward": 1.0})
    assert not dec2.promoted and dec2.ignored == ["intrinsic_reward", "reward"]
    # the learner keeps learning from the restored state
    after = learner.learn(UpdatePlan(trigger="test", max_steps=1))[0]
    assert after.param_groups and after.hash_before == {g: h for g, h in good_hashes.items() if g in after.hash_before}
