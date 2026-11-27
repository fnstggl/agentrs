"""Complete learner state on the real CamLiveModel (debug scale): save / load, identical continuation, tolerant
loading, weights-only export and the fresh-model loader (CL-05, CL-06, spec 15.2 and 10.4)."""
import json
import os

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import OutcomeKind
from cam_live.credit import checkpoint as ckpt
from cam_live.credit.diagnose import probe_predictions, register_heads_view
from cam_live.credit.labkit import _canonical, build_lab, lab_functions, register_lab_heads
from cam_live.credit.learner import LearnerSettings
from cam_live.credit.retention import AnchorSet, RetentionProvider
from cam_live.learning_api import UpdatePlan, encode_decisions, matured_outcomes
from cam_live.nn.api import GROUP_REWARD

STEP = UpdatePlan(trigger="test", max_steps=1)


def _cfg():
    cfg = CamLiveConfig.debug()
    cfg.optim.adapter_every = 2
    return cfg


def _build(path, cfg):
    lab = build_lab(path, cfg, bridge=True, beta=1e-6, settings=LearnerSettings.from_cfg(cfg, val_fraction=0.25, grad_last=4,
                                                                                        batch_size=12))
    rp = RetentionProvider(AnchorSet(lab_functions(lab.world, lab.policy), max_per_family=12), n_anchors=4)
    lab.learner.register(rp)
    return lab, rp


def _fast(lab, n=3):
    recs = [r for r in lab.store.all() if matured_outcomes(r)][:n]
    batch = encode_decisions(lab.model, lab.store, recs, grad=False)
    qs = [[_canonical(matured_outcomes(r)[0][0])] for r in recs]
    phi = lab.model.features(batch.z_chosen[:, None], lab.model.encode_queries(qs)[0])[:, 0, 0]
    for i, r in enumerate(recs):
        lab.learner.fast_update(phi[i], float(matured_outcomes(r)[0][1].value), OutcomeKind.COUNT, exposure=qs[i][0].exposure)


def _digest(recs):
    return [(r.timescale, r.losses, r.hash_after, r.exp_ids, r.grad_norms) for r in recs]


def test_saved_learner_state_is_complete_and_continues_identically(lab_factory, tmp_path):
    cfg = _cfg()
    lab = lab_factory(n_worlds=10, collect=True, cfg=cfg, bridge=True, beta=1e-6,
                      settings=LearnerSettings.from_cfg(cfg, val_fraction=0.25, grad_last=4, batch_size=12))
    learner, store = lab.learner, lab.store
    rp = RetentionProvider(AnchorSet(lab_functions(lab.world, lab.policy), max_per_family=12), n_anchors=4)
    learner.register(rp)
    rp.anchors.add(lab.model, store, store.all()[:12])
    _fast(lab)
    plan = learner.poll()
    learner.learn(UpdatePlan(trigger=plan.trigger, max_steps=3, exp_ids=plan.exp_ids))      # credit assigned, 3 + 1 updates
    root = str(tmp_path / "ckpt")
    info = ckpt.save_learner(root, learner, current_task={"objective_id": "obj-1"}, auth_config_hash="abc",
                             reward_version="r0", lifetime_context={"h": torch.arange(4.0)})
    # --- everything spec 15.2 lists is in the checkpoint
    st = ckpt.load_checkpoint(root)
    assert set(st.weights) >= {"experiential", "heads", "semantic_adapters", "buffers"}
    assert set(st.optimizers) == {"experiential", "heads", "semantic_adapters"}
    assert st.fast_state["post"]["count"]["n"] == 3 and torch.equal(st.lifetime_context["h"], torch.arange(4.0))
    assert st.current_task == {"objective_id": "obj-1"} and st.auth_config_hash == "abc" and st.reward_version == "r0"
    assert st.replay_state["draws"] > 0 and st.pending_pointer["head_seq"] == store.head_seq()
    assert st.schedules["counters"]["frequent"] == 3 and st.schedules["counters"]["slow"] == 1
    assert st.schedules["sched_step"]["experiential"] == 3 and st.schedules["probe_ids"] == learner.probe_ids
    assert st.rng["torch"] is not None and st.model_version == "m3.1" == info.model_version
    assert st.arch_signature["scale"] == "debug" and "credit_rudder" in st.arch_signature["heads"]
    assert st.extra["config"]["core"]["scale"] == "debug" and st.extra["credit"]["selector"]["default"] == "simple"
    assert set(st.aux_state["providers"]["retention"]["anchors"]["refs"]) == set(rp.anchors.refs)
    assert ckpt.verify_checkpoint(info.path) == []
    # --- the original continues; then a fresh learner over the same store resumes from the checkpoint
    cont = [r for _ in range(4) for r in learner.learn(STEP)]
    final = lab.model.group_hashes()
    fast_mu = learner.fast_state.posterior(OutcomeKind.COUNT)["mu"].clone()
    path = store.path
    lab.close()
    lab2, rp2 = _build(path, _cfg())
    try:
        assert lab2.model.group_hashes() != final
        rep = ckpt.load_learner(root, lab2.learner, expected_auth_hash="abc")
        assert rep["checkpoint_id"] == info.checkpoint_id and rep["weights"].clean
        assert rep["arch_signature_differences"] == [] and rep["auth_config_changed"] is False
        assert all(v["fresh"] == [] and v["mismatched"] == [] for v in rep["optimizers"].values())
        assert rep["provider_state"] == {"restored": ["retention"], "missing": []} and rep["fast_state"] == "restored"
        assert lab2.learner.counters["frequent"] == 3 and lab2.learner.model_version == "m3.1"
        assert set(rp2.anchors.refs) == set(rp.anchors.refs)
        assert lab2.learner.fast_state.n_updates(OutcomeKind.COUNT) == 3
        resumed = [r for _ in range(4) for r in lab2.learner.learn(STEP)]
        assert _digest(resumed) == _digest(cont)                # same batches, losses, gradients and weights, bit for bit
        assert lab2.model.group_hashes() == final
        assert torch.equal(lab2.learner.fast_state.posterior(OutcomeKind.COUNT)["mu"], fast_mu)
    finally:
        lab2.close()


def test_a_head_added_after_the_checkpoint_is_reported_not_fatal_and_strict_loading_refuses(lab_factory, tmp_path):
    cfg = _cfg()
    lab = lab_factory(n_worlds=6, cfg=cfg, credit=False)
    lab.learner.learn(STEP)
    root = str(tmp_path / "ckpt")
    ckpt.save_learner(root, lab.learner)
    h = lab.model.group_hashes()
    lab.model.register_head("added_later", torch.nn.Linear(lab.model.d_model, 3))
    with torch.no_grad():
        for p in lab.model.head("world").parameters():
            p.add_(1.0)
    with pytest.warns(UserWarning, match="tolerant"):
        rep = ckpt.load_learner(root, lab.learner)
    assert sorted(rep["weights"].missing) == ["heads.added_later.bias", "heads.added_later.weight"]
    assert rep["weights"].unexpected == [] and "heads" in rep["arch_signature_differences"]
    assert rep["optimizers"]["heads"]["fresh"] == ["heads.added_later.weight", "heads.added_later.bias"] or \
        set(rep["optimizers"]["heads"]["fresh"]) == {"heads.added_later.weight", "heads.added_later.bias"}
    assert lab.model.group_hashes()["experiential"] == h["experiential"]
    out = lab.learner.learn(STEP)[0]                            # the new head trains from its initial values
    assert out.param_groups
    with pytest.raises(ckpt.CheckpointMismatch):
        ckpt.load_learner(root, lab.learner, strict=True)


def test_weights_only_export_builds_a_fresh_model_without_experience_and_with_reset_fast_beliefs(lab_factory, tmp_path):
    cfg = _cfg()
    lab = lab_factory(n_worlds=8, cfg=cfg, bridge=True, settings=LearnerSettings.from_cfg(cfg, grad_last=4))
    model, store, learner = lab.model, lab.store, lab.learner
    psi = model.register_head("r_psi", torch.nn.Linear(model.d_model, 1), GROUP_REWARD)        # learned reward travels too
    learner.learn(UpdatePlan(trigger="test", max_steps=4))
    _fast(lab)
    out_dir = str(tmp_path / "export")
    meta = ckpt.export_learner_weights(learner, out_dir)
    assert sorted(os.listdir(out_dir)) == ["config.json", "weights.safetensors"]
    keys = ckpt.export_keys(out_dir)
    assert any(k.startswith("reward::heads.r_psi") for k in keys) and any(k.startswith("semantic_adapters::") for k in keys)
    assert not any("backbone.lm.tok" in k or "optim" in k for k in keys) and meta["config"]["core"]["scale"] == "debug"
    recs = [r for r in store.all() if r.result.expected_outcomes][:4]
    probe = [{"items": [i.to_dict() for i in store.items(r.context_item_refs)],
              "actions": [c.action.to_dict() for c in r.decision.candidates], "goal": r.goal_text,
              "query": _canonical(r.result.expected_outcomes[0]).to_dict()} for r in recs]
    want = probe_predictions(model, {"world": lab.world, "policy": lab.policy, "rudder": lab.rudder}, probe)

    def heads(m):
        register_lab_heads(m, None)
        m.register_head("r_psi", torch.nn.Linear(m.d_model, 1), GROUP_REWARD)

    fresh = ckpt.load_exported_model(out_dir, register_heads=heads)
    assert fresh.model is not model and fresh.report.clean and fresh.config["format"] == ckpt.EXPORT_FORMAT
    got = probe_predictions(fresh.model, register_heads_view(fresh.model), probe)
    assert set(got) == set(want) and all(torch.equal(got[k], want[k]) for k in want)
    assert fresh.model.group_hashes() == model.group_hashes()
    # task-specific fast beliefs do not travel: the fresh state sits at the learned prior, the source has evidence
    assert fresh.fast_state.n_updates(OutcomeKind.COUNT) == 0 and learner.fast_state.n_updates(OutcomeKind.COUNT) == 3
    # heads are code: without them the loader refuses instead of silently dropping learned tensors
    with pytest.raises(ckpt.CheckpointMismatch):
        ckpt.load_exported_model(out_dir, register_heads=lambda m: register_lab_heads(m, None, credit=False))
    with pytest.warns(UserWarning, match="tolerant"):
        tol = ckpt.load_exported_model(out_dir, register_heads=lambda m: register_lab_heads(m, None, credit=False),
                                       strict=False)
    assert any("credit_rudder" in k for k in tol.report.unexpected) and any("r_psi" in k for k in tol.report.unexpected)
    assert tol.report.missing == []
