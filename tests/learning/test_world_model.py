"""World-model behaviour: conditioning, alignment, no future leakage, probabilistic outputs,
uncertainty, unseen actions, and learning a held-out synthetic dynamics problem better than baselines."""

import torch

from agentrs.data.diff import N_EVENTS, StateDiff
from agentrs.environments.base import ElementState, Observation
from agentrs.evaluation.wm_eval import FrequencyBaseline, evaluate_world_model
from agentrs.models.world_model import N_OUT, WorldModel, WorldModelConfig, bootstrap_mask
from agentrs.training.common import OptimConfig
from agentrs.training.world_model import WMTrainConfig, make_wm_examples, train_world_model


def _obs():
    return Observation(env_id="t", url="/form", title="Form",
                       elements=[ElementState(eid="e1", role="textbox", name="Name", value=""),
                                 ElementState(eid="e2", role="button", name="Submit")])


def test_input_contains_action_and_only_current_observation(scratch_bb, demo_data):
    train, _ = demo_data
    wm = WorldModel(WorldModelConfig(backbone=scratch_bb, n_heads=2))
    for ex in make_wm_examples(wm, train)[:40]:
        t = ex.transition
        assert f"ACTION: {t.action}\nEFFECT:" in ex.prompt  # action conditioning
        assert t.observation.to_text(wm.cfg.max_obs_elements) in ex.prompt  # before-observation aligned
        assert ex.diff_text == t.diff_text  # target computed from the same transition's after-observation
        nxt = t.next_observation.to_text(wm.cfg.max_obs_elements)
        if nxt != t.observation.to_text(wm.cfg.max_obs_elements):
            assert nxt not in ex.prompt  # the future observation never appears in the input


def test_probabilistic_outputs_and_uncertainty(scratch_bb):
    wm = WorldModel(WorldModelConfig(backbone=scratch_bb, n_heads=4))
    pred = wm.predict("Type 'a'", _obs(), 'type(e1, "a")', n_samples=3, greedy=False,
                      generator=torch.Generator().manual_seed(0))
    for p in pred.event_probs + [pred.reward_prob, pred.terminal_prob]:
        assert 0.0 <= p <= 1.0
    assert all(s >= 0 for s in pred.event_std) and pred.epistemic >= 0
    assert len(pred.diffs) == 3 and all(lp <= 0 for _, lp in pred.diffs)
    assert len(pred.next_observations) == 3
    # identical heads => zero epistemic uncertainty
    with torch.no_grad():
        for prm in (wm.heads.w1, wm.heads.b1, wm.heads.w2, wm.heads.b2):
            prm.copy_(prm[:1].expand_as(prm))
    m, s = wm.predict_heads([wm.prompt("g", _obs(), "click(e2)")])
    assert float(s.max()) < 1e-6


def test_unseen_actions_handled(scratch_bb):
    wm = WorldModel(WorldModelConfig(backbone=scratch_bb, n_heads=2))
    for a in ['navigate("/never/seen")', "click(e999)", 'select(e1, "Zzz")', "skill(foo)"]:
        pred = wm.predict("g", _obs(), a, n_samples=1)
        assert all(torch.isfinite(torch.tensor(pred.event_probs)))
        assert isinstance(pred.next_observations[0], Observation)


def test_bootstrap_masks_are_deterministic_and_cover_examples():
    m1 = bootstrap_mask([f"k{i}" for i in range(50)], 5, 0.5, "cpu")
    m2 = bootstrap_mask([f"k{i}" for i in range(50)], 5, 0.5, "cpu")
    assert torch.equal(m1, m2) and bool((m1.sum(0) > 0).all())
    assert 0.3 < float(m1.mean()) < 0.8


def test_learns_heldout_dynamics_better_than_baselines(scratch_bb, demo_data):
    """Small learnable dynamics problem (3 MiniWeb families): after training, held-out
    event prediction beats the action-type-conditional frequency baseline."""
    torch.manual_seed(0)
    train, val = demo_data
    wm = WorldModel(WorldModelConfig(backbone=scratch_bb, n_heads=3))
    cfg = WMTrainConfig(steps=250, batch_size=16, eval_every=250,
                        optim=OptimConfig(lr=2e-3, total_steps=250, warmup_steps=10, schedule="cosine"))
    r = train_world_model(wm, train, val, cfg)
    assert r["after"]["val_loss"] < r["before"]["val_loss"]
    assert r["after"]["val_diff_nll_per_token"] < r["before"]["val_diff_nll_per_token"]
    res = evaluate_world_model(wm, train, val, n_generate=10, n_drift=3)
    assert res["model"]["event_nll"] < res["action_type"]["event_nll"] < res["marginal"]["event_nll"]
    assert res["model"]["terminal_nll"] < res["action_type"]["terminal_nll"]
    _ = (FrequencyBaseline, N_OUT, N_EVENTS, StateDiff)
