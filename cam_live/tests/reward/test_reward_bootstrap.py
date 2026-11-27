"""Cold-start bootstrap signal, RewardMixer promotion, noisy-TV behaviour (spec 7.6, 7.8, Part 8; MR-08, MR-09)."""
import math

import pytest
import torch

from cam_live.reward import bootstrap as B
from cam_live.reward.integrity import NoisyTVGate, depends_on, noisy_tv_audit
from cam_live.reward.intrinsic import IntrinsicReward
from cam_live.reward.signals import N_SIGNALS

DT = torch.float64
t = lambda *v: torch.tensor(v, dtype=DT)


def test_learning_progress_is_improvement_on_the_same_targets_not_raw_error():
    g = torch.Generator().manual_seed(0)
    # a learnable target (constant 3.0) and an unpredictable one (unit noise); the model is a running mean
    y_learn, y_noise = torch.full((400,), 3.0, dtype=DT), 5.0 * torch.randn(400, dtype=DT, generator=g)
    lp = {}
    for name, y in (("learnable", y_learn), ("noise", y_noise)):
        before = (y[200:] - 0.0) ** 2                         # prediction before learning: 0
        after = (y[200:] - y[:200].mean()) ** 2               # after fitting on the first half, same targets
        lp[name] = B.learning_progress(before, after)
        assert lp[name].shape == (200,)
    assert float(lp["learnable"].mean()) == pytest.approx(9.0)
    assert abs(float(lp["noise"].mean())) < 0.5 < float(((y_noise[200:]) ** 2).mean())   # big error, no progress
    with pytest.raises(ValueError):
        B.learning_progress(t(1.0, 2.0), t(1.0))
    tr = B.LearningProgressTracker()
    tr.record_before("probe-7", 2.5)
    assert tr.progress("probe-7", 1.0) == pytest.approx(1.5) and tr.progress("probe-7", 1.0) is None
    assert tr.progress("never-seen", 0.1) is None             # no earlier loss on those targets: no progress claimed


def test_bootstrap_reward_is_labelled_bounded_and_made_of_its_named_components():
    boot = B.BootstrapReward(bound=1.0)
    out = boot(learning_progress=t(0.5, -0.2, 50.0), delta_v=t(0.1, -0.3, 0.9),
               epistemic_before=t(1.0, 1.0, 1.0), epistemic_after=t(0.5, 1.0, 0.0),
               verified_delta=t(0.0, 0.0, 400.0), verified_mask=t(0.0, 0.0, 1.0))
    assert out.label == "bootstrap" and out.version == B.BOOTSTRAP_VERSION
    assert set(out.components) == {"learning_progress", "delta_v", "exploration", "verified"}
    assert float(out.total.abs().max()) < 1.0
    assert torch.allclose(out.total, torch.tanh(sum(out.components.values())))
    assert out.components["exploration"].tolist() == [0.25, 0.0, 0.5]
    assert out.components["verified"][:2].tolist() == [0.0, 0.0] and float(out.components["verified"][2]) > 0
    only = boot(delta_v=t(0.2))
    assert float(only.total) == pytest.approx(math.tanh(0.2))


def test_exploration_bonus_follows_epistemic_not_aleatoric_uncertainty():
    boot = B.BootstrapReward()
    # same epistemic reduction under tiny and huge observation noise / prediction error: same bonus
    a = boot(epistemic_before=t(0.8), epistemic_after=t(0.4)).components["exploration"]
    assert float(a) == pytest.approx(0.25)
    # no epistemic reduction: nothing, however surprising the observation was (the bonus has no error input)
    assert float(boot(epistemic_before=t(0.4), epistemic_after=t(0.4)).components["exploration"]) == 0.0
    assert float(boot(epistemic_before=t(0.4), epistemic_after=t(0.6)).components["exploration"]) == 0.0
    import inspect
    assert not any("aleatoric" in p or "error" in p for p in inspect.signature(boot.__call__).parameters)


def _stream(learnable: bool, n: int = 240, seed: int = 0):
    """An agent repeatedly observing one context. Conjugate-normal belief about the mean of the observations.
    learnable: fixed mean, small noise (uncertainty keeps shrinking fast at first, then is exhausted).
    not learnable (noisy TV): pure noise with a large variance — epistemic std of the *mean* still shrinks as
    1/sqrt(n) and then flattens, while the prediction error stays large forever."""
    g = torch.Generator().manual_seed(seed)
    sigma = 0.1 if learnable else 5.0
    ys = (2.0 if learnable else 0.0) + sigma * torch.randn(n, dtype=DT, generator=g)
    epi_b, epi_a, err, lp = [], [], [], []
    mean, cnt = 0.0, 0
    for y in ys.tolist():
        epi_b.append(sigma / math.sqrt(cnt + 1))
        err.append(abs(y - mean))
        new_mean = (mean * cnt + y) / (cnt + 1)
        lp.append((y - mean) ** 2 - (y - new_mean) ** 2 if cnt else 0.0)
        mean, cnt = new_mean, cnt + 1
        epi_a.append(sigma / math.sqrt(cnt + 1))
    return epi_b, epi_a, err, lp


def test_noisy_tv_reward_for_irreducible_randomness_decays():
    boot = B.BootstrapReward(B.BootstrapWeights(learning_progress=0.0, delta_v=0.0, exploration=1.0, verified=0.0))
    epi_b, epi_a, err, _ = _stream(learnable=False)
    paid = [float(boot(epistemic_before=t(b), epistemic_after=t(a), context_keys=["tv"]).total)
            for b, a in zip(epi_b, epi_a)]
    audit = noisy_tv_audit(paid, epi_a)
    assert audit["epistemic_flat_in_tail"] and audit["passed"], audit
    assert sum(paid[:20]) > 1.0 and sum(paid[-60:]) < 1e-3                 # paid while learning, then nothing
    assert sum(err[-60:]) / 60 > 2.0                                        # although surprise never went away
    # the audit has power: a curiosity signal that pays raw prediction error fails it on the same stream
    naive = [math.tanh(e) for e in err]
    assert not noisy_tv_audit(naive, epi_a)["passed"]
    # total reward collected from the random stream is finite and small compared with its length
    assert sum(paid) < 0.05 * len(paid)


def test_noisy_tv_gate_reopens_when_uncertainty_falls_again():
    gate = NoisyTVGate(window=4, min_rel_drop=0.05)
    for _ in range(20):
        g_flat = gate.update("ctx", 1.0)
    assert g_flat < 1e-3
    e = 1.0
    for _ in range(12):
        e *= 0.7
        g_open = gate.update("ctx", e)
    assert g_open > 0.9 and gate.gate("other") == 1.0
    twin = NoisyTVGate(window=4); twin.load_state_dict(gate.state_dict())
    assert twin.gate("ctx") == gate.gate("ctx")


def test_mixer_starts_as_pure_bootstrap_and_logs_which_component_produced_what():
    mixer = B.RewardMixer()
    rb, rl = t(0.3, -0.1), t(0.9, 0.9)
    out = mixer.combine(rb, rl, reward_version="rpsi1-x", exp_ids=["e1", "e2"],
                        bootstrap_components={"delta_v": t(0.3, -0.1)})
    assert mixer.mix == 1.0 and out.label == "bootstrap" and torch.equal(out.total, rb)
    entry = mixer.log[-1]
    assert entry["from_learned"] == [0.0, 0.0] and entry["from_bootstrap"] == rb.tolist()
    assert entry["exp_ids"] == ["e1", "e2"] and entry["reward_version"] == "rpsi1-x" and entry["label"] == "bootstrap"
    assert entry["bootstrap_components"]["delta_v"] == [0.3, -0.1]
    f = out.record_fields(0)
    assert f["intrinsic_reward"] == pytest.approx(0.3) and f["bootstrap_reward"] == pytest.approx(0.3)
    assert f["reward_version"].startswith("bootstrap:mix=1.000:")           # never labelled as learned
    mixer.mix = 0.25
    out = mixer.combine(rb, rl, reward_version="rpsi1-x")
    assert torch.allclose(out.total, 0.25 * rb + 0.75 * rl)
    assert mixer.log[-1]["from_learned"] == pytest.approx([0.675, 0.675])
    # D12: R_psi is never labelled learned / mixed before a meta-update was applied *and* promoted
    assert out.label == "bootstrap" and out.record_fields(0)["reward_version"].startswith("bootstrap:mix=0.250:")
    mixer.mix = 0.0
    assert mixer.combine(rb, rl, reward_version="v").label == "bootstrap"          # random R_psi, mix 0
    mixer.note_meta_update()
    assert mixer.combine(rb, rl, reward_version="v").label == "bootstrap"          # applied, not promoted
    mixer.mix, mixer.min_independent = 0.5, 2
    assert mixer.update([0.1, 0.2, 0.15]).action == "lower" and mixer.promoted
    assert mixer.combine(rb, rl, reward_version="v").label == "mixed"
    mixer.mix = 0.0
    assert mixer.combine(rb, rl, reward_version="v").label == "learned"
    twin = B.RewardMixer(); twin.load_state_dict(mixer.state_dict())
    assert twin.promoted and twin.meta_updates_applied == 1 and twin.promotions == 1
    with pytest.raises(ValueError):
        B.RewardMixer(mix=1.5)


def test_promotion_before_any_meta_update_does_not_earn_the_learned_label():
    """D12: a 'lower' decision taken before psi was ever meta-updated is not a promotion of a learned reward."""
    mixer = B.RewardMixer(mix=0.5, step=0.5, min_independent=2)
    assert mixer.update([0.1, 0.2, 0.15]).action == "lower" and mixer.mix == 0.0
    rb, rl = t(0.3, -0.1), t(0.9, 0.9)
    assert not mixer.promoted and mixer.combine(rb, rl, reward_version="v").label == "bootstrap"


def test_mixing_is_not_a_gradient_path_into_the_reward_parameters():
    R = IntrinsicReward(8, d_hidden=16, d_life=4).double()
    g = torch.Generator().manual_seed(0)
    rn = lambda *s: torch.randn(*s, dtype=DT, generator=g)
    r_learned = R(rn(3, 8), rn(3, 8), rn(3, 8), rn(3, 8), rn(3, N_SIGNALS))
    assert depends_on(r_learned, R)
    out = B.RewardMixer(mix=0.5).combine(t(0.1, 0.2, 0.3), r_learned, reward_version=R.version)
    assert not out.total.requires_grad and not depends_on(out.total, R)


def test_promotion_lowers_mix_only_on_held_out_evidence_of_no_harm_and_moves_back():
    """lower on shown non-inferiority, raise (revert) on shown harm, hold while inconclusive or too thin."""
    mixer = B.RewardMixer(step=0.1, min_independent=8)
    hold = mixer.update([0.5, 0.4, 0.6])                                    # 3 independent outcomes: not enough
    assert hold.action == "hold" and mixer.mix == 1.0 and "insufficient_independent_evidence:3<8" in hold.reason
    good = [0.05, 0.02, 0.08, 0.01, 0.04, 0.06, 0.03, 0.05, 0.02, 0.07]
    d = mixer.update(good)
    assert d.action == "lower" and mixer.mix == pytest.approx(0.9) and d.lower_bound > 0 and d.n_independent == 10
    d = mixer.update(good)
    assert mixer.mix == pytest.approx(0.8)
    harmful = [-x for x in good]
    d = mixer.update(harmful)                                               # the learned reward hurts: move back
    assert d.action == "raise" and mixer.mix == pytest.approx(0.9) and d.upper_bound < 0
    noisy = [0.5, -0.6, 0.4, -0.5, 0.6, -0.4, 0.5, -0.6, 0.1, -0.1]         # inconclusive: neither lowered nor raised
    d = mixer.update(noisy)
    assert d.action == "hold" and d.reason == "inconclusive_evidence" and mixer.mix == pytest.approx(0.9)
    assert mixer.update(harmful).mix_after == pytest.approx(1.0)
    assert mixer.update(harmful).mix_after == 1.0                           # capped
    assert mixer.update([float("nan")] * 10).action == "hold"
    low = B.RewardMixer(mix=0.05, step=0.1, min_independent=8)
    assert low.update(good).mix_after == 0.0                                # floored
    assert [x["label"] for x in mixer.log].count("promotion") == 8
    # non-inferiority margin relative to the utility scale: a small mean loss inside the margin still promotes
    near = [x - 0.03 for x in good]                                         # mean -0.01, lower bound ~ -0.025
    strict = B.RewardMixer(min_independent=8)
    assert strict.update(near).action == "hold"
    lenient = B.RewardMixer(min_independent=8, rel_margin=0.05)
    dec = lenient.update(near, scale=1.0)
    assert dec.action == "lower" and dec.margin == pytest.approx(0.05) and lenient.mix == pytest.approx(0.9)
    twin = B.RewardMixer(); twin.load_state_dict(mixer.state_dict())
    assert twin.mix == mixer.mix


def test_paired_policy_value_diffs_count_each_independent_group_once():
    lp_l = torch.log(t(0.5, 0.5, 0.5, 0.9, 0.9, 0.2))
    lp_b = torch.log(t(0.25, 0.25, 0.25, 0.3, 0.3, 0.4))
    mu = torch.log(t(0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    U = t(1.0, 1.0, 1.0, 2.0, 2.0, -1.0)
    groups = torch.tensor([7, 7, 7, 3, 3, 9])
    d = B.paired_policy_value_diffs(lp_l, lp_b, mu, U, groups)
    assert len(d) == 3                                                      # three outcomes, not six rows
    assert d == pytest.approx([(1.8 - 0.6) * 2.0, (1.0 - 0.5) * 1.0, (0.4 - 0.8) * -1.0])
