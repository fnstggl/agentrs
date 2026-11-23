"""Mandatory tests 1, 2, 3 and 9, plus program, memory, replay and checkpoint integrity."""
import numpy as np
import pytest
import torch

from crucible.research.learner import EnvSpec
from superlearner.unified.config import small
from superlearner.unified.encoders.programs import (ADD, DEFAULT, EOS, SET, WAIT, Grammar, encode, execute, features,
                                                    parse, structured_candidates)
from superlearner.unified.interfaces import ObservationBatch, masked_log_softmax
from superlearner.unified.learning.replay import EpisodeReplay
from superlearner.unified.models.unified import UnifiedModel
from superlearner.unified.runtime.learner import UnifiedLearner
from superlearner.unified.runtime.toy import ToyVecEnv, run_toy

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tiny():
    return small().with_(d_model=48, obs_layers=1, obs_heads=2, obs_latent_tokens=4, deter_dim=64, stoch_vars=4,
                         stoch_classes=4, memory_slots=3, memory_dim=48, action_dim=48, goal_dim=48, batch_size=4,
                         seq_len=10, burn_in=2, candidates=6, updates_per_step=0.5, multi_horizons=(2, 4))


def obs_batch(b, n=33, device=DEV, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return ObservationBatch(torch.randn(b, n, device=device, generator=g), torch.zeros(b, n, device=device),
                            torch.ones(b, n, dtype=torch.bool, device=device), torch.arange(n, device=device).expand(b, -1))


@pytest.fixture(scope="module")
def filled():
    lr = UnifiedLearner(EnvSpec(), 0, DEV, tiny())
    run_toy(lr, ToyVecEnv(4, horizon=20), 24, learn=True, updates=False)
    return lr


# ---------------------------------------------------------------------- Test 1
def test_tensor_integrity():
    c = tiny()
    m = UnifiedModel(c).to(DEV)
    o = obs_batch(5)
    e, pooled = m.encode_observation(o)
    assert e.shape == (5, c.obs_latent_tokens, c.d_model) and pooled.shape == (5, c.d_model)
    g = torch.zeros(5, c.goal_dim, device=DEV)
    s = m.initial_belief(5, DEV)
    a = torch.zeros(5, c.action_dim, device=DEV)
    b = m.update_belief(s, a, o, torch.ones(5, device=DEV), g)
    assert b.h.shape == (5, c.deter_dim) and b.z.shape == (5, c.stoch_vars, c.stoch_classes)
    assert b.memory.shape == (5, c.memory_slots, c.memory_dim) and b.u.shape == (5, c.d_model)
    assert torch.allclose(b.z.detach().sum(-1), torch.ones(5, c.stoch_vars, device=DEV))
    assert m.decode_observation(b.u).shape == (5, 33)
    assert m.estimate_value(b.u, g).shape == (5,)


def test_missing_observations_do_not_break_encoding():
    m = UnifiedModel(tiny()).to(DEV)
    o = obs_batch(2)
    o.present[0] = False
    _, p = m.encode_observation(o)
    assert torch.isfinite(p).all()


def test_masked_policy_never_selects_invalid():
    logits = torch.tensor([[10.0, 0.0, -1.0]])
    mask = torch.tensor([[False, True, True]])
    lp = masked_log_softmax(logits, mask)
    assert lp[0, 0] < -1e8 and torch.exp(lp[0, 1:]).sum().item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        masked_log_softmax(logits, torch.zeros_like(mask))


def test_candidate_scorer_is_permutation_equivariant():
    c = tiny()
    m = UnifiedModel(c).to(DEV).eval()
    u, g = torch.randn(2, c.d_model, device=DEV), torch.randn(2, c.goal_dim, device=DEV)
    e = torch.randn(2, 6, c.action_dim, device=DEV)
    mask = torch.ones(2, 6, dtype=torch.bool, device=DEV)
    perm = torch.randperm(6, device=DEV)
    a = m.score_candidates(u, g, e, mask)
    b = m.score_candidates(u, g, e[:, perm], mask)
    assert torch.allclose(a[:, perm], b, atol=1e-4)


# ---------------------------------------------------------------------- Test 2
def test_gradients_reach_every_trainable_parameter(filled):
    t = filled.trainer
    batch = filled.replay.sample(4, 40)          # whole episodes, so the initial state is trained too
    before = {k: [p.detach().clone() for p in v] for k, v in t.groups.items()}
    t.update(batch)     # distributional heads start at exactly zero output, so their inner layers move from step 2
    t.update(filled.replay.sample(4, 40))
    for name, params in t.groups.items():
        dead = [i for i, p in enumerate(params) if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not dead, f"{name}: {len(dead)} of {len(params)} parameters got no finite gradient"
        moved = sum(int(not torch.equal(a, b)) for a, b in zip(before[name], params))
        assert moved > 0.9 * len(params), f"{name}: only {moved}/{len(params)} changed"


def test_gradient_ownership(filled):
    """Actor and critic losses must not reach the belief layers in the default configuration."""
    t, m = filled.trainer, filled.model
    b = filled.replay.sample(4, 10).to(DEV)
    s, goal, _, _ = t.unroll(b, 2)
    b.mask = t.training_mask(b, 2)
    actor, critic, *_ = t.control_loss(b, s, goal, 0)
    for o in t.opt.values():
        o.zero_grad(set_to_none=True)
    (actor + critic).backward()
    assert all(p.grad is None for p in t.groups["world"])
    assert any(p.grad is not None for p in t.groups["actor"])


# ---------------------------------------------------------------------- Test 3
def test_posterior_uses_observation_and_prior_does_not():
    c = tiny()
    m = UnifiedModel(c).to(DEV)
    g = torch.zeros(3, c.goal_dim, device=DEV)
    s = m.initial_belief(3, DEV)
    a = torch.zeros(3, c.action_dim, device=DEV)
    dt = torch.ones(3, device=DEV)
    b1 = m.update_belief(s, a, obs_batch(3, seed=1), dt, g)
    b2 = m.update_belief(s, a, obs_batch(3, seed=2), dt, g)
    assert torch.allclose(b1.prior_logits, b2.prior_logits)
    assert not torch.allclose(b1.post_logits, b2.post_logits)


def test_no_future_information_in_replay_unroll(filled):
    t = filled.trainer
    b = filled.replay.sample(4, 10).to(DEV)
    torch.manual_seed(5)
    s1, *_ = t.unroll(b, 0)
    b.values = b.values.clone()
    b.values[:, 6:] += 5.0
    b.deltas = b.deltas.clone()
    b.deltas[:, 6:] -= 3.0
    torch.manual_seed(5)
    s2, *_ = t.unroll(b, 0)
    assert torch.allclose(s1.h[:, :7], s2.h[:, :7], atol=1e-5)
    assert torch.allclose(s1.post_logits[:, :6], s2.post_logits[:, :6], atol=1e-5)
    assert not torch.allclose(s1.post_logits[:, 6:], s2.post_logits[:, 6:], atol=1e-3)


# ---------------------------------------------------------------------- Test 9
def test_imagination_has_no_observation_channel():
    c = tiny()
    m = UnifiedModel(c).to(DEV)
    g = torch.zeros(2, c.goal_dim, device=DEV)
    a = torch.zeros(2, c.action_dim, device=DEV)
    b = m.update_belief(m.initial_belief(2, DEV), a, obs_batch(2), torch.ones(2, device=DEV), g)
    nxt, out = m.predict_future(b, a, torch.ones(2, device=DEV), g)
    assert nxt.post_logits is None and set(out) == {"reward", "continue", "reward_std"}
    assert "embed" not in m.rssm.imagine.__code__.co_varnames


def test_imagination_gate_stays_closed_until_calibrated(filled):
    t = filled.trainer
    t.heldout_r2 = float("nan")
    b = filled.replay.sample(4, 10).to(DEV)
    s, goal, *_ = t.unroll(b, 2)
    b.mask = t.training_mask(b, 2)
    assert t.control_loss(b, s, goal, 0)[2]["ac/imag"].item() == 0.0
    t.heldout_r2 = -0.5
    assert t.control_loss(b, s, goal, 0)[2]["ac/imag"].item() == 0.0


# ---------------------------------------------------------------------- programs
def test_program_roundtrip_and_execution():
    st = [("set", 2, 0.5), ("wait", 2), ("add", 2, -0.3), ("wait", 1)]
    p = parse(encode(st))
    sched = execute(p, np.zeros(12), np.ones(12))
    assert sched.shape == (3, 12)
    assert sched[:, 2].tolist() == pytest.approx([0.5, 0.5, 0.2], abs=1e-6)


def test_parser_rejects_malformed_programs():
    good = encode([("wait", 1)])
    for bad in ([EOS] + [0] * 19, [1, SET, 2], [1, WAIT, 3, EOS], [1, EOS] + [0] * 18):
        assert parse(np.array(bad)) is None
    corrupted = good.copy()
    corrupted[-1] = WAIT
    assert parse(corrupted) is None


def test_program_cannot_touch_unavailable_lever():
    mask = np.zeros(12)
    mask[1] = 1
    assert execute([("set", 5, 0.1)], np.zeros(12), mask) is None
    f, ticks = features([("set", 5, 0.1)], np.zeros(12), mask)
    assert ticks == 0 and f.sum() == 0


def test_generator_samples_always_parse_and_respect_masks():
    c = tiny()
    m = UnifiedModel(c).to(DEV)
    b = 256
    lm = torch.zeros(b, 12, device=DEV)
    lm[:, [1, 4]] = 1
    t, lp = m.generator.sample(torch.randn(b, c.d_model, device=DEV), torch.randn(b, c.goal_dim, device=DEV), lm)
    mask = lm[0].cpu().numpy()
    for row in t.cpu().numpy():
        st = parse(row)
        assert st is not None, row
        assert all(s[0] == "wait" or mask[s[1]] > 0.5 for s in st)
    again = m.generator.log_prob(torch.randn(b, c.d_model, device=DEV), torch.randn(b, c.goal_dim, device=DEV), t, lm)
    assert torch.isfinite(again).all() and (lp <= 0).all()


def test_generator_log_prob_matches_sampling():
    c = tiny()
    m = UnifiedModel(c).to(DEV).eval()
    u, g = torch.randn(32, c.d_model, device=DEV), torch.randn(32, c.goal_dim, device=DEV)
    lm = torch.ones(32, 12, device=DEV)
    t, lp = m.generator.sample(u, g, lm)
    assert torch.allclose(m.generator.log_prob(u, g, t, lm), lp, atol=1e-3)


# ---------------------------------------------------------------------- memory
def test_memory_write_is_selective_and_bounded():
    c = tiny().with_(memory_slots=8)
    m = UnifiedModel(c).to(DEV).rssm.memory
    mem = torch.randn(2, 8, c.memory_dim, device=DEV)
    ev = torch.randn(2, c.deter_dim + c.stoch_vars * c.stoch_classes, device=DEV)
    new, share = m.write(mem, ev, torch.zeros(2, device=DEV))
    assert new.shape == mem.shape and ((share >= 0) & (share <= 1)).all()
    assert (share.sum(-1) <= 1 + 1e-5).all()
    assert share.std(-1).mean() > 0


# ---------------------------------------------------------------------- replay
def _ep(n, reward=0.0):
    k, l, f = 2, DEFAULT.max_len, DEFAULT.n_features
    return {"values": np.zeros((n, 33)), "deltas": np.zeros((n, 33)), "present": np.ones((n, 33), bool), "extra": np.zeros((n, 33, 4)),
            "goal": np.zeros((n, 4)), "cand_tokens": np.zeros((n, k, l), dtype=np.int16),
            "cand_features": np.zeros((n, k, f)), "cand_mask": np.ones((n, k), bool), "cand_dur": np.ones((n, k)), "cand_bonus": np.zeros((n, k)), "rscale": np.ones(n),
            "action": np.zeros(n, dtype=np.int64), "behavior_log_prob": np.zeros(n), "reward": np.full(n, reward),
            "dt": np.ones(n), "cont": np.ones(n), "first": np.eye(n)[0]}


def test_reservoir_is_uniform_over_everything_seen():
    survive = 0
    for s in range(300):
        r = EpisodeReplay(capacity_per_family=10, recent_per_family=1, holdout=0.0, seed=s)
        r.add(_ep(3, reward=1.0), "f")
        for _ in range(39):
            r.add(_ep(3), "f")
        survive += any(e["reward"][0] == 1.0 for e in r.reservoir["f"])
    assert 0.15 < survive / 300 < 0.35      # expected 10/40


def test_families_do_not_evict_each_other():
    r = EpisodeReplay(capacity_per_family=5, holdout=0.0)
    for _ in range(5):
        r.add(_ep(3), "old")
    for _ in range(500):
        r.add(_ep(3), "new")
    assert len(r.reservoir["old"]) == 5


def test_replay_pads_and_masks():
    r = EpisodeReplay(holdout=0.0)
    r.add(_ep(4), "f")
    b = r.sample(2, 8)
    assert b.mask.sum(1).tolist() == [4.0, 4.0] and b.cand_mask[:, 4:, 0].all() and (b.cont[:, 4:] == 0).all()


# ---------------------------------------------------------------------- checkpoint
def test_checkpoint_restores_weights_and_excludes_context(filled, tmp_path):
    path = str(tmp_path / "c.pt")
    filled.save(path)
    other = UnifiedLearner(EnvSpec(), 1, DEV, tiny())
    other.load(path)
    a, b = filled.model.state_dict(), other.model.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert "state" not in torch.load(path, map_location="cpu", weights_only=False)
    assert other.trainer.steps == filled.trainer.steps


def test_frozen_evaluation_changes_nothing(filled):
    before = {k: v.clone() for k, v in filled.model.state_dict().items()}
    n = len(filled.replay)
    run_toy(filled, ToyVecEnv(4, horizon=12), 8, learn=False)
    assert len(filled.replay) == n
    assert all(torch.equal(before[k], v) for k, v in filled.model.state_dict().items())
