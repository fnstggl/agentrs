"""Mandatory tests 4, 5, 6 and 7: short real training runs on controlled worlds, judged on held-out behaviour."""
import numpy as np
import pytest
import torch

from crucible.research.learner import EnvSpec
from superlearner.unified.config import small
from superlearner.unified.encoders.programs import encode, parse
from superlearner.unified.interfaces import BeliefState
from superlearner.unified.runtime.learner import UnifiedLearner
from superlearner.unified.runtime.toy import ToyVecEnv, run_toy

DEV = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.slow


def cfg(**kw):
    base = small().with_(d_model=96, obs_layers=1, obs_heads=2, obs_latent_tokens=4, deter_dim=128, stoch_vars=8,
                         stoch_classes=8, memory_slots=4, memory_dim=96, action_dim=96, goal_dim=96, batch_size=32,
                         seq_len=12, burn_in=0, candidates=8, updates_per_step=0.1, multi_horizons=(2, 4),
                         lr_world=1e-3, lr_actor=1e-3, lr_critic=1e-3, generator_lr=1e-3, entropy_coef=3e-3,
                         use_imagination=False)
    return base.with_(**kw)


def train(lr, env, rounds):
    for _ in range(rounds):
        run_toy(lr, env, env.n, learn=True)


# ---------------------------------------------------------------------- Test 4
def test_world_model_prediction_improves_on_held_out_episodes():
    lr = UnifiedLearner(EnvSpec(), 0, DEV, cfg(use_generator=False))
    lr.replay.holdout = 0.25
    env = ToyVecEnv(16, horizon=12, seed=1)
    run_toy(lr, env, 256, learn=True, updates=False)
    held = lr.replay.sample_holdout(64, 12)           # about 64 held-out episodes, so the R² is not dominated by noise
    assert held is not None
    before = lr.trainer.heldout_prediction(held)["heldout/reward_r2"]
    for _ in range(250):
        lr.trainer.update(lr.replay.sample(16, 12))
    after = lr.trainer.heldout_prediction(held)["heldout/reward_r2"]
    assert after > before + 0.2 and after > 0.3, (before, after)


# ---------------------------------------------------------------------- Test 5
def _cue_score(window: int, seed: int) -> float:
    """CAM 7.0 memory: causal attention over the last ``window`` events. Window 1 sees only the current event."""
    c = cfg(use_generator=False, use_adaptive=False, candidates=6, core="attention", event_window=window,
            lr_world=3e-4, lr_actor=3e-4, lr_critic=3e-4)
    lr = UnifiedLearner(EnvSpec(), seed, DEV, c)
    # Two opposed one-tick programs make the decision depend only on remembering the cue.
    lr.proposer = lambda rng, mask, k: ([encode([("set", 0, 0.8), ("wait", 1)]), encode([("set", 0, -0.8), ("wait", 1)])]
                                        * k)[:k]
    env = ToyVecEnv(16, "cue_final", horizon=8, seed=seed)
    train(lr, env, 250)
    score = run_toy(lr, ToyVecEnv(64, "cue_final", horizon=8, seed=100 + seed), 128, learn=False, explore=False)
    return score * 8 / 0.5          # +1 = always right on the rewarded last tick, 0 = chance


def test_event_memory_beats_single_event_control():
    """Only the last of eight ticks is rewarded, by the cue shown on the first: no reward along the way reveals the
    answer, so a learner without memory of the first event is at chance."""
    with_mem = _cue_score(16, 3)
    without = _cue_score(1, 3)
    assert with_mem > 0.6 and with_mem > without + 0.4, (with_mem, without)


# ---------------------------------------------------------------------- Test 6
def test_policy_probabilities_move_toward_rewarded_action():
    c = cfg(use_generator=False, candidates=4)
    lr = UnifiedLearner(EnvSpec(), 0, DEV, c)
    good, bad = encode([("set", 0, 0.6), ("wait", 1)]), encode([("set", 0, -0.6), ("wait", 1)])
    lr.proposer = lambda rng, mask, k: [good, bad, good, bad][:k]
    env = ToyVecEnv(16, horizon=8, seed=2)

    def p_good() -> float:
        probe = ToyVecEnv(16, horizon=8, seed=3)
        obs = probe.reset()
        lr.reset_context(16)
        was = lr.frozen
        lr.freeze(True)
        lr.act(obs, explore=False)
        lr.freeze(was)
        idx = torch.arange(16, device=DEV)
        toks = np.stack([np.stack([good, bad, good, bad])] * 16)
        from superlearner.unified.encoders.programs import features
        f = np.zeros((16, 4, lr.model.action_encoder.n_struct), dtype=np.float32)
        for k, t in enumerate((good, bad, good, bad)):
            f[:, k, :lr.lang.n_features] = features(parse(t), obs[0, lr.spec.last_action], obs[0, lr.spec.lever_mask])[0]
        with torch.no_grad():
            e = lr.model.action_encoder(torch.as_tensor(toks, device=DEV), torch.as_tensor(f, device=DEV))
            g = lr.model.encode_goal(type("G", (), {"features": torch.as_tensor(lr._goal(obs, np.arange(16)), device=DEV)})())
            lp = lr.model.score_candidates(lr.state.u[idx], g, e, torch.ones(16, 4, dtype=torch.bool, device=DEV))
        return float(lp.exp()[:, [0, 2]].sum(1).mean())

    before = p_good()
    train(lr, env, 40)
    after = p_good()
    assert after > 0.85 and after > before + 0.2, (before, after)
    score = run_toy(lr, ToyVecEnv(32, horizon=8, seed=4), 32, learn=False, explore=False)
    assert score > 0.9


# ---------------------------------------------------------------------- Test 7
def test_trainable_generator_learns_a_program_the_fixed_proposer_cannot_produce():
    """The rewarded composition needs two specific levers at once, which the fixed proposer below never emits."""
    target = {2: 0.6, 4: -0.6}
    c = cfg(candidates=8, generator_fraction=0.75, generator_lr=2e-3, awr_temperature=0.5)
    lr = UnifiedLearner(EnvSpec(), 0, DEV, c)
    hold = encode([("wait", 1)])
    lr.proposer = lambda rng, mask, k: [hold] * k
    env = ToyVecEnv(16, horizon=6, seed=5, target=target)

    @torch.no_grad()
    def generated_reward(n=512) -> float:
        u = torch.zeros(n, c.d_model, device=DEV)
        probe = ToyVecEnv(1, horizon=6, target=target)
        obs = probe.reset()
        lr.reset_context(1)
        was = lr.frozen
        lr.freeze(True)
        lr.act(obs, explore=False)
        lr.freeze(was)
        u = lr.state.u[0].expand(n, -1)
        g = lr.model.encode_goal(type("G", (), {"features": torch.as_tensor(lr._goal(obs, np.arange(1)), device=DEV)})())
        tok, _ = lr.model.generate_candidates(u, g.expand(n, -1), torch.as_tensor(obs[0, lr.spec.lever_mask], device=DEV).expand(n, -1))
        out = []
        for row in tok.cpu().numpy():
            cur = np.zeros(12)
            for st in parse(row):
                if st[0] == "set":
                    cur[st[1]] = st[2]
                elif st[0] == "add":
                    cur[st[1]] = np.clip(cur[st[1]] + st[2], -1, 1)
            out.append(1 - 2 * np.mean([(cur[k] - v) ** 2 for k, v in target.items()]))
        return float(np.mean(out))

    before = generated_reward()
    train(lr, env, 120)
    after = generated_reward()
    assert after > before + 0.25, (before, after)
