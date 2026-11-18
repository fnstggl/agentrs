"""Product-family fix: training-only goal curriculum and the failure-horizon cost."""
from __future__ import annotations

import torch

from agentrs.cam.envs import ProductEnv
from agentrs.cam.model import CAM, CAMConfig
from agentrs.cam.runtime import Workers

TINY = dict(d=64, belief=96, layers=1, heads=2, n_options=4)


def _target(seed, split, difficulty):
    e = ProductEnv()
    e.difficulty = difficulty
    e.reset(seed, split)
    return e.target, e.sizes.tolist(), e.budget


def test_curriculum_scales_only_training_goals():
    for seed in range(5):
        for split in ("val", "test"):  # evaluation goals and hidden markets are bit-identical at any difficulty
            assert _target(seed, split, 0.1) == _target(seed, split, 1.0)
        full, sizes, budget = _target(seed, "train", 1.0)
        easy, sizes2, budget2 = _target(seed, "train", 0.1)
        assert sizes == sizes2 and budget == budget2  # same hidden market, only the goal is easier
        assert abs(easy - max(50.0, full * 0.1)) < 1e-6 and easy < full


def test_workers_apply_difficulty_on_train_split_only():
    torch.manual_seed(0)
    m = CAM(CAMConfig(**TINY))
    tr = Workers(m, 4, {"product": 1.0}, "train", difficulty={"product": 0.2}, gen_seed=1)
    va = Workers(m, 4, {"product": 1.0}, "val", difficulty={"product": 0.2}, gen_seed=1)
    assert tr._new_env()[2].difficulty == 0.2 and va._new_env()[2].difficulty == 1.0


def test_failure_horizon_cost_removes_the_reward_for_ending_early():
    torch.manual_seed(0)
    m = CAM(CAMConfig(**TINY))
    runs = {}
    for flag in (False, True):  # identical seeds -> identical trajectories; only the terminal reward differs
        w = Workers(m, 8, {"product": 1.0}, "train", gen_seed=5, fail_horizon_cost=flag)
        runs[flag] = w.collect(fixed=[("product", i) for i in range(16)], version="v0")
    a = {e.seed: e for e in runs[False]}
    n_early = 0
    for e in runs[True]:
        o = a[e.seed]
        assert [s.act_tokens for s in e.steps] == [s.act_tokens for s in o.steps]
        for s1, s0 in zip(e.steps[:-1], o.steps[:-1]):
            assert s1.reward == s0.reward
        extra = 0.0 if e.success else 0.005 * max(0, ProductEnv.max_steps - len(e.steps))
        assert abs((o.steps[-1].reward - e.steps[-1].reward) - extra) < 1e-9
        n_early += (not e.success) and len(e.steps) < ProductEnv.max_steps
        # every failed episode now pays at least the step cost of the full horizon (undiscounted sum)
        if not e.success:
            costs = sum(0.005 for _ in e.steps) + extra
            assert abs(costs - 0.005 * max(ProductEnv.max_steps, len(e.steps))) < 1e-9
    assert n_early > 0  # the untrained policy does end some episodes early (bankruptcy): the case being fixed


def test_malformed_curriculum_config_is_rejected():
    import pytest
    import yaml

    from agentrs.cam.train import TrainConfig

    bad = yaml.safe_load("{product: {start:0.1}}")  # missing space: YAML reads 'start:0.1' as a key
    with pytest.raises(ValueError):
        TrainConfig(curriculum=bad)
    ok = TrainConfig(curriculum=yaml.safe_load("{product: {start: 0.1, lo: 0.15, hi: 0.5}}"))
    assert ok.curriculum["product"]["start"] == 0.1
