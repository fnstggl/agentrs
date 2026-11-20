"""sl-exp-v18 anchor penalty: off by default (identical updates), zero against itself, and the anchor model's
recurrent state is carried exactly across segments."""
import copy

import numpy as np
import pytest
import torch

from agentrs.sl import worlds as W
from agentrs.sl.learner import PPO
from agentrs.sl.model import SLConfig, SLCore
from agentrs.sl.runner import Runner
from agentrs.sl.world import RewardContract, RewardSpec

MC = dict(d=32, dh=48, layers=1, heads=2, dk=16, dv=16, mem_slots=16, skills=4, ds=8, da=32, ens=2, fam_gate_n0=64,
          head_bank=["cam/bizdev"])
RC = RewardContract(RewardSpec(native=1.0, success=1.0, shaping=0.5, time_cost=0.0, violation=1.0, gamma=0.99))
FAMS = {"cam/bizdev": 0.5, "cam/negotiation": 0.5}


def _setup(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = SLCore(SLConfig(**MC))
    r = Runner(m, FAMS, W.make_world, W.sample_task, 8, trial_episodes=3, split="train", seed=seed, mode="full",
               contract=RC, device="cpu", max_episode_steps=200)
    return m, r


def _seg(r, T=12):
    with torch.no_grad():
        return r.collect(T)


def test_anchor_off_is_unchanged():
    m, r = _setup()
    try:
        seg = _seg(r)
    finally:
        r.close()
    m2 = copy.deepcopy(m)
    a, b = PPO(m, {"mb_streams": 4}), PPO(m2, {"mb_streams": 4, "anchor": 0.0})
    b.anchor_model = copy.deepcopy(m)  # present but coefficient 0: must not change anything
    np.random.seed(1)
    sa = a.update(seg, "cpu")
    np.random.seed(1)
    sb = b.update(seg, "cpu")
    assert "anchor_kl" not in sa and "anchor_kl" not in sb
    assert sa == sb
    for (n, p), (_, q) in zip(m.named_parameters(), m2.named_parameters()):
        assert torch.equal(p, q), n


def test_anchor_against_itself_is_zero_and_state_is_carried():
    m, r = _setup(3)
    try:
        ppo = PPO(m, {"mb_streams": 4, "anchor": 1.0, "lr": 0.0, "epochs": 1})
        ppo.anchor_model = copy.deepcopy(m).eval()
        kls = []
        for _ in range(4):  # several segments: rows continue trials across segment boundaries
            seg = _seg(r)
            hix = torch.stack([s["hix"] for s in seg["steps"]])
            assert bool((hix == 0).any()), "no shared-head decisions collected"
            st = ppo.update(seg, "cpu")
            kls.append(st["anchor_kl"])
    finally:
        r.close()
    # lr = 0: the policy never changes, so the anchor equals the policy; any mismatch would come from the anchor's
    # recurrent state not being carried exactly from one segment to the next
    assert max(kls) < 1e-6, kls


def test_anchor_pulls_toward_source():
    m, r = _setup(5)
    try:
        src = copy.deepcopy(m).eval()
        ppo = PPO(m, {"mb_streams": 4, "anchor": 1.0})
        ppo.anchor_model = src
        seg = _seg(r)
        first = ppo.update(seg, "cpu")["anchor_kl"]
        later = [ppo.update(_seg(r), "cpu")["anchor_kl"] for _ in range(3)]
    finally:
        r.close()
    assert first >= 0 and all(k >= 0 for k in later)
    assert max(later) > 0  # the policy moved away from the frozen source and the penalty sees it
