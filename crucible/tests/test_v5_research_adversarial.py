"""Workstream N: the instrument against adversarial learners and the control learners' known mechanics
(DEF-N-2, DEF-N-4, DEF-N-6 regressions; SP-15.8 hooks in `crucible.research.v5.adversarial`)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from crucible.research.v5 import adversarial as A
from crucible.research.v5 import fresh as FR
from crucible.research.v5 import learners as LR
from crucible.research.v5.distribution import build_registry
from crucible.research.v5.episodes import run_episode, score_episode
from crucible.research.v5.sampler import build_env

FAM = "t1:rapid:cx_balancing_loop"


@pytest.fixture(scope="module")
def reg():
    return build_registry()


def _train(L, reg, n=4, reps=2, seed=0):
    rng = np.random.default_rng(seed)
    for s in range(310, 310 + n):
        for _ in range(reps):
            run_episode(build_env(reg[FAM], s, {}, {"mode": "assigned"}), L, learn=True, rng=rng, verify=False)


def test_instrument_worlds_are_observation_only_and_learners_read_only_their_cash(reg):
    """DEF-N-4: the v5 instrument builds its worlds through crucible.experience in its default mode; no reward
    reaches a learner (scalar 0.0) and the control learners' signal is the change of their own cash."""
    env = build_env(reg[FAM], 3, {}, {"mode": "assigned"})
    assert env.reward_mode == "observation_only"
    a = env.agents()[0]
    run_episode(env, LR.make("adapter", 0), learn=False, rng=np.random.default_rng(0))
    assert env.reward(a) == 0.0
    seen = []

    class Spy(LR.Adapter):
        def observe(self, k, s):
            seen.append(s)
            super().observe(k, s)
    env2 = build_env(reg[FAM], 3, {}, {"mode": "assigned"})
    rec = run_episode(env2, Spy(0), learn=False, rng=np.random.default_rng(0))
    assert len(seen) == rec["ticks"] and abs(sum(seen) - rec["observed_signal_sum"]) < 1e-6


def test_fresh_child_is_jailed_and_checks_its_confinement(reg, tmp_path):
    w = LR.make("weights", 1)
    _train(w, reg)
    man = FR.save_checkpoint(w, str(tmp_path / "w"))
    r = FR.weights_only_eval(man, [{"entry": FAM, "seed": 1_000_101}])
    assert r["ok"], r
    assert set(r["checks"]) == set(FR.CHECKS) and r["checks"]["filesystem_confined"] and r["jailed"]
    open_ = FR.weights_only_eval(man, [{"entry": FAM, "seed": 1_000_101}], jail=False)
    assert not open_["ok"] and open_["checks"]["filesystem_confined"] is False


def test_configuration_smuggling_is_refused(reg, tmp_path):
    L = LR.make("adv_config_smuggle", 3)
    _train(L, reg, n=2, reps=1)
    man = FR.save_checkpoint(L, str(tmp_path / "s"))
    assert json.load(open(os.path.join(man["dir"], "arch.json")))["cfg"]["table"]
    r = FR.weights_only_eval(man, [{"entry": FAM, "seed": 310}])
    assert not r["ok"] and r["checks"]["authorised_configuration"] is False and "table" in r["refused"]


def test_file_stash_outside_the_checkpoint_is_unreadable_under_the_jail(reg, tmp_path):
    L = LR.make("adv_file_stash", 9_901)
    try:
        os.unlink(L._stash())
    except OSError:
        pass
    worlds = [{"entry": FAM, "seed": s} for s in (1_000_101, 1_000_102)]
    L.frozen = True                                         # scored as the weights-only child plays: no search
    base = np.mean([score_episode(reg, w["entry"], w["seed"], L, learn=False, rng=np.random.default_rng(0))["score"]
                    for w in worlds])
    L.frozen = False
    _train(L, reg)
    try:
        man = FR.save_checkpoint(L, str(tmp_path / "f"))
        jailed = FR.weights_only_eval(man, worlds)
        unjailed = FR.weights_only_eval(man, worlds, jail=False)
    finally:
        os.unlink(L._stash())
    fj = np.mean([x["score"] for x in jailed["scores"]])
    fu = np.mean([x["score"] for x in unjailed["scores"]])
    assert jailed["ok"] and abs(fj - base) < 1e-9          # the jail credits nothing to the stash
    assert fu - base > A.GAIN_MIN                           # without the jail the stash would have worked


def test_weights_replay_is_memory_not_parameters(reg):
    w = LR.make("weights", 2)
    _train(w, reg, n=2, reps=1)
    assert len(w.memory_state()["replay"]) == 2 and set(w.parameters()) == {"W1", "b1", "W2", "b2"}
    w.clear_memory()
    assert w.memory_state() == {"replay": []}


def test_transfer_learner_credits_its_strategy_from_the_second_episode(reg):
    """Regression: the bandit was credited only at a task's 4th episode, so a battery playing each training world 3
    times left the transfer control untrained."""
    T = LR.make("transfer", 4)
    _train(T, reg, n=3, reps=3)
    assert T.bandit.n.sum() == 3
    T2 = LR.make("transfer", 4)
    _train(T2, reg, n=3, reps=4)
    assert T2.bandit.n.sum() == 3                            # a task counts once, whatever its episode count


def test_generator_has_no_fixed_rule_shortcut():
    """DEF-N-6: fixed public rules (the audit's anti-default rule, quartiles, centre) reach less than SHORTCUT_MAX
    of the oracle's improvement in the median calibrated generated world."""
    r = A.calibration_shortcut(n_structures=8, seeds=(0,))
    assert r["worlds"] >= 6
    assert r["passed"], r["median_share_of_oracle"]
