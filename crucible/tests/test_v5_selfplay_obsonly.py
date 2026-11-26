"""The 4.0 numeric self-play stack (`crucible.selfplay`) under Clarification 2 (workstream K regressions).

Audit K (docs/v5/audit/K_selfplay.md 3.1, skeptic S.1/S.6) found the arena defaulting to the shaped reward, the
mechanism's objective quantity (`score`) and every seat's money leaving the environment with each step, a "joint"
objective that hands a learner its partner's private net, a scripted climber that reads the (zeroed) reward slot and
a PG learner that counts optimiser steps that change nothing. These tests pin the repairs (DEF-K-2, DEF-K-3, DEF-K-4).
"""
import json

import numpy as np
import pytest
import torch

from crucible.experience.modes import LEGACY, OBSERVATION_ONLY
from crucible.research.learner import EnvSpec
from crucible.selfplay.arena import Individual, KernelSelfPlayEnv, SelfPlayEnv, arena_task, play
from crucible.selfplay.learners import ClimberLearner, PGLearner

torch.set_num_threads(1)
SPEC = EnvSpec()


def _distinct(v: float, legit: list[float]) -> bool:
    return abs(v) > 1e-6 and all(abs(v - x) > 1e-6 for x in legit)


@pytest.mark.parametrize("family", ["abm_duopoly", "tech_handoff"])
def test_default_is_observation_only_and_nothing_of_another_seat_leaves_a_step(family):
    task = arena_task(family, 3)
    env = SelfPlayEnv([task] * 2, SPEC)
    assert env.reward_mode == OBSERVATION_ONLY
    env.reset()
    rng = np.random.default_rng(0)
    sc0 = SPEC.scalars.start
    for _ in range(12):
        acts = [rng.uniform(-1, 1, size=(2, SPEC.act_dim)).astype(np.float32) for _ in range(env.n_seats)]
        obs, rew, done, info = env.step(acts)
        assert float(np.abs(rew).sum()) == 0.0
        assert info == {}                                   # every seat's money and status is operator data
        op = env.last_info
        for s in range(env.n_seats):
            o = obs[s]
            assert np.all(o[:, sc0] == 0.0)                 # the 4.0 reward slot
            # what a seat sees of money is its own observed revenue and cost, nothing else
            scale = np.asarray(env._scale)
            assert np.allclose(o[:, sc0 + 1] * scale, op["revenue_cents"][s], rtol=1e-5, atol=1e-2)
            assert np.allclose(o[:, sc0 + 2] * scale, op["cost_cents"][s], rtol=1e-5, atol=1e-2)
            # no value of the other seat's private net (or the team net) appears in this seat's observation
            other = [x for x in range(env.n_seats) if x != s]
            for i in range(env.n):
                legit = [float(op["revenue_cents"][s][i]), float(op["cost_cents"][s][i]),
                         float(op["net_cents"][s][i])]
                secrets = [float(op["net_cents"][x][i]) for x in other]
                secrets.append(float(sum(op["net_cents"][x][i] for x in range(env.n_seats))))
                seen = set(np.round(o[i].astype(np.float64) * float(scale[i]), 2).tolist()) | \
                    set(np.round(o[i].astype(np.float64), 4).tolist())
                for v in secrets:
                    if _distinct(v, legit):
                        assert round(v, 2) not in seen and round(v, 4) not in seen
        if np.asarray(done).all():
            break


def test_legacy_mode_keeps_the_4_0_step_contract():
    task = arena_task("abm_duopoly", 3)
    env = SelfPlayEnv([task], SPEC, reward_mode=LEGACY)
    env.reset()
    _, rew, _, info = env.step([None, None])
    assert set(info) == {"net_cents", "revenue_cents", "cost_cents", "status", "score"}
    assert env.last_info is info


def test_kernel_twin_hides_the_objective_quantity_and_other_seats():
    """tech_handoff's objective quantity is the team net, (sum revenue - sum cost) / n: with it a seat recovers its
    partner's private net as 2*score - own (skeptic S.1). In observation_only no participant-visible channel of the
    kernel twin carries it, and the step info is empty."""
    task = arena_task("tech_handoff", 3)
    seen_keys = {}
    for mode in (OBSERVATION_ONLY, LEGACY):
        k = KernelSelfPlayEnv(task, SPEC, reward_mode=mode)
        k.reset()
        infos = [k.step([None, None])[3] for _ in range(4)]
        a = k.actors[0]
        res = k.world.execute(a, {"tool": "rig.read", "args": {"rig": "r1"}}).to_dict()
        assert res["ok"], res
        keys = set(json.dumps(res["result"]).replace('"', " ").split())
        for e in k.world.events_range(0):
            if e.type.startswith("rig") and k.world.visible(e, k.world.principals_of(a)):
                keys |= set(k.world.participant_event(e).payload)
        seen_keys[mode] = keys
        if mode == OBSERVATION_ONLY:
            assert all(i == {} for i in infos)
            assert k.last_info["net_cents"].shape == (2, 1)
    assert "score" not in seen_keys[OBSERVATION_ONLY]
    assert "score" in seen_keys[LEGACY]                  # the 4.0 contract, explicitly selected


@pytest.mark.parametrize("kind", ["joint", "score", "mixed"])
def test_privileged_objectives_are_refused_outside_legacy(kind):
    task = arena_task("abm_duopoly", 3)
    env = SelfPlayEnv([task], SPEC)
    a = Individual("a", PGLearner(SPEC, seed=1), objective={"kind": kind, "weight": 0.5} if kind == "mixed" else kind)
    b = Individual("b", PGLearner(SPEC, seed=2))
    with pytest.raises(ValueError, match="privileged"):
        play(env, [a, b])


def test_learners_learn_from_their_own_observed_income_in_observation_only():
    task = arena_task("abm_duopoly", 3)
    env = SelfPlayEnv([task] * 2, SPEC)
    pg = PGLearner(SPEC, seed=3)
    climber = ClimberLearner(SPEC, seed=4)
    a, b = Individual("pg", pg), Individual("cl", climber)
    w0 = pg.parameters_flat().copy()
    counted = 0
    for k in range(3):
        play(env, [a, b], noise=2 * k)
        counted = pg.usage().updates
    assert np.any(pg.parameters_flat() != w0)           # the policy changed from its own observed revenue - cost
    assert counted > 0
    # the climber moved its settings away from the public defaults: it measured its own income
    assert climber.memory is not None and np.abs(climber.memory).sum() > 0.0
    assert b.learns is False and a.learns is True       # a scripted operator is not counted as a learner


def test_pg_counts_no_update_when_its_signal_is_identically_zero():
    task = arena_task("abm_duopoly", 3)
    env = SelfPlayEnv([task], SPEC)
    pg = PGLearner(SPEC, seed=5)
    obs = env.reset()
    pg.reset_context(1)
    w0 = pg.parameters_flat().copy()
    while not env.done.all():
        act = np.asarray(pg.act(obs[0], explore=True), dtype=np.float32)
        nxt, _, done, _ = env.step([act, None])
        pg.observe(obs[0], act, np.zeros(1, dtype=np.float32), done, nxt[0])
        obs = nxt
    out = pg.update()
    assert out["optimizer_steps"] >= 1
    assert out["updates"] == 0 and out["policy_changed"] is False
    assert np.all(pg.parameters_flat() == w0)
