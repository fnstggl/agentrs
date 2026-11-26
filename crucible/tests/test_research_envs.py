"""Tasks and numeric environments: a task is the same problem in the simulator and in the kernel, nothing private
reaches a learner, and the normalisation means what it says."""
import json

import numpy as np
import pytest

from crucible.research import tasks as T
from crucible.research.baselines import HoldLearner, RandomLearner
from crucible.research.bridge import KernelMechEnv
from crucible.research.envs import MechVecEnv, money_scale, run_episodes
from crucible.research.learner import EnvSpec
LEGACY = "legacy_shaped_reward"   # 4.0 baseline: the shaped reward is selected explicitly (Clarification 2, OO.8)

SPEC = EnvSpec()


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    import os
    d = tmp_path_factory.mktemp("crucible-cache")
    old = os.environ.get("CRUCIBLE_CACHE")
    os.environ["CRUCIBLE_CACHE"] = str(d)
    yield d
    if old is None:
        os.environ.pop("CRUCIBLE_CACHE", None)
    else:
        os.environ["CRUCIBLE_CACHE"] = old


@pytest.fixture(scope="module")
def val_tasks(cache):
    return T.build_tasks([("scm_additive", T.VAL_SEED0), ("scm_additive", T.VAL_SEED0 + 1), ("scm_delay", T.VAL_SEED0),
                          ("tech_pipeline", T.VAL_SEED0)])


def test_every_family_fits_the_env_spec_and_partitions_are_disjoint():
    fams = T.families_by(seats=None)
    assert len(fams) >= 20
    for f in fams:
        t = T.make_task(f, 0, validate=False)
        eng = t.sim(money=False).engine
        assert len(eng.levers(0)) <= SPEC.max_levers and len(eng.gauges(0)) <= SPEC.max_gauges, f
    train, val, test = (set(T.families_by(s, seats=None)) for s in ("train", "val", "test"))
    assert train and val and test and not (train & val) and not (train & test) and not (val & test)
    assert {T.MECH_FAMILIES[f].engine for f in train} == {"scm", "abm"}          # techsys is never trained on
    assert all(T.MECH_FAMILIES[f].transfer == "cross_impl" for f in fams if T.MECH_FAMILIES[f].engine == "techsys")


def test_tasks_are_deterministic_cached_and_have_distinct_world_seeds(val_tasks, cache):
    a = val_tasks[0]
    b = T.make_task(a.family, a.seed, validate=True)
    assert a == b                                                   # rebuilt from scratch: identical
    assert T.build_tasks([(a.family, a.seed)])[0] == a              # and from the cache
    assert len({t.world_seed for t in val_tasks}) == len(val_tasks)
    assert a.validated and a.reference_net > a.default_net and a.goal_net_cents is not None
    assert a.endowment == T.ENDOWMENT_TICKS * a.income
    json.dumps(a.to_dict())
    assert T.Task.from_dict(json.loads(json.dumps(a.to_dict()))) == a


def test_public_view_of_a_task_contains_no_private_number(val_tasks):
    for t in val_tasks:
        pub = json.dumps(t.public())
        for secret in (t.reference_net, t.default_net, t.random_net):
            assert f"{secret:.3f}"[:8] not in pub
        assert set(t.public()) == {"task", "horizon_ticks", "starting_balance_cents", "objective", "goal_net_cents"}
        assert str(t.income) not in pub                              # the facility's demand income is not public


def test_normalisation_hold_scores_zero_random_scores_below_and_reference_scores_about_one(val_tasks):
    env = MechVecEnv(val_tasks, SPEC, reward_mode=LEGACY)
    hold = run_episodes(env, HoldLearner(SPEC))
    rand = run_episodes(env, RandomLearner(SPEC, seed=1))
    for t, h, r in zip(val_tasks, hold, rand):
        # the private default was estimated on other noise seeds: the held episode is near 0, not exactly 0
        assert abs(h["score"]) < 0.35, (t.id, h["score"])
        assert r["score"] < h["score"] + 0.05 and r["success"] is False
        assert h["ticks_fallback"] == 0 and h["ticks_idle"] == 0
    # the reference settings, replayed as a constant action, reach the objective
    from crucible.mechanisms.validity import check_mechanism
    t = val_tasks[0]
    ref = check_mechanism(t.mechanism())["economics"][0]["reference"][0]["settings"]
    sim = t.sim()
    out = sim.run(ref, t.horizon)
    net = float(out["net"][:, 0].mean())
    assert t.score(net) > 0.8 and t.success(float(out["net"][:, 0].sum()))


def test_observation_layout_masks_and_reward_units(val_tasks):
    env = MechVecEnv(val_tasks[:2], SPEC, reward_mode=LEGACY)
    obs = env.reset()
    assert obs.shape == (2, SPEC.obs_dim) and obs.dtype == np.float32
    eng = env.sims[0].engine
    assert obs[0, SPEC.lever_mask].sum() == len(eng.levers(0)) and obs[0, SPEC.gauge_mask].sum() == len(eng.gauges(0))
    assert np.all(obs[:, SPEC.gauges] == 0) and obs[0, SPEC.scalars][3] == 1.0 and obs[0, SPEC.scalars][4] == 0.0
    act = np.zeros((2, SPEC.act_dim), np.float32)
    nxt, rew, done, info = env.step(act)
    sc = nxt[0, SPEC.scalars]
    assert rew[0] == pytest.approx(info["net_cents"][0] / money_scale(val_tasks[0]), rel=1e-6)
    assert sc[0] == pytest.approx(rew[0]) and sc[1] - sc[2] == pytest.approx(rew[0], abs=1e-5)
    assert sc[4] == pytest.approx(1 / val_tasks[0].horizon)
    # masked action entries are ignored: garbage there changes nothing
    env2 = MechVecEnv(val_tasks[:2], SPEC, reward_mode=LEGACY)
    env2.reset()
    junk = act.copy()
    junk[:, len(eng.levers(0)):] = 0.77
    k = min(len(e.sims[i].engine.levers(0)) for e in (env2,) for i in range(2))
    junk[:, :k] = 0.0
    n2, r2, _, _ = env2.step(junk)
    assert r2[0] == rew[0]
    with pytest.raises(ValueError):
        env.step(np.zeros((2, 3)))


def test_replaying_a_task_gives_the_same_episode_and_new_tasks_can_be_swapped_in(val_tasks):
    env = MechVecEnv(val_tasks[:2], SPEC, reward_mode=LEGACY)
    a = run_episodes(env, RandomLearner(SPEC, seed=5))
    b = run_episodes(env, RandomLearner(SPEC, seed=5))
    assert [x["net_cents"] for x in a] == [x["net_cents"] for x in b]
    obs = env.reset(which=np.array([True, False]), tasks=[val_tasks[2]])
    assert env.tasks[0].id == val_tasks[2].id and not env.done[0] and env.done[1]
    assert obs[0, SPEC.lever_mask].sum() == len(env.sims[0].engine.levers(0))


def test_an_operator_who_overspends_runs_out_of_money_and_the_record_says_so(val_tasks):
    t = val_tasks[0]
    env = MechVecEnv([t], SPEC, reward_mode=LEGACY)

    class MaxEverything(HoldLearner):
        def act(self, obs, explore=True):
            return np.ones((obs.shape[0], self.spec.act_dim), np.float32)
    rec = run_episodes(env, MaxEverything(SPEC))[0]
    assert rec["net_cents"] < 0 and rec["score"] < 0
    assert rec["funds_end_cents"] >= 0                               # the ledger never lets the balance go negative
    assert rec["funds_end_cents"] == t.endowment + rec["net_cents"]


@pytest.mark.parametrize("which", [0, 2, 3])
def test_the_kernel_world_and_the_simulator_are_the_same_problem(val_tasks, which):
    """One learner, the same actions: observations and rewards from `rig.*` tool calls in a kernel world equal the
    batched simulator's, tick for tick (float32 observations compared exactly)."""
    t = val_tasks[which]
    ticks = 30
    sim_env = MechVecEnv([t], SPEC, reward_mode=LEGACY)
    ker_env = KernelMechEnv(t, SPEC, reward_mode=LEGACY)
    la, lb = RandomLearner(SPEC, seed=9, hold=3), RandomLearner(SPEC, seed=9, hold=3)
    oa, ob = sim_env.reset(), ker_env.reset()
    la.reset_context(1)
    lb.reset_context(1)
    assert np.array_equal(oa, ob)
    for k in range(ticks):
        a, b = la.act(oa), lb.act(ob)
        assert np.array_equal(a, b)
        oa, ra, _, ia = sim_env.step(a)
        ob, rb, _, ib = ker_env.step(b)
        assert np.array_equal(oa, ob), (k, np.flatnonzero(oa != ob))
        assert ra[0] == rb[0] and ia["net_cents"][0] == ib["net_cents"][0]
    # the kernel run is an ordinary recorded agent trajectory: tool calls, and a log that replays
    assert sum(1 for c in ker_env.calls if c["tool"] == "rig.set") == ticks
    from crucible.core.kernel import World
    w = ker_env.world
    assert w.check_invariants() == []
    assert World.replay(w.spec, [e.to_dict() for e in w.log]).digest() == w.digest()
