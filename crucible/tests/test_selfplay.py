"""Self-play on hidden mechanisms: per-seat observations with no leak between seats, the kernel-hosted twin equal
to the arena tick for tick, individuals that own their weights, buffers, state and budget, objectives, the
persistent league (archive, held-out separation, ratings, matchmaking), the six interaction modes, the collusion
and reward-inflation audit, and short real training runs in which an adaptive learner's policy depends on its
opponent."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from crucible.core.kernel import World
from crucible.mechanisms import base as B
from crucible.research import tasks as T
from crucible.research.baselines import HoldLearner, RandomLearner
from crucible.research.envs import MechVecEnv
from crucible.research.learner import EnvSpec
from crucible.selfplay import audit as AU
from crucible.selfplay import modes as M
from crucible.selfplay import profiles as P
from crucible.selfplay.arena import (Individual, KernelSelfPlayEnv, Objective, SelfPlayEnv, arena_task, play,
                                     summarise, with_noise)
from crucible.selfplay.league import SelfPlayLeague
from crucible.selfplay.learners import (ClimberLearner, FixedLearner, PGLearner, SPSALearner, learner_config,
                                        make_learner)

LEGACY = "legacy_shaped_reward"   # 4.0 baseline: the shaped reward is selected explicitly (Clarification 2, OO.8)
SPEC = EnvSpec()


# ====================================================================== a two-seat engine with no coupling at all
@B.register_engine
class _Iso(B.Engine):
    """Two seats that have nothing to do with each other: whatever one does, the other's readings and money are
    the same. Seat 1 has secrets (large distinctive numbers) that must never show up in seat 0's observation."""
    engine = "selfplay_iso_test"

    def levers(self, seat):
        return [B.Lever("a0", 0.0, 10.0, 2.0), B.Lever("b0", -1.0, 1.0, 0.0)] if seat == 0 else \
            [B.Lever("a1", 0.0, 100.0, 50.0), B.Lever("b1", 0.0, 1.0, 0.5), B.Lever("c1", 0.0, 5.0, 1.0)]

    def gauges(self, seat):
        return [B.Gauge("g0"), B.Gauge("h0")] if seat == 0 else [B.Gauge("g1"), B.Gauge("h1"), B.Gauge("i1")]

    def brief(self, seat):
        return "test"

    def init_state(self, rng):
        return {"x": np.zeros(2)}

    def step(self, state, settings, rng, t):
        z = rng.standard_normal(2)
        a0, b0 = settings[0]["a0"], settings[0]["b0"]
        a1, b1, c1 = settings[1]["a1"], settings[1]["b1"], settings[1]["c1"]
        x = state["x"] * 0.5 + np.array([a0, a1])
        readings = [{"g0": float(x[0] + z[0]), "h0": float(b0 * 3.0)},
                    {"g1": float(7777.0 + x[1] + z[1]), "h1": float(8888.0 * b1), "i1": float(9999.0 + c1)}]
        return B.StepResult({"x": x}, readings, [40.0 * a0, 3.0 * a1 + 600.0 * c1], [2.0 * a0 * a0, 31.0 + a1],
                            [a0, a1], {"secret": 4242.4242})

    def truth(self):
        return {"engine": self.engine, "family": None, "nodes": [], "edges": [], "regimes": [], "motifs": [], "notes": ""}


@pytest.fixture()
def iso_task():
    mech = {"engine": "selfplay_iso_test", "family": "selfplay_iso_test", "seed": 0,
            "config": {"n_seats": 2, "horizon": 24, "income_cents_per_tick": 5000}}
    T._MECHS[("selfplay_iso_test", 0)] = mech
    yield T.Task("selfplay_iso_test", 0, 0, 77, 5000, 100_000, 24, 2, engine="selfplay_iso_test")
    T._MECHS.pop(("selfplay_iso_test", 0), None)


def _run(env, acts0, acts1, funds=None):
    obs = env.reset(funds=funds)
    out0, out1, rew = [obs[0].copy()], [obs[1].copy()], []
    for a0, a1 in zip(acts0, acts1):
        obs, r, done, info = env.step([a0, a1])
        out0.append(obs[0].copy())
        out1.append(obs[1].copy())
        rew.append(r.copy())
    return np.array(out0), np.array(out1), np.array(rew)


# ====================================================================== 1. the arena
def test_each_seat_sees_only_its_own_view_and_nothing_of_the_other_leaks(iso_task):
    env = SelfPlayEnv([iso_task], SPEC, reward_mode=LEGACY)
    rng = np.random.default_rng(0)
    n = iso_task.horizon
    a0 = rng.uniform(-1, 1, (n, 1, SPEC.act_dim)).astype(np.float32)
    a1 = rng.uniform(-1, 1, (n, 1, SPEC.act_dim)).astype(np.float32)
    a1_other = rng.uniform(-1, 1, (n, 1, SPEC.act_dim)).astype(np.float32)
    o0, o1, r = _run(env, a0, a1)
    # the other seat acts completely differently and is twice as rich: seat 0's stream is bit-identical
    p0, p1, q = _run(env, a0, a1_other, funds=[[100_000, 200_000]])
    assert np.array_equal(o0, p0) and np.array_equal(r[:, 0], q[:, 0])
    assert not np.array_equal(o1, p1) and not np.array_equal(r[:, 1], q[:, 1])
    # the layout: own gauges and masks, own settings, own money, the clock
    assert o0.shape == (n + 1, 1, SPEC.obs_dim) and o0.dtype == np.float32
    assert o0[5, 0, SPEC.gauge_mask].sum() == 2 and o1[5, 0, SPEC.gauge_mask].sum() == 3
    assert o0[5, 0, SPEC.lever_mask].sum() == 2 and o1[5, 0, SPEC.lever_mask].sum() == 3
    assert np.allclose(o0[5, 0, SPEC.last_action][:2], np.clip(a0[4, 0, :2], -1, 1), atol=1e-6)
    assert np.allclose(o1[5, 0, SPEC.last_action][:3], np.clip(a1[4, 0, :3], -1, 1), atol=1e-6)
    sim = env.sims[0]
    assert o0[-1, 0, SPEC.scalars][3] == np.float32(sim.funds[0] / iso_task.endowment)
    assert o0[-1, 0, SPEC.scalars][4] == 1.0
    # seat 1's secrets (readings near 7777, 8888 x, 9999; the engine's private 4242.4242) are nowhere in seat 0's
    big = np.log1p(4000.0)                                   # symlog of anything that large
    assert o0[:, 0, SPEC.gauges].max() < big and o1[1:, 0, SPEC.gauges].max() > big
    # reward: own net in the public money unit; per-seat records add up; private diagnostics stay operator-side
    rec = env.drain()[0]
    assert rec["seats"][0]["net_cents"] == pytest.approx(float(r[:, 0, 0].sum()) * (iso_task.endowment / 20), rel=1e-5)
    assert rec["private"]["secret"] == pytest.approx(4242.4242 * n)
    assert rec["joint_net_per_tick"] == pytest.approx(sum(s["net_per_tick"] for s in rec["seats"]))
    with pytest.raises(ValueError):
        env.step([a0[0]])                                     # one action array per seat
    with pytest.raises(ValueError):
        SelfPlayEnv([iso_task, arena_task("scm_additive", 0)], SPEC, reward_mode=LEGACY)   # mixed numbers of seats


def test_seat_view_matches_the_single_agent_environment_layout():
    """A seat's observation in the arena has the layout of `research.envs.MechVecEnv` for that seat: at the start
    of a match (nothing has happened, both hold the defaults) the two are equal."""
    for fam in ("tech_handoff", "abm_duopoly"):
        t = arena_task(fam, 0)
        arena = SelfPlayEnv([t], SPEC, reward_mode=LEGACY)
        obs = arena.reset()
        for seat in (0, 1):
            single = MechVecEnv([t], SPEC, seat=seat, reward_mode=LEGACY)
            assert np.array_equal(obs[seat], single.reset())
        hold = [np.asarray(o[:, SPEC.last_action]) for o in obs]
        o2, r, _, info = arena.step(hold)
        sim = t.sim(held=[True, True], funds=[t.endowment] * 2)
        res = sim.step(None)
        for seat in (0, 1):
            assert info["net_cents"][seat, 0] == res.revenue[seat] - res.cost[seat]
            names = [g.name for g in sim.engine.gauges(seat)]
            want = np.sign([res.readings[seat][g] for g in names]) * np.log1p(np.abs([res.readings[seat][g] for g in names]))
            assert np.allclose(o2[seat][0, SPEC.gauges][:len(names)], want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fam,noise", [("tech_handoff", 0), ("abm_duopoly", 3), ("tech_shared_pool", 1)])
def test_kernel_hosted_match_equals_the_arena_tick_for_tick(fam, noise):
    """Two admitted participants operating their seats with rig.set / rig.read in a kernel world get exactly the
    observations and rewards of the kernel-free arena, for both seats, at every tick."""
    t = with_noise(arena_task(fam, 0), noise)
    ticks = 30
    arena, kernel = SelfPlayEnv([t], SPEC, reward_mode=LEGACY), KernelSelfPlayEnv(t, SPEC, handles=("ana", "ben"), reward_mode=LEGACY)
    la = [RandomLearner(SPEC, seed=5 + s, hold=3) for s in range(2)]
    lb = [RandomLearner(SPEC, seed=5 + s, hold=3) for s in range(2)]
    oa, ob = arena.reset(), kernel.reset()
    for s in range(2):
        assert np.array_equal(oa[s], ob[s])
    for k in range(ticks):
        a = [l.act(o) for l, o in zip(la, oa)]
        b = [l.act(o) for l, o in zip(lb, ob)]
        oa, ra, _, ia = arena.step(a)
        ob, rb, _, ib = kernel.step(b)
        for s in range(2):
            assert np.array_equal(oa[s], ob[s]), (fam, k, s, np.flatnonzero(oa[s] != ob[s]))
        assert np.array_equal(ra, rb) and np.array_equal(ia["net_cents"], ib["net_cents"])
        assert [x[0] for x in ia["status"]] == [x[0] for x in ib["status"]]
    w = kernel.world
    assert w.check_invariants() == []
    # recorded actions in a shared world: one rig.set per seat and tick, a log that replays, the metric rig_net
    for s in range(2):
        assert sum(1 for c in kernel.calls[s] if c["tool"] == "rig.set") == ticks
        assert kernel.rig_net(s) == float(np.sum([row["net"][s] for row in _trace(t, la, ticks)]))
        assert kernel.balance(s) == arena.sims[0].funds[s]
    assert World.replay(w.spec, [e.to_dict() for e in w.log]).digest() == w.digest()
    assert AU.ledger_conservation(w)["ok"]


def _trace(task, learners, ticks):
    env = SelfPlayEnv([task], SPEC, trace=True, reward_mode=LEGACY)
    ls = [RandomLearner(SPEC, seed=l.seed, hold=l.hold) for l in learners]
    obs = env.reset()
    for _ in range(ticks):
        obs, *_ = env.step([l.act(o) for l, o in zip(ls, obs)])
    return env.trace[0]


# ====================================================================== 2. individuals
class _Spy(FixedLearner):
    """Plays one constant settings vector and writes down everything it is given."""
    name, kind = "spy", "scripted"

    def __init__(self, spec, level):
        super().__init__(spec, action=np.full(spec.act_dim, level, dtype=np.float32))
        self.seen, self.rewards, self.resets = [], [], 0

    def reset_context(self, n, which=None):
        self.resets += 1

    def observe(self, obs, action, reward, done, next_obs):
        self.seen.append((obs.copy(), next_obs.copy()))
        self.rewards.append(np.asarray(reward).copy())


def test_an_individual_is_given_its_own_observation_and_its_own_objective_only():
    t = arena_task("tech_handoff", 0)
    env = SelfPlayEnv([t, t], SPEC, trace=True, reward_mode=LEGACY)
    own = Individual("own", _Spy(SPEC, 0.3), "own", seat=0)
    joint = Individual("joint", _Spy(SPEC, -0.3), "joint", seat=1)
    recs = play(env, [own, joint], noise=[4, 5])
    assert own.learner.resets == joint.learner.resets == 1 and len(own.learner.seen) == t.horizon
    # replay the same two worlds by hand to know what each seat's observations were
    ref = SelfPlayEnv([with_noise(t, 4), with_noise(t, 5)], SPEC, reward_mode=LEGACY)
    obs = ref.reset()
    cent = 1.0 / ref._scale[0]
    for k in range(t.horizon):
        nxt, rew, _, info = ref.step([own.learner.act(obs[0]), joint.learner.act(obs[1])])
        for seat, ind in ((0, own), (1, joint)):
            got_obs, got_next = ind.learner.seen[k]
            assert np.array_equal(got_obs, obs[seat]) and np.array_equal(got_next, nxt[seat])
        assert np.array_equal(own.learner.rewards[k], rew[0])                       # its own net
        assert np.allclose(joint.learner.rewards[k], rew.mean(axis=0), atol=1e-7)   # the team mean, by its objective
        # in this family the mechanism's own score is the joint outcome (up to the rounding of payments to cents)
        assert np.allclose(Objective("score").reward(1, rew, (info["score"] / ref._scale[0]).astype(np.float32)),
                           rew.mean(axis=0), atol=cent)
        obs = nxt
    assert abs(np.sum(own.learner.rewards) - np.sum(joint.learner.rewards)) > 100 * cent    # own net is not the joint one
    assert Objective("mixed", 0.5).reward(0, np.array([[2.0], [4.0]])) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        Objective("selfless")
    # experience history, identity and accounting are per individual
    assert own.history[-1]["with"] == ["joint"] and joint.history[-1]["with"] == ["own"]
    assert own.matches == 2 and own.env_steps == 2 * t.horizon == own.learner.usage().env_steps
    assert own.lifetime_net_cents == pytest.approx(sum(r["seats"][0]["net_cents"] for r in recs))
    with pytest.raises(ValueError):
        play(env, [joint, own])                                                    # bound to the other seats
    with pytest.raises(ValueError):
        play(env, [own, own])


def test_individuals_share_no_weights_buffers_or_state():
    t = arena_task("tech_handoff", 0)
    env = SelfPlayEnv([t] * 4, SPEC, reward_mode=LEGACY)
    for algo, cls in (("pg", PGLearner), ("spsa", SPSALearner)):
        cfg = learner_config(algo, 3)
        a = Individual("a", make_learner(SPEC, **cfg), "own", seat=0, config=cfg)
        b = Individual("b", make_learner(SPEC, **cfg), "own", seat=1, config=cfg)      # same recipe, same seed
        assert isinstance(a.learner, cls) and a.learner is not b.learner
        if algo == "pg":
            pa = list(a.learner.pi.parameters()) + list(a.learner.vf.parameters()) + [a.learner.log_std]
            pb = list(b.learner.pi.parameters()) + list(b.learner.vf.parameters()) + [b.learner.log_std]
            assert not {p.data_ptr() for p in pa} & {p.data_ptr() for p in pb}
            assert a.learner.opt_pi is not b.learner.opt_pi and a.learner.gen is not b.learner.gen
            before = b.learner.parameters_flat().copy()
        for it in range(2):
            obs = env.reset(tasks=[with_noise(t, 10 * it + i) for i in range(4)])
            a.learner.reset_context(4)
            b.learner.reset_context(4)
            seen_a = []
            while not env.done.all():
                act = [a.learner.act(obs[0]), b.learner.act(obs[1])]
                nxt, rew, done, _ = env.step(act)
                a.learner.observe(obs[0], act[0], rew[0], done, nxt[0])               # only a learns
                seen_a.append(obs[0])
                obs = nxt
            if algo == "pg":
                # the rollout buffer holds a's own decisions only: one entry per decision of its own episode
                assert len(a.learner.buffer) == -(-t.horizon // a.learner.hp["hold"]) and b.learner.buffer == []
                assert all(e["z"].shape == (4, 2 * SPEC.obs_dim) for e in a.learner.buffer)
            assert a.learner.update()["updates"] >= 1 and b.learner.update()["updates"] == 0
        if algo == "pg":
            assert a.learner.buffer == []                                           # on-policy: discarded after use
            assert np.array_equal(b.learner.parameters_flat(), before)             # b never learned: untouched
            assert not np.array_equal(a.learner.parameters_flat(), before)
            assert not np.shares_memory(a.learner.o_mean, b.learner.o_mean)
        else:
            assert not np.shares_memory(a.learner.theta, b.learner.theta)
            assert a.learner.k == 2 and b.learner.k == 0
            assert not np.array_equal(a.learner.theta, b.learner.theta)             # b still at its seat's defaults


@pytest.mark.parametrize("algo", ["spsa", "pg", "climber"])
def test_checkpoints_restore_an_individual_and_training_resumes_exactly(algo, tmp_path):
    t = arena_task("tech_shared_pool", 0)
    env = SelfPlayEnv([t] * 4, SPEC, reward_mode=LEGACY)

    def fresh():
        ca, cb = learner_config(algo, 1), learner_config(algo, 2)
        return [Individual("a", make_learner(SPEC, **ca), "own", seat=0, config=ca, tags=("x",)),
                Individual("b", make_learner(SPEC, **cb), "own", seat=1, config=cb, parent="root", generation=3)]

    one = fresh()
    for it in range(4):
        play(env, one, noise=100 + 4 * it)
    two = fresh()
    for it in range(2):
        play(env, two, noise=100 + 4 * it)
    for ind in two:
        ind.save(str(tmp_path / ind.id))
    back = [Individual.load(str(tmp_path / i), SPEC) for i in ("a", "b")]
    assert [(i.id, i.seat, i.parent, i.generation, i.tags, i.objective.kind, i.matches) for i in back] == \
        [("a", 0, None, 0, ("x",), "own", 8), ("b", 1, "root", 3, (), "own", 8)]
    assert back[0].history == two[0].history and back[0].learner is not two[0].learner
    for it in range(2, 4):
        play(env, back, noise=100 + 4 * it)
    ev1 = play(env, one, noise=900, learn=False)
    ev2 = play(env, back, noise=900, learn=False)
    if algo != "climber":                         # the scripted climber does not checkpoint its random stream
        assert [r["seats"] for r in ev1] == [r["seats"] for r in ev2]
        if algo == "spsa":
            assert np.array_equal(one[0].learner.theta, back[0].learner.theta)
        else:
            assert np.array_equal(one[0].learner.parameters_flat(), back[0].learner.parameters_flat())
    # frozen play is repeatable and changes nothing
    snap = [json.dumps(i.meta()["history"]) for i in one]
    ev3 = play(env, one, noise=900, learn=False)
    if algo != "climber":
        assert [r["seats"] for r in ev3] == [r["seats"] for r in ev1]
    one[0].learner.freeze(True)
    assert play(env, one, noise=901)[0]["learned"] == [False, True]                # a frozen learner never learns
    assert len(snap) == 2


def test_budgets_are_per_individual_and_an_individual_that_goes_broke_idles():
    t = arena_task("tech_shared_pool", 0)
    env = SelfPlayEnv([t, t], SPEC, reward_mode=LEGACY)
    hi = np.ones(SPEC.act_dim, dtype=np.float32)                                    # everything at maximum: costly
    burner = Individual("burner", FixedLearner(SPEC, action=hi), "own", seat=0, bankroll=int(0.8 * t.endowment))
    saver = Individual("saver", HoldLearner(SPEC), "own", seat=1, funds_scale=0.5)
    assert burner.stake(t, 2) == int(0.8 * t.endowment) // 2 and saver.stake(t, 2) == t.endowment // 2
    rows = []
    for k in range(4):
        recs = play(env, [burner, saver], noise=2 * k, learn=False)
        rows.append((burner.bankroll, sum(r["seats"][0]["ticks_idle"] + r["seats"][0]["ticks_fallback"] for r in recs),
                     [r["seats"][0]["funds_start"] for r in recs], [r["seats"][1]["funds_start"] for r in recs]))
        assert all(r["seats"][0]["funds_end"] >= 0 for r in recs)                   # never overdrawn
    bank = [r[0] for r in rows]
    assert bank[0] < 0.8 * t.endowment and bank == sorted(bank, reverse=True) and bank[-1] >= 0
    assert rows[-1][2][0] == bank[-2] // 2                                          # the stake is what is left
    assert rows[-1][1] > 0                                                          # out of money: fallback or idle
    assert all(r[3] == [t.endowment // 2] * 2 for r in rows)                        # the other's funding is its own
    assert saver.bankroll is None and burner.history[-1]["idle_ticks"] >= 0
    # with nothing at all it produces nothing and is charged nothing
    broke = Individual("broke", FixedLearner(SPEC, action=hi), "own", seat=0, bankroll=0)
    recs = play(env, [broke, saver], noise=50, learn=False)
    assert all(r["seats"][0]["ticks_idle"] + r["seats"][0]["ticks_fallback"] == t.horizon for r in recs)
    assert broke.bankroll == pytest.approx(sum(r["seats"][0]["net_cents"] for r in recs)) and broke.bankroll >= 0


# ====================================================================== 3. learners adapt, and to their opponent
def test_independent_learners_improve_a_joint_outcome_neither_reaches_alone():
    """tech_handoff, both algorithms, a short real run: the joint net of the trained pair is above the defaults
    and above what either trained member reaches with a partner that leaves its half alone."""
    t = arena_task("tech_handoff", 0)
    for algos in (("spsa", "spsa"), ("pg", "pg")):
        env = SelfPlayEnv([t] * 6, SPEC, reward_mode=LEGACY)
        inds = [Individual(f"{a}{s}", make_learner(SPEC, a, 10 + s), "joint", seat=s) for s, a in enumerate(algos)]
        for it in range(14):
            play(env, inds, noise=1000 + 6 * it)
        pair = M.evaluate(t, inds, n_slots=6, reward_mode=LEGACY)
        base = M.evaluate(t, [None, None], n_slots=6, reward_mode=LEGACY)
        solo = [M.evaluate(t, [inds[0], None], n_slots=6, reward_mode=LEGACY), M.evaluate(t, [None, inds[1]], n_slots=6, reward_mode=LEGACY)]
        assert pair["joint_net_per_tick"] > base["joint_net_per_tick"] + 30, (algos, pair["joint_net_per_tick"])
        assert pair["joint_net_per_tick"] > max(s["joint_net_per_tick"] for s in solo), algos


def test_what_a_learner_learns_depends_on_the_opponent_it_was_trained_against():
    """tech_shared_pool is a game of chicken over a shared pool. The same learner (same algorithm, same seed) is
    trained against a partner that takes the large slice of the pool and against one that shows restraint. The two
    copies learn different settings of the pool-hungry lever, and each does better against its own training
    opponent than the other copy does (the 2x2 that shows adaptation to the opponent, not just learning)."""
    t = arena_task("tech_shared_pool", 0)
    cache = next(lv["name"] for lv in t.mechanism()["config"]["levers"] if lv["kind"] == "cache" and lv["seat"] == 0)
    j = [lv.name for lv in M._env(t, 6).sims[0].engine.levers(0)].index(cache)
    trained, net = {}, {}
    for opp in ("grab", "restraint"):
        env = SelfPlayEnv([t] * 6, SPEC, reward_mode=LEGACY)
        me = Individual(f"vs_{opp}", make_learner(SPEC, "spsa", 5), "own", seat=0)
        other = P.profile_individual(t, 1, opp, SPEC)
        for it in range(24):
            play(env, [me, other], noise=2000 + 6 * it)
        trained[opp] = me
    for tr, me in trained.items():
        for opp in ("grab", "restraint"):
            net[(tr, opp)] = M.evaluate(t, [me, P.profile_individual(t, 1, opp, SPEC)], n_slots=6, reward_mode=LEGACY)["net_per_tick"][0]
    th = {k: v.learner.theta[j] for k, v in trained.items()}
    assert th["restraint"] > th["grab"] + 0.1, th               # more cache when the partner leaves room for it
    own = net[("grab", "grab")] + net[("restraint", "restraint")]
    cross = net[("restraint", "grab")] + net[("grab", "restraint")]
    assert net[("grab", "grab")] > net[("restraint", "grab")], net
    assert own > cross, net


def test_learner_contract_details():
    t = arena_task("tech_handoff", 0)
    env = SelfPlayEnv([t] * 3, SPEC, reward_mode=LEGACY)
    obs = env.reset()
    sp = SPSALearner(SPEC, seed=1)
    sp.reset_context(3)
    a = sp.act(obs[0])
    # theta starts at the public defaults; slots 0 and 1 are the two signs of one direction, slot 2 plays theta
    assert np.allclose(sp.theta[sp.mask], obs[0][0, SPEC.last_action][sp.mask])
    c = sp.gains()[1]
    d0, d1 = a[0] - sp.theta, a[1] - sp.theta
    inside = sp.mask & (np.abs(a[0]) < 1) & (np.abs(a[1]) < 1)                    # not cut at the edge of a range
    assert inside.sum() >= 2 and np.allclose(d0[inside], -d1[inside], atol=1e-6)
    assert np.allclose(np.abs(d0[inside]), c, atol=1e-6)
    assert np.allclose(a[2], sp.theta) and np.all(a[:, ~sp.mask] == 0)
    assert np.allclose(sp.act(obs[0], explore=False), sp.theta)
    sp.freeze(True)
    assert np.allclose(sp.act(obs[0]), sp.theta) and sp.update()["updates"] == 0
    a0, c0 = SPSALearner(SPEC).gains()
    late = SPSALearner(SPEC)
    late.k = 500
    assert late.gains()[0] < a0 and late.gains()[1] < c0
    # the neural policy starts as "hold the defaults" and is deterministic without exploration
    pg = PGLearner(SPEC, seed=2)
    pg.reset_context(3)
    assert np.allclose(pg.act(obs[1], explore=False), obs[1][:, SPEC.last_action], atol=1e-6)
    pg.freeze(True)
    pg.reset_context(3)
    pg.act(obs[1])
    pg.observe(obs[1], a, np.zeros(3, dtype=np.float32), np.ones(3, bool), obs[1])
    assert pg.buffer == [] and pg.update()["updates"] == 0
    # the scripted climber reads its own net from its observation, so it adapts with or without `observe`
    cl = ClimberLearner(SPEC, seed=3, block=4, persist=False)
    env1 = SelfPlayEnv([t], SPEC, reward_mode=LEGACY)
    o = env1.reset()
    cl.reset_context(1)
    acts = []
    while not env1.done.all():
        x = cl.act(o[0])
        acts.append(x.copy())
        o, *_ = env1.step([x, None])
    assert len({tuple(np.round(x[0], 4)) for x in acts}) > 5
    with pytest.raises(ValueError):
        make_learner(SPEC, "no_such_learner")


def test_registry_learners_are_individuals_like_any_other(tmp_path):
    """Learners of the `research.learners` registry (written elsewhere) plug into the arena through the learner
    contract alone: a recurrent PPO individual trains against a scripted experimenter, each on its own
    observations, and its checkpoint restores."""
    reg = pytest.importorskip("crucible.research.learners")
    need = {"ppo_gru", "hill_climb", "probe_exploit"}
    if not need <= set(reg.available()):
        pytest.skip(f"registry does not have {sorted(need - set(reg.available()))} yet")
    t = arena_task("tech_handoff", 0)
    env = SelfPlayEnv([t, t], SPEC, reward_mode=LEGACY)
    ca = learner_config("ppo_gru", 1, rollout_steps=2 * t.horizon)
    a = Individual("ppo", make_learner(SPEC, **ca), "joint", seat=0, config=ca)
    b = Individual("climb", make_learner(SPEC, "hill_climb", 2), "own", seat=1, config=learner_config("hill_climb", 2))
    for it in range(2):
        recs = play(env, [a, b], noise=2 * it)
    assert a.arch == "ppo_gru" and a.learner.usage().updates >= 1 and a.history[-1]["updates"] >= 1
    assert any(abs(x) > 1e-3 for r in recs for x in np.array(r["seats"][1]["mean_action"])
               - np.array(M.evaluate(t, [None, None], n_slots=2, reward_mode=LEGACY)["mean_action"][1]))        # the climber moved levers
    a.save(str(tmp_path / "ppo"))
    back = Individual.load(str(tmp_path / "ppo"), SPEC)
    probe = Individual("probe", make_learner(SPEC, "probe_exploit", 3), "own", seat=1)
    e1 = M.evaluate(t, [a, probe], n_slots=2, reward_mode=LEGACY)
    e2 = M.evaluate(t, [back, probe], n_slots=2, reward_mode=LEGACY)
    assert e1["net_per_tick"] == e2["net_per_tick"] and back.learner is not a.learner


# ====================================================================== 4. the league
@pytest.fixture(scope="module")
def league(tmp_path_factory):
    t = arena_task("abm_duopoly", 1)
    lg = SelfPlayLeague(str(tmp_path_factory.mktemp("league")), t, SPEC, "competition", n_slots=2, reward_mode=LEGACY)
    lg.new("a", "spsa", 1)
    lg.new("b", "pg", 2)
    M.train(lg, ["a", "b"], 2, eval_every=0, snapshot_every=1)
    return lg


def test_league_archive_holds_frozen_copies_with_their_own_weights(league):
    lg = league
    assert lg.archive() == ["a@0", "b@0", "a@1", "b@1", "a@2", "b@2"] and lg.archive(of="a") == ["a@0", "a@1", "a@2"]
    assert lg.lineage("a@2") == ["a@2", "a"] and lg.entries["a@2"].generation == 1
    assert lg.info["a@1"] == {**lg.info["a@1"], "role": "snapshot", "stage": 1, "arch": "spsa"}
    s0, s2, live = lg.get("a@0"), lg.get("a@2"), lg.get("a")
    assert s0.learner.frozen and s2.learner.frozen and not live.learner.frozen and s2.learner is not live.learner
    assert np.array_equal(s2.learner.theta, live.learner.theta) and not np.array_equal(s0.learner.theta, live.learner.theta)
    theta2 = s2.learner.theta.copy()
    b2 = lg.get("b@2").learner.parameters_flat().copy()
    lg.train_match(["a", "b"], 5000)                              # the live ones move on, the archive does not
    assert not np.array_equal(live.learner.theta, theta2) and np.array_equal(s2.learner.theta, theta2)
    assert np.array_equal(lg.get("b@2").learner.parameters_flat(), b2)
    assert not np.array_equal(lg.get("b").learner.parameters_flat(), b2)
    # an archived stage never learns, whatever a match asks for
    rec = lg.play(["a@2", "b"], 5001, learn=True)
    assert rec.outcomes["a@2"]["learned"] is False and np.array_equal(s2.learner.theta, theta2)
    with pytest.raises(ValueError):
        lg.snapshot("a", 2)


def test_league_ratings_update_from_rated_matches_only(league):
    lg = league
    before = {p: (lg.entries[p].elo, lg.entries[p].mu, lg.entries[p].sigma, lg.entries[p].games) for p in ("a@0", "a@2")}
    rec = lg.play(["a@0", "a@2"], 6000, rated=False, tag="eval")
    assert {p: (lg.entries[p].elo, lg.entries[p].mu, lg.entries[p].sigma, lg.entries[p].games) for p in before} == before
    rec = lg.play(["a@0", "a@2"], 6000)
    (x, y, s), = rec.results
    win, lose = (x, y) if s == 1.0 else (y, x)
    assert s in (0.0, 1.0) and rec.scores[win] > rec.scores[lose]
    assert lg.entries[win].elo > before[win][0] and lg.entries[lose].elo < before[lose][0]
    assert lg.entries[win].elo + lg.entries[lose].elo == pytest.approx(before[win][0] + before[lose][0])
    assert lg.entries[win].mu > before[win][1] and lg.entries[win].sigma < before[win][2]
    assert lg.entries[win].games == before[win][3] + 1
    # scores are nets from independent demand; the record keeps who sat where and the private quantities
    assert rec.scores == {p: rec.outcomes[p]["net_per_tick"] for p in rec.participants}
    assert rec.seats == {"a@0": 0, "a@2": 1} and "units_market" in rec.private
    # ratings can be recomputed from the rated records alone (a fresh start for everyone, no inherited rating)
    first = lg.rerate()
    rated = [r for r in lg.records if r.rated]
    assert first == lg.rerate() and len(rated) < len(lg.records)
    assert {p: e.games for p, e in lg.entries.items()} == {p: sum(p in r.participants for r in rated) for p in lg.entries}
    assert lg.rerate(tags=["no_such_tag"])[0]["elo"] == 1000.0 and lg.rerate() == first


def test_heldout_population_never_trains_anybody_and_its_evaluation_is_logged(league):
    lg = league
    ho = lg.add_heldout_population(profiles=("aggressive", "passive"))
    aside = lg.snapshot("b", "set_aside", heldout=True)
    assert set(ho) == {"ho:hold", "ho:random", "ho:climber", "ho:climber_funds0.25", "ho:climber_funds4",
                       "ho:aggressive", "ho:passive"}
    assert set(lg.heldout_pool()) == set(ho) | {aside} and not set(lg.training_pool()) & set(lg.heldout_pool())
    assert aside not in lg.archive() and aside in lg.archive(include_heldout=True)
    assert lg.get("ho:climber_funds0.25").funds_scale == 0.25 and lg.get("ho:aggressive").privileged
    rng = np.random.default_rng(0)
    for method in ("uniform", "pfsp", "recent"):
        drawn = {o for _ in range(60) for o in lg.sample_opponents("a", 2, method, rng=rng)}
        assert drawn and not drawn & set(lg.heldout_pool()) and "a" not in drawn
    for bad in (["a", "ho:hold"], ["ho:random", "b"], ["a", aside]):
        with pytest.raises(ValueError, match="held-out"):
            lg.play(bad, 1)
        with pytest.raises(ValueError, match="held-out"):
            lg.train_match(bad, 1)
    with pytest.raises(ValueError, match="nobody learns"):
        lg.play(["a", "ho:hold"], 1, heldout_eval=True, learn=True)
    ratings = {p: (e.elo, e.mu, e.games) for p, e in lg.entries.items()}
    wr = lg.win_rate("a", "b")
    theta, n_rec = lg.get("a").learner.theta.copy(), len(lg.records)
    out = lg.evaluate_heldout("a", [7000], both_seatings=False)
    assert set(out["vs"]) == set(lg.heldout_pool()) and all(v["games"] == 1 for v in out["vs"].values())
    new = lg.records[n_rec:]
    assert len(new) == len(lg.heldout_pool()) and all(r.heldout_eval and not r.rated for r in new)
    assert {p: (e.elo, e.mu, e.games) for p, e in lg.entries.items()} == ratings      # no rating moved
    assert lg.win_rate("a", "b") == wr and np.array_equal(lg.get("a").learner.theta, theta)
    assert lg.win_rate("a", "ho:hold") == 0.5 and lg.win_rate("a", "ho:hold", include_heldout_eval=True) != 0.5
    with open(os.path.join(lg.root, "heldout_log.jsonl")) as fh:
        log = [json.loads(x) for x in fh]
    assert log == lg.heldout_log and log[-1]["policy"] == "a" and log[-1]["matches"] == len(new)
    # the resource-constrained variants really start with different funds
    by = {r.participants[1]: r for r in new}
    assert by["ho:climber_funds0.25"].outcomes["ho:climber_funds0.25"]["net_per_tick"] is not None
    assert not any(not r.heldout_eval and any(lg.entries[p].heldout for p in r.participants) for r in lg.records)


def test_prioritised_sampling_prefers_opponents_the_learner_loses_to(league):
    lg = league
    from crucible.training.league import MatchRecord
    k = len(lg.records)
    for i in range(6):                                  # recorded results: a loses to a@1, beats a@0
        for opp, s in (("a@1", 0.0), ("a@0", 1.0)):
            r = MatchRecord(f"x{i}{opp}", lg.task.family, i, "", "competition", 1.0, ["a", opp], {}, {}, {}, [["a", opp, s]])
            r.tag, r.rated = "synthetic", False
            lg.records.append(r)
    rng = np.random.default_rng(1)
    pool_excl = [p for p in lg.entries if p not in ("a@0", "a@1")]
    draws = [lg.sample_opponents("a", 1, "pfsp", exclude=pool_excl, rng=rng)[0] for _ in range(300)]
    assert lg.win_rate("a", "a@1") < 0.2 < 0.8 < lg.win_rate("a", "a@0")
    assert draws.count("a@1") > 5 * draws.count("a@0")
    del lg.records[k:]


def test_league_is_persistent_on_disk(league):
    lg = league
    lg.save()
    back = SelfPlayLeague.load(lg.root, SPEC)
    assert list(back.entries) == list(lg.entries) and back.ratings() == lg.ratings()
    assert back.info == json.loads(json.dumps(lg.info)) and back.heldout_log == lg.heldout_log
    assert [r.to_dict() for r in back.records] == json.loads(json.dumps([r.to_dict() for r in lg.records]))
    assert sorted(back.live) == sorted(lg.live) and back.heldout_pool() == lg.heldout_pool()
    assert np.array_equal(back.get("a").learner.theta, lg.get("a").learner.theta)
    assert np.array_equal(back.get("b").learner.parameters_flat(), lg.get("b").learner.parameters_flat())
    assert np.array_equal(back.get("a@2").learner.theta, lg.get("a@2").learner.theta)
    assert back.get("a@2").learner.frozen and back.get("b@1").learner.frozen
    assert back.get("a").history == lg.get("a").history and back.get("a").matches == lg.get("a").matches
    # the reloaded league plays the same match with the same result
    r1, r2 = lg.play(["a@1", "b@1"], 8000, rated=False), back.play(["a@1", "b@1"], 8000, rated=False)
    assert r1.outcomes == r2.outcomes
    dest = os.path.join(lg.root, "export")
    assert len(back.export_checkpoints(["a", "b", "a@0"], dest)) == 3
    assert Individual.load(os.path.join(dest, "b"), SPEC).learner.frozen


# ====================================================================== 5. the six modes
def test_the_six_interaction_modes_run_with_their_own_objectives_and_analyses(tmp_path):
    assert set(M.MODES) == set(M.MODE_NAMES) and len(M.MODES) == 6
    duo, hand, pool = arena_task("abm_duopoly", 0), arena_task("tech_handoff", 0), arena_task("tech_shared_pool", 0)
    kw = dict(iters=3, n_slots=2)
    comp = M.run_mode("competitive", duo, str(tmp_path / "comp"), algos=("spsa", "pg"), snapshot_every=1, **kw, reward_mode=LEGACY)
    assert comp["objective"] == "own net income (rig_net)" and comp["arch"] == ["spsa", "pg"]
    assert len(comp["curve"]) == 3 and set(comp["lagged"]["lever0"]) == {"lags", "corr", "n"}
    assert comp["league"].mode == "competition" and len(comp["league"].archive()) == 8
    coop = M.run_mode("cooperative", hand, str(tmp_path / "coop"), **kw, reward_mode=LEGACY)
    assert coop["objective"].startswith("mean net income of all seats") and coop["league"].mode == "cooperation"
    assert len(coop["with_default_partner"]) == 2 and "joint_gain_over_best_single" in coop
    rec = coop["league"].records[0]
    assert rec.scores[rec.participants[0]] == rec.scores[rec.participants[1]] == pytest.approx(2 * rec.joint_net_per_tick / 2 * 1.0)
    mixed = M.run_mode("mixed_motive", pool, str(tmp_path / "mixed"), **kw, reward_mode=LEGACY)
    assert mixed["both_restrain"]["joint_net_per_tick"] > mixed["both_grab"]["joint_net_per_tick"]
    assert mixed["position"] is not None and mixed["league"].mode == "mixed"
    members = [{"id": "s1", "algo": "spsa", "seed": 1}, {"id": "s2", "algo": "spsa", "seed": 2, "c": 0.2},
               {"id": "p1", "algo": "pg", "seed": 3}, {"id": "c1", "algo": "climber", "seed": 4}]
    pop = M.run_mode("population", duo, str(tmp_path / "pop"), members=members, **kw, reward_mode=LEGACY)
    assert sorted(pop["arch"]) == ["climber", "pg", "spsa", "spsa"] and len(pop["ratings"]) == 4
    assert all(len(r["pairs"]) == 2 and sorted(sum(r["pairs"], [])) == ["c1", "p1", "s1", "s2"] for r in pop["curve"])
    assert sum(r["games"] for r in pop["ratings"]) == 2 * 2 * 3 and np.array(pop["head_to_head"]["net"]).shape == (4, 4)
    hist = M.run_mode("historical", duo, str(tmp_path / "hist"), snapshot_every=1, p_archive=1.0, iters=3, n_slots=2, reward_mode=LEGACY)
    opps = [o for r in hist["curve"] for o in r["opponents"].values()]
    assert all("@" in o for o in opps) and hist["sampling"] == "pfsp"            # only archived stages were met
    assert hist["history"]["stages"] == [0, 1, 2, 3] and len(hist["history"]["elo"]) == 4
    lg = hist["league"]
    assert all(not lg.get(o).learns for o in set(opps))
    cross = M.run_mode("cross_play", hand, str(tmp_path / "cross"), objective="joint", iters=2, n_slots=2,
                       pairs=[{"algos": ("spsa", "spsa"), "seeds": (1, 2)}, {"algos": ("pg", "pg"), "seeds": (3, 4)}], reward_mode=LEGACY)
    m = cross["matrix"]
    j = np.array(m["joint"])
    assert j.shape == (2, 2) and m["gap_joint"] == pytest.approx(np.diag(j).mean() - (j[0, 1] + j[1, 0]) / 2)
    assert m["arch0"] == ["spsa", "pg"] and np.allclose(np.array(m["net0"]) + np.array(m["net1"]), j)
    json.dumps(M.strip(cross)), json.dumps(M.strip(pop)), json.dumps(M.strip(hist))
    with pytest.raises(ValueError):
        M.run_mode("population", hand, str(tmp_path / "bad"), members=members, **kw, reward_mode=LEGACY)   # seats differ in that world
    with pytest.raises(ValueError):
        M.run_mode("friendly", duo, str(tmp_path / "bad2"), reward_mode=LEGACY)


def test_mode_analysis_functions():
    x = np.cumsum(np.random.default_rng(0).standard_normal(200))
    y = np.concatenate([[0.0, 0.0], x[:-2]])                   # y repeats x two steps later
    lr = M.lagged_response(x, y, 4)
    assert lr["lags"][int(np.nanargmax([c if c is not None else -9 for c in lr["corr"]]))] == 2
    assert lr["corr"][lr["lags"].index(2)] > 0.99 and abs(lr["corr"][lr["lags"].index(-2)]) < 0.3
    assert M.commons_position(5.0, 10.0, 0.0) == 0.5 and M.commons_position(1.0, 0.0, 0.0) is None
    assert M.seat_bound(arena_task("tech_handoff", 0)) and not M.seat_bound(arena_task("abm_duopoly", 0))
    assert not set(M.EVAL_SEEDS) & set(M.HELDOUT_SEEDS) and min(M.EVAL_SEEDS + M.HELDOUT_SEEDS) > M.TRAIN_SEED0 + 500_000


# ====================================================================== 6. collusion and reward inflation
def _schedule(task, seed=0, every=4):
    rng = np.random.default_rng(seed)
    acts = {}

    def f(seat, t):
        key = (seat, t // every)
        if key not in acts:
            acts[key] = rng.uniform(-1, 1, SPEC.act_dim).astype(np.float32)
        return acts[key]
    return f


@pytest.mark.parametrize("fam", ["abm_duopoly", "tech_handoff"])
def test_no_transfer_between_participants_moves_the_objective(fam):
    """One direction, the other, a ring through a third participant, and through organisations: `rig_net` of every
    seat holder is identical to the cent with and without the payments; the bank balances differ by exactly them."""
    t = arena_task(fam, 0)
    e = int(t.endowment)
    big = 50 * e                                                # funds never bind, whatever the random settings cost
    pay = [{"t": 2, "from": 0, "to": 1, "amount": e // 3},                                    # A -> B
           {"t": 5, "from": 1, "to": 0, "amount": e // 7},                                    # B -> A
           {"t": 8, "from": 0, "to": "carol", "amount": e // 5},                              # a ring A -> C -> B -> A
           {"t": 8, "from": "carol", "to": 1, "amount": e // 5},
           {"t": 9, "from": 1, "to": 0, "amount": e // 5},
           {"t": 12, "from": 0, "to": "org:alpha", "owner": 0, "dst_owner": 0, "amount": e // 4},   # A's organisation
           {"t": 13, "from": "org:alpha", "to": "org:beta", "owner": 0, "dst_owner": 1, "amount": e // 6},
           {"t": 14, "from": "org:beta", "to": 0, "owner": 1, "amount": e // 8},              # ... and back to A
           {"t": 20, "from": 1, "to": 0, "amount": 1}]
    r = AU.transfer_invariance(t, _schedule(t), pay, ticks=24, funds=[big, big], extra=["carol"], reward_mode=LEGACY)
    assert r["payments"] == len(pay) and not r["binding"] and r["invariants"] == []
    assert r["objective_moved_cents"] == [0.0, 0.0]
    assert r["rig_net_with"] == r["rig_net_without"] and r["revenue_with"] == r["revenue_without"]
    assert r["balance_change_cents"] == r["net_received_cents"] and r["net_received_cents"][0] != 0
    assert r["orgs"] == {"org:alpha": e // 4 - e // 6, "org:beta": e // 6 - e // 8}
    base, paid = r["runs"]
    assert paid["rig_net"] == paid["net_from_reports"] == base["net_from_reports"]
    # the payments are on the ledger for anyone auditing the pair: recorded, not hidden, and not part of any net
    bt = AU.bilateral_transfers(paid["env"].world, paid["env"].actors)
    # (A->B, B->A twice, B's last cent, organisation to organisation, organisation back to A; not the legs via carol)
    assert len(bt["transfers"]) == 6 and bt["total_cents"] == e // 3 + e // 7 + e // 5 + e // 6 + e // 8 + 1
    assert AU.bilateral_transfers(base["env"].world, base["env"].actors)["total_cents"] == 0
    # conservation on the ledger: holders were paid by the demand account only, never more than it was given
    for run in (base, paid):
        lc = AU.ledger_conservation(run["env"].world)
        assert lc["ok"] and lc["paid"] <= lc["funded"] == 24 * t.income and lc["earned_by_holders"] == lc["paid"]
        assert sum(run["revenue"]) == lc["paid"]


def test_money_from_a_partner_changes_what_a_broke_seat_can_run_not_where_revenue_comes_from():
    """The one way a payment matters: a holder too poor to pay for its settings. The partner's money lets it run
    them; what it then earns is still paid by the independent demand, and the audit says that funds were binding."""
    t = arena_task("tech_handoff", 0)
    hi = np.ones(SPEC.act_dim, dtype=np.float32)
    sched = lambda seat, tick: hi                               # noqa: E731 - everything at maximum: expensive
    poor = int(0.02 * t.endowment)
    r = AU.transfer_invariance(t, sched, [{"t": 0, "from": 1, "to": 0, "amount": 5 * t.endowment}], ticks=20,
                               funds=[poor, 10 * t.endowment], reward_mode=LEGACY)
    assert r["binding"] and r["objective_moved_cents"][0] != 0
    base, paid = r["runs"]
    assert any(s != "ok" for s in base["status"][0]) and all(s == "ok" for s in paid["status"][0])
    for run in (base, paid):
        lc = AU.ledger_conservation(run["env"].world)
        assert lc["ok"] and sum(run["revenue"]) == lc["paid"] <= lc["funded"]


def test_joint_outcomes_never_exceed_what_independent_demand_paid():
    recs = []
    for fam in ("abm_duopoly", "abm_commons", "abm_complement", "tech_handoff", "tech_shared_pool"):
        t = arena_task(fam, 0)
        env = SelfPlayEnv([t, t], SPEC, reward_mode=LEGACY)
        for k, pair in enumerate(([RandomLearner(SPEC, seed=1, hold=5), RandomLearner(SPEC, seed=2, hold=5)],
                                  [FixedLearner(SPEC, action=-np.ones(SPEC.act_dim)), FixedLearner(SPEC, action=np.ones(SPEC.act_dim))])):
            inds = [Individual(f"x{s}", l, "own") for s, l in enumerate(pair)]
            recs += play(env, inds, noise=[10 * k, 10 * k + 1], learn=False, explore=True)
    c = AU.conservation(recs)
    assert c["matches"] == 20 and c["violations"] == [] and 0 < c["max_revenue_over_demand_income"] <= 1.0
    for r in recs:
        assert sum(s["revenue_cents"] for s in r["seats"]) == r["paid_by_demand_cents"] <= r["demand_income_cents"]
        if r["engine"] == "abm":                            # the engine's own bound: customers' budgets and income
            assert r["private"]["paid_seats"] <= r["private"]["paid_market"] <= r["revenue_bound_cents"]
    bad = json.loads(json.dumps(recs[0]))
    bad["seats"][0]["revenue_cents"] += bad["demand_income_cents"]
    assert AU.conservation([bad])["violations"]             # the check does fire on an inflated record


def _profile_pair(task, names, noise=(0, 1, 2, 3)):
    env = M._env(task, len(noise))
    inds = [None if n is None else P.profile_individual(task, s, n, SPEC) for s, n in enumerate(names)]
    for i in inds:
        if i is not None and not M.seat_bound(task):
            i.seat = None
    return play(env, inds, noise=list(noise), learn=False)


def test_audit_tells_value_creating_cooperation_from_tacit_collusion():
    duo = arena_task("abm_duopoly", 0)
    compete, collude = _profile_pair(duo, ("compete", "compete")), _profile_pair(duo, ("collude", "collude"))
    c = AU.classify_pair(collude, compete)
    ev = c["evidence"]
    # both firms raise price: both nets up, fewer customers served, each paying more: extraction, not value
    assert c["class"] == "tacit_collusion" and c["matched"] and all(c["flags"]["seats_better"])
    assert ev["value_delivered"]["relative_change"] < 0 and ev["price_per_unit"]["relative_change"] > 0.1
    assert ev["paired"]["joint_net_per_tick"]["mean_diff"] > 0 and ev["paired"]["value_delivered"]["mean_diff"] < 0
    assert c["circular_manipulation"]["possible"] is False and c["circular_manipulation"]["counted_in_any_net"] == 0.0
    assert "forbidden" in c["note"]                          # recorded for investigation, not suppressed
    assert AU.classify_pair(compete, collude)["class"] == "no_joint_gain"
    war = _profile_pair(duo, ("aggressive", "aggressive"))
    assert AU.classify_pair(war, compete)["class"] == "no_joint_gain"
    lop = AU.classify_pair(_profile_pair(duo, ("aggressive", "passive")), _profile_pair(duo, ("passive", "passive")))
    assert lop["class"] in ("one_sided", "no_joint_gain") and lop["flags"]["seats_worse"][1]
    # a pipeline run by two seats: both doing their part completes more work than either with an idle partner
    hand = arena_task("tech_handoff", 0)
    both = _profile_pair(hand, ("tuned", "tuned"))
    ref = AU.matched_reference(_profile_pair(hand, ("tuned", None)), _profile_pair(hand, (None, "tuned")))
    v = AU.classify_pair(both, ref)
    assert v["class"] == "value_creating_cooperation" and v["flags"]["more_value_delivered"]
    assert v["evidence"]["value_delivered"]["pair"] > v["evidence"]["value_delivered"]["reference"]
    assert v["evidence"]["value_completed"]["pair"] > v["evidence"]["value_completed"]["reference"]
    # a shared pool: mutual restraint against mutual over-use
    pool = arena_task("tech_shared_pool", 0)
    r = AU.classify_pair(_profile_pair(pool, ("restraint", "restraint")), _profile_pair(pool, ("grab", "grab")))
    assert r["class"] == "value_creating_cooperation"
    assert r["evidence"]["shared_pool_utilisation"]["pair"] < r["evidence"]["shared_pool_utilisation"]["reference"]
    # payments between the pair are reported next to the nets and never added to them
    paid = AU.classify_pair(collude, compete, transfers_cents=123456.0)
    assert paid["circular_manipulation"]["payments_between_the_pair_cents"] == 123456.0
    assert paid["evidence"]["joint_net_per_tick"] == ev["joint_net_per_tick"]
    with pytest.raises(ValueError):
        AU.matched_reference(compete, collude[:2])


def test_profiles_are_operator_side_fixtures():
    duo = arena_task("abm_duopoly", 0)
    a = P.profile_individual(duo, 0, "aggressive", SPEC)
    v = P.natural_view(duo, 0, a.learner.action)
    assert a.privileged and a.learner.frozen and "profile" in a.tags
    assert v["price"] == pytest.approx(0.95, abs=0.01) and v["outreach"] == pytest.approx(0.30, abs=0.01)
    assert P.natural_view(duo, 0, P.profile_action(duo, 0, "passive"))["price"] > 1.4
    assert set(P.available(arena_task("tech_shared_pool", 0))) >= {"default", "restraint", "grab", "tuned"}
    assert "restraint" not in P.available(arena_task("tech_handoff", 0))
    hand = arena_task("tech_handoff", 0)
    assert P.profile_settings(hand, 1, "default") == M._env(hand, 1).sims[0].engine.defaults(1)
