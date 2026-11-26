"""Baseline learners (`crucible.research.learners`): every registered learner honours the learner contract, the
scripted experimenters find what there is to find on mechanisms written out by hand, a short real training run of
each trainable learner improves its frozen score, and a learner run through `rig.*` tool calls in a kernel world
acts and is paid exactly as in the batched simulator."""
import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crucible.mechanisms.scm import handmade  # noqa: E402
from crucible.research import learners, tasks as T  # noqa: E402
from crucible.research.envs import MechVecEnv, run_episodes  # noqa: E402
from crucible.research.learner import EnvSpec  # noqa: E402
from crucible.research.learners.training import evaluate, pool_stream, summarise, train, train_pool  # noqa: E402
LEGACY = "legacy_shaped_reward"   # 4.0 baseline: the shaped reward is selected explicitly (Clarification 2, OO.8)

torch.set_num_threads(2)
SPEC = EnvSpec()
ALL = learners.available()
# small settings so that a weight update happens within a couple of short episodes
FAST = {"ppo_gru": dict(hidden=32, rollout_steps=200, epochs=2, minibatches=2),
        "ppo_gru_flat": dict(hidden=32, rollout_steps=200, epochs=2, minibatches=2),
        "model_based": dict(min_episodes=4, batch_episodes=4, updates_per_episode=1.0, feat_hidden=16),
        "evo_probe": dict(population=4),
        "meta_np": dict(d=16, heads=2, enc_blocks=1, dec_blocks=1, min_episodes=2, batch=4, updates_per_episode=1.0)}
TRAINABLE = [n for n in ALL if learners.describe(n)["learns_across_episodes"]]
STOCHASTIC = ["random", "hill_climb", "ppo_gru", "ppo_gru_flat", "model_based", "evo_probe", "meta_np"]


def make(name, seed=0, **kw):
    return learners.make(name, SPEC, seed=seed, **dict(FAST.get(name, {}), **kw))


# ---------------------------------------------------------------------------- mechanisms written out by hand
@dataclass(frozen=True)
class HandTask(T.Task):
    """A task over a hand-made mechanism (no registered family): the mechanism is carried by the task."""
    mech: dict | None = None

    def mechanism(self) -> dict:
        return self.mech


def _hand_task(mech: dict, k: int, horizon: int, best: dict | None, family: str = "hand") -> HandTask:
    inc = int(mech["config"]["income_cents_per_tick"])
    base = dict(family=family, seed=k, mech_seed=k, world_seed=1000 + k, income=inc,
                endowment=T.ENDOWMENT_TICKS * inc, horizon=horizon, n_seats=1, validated=True, engine="scm",
                transfer="base", split="train", mech=mech)

    def net(x):
        return float(np.mean([HandTask(**dict(base, world_seed=5000 + s)).sim().run(x, horizon)["net"][:, 0].mean()
                              for s in range(3)]))
    d = net(None)
    r = net(best) if best is not None else d + 1.0
    return HandTask(**dict(base, default_net=d, reference_net=r, goal_net_cents=int(round(horizon * (d + r) / 2))))


def easy_task(k: int, horizon: int = 60, noise: float = 0.02) -> HandTask:
    """Lever `a` pays in proportion (best at the top of its range), `b` pays with diminishing returns against a
    linear cost (best in the interior), `c` and `d` only cost. Order, ranges, weights and costs vary with k.
    Leaving everything alone earns about nothing, random settings lose money. The reference is the best of a
    grid of constant settings of (a, b) on the noise-free steady state, its net measured by simulation."""
    rng = np.random.default_rng([k, 99])
    wa, wb = rng.uniform(0.5, 0.9), rng.uniform(0.4, 0.8)
    specs = [{"name": "a", "lo": 0.0, "hi": float(rng.choice([1.0, 10.0])), "cost": float(rng.uniform(100, 200))},
             {"name": "b", "lo": 0.0, "hi": float(rng.choice([1.0, 100.0])), "cost": float(rng.uniform(250, 400))},
             {"name": "c", "lo": -1.0, "hi": 1.0, "cost": float(rng.uniform(900, 1300))},
             {"name": "d", "lo": 0.0, "hi": 4.0, "cost": float(rng.uniform(500, 800))}]
    for s in specs:
        s["default"] = s["lo"]
    kb = 4.0
    mech = handmade(levers=[specs[i] for i in rng.permutation(4)],
                    latents=[{"id": "m", "terms": [{"kind": "linear", "src": "a", "w": 1.0}], "noise": noise}],
                    outcome={"terms": [{"kind": "linear", "src": "m", "w": wa},
                                       {"kind": "saturating", "src": "b", "w": wb, "k": kb},
                                       {"kind": "linear", "src": "c", "w": 0.05}],
                             "bias": 0.3, "noise": noise, "cap": 3.0},
                    price=10.0, units=100.0, overhead=300.0, horizon=horizon)
    ub = np.linspace(0.0, 1.0, 201)
    gain = 1000.0 * wb * (1 - np.exp(-kb * ub)) / (1 - np.exp(-kb)) - specs[1]["cost"] * ub
    best = {"a": specs[0]["hi"], "b": float(ub[int(np.argmax(gain))]) * specs[1]["hi"]}
    return _hand_task(mech, k, horizon, best, "hand_easy")


def regular_task(k: int, horizon: int = 24, noise: float = 0.02) -> HandTask:
    """Worlds that share a regularity a prior can internalise: a lever that rests at the bottom of its range pays
    when raised, a lever that rests in the middle only costs (it is best turned down). Order, ranges, weights and
    costs vary with k; nothing but the resting position tells the two kinds apart before acting."""
    rng = np.random.default_rng([k, 77])
    pay, cost = [], []
    for i in range(2):
        pay.append({"name": f"p{i}", "lo": 0.0, "hi": float(rng.choice([1.0, 10.0, 100.0])), "default": 0.0,
                    "cost": float(rng.uniform(100, 250))})
    w = [float(rng.uniform(0.5, 0.9)) for _ in range(2)]
    for i in range(2):
        hi = float(rng.choice([1.0, 10.0]))
        cost.append({"name": f"c{i}", "lo": 0.0, "hi": hi, "default": hi / 2, "cost": float(rng.uniform(300, 600))})
    specs = pay + cost
    mech = handmade(levers=[specs[i] for i in rng.permutation(4)],
                    latents=[{"id": "m", "terms": [{"kind": "linear", "src": "p0", "w": 1.0}], "noise": noise}],
                    outcome={"terms": [{"kind": "linear", "src": "m", "w": w[0]},
                                       {"kind": "linear", "src": "p1", "w": w[1]}],
                             "bias": 0.4, "noise": noise, "cap": 3.0},
                    price=10.0, units=100.0, overhead=400.0, horizon=horizon)
    best = {"p0": pay[0]["hi"], "p1": pay[1]["hi"], "c0": 0.0, "c1": 0.0}
    return _hand_task(mech, k, horizon, best, "hand_regular")


def regime_task(k: int = 0, horizon: int = 120, at: int = 60) -> HandTask:
    """At tick `at` lever `a` stops paying and lever `b` starts to: the settings that were best lose money."""
    mech = handmade(levers=[{"name": "a", "cost": 150.0}, {"name": "b", "cost": 150.0}, {"name": "c", "cost": 900.0}],
                    latents=[{"id": "m", "terms": [{"kind": "linear", "src": "c", "w": 0.02}], "noise": 0.01}],
                    outcome={"terms": [{"kind": "linear", "src": "a", "w": 0.8, "w2": 0.0},
                                       {"kind": "linear", "src": "b", "w": 0.0, "w2": 0.8},
                                       {"kind": "linear", "src": "m", "w": 1.0}], "bias": 0.3, "noise": 0.02, "cap": 3.0},
                    price=10.0, units=100.0, overhead=300.0, horizon=horizon, regime_at=at)
    return _hand_task(mech, k, horizon, None, "hand_regime")


@pytest.fixture(scope="module")
def easy():
    return [easy_task(k) for k in range(8)]


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    d = tmp_path_factory.mktemp("crucible-cache")
    old = os.environ.get("CRUCIBLE_CACHE")
    os.environ["CRUCIBLE_CACHE"] = str(d)
    yield d
    if old is None:
        os.environ.pop("CRUCIBLE_CACHE", None)
    else:
        os.environ["CRUCIBLE_CACHE"] = old


# ---------------------------------------------------------------------------- helpers
def rollout(lr, tasks, explore=False, learn=False, ticks=None, other=None, junk=False):
    """Drive `lr` on one episode of every task. Returns (actions [t, n, act], rewards [t, n], records, and the
    actions of `other`, a second learner shown the same observations - with garbage in the masked slots if
    `junk`)."""
    env = MechVecEnv(list(tasks), SPEC, reward_mode=LEGACY)
    obs = env.reset()
    lr.reset_context(env.n)
    if other is not None:
        other.reset_context(env.n)
    acts, rews, oacts, upd = [], [], [], 0
    while not env.done.all() and (ticks is None or len(acts) < ticks):
        a = lr.act(obs, explore=explore)
        if other is not None:
            o2 = obs.copy()
            if junk:
                o2[:, SPEC.gauges][obs[:, SPEC.gauge_mask] == 0] = 3.3
                o2[:, SPEC.last_action][obs[:, SPEC.lever_mask] == 0] = 0.77
            oacts.append(other.act(o2, explore=explore).copy())
        nxt, r, d, _ = env.step(a)
        if learn:
            lr.observe(obs, a, r, d, nxt)
            upd += int(lr.update().get("updates", 0))
        acts.append(np.array(a, copy=True))
        rews.append(r.copy())
        obs = nxt
    return np.stack(acts), np.stack(rews), env.drain(), (np.stack(oacts) if oacts else None), upd


def fingerprint(lr, tmp_path) -> str:
    """A digest of everything `save` writes: the learner's long-term state, bit for bit."""
    path = str(tmp_path / f"fp-{lr.name}.ckpt")
    lr.save(path)
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, torch.Tensor):
            h.update(x.detach().cpu().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(x.tobytes())
        else:
            h.update(repr(x).encode())
    try:
        walk(torch.load(path, map_location="cpu", weights_only=False))
    except Exception:                                   # noqa: BLE001 - a JSON checkpoint
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def train_until_update(lr, tasks, max_passes=12) -> int:
    lr.freeze(False)
    upd = 0
    for _ in range(max_passes):
        upd += rollout(lr, tasks, explore=True, learn=True)[4]
        if upd:
            break
    return upd


# ---------------------------------------------------------------------------- the registry
def test_registry_lists_describes_and_builds_every_learner():
    assert {"hold", "random", "probe_exploit", "hill_climb", "ppo_gru", "model_based"} <= set(ALL)
    for name in ALL:
        d = learners.describe(name)
        assert {"kind", "has_weights", "learns_across_episodes", "defaults", "doc"} <= set(d) and d["doc"]
        lr = learners.make(name, SPEC, seed=3)
        assert lr.name == name and lr.spec == SPEC and lr.seed == 3 and lr.kind == d["kind"]
        assert (lr.usage().params > 0) == d["has_weights"]
        if d["defaults"]:                              # every documented option is accepted
            k = sorted(d["defaults"])[0]
            learners.make(name, SPEC, **{k: d["defaults"][k]})
    assert learners.describe("ppo_gru")["kind"] == "model_free" and learners.describe("model_based")["kind"] == "model_based"
    assert not learners.describe("probe_exploit")["has_weights"] and learners.describe("ppo_gru")["has_weights"]
    with pytest.raises(KeyError):
        learners.make("no_such_learner")
    with pytest.raises(TypeError):
        learners.make("ppo_gru", SPEC, no_such_option=1)
    assert learners.make("ppo_gru").spec == EnvSpec()   # the spec is optional


# ---------------------------------------------------------------------------- contract conformance
@pytest.mark.parametrize("name", ALL)
def test_actions_have_the_contract_shape_and_bounds_and_masked_slots_are_ignored(name, easy):
    tasks = [easy[0], easy[1], regime_task()]          # 4, 4 and 3 levers; two horizons
    a, b = make(name, seed=5), make(name, seed=5)
    acts, rews, recs, other, _ = rollout(a, tasks, other=b, junk=True)
    assert acts.shape[1:] == (3, SPEC.act_dim) and acts.dtype == np.float32
    assert np.isfinite(acts).all() and acts.min() >= -1.0 and acts.max() <= 1.0
    assert len(recs) == 3 and acts.shape[0] == max(t.horizon for t in tasks)
    # garbage in the masked parts of the observation changes nothing on the levers that exist
    for i, t in enumerate(tasks):
        n = len(t.sim().engine.levers(0))
        assert np.array_equal(acts[:, i, :n], other[:, i, :n]), name
    assert np.isfinite(other).all() and np.abs(other).max() <= 1.0
    # explore=True also respects the bounds
    acts2 = rollout(make(name, seed=5), tasks, explore=True, ticks=20)[0]
    assert np.isfinite(acts2).all() and np.abs(acts2).max() <= 1.0


@pytest.mark.parametrize("name", ALL)
def test_reset_context_clears_context_but_not_weights(name, easy, tmp_path):
    tasks = easy[:2]
    lr = make(name, seed=1)
    if name in TRAINABLE:
        assert train_until_update(lr, easy[:4]) > 0     # weights that are not the initial ones
    lr.freeze(True)
    fp = fingerprint(lr, tmp_path)
    ref = rollout(lr, tasks, ticks=24)[0]
    again = rollout(lr, tasks, ticks=24)[0]             # rollout resets the context first
    if name != "random":                                # (the random baseline continues its stream by design)
        assert np.array_equal(ref, again), "a fresh context must not remember the previous episode"
        assert not np.array_equal(ref[4:14, 0], ref[14:24, 0]) or name == "hold"
        # reset one slot in mid-episode: it starts over, the other slot is not disturbed
        env = MechVecEnv(list(tasks), SPEC, reward_mode=LEGACY)
        obs = env.reset()
        lr.reset_context(2)
        got = []
        for t in range(24):
            if t == 10:
                which = np.array([True, False])
                obs = env.reset(which=which)
                lr.reset_context(2, which=which)
            a = lr.act(obs, explore=False)
            got.append(a.copy())
            obs = env.step(a)[0]
        got = np.stack(got)
        assert np.array_equal(got[:, 1], ref[:, 1])
        assert np.array_equal(got[10:, 0], ref[:14, 0])
    assert fingerprint(lr, tmp_path) == fp


@pytest.mark.parametrize("name", ["probe_exploit", "hill_climb", "model_based", "evo_probe"])
def test_a_context_reset_in_mid_episode_plans_for_the_ticks_that_are_left(name, easy):
    """The protocols reset a learner's context inside an episode (the context-reset control). The learners that
    plan against the horizon read it off the clock, and must then plan for what is left, not for a whole episode."""
    lr = make(name)
    env = MechVecEnv(easy[:1], SPEC, reward_mode=LEGACY)
    obs = env.reset()
    lr.reset_context(1)
    for t in range(44):
        if t == 36:
            assert lr._slots[0].h == easy[0].horizon
            lr.reset_context(1)
        obs = env.step(lr.act(obs, explore=False))[0]
    assert lr._slots[0].h == easy[0].horizon - 36


@pytest.mark.parametrize("name", ALL)
def test_freeze_blocks_every_weight_change_bit_for_bit(name, easy, tmp_path):
    lr = make(name, seed=2)
    if name in TRAINABLE:
        assert train_until_update(lr, easy[:4]) > 0
        before = fingerprint(lr, tmp_path)
        assert train_until_update(lr, easy[:4]) > 0
        assert fingerprint(lr, tmp_path) != before, "training must change the long-term state"
    lr.freeze(True)
    fp = fingerprint(lr, tmp_path)
    for _ in range(3):                                  # experience offered while frozen is refused
        upd = rollout(lr, easy[:4], explore=True, learn=True)[4]
        assert upd == 0 and lr.update().get("updates", 0) == 0
    run_episodes(MechVecEnv(easy[:4], SPEC, reward_mode=LEGACY), lr, explore=False, learn=True)
    assert fingerprint(lr, tmp_path) == fp
    assert lr.frozen
    lr.freeze(False)
    assert not lr.frozen
    if name in TRAINABLE:                               # nothing seen while frozen was kept for later
        n_before = lr.usage().updates
        assert lr.update().get("updates", 0) == 0 and lr.usage().updates == n_before


@pytest.mark.parametrize("name", ALL)
def test_save_load_round_trip_reproduces_actions_exactly(name, easy, tmp_path):
    a = make(name, seed=4)
    if name in TRAINABLE:
        assert train_until_update(a, easy[:4]) > 0
    path = str(tmp_path / f"{name}.ckpt")
    a.save(path)
    b = make(name, seed=4)
    if name in TRAINABLE:
        assert fingerprint(b, tmp_path) != fingerprint(a, tmp_path)
    b.load(path)
    assert fingerprint(b, tmp_path) == fingerprint(a, tmp_path)
    a.freeze(True)
    b.freeze(True)
    if name == "random":
        a = make(name, seed=4)                          # no weights: equal from equal starts
    ra, rb = rollout(a, easy[:3]), rollout(b, easy[:3])
    assert np.array_equal(ra[0], rb[0]) and np.array_equal(ra[1], rb[1])
    # a checkpoint holds no episodic context: loading in mid-episode state of another learner changes nothing
    assert b.usage().params == a.usage().params


@pytest.mark.parametrize("name", ALL)
def test_a_seed_determines_everything(name, easy, tmp_path):
    def run(seed):
        lr = make(name, seed=seed)
        out = [rollout(lr, easy[:4], explore=True, learn=name in TRAINABLE)[0] for _ in range(3)]
        lr.freeze(True)
        return np.concatenate(out), rollout(lr, easy[:2])[0], fingerprint(lr, tmp_path)
    a, b, c = run(11), run(11), run(12)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    if name in STOCHASTIC:
        assert not np.array_equal(a[0], c[0])


@pytest.mark.parametrize("name", ALL)
def test_usage_accounting_adds_up(name, easy):
    lr = make(name, seed=6)
    tasks = [easy[0], easy[1], regime_task()]
    steps = sum(t.horizon for t in tasks)
    u = lr.usage()
    assert u.env_steps == 0 and u.updates == 0 and u.search_evals == 0
    run_episodes(MechVecEnv(tasks, SPEC, reward_mode=LEGACY), lr, explore=False)
    assert u.env_steps == steps                         # counted once, by the harness
    if name not in ("hold", "random"):
        assert u.infer_cpu_s > 0
    assert u.train_cpu_s == 0 and u.updates == 0        # acting is not training
    d = learners.describe(name)
    if name == "model_based":
        per = lr.p["grid"]                              # one grid per lever per decision, every `dwell` ticks
        want = sum(((t.horizon - 1) // lr.p["dwell"]) * len(t.sim().engine.levers(0)) * per for t in tasks)
        assert u.search_evals == want
        assert u.params == sum(q.numel() for q in lr.net.parameters())
    elif name == "meta_np":                             # searches over settings with its model at every decision
        assert u.search_evals > 0
        assert u.params == sum(q.numel() for q in lr.net.parameters())
    else:
        assert u.search_evals == 0
    if name.startswith("ppo_gru"):
        assert u.params == sum(q.numel() for q in lr.net.parameters())
    if d["learns_across_episodes"]:
        returned = 0
        for _ in range(6):
            returned += rollout(lr, easy[:4], explore=True, learn=True)[4]
        assert returned > 0 and u.updates == returned   # optimiser steps reported once
        assert u.extra["observed_steps"] == 6 * sum(t.horizon for t in easy[:4])
        assert u.train_cpu_s > 0
    else:
        assert rollout(lr, easy[:2], explore=True, learn=True)[4] == 0 and u.updates == 0 and u.params == 0
    assert set(u.to_dict()) >= {"env_steps", "updates", "train_cpu_s", "infer_cpu_s", "search_evals", "params"}


# ---------------------------------------------------------------------------- what the learners achieve
def test_scripted_experimenters_beat_hold_and_random_on_easy_mechanisms(easy):
    sc = {n: np.array([r["score"] for r in evaluate(learners.make(n, SPEC, seed=1), easy, reward_mode=LEGACY)])
          for n in ("hold", "random", "probe_exploit", "hill_climb")}
    assert abs(sc["hold"].mean()) < 0.05 and sc["random"].mean() < -0.1
    assert sc["probe_exploit"].mean() > sc["hold"].mean() + 0.3
    assert sc["probe_exploit"].mean() > sc["random"].mean() + 0.5
    assert (sc["probe_exploit"] > sc["hold"] + 0.15).all() and (sc["probe_exploit"] > sc["random"]).all()
    assert sc["hill_climb"].mean() > sc["hold"].mean() + 0.1 and sc["hill_climb"].mean() > sc["random"].mean() + 0.3
    # the experiment costs the first half of the episode and pays in the second
    recs = evaluate(learners.make("probe_exploit", SPEC), easy, reward_mode=LEGACY)
    assert np.mean([r["score_second_half"] for r in recs]) > np.mean([r["score_first_half"] for r in recs]) + 0.3
    assert np.mean([r["score_second_half"] for r in recs]) > 0.7


def test_probe_exploit_probes_again_when_net_collapses():
    t = regime_task()
    out = {}
    for reprobe in (True, False):
        lr = learners.make("probe_exploit", SPEC, reprobe=reprobe)
        rews = rollout(lr, [t])[1]
        out[reprobe] = (lr._slots[0].probes, float(rews[-25:].mean()), float(rews[40:58].mean()))
    assert out[True][0] == 2 and out[False][0] == 1                 # one new experiment after the switch
    assert out[True][2] > 0.1                                       # it had found the first regime's settings
    assert out[True][1] > out[False][1] + 0.1                       # and recovers in the second; without, it does not


def test_model_based_identifies_an_easy_mechanism_within_the_episode(easy):
    recs = evaluate(learners.make("model_based", SPEC, seed=0), easy, reward_mode=LEGACY)
    hold = evaluate(learners.make("hold", SPEC), easy, reward_mode=LEGACY)
    assert np.mean([r["score"] for r in recs]) > np.mean([r["score"] for r in hold]) + 0.5
    assert np.mean([r["success"] for r in recs]) >= 0.75
    assert np.mean([r["score_second_half"] for r in recs]) > np.mean([r["score_first_half"] for r in recs]) + 0.15


def test_a_short_training_run_improves_the_frozen_score_of_ppo_gru(easy):
    """Real training (recurrent PPO, about 29k environment steps on 8 easy worlds): the frozen policy with fresh
    context earns clearly more than the same network before training, and more than leaving the levers alone."""
    lr = learners.make("ppo_gru", SPEC, seed=0, hidden=32, rollout_steps=960, lr=1e-3)
    before = summarise(evaluate(lr, easy, reward_mode=LEGACY))["score"]
    curve = train(lr, pool_stream(easy, 0), total_steps=28_800, n_envs=16, eval_tasks=easy, eval_every=14_400, reward_mode=LEGACY)
    after = summarise(evaluate(lr, easy, reward_mode=LEGACY))["score"]
    assert lr.usage().updates > 0 and lr.usage().env_steps >= 28_800
    assert len(curve) == 3 and curve[0]["steps"] == 0 and curve[-1]["steps"] >= 28_800
    assert curve[-1]["updates"] == lr.usage().updates and curve[-1]["train_cpu_s"] > 0
    assert after > before + 0.15, (before, after)
    assert after > 0.1, after                                        # better than never touching a lever


def test_a_short_training_run_improves_the_frozen_score_of_model_based():
    """Real training of the prior (meta-learned Bayesian last layer) on short episodes of worlds that share a
    regularity (which levers pay can be read off where they rest): weights frozen and context fresh, the trained
    learner identifies a world faster than the untrained one, on the training worlds and on worlds of the same
    kind it never saw. The predictive likelihood of held-out episodes improves too."""
    tasks = [regular_task(k) for k in range(12)]
    unseen = [regular_task(k) for k in range(100, 112)]
    lr = learners.make("model_based", SPEC, seed=0, min_episodes=24, batch_episodes=16)
    before, before_unseen = summarise(evaluate(lr, tasks, reward_mode=LEGACY))["score"], summarise(evaluate(lr, unseen, reward_mode=LEGACY))["score"]
    assert lr.usage().updates == 0
    curve = train(lr, pool_stream(tasks, 0), total_steps=3_600, n_envs=12, eval_tasks=tasks, eval_every=1_800, reward_mode=LEGACY)
    lr.freeze(True)
    after, after_unseen = summarise(evaluate(lr, tasks, reward_mode=LEGACY))["score"], summarise(evaluate(lr, unseen, reward_mode=LEGACY))["score"]
    assert lr.usage().updates >= 100 and curve[-1]["updates"] == lr.usage().updates
    assert after > before + 0.1, (before, after)
    assert after_unseen > before_unseen + 0.05, (before_unseen, after_unseen)
    # what was learned is a better predictive model: episodes of unseen worlds, collected by an untrained learner
    collector = learners.make("model_based", SPEC, seed=5)
    for _ in range(3):
        run_episodes(MechVecEnv(unseen, SPEC, reward_mode=LEGACY), collector, explore=True, learn=True)
    episodes = list(collector._buffer)
    assert len(episodes) == 36 and collector.usage().updates == 0
    assert lr.heldout_nll(episodes) < collector.heldout_nll(episodes) - 0.2


# ---------------------------------------------------------------------------- the same problem through tools
@pytest.mark.parametrize("name", ["probe_exploit", "hill_climb", "model_based", "ppo_gru", "evo_probe"])
def test_a_learner_acts_and_is_paid_identically_through_kernel_tool_calls(name):
    """One learner, two hosts: the batched simulator and a kernel world operated with `rig.set` / `rig.read`.
    Same observations, hence the same actions, and the same rewards, tick for tick."""
    from crucible.research.bridge import KernelMechEnv
    t = T.make_task("scm_additive", 3, validate=False)
    ticks = 40
    sim_env, ker_env = MechVecEnv([t], SPEC, reward_mode=LEGACY), KernelMechEnv(t, SPEC, reward_mode=LEGACY)
    la, lb = make(name, seed=9), make(name, seed=9)
    oa, ob = sim_env.reset(), ker_env.reset()
    la.reset_context(1)
    lb.reset_context(1)
    moved = 0
    for k in range(ticks):
        a, b = la.act(oa, explore=False), lb.act(ob, explore=False)
        assert np.array_equal(a, b), (name, k)
        moved += int(not np.array_equal(a, oa[:, SPEC.last_action]))
        oa, ra, _, ia = sim_env.step(a)
        ob, rb, _, ib = ker_env.step(b)
        assert np.array_equal(oa, ob), (name, k)
        assert ra[0] == rb[0] and ia["net_cents"][0] == ib["net_cents"][0]
    assert moved > 5                                                 # the learner really operated the facility
    assert sum(1 for c in ker_env.calls if c["tool"] == "rig.set") == ticks


def test_the_training_pool_is_the_training_stream_from_a_cache(cache):
    fams = ["scm_additive", "scm_gate"]
    pool = train_pool(fams, 2)
    stream = T.train_stream(fams)
    assert pool == [next(stream) for _ in range(4)]
    assert train_pool(fams, 2) == pool                               # second time: from disk
    assert all(not t.validated and t.seed <= T.TRAIN_SEED_MAX for t in pool)
    s = pool_stream(pool, shuffle_seed=3)
    first, second = [next(s).id for _ in range(4)], [next(s).id for _ in range(4)]
    assert sorted(first) == sorted(second) == sorted(t.id for t in pool)
