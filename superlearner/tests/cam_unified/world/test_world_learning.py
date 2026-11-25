"""Learning tests on synthetic sequence worlds (spec 6.6 / 7.4).

Default run: the world model learns (a) a delayed vs an immediate effect, (b) a non-additive interaction of two
simultaneous interventions, (c) open-loop multistep predictions (irregular elapsed times, nonlinear dynamics) that
beat no-change / trend-persistence / key-mean / per-key linear-regression baselines at every horizon, (d) filtered
predictions that follow an unannounced mechanism change, and later predictions that follow a changed intermediate
event (spec 6.6(4), 6.6(6)).
Slow (CAM_RUN_SLOW=1): on ProbeWorld models from three training seeds, the closed-loop planner picks the
immediately costly experiment that enables the later payoff by a margin, imagination changes the real action, the
three planners are compared at matched compute, and a policy trained on imagined targets learns to experiment --
with ablations showing that the real-experience term alone cannot teach it and that imagination without belief
revision does not learn to probe.
"""
import math
import time

import pytest
import torch

from superlearner.cam_unified.interfaces import Action, Provenance, SequenceBatch
from superlearner.cam_unified.testing import DoubleCore, tiny_config
from superlearner.cam_unified.world.baselines import (KeyMeanBaseline, LinearRegressionBaseline, NoChangeBaseline,
                                                      TrendPersistenceBaseline, WorldModelForecaster,
                                                      evaluate_multistep)
from superlearner.cam_unified.world.dynamics import WorldModel
from superlearner.cam_unified.world.imagination import Candidates, Imagination, PolicyContext
from superlearner.cam_unified.world.imagine_train import (combine_real_and_imagined, imagination_actor_critic_loss,
                                                          imagined_targets)
from superlearner.cam_unified.world.planner import PolicyRolloutPlanner, compare_planners

from world_synth import (DelayWorld, HypothesisBelief, InteractionWorld, InvestWorld, ProbeWorld, StockWorld,
                         SwitchWorld, batch_to, device, probe_candidates, probe_hooks, real_start, train)

DEV = device()


def _gen():
    return torch.Generator(device=DEV).manual_seed(0)


def _wm(cfg, **kw):
    torch.manual_seed(0)
    return WorldModel(cfg, DoubleCore(cfg), **kw).to(DEV)


DELAY_STEPS, DELAY_LR, DELAY_LR_END = 700, 4e-3, 0.1
DELAY_TRUTH = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0, 0, 0]])


def _delay_run(model_seed: int = 0):
    """Train the DelayWorld model and forecast the three plans open loop from step 2 -> (pred [plan, h], loss
    history, world, model, test batch).

    Recipe (diagnosis in world/results/delay_effect_diagnosis.json): the immediate effect is learned within ~100
    steps, the delayed one (credit through two open-loop transitions) only between ~250 and ~550 steps, and with a
    constant learning rate 4e-3 the loss keeps spiking (max loss over the last 100 steps 10-18 vs ~6-8 final), which
    transiently erases the delayed effect first. The old recipe (450 steps, constant lr) stopped inside that
    transient, so non-deterministic CUDA kernels decided pass or fail (1 of 6 seeds failed, 2 of 4 runs in the merged
    suite; a decay to 0.1x within 450 steps fails 4 of 6 - it is under-training, not a marginal threshold). 700 steps
    with a linear decay to 0.1x passed 18 of 18 runs (12 model seeds, 6 of them repeated) with max error <= 0.053."""
    cfg = tiny_config()
    w = DelayWorld(cfg.d_model)
    batches = [batch_to(w.batch(64, seed=i), DEV) for i in range(16)]
    torch.manual_seed(model_seed)
    wm = WorldModel(cfg, DoubleCore(cfg), train_horizons=(1, 2, 3, 4), open_loop_samples=2,
                    open_loop_starts=4).to(DEV)
    hist = train(wm, batches, DELAY_STEPS, lr=DELAY_LR, lr_end_frac=DELAY_LR_END, seed=model_seed)
    wm.eval()
    plans = [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    tb = batch_to(w.batch(3, seed=999, plans=plans), DEV)
    f = wm.forecast(tb, 2, (1, 2, 3), n_samples=8, generator=_gen())
    pred = torch.stack([f[h]["mean"][:, 0] for h in (1, 2, 3)], 1).cpu()        # [plan, horizon]
    return pred, hist, w, wm, tb, batches


def _chance_max_err(batches) -> float:
    """Max error of the climatology forecast (the training mean of y for every plan and horizon): what a model
    that represents neither effect scores against the same truth table."""
    ys = torch.cat([s.outcomes.y[s.outcomes.y_mask].cpu() for b in batches for s in b.steps])
    return float((torch.full_like(DELAY_TRUTH, float(ys.mean())) - DELAY_TRUTH).abs().max())


def test_learns_delayed_versus_immediate_effects():
    """A has an immediate unit effect on y, B the same effect two decisions later (spec 6.6(1),(2))."""
    t0 = time.time()
    pred, hist, w, wm, tb, batches = _delay_run(0)
    chance = _chance_max_err(batches)
    print(f"delay world: {time.time() - t0:.0f}s, loss {hist[0]:.2f}->{hist[-1]:.2f}, predictions\n{pred}\n"
          f"climatology max error {chance:.2f}")
    assert chance >= 0.3                          # the criterion separates learning from the chance forecast
    assert float((pred - DELAY_TRUTH).abs().max()) < 0.3, pred
    # A07 (learned): from the same filtered state, contemplated A vs noop predict different consequences
    with torch.no_grad():
        Z = wm.filter(SequenceBatch(tb.steps[:3]))[-1][2:3]
        ob = tb.steps[2].outcomes
        q = type(ob)(ob.q.select([0]), ob.lik[:1], ob.n_classes[:1], ob.y[:1], ob.censored[:1], ob.y_mask[:1],
                     ob.channel[:1])
        yA = wm.outcome_dist(wm.prior(Z, w.act_tokens([1]).to(DEV), None, 1.0, sample=False)[0], q).mean()
        y0 = wm.outcome_dist(wm.prior(Z, w.act_tokens([0]).to(DEV), None, 1.0, sample=False)[0], q).mean()
    assert float(yA - y0) > 0.7


@pytest.mark.slow
def test_delayed_effect_learning_is_reliable_across_model_seeds():
    """The same criterion on three further model seeds (all must pass; the diagnosis file has 18 runs)."""
    errs = []
    for s in (1, 2, 3):
        pred, *_ = _delay_run(s)
        errs.append(float((pred - DELAY_TRUTH).abs().max()))
        print(f"model seed {s}: delayed {float(pred[1, 2]):.3f} immediate {float(pred[0, 0]):.3f} max err {errs[-1]:.3f}")
    assert max(errs) < 0.3, errs


def test_learns_interaction_of_two_interventions():
    """y = 0.5x - 0.5w + 2xw for two simultaneous interventions; additive regression cannot represent it."""
    cfg = tiny_config()
    w = InteractionWorld(cfg.d_model)
    raw = [w.batch(64, seed=i) for i in range(10)]
    wm = _wm(cfg, train_horizons=(1, 2), open_loop_samples=2, open_loop_starts=2)
    t0 = time.time()
    train(wm, [batch_to(b, DEV) for b in raw], 170, lr=2e-3)
    wm.eval()
    tb = w.batch(4, seed=77, plans=[[0, a, 0, 0] for a in range(4)])
    pred = wm.forecast(batch_to(tb, DEV), 1, (1,), n_samples=8, generator=_gen())[1]["mean"][:, 0].cpu()
    truth = torch.tensor([w.truth(a) for a in range(4)])
    contrast = float(pred[3] - pred[1] - pred[2] + pred[0])
    lin = LinearRegressionBaseline(horizons=(1,)).fit(raw).predict(tb, 1, (1,))[1][:, 0]
    lin_contrast = float(lin[3] - lin[1] - lin[2] + lin[0])
    print(f"interaction world: {time.time() - t0:.0f}s, wm {pred.tolist()} truth {truth.tolist()} "
          f"contrast {contrast:.2f} (linear regression {lin_contrast:.2f})")
    assert float((pred - truth).abs().max()) < 0.3
    assert abs(contrast - 2.0) < 0.4 and abs(lin_contrast) < 0.5


def test_multistep_open_loop_beats_baselines():
    """Irregular dt, decay and an overload crash: open-loop error vs horizon against all baselines (C01)."""
    cfg = tiny_config()
    w = StockWorld(cfg.d_model)
    raw = [w.batch(64, seed=i) for i in range(16)]
    H = (1, 2, 4, 8)
    wm = _wm(cfg, train_horizons=H, open_loop_samples=2, open_loop_starts=3)
    t0 = time.time()
    hist = train(wm, [batch_to(b, DEV) for b in raw], 450, lr=2e-3)
    wm.eval()
    test = [w.batch(64, seed=1000 + i) for i in range(2)]

    class OnDevice:
        def __init__(self, f):
            self.f = f

        def fit(self, b):
            return self

        def predict(self, b, s, hs):
            return self.f.predict(batch_to(b, DEV), s, hs)

    fc = {"world_model": OnDevice(WorldModelForecaster(wm, 8)), "no_change": NoChangeBaseline().fit(raw),
          "trend_persistence": TrendPersistenceBaseline().fit(raw), "key_mean": KeyMeanBaseline().fit(raw),
          "linear_regression": LinearRegressionBaseline(horizons=H).fit(raw)}
    res = evaluate_multistep(fc, test, H, starts=[1, 2, 3])
    table = {n: {h: round(v["mse"], 4) for h, v in d.items()} for n, d in res.items()}
    print(f"stock world: {time.time() - t0:.0f}s, loss {hist[0]:.2f}->{hist[-1]:.2f}, open-loop MSE by horizon "
          f"{table}")
    for h in H:
        assert res["world_model"][h]["n"] == res["no_change"][h]["n"] > 100
        for name in ("no_change", "trend_persistence", "key_mean", "linear_regression"):
            assert res["world_model"][h]["mse"] < res[name][h]["mse"], (h, name, table)
    # the model's own open-loop training errors are reported per horizon
    out = wm.loss(batch_to(test[0], DEV))
    assert set(out["horizon_errors"]) == set(H)


def test_predictions_follow_a_mechanism_change_and_a_changed_intermediate_event():
    """Spec 6.6(4): the effect of A flips sign at an unannounced time; once a changed outcome has been observed the
    filtered forecasts follow the new mechanism (and not before: histories up to the change are identical).
    Spec 6.6(6): from the same state, the intermediate event code of a switched vs an unswitched transition
    (posterior latents of the two real episodes) changes the later open-loop predictions accordingly."""
    cfg = tiny_config()
    w = SwitchWorld(cfg.d_model)
    batches = [batch_to(w.batch(64, seed=i), DEV) for i in range(16)]
    wm = _wm(cfg, train_horizons=(1, 2, 3, 4), open_loop_samples=2, open_loop_starts=4)
    t0 = time.time()
    train(wm, batches, 450, lr=4e-3)
    wm.eval()
    preds, eps = {}, {}
    for name, force in (("switch", {"s": 1.0, "sw": 4}), ("none", {"s": 1.0, "sw": None}),
                        ("neg_switch", {"s": -1.0, "sw": 4})):
        w.force = force
        eps[name] = tb = batch_to(w.batch(1, seed=5, plans=[[1] * w.T]), DEV)
        w.force = None
        preds[name] = torch.tensor([float(wm.forecast(tb, t, (1,), n_samples=16, generator=_gen())[1]["mean"][0, 0])
                                    for t in range(2, 9)])                    # prediction of Y_t from O_<=t
    print(f"switch world: {time.time() - t0:.0f}s, filtered predictions of Y_2..Y_8:",
          {k: [round(float(x), 2) for x in v] for k, v in preds.items()})
    sw, no, neg = preds["switch"], preds["none"], preds["neg_switch"]
    torch.testing.assert_close(sw[:3], no[:3])          # identical histories up to the change: no future leakage
    assert bool((sw[3:] < -0.7).all()) and bool((neg[3:] > 0.7).all())      # Y_5.. after observing the change
    assert bool((no[3:] > 0.2).all()) and bool((no[3:] - sw[3:] > 1.0).all())
    # 6.6(6): intervene on the intermediate event of the transition 4 -> 5
    later = {}
    with torch.no_grad():
        Z4 = wm.filter(SequenceBatch(eps["none"].steps[:5]))[-1]                 # the shared state before A_4
    for name in ("switch", "none"):
        tb = eps[name]
        with torch.no_grad():
            Zf = wm.filter(SequenceBatch(tb.steps[:6]))
            torch.testing.assert_close(Zf[4], Z4)
            # the event code: posterior latent of transition 4 -> 5 of this episode (sees O_5, i.e. Y_4)
            z = wm.sample_latent(wm.posterior_logits(Z4, tb.steps[4].act, None, 1.0, Zf[5]), sample=False)
            Z5 = wm.transition(Z4, tb.steps[4].act, None, torch.ones(1, device=DEV), z)
            ev = float(wm.outcome_dist(Z5, tb.steps[4].outcomes).mean()[0, 0])
            Z6, _ = wm.prior(Z5, tb.steps[5].act, None, 1.0, sample=False)
            later[name] = (ev, float(wm.outcome_dist(Z6, tb.steps[5].outcomes).mean()[0, 0]))
    print("intermediate event -> later prediction:", later)
    assert later["switch"][0] < -0.5 and later["none"][0] > 0.5                # the two intermediate events
    assert later["switch"][1] < -0.4 and later["none"][1] > 0.4 and later["none"][1] - later["switch"][1] > 1.0


# ------------------------------------------------------------------------------------------------ slow: planning

PROBE_SEEDS = (0, 1, 2)


def _train_probe_model(seed: int):
    cfg = tiny_config()
    w = ProbeWorld(cfg.d_model)
    bel = HypothesisBelief(cfg, w)
    batches = [batch_to(w.batch(128, seed=i + 1000 * seed), DEV) for i in range(24)]   # 3072 distinct episodes
    torch.manual_seed(seed)
    wm = WorldModel(cfg, DoubleCore(cfg), train_horizons=(1, 2, 3), open_loop_samples=4,
                    open_loop_starts=None).to(DEV)
    t0 = time.time()
    train(wm, batches, 1000, lr=2e-3, belief=bel, lr_end_frac=0.1, seed=seed)   # imagination-consistent open loop
    print(f"probe world (training seed {seed}) trained in {time.time() - t0:.0f}s")
    return cfg, w, bel, wm.eval()


@pytest.fixture(scope="module")
def probe_model():
    return _train_probe_model(PROBE_SEEDS[0])


@pytest.fixture(scope="module")
def probe_models(probe_model):
    return [probe_model] + [_train_probe_model(s) for s in PROBE_SEEDS[1:]]


@pytest.mark.slow
def test_planner_chooses_costly_experiment_that_enables_later_payoff(probe_models):
    """Spec 7.4: probe costs 1 now; only after it can the agent exploit (+10 if h=1) or fall back to safe (+0.5).
    The myopic choice is safe; the policy alone (uncertain belief) would wait; multistep belief-aware imagination
    picks the experiment, i.e. the imagined consequences change the real action. Checked for models from three
    training seeds, with a minimum probe-minus-safe margin (true values: probe -1 + e^-kappa * 5.25 ~ 4.0,
    safe 0.5)."""
    margins = []
    for seed, (cfg, w, bel, wm) in zip(PROBE_SEEDS, probe_models):
        Z, fast = real_start(wm, w, bel, 2, DEV)
        hooks, act_tokens = probe_hooks(w, bel, DEV)
        cand = probe_candidates(w, 2, DEV)
        im = Imagination(wm, horizon=2, outcome_sampling="marginal")
        res = PolicyRolloutPlanner(im, 4096, horizon=2).plan(Z, fast, None, cand, hooks, generator=_gen())
        myopic = PolicyRolloutPlanner(im, 4096, horizon=1).plan(Z, fast, None, cand, hooks, generator=_gen())
        traj = res.info["trajectory"]
        probe = traj.first_candidate == 1
        safe = traj.first_candidate == 2
        y = traj.steps[0].outcomes.y[:, 0]
        ex_pos = float(traj.steps[1].reward[probe & (y > 0.5)].mean())
        margin = float((res.scores[:, 1] - res.scores[:, 2]).min())
        margins.append(margin)
        print(f"seed {seed}: imagined value per candidate (wait, probe, safe, exploit):",
              [[round(float(x), 3) for x in r] for r in res.scores], "myopic:",
              [[round(float(x), 3) for x in r] for r in myopic.scores],
              f"probe-safe margin {margin:.2f}, E[r | exploit after imagined positive probe] {ex_pos:.2f}")
        assert res.choice.tolist() == [1, 1], (seed, res.scores)
        assert margin > 1.0, (seed, res.scores)
        assert all(a.tool == "probe" and a.source.value == "cam_planner" for a in res.actions)
        # one-step lookahead never picks the experiment: it only sees its immediate cost
        assert 1 not in myopic.choice.tolist() and bool((myopic.scores[:, 1] < -0.5).all())
        own, _ = hooks.policy_fn(PolicyContext(R=Z, Z=Z, goal=None, belief=bel.tokens(fast), fast=fast, step=0,
                                               prev_actions=[], branch_of=torch.arange(2), first_candidate=None))
        # ^ the policy alone, acting on the real (uninformed) belief
        assert {a.tool for a in own} == {"wait"}
        # end-to-end A08: alternatives rolled over several decisions with revised imagined beliefs
        nxt = [a.tool for a in traj.steps[1].action]
        assert {nxt[i] for i in torch.nonzero(probe & (y > 0.5)).flatten().tolist()} == {"exploit"}
        assert {nxt[i] for i in torch.nonzero(probe & (y < 0.5)).flatten().tolist()} == {"safe"}
        assert float(traj.steps[0].belief_ess[probe].mean()) < cfg.n_particles - 1      # beliefs revised
        assert float(traj.steps[0].reward[probe].mean()) < -0.5 < float(traj.steps[0].reward[safe].mean())
        assert ex_pos > 5.0                                    # exploit after a positive imagined probe pays
        assert traj.provenance == Provenance.IMAGINED
    print("probe-minus-safe margins over training seeds:", [round(m, 2) for m in margins])
    assert sum(margins) / len(margins) > 1.5


@pytest.fixture(scope="module")
def invest_model():
    cfg = tiny_config()
    w = InvestWorld(cfg.d_model)
    batches = [batch_to(w.batch(64, seed=i), DEV) for i in range(10)]
    wm = _wm(cfg, train_horizons=(1, 2, 3), open_loop_samples=2, open_loop_starts=None)
    train(wm, batches, 300, lr=2e-3)
    return cfg, w, wm.eval()


@pytest.mark.slow
def test_planner_comparison_at_matched_compute(probe_model, invest_model):
    """Beam vs CEM vs policy rollouts with the same transition budget. On InvestWorld (deterministic: invest -1,
    then sell +5) policy rollouts, CEM and a full-width beam find the investment, while a width-2 beam prunes it
    after its immediate cost. On ProbeWorld only the closed-loop policy rollouts value the experiment: open-loop
    plans must commit to the later action before seeing the result."""
    cfg, w, wm = invest_model
    B, n = 2, 3
    sb = batch_to(w.batch(B, seed=5, plans=[[0] * w.T] * B), DEV)
    Z = wm.filter(SequenceBatch(sb.steps[:1]))[-1]
    names = w.actions
    cand = Candidates([[Action(a) for a in names] for _ in range(B)], w.act_tokens(list(range(n)) * B).to(DEV))

    def policy(ctx):                     # sells after investing, otherwise idles
        ids = [2 if a.tool == "invest" else 0 for a in ctx.prev_actions]
        return [Action(names[i]) for i in ids], w.act_tokens(ids).to(DEV)

    from superlearner.cam_unified.world.imagination import ImaginationHooks
    from superlearner.cam_unified.world.planner import BeamPlanner
    im = Imagination(wm, horizon=2)
    hk = ImaginationHooks(policy_fn=policy)
    res = compare_planners(im, 144, Z, None, None, cand, hk, horizon=2)
    res["beam_full_width"] = BeamPlanner(im, 144, horizon=2, beam_width=n).plan(Z, None, None, cand, hk,
                                                                                 generator=_gen())
    report = {k: (r.choice.tolist(), [[round(float(x), 2) for x in s] for s in r.scores], r.n_model_evals)
              for k, r in res.items()}
    print("invest world planners:", report)
    for k, r in res.items():
        assert r.n_model_evals <= 144 * B
    for k in ("policy_rollout", "cem", "beam_full_width"):
        assert res[k].choice.tolist() == [1, 1], (k, report)
    # a narrow beam (width 2 of 3) prunes the immediately costly investment at depth 1
    assert not bool(torch.isfinite(res["beam"].scores[:, 1]).any()), report
    cfg, w, bel, wm = probe_model
    Z, fast = real_start(wm, w, bel, 2, DEV)
    hooks, _ = probe_hooks(w, bel, DEV)
    im = Imagination(wm, horizon=2, outcome_sampling="marginal")
    res = compare_planners(im, 2048, Z, fast, None, probe_candidates(w, 2, DEV), hooks, horizon=2)
    report = {k: (r.choice.tolist(), [[round(float(x), 2) if math.isfinite(float(x)) else None for x in s]
                                      for s in r.scores], r.n_model_evals) for k, r in res.items()}
    print("probe world planners:", report)
    evals = [r.n_model_evals for r in res.values()]
    assert max(evals) <= 2048 * 2 and min(evals) >= 2048 * 2 // 3
    assert res["policy_rollout"].choice.tolist() == [1, 1]
    cl = res["policy_rollout"].scores[:, 1]
    cem = res["cem"].scores[:, 1]
    assert bool((cl > cem + 0.25).all()), report      # open loop cannot value the information of the probe


def _imagination_policy_training(cfg, w, bel, wm, arm: str, seed: int, iters: int = 40, N: int = 48):
    """Train a first-decision policy pi(a0) = softmax(logits) for ProbeWorld from the real start state.

    arm "imagined": imagination actor-critic loss on imagined lambda-return targets only;
    arm "combined": the same imagined loss plus the real-experience term through ``combine_real_and_imagined``;
    arm "real_only": the real-experience term alone (no imagination);
    arm "no_belief_update": imagined targets from rollouts whose imagined probe results never revise the belief.
    The real-experience term is a value correction from actual ProbeWorld returns (4 real episodes per update,
    exact returns of the belief-driven follow-up policy): it corrects the critic V(s0) that is the imagined
    advantages' baseline but has no policy-gradient path, so it cannot by itself teach the policy to experiment."""
    Z, fast = real_start(wm, w, bel, 1, DEV)
    hooks, act_tokens = probe_hooks(w, bel, DEV, update=(arm != "no_belief_update"))
    im = Imagination(wm, horizon=2, outcome_sampling="marginal")
    logits = torch.zeros(4, device=DEV, requires_grad=True)
    v0 = torch.zeros((), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logits, v0], lr=0.1)
    g = torch.Generator(device=DEV).manual_seed(seed)
    rng = torch.Generator().manual_seed(100 + seed)

    def real_return(a, h):               # the actual ProbeWorld with the same belief-driven follow-up policy
        if a == 1:
            return -1.0 + math.exp(-cfg.kappa) * (10.0 if h == 1 else 0.5)
        return {0: 0.0, 2: 0.5, 3: 10.0 if h == 1 else -10.0}[a]

    imag = None
    for it in range(iters):
        pi = torch.softmax(logits, 0)
        imag = None
        if arm != "real_only":
            a0 = torch.multinomial(pi.detach(), N, replacement=True, generator=g)
            cand = Candidates([[Action(w.actions[i]) for i in a0.tolist()]], act_tokens(a0.tolist()))
            traj = im.rollout(Z, fast, None, cand, hooks, horizon=2, n_samples=1, generator=g)
            tg = imagined_targets(traj, lam=0.95, kappa=cfg.kappa, value0=v0.detach().expand(N),
                                  error_threshold=5.0)
            H = len(traj.steps)
            logp = torch.zeros(H, N, device=DEV)
            logp[0] = torch.log_softmax(logits, 0)[a0]
            vals = torch.zeros(H, N, device=DEV)
            vals[0] = v0.expand(N)
            mask = torch.zeros(H, N, device=DEV)
            mask[0] = 1.0
            imag = imagination_actor_critic_loss(tg, logp, vals, step_mask=mask, entropy_coef=0.0)
        real = None
        if arm in ("combined", "real_only"):
            ar = torch.multinomial(pi.detach().cpu(), 4, replacement=True, generator=rng)
            hr = torch.randint(0, 2, (4,), generator=rng)
            Rr = torch.tensor([real_return(int(a), int(h)) for a, h in zip(ar, hr)], device=DEV)
            real = {"loss": 0.5 * ((v0 - Rr) ** 2).mean(), "provenance": Provenance.REAL.value}
        if arm == "combined":
            loss = combine_real_and_imagined(real, imag, imagined_weight=1.0)["loss"]
        elif arm == "real_only":
            loss = combine_real_and_imagined(real, None)["loss"]
        else:
            loss = imag["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(logits, 0).tolist(), imag


@pytest.mark.slow
def test_policy_trained_on_imagined_targets_learns_to_experiment(probe_model):
    """C04: a first-decision policy trained with imagined lambda-return targets (uncertainty-weighted, flagged
    imagined) learns to probe -- with imagination alone and combined with a real-experience correction -- while
    the real-experience term alone leaves the policy where it started and imagination without belief revision
    does not learn to probe. Imagined-only updates through ``combine_real_and_imagined`` are refused."""
    cfg, w, bel, wm = probe_model
    res = {}
    for arm in ("imagined", "combined", "real_only", "no_belief_update"):
        for seed in (0, 1):
            pi, imag = _imagination_policy_training(cfg, w, bel, wm, arm, seed)
            res[(arm, seed)] = pi
            if imag is not None:
                assert imag["provenance"] == Provenance.IMAGINED.value
    print("policy after training (wait, probe, safe, exploit):",
          {f"{a}/{s}": [round(p, 3) for p in pi] for (a, s), pi in res.items()})
    for seed in (0, 1):
        for arm in ("imagined", "combined"):
            pi = res[(arm, seed)]
            assert pi[1] == max(pi) and pi[1] > 0.5, (arm, seed, pi)
        # the real term alone cannot move the policy: it has no policy-gradient path
        assert max(abs(p - 0.25) for p in res[("real_only", seed)]) < 1e-6, res[("real_only", seed)]
        # without imagined belief revision the experiment looks like a pure cost: the policy does not learn to
        # probe (it settles on whichever immediate action the model values most)
        pi = res[("no_belief_update", seed)]
        assert pi[1] < 0.15 and pi[1] != max(pi), (seed, pi)
    _, imag = _imagination_policy_training(cfg, w, bel, wm, "imagined", 0, iters=1)
    with pytest.raises(ValueError):
        combine_real_and_imagined(None, imag)
