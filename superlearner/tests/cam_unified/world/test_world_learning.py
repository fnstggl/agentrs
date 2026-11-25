"""Learning tests on synthetic sequence worlds (spec 6.6 / 7.4).

Default run: the world model learns (a) a delayed vs an immediate effect, (b) a non-additive interaction of two
simultaneous interventions, (c) open-loop multistep predictions (irregular elapsed times, nonlinear dynamics) that
beat no-change / trend-persistence / key-mean / per-key linear-regression baselines at every horizon.
Slow (CAM_RUN_SLOW=1): on a learned ProbeWorld model, the closed-loop planner picks the immediately costly
experiment that enables the later payoff, imagination changes the real action, the three planners are compared at
matched compute, and a policy trained on imagined targets learns to experiment.
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
                         batch_to, device, probe_candidates, probe_hooks, real_start, train)

DEV = device()


def _gen():
    return torch.Generator(device=DEV).manual_seed(0)


def _wm(cfg, **kw):
    torch.manual_seed(0)
    return WorldModel(cfg, DoubleCore(cfg), **kw).to(DEV)


def test_learns_delayed_versus_immediate_effects():
    """A has an immediate unit effect on y, B the same effect two decisions later (spec 6.6(1),(2))."""
    cfg = tiny_config()
    w = DelayWorld(cfg.d_model)
    batches = [batch_to(w.batch(64, seed=i), DEV) for i in range(16)]
    wm = _wm(cfg, train_horizons=(1, 2, 3, 4), open_loop_samples=2, open_loop_starts=4)
    t0 = time.time()
    hist = train(wm, batches, 260, lr=2e-3)
    wm.eval()
    plans = [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    tb = batch_to(w.batch(3, seed=999, plans=plans), DEV)
    f = wm.forecast(tb, 2, (1, 2, 3), n_samples=8, generator=_gen())
    pred = torch.stack([f[h]["mean"][:, 0] for h in (1, 2, 3)], 1).cpu()        # [plan, horizon]
    truth = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0, 0, 0]])
    print(f"delay world: {time.time() - t0:.0f}s, loss {hist[0]:.2f}->{hist[-1]:.2f}, predictions\n{pred}")
    assert float((pred - truth).abs().max()) < 0.3, pred
    # A07 (learned): from the same filtered state, contemplated A vs noop predict different consequences
    Z = wm.filter(SequenceBatch(tb.steps[:3]))[-1][2:3]
    ob = tb.steps[2].outcomes
    q = type(ob)(ob.q.select([0]), ob.lik[:1], ob.n_classes[:1], ob.y[:1], ob.censored[:1], ob.y_mask[:1],
                 ob.channel[:1])
    yA = wm.outcome_dist(wm.prior(Z, w.act_tokens([1]).to(DEV), None, 1.0, sample=False)[0], q).mean()
    y0 = wm.outcome_dist(wm.prior(Z, w.act_tokens([0]).to(DEV), None, 1.0, sample=False)[0], q).mean()
    assert float(yA - y0) > 0.7


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
    hist = train(wm, [batch_to(b, DEV) for b in raw], 260, lr=2e-3)
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


# ------------------------------------------------------------------------------------------------ slow: planning

@pytest.fixture(scope="module")
def probe_model():
    cfg = tiny_config()
    w = ProbeWorld(cfg.d_model)
    bel = HypothesisBelief(cfg, w)
    batches = [batch_to(w.batch(128, seed=i), DEV) for i in range(24)]       # 3072 distinct episodes
    wm = _wm(cfg, train_horizons=(1, 2, 3), open_loop_samples=4, open_loop_starts=None)
    t0 = time.time()
    train(wm, batches, 600, lr=2e-3, belief=bel, lr_end_frac=0.1)   # imagination-consistent open loop
    print(f"probe world trained in {time.time() - t0:.0f}s")
    return cfg, w, bel, wm.eval()


@pytest.mark.slow
def test_planner_chooses_costly_experiment_that_enables_later_payoff(probe_model):
    """Spec 7.4: probe costs 1 now; only after it can the agent exploit (+10 if h=1) or fall back to safe (+0.5).
    The myopic choice is safe; the policy alone (uncertain belief) would wait; multistep belief-aware imagination
    picks the experiment, i.e. the imagined consequences change the real action."""
    cfg, w, bel, wm = probe_model
    Z, fast = real_start(wm, w, bel, 2, DEV)
    hooks, act_tokens = probe_hooks(w, bel, DEV)
    cand = probe_candidates(w, 2, DEV)
    im = Imagination(wm, horizon=2, outcome_sampling="marginal")
    res = PolicyRolloutPlanner(im, 4096, horizon=2).plan(Z, fast, None, cand, hooks, generator=_gen())
    myopic = PolicyRolloutPlanner(im, 4096, horizon=1).plan(Z, fast, None, cand, hooks, generator=_gen())
    print("imagined value per candidate (wait, probe, safe, exploit):", res.scores.tolist(),
          "myopic:", myopic.scores.tolist())
    assert res.choice.tolist() == [1, 1], res.scores
    assert all(a.tool == "probe" and a.source.value == "cam_planner" for a in res.actions)
    # one-step lookahead never picks the experiment: it only sees its immediate cost
    assert 1 not in myopic.choice.tolist() and bool((myopic.scores[:, 1] < -0.5).all())
    own, _ = hooks.policy_fn(PolicyContext(R=Z, Z=Z, goal=None, belief=bel.tokens(fast), fast=fast, step=0,
                                           prev_actions=[], branch_of=torch.arange(2), first_candidate=None))
    # ^ the policy alone, acting on the real (uninformed) belief
    assert {a.tool for a in own} == {"wait"}
    # end-to-end A08: alternatives rolled over several decisions with revised imagined beliefs
    traj = res.info["trajectory"]
    probe = traj.first_candidate == 1
    safe = traj.first_candidate == 2
    y = traj.steps[0].outcomes.y[:, 0]
    nxt = [a.tool for a in traj.steps[1].action]
    assert {nxt[i] for i in torch.nonzero(probe & (y > 0.5)).flatten().tolist()} == {"exploit"}
    assert {nxt[i] for i in torch.nonzero(probe & (y < 0.5)).flatten().tolist()} == {"safe"}
    assert float(traj.steps[0].belief_ess[probe].mean()) < cfg.n_particles - 1      # beliefs revised
    assert float(traj.steps[0].reward[probe].mean()) < -0.5 < float(traj.steps[0].reward[safe].mean())
    ex_pos = traj.steps[1].reward[probe & (y > 0.5)].mean()
    assert float(ex_pos) > 3.0                                     # exploit after a positive imagined probe pays
    assert traj.provenance == Provenance.IMAGINED


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


@pytest.mark.slow
def test_policy_trained_on_imagined_targets_learns_to_experiment(probe_model):
    """C04: a first-decision policy trained with imagined lambda-return targets (uncertainty-weighted, flagged
    imagined) plus a small real-experience term learns to probe; imagined-only updates are refused."""
    cfg, w, bel, wm = probe_model
    Z, fast = real_start(wm, w, bel, 1, DEV)
    hooks, act_tokens = probe_hooks(w, bel, DEV)
    im = Imagination(wm, horizon=2, outcome_sampling="marginal")
    logits = torch.zeros(4, device=DEV, requires_grad=True)
    v0 = torch.zeros((), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logits, v0], lr=0.1)
    g = torch.Generator(device=DEV).manual_seed(0)
    rng = torch.Generator().manual_seed(1)
    N = 48

    def real_return(a, h):               # the actual ProbeWorld with the same belief-driven follow-up policy
        if a == 1:
            return -1.0 + math.exp(-cfg.kappa) * (10.0 if h == 1 else 0.5)
        return {0: 0.0, 2: 0.5, 3: 10.0 if h == 1 else -10.0}[a]

    for it in range(40):
        pi = torch.softmax(logits, 0)
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
        # a little real experience every update
        ar = torch.multinomial(pi.detach(), 4, replacement=True, generator=g).cpu()
        hr = torch.randint(0, 2, (4,), generator=rng)
        Rr = torch.tensor([real_return(int(a), int(h)) for a, h in zip(ar, hr)], device=DEV)
        lp_r = torch.log_softmax(logits, 0)[ar.to(DEV)]
        real = {"loss": -((Rr - v0.detach()) * lp_r).mean() + 0.5 * ((v0 - Rr) ** 2).mean()}
        tot = combine_real_and_imagined(real, imag, imagined_weight=1.0)
        opt.zero_grad()
        tot["loss"].backward()
        opt.step()
    pi = torch.softmax(logits, 0).tolist()
    print("policy after imagination training (wait, probe, safe, exploit):", [round(p, 3) for p in pi])
    assert pi[1] == max(pi) and pi[1] > 0.5
    with pytest.raises(ValueError):
        combine_real_and_imagined(None, imag)
