"""Mechanism particles (A06): weighting concentrates on the data-generating hypothesis, ESS-triggered systematic
resampling, misspecification detection with uncertainty expansion, learned prior/proposal trained through the
support/query predictive log-likelihood."""
import math

import pytest
import torch

from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.beliefs import BeliefModule, detach_fast, systematic_resample
from superlearner.cam_unified.interfaces import FastState, Likelihood, OutcomeBatch, TokenKind, TokenSet
from superlearner.cam_unified.testing import DoubleOutcomeHead, rand_outcomes, rand_tokens, tiny_config


def known_head(cfg, sigma: float) -> DoubleOutcomeHead:
    """DoubleOutcomeHead wired as an exact model: y_q ~ N(m[0], sigma^2) for particle m (queries with zero tokens)."""
    h = DoubleOutcomeHead(cfg)
    D = cfg.d_model
    with torch.no_grad():
        for p in h.parameters():
            p.zero_()
        h.cond.weight[D + 0, 0] = 1.0                  # FiLM shift b[0] = m[0]
        h.cond.bias[D + 1] = math.log(sigma ** 2)      # b[1] = log variance
        h.out.weight[0, 0] = 1.0                       # mean = feat'[0]
        h.out.weight[1, 1] = 1.0                       # log variance = feat'[1]
    return h


def outcomes(y: torch.Tensor, D: int, mask=None) -> OutcomeBatch:
    """Gaussian outcome batch with zero query tokens; y [B,Q]."""
    B, Q = y.shape
    q = TokenSet(torch.zeros(B, Q, D), torch.ones(B, Q, dtype=torch.bool), torch.full((B, Q), int(TokenKind.QUERY)),
                 torch.zeros(B, Q), torch.full((B, Q), -1), None)
    return OutcomeBatch(q, torch.full((B, Q), int(Likelihood.GAUSSIAN)), torch.zeros(B, Q, dtype=torch.long), y,
                        torch.zeros(B, Q, dtype=torch.bool),
                        torch.ones(B, Q, dtype=torch.bool) if mask is None else mask,
                        torch.full((B, Q), -1), [[f"y{i}" for i in range(Q)] for _ in range(B)])


def two_hypotheses(bm: BeliefModule, B: int, a: float, b: float, scale: float = 0.05, n_a=None) -> FastState:
    fs = bm.init_fast(B)
    K = bm.K
    n_a = K // 2 if n_a is None else n_a
    p = torch.zeros(B, K, bm.Dp)
    p[:, :n_a, 0] = a
    p[:, n_a:, 0] = b
    return FastState(p, fs.logw, torch.full_like(p, scale), fs.bayes_mu, fs.bayes_P, fs.bayes_noise, fs.bayes_n,
                     fs.surprise, fs.steps, fs.channels)


def _setup(**over):
    cfg = tiny_config(**over)
    core = UnifiedCore(cfg)
    return cfg, core


def test_weights_concentrate_on_the_generating_hypothesis():
    torch.manual_seed(0)
    cfg, core = _setup()
    bm = BeliefModule(cfg, misspec_enabled=False)
    head = known_head(cfg, 0.5)
    B = 4
    fs = two_hypotheses(bm, B, +1.0, -1.0)
    Z = core.init_state(B).detach()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for t in range(12):
            y = 1.0 + 0.5 * torch.randn(B, 1, generator=g)
            fs, info = bm.update(fs, Z, None, outcomes(y, cfg.d_model), head, generator=g)
            assert info["loglik"].shape == (B, bm.K) and torch.isfinite(info["loglik"]).all()
    w = fs.logw.exp()
    mass_a = (w * (fs.particles[..., 0] > 0).float()).sum(-1)
    assert (mass_a > 0.99).all(), mass_a
    assert torch.allclose(fs.logw.exp().sum(-1), torch.ones(B), atol=1e-5)
    mm = bm.mixture_moments(fs, torch.zeros(B, 1, cfg.d_model), outcomes(torch.zeros(B, 1), cfg.d_model), head)
    assert torch.allclose(mm["mean"], torch.ones(B, 1), atol=0.15)
    assert (fs.steps == 12).all()


def test_single_update_is_exact_importance_weighting():
    cfg, core = _setup()
    bm = BeliefModule(cfg, resample_enabled=False, misspec_enabled=False)
    head = known_head(cfg, 0.7)
    B = 3
    fs = bm.init_fast(B)
    y = torch.tensor([[0.3, -0.2], [1.0, 1.5], [-2.0, 0.0]])
    with torch.no_grad():
        fs2, info = bm.update(fs, core.init_state(B), None, outcomes(y, cfg.d_model), head)
    m0 = fs.particles[..., 0]
    ll = torch.distributions.Normal(m0.unsqueeze(-1), 0.7).log_prob(y.unsqueeze(1)).sum(-1)
    assert torch.allclose(info["loglik"], ll, atol=1e-5)
    ref = torch.log_softmax(fs.logw + ll, -1)
    assert torch.allclose(fs2.logw, ref, atol=1e-5)
    assert torch.allclose(info["pred_ll"], torch.logsumexp(fs.logw + ll, -1), atol=1e-5)
    w = ref.exp()
    assert torch.allclose(info["ess"], 1 / (w * w).sum(-1), atol=1e-4)


def test_rows_without_observations_are_untouched():
    cfg, core = _setup()
    bm = BeliefModule(cfg)
    head = known_head(cfg, 0.5)
    fs = bm.init_fast(2)
    mask = torch.tensor([[True], [False]])
    with torch.no_grad():
        fs2, info = bm.update(fs, core.init_state(2), None, outcomes(torch.tensor([[3.0], [3.0]]), cfg.d_model, mask),
                              head)
    assert torch.equal(fs2.particles[1], fs.particles[1]) and torch.equal(fs2.logw[1], fs.logw[1])
    assert not torch.equal(fs2.particles[0], fs.particles[0])
    assert fs2.steps.tolist() == [1, 0] and info["surprise_z"][1] == 0


def test_systematic_resampling_counts():
    w = torch.tensor([[0.5, 0.25, 0.125, 0.125, 0, 0, 0, 0], [0.9, 0.1 / 7, 0.1 / 7, 0.1 / 7, 0.1 / 7, 0.1 / 7,
                                                          0.1 / 7, 0.1 / 7]])
    for seed in range(20):
        idx = systematic_resample(w, torch.Generator().manual_seed(seed))
        counts = torch.stack([torch.bincount(idx[b], minlength=8) for b in range(2)])
        expect = w * 8
        assert ((counts >= expect.floor() - 1e-6) & (counts <= expect.ceil() + 1e-6)).all(), (counts, expect)


def test_ess_collapse_triggers_resampling_and_rejuvenation():
    torch.manual_seed(0)
    cfg, core = _setup()
    bm = BeliefModule(cfg, misspec_enabled=False, rejuv=0.5)
    head = known_head(cfg, 0.2)
    B = 2
    fs = two_hypotheses(bm, B, +1.0, -1.0, scale=0.05, n_a=2)        # ESS after the update ~ 2 < 0.5 K
    Z = core.init_state(B).detach()
    with torch.no_grad():
        fs2, info = bm.update(fs, Z, None, outcomes(torch.tensor([[1.0], [1.0]]), cfg.d_model), head,
                              generator=torch.Generator().manual_seed(0))
    assert (info["ess"] < cfg.ess_frac * bm.K).all() and info["resampled"].all()
    assert torch.allclose(fs2.logw, torch.full_like(fs2.logw, -math.log(bm.K)))
    assert (fs2.particles[..., 0] > 0.5).all(), "every resampled particle descends from the supported hypothesis"
    assert fs2.particles[..., 0].std(-1).min() > 1e-3, "rejuvenation keeps resampled particles distinct"
    assert bm.counters["resampled_rows"] == 2
    # uninformative data: no resampling
    flat = known_head(cfg, 50.0)
    with torch.no_grad():
        fs3, info3 = bm.update(fs, Z, None, outcomes(torch.tensor([[1.0], [1.0]]), cfg.d_model), flat)
    assert not info3["resampled"].any() and (info3["ess"] > 0.99 * bm.K).all()


def _run(bm, head, fs, Z, ys, D, g):
    lls, infos = [], []
    with torch.no_grad():
        for y in ys:
            fs, info = bm.update(fs, Z, None, outcomes(y, D), head, generator=g)
            lls.append(info["pred_ll"])
            infos.append(info)
    return fs, torch.stack(lls), infos


@pytest.mark.parametrize("stat", ["rank", "moment"])
def test_misspecification_detected_and_uncertainty_expanded(stat):
    """Data from neither hypothesis (+-1, sharp) but from N(4, 0.3^2): the surprise statistic crosses misspec_z, fresh
    particles are injected and scales inflated; afterwards the belief explains the data far better than the same
    filter without expansion."""
    cfg, core = _setup(n_particles=16)
    head = known_head(cfg, 0.3)
    B = 4
    Z = core.init_state(B).detach()
    gy = torch.Generator().manual_seed(2)
    ys = [4.0 + 0.3 * torch.randn(B, 1, generator=gy) for _ in range(40)]
    out = {}
    for enabled in (True, False):
        torch.manual_seed(0)
        bm = BeliefModule(cfg, misspec_enabled=enabled, inflate=3.0, surprise_stat=stat)
        fs0 = two_hypotheses(bm, B, +1.0, -1.0, scale=0.05)
        fs, lls, infos = _run(bm, head, fs0, Z, ys, cfg.d_model, torch.Generator().manual_seed(3))
        out[enabled] = (bm, fs, lls, infos)
    bm, fs, lls, infos = out[True]
    exp_steps = [t for t, i in enumerate(infos) if i["expanded"].any()]
    assert exp_steps and exp_steps[0] <= 6, exp_steps
    first = infos[exp_steps[0]]
    assert (first["cusum"][first["expanded"]] > bm.cusum_h).all() and first["alarm"][first["expanded"]].all()
    assert (first["window_z"][first["expanded"]] > cfg.misspec_z).all()
    n_exp_log = sum(r["kind"] == "expansion" for r in bm.misspec_log)
    assert bm.counters["expanded_rows"] >= B and n_exp_log == bm.counters["expanded_rows"]
    assert bm.misspec_log[0]["cusum"] > bm.cusum_h and bm.misspec_log[0]["window_z"] > cfg.misspec_z
    late_on = out[True][2][-10:].mean()
    late_off = out[False][2][-10:].mean()
    assert late_on > late_off + 5.0, (late_on, late_off)
    assert late_on > -2.0, "after expansion the belief fits the new mechanism"
    assert out[False][0].counters["expanded_rows"] == 0


def test_no_false_alarm_when_a_hypothesis_is_correct():
    cfg, core = _setup(n_particles=16)
    head = known_head(cfg, 0.5)
    B = 8
    torch.manual_seed(0)
    bm = BeliefModule(cfg)
    fs0 = two_hypotheses(bm, B, +1.0, -1.0, scale=0.02)
    gy = torch.Generator().manual_seed(5)
    ys = [1.0 + 0.5 * torch.randn(B, 1, generator=gy) for _ in range(60)]
    fs, lls, infos = _run(bm, head, fs0, core.init_state(B).detach(), ys, cfg.d_model,
                          torch.Generator().manual_seed(6))
    zs = torch.stack([i["surprise_z"] for i in infos])
    wz = torch.stack([i["window_z"] for i in infos])
    print(f"null surprise z: mean {float(zs.mean()):.3f} sd {float(zs.std()):.3f}; max window z {float(wz.max()):.2f}")
    assert bm.counters["expanded_rows"] == 0
    assert abs(float(zs.mean())) < 0.25 and 0.75 < float(zs.std()) < 1.25, (zs.mean(), zs.std())
    h = torch.stack([i["pred_entropy"] for i in infos[20:]])
    true_h = 0.5 * math.log(2 * math.pi * math.e * 0.25)
    assert abs(float(h.mean()) - true_h) < 0.1, "predicted entropy of the concentrated belief = N(1, 0.5^2) entropy"


def test_belief_tokens_are_a_particle_set_with_summary():
    cfg, core = _setup()
    bm = BeliefModule(cfg)
    fs = bm.init_fast(3)
    fs.logw = torch.log_softmax(torch.randn(3, bm.K), -1)
    tk = bm.tokens(fs)
    assert tk.x.shape == (3, bm.K + 1, cfg.d_model) and (tk.kind == TokenKind.BELIEF).all() and tk.mask.all()
    perm = torch.randperm(bm.K)
    fsp = FastState(fs.particles[:, perm], fs.logw[:, perm], fs.pscale[:, perm], fs.bayes_mu, fs.bayes_P,
                    fs.bayes_noise, fs.bayes_n, fs.surprise, fs.steps, fs.channels)
    tkp = bm.tokens(fsp)
    assert torch.allclose(tk.x[:, perm], tkp.x[:, :bm.K], atol=1e-5)
    assert torch.allclose(tk.x[:, -1], tkp.x[:, -1], atol=1e-5), "summary is permutation invariant"
    head = known_head(cfg, 0.5)
    with torch.no_grad():
        fs2, _ = bm.update(fs, core.init_state(3), None, outcomes(torch.ones(3, 1), cfg.d_model), head)
    assert (bm.tokens(fs2).x - tk.x).abs().max() > 1e-3, "the belief update is visible to the core"
    Z, _ = core(core.init_state(3), tk)
    assert torch.isfinite(Z).all()


def test_action_conditioning_and_any_outcome_head():
    """With the generic (untrained) double head and mixed likelihood outcomes the update runs, depends on the action
    tokens, and is differentiable into the action-conditioning block and the head."""
    torch.manual_seed(0)
    cfg, core = _setup()
    bm = BeliefModule(cfg)
    head = DoubleOutcomeHead(cfg)
    B = 3
    fs = bm.init_fast(B)
    Z = core.init_state(B)
    oc = rand_outcomes(B, 6, cfg.d_model, seed=1)
    act = rand_tokens(B, 2, cfg.d_model, TokenKind.ACTION, ragged=False, seed=2)
    with torch.no_grad():                       # make the action path non-trivial
        bm.act_cond.attn.o.weight.normal_(std=0.05)
    fs1, i1 = bm.update(fs, Z, act, oc, head, generator=torch.Generator().manual_seed(0))
    act2 = rand_tokens(B, 2, cfg.d_model, TokenKind.ACTION, ragged=False, seed=9)
    _, i2 = bm.update(fs, Z, act2, oc, head, generator=torch.Generator().manual_seed(0))
    assert (i1["loglik"] - i2["loglik"]).abs().max() > 1e-4
    (-i1["pred_ll"].mean() + bm.tokens(fs1).x.pow(2).mean()).backward()
    assert bm.act_cond.attn.o.weight.grad.abs().max() > 0 and head.out.weight.grad.abs().max() > 0


def test_learned_prior_and_proposal_train_through_query_likelihood():
    """Support/query adaptation: tasks y ~ N(theta, 0.3^2), theta ~ U(-8, 8) (far outside the initial prior N(0, 1)).
    Training the prior, proposal and event summary on the query predictive log-likelihood after 4 support updates
    improves the held-out query log-likelihood by a large margin."""
    torch.manual_seed(0)
    cfg, core = _setup()
    bm = BeliefModule(cfg, n_mc=2, misspec_enabled=False)
    head = known_head(cfg, 0.3)
    for p in head.parameters():
        p.requires_grad_(False)
    D = cfg.d_model
    Zs = core.init_state(32).detach()

    def episode(B, g):
        theta = torch.rand(B, 1, generator=g) * 16 - 8
        fs = bm.init_fast(B, generator=g)
        Z = Zs[:B]
        for _ in range(4):
            y = theta + 0.3 * torch.randn(B, 1, generator=g)
            fs, info = bm.update(fs, Z, None, outcomes(y, D), head, generator=g)
        yq = theta + 0.3 * torch.randn(B, 4, generator=g)
        return bm.predictive_log_prob(fs, torch.zeros(B, 4, D), outcomes(yq, D), head).mean()

    def evaluate():
        with torch.no_grad():
            return float(sum(episode(32, torch.Generator().manual_seed(1000 + i)) for i in range(4)) / 4)
    def pull_toward(y0):
        with torch.no_grad():
            fs = bm.init_fast(32, generator=torch.Generator().manual_seed(7))
            fs1, _ = bm.update(fs, Zs, None, outcomes(torch.full((32, 1), y0), D), head,
                               generator=torch.Generator().manual_seed(8))
        return float((fs1.logw.exp() * fs1.particles[..., 0]).sum(-1).mean())
    before = evaluate()
    pull_before = pull_toward(6.0)
    opt = torch.optim.Adam([p for p in bm.parameters() if p.requires_grad], lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for step in range(150):
        loss = -episode(32, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = evaluate()
    print(f"query log-lik per 4 points: before {before:.2f} after {after:.2f}")
    assert after > before + 30.0 and after > -12.0, (before, after)
    # the learned proposal moves the hypotheses toward the evidence: one observation y = 6 pulls the particles there
    pull_after = pull_toward(6.0)
    print(f"weighted particle mean after one observation at 6.0: untrained {pull_before:.2f} trained {pull_after:.2f}")
    assert pull_after > pull_before + 1.0, (pull_before, pull_after)


def test_detach_fast_cuts_the_graph():
    cfg, core = _setup()
    bm = BeliefModule(cfg)
    fs = bm.init_fast(2)
    assert fs.particles.requires_grad
    d = detach_fast(fs)
    assert not d.particles.requires_grad and torch.equal(d.particles, fs.particles)
    Zc = torch.randn(2, cfg.n_state, cfg.d_model)
    with torch.no_grad():
        bm.prior_cond[1].weight.normal_()
    fz = bm.init_fast(2, Z=Zc)
    mu, _ = bm.prior(2, Zc)
    assert (mu[0] - mu[1]).abs().max() > 1e-3, "the prior is conditioned on Z"


# ---------------------------------------------------------------------------- sequential test: false alarms, backoff


def test_cusum_threshold_gives_the_target_average_run_length():
    """h from Siegmund's approximation gives the requested in-control average run length for N(0,1) increments
    (window long enough that the windowed CUSUM equals the full one)."""
    cfg, core = _setup(misspec_window=96)
    bm = BeliefModule(cfg, misspec_arl=200.0)
    g = torch.Generator().manual_seed(0)
    n, T, W = 4000, 3000, cfg.misspec_window
    win = torch.full((n, W), float("nan"))
    first = torch.full((n,), T, dtype=torch.long)
    for t in range(T):
        win = torch.cat([win[:, 1:], torch.randn(n, 1, generator=g)], 1)
        s, _ = bm.cusum_stat(win)
        new = (s > bm.cusum_h) & (first == T)
        first[new] = t + 1
    arl = float(first.float().mean())
    print(f"CUSUM k={bm.cusum_k:.2f} h={bm.cusum_h:.2f}: simulated ARL0 {arl:.0f} (target 200)")
    assert (first < T).float().mean() > 0.99
    assert 160 < arl < 250, arl


def test_long_horizon_false_alarm_rate_under_a_correct_hypothesis():
    """96k row-steps of data from a correct hypothesis: the CUSUM detector stays within a false-alarm budget of
    0.05 per 1000 row-steps (the former window rule, sum(z)/sqrt(n) > misspec_z, fires ~0.4 per 1000 on the same
    z stream)."""
    cfg, core = _setup(n_particles=16)
    head = known_head(cfg, 0.5)
    B, T = 96, 1000
    torch.manual_seed(0)
    bm = BeliefModule(cfg)
    fs = two_hypotheses(bm, B, +1.0, -1.0, scale=0.02)
    Z = core.init_state(B).detach()
    gy, g = torch.Generator().manual_seed(5), torch.Generator().manual_seed(6)
    old_rule = 0
    zs = []
    with torch.no_grad():
        for t in range(T):
            y = 1.0 + 0.5 * torch.randn(B, 1, generator=gy)
            fs, info = bm.update(fs, Z, None, outcomes(y, cfg.d_model), head, generator=g)
            old_rule += int(((info["window_z"] > cfg.misspec_z) & (t >= bm.min_steps - 1)).sum())
            zs.append(info["surprise_z"])
    zs = torch.stack(zs)
    rate = 1000.0 * bm.counters["alarm_rows"] / (B * T)
    print(f"null {B * T} row-steps: CUSUM alarms {bm.counters['alarm_rows']} ({rate:.3f}/1000), expansions "
          f"{bm.counters['expanded_rows']}; old window rule would fire {old_rule} ({1000.0 * old_rule / (B * T):.3f}"
          f"/1000); z mean {float(zs.mean()):.3f} sd {float(zs.std()):.3f}")
    assert rate <= 0.05, rate
    assert old_rule > 10 * max(1, bm.counters["alarm_rows"])
    assert abs(float(zs.mean())) < 0.05 and 0.9 < float(zs.std()) < 1.1


def test_persistent_misspecification_backs_off_and_caps_uncertainty():
    """Data no particle can represent (outcome noise 30, head noise 0.3): a few expansions, then the row is flagged
    as persistently misspecified — no further weight resets, Sigma pinned at max_scale_mult x prior scale, particles
    bounded and finite."""
    cfg, core = _setup(n_particles=16)
    head = known_head(cfg, 0.3)
    B, T = 2, 400
    torch.manual_seed(0)
    bm = BeliefModule(cfg)
    fs = two_hypotheses(bm, B, +1.0, -1.0, scale=0.05)
    Z = core.init_state(B).detach()
    gy, g = torch.Generator().manual_seed(2), torch.Generator().manual_seed(3)
    exp_t, pers_t, pmax = [], [], []
    with torch.no_grad():
        for t in range(T):
            y = 30.0 * torch.randn(B, 1, generator=gy)
            fs, info = bm.update(fs, Z, None, outcomes(y, cfg.d_model), head, generator=g)
            exp_t.append(info["expanded"].clone())
            pers_t.append(info["persistent"].clone())
            pmax.append(float(fs.particles.abs().max()))
    E, P = torch.stack(exp_t), torch.stack(pers_t)
    _, sc0 = bm.prior(B, Z)
    cap = bm.max_scale_mult * float(sc0.detach().max())
    print(f"persistent misspecification over {T} steps x {B} rows: expansions {int(E.sum())}, persistent row-steps "
          f"{int(P.sum())}, episodes {bm.counters['persistent_episodes']}; max pscale {float(fs.pscale.max()):.2f} "
          f"(cap {cap:.2f}); max |particle| {max(pmax):.1f}")
    assert int(E.sum()) <= 10 * B, int(E.sum())
    assert P.any(0).all(), "every row is flagged as persistently misspecified"
    for b in range(B):
        t0 = int(P[:, b].float().argmax())
        assert not E[t0:, b].any(), "no expansions (weight resets) after the row is flagged persistent"
    assert float(fs.pscale.max()) <= cap * (1 + 1e-5)
    assert torch.isfinite(fs.particles).all() and max(pmax) < 1e3
    assert any(r["kind"] == "persistent" for r in bm.misspec_log)
    assert 1 <= bm.counters["persistent_episodes"] <= 4 * B, bm.counters["persistent_episodes"]
    # the window rule without backoff/cap expanded every min_steps steps here (199 expansions, |m| ~ 2e4)


class _ExpSurvival:
    """Exponential event-time distribution with right censoring (test double of a SURVIVAL OutcomeDist)."""

    def __init__(self, rate):
        self.rate = rate

    def log_prob(self, y, censored=None):
        lp = torch.log(self.rate) - self.rate * y
        return lp if censored is None else torch.where(censored, -self.rate * y, lp)

    def sample(self, generator=None):
        return torch.distributions.Exponential(self.rate).sample()

    @property
    def mean(self):
        return 1.0 / self.rate

    @property
    def variance(self):
        return 1.0 / self.rate ** 2


class _ExpSurvivalHead(torch.nn.Module):
    def features(self, Z, outcomes):
        return outcomes.q.x

    def dist(self, feat, outcomes, cond=None):
        B, Q, _ = feat.shape
        K = cond.shape[1]
        return _ExpSurvival(torch.exp(cond[..., :1]).expand(B, K, Q))


def test_censored_survival_surprise_is_calibrated():
    """Right-censored event times from the correct hypothesis (rate 1, censoring at 1 day: ~37% censored): with the
    censored observation process simulated in the Monte Carlo draws the rank z is N(0,1); comparing against
    uncensored draws (the former computation) is biased."""
    cfg, core = _setup(n_particles=8)
    torch.manual_seed(0)
    bm = BeliefModule(cfg, misspec_enabled=False, resample_enabled=False)
    head = _ExpSurvivalHead()
    B, T = 64, 60
    fs = bm.init_fast(B)
    fs = FastState(torch.zeros_like(fs.particles), fs.logw, torch.full_like(fs.pscale, 1e-3), fs.bayes_mu, fs.bayes_P,
                   fs.bayes_noise, fs.bayes_n, fs.surprise, fs.steps, fs.channels)
    with torch.no_grad():
        bm.prop[2].weight.zero_()
    bm.jitter = 0.0
    Z = core.init_state(B).detach()
    g = torch.Generator().manual_seed(1)
    z_fix, z_old, frac_cen = [], [], []
    with torch.no_grad():
        for _ in range(T):
            t_ev = torch.distributions.Exponential(torch.ones(B, 1)).sample()
            cen = t_ev > 1.0
            y = torch.where(cen, torch.ones_like(t_ev), t_ev)
            ob = outcomes(y, cfg.d_model)
            ob = OutcomeBatch(ob.q, torch.full((B, 1), int(Likelihood.SURVIVAL)), ob.n_classes, y, cen, ob.y_mask,
                              ob.channel, ob.keys)
            feat = head.features(Z, ob)
            ll, dist = bm.particle_loglik(fs, feat, ob, head)
            pred = torch.logsumexp(fs.logw + ll, -1)
            z_fix.append(bm.surprise_z(dist, fs.logw, ob.y_mask, pred, g, y_obs=y, censored=cen, lik=ob.lik)[0])
            z_old.append(bm.surprise_z(dist, fs.logw, ob.y_mask, pred, g)[0])
            fs, info = bm.update(fs, Z, None, ob, head, generator=g)
            frac_cen.append(float(cen.float().mean()))
    zf, zo = torch.cat(z_fix), torch.cat(z_old)
    print(f"censored survival ({sum(frac_cen) / T:.2f} censored): z with censored draws mean {float(zf.mean()):.3f} "
          f"sd {float(zf.std()):.3f}; with uncensored draws mean {float(zo.mean()):.3f} sd {float(zo.std()):.3f}")
    assert abs(float(zf.mean())) < 0.1 and 0.85 < float(zf.std()) < 1.15
    assert abs(float(zo.mean())) > 0.25 or abs(float(zo.std()) - 1) > 0.25
