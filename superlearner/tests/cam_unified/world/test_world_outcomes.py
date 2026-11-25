"""OutcomeHead / MixedOutcomeDist: every likelihood against torch.distributions references, censoring, masks,
particle FiLM conditioning, sampling moments and learning."""
import math

import pytest
import torch
import torch.distributions as td

from superlearner.cam_unified.interfaces import Likelihood, OutcomeBatch, TokenKind, TokenSet
from superlearner.cam_unified.testing import rand_outcomes, rand_tokens, tiny_config
from superlearner.cam_unified.world.outcomes import MixedOutcomeDist, OutcomeHead, default_survival_edges

L = Likelihood


def _dist(B=4, Q=12, seed=0, Cm=8, scale=1.0, K=None):
    g = torch.Generator().manual_seed(seed)
    edges = default_survival_edges(8, 0.5, 30.0)
    P = 2 + 3 + 1 + Cm + 2 + edges.numel()
    shape = (B, Q, P) if K is None else (B, K, Q, P)
    raw = torch.randn(*shape, generator=g) * scale
    lik = torch.tensor([i % 6 for i in range(Q)]).repeat(B, 1)
    ncls = torch.where(lik == L.CATEGORICAL, torch.tensor(5), torch.tensor(0))
    if K is not None:
        lik, ncls = lik.unsqueeze(1), ncls.unsqueeze(1)
    return MixedOutcomeDist(raw, lik, ncls, edges, Cm), lik.expand(shape[:-1]), edges


def _targets(lik, seed=1):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(lik.shape, generator=g) * 2
    y = torch.where(lik == L.BERNOULLI, (y > 0).float(), y)
    y = torch.where(lik == L.CATEGORICAL, torch.randint(0, 5, lik.shape, generator=g).float(), y)
    y = torch.where(lik == L.COUNT, torch.randint(0, 30, lik.shape, generator=g).float(), y)
    y = torch.where(lik == L.SURVIVAL, torch.rand(lik.shape, generator=g) * 40, y)
    return y


def test_log_prob_matches_torch_distributions_for_every_likelihood():
    d, lik, edges = _dist()
    y = _targets(lik)
    lp = d.log_prob(y)
    sel = lambda k: lik == k
    m = sel(L.GAUSSIAN)
    ref = td.Normal(d.g_mu[m], d.g_std[m]).log_prob(y[m])
    torch.testing.assert_close(lp[m], ref, rtol=1e-5, atol=1e-5)
    m = sel(L.STUDENT_T)
    ref = td.StudentT(d.t_df[m], d.t_loc[m], d.t_scale[m]).log_prob(y[m])
    torch.testing.assert_close(lp[m], ref, rtol=1e-5, atol=1e-5)
    m = sel(L.BERNOULLI)
    ref = td.Bernoulli(logits=d.b_logit[m]).log_prob(y[m])
    torch.testing.assert_close(lp[m], ref, rtol=1e-5, atol=1e-5)
    m = sel(L.CATEGORICAL)
    ref = td.Categorical(logits=d.raw[..., 6:6 + 8][m][:, :5]).log_prob(y[m].long())   # classes >= 5 masked
    torch.testing.assert_close(lp[m], ref, rtol=1e-5, atol=1e-5)
    m = sel(L.COUNT)
    r, mu = d.nb_total_count[m], d.nb_mean[m]
    ref = td.NegativeBinomial(total_count=r, logits=torch.log(mu) - torch.log(r)).log_prob(y[m])
    torch.testing.assert_close(lp[m], ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(td.NegativeBinomial(total_count=r, logits=torch.log(mu) - torch.log(r)).mean, mu, rtol=1e-4)
    m = sel(L.SURVIVAL)
    pmf = d.surv_bin_probs[m]
    torch.testing.assert_close(pmf.sum(-1), torch.ones(pmf.shape[0]), rtol=1e-5, atol=1e-5)
    j = (torch.searchsorted(edges, y[m], right=True) - 1).clamp(0, edges.numel() - 1)
    ref = td.Categorical(probs=pmf).log_prob(j)
    torch.testing.assert_close(lp[m], ref, rtol=1e-4, atol=1e-5)


def test_survival_censoring_is_the_survival_mass():
    d, lik, edges = _dist(seed=3)
    y = _targets(lik, seed=4)
    cens = torch.ones_like(lik, dtype=torch.bool)
    lp_c = d.log_prob(y, censored=cens)
    lp_e = d.log_prob(y, censored=~cens)
    m = lik == L.SURVIVAL
    pmf = d.surv_bin_probs[m]
    j = (torch.searchsorted(edges, y[m], right=True) - 1).clamp(0, edges.numel() - 1)
    # censored at c in bin j: P(T >= e_j) = sum_{i >= j} P_i
    tail = torch.stack([pmf[n, j[n]:].sum() for n in range(pmf.shape[0])])
    torch.testing.assert_close(lp_c[m].exp(), tail, rtol=1e-4, atol=1e-6)
    assert bool((lp_c[m] >= lp_e[m] - 1e-6).all())          # surviving through the start >= event in the bin
    # censored in bin 0 carries no information; censoring later costs survival mass
    zero = d.log_prob(torch.zeros_like(y), censored=cens)[m]
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-6, rtol=0)
    # censoring does not change other likelihood families
    torch.testing.assert_close(lp_c[~m], lp_e[~m])
    # raising an early hazard lowers the likelihood of being censored later
    raw = d.raw.clone()
    s0 = 2 + 3 + 1 + 8 + 2
    raw[..., s0 + 0] += 3.0
    d2 = MixedOutcomeDist(raw, d.lik, d.n_classes, edges, 8)
    late = torch.full_like(y, 20.0)            # a late censoring time (only the survival entries are observed)
    assert bool((d2.log_prob(late, censored=cens, mask=m)[m] < d.log_prob(late, censored=cens, mask=m)[m]).all())


def test_mean_variance_entropy_match_references():
    d, lik, edges = _dist(seed=5)
    mean, var, ent = d.mean(), d.variance(), d.entropy()
    m = lik == L.GAUSSIAN
    ref = td.Normal(d.g_mu[m], d.g_std[m])
    torch.testing.assert_close(ent[m], ref.entropy(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(var[m], ref.variance)
    m = lik == L.STUDENT_T
    ref = td.StudentT(d.t_df[m], d.t_loc[m], d.t_scale[m])
    torch.testing.assert_close(ent[m], ref.entropy(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(var[m], ref.variance, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mean[m], ref.mean)
    m = lik == L.BERNOULLI
    ref = td.Bernoulli(logits=d.b_logit[m])
    torch.testing.assert_close(ent[m], ref.entropy(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mean[m], ref.mean)
    torch.testing.assert_close(var[m], ref.variance)
    m = lik == L.CATEGORICAL
    ref = td.Categorical(logits=d.raw[..., 6:6 + 8][m][:, :5])
    torch.testing.assert_close(ent[m], ref.entropy(), rtol=1e-5, atol=1e-5)
    cls = torch.arange(5).float()
    torch.testing.assert_close(mean[m], (ref.probs * cls).sum(-1), rtol=1e-5, atol=1e-5)
    m = lik == L.COUNT
    r, mu = d.nb_total_count[m], d.nb_mean[m]
    ref = td.NegativeBinomial(total_count=r, logits=torch.log(mu) - torch.log(r))
    torch.testing.assert_close(mean[m], ref.mean, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(var[m], ref.variance, rtol=1e-4, atol=1e-5)
    k = torch.arange(2000).float().unsqueeze(-1)
    lp = ref.log_prob(k)
    ent_ref = -(lp.exp() * lp).sum(0)
    torch.testing.assert_close(ent[m], ent_ref, rtol=1e-3, atol=1e-3)
    m = lik == L.SURVIVAL
    pmf = d.surv_bin_probs[m]
    torch.testing.assert_close(ent[m], td.Categorical(probs=pmf).entropy(), rtol=1e-4, atol=1e-5)


def test_sampling_matches_moments_and_is_reproducible():
    d, lik, edges = _dist(B=1, Q=12, seed=7, scale=0.5)
    n = 20000
    raw = d.raw.expand(n, -1, -1)
    big = MixedOutcomeDist(raw, d.lik.expand(n, -1), d.n_classes.expand(n, -1), edges, 8)
    s1 = big.sample(torch.Generator().manual_seed(0))
    s2 = big.sample(torch.Generator().manual_seed(0))
    assert torch.equal(s1, s2)
    emp_m, emp_v = s1.mean(0), s1.var(0)
    for q in range(12):
        k = int(lik[0, q])
        if k == L.SURVIVAL:      # within-bin uniform sampling differs from the mid-point mean by < half a bin
            continue
        mu, var = float(d.mean()[0, q]), float(d.variance()[0, q])
        assert abs(emp_m[q] - mu) < 4 * math.sqrt(var / n) + 1e-3, (k, emp_m[q], mu)
        if k != L.STUDENT_T:     # heavy tails: variance estimate is noisy
            assert abs(emp_v[q] - var) / (var + 1e-6) < 0.1, (k, emp_v[q], var)
    m = lik[0] == L.SURVIVAL
    js = (torch.searchsorted(edges, s1[:, m].contiguous(), right=True) - 1).clamp(0, edges.numel() - 1)
    for qi, q in enumerate(torch.nonzero(m).flatten()):
        freq = torch.bincount(js[:, qi], minlength=edges.numel()).float() / n
        torch.testing.assert_close(freq, d.surv_bin_probs[0, q], atol=0.015, rtol=0)
    assert bool((s1[:, lik[0] == L.CATEGORICAL] < 5).all())   # never samples masked classes


def test_head_shapes_masks_and_particle_film():
    cfg = tiny_config()
    torch.manual_seed(0)
    head = OutcomeHead(cfg)
    B, Q, S, K = 3, 7, cfg.n_state, cfg.n_particles
    ob = rand_outcomes(B, Q, cfg.d_model, seed=2)
    Z = torch.randn(B, S, cfg.d_model)
    f = head.features(Z, ob)
    assert f.shape == (B, Q, cfg.d_model)
    d = head.dist(f, ob)
    assert d.log_prob(ob.y, ob.censored).shape == (B, Q)
    for v in (d.mean(), d.variance(), d.entropy(), d.sample()):
        assert v.shape == (B, Q) and torch.isfinite(v).all()
    parts = torch.randn(B, K, cfg.d_particle)
    dk = head.dist(f, ob, cond=parts)
    assert dk.log_prob(ob.y, ob.censored).shape == (B, K, Q)
    # FiLM is zero-initialised: every particle starts at the unconditioned prediction
    torch.testing.assert_close(dk.mean(), d.mean().unsqueeze(1).expand(B, K, Q))
    with torch.no_grad():
        head.film.weight.normal_(0, 0.5)
    dk2 = head.dist(f, ob, cond=parts)
    assert (dk2.mean()[:, 0] - dk2.mean()[:, 1]).abs().max() > 1e-3
    # mask zeroes unobserved entries
    mask = torch.zeros(B, Q, dtype=torch.bool)
    mask[:, 0] = True
    lp = d.log_prob(ob.y, ob.censored, mask)
    assert bool((lp[:, 1:] == 0).all()) and bool((lp[:, 0] != 0).all())
    # queries attend to Z: features change with Z, but not with key-padded tokens
    zm = torch.ones(B, S, dtype=torch.bool)
    zm[:, -2:] = False
    f1 = head.features(Z, ob, mask=zm)
    Z2 = Z.clone()
    Z2[:, -2:] += 5.0
    torch.testing.assert_close(head.features(Z2, ob, mask=zm), f1)
    assert (head.features(Z + torch.randn_like(Z), ob) - f).abs().max() > 1e-3
    ts = TokenSet.from_dense(Z, TokenKind.STATE, mask=zm)
    torch.testing.assert_close(head.features(ts, ob), f1)


def test_gradients_finite_for_extreme_targets():
    cfg = tiny_config()
    head = OutcomeHead(cfg)
    ob = rand_outcomes(4, 12, cfg.d_model, seed=3)
    ob.y = torch.where(ob.lik == L.GAUSSIAN, torch.full_like(ob.y, 1e6), ob.y)
    ob.y = torch.where(ob.lik == L.COUNT, torch.full_like(ob.y, 5e4), ob.y)
    ob.y = torch.where(ob.lik == L.SURVIVAL, torch.full_like(ob.y, 1e4), ob.y)
    Z = torch.randn(4, cfg.n_state, cfg.d_model, requires_grad=True)
    loss = head.nll(Z, ob, cond=torch.randn(4, cfg.n_particles, cfg.d_particle))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    assert torch.isfinite(Z.grad).all()


def test_head_learns_every_likelihood_from_state():
    """Each query type's outcome is a function of a latent scalar stored in Z; the head must learn it."""
    torch.manual_seed(0)
    cfg = tiny_config()
    head = OutcomeHead(cfg)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    B, Q, D = 64, 6, cfg.d_model
    qtok = rand_tokens(1, Q, D, TokenKind.QUERY, ragged=False, seed=11)
    lik = torch.arange(6).repeat(B, 1)
    ncls = torch.where(lik == L.CATEGORICAL, torch.tensor(4), torch.tensor(0))
    direction = torch.randn(D)

    def batch(g):
        u = torch.rand(B, generator=g) * 2 - 1                                   # latent cause in [-1, 1]
        Z = torch.randn(B, cfg.n_state, D, generator=g) * 0.1 + u.view(B, 1, 1) * direction
        y = torch.stack([3 * u + 0.1 * torch.randn(B, generator=g),                # gaussian
                         -2 * u + 0.1 * torch.randn(B, generator=g),               # student-t
                         (torch.rand(B, generator=g) < torch.sigmoid(4 * u)).float(),
                         ((u + 1) * 2).floor().clamp(0, 3),                        # categorical, 4 classes
                         torch.poisson(torch.exp(1.5 + 1.5 * u), generator=g),     # counts
                         torch.exp(1.0 - 2 * u) * (0.8 + 0.4 * torch.rand(B, generator=g))], 1)  # event times
        cens = torch.zeros(B, Q, dtype=torch.bool)
        cens[:, 5] = y[:, 5] > 8.0
        y[:, 5] = y[:, 5].clamp(max=8.0)
        q = TokenSet(qtok.x.expand(B, -1, -1), torch.ones(B, Q, dtype=torch.bool), qtok.kind.expand(B, -1),
                     torch.zeros(B, Q), torch.full((B, Q), -1))
        return Z, OutcomeBatch(q, lik, ncls, y, cens, torch.ones(B, Q, dtype=torch.bool), torch.full((B, Q), -1)), u

    g = torch.Generator().manual_seed(1)
    Zt, obt, ut = batch(torch.Generator().manual_seed(99))
    with torch.no_grad():
        start = -head.dist(head.features(Zt, obt), obt).log_prob(obt.y, obt.censored).mean(0)
    for _ in range(300):
        Z, ob, _ = batch(g)
        loss = head.nll(Z, ob)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        d = head.dist(head.features(Zt, obt), obt)
        end = -d.log_prob(obt.y, obt.censored).mean(0)
        mean = d.mean()
    assert bool((end < start - 0.3).all()), (start, end)
    # predicted means track the latent cause with the right sign
    sign = torch.tensor([1, -1, 1, 1, 1, -1.0])
    for q in range(6):
        c = torch.corrcoef(torch.stack([mean[:, q], ut]))[0, 1]
        assert float(c * sign[q]) > 0.8, (q, float(c))


def test_categorical_and_count_inputs_are_validated_not_clipped():
    """Nothing is silently merged or clipped: unrepresentable categorical queries and out-of-range observed classes
    or counts raise; unobserved (masked) entries are not validated."""
    edges = default_survival_edges(8, 0.5, 30.0)
    Cm = 8
    P = 2 + 3 + 1 + Cm + 2 + edges.numel()
    raw = torch.randn(2, 3, P, generator=torch.Generator().manual_seed(0))
    lik = torch.tensor([[L.CATEGORICAL, L.COUNT, L.GAUSSIAN]]).repeat(2, 1)
    with pytest.raises(ValueError):                                   # 50 classes > max_classes 8
        MixedOutcomeDist(raw, lik, torch.tensor([[50, 0, 0]]).repeat(2, 1), edges, Cm)
    with pytest.raises(ValueError):                                   # a categorical query needs >= 1 class
        MixedOutcomeDist(raw, lik, torch.tensor([[0, 0, 0]]).repeat(2, 1), edges, Cm)
    d = MixedOutcomeDist(raw, lik, torch.tensor([[4, 0, 0]]).repeat(2, 1), edges, Cm)
    ok = torch.tensor([[3.0, 7.0, 0.3], [0.0, 0.0, -2.0]])
    assert torch.isfinite(d.log_prob(ok)).all()
    for bad in ([[7.0, 7.0, 0.3], [0.0, 0.0, 0.0]],          # class 7 of a 4-class query
                [[-1.0, 7.0, 0.3], [0.0, 0.0, 0.0]],         # negative class
                [[1.5, 7.0, 0.3], [0.0, 0.0, 0.0]],          # non-integer class
                [[1.0, -3.0, 0.3], [0.0, 0.0, 0.0]],         # negative count
                [[1.0, 2.5, 0.3], [0.0, 0.0, 0.0]],          # non-integer count
                [[1.0, 1e10, 0.3], [0.0, 0.0, 0.0]]):        # count beyond exp(count_log_max)
        with pytest.raises(ValueError):
            d.log_prob(torch.tensor(bad))
        # the same values on an unobserved entry are ignored (no validation, log-prob 0)
        m = torch.ones(2, 3, dtype=torch.bool)
        m[0, :2] = False
        lp = d.log_prob(torch.tensor(bad), mask=m)
        assert bool((lp[0, :2] == 0).all()) and torch.isfinite(lp).all()
    # the old silent merge is gone: distinct classes have distinct probabilities
    lp = d.log_prob(torch.tensor([[0.0, 1.0, 0.0], [3.0, 1.0, 0.0]]))
    torch.testing.assert_close(lp[:, 0], d.c_logp[:, 0, :4].gather(-1, torch.tensor([[0], [3]])).squeeze(-1))
    # a head with enough classes represents a 50-class outcome exactly
    cfg = tiny_config()
    head = OutcomeHead(cfg, max_classes=64)
    ob = OutcomeBatch(rand_tokens(1, 1, cfg.d_model, TokenKind.QUERY, ragged=False), torch.tensor([[L.CATEGORICAL]]),
                      torch.tensor([[50]]), torch.tensor([[45.0]]), torch.zeros(1, 1, dtype=torch.bool),
                      torch.ones(1, 1, dtype=torch.bool), torch.full((1, 1), -1))
    with torch.no_grad():
        dd = head.dist(head.features(torch.randn(1, cfg.n_state, cfg.d_model), ob), ob)
        assert float(dd.cat_probs[0, 0, 50:].sum()) == 0.0 and float(dd.cat_probs[0, 0, :50].sum()) > 0.999
        torch.testing.assert_close(dd.log_prob(ob.y)[0, 0], dd.c_logp[0, 0, 45])
        assert float(dd.log_prob(ob.y)) != float(dd.log_prob(torch.tensor([[31.0]])))   # class 45 is not class 31


def test_count_mean_is_soft_bounded_with_live_gradients_and_exact_for_large_counts():
    from superlearner.cam_unified.world.outcomes import softbound
    edges = default_survival_edges(8, 0.5, 30.0)
    Cm = 8
    P = 2 + 3 + 1 + Cm + 2 + edges.numel()
    n0 = 2 + 3 + 1 + Cm
    raws = torch.tensor([-30.0, 0.0, 5.0, 15.0, 20.0, 40.0])
    raw = torch.zeros(len(raws), 1, P)
    raw[:, 0, n0] = raws
    raw.requires_grad_(True)
    lik = torch.full((len(raws), 1), int(L.COUNT))
    d = MixedOutcomeDist(raw, lik, torch.zeros_like(lik), edges, Cm)
    lm = d.n_logm[:, 0]
    assert bool((lm[1:] > lm[:-1]).all()) and float(lm.max()) < 20.0 and float(lm.min()) > -12.0
    lm.sum().backward()
    g = raw.grad[:, 0, n0]
    assert bool((g > 0).all()), g                          # no dead gradient anywhere (hard clamp gave 0 at raw 20)
    assert float(g[3]) > 0.3 and float(g[4]) > 0.2          # the old clamp at 15 had zero gradient here
    torch.testing.assert_close(softbound(torch.tensor(4.0), -12.0, 20.0), torch.tensor(4.0))   # identity at centre
    # large counts: the float64 evaluation agrees with a float64 torch.distributions reference
    raw2 = torch.zeros(1, 1, P)
    raw2[0, 0, n0] = 16.0
    raw2[0, 0, n0 + 1] = 3.0
    d2 = MixedOutcomeDist(raw2, lik[:1], torch.zeros_like(lik[:1]), edges, Cm)
    k = torch.tensor([[float(round(float(d2.nb_mean)))]])
    r, mu = d2.nb_total_count.double(), d2.nb_mean.double()
    ref = torch.distributions.NegativeBinomial(total_count=r, logits=torch.log(mu) - torch.log(r)).log_prob(k.double())
    torch.testing.assert_close(d2.log_prob(k).double(), ref, rtol=1e-6, atol=1e-6)
