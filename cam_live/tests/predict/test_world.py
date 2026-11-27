"""WorldHeads / world_nll / LatentTransition on synthetic data-generating processes with known ground truth."""
import math

import numpy as np
import pytest
import torch

from cam_live.contracts import (OutcomeKind, OutcomeObservation, OutcomeQuery, OutcomeStatus, Provenance)
from cam_live.nn import api as nn_api
from cam_live.predict import dists as D
from cam_live.predict import world as W
from cam_live.predict.imagination import ImaginedDataError
from cam_live.predict.world import (K_BINARY, K_CAT, K_COUNT, K_REAL, K_TTE, KindNormaliser, LatentTransition,
                                    WorldConfig, WorldHeads, WorldTargets, bootstrap_weights, build_targets,
                                    transition_nll, world_nll)

DM = 16
HOUR, DAY = 3600.0, 86400.0


def ones(*shape):
    return torch.ones(*shape, dtype=torch.bool)


def zeros(*shape):
    return torch.zeros(*shape, dtype=torch.bool)


def test_kind_index_matches_tensor_contract():
    assert W.KIND_INDEX == nn_api.KIND_INDEX


# ----------------------------------------------------------------------------------------------------------------
# action-conditioned count rates with delayed arrival, several horizons
# ----------------------------------------------------------------------------------------------------------------
def _count_world(gen):
    n_act = 6
    e_a = torch.randn(n_act, DM, generator=gen)
    lam = torch.tensor([2.0, 5.0, 0.5, 12.0, 8.0, 1.0])                    # eventual expected count per action
    tau = torch.tensor([HOUR, DAY, 2 * HOUR, 2 * DAY, 0.5 * HOUR, 12 * HOUR])   # mean arrival delay per action
    horizons = torch.tensor([0.5 * HOUR, HOUR, 6 * HOUR, DAY, 3 * DAY])
    e_q = torch.randn(1, DM, generator=gen)
    return e_a, lam, tau, horizons, e_q


def _true_count(lam, tau, a, h):
    return lam[a] * (1.0 - torch.exp(-h / tau[a]))


def test_recovers_action_conditioned_count_rates_and_delayed_effects(gen, measure):
    e_a, lam, tau, horizons, e_q = _count_world(gen)
    n = 4000
    a = torch.randint(0, len(lam), (n,), generator=gen)
    z = (e_a[a] + 0.1 * torch.randn(n, DM, generator=gen))[:, None]
    h = horizons[torch.randint(0, len(horizons), (n,), generator=gen)]
    y = torch.poisson(_true_count(lam, tau, a, h), generator=gen)
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=64, n_members=3, n_hazard_bins=6, max_categories=4))
    opt = torch.optim.Adam(heads.parameters(), lr=3e-3)
    kind = torch.full((128, 1), K_COUNT)
    first = None
    for _ in range(600):
        idx = torch.randint(0, n, (128,), generator=gen)
        out = heads(z[idx], e_q.expand(128, 1, DM), kind, h[idx, None].double())
        loss, met = world_nll(out, WorldTargets(y[idx, None], ones(128, 1), zeros(128, 1)))
        first = met["nll_count"] if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    aa = torch.arange(len(lam)).repeat_interleave(len(horizons))
    hh = horizons.repeat(len(lam))
    with torch.no_grad():
        out = heads(e_a[aa][:, None], e_q.expand(len(aa), 1, DM), torch.full((len(aa), 1), K_COUNT), hh[:, None].double())
        pred = out.predictive()["mean"][:, 0, 0]
    true = _true_count(lam, tau, aa, hh)
    rel = float(((pred - true).abs() / (true + 0.5)).mean())
    corr = float(torch.corrcoef(torch.stack([true, pred]))[0, 1])
    assert met["nll_count"] < first - 0.5
    measure("world_count_mean_relative_error", rel)
    measure("world_count_pred_true_corr", corr)
    assert rel < 0.2 and corr > 0.97, (rel, corr)
    # delayed effect: the slow action (tau = 2 days) must be predicted to yield far less at 1 hour than at 3 days,
    # and the fast one (tau = 30 min) to be nearly saturated after 6 hours
    grid = pred.reshape(len(lam), len(horizons))
    assert grid[3, 1] < 0.25 * grid[3, 4]
    assert grid[4, 2] > 0.8 * grid[4, 4]
    # multi-horizon by construction: predictions are monotone in the horizon for the slow action
    assert bool((grid[3, 1:] > grid[3, :-1]).all())


# ----------------------------------------------------------------------------------------------------------------
# every outcome kind, with pending entries
# ----------------------------------------------------------------------------------------------------------------
def _mixed_batch(n, gen):
    w = torch.randn(DM, generator=gen) / math.sqrt(DM)
    e_q = torch.randn(5, DM, generator=gen)
    z = torch.randn(n, 1, DM, generator=gen)
    u = z[:, 0] @ w
    kind = torch.tensor([K_REAL, K_COUNT, K_BINARY, K_TTE, K_CAT]).expand(n, 5).contiguous()
    horizon = torch.tensor([HOUR, HOUR, HOUR, 10.0, HOUR], dtype=torch.float64).expand(n, 5).contiguous()
    y = torch.zeros(n, 5)
    y[:, 0] = 2.0 * u + 0.5 * torch.randn(n, generator=gen)
    mu = torch.exp(1.0 + 0.7 * u)
    y[:, 1] = torch.poisson(mu, generator=gen)
    p = torch.sigmoid(1.5 * u)
    y[:, 2] = torch.bernoulli(p, generator=gen)
    rate = 0.1 * torch.exp(0.8 * u)
    t = -torch.log(torch.rand(n, generator=gen).clamp_min(1e-12)) / rate
    happened = t < 10.0
    y[:, 3] = torch.where(happened, t, torch.full_like(t, 10.0))
    logits = torch.stack([2 * u, -2 * u, torch.zeros_like(u)], -1)
    y[:, 4] = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1).float()
    observed = ones(n, 5)
    observed[:, 3] = happened
    censored = zeros(n, 5)
    censored[:, 3] = ~happened
    oracle = {"real": 0.5 * math.log(2 * math.pi * math.e * 0.25),
              "count": float(-D.poisson_log_prob(y[:, 1], mu).mean()),
              "binary": float(-D.bernoulli_log_prob(y[:, 2], 1.5 * u).mean()),
              "categorical": float(-D.categorical_log_prob(y[:, 4].long(), logits).mean())}
    # exact discretised hazard of an exponential time on 8 equal bins of [0, 10]: h = 1 - exp(-rate * width)
    h_true = (1.0 - torch.exp(-rate * 10.0 / 8)).clamp(1e-9, 1 - 1e-9)[:, None].expand(n, 8)
    oracle["time_to_event"] = float(-D.hazard_log_prob(y[:, 3], happened, torch.log(h_true) - torch.log1p(-h_true),
                                                       D.uniform_edges(torch.tensor(10.0), 8)).mean())
    expo = torch.ones(n, 5)
    expo[:, 1] = 2.0                                                # the count is reported over an exposure of 2
    ncat = torch.full((n, 5), 3)
    return dict(z=z, e_q=e_q, kind=kind, horizon=horizon, y=y, observed=observed, censored=censored, oracle=oracle,
                expo=expo, ncat=ncat, p_event=1.0 - torch.exp(-rate * 10.0), u=u)


def test_all_kinds_learn_with_pending_entries(gen, measure):
    d = _mixed_batch(3000, gen)
    n = 3000
    pending = torch.rand(n, 5, generator=gen) < 0.3                  # 30 % of the outcomes have not matured
    y_seen = torch.where(pending, torch.full_like(d["y"], float("nan")), d["y"])
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=64, n_members=3, n_hazard_bins=8, max_categories=5))
    opt = torch.optim.Adam(heads.parameters(), lr=3e-3)
    norm = KindNormaliser()

    def forward(idx):
        return heads(d["z"][idx], d["e_q"].expand(len(idx), 5, DM), d["kind"][idx], d["horizon"][idx],
                     exposure=d["expo"][idx], n_categories=d["ncat"][idx])

    first = None
    for _ in range(700):
        idx = torch.randint(0, n, (128,), generator=gen)
        tg = WorldTargets(y_seen[idx], d["observed"][idx] & ~pending[idx], d["censored"][idx] & ~pending[idx])
        loss, met = world_nll(forward(idx), tg, normaliser=norm)
        assert math.isfinite(float(loss.detach()))
        first = dict(met) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        full = torch.arange(n)
        out = forward(full)
        _, met = world_nll(out, WorldTargets(d["y"], d["observed"], d["censored"]))
        pred = out.predictive()
    measure("world_nll_by_kind", {k[4:]: round(v, 4) for k, v in met.items() if k.startswith("nll_")})
    measure("world_nll_oracle_by_kind", {k: round(v, 4) for k, v in d["oracle"].items()})
    for name in ("real", "count", "binary", "categorical", "time_to_event"):
        assert met[f"nll_{name}"] < d["oracle"][name] + 0.15, (name, met[f"nll_{name}"], d["oracle"][name])
        assert met[f"nll_{name}"] < first[f"nll_{name}"]
    # count: the head predicts a rate per unit exposure; the predictive mean is rate * exposure
    mu = torch.exp(1.0 + 0.7 * d["u"])
    assert float(torch.corrcoef(torch.stack([pred["mean"][:, 0, 1].log(), mu.log()]))[0, 1]) > 0.97
    # time-to-event: probability of the event within the horizon matches the exponential ground truth
    surv_end = D.hazard_survival(out.hazard_logits)[..., -1].mean(0)[:, 0, 3]
    assert float((1.0 - surv_end - d["p_event"]).abs().mean()) < 0.1
    assert float(torch.corrcoef(torch.stack([1.0 - surv_end, d["p_event"]]))[0, 1]) > 0.9
    # every kind reports a positive aleatoric spread, and the summary is finite
    assert all(bool(torch.isfinite(v).all()) for v in pred.values())
    assert bool((pred["aleatoric_std"] > 0).all())


def test_pending_targets_contribute_exactly_zero_gradient(gen):
    d = _mixed_batch(32, gen)
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=32, n_members=3, n_hazard_bins=8, max_categories=5))
    out = heads(d["z"], d["e_q"].expand(32, 5, DM), d["kind"], d["horizon"], exposure=d["expo"],
                n_categories=d["ncat"])
    # (a) nothing matured: zero loss, zero gradient everywhere
    loss, met = world_nll(out, WorldTargets(torch.full((32, 5), float("nan")), zeros(32, 5), zeros(32, 5)))
    loss.backward()
    assert float(loss.detach()) == 0.0 and met["n_targets"] == 0
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in heads.parameters())

    # (b) the value stored under a pending / unobservable slot cannot influence the gradient in any way
    pending = torch.rand(32, 5, generator=gen) < 0.5

    def grads(fill):
        heads.zero_grad()
        o = heads(d["z"], d["e_q"].expand(32, 5, DM), d["kind"], d["horizon"], exposure=d["expo"],
                  n_categories=d["ncat"])
        y = torch.where(pending, torch.full_like(d["y"], fill), d["y"])
        l, _ = world_nll(o, WorldTargets(y, d["observed"] & ~pending, d["censored"] & ~pending))
        l.backward()
        return torch.cat([p.grad.flatten() for p in heads.parameters() if p.grad is not None]), float(l.detach())

    g_nan, l_nan = grads(float("nan"))
    g_big, l_big = grads(1e30)
    g_neg, l_neg = grads(-7.0)
    assert torch.isfinite(g_nan).all() and float(g_nan.abs().max()) > 0
    assert torch.equal(g_nan, g_big) and torch.equal(g_nan, g_neg) and l_nan == l_big == l_neg

    # (c) a censored flag on a non-time-to-event query is not a target
    cen = zeros(32, 5)
    cen[:, 0] = True
    tg = WorldTargets(d["y"], zeros(32, 5), cen)
    assert int(tg.mask(d["kind"]).sum()) == 0


def test_candidate_selection_and_per_candidate_queries(gen):
    b, n, q = 4, 3, 2
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=32, n_members=2, n_features=6))
    z = torch.randn(b, n, DM, generator=gen)
    e_q4 = torch.randn(b, n, q, DM, generator=gen)
    kind = torch.tensor([[K_REAL, K_BINARY]]).expand(b, q)
    hor = torch.full((b, q), DAY, dtype=torch.float64)
    phi = torch.randn(b, n, q, 6, generator=gen)
    out = heads(z, e_q4, kind, hor, phi)
    assert out.raw.shape[:4] == (2, b, n, q)
    y = torch.randn(b, q)
    y[:, 1] = (y[:, 1] > 0).float()
    tg = WorldTargets(y, ones(b, q), zeros(b, q))
    chosen = torch.tensor([0, 2, 1, 1])
    loss, _ = world_nll(out, tg, chosen=chosen)
    manual = torch.stack([-out.log_prob(y)[:, i, chosen[i]] for i in range(b)], 1)        # [M, B, Q]
    expect = manual[..., 0].mean() + manual[..., 1].mean()                                # per-kind means
    assert torch.allclose(loss, expect, atol=1e-6)
    with pytest.raises(ValueError):
        world_nll(out, tg)                                           # ambiguous without `chosen`
    with pytest.raises(ValueError):
        heads(z, e_q4, kind, hor)                                    # phi is required when n_features > 0
    # phi is really used
    out2 = heads(z, e_q4, kind, hor, phi + 1.0)
    assert not torch.allclose(out.raw, out2.raw)


def test_per_kind_normalisation_stops_one_kind_dominating(gen, measure):
    n = 64
    z = torch.randn(n, 1, DM, generator=gen)
    e_q = torch.randn(2, DM, generator=gen).expand(n, 2, DM)
    kind = torch.tensor([[K_BINARY, K_COUNT]]).expand(n, 2)
    hor = torch.full((n, 2), DAY, dtype=torch.float64)
    y = torch.stack([torch.bernoulli(torch.full((n,), 0.5), generator=gen),
                     torch.poisson(torch.full((n,), 20000.0), generator=gen)], 1)        # counts in the tens of thousands
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=32, n_members=2))

    def share(normaliser):
        norms = {}
        for name, col in (("binary", 0), ("count", 1)):
            heads.zero_grad()
            obs = zeros(n, 2)
            obs[:, col] = True
            loss, _ = world_nll(heads(z, e_q, kind, hor), WorldTargets(y, obs, zeros(n, 2)), normaliser=normaliser)
            loss.backward()
            norms[name] = float(heads.in_z.weight.grad.norm())
        return norms["count"] / norms["binary"]

    raw = share(None)
    normed = share(KindNormaliser())
    measure("world_trunk_grad_ratio_count_over_binary_raw", raw)
    measure("world_trunk_grad_ratio_count_over_binary_normalised", normed)
    assert raw > 1e3                     # un-normalised, the count likelihood owns the shared trunk
    assert normed < 30 and normed < raw / 100
    # the metrics still report the plain per-kind NLL
    _, met = world_nll(heads(z, e_q, kind, hor), WorldTargets(y, ones(n, 2), zeros(n, 2)), normaliser=KindNormaliser())
    assert met["nll_count"] > 1e3 and met["nll_binary"] < 5


# ----------------------------------------------------------------------------------------------------------------
# epistemic uncertainty
# ----------------------------------------------------------------------------------------------------------------
def _fit_real(n, seed, members=5, steps=400):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(DM, generator=g) / math.sqrt(DM)
    e_q = torch.randn(1, DM, generator=g)
    z = torch.randn(n, 1, DM, generator=g)
    y = torch.sin(z[:, 0] @ w) * 2 + 0.3 * torch.randn(n, generator=g)
    torch.manual_seed(seed)
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=64, n_members=members, n_hazard_bins=4, max_categories=4,
                                   seed=seed))
    opt = torch.optim.Adam(heads.parameters(), lr=3e-3)
    boot = bootstrap_weights([f"rec{i}" for i in range(n)], members, seed)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(n, 128),), generator=g)
        k = len(idx)
        out = heads(z[idx], e_q.expand(k, 1, DM), torch.full((k, 1), K_REAL), torch.full((k, 1), HOUR, dtype=torch.float64))
        loss, _ = world_nll(out, WorldTargets(y[idx, None], ones(k, 1), zeros(k, 1)), member_weights=boot[:, idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def epistemic(zz):
        o = heads(zz, e_q.expand(len(zz), 1, DM), torch.full((len(zz), 1), K_REAL),
                  torch.full((len(zz), 1), HOUR, dtype=torch.float64))
        return float(o.predictive()["epistemic_std"].mean())

    z_in = torch.randn(512, 1, DM, generator=g)
    z_off = torch.randn(512, 1, DM, generator=g) + 4.0
    return epistemic(z_in), epistemic(z_off)


def test_epistemic_uncertainty_shrinks_with_data_and_is_larger_off_distribution(measure):
    small = [_fit_real(24, s) for s in (0, 1)]
    large = [_fit_real(2048, s) for s in (0, 1)]
    measure("world_epistemic_std_n24_in_dist", [s[0] for s in small])
    measure("world_epistemic_std_n2048_in_dist", [l[0] for l in large])
    measure("world_epistemic_std_n2048_off_dist", [l[1] for l in large])
    for (s_in, _), (l_in, l_off) in zip(small, large):
        assert l_in < 0.6 * s_in, (s_in, l_in)           # more data -> less reducible uncertainty on-distribution
        assert l_off > 2.0 * l_in, (l_in, l_off)         # away from the data the members still disagree


def test_bootstrap_weights_are_deterministic_poisson():
    keys = [f"k{i}" for i in range(4000)]
    w = bootstrap_weights(keys, 3, seed=7)
    assert torch.equal(w, bootstrap_weights(keys, 3, seed=7))
    assert not torch.equal(w, bootstrap_weights(keys, 3, seed=8))
    assert not torch.equal(w[0], w[1])
    np.testing.assert_allclose(w.mean(1), 1.0, atol=0.06)
    np.testing.assert_allclose(w.var(1), 1.0, atol=0.12)
    np.testing.assert_allclose((w == 0).float().mean(1), math.exp(-1), atol=0.03)


def test_thompson_means_draw_whole_members(gen):
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=32, n_members=4))
    out = heads(torch.randn(3, 5, DM, generator=gen), torch.randn(3, 2, DM, generator=gen),
                torch.tensor([[K_REAL, K_COUNT]]).expand(3, 2), torch.full((3, 2), DAY, dtype=torch.float64))
    mm = out.member_means()
    draw = out.thompson_means(gen)
    for b in range(3):
        assert any(torch.equal(draw[b], mm[m, b]) for m in range(4))
    seen = {tuple(out.thompson_means(gen)[0].flatten().tolist()) for _ in range(40)}
    assert len(seen) > 1


# ----------------------------------------------------------------------------------------------------------------
# contract records
# ----------------------------------------------------------------------------------------------------------------
def test_to_distribution_for_every_kind(gen):
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=32, n_members=3, n_hazard_bins=5, max_categories=6))
    kind = torch.tensor([[K_REAL, K_COUNT, K_BINARY, K_TTE, K_CAT]])
    out = heads(torch.randn(1, 2, DM, generator=gen), torch.randn(1, 5, DM, generator=gen), kind,
                torch.full((1, 5), DAY, dtype=torch.float64), exposure=torch.tensor([[1.0, 4.0, 1.0, 1.0, 1.0]]),
                n_categories=torch.tensor([[1, 1, 1, 1, 3]]), real_loc=torch.tensor([[100.0, 0, 0, 0, 0]]),
                real_scale=torch.tensor([[10.0, 1, 1, 1, 1]]))
    pred = out.predictive()
    fams = ["student_t", "neg_binomial", "bernoulli", "discrete_hazard", "categorical"]
    for q, fam in enumerate(fams):
        rec = out.to_distribution(0, 1, q)
        assert rec.family == fam and rec.epistemic_std is not None and rec.epistemic_std >= 0
        d = D.from_record(rec)
        if fam in ("student_t", "neg_binomial", "bernoulli", "discrete_hazard"):
            np.testing.assert_allclose(float(d.mean), float(pred["mean"][0, 1, q].detach()), rtol=2e-4)
    # real predictions are reported in the metric's own units
    assert abs(out.to_distribution(0, 1, 0).mean - 100.0) < 60.0
    # exact mixtures: the recorded likelihood equals the mean of the members' likelihoods
    y1 = torch.ones(1, 5)
    y1[0, 3], y1[0, 4] = 0.3 * DAY, 2.0
    member_lik = torch.exp(out.log_prob(y1)[:, 0, 1])                # [M, Q]
    for q in (2, 3, 4):
        got = math.exp(D.record_log_prob(out.to_distribution(0, 1, q), float(y1[0, q])))
        np.testing.assert_allclose(got, float(member_lik[:, q].detach().mean()), rtol=1e-4)
    # moment-matched count: total variance = aleatoric + epistemic
    nb = D.from_record(out.to_distribution(0, 1, 1))
    np.testing.assert_allclose(float(nb.variance), float(pred["total_std"][0, 1, 1]) ** 2, rtol=1e-3)
    assert len(out.to_distribution(0, 0, 4).params["probs"]) == 3


def test_build_targets_masks_pending_and_refuses_imagined():
    def q(i, kind, **kw):
        return OutcomeQuery(query_id=f"q{i}", metric=f"m{i}", kind=kind, horizon_s=DAY, **kw)

    rows = [
        [(q(0, OutcomeKind.COUNT, exposure=3.0), OutcomeObservation("q0", OutcomeStatus.OBSERVED, 4.0)),
         (q(1, OutcomeKind.TIME_TO_EVENT), OutcomeObservation("q1", OutcomeStatus.CENSORED, DAY)),
         (q(2, OutcomeKind.BINARY), OutcomeObservation("q2", OutcomeStatus.PENDING))],
        [(q(3, OutcomeKind.REAL), OutcomeObservation("q3", OutcomeStatus.UNOBSERVABLE, 9.0)),
         (q(4, OutcomeKind.CATEGORICAL, categories=["a", "b", "c"]), None)],
    ]
    meta, tg = build_targets(rows)
    assert meta.kind_ids.tolist() == [[K_COUNT, K_TTE, K_BINARY], [K_REAL, K_CAT, 0]]
    assert meta.q_mask.tolist() == [[True, True, True], [True, True, False]]
    assert tg.mask(meta.kind_ids).tolist() == [[True, True, False], [False, False, False]]
    assert float(meta.exposure[0, 0]) == 3.0 and int(meta.n_categories[1, 1]) == 3
    assert float(tg.y[1, 0]) == 0.0                              # an unobservable value is not copied into a target
    # a censored real-valued observation is not a target either
    _, tg2 = build_targets([[(q(5, OutcomeKind.REAL), OutcomeObservation("q5", OutcomeStatus.CENSORED, 1.0))]])
    assert not bool(tg2.mask(torch.tensor([[K_REAL]])).any())
    with pytest.raises(ValueError):
        build_targets([[(q(6, OutcomeKind.REAL), OutcomeObservation("other", OutcomeStatus.OBSERVED, 1.0))]])

    class Imagined(OutcomeObservation):
        imagined = True

    with pytest.raises(ImaginedDataError):
        build_targets([[(q(7, OutcomeKind.REAL), Imagined("q7", OutcomeStatus.OBSERVED, 1.0))]])


# ----------------------------------------------------------------------------------------------------------------
# latent transition
# ----------------------------------------------------------------------------------------------------------------
def _latent_dynamics(gen):
    a_mat = torch.randn(DM, DM, generator=gen) / math.sqrt(DM)

    def step(b, z, dt, noise=0.05):
        gain = 1.0 - torch.exp(-dt / HOUR)                           # longer steps move the belief further
        return b + gain[:, None] * torch.tanh(z @ a_mat) + noise * torch.randn(b.shape, generator=gen)
    return step


def test_latent_transition_learns_dynamics_and_masks_missing_next(gen, measure):
    step = _latent_dynamics(gen)
    n = 3000
    b = torch.randn(n, DM, generator=gen)
    z = torch.randn(n, DM, generator=gen)
    dt = torch.tensor([600.0, HOUR, 4 * HOUR])[torch.randint(0, 3, (n,), generator=gen)]
    b_next = step(b, z, dt)
    tr = LatentTransition(DM, hidden=96, n_members=3)
    opt = torch.optim.Adam(tr.parameters(), lr=3e-3)
    first = None
    for _ in range(500):
        idx = torch.randint(0, n, (128,), generator=gen)
        loss, met = transition_nll(tr(z[idx], dt[idx], b[idx]), b_next[idx])
        first = met if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = tr(z, dt, b)
        _, met = transition_nll(out, b_next)
        persistence = float(((b - b_next) ** 2).mean() / b_next.var(0).mean())
    assert out.imagined is True
    measure("transition_nmse", met["transition_nmse"])
    measure("transition_persistence_nmse", persistence)
    assert met["transition_nll"] < first["transition_nll"] - 0.5
    assert met["transition_nmse"] < 0.05 and met["transition_nmse"] < 0.3 * persistence
    # the learned aleatoric spread is close to the true process noise (0.05)
    assert 0.03 < float(out.aleatoric_var.sqrt().mean()) < 0.12
    # dt is used: the same (z, b) with a longer step moves further
    with torch.no_grad():
        short = (tr(z[:64], torch.full((64,), 600.0), b[:64]).mean - b[:64]).norm(dim=-1).mean()
        long = (tr(z[:64], torch.full((64,), 4 * HOUR), b[:64]).mean - b[:64]).norm(dim=-1).mean()
        off = tr(z[:256] * 4 + 3, dt[:256], b[:256]).epistemic_var.mean()
        on = tr(z[:256], dt[:256], b[:256]).epistemic_var.mean()
    assert float(long) > 2.0 * float(short)
    assert float(off) > 3.0 * float(on)                             # its own uncertainty grows off-distribution
    # rows without a next belief contribute exactly zero, whatever their target holds
    mask = torch.rand(128, generator=gen) < 0.5
    tgt = b_next[:128].clone()

    def grads(fill):
        tr.zero_grad()
        t2 = torch.where(mask[:, None], tgt, torch.full_like(tgt, fill))
        l, m = transition_nll(tr(z[:128], dt[:128], b[:128]), t2, mask=mask)
        l.backward()
        return torch.cat([p.grad.flatten() for p in tr.parameters()]), m

    g1, m1 = grads(0.0)
    g2, _ = grads(1e20)
    assert torch.equal(g1, g2) and m1["n"] == float(mask.sum())
    tr.zero_grad()
    l0, _ = transition_nll(tr(z[:8], dt[:8], b[:8]), b_next[:8], mask=zeros(8))
    l0.backward()
    assert float(l0.detach()) == 0.0 and all(float(p.grad.abs().max()) == 0.0 for p in tr.parameters() if p.grad is not None)


def test_transition_target_is_detached_and_sampling_is_seeded(gen):
    tr = LatentTransition(DM, hidden=32, n_members=2)
    b_next = torch.randn(4, DM, generator=gen, requires_grad=True)
    out = tr(torch.randn(4, DM, generator=gen), torch.full((4,), 60.0), torch.randn(4, DM, generator=gen))
    loss, _ = transition_nll(out, b_next)
    loss.backward()
    assert b_next.grad is None                                       # the real next belief is a target, not a variable
    s1 = out.sample(torch.Generator().manual_seed(5))
    s2 = out.sample(torch.Generator().manual_seed(5))
    assert torch.equal(s1, s2) and s1.shape == (4, DM)
