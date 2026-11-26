"""Self-evaluation, intrinsic reward and credit heads on synthetic data with known answers."""
import math

import torch

from superlearner.cam_unified.credit.heads import (HindsightModel, OutcomeEncoder, ReturnDecomposer,
                                                    contribution_coefficients)
from superlearner.cam_unified.selfeval.evaluator import (FAILURE, PENDING, SUCCESS, ProspectEvaluator,
                                                          ProspectTargets, calibration_report)
from superlearner.cam_unified.selfeval.intrinsic import N_SIGNALS, ExperienceSignals, IntrinsicReward, LifetimeValue

torch.set_num_threads(4)


def _synth(n, D, g):
    R = torch.randn(n, 4, D, generator=g)
    p = torch.sigmoid(3 * R[:, :, 0].mean(1))
    y = torch.bernoulli(p, generator=g)
    st = torch.where(y > 0, SUCCESS, FAILURE)
    st = torch.where(torch.rand(n, generator=g) < 0.25, PENDING, st)
    t = 1 + 10 * torch.rand(n, generator=g)
    return R, p, y, st, t


def test_pending_outcomes_never_train_success_as_failure():
    D = 32
    ev = ProspectEvaluator(D, n_heads=4)
    R = torch.randn(8, 4, D)
    out = ev(R, None, torch.full((8,), 10.0), torch.zeros(8, dtype=torch.long))
    st = torch.full((8,), PENDING)
    L = ev.losses(out, ProspectTargets(st, torch.full((8,), 3.0)))
    g = torch.autograd.grad(L["success"], out.success_logit, allow_unused=True, retain_graph=True)[0]
    assert g is None or float(g.abs().max()) == 0.0
    assert L["n_resolved"] == 0
    # censored time still trains the hazard (survived 3 days)
    assert float(L["time"]) > 0


def test_evaluator_learns_calibrated_success_predictions():
    g = torch.Generator().manual_seed(0)
    D = 32
    ev = ProspectEvaluator(D, n_heads=4)
    opt = torch.optim.Adam(ev.parameters(), 3e-3)
    for _ in range(250):
        R, p, y, st, t = _synth(256, D, g)
        out = ev(R, None, torch.full((256,), 30.0), torch.zeros(256, dtype=torch.long))
        L = ev.losses(out, ProspectTargets(st, t))
        opt.zero_grad(); L["total"].backward(); opt.step()
    R, p, y, st, t = _synth(3000, D, g)
    with torch.no_grad():
        out = ev(R, None, torch.full((3000,), 30.0), torch.zeros(3000, dtype=torch.long))
    rep = calibration_report(out.p_success, y)
    oracle = float(((p - y) ** 2).mean())
    assert rep["brier"] < oracle + 0.02 and rep["brier_skill"] > 0.1
    assert abs(rep["calib_slope"] - 1) < 0.25 and rep["ece"] < 0.06


def test_calibration_report_on_known_cases():
    g = torch.Generator().manual_seed(1)
    p = torch.rand(20000, generator=g)
    y = torch.bernoulli(p, generator=g)
    rep = calibration_report(p, y)
    assert rep["ece"] < 0.02 and abs(rep["calib_slope"] - 1) < 0.05 and abs(rep["calib_intercept"]) < 0.05
    over = calibration_report(torch.sigmoid(3 * torch.logit(p.clamp(1e-3, 1 - 1e-3))), y)
    assert over["calib_slope"] < 0.5          # over-confident predictions have slope < 1


def test_intrinsic_reward_is_bounded_versioned_and_goal_free_capable():
    D = 32
    ir = IntrinsicReward(D, d_hidden=64, scale=1.0)
    with torch.no_grad():
        for p in ir.parameters():
            p.normal_(0, 3.0)
    s = ExperienceSignals.zeros(16).stack() + 100 * torch.randn(16, N_SIGNALS)
    r = ir(torch.randn(16, D), torch.randn(16, D), torch.randn(16, D), None, s)
    assert float(r.abs().max()) <= 1.0 + 1e-6
    v = ir.version
    ir.bump_version()
    assert ir.version == v + 1
    lv = LifetimeValue(D, 64)
    assert lv(torch.randn(4, D)).shape == (4,)


def test_return_decomposer_conserves_discounted_and_undiscounted_returns():
    torch.manual_seed(0)
    rd = ReturnDecomposer(16, d_model=32, n_layers=1, n_heads=4)
    with torch.no_grad():
        rd.head.weight.normal_(0, 0.5)
    x = torch.randn(5, 12, 16)
    mask = torch.ones(5, 12, dtype=torch.bool)
    mask[2, 7:] = False
    G = torch.randn(5) * 100
    red = rd.redistribute(x, mask, G)
    assert float((red.contrib.sum(1) - G).abs().max()) < 1e-3
    assert float(red.contrib[2, 7:].abs().max()) == 0.0
    tau = torch.cumsum(torch.rand(5, 12), 1)
    gt = torch.exp(-0.1 * tau)
    red = rd.redistribute(x, mask, G, gamma_tau=gt)
    assert float(((red.contrib * gt).sum(1) - G).abs().max()) < 1e-3


def test_return_decomposer_learns_which_early_event_matters():
    """Only the event at step 2 determines the delayed return; after training the redistribution puts the credit
    there, not on the final step (RUDDER's purpose)."""
    torch.manual_seed(0)
    T, d = 20, 8
    rd = ReturnDecomposer(d, d_model=48, n_layers=2, n_heads=4)
    opt = torch.optim.Adam(rd.parameters(), 2e-3)

    def batch(n):
        x = torch.randn(n, T, d)
        key = (torch.rand(n) < 0.5).float()
        x[:, 2, 0] = 3 * (2 * key - 1)
        G = 10 * key + 0.1 * torch.randn(n)
        return x, G
    for _ in range(300):
        x, G = batch(128)
        L = rd.loss(x, torch.ones(128, T, dtype=torch.bool), G)
        opt.zero_grad(); L.backward(); opt.step()
    x, G = batch(256)
    red = rd.redistribute(x, torch.ones(256, T, dtype=torch.bool), G)
    c = red.contrib.clone()
    c[:, 0] = c[:, 0] - red.offset                       # the offset E[G] is a baseline, not credit
    share_step2 = c[:, 2].abs().mean() / c.abs().sum(1).mean()
    assert float(share_step2) > 0.5


def test_hindsight_model_recovers_contribution_coefficients():
    """Three actions with pi = (0.5, 0.3, 0.2); outcome u in {0,1}: P(u=1|a) = (0.9, 0.1, 0.5). The exact
    hindsight distribution is h(a|u=1) = pi(a) P(u=1|a) / sum_a' pi(a') P(u=1|a')."""
    torch.manual_seed(0)
    D, A, du = 16, 3, 4
    pi = torch.tensor([0.5, 0.3, 0.2])
    pu = torch.tensor([0.9, 0.1, 0.5])
    hm = HindsightModel(D, d_u=du, d_hidden=64)
    enc = OutcomeEncoder(2, d_u=du, d_hidden=16)
    cand = torch.randn(1, A, D).expand(512, -1, -1)
    mask = torch.ones(512, A, dtype=torch.bool)
    b = torch.zeros(512, D)
    opt = torch.optim.Adam(list(hm.parameters()) + list(enc.parameters()), 1e-3)
    for _ in range(1500):
        a = torch.multinomial(pi, 512, replacement=True)
        u = torch.bernoulli(pu[a])
        uf = enc(torch.stack([u, 1 - u], -1))
        L = hm.loss(b, uf, None, cand, mask, a)
        opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        uf = enc(torch.tensor([[1.0, 0.0]])).expand(512, -1)
        lg = hm.logits(b, uf, None, cand, mask)[:1]
        w = contribution_coefficients(lg, pi.log()[None], mask[:1])[0]
    h_exact = pi * pu / (pi * pu).sum()
    w_exact = h_exact / pi - 1
    assert float((w - w_exact).abs().max()) < 0.12
