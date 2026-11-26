"""GoalEvaluator: corrective gradients for over-confidence, prefix training, censoring, decision quality vs luck."""
import math

import numpy as np
import pytest
import torch

from cam_live.contracts import GoalStatus, VerifiedOutcome
from cam_live.predict import calibration as C
from cam_live.predict.evaluator import (EvalTargets, EvaluatorConfig, GoalEvaluator, UnverifiedOutcomeError,
                                        action_quality, delta_v, evaluator_loss, outcome_decomposition,
                                        prefix_targets, prefix_targets_from_outcome)

DM, T, DAY = 16, 5, 86400.0


def _trajectories(n, gen, direction):
    """n objectives, T prefixes each. A hidden quality s decides success; the belief reveals s progressively."""
    s = torch.randn(n, generator=gen)
    p_true = torch.sigmoid(1.5 * s - 1.0)                              # base rate about 0.3
    success = torch.bernoulli(p_true, generator=gen)
    frac = (torch.arange(T) + 1.0) / T
    b = s[:, None, None] * frac[None, :, None] * direction + 0.4 * torch.randn(n, T, DM, generator=gen)
    e_g = torch.randn(n, DM, generator=gen)
    t0 = 1000.0
    prefix_ts = [t0 + i * DAY for i in range(T)]
    deadline = t0 + T * DAY
    ach_frac = torch.rand(n, generator=gen)
    hor, tgts = [], []
    for i in range(n):
        ok = bool(success[i])
        achieved_at = t0 + (T - 1) * DAY + float(ach_frac[i]) * DAY      # achieved during the last day
        value = float(torch.poisson(20.0 * torch.sigmoid(s[i]), generator=gen))
        res_at = [[60.0 * k, 0.5 * k, 0.0] for k in range(T)]
        h, tg = prefix_targets(prefix_ts, status=GoalStatus.ACHIEVED if ok else GoalStatus.FAILED,
                               evaluated_at=achieved_at if ok else deadline, deadline=deadline,
                               signature_checked=True, value=value, achieved_at=achieved_at if ok else None,
                               resources_final=[60.0 * T, 0.5 * T, 0.0], resources_at_prefix=res_at)
        hor.append(h)
        tgts.append(tg)
    tgt = EvalTargets.stack(tgts)                                      # [n, T, ...]
    return dict(b=b, e_g=e_g, horizon=torch.stack(hor), tgt=tgt, success=success, p_true=p_true, s=s)


def _overconfident(seed=0):
    torch.manual_seed(seed)
    ev = GoalEvaluator(EvaluatorConfig(d_model=DM, hidden=64, n_time_bins=6))
    with torch.no_grad():
        ev.out.bias[0] = math.log(0.8 / 0.2)                           # says "80 %" for everything
    return ev


def test_overconfident_evaluator_receives_corrective_gradients_and_calibration_improves(gen):
    direction = torch.randn(DM, generator=gen)
    direction = direction / direction.norm() * 2.0
    train = _trajectories(700, gen, direction)
    held = _trajectories(500, gen, direction)
    ev = _overconfident()
    scale = torch.full((1,), 20.0)
    res_scale = torch.tensor([300.0, 2.5, 1.0])                       # natural size of each resource budget

    def run(d):
        out = ev(d["b"], d["e_g"], d["horizon"], scale=scale.expand(len(d["b"])))
        return out

    with torch.no_grad():
        p0 = run(held).p_success
    y_held = held["success"][:, None].expand(-1, T).reshape(-1)
    before = C.binary_report(p0.reshape(-1), y_held)
    assert abs(before["mean_p"] - 0.8) < 0.03 and before["base_rate"] < 0.4

    # corrective gradient: the model said 80 %, reality said ~30 % -> dL/dlogit = mean(p - y) > 0, so a descent
    # step lowers the stated probability
    out = run(train)
    out.success_logit.retain_grad()
    loss, met = evaluator_loss(out, train["tgt"])
    loss.backward()
    g = out.success_logit.grad
    y_tr = train["success"][:, None].expand(-1, T)
    assert float(ev.out.bias.grad[0]) > 0.3
    assert float(g[y_tr == 0].min()) > 0 and float(g[y_tr == 1].max()) < 0   # each failure pushes down, each success up
    with torch.no_grad():
        for p in ev.parameters():
            if p.grad is not None:                                     # (the Q input layer is unused by V)
                p -= 0.5 * p.grad
        assert float(run(held).p_success.mean()) < float(p0.mean()) - 0.02

    opt = torch.optim.Adam(ev.parameters(), lr=2e-3)
    n = len(train["b"])
    for _ in range(400):
        idx = torch.randint(0, n, (64,), generator=gen)
        o = ev(train["b"][idx], train["e_g"][idx], train["horizon"][idx], scale=scale.expand(64))
        l, _ = evaluator_loss(o, train["tgt"].index(idx), res_scale=res_scale)
        opt.zero_grad()
        l.backward()
        opt.step()
    with torch.no_grad():
        o = run(held)
    p1 = o.p_success
    after = C.binary_report(p1.reshape(-1), y_held, clusters=np.repeat(np.arange(500), T))
    assert after["brier"]["mean"] < before["brier"]["mean"] - 0.15
    assert after["nll"]["mean"] < before["nll"]["mean"] - 0.3
    assert after["ece"]["ece"] < 0.08 < before["ece"]["ece"] - 0.3
    assert abs(after["mean_p"] - after["base_rate"]) < 0.05
    cal = after["calibration"]
    assert cal["identifiable"] and 0.6 < cal["slope"] < 1.5 and abs(cal["intercept"]) < 0.5
    # prefix training: later prefixes have seen more and must be sharper than early ones
    nll_by_prefix = C.binary_nll(p1, held["success"][:, None].expand(-1, T)).reshape(500, T).mean(0)
    assert nll_by_prefix[-1] < nll_by_prefix[0] - 0.05
    # the first prefix is still better than the constant base rate would be over-confident: close to prior entropy
    base = held["success"].mean()
    prior_nll = float(-(base * torch.log(base) + (1 - base) * torch.log(1 - base)))
    assert nll_by_prefix[0] < prior_nll + 0.03

    # the other heads learned too -------------------------------------------------------------------------
    # time to achievement (censored for failures): the hazard head's P(achieved in time) tracks the truth
    p_time = o.p_achieved_in_time()
    assert abs(float(p_time[:, 0].mean()) - float(base)) < 0.08
    assert float(torch.corrcoef(torch.stack([p_time[:, -1], held["p_true"]]))[0, 1]) > 0.5
    ach = held["success"] == 1
    # achieved objectives were all achieved during the final day: at the last prefix the remaining time is < 1 day
    assert float(o.expected_time()[ach, -1].mean()) < DAY
    # magnitude: negative binomial mean tracks 20 * sigmoid(s)
    mag = o.magnitude_count().mean[:, -1]
    assert float(torch.corrcoef(torch.stack([mag, 20.0 * torch.sigmoid(held["s"])]))[0, 1]) > 0.8
    # resources: the remaining expenditure shrinks as the trajectory advances (true: 60 s and 0.5 per step left)
    rem = o.resources_expected(res_scale)[..., 0].mean(0)
    assert bool((rem[1:] < rem[:-1]).all())
    np.testing.assert_allclose(rem, [300.0, 240.0, 180.0, 120.0, 60.0], rtol=0.2)


def test_prefix_targets_semantics():
    ts = [0.0, 10.0, 20.0, 30.0]
    h, tg = prefix_targets(ts, status=GoalStatus.ACHIEVED, evaluated_at=25.0, deadline=40.0, signature_checked=True,
                           value=7.0, achieved_at=24.0, resources_final=[100.0, 5.0],
                           resources_at_prefix=[[0, 0], [30, 1], [60, 2], [100, 5]])
    assert h.tolist() == [40.0, 30.0, 20.0, 10.0]
    assert tg.success_mask.tolist() == [True, True, True, False]           # the 4th prefix is after the outcome
    assert tg.success[:3].tolist() == [1.0, 1.0, 1.0] and tg.magnitude[:3].tolist() == [7.0] * 3
    assert tg.time[:3].tolist() == [24.0, 14.0, 4.0] and bool(tg.time_event.all())
    assert tg.resources[:3, :2].tolist() == [[100.0, 5.0], [70.0, 4.0], [40.0, 3.0]]
    assert tg.resources_mask.tolist() == [[True, True, False]] * 3 + [[False] * 3]
    np.testing.assert_allclose(tg.weight.tolist(), [1 / 3, 1 / 3, 1 / 3, 0.0])
    # failed: success 0 everywhere, time right-censored at each prefix's remaining horizon
    h, tg = prefix_targets(ts, status=GoalStatus.FAILED, evaluated_at=40.0, deadline=40.0, signature_checked=True)
    assert tg.success.tolist() == [0.0] * 4 and tg.success_mask.all() and not tg.time_event.any()
    assert tg.time.tolist() == h.tolist() and not tg.resources_mask.any()
    # pending stays pending: no target of any kind
    _, tg = prefix_targets(ts, status=GoalStatus.PENDING, evaluated_at=40.0, deadline=40.0, signature_checked=True)
    assert not (tg.success_mask.any() or tg.magnitude_mask.any() or tg.time_mask.any() or tg.resources_mask.any())
    assert float(tg.weight.sum()) == 0.0
    # only verified outcomes may supervise
    with pytest.raises(UnverifiedOutcomeError):
        prefix_targets(ts, status=GoalStatus.ACHIEVED, evaluated_at=25.0, deadline=40.0, signature_checked=False)
    with pytest.raises(ValueError):
        prefix_targets(ts, status="maybe", evaluated_at=25.0, deadline=40.0, signature_checked=True)
    signed = VerifiedOutcome("obj", GoalStatus.ACHIEVED, value=3.0, evaluated_at=25.0, signature="abc")
    _, tg = prefix_targets_from_outcome(signed, ts, deadline=40.0, signature_checked=True)
    assert tg.magnitude[0] == 3.0 and int(tg.success_mask.sum()) == 3
    with pytest.raises(UnverifiedOutcomeError):
        prefix_targets_from_outcome(VerifiedOutcome("obj", GoalStatus.ACHIEVED), ts, deadline=40.0,
                                    signature_checked=True)


def test_pending_objective_gives_zero_loss_and_zero_gradient(gen):
    ev = GoalEvaluator(EvaluatorConfig(d_model=DM, hidden=32, n_time_bins=4))
    _, tg = prefix_targets([0.0, 1.0, 2.0], status=GoalStatus.PENDING, evaluated_at=9.0, deadline=9.0,
                           signature_checked=True)
    tg.magnitude[:] = float("nan")
    out = ev(torch.randn(3, DM, generator=gen), torch.randn(3, DM, generator=gen), torch.tensor([9.0, 8.0, 7.0]))
    loss, met = evaluator_loss(out, tg)
    loss.backward()
    assert float(loss.detach()) == 0.0 and met["eval_success_n"] == 0
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in ev.parameters())


def test_bce_is_stable_and_outputs_are_not_zero_initialised(gen):
    ev = GoalEvaluator(EvaluatorConfig(d_model=DM, hidden=32, n_time_bins=4, d_ctx=3))
    b, e_g = torch.randn(6, DM, generator=gen), torch.randn(6, DM, generator=gen)
    hz, ctx = torch.full((6,), DAY), torch.randn(6, 3, generator=gen)
    out = ev(b, e_g, hz, ctx)
    assert float(out.success_logit.detach().std()) > 1e-4                         # small random init, not a dead zero head
    _, tg = prefix_targets([0.0] * 6, status=GoalStatus.ACHIEVED, evaluated_at=5.0, deadline=DAY,
                           signature_checked=True, value=3.0)
    loss, _ = evaluator_loss(out, tg)
    loss.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in ev.named_parameters()
               if not n.startswith("in_z"))                             # every V-path parameter learns from step one
    # extreme logits: finite loss and finite gradients
    with torch.no_grad():
        ev.out.bias[0] = 1e4
    ev.zero_grad()
    tg.success[:] = 0.0
    loss, _ = evaluator_loss(ev(b, e_g, hz, ctx), tg, weights={"magnitude": 0, "time": 0, "resources": 0})
    loss.backward()
    assert math.isfinite(float(loss.detach())) and float(loss.detach()) > 9e3
    assert all(bool(torch.isfinite(p.grad).all()) for p in ev.parameters() if p.grad is not None)
    # the policy context is an input, and it is required when configured
    assert not torch.allclose(ev(b, e_g, hz, ctx).raw, ev(b, e_g, hz, ctx + 1.0).raw)
    with pytest.raises(ValueError):
        ev(b, e_g, hz)
    # horizon and goal are inputs too
    assert not torch.allclose(ev(b, e_g, hz, ctx).raw, ev(b, e_g, hz * 30, ctx).raw)
    assert not torch.allclose(ev(b, e_g, hz, ctx).raw, ev(b, e_g.roll(1, 0), hz, ctx).raw)
    # shapes: prefixes [B, T, D] and candidates [B, N, D]
    assert ev(torch.randn(6, 4, DM, generator=gen), e_g, hz, ctx).success_logit.shape == (6, 4)
    q = ev.forward_candidates(torch.randn(6, 7, DM, generator=gen), e_g, hz, ctx)
    assert q.success_logit.shape == (6, 7) and q.time_hazard_logits.shape == (6, 7, 4)


def test_real_valued_magnitude_head_learns(gen):
    ev = GoalEvaluator(EvaluatorConfig(d_model=DM, hidden=48, n_time_bins=4))
    w = torch.randn(DM, generator=gen) / math.sqrt(DM)
    b = torch.randn(1500, DM, generator=gen)
    y = 3.0 * (b @ w) + 0.5 * torch.randn(1500, generator=gen)
    e_g = torch.zeros(1500, DM)
    opt = torch.optim.Adam(ev.parameters(), lr=3e-3)
    false, true = torch.zeros(1500, dtype=torch.bool), torch.ones(1500, dtype=torch.bool)
    tg = EvalTargets(success=torch.zeros(1500), success_mask=false, magnitude=y, magnitude_mask=true,
                     magnitude_is_count=false, time=torch.zeros(1500), time_event=false, time_mask=false,
                     resources=torch.zeros(1500, 3), resources_mask=torch.zeros(1500, 3, dtype=torch.bool))
    for _ in range(300):
        idx = torch.randint(0, 1500, (128,), generator=gen)
        loss, met = evaluator_loss(ev(b[idx], e_g[idx], torch.full((128,), DAY)), tg.index(idx))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        d = ev(b, e_g, torch.full((1500,), DAY)).magnitude_real()
    assert float(torch.corrcoef(torch.stack([d.mean, 3.0 * (b @ w)]))[0, 1]) > 0.97
    assert 0.3 < float(d.scale.mean()) < 0.9                              # true noise 0.5


def test_delta_v_is_a_detached_signal(gen):
    ev = GoalEvaluator(EvaluatorConfig(d_model=DM, hidden=32, n_time_bins=4))
    b0 = torch.randn(5, DM, generator=gen, requires_grad=True)
    b1 = torch.randn(5, DM, generator=gen)
    e_g, h = torch.randn(5, DM, generator=gen), torch.full((5,), DAY)
    sig = delta_v(ev, b0, b1, e_g, h, h - 3600.0)
    assert set(sig) == {"p_prev", "p_next", "dv_prob", "dv_logit"}
    assert all(not v.requires_grad for v in sig.values())
    assert torch.allclose(sig["dv_prob"], sig["p_next"] - sig["p_prev"])
    assert float(delta_v(ev, b1, b1, e_g, h, h)["dv_prob"].abs().max()) == 0.0      # nothing changed -> no signal
    live = delta_v(ev, b0, b1, e_g, h, h, detach=False)
    assert live["dv_logit"].requires_grad
    assert bool((sig["dv_prob"].abs() <= 1).all())


def test_action_quality_separates_a_good_decision_from_a_lucky_outcome(gen):
    """Bandit with four candidate actions of very different success probability. The evaluator's Q read-out is
    trained on logged (uniformly chosen) decisions and their binary outcomes; afterwards, decisions are judged."""
    p_arm = torch.tensor([0.1, 0.3, 0.6, 0.85])
    e_arm = torch.randn(4, DM, generator=gen)
    ev = GoalEvaluator(EvaluatorConfig(d_model=DM, hidden=48, n_time_bins=4))
    opt = torch.optim.Adam(ev.parameters(), lr=3e-3)
    n = 4000
    arm = torch.randint(0, 4, (n,), generator=gen)
    b = torch.randn(n, DM, generator=gen)
    z = e_arm[arm] + 0.2 * b                                              # decision latent of the chosen candidate
    y = torch.bernoulli(p_arm[arm], generator=gen)
    e_g, hz = torch.zeros(1, DM), torch.full((1,), DAY)
    false = torch.zeros(128, dtype=torch.bool)
    for _ in range(400):
        idx = torch.randint(0, n, (128,), generator=gen)
        out = ev.forward_candidates(z[idx, None], e_g.expand(128, DM), hz.expand(128))
        tg = EvalTargets(success=y[idx, None], success_mask=~false[:, None], magnitude=y[idx, None] * 0,
                         magnitude_mask=false[:, None], magnitude_is_count=false[:, None], time=y[idx, None] * 0,
                         time_event=false[:, None], time_mask=false[:, None], resources=torch.zeros(128, 1, 3),
                         resources_mask=torch.zeros(128, 1, 3, dtype=torch.bool))
        loss, _ = evaluator_loss(out, tg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m = 2000
    b2 = torch.randn(m, DM, generator=gen)
    z_all = e_arm[None] + 0.2 * b2[:, None]                               # [m, 4, D] every candidate
    with torch.no_grad():
        q = ev.forward_candidates(z_all, e_g.expand(m, DM), hz.expand(m)).p_success
    np.testing.assert_allclose(q.mean(0), p_arm, atol=0.08)
    probs = torch.full((m, 4), 0.25)
    chosen = torch.randint(0, 4, (m,), generator=gen)
    y2 = torch.bernoulli(p_arm[chosen], generator=gen)
    aq = action_quality(q, probs, chosen)
    dec = outcome_decomposition(y2, aq["q_chosen"], aq["baseline"])
    assert torch.allclose(dec["quality"] + dec["luck"], dec["surprise"], atol=1e-6)
    truly_good = p_arm[chosen] > p_arm.mean()
    by_quality = ((aq["quality"] > 0) == truly_good).float().mean()
    by_outcome = ((y2 > 0.5) == truly_good).float().mean()
    assert float(by_quality) > 0.97 and float(by_outcome) < 0.85          # the raw outcome confuses luck with skill
    lucky = (chosen == 0) & (y2 == 1)                                     # worst action, happened to succeed
    unlucky = (chosen == 3) & (y2 == 0)                                   # best action, happened to fail
    assert int(lucky.sum()) > 10 and int(unlucky.sum()) > 10
    assert float(dec["quality"][lucky].max()) < 0 < float(dec["luck"][lucky].min())
    assert float(dec["luck"][unlucky].max()) < 0 < float(dec["quality"][unlucky].min())
    assert float(aq["regret"][chosen == 3].abs().max()) < 0.1 and float(aq["regret"][chosen == 0].min()) > 0.5
    # masked candidates are excluded from the baseline and from the best alternative
    mask = torch.tensor([[True, True, False, False]]).expand(m, 4)
    aqm = action_quality(q, probs, torch.zeros(m, dtype=torch.long), mask)
    np.testing.assert_allclose(aqm["baseline"], q[:, :2].mean(1), rtol=1e-5)
    np.testing.assert_allclose(aqm["regret"], q[:, :2].max(1).values - q[:, 0], rtol=1e-5, atol=1e-6)
