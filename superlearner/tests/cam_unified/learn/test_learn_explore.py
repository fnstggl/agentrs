"""Learning progress (hindsight, held-out queries) with epistemic/aleatoric separation: pure noise earns nothing,
informative observations earn LP, and a policy trained on LP prefers an informative probe over a noisy TV while a
raw-surprise learner prefers the TV."""
import math

import pytest
import torch

from learn_doubles import ArmBelief, ParticleThetaBelief, ScalarStep, arm_learning_progress
from superlearner.cam_unified.interfaces import Action
from superlearner.cam_unified.learn.diagnostics import ProbeBanditAdapter
from superlearner.cam_unified.learn.explore import (categorical_decomposition, combine_signal_frac,
                                                    exploration_reward, gaussian_decomposition,
                                                    hindsight_learning_progress, mixture_log_prob, noise_corrected_lp,
                                                    surprise_reward)
from superlearner.cam_unified.learn.meta import TrialRunner


def _scenario(B, support_informative, seed=0, H=6, naive_shift=0.0):
    g = torch.Generator().manual_seed(seed)
    thetas = torch.linspace(-2, 2, 9)
    bel = ParticleThetaBelief(thetas, naive_shift)
    true = thetas[torch.randint(0, 9, (B,), generator=g)]
    steps = []
    # step 0 = support: informative (x = 1, y = theta + small noise) or pure noise (x = 0, y ~ N(0, 3^2))
    if support_informative:
        steps.append(ScalarStep(torch.ones(B), true + 0.5 * torch.randn(B, generator=g), torch.full((B,), 0.5)))
    else:
        steps.append(ScalarStep(torch.zeros(B), 3.0 * torch.randn(B, generator=g), torch.full((B,), 3.0)))
    for _ in range(H):                                   # later real observations = queries (scored in hindsight)
        x = torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0)
        steps.append(ScalarStep(x, true * x + 0.5 * torch.randn(B, generator=g), torch.full((B,), 0.5)))
    b0 = bel.init(B)
    beliefs = [b0]
    for t in range(len(steps) - 1):
        beliefs.append(bel.update(beliefs[-1], steps[t], t))
    return bel, beliefs, steps


def _lp_and_rho(bel, beliefs, steps, H=6):
    r = hindsight_learning_progress(beliefs, steps, bel.update, bel.log_prob, horizon=H)
    m, v, lw = bel.moments(beliefs[0], steps[0])
    rho = gaussian_decomposition(m, v, lw)["signal_frac"][:, 0]
    return r["lp"][:, 0], rho, r["n_query"][:, 0]


def test_lp_is_zero_for_pure_noise_and_positive_for_informative_observations():
    B = 512
    bel, beliefs, steps = _scenario(B, support_informative=True)
    lp_inf, rho_inf, nq = _lp_and_rho(bel, beliefs, steps)
    assert torch.all(nq == 6)
    assert lp_inf.mean() > 1.0 and (lp_inf > 0).float().mean() > 0.8
    assert rho_inf.mean() > 0.8
    bel, beliefs, steps = _scenario(B, support_informative=False)
    lp_noise, rho_noise, _ = _lp_and_rho(bel, beliefs, steps)
    assert lp_noise.abs().max() < 1e-5                    # hypotheses agree on noise: the belief does not move
    assert rho_noise.abs().max() < 1e-6
    r_inf = exploration_reward(lp_inf, rho_inf, coef=1.0)
    r_noise = exploration_reward(lp_noise, rho_noise, coef=1.0)
    assert r_inf.mean() > 1.0 and r_noise.abs().max() < 1e-5


def test_noise_correction_removes_credit_when_an_update_reacts_to_noise():
    """A (bad) learned update that shifts hypotheses by the innovation changes predictions after pure noise: the raw
    hindsight LP is then non-zero (and on average harmful), the epistemic correction gives exactly zero."""
    B = 512
    bel, beliefs, steps = _scenario(B, support_informative=False, naive_shift=0.3, seed=1)
    lp, rho, _ = _lp_and_rho(bel, beliefs, steps)
    assert lp.abs().mean() > 0.05 and lp.mean() < 0
    assert noise_corrected_lp(lp, rho).abs().max() < 1e-6
    bel, beliefs, steps = _scenario(B, support_informative=True, naive_shift=0.3, seed=1)
    lp_i, rho_i, _ = _lp_and_rho(bel, beliefs, steps)
    assert noise_corrected_lp(lp_i, rho_i).mean() > 0.5


def test_queries_are_disjoint_from_support_and_stay_in_the_episode():
    B = 8
    bel, beliefs, steps = _scenario(B, True, H=4)
    ep = torch.zeros(B, 5, dtype=torch.long)
    ep[:, 2:] = 1                                       # an episode boundary after step 1
    r = hindsight_learning_progress(beliefs, steps, bel.update, bel.log_prob, horizon=4, episode=ep)
    assert torch.all(r["n_query"][:, 0] == 1) and torch.all(r["n_query"][:, 2] == 2)
    # scoring never feeds a query into the belief: the LP of step 0 over query 1 is unchanged when a later query
    # (step 3) is perturbed, which would matter only if step 3 had been absorbed into the scoring belief
    r1 = hindsight_learning_progress(beliefs, steps, bel.update, bel.log_prob, horizon=1)["lp"][:, 0]
    steps2 = list(steps)
    steps2[3] = ScalarStep(steps[3].x, steps[3].y + 50.0, steps[3].noise_sd)
    r2 = hindsight_learning_progress(beliefs, steps2, bel.update, bel.log_prob, horizon=1)["lp"][:, 0]
    assert torch.allclose(r1, r2)

    class S:
        def __init__(self, ids):
            self.event_ids = ids
    with pytest.raises(ValueError):
        hindsight_learning_progress([0, 0], [S(["e1"]), S(["e1"])], lambda b, s, t: b,
                                    lambda b, s, t: torch.zeros(1), 1)


def test_uncertainty_decompositions():
    logw = torch.log(torch.tensor([[0.5, 0.5]]))
    d = gaussian_decomposition(torch.tensor([[[1.0], [3.0]]]), torch.tensor([[[0.5], [1.5]]]), logw)
    assert d["epistemic"].item() == pytest.approx(1.0) and d["aleatoric"].item() == pytest.approx(1.0)
    assert d["total"].item() == pytest.approx(2.0) and d["signal_frac"].item() == pytest.approx(0.5)
    assert d["info_gain"].item() == pytest.approx(0.5 * math.log(2.0))
    p_agree = torch.tensor([[[[0.3, 0.7]], [[0.3, 0.7]]]])          # [B,K,Q,C]
    c = categorical_decomposition(p_agree, logw)
    assert c["epistemic"].item() == pytest.approx(0.0, abs=1e-7) and c["signal_frac"].item() == pytest.approx(0, abs=1e-6)
    p_dis = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    c2 = categorical_decomposition(p_dis, logw)
    assert c2["epistemic"].item() == pytest.approx(math.log(2), rel=1e-4) and c2["signal_frac"].item() == pytest.approx(1)
    assert combine_signal_frac(torch.tensor([[0.0, 0.0]])).item() == 0.0
    assert combine_signal_frac(torch.tensor([[0.0, 1.0]])).item() == 1.0
    lp = torch.log(torch.tensor([[[0.2, 0.5]], [[0.4, 0.1]]])).permute(1, 0, 2)   # [B=1,K=2,Q=2]
    got = mixture_log_prob(lp, logw).item()
    assert got == pytest.approx(math.log(0.5 * 0.2 * 0.5 + 0.5 * 0.4 * 0.1), rel=1e-5)


class _ExperimentChooser:
    """Decision 0: probe or watch (learned logits); later decisions pull a uniformly random arm (scripted)."""

    def __init__(self, seed=0):
        self.theta = torch.zeros(2, requires_grad=True)
        self.gen = torch.Generator().manual_seed(seed)
        self.version = 0

    def initial_state(self, B):
        return {"first": torch.ones(B, dtype=torch.bool)}

    def act(self, state, obs, tools, budget, goals, active, greedy=False):
        B = len(obs)
        p = torch.softmax(self.theta.detach(), -1)
        c = torch.multinomial(p.expand(B, 2), 1, generator=self.gen).squeeze(1)
        arm = torch.randint(0, 6, (B,), generator=self.gen)
        acts = []
        for b in range(B):
            if bool(state["first"][b]):
                acts.append(Action("probe" if int(c[b]) == 0 else "watch"))
            else:
                acts.append(Action("pull", {"arm": int(arm[b])}))
        rec = {"choice": torch.where(state["first"], c, torch.full_like(c, -1))}
        new = {"first": state["first"] & ~active}
        return acts, torch.zeros(B), torch.zeros(B), new, rec


@pytest.mark.parametrize("reward_kind", ["lp", "surprise"])
def test_lp_trained_experimenter_prefers_information_surprise_trained_prefers_noise(reward_kind):
    torch.manual_seed(0)
    B = 64
    ad = ProbeBanditAdapter(B, horizon=4, noise_tv=True)
    runner = TrialRunner(ad, ad.evaluator)
    pol = _ExperimentChooser()
    opt = torch.optim.Adam([pol.theta], lr=0.1)
    belief = ArmBelief(6)
    for it in range(40):
        res = runner.run(pol, list(range(it * B, (it + 1) * B)))
        choice = res.traj.extras["choice"][:, 0]
        if reward_kind == "lp":
            r = arm_learning_progress(res, belief, horizon=3, coef=1.0)["intrinsic"][:, 0]
        else:
            ll0 = torch.stack([belief.loglik(res.observations[b][0]) for b in range(B)])
            r = surprise_reward(torch.logsumexp(belief.init(B) + ll0, -1))
        logp = torch.log_softmax(pol.theta, -1)[choice]
        loss = -((r - r.mean()) * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    p_probe = torch.softmax(pol.theta.detach(), -1)[0].item()
    if reward_kind == "lp":
        assert p_probe > 0.9, p_probe
    else:
        assert p_probe < 0.1, p_probe


def test_intrinsic_reward_is_finite_for_nonfinite_lp_and_zero_for_pure_noise():
    lp = torch.tensor([float("-inf"), float("inf"), float("nan"), 3.0, -50.0])
    rho = torch.tensor([0.0, 0.0, 1.0, 0.5, 1.0])
    assert noise_corrected_lp(lp, rho).tolist() == [0.0, 0.0, 0.0, 1.5, -50.0]
    r = exploration_reward(lp, rho, coef=2.0, clip=10.0)
    assert torch.isfinite(r).all() and r.tolist() == [0.0, 0.0, 0.0, 3.0, -20.0]
    raw = exploration_reward(lp, None, coef=1.0, clip=10.0)
    assert raw.tolist() == [-10.0, 10.0, 0.0, 3.0, -10.0]
    assert torch.equal(exploration_reward(lp, rho, 1.0, enabled=False), torch.zeros(5))
    # unobserved outcomes may carry nan particle log-likelihoods
    plp = torch.tensor([[[0.0, float("nan")], [-1.0, float("nan")]]])
    out = mixture_log_prob(plp, torch.log(torch.tensor([[0.5, 0.5]])), torch.tensor([[True, False]]))
    assert torch.isfinite(out).all()
