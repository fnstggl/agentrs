"""Meta-gradient exactness measured by relative error and z-scores, never cosine alone (IR-06, Exp J)."""
import torch

from superlearner.cam_unified.credit import exact as X
from superlearner.cam_unified.credit import lifetime_exact as LE
from superlearner.cam_unified.credit import meta_exact as M

torch.set_num_threads(4)


def test_zheng_full_backprop_is_unbiased_and_truncation_is_biased_despite_high_cosine():
    eta = 0.3 * torch.randn(3, 3, dtype=LE.DT, generator=torch.Generator().manual_seed(1))
    g, J = LE.exact_grad(eta)
    e2 = eta.clone()
    e2[0, 2] += 1e-5
    e3 = eta.clone()
    e3[0, 2] -= 1e-5
    fd = (float(LE.exact_J(e2)) - float(LE.exact_J(e3))) / 2e-5
    assert abs(fd - float(g[0, 2])) < 1e-8
    full = LE.compare(*LE.mc_estimate(eta, 60000, seed=0), g)
    tr1 = LE.compare(*LE.mc_estimate(eta, 60000, trunc=1, seed=0), g)
    assert full["max_abs_z"] < 5.0 and full["rel_err"] < 0.05
    assert tr1["rel_err"] > 0.3 and tr1["max_abs_z"] > 10.0
    assert tr1["cos"] > 0.99            # why cosine tests cannot detect truncation bias


def test_sampled_meta_gradient_with_inner_score_term_is_unbiased_for_k_ge_2():
    mdp = X.key_to_door(n_distract=4, key_cost=0.3, apple_reward=0.2, door_reward=1.0)
    th0 = torch.zeros(mdp.S, mdp.A, dtype=X.DT)
    psi = 0.1 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=X.DT, generator=torch.Generator().manual_seed(0))
    K = 3
    ge, J = M.exact_meta_gradient(mdp, psi, th0, K, 1.0)
    gen = torch.Generator().manual_seed(7)
    rep = 400
    G = torch.stack([M.sampled_meta_gradient(mdp, psi, th0, K, 1.0, 64, 256, gen, score_baseline=J)
                     for _ in range(rep)]).reshape(rep, -1)
    m, se = G.mean(0), G.std(0) / rep ** 0.5
    ex = ge.reshape(-1)
    mask = (ex.abs() > 1e-6) | (m.abs() > 1e-6)
    z = float(((m - ex)[mask] / se[mask].clamp_min(1e-12)).abs().max())
    assert z < 5.0, z
    assert float((m - ex).norm() / ex.norm()) < 0.05
