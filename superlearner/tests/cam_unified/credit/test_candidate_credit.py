"""CAM-scale restricted candidate-set COCOA / R-HCA estimators (CR-06, CR-07) against exact references on a tabular
candidate wrapper: each state offers K candidates drawn from a proposal; the selector is softmax over the
candidates' base logits. Expectations are EXACT (enumeration of every trajectory of the wrapped process)."""
import pytest
import torch

from superlearner.cam_unified.credit import candidates as C
from superlearner.cam_unified.credit import exact as X
from superlearner.cam_unified.credit.heads import HindsightModel

torch.set_num_threads(4)


@pytest.fixture(scope="module", params=[0.0, 0.6])
def wrapped(request):
    base = X.noisy_key_to_door(sigma0=request.param, n_distract=1)
    cm = C.wrap(base, K=2, seed=3)
    theta = 0.5 * torch.randn(base.S, base.A, dtype=X.DT, generator=torch.Generator().manual_seed(0))
    batch, probs = X.enumerate_batch(cm.mdp, cm.aug_theta(theta))
    w, rho = C.exact_coefficients(cm, theta)
    return cm, theta, batch, probs, w, rho


def _expected_grad(cm, theta, batch, probs, fn):
    th = theta.detach().clone().requires_grad_(True)
    st = cm.steps(batch, th)
    s = fn(st)
    (g,) = torch.autograd.grad((probs * s).sum(), th)
    return g


def test_wrapper_is_a_valid_process_and_selector_gradient_is_nonzero(wrapped):
    cm, theta, batch, probs, w, rho = wrapped
    assert abs(float(probs.sum()) - 1) < 1e-12
    g = cm.exact_gradient(theta)
    assert float(g.norm()) > 1e-3
    # the wrapped J equals the base J under the induced (marginal) policy only through the candidates: sanity
    J = float(cm.expected_return(theta))
    assert float((probs * batch.r.sum(1)).sum()) == pytest.approx(J, abs=1e-12)


def test_cocoa_candidate_estimator_is_exactly_unbiased(wrapped):
    cm, theta, batch, probs, w, rho = wrapped
    wp = C.batch_coefficients(w, batch)
    g = _expected_grad(cm, theta, batch, probs,
                       lambda st: C.cocoa_surrogate(st["sel_logits"], st["cand_mask"], st["chosen"], batch.r, wp))
    ex = cm.exact_gradient(theta)
    assert float((g - ex).norm() / ex.norm()) < 1e-10


def test_rhca_candidate_estimator_is_exactly_unbiased(wrapped):
    cm, theta, batch, probs, w, rho = wrapped
    rp = C.batch_coefficients(rho, batch)
    g = _expected_grad(cm, theta, batch, probs,
                       lambda st: C.rhca_surrogate(st["sel_logits"], st["cand_mask"], batch.r, rp))
    ex = cm.exact_gradient(theta)
    assert float((g - ex).norm() / ex.norm()) < 1e-10


def test_wrong_coefficients_are_biased_so_the_test_has_power(wrapped):
    cm, theta, batch, probs, w, rho = wrapped
    zero = torch.zeros_like(C.batch_coefficients(rho, batch))
    g = _expected_grad(cm, theta, batch, probs,
                       lambda st: C.rhca_surrogate(st["sel_logits"], st["cand_mask"], batch.r, zero))
    ex = cm.exact_gradient(theta)
    assert float((g - ex).norm() / ex.norm()) > 0.1


def _per_traj_key_grad(cm, theta, batch, fn):
    """Per-trajectory gradient on the t=0 row of theta [N, A]. Each trajectory's surrogate depends only on its own
    selector scores, so one backward through leaf scores gives every per-trajectory gradient."""
    st = cm.steps(batch, theta.detach())
    lg = st["sel_logits"].detach().clone().requires_grad_(True)
    fn({**st, "sel_logits": lg}).sum().backward()
    g0 = lg.grad[:, 0, :]                                               # d s_n / d score of each t=0 slot
    out = torch.zeros(g0.shape[0], cm.base.A, dtype=g0.dtype)
    out.scatter_add_(1, st["cand"][:, 0, :], g0)                         # slot score j = theta[s0, C_j]
    return out


def test_rhca_key_decision_variance_is_invariant_to_immediate_noise_on_candidates():
    """R-HCA's defining property (Prop. 2-3) carries over to candidate sets: the key decision's variance does not
    grow with the noise of its own immediate reward, COCOA's does (exact variances by enumeration)."""
    out = {}
    for s0 in (0.0, 1.5):
        base = X.noisy_key_to_door(sigma0=s0, n_distract=1)
        cm = C.wrap(base, K=2, seed=3)
        theta = 0.5 * torch.randn(base.S, base.A, dtype=X.DT, generator=torch.Generator().manual_seed(0))
        batch, probs = X.enumerate_batch(cm.mdp, cm.aug_theta(theta))
        w, rho = C.exact_coefficients(cm, theta)
        wp, rp = C.batch_coefficients(w, batch), C.batch_coefficients(rho, batch)
        res = {}
        for name, fn in (("cocoa", lambda st: C.cocoa_surrogate(st["sel_logits"], st["cand_mask"], st["chosen"],
                                                                 batch.r, wp)),
                         ("rhca", lambda st: C.rhca_surrogate(st["sel_logits"], st["cand_mask"], batch.r, rp))):
            G = _per_traj_key_grad(cm, theta, batch, fn)
            m = (probs[:, None] * G).sum(0)
            res[name] = float((probs * ((G - m) ** 2).sum(1)).sum())
        out[s0] = res
    assert out[1.5]["rhca"] == pytest.approx(out[0.0]["rhca"], rel=1e-9, abs=1e-12)
    assert out[1.5]["cocoa"] > out[0.0]["cocoa"] + 0.1


class _ExactHindsight:
    """Test double: a hindsight 'model' returning the exact hindsight log-probabilities of the wrapped process."""

    def __init__(self, cm, theta, inclusive):
        D, Dbar = X.future_outcome_counts(cm.mdp, cm.aug_theta(theta))
        pi = X.policy(cm.aug_theta(theta))
        if inclusive:
            onehot = torch.nn.functional.one_hot(cm.mdp.outcome, cm.mdp.U).to(X.DT)
            D = D + torch.einsum("tsap,tsapu->tsau", cm.mdp.P, onehot)
        h = pi[None, :, :, None] * D / torch.einsum("sa,tsau->tsu", pi, D).clamp_min(1e-300)[:, :, None, :]
        self.h = h                                                         # [T, S~, K, U]

    def logits(self, b_t, u, e_g, cand, cand_mask, logp_policy, elapsed):
        t, s, uk = b_t[:, 0].long(), b_t[:, 1].long(), u[:, 0].long()
        return self.h[t, s, :, uk].clamp_min(1e-300).log()


def test_pair_coefficients_from_a_hindsight_model_reproduce_the_exact_estimator(wrapped):
    cm, theta, batch, probs, w, rho = wrapped
    N, T = batch.s.shape
    K = cm.K
    sel = X.policy(cm.aug_theta(theta))[batch.s]                          # [N, T, K]
    b_t = torch.stack([torch.arange(T)[None, :].expand(N, T), batch.s], -1).to(X.DT)
    u = batch.u[..., None].to(X.DT)
    cand = torch.zeros(N, T, K, 1, dtype=X.DT)
    mask = torch.ones(N, T, K, dtype=torch.bool)
    for method, table, incl in (("cocoa", w, False), ("rhca", rho, True)):
        hm = _ExactHindsight(cm, theta, incl)
        pc = C.pair_coefficients(hm, b_t, u, cand, mask, sel.log(), method=method, clip=1e9).to(X.DT)
        ref = C.batch_coefficients(table, batch)
        lo = 1 if method == "cocoa" else 0
        tri = (torch.arange(T)[None, :] >= torch.arange(T)[:, None] + lo)[None, :, None, :]
        reachable = (sel[..., None] > 0) & tri
        # where the outcome is reachable from s~_t the learned path equals the exact coefficient
        ok = reachable & (ref != 0) if method == "cocoa" else reachable
        assert float((pc - ref)[ok].abs().max()) < 1e-6, method        # float32 path (CAM precision)


def test_coverage_report_counts_candidates_and_hindsight_mass():
    sel = torch.tensor([[[0.7, 0.3, 0.0], [0.5, 0.5, 0.0]]])
    mask = torch.tensor([[[True, True, False], [True, True, False]]])
    chosen = torch.tensor([[0, 1]])
    h = torch.tensor([[[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]]])
    rep = C.coverage_report(sel, chosen, mask, h_probs=h, better_in_set=torch.tensor([[True, False]]))
    assert rep["K_hist"] == {2: 2} and rep["n"] == 2
    assert rep["hindsight_mass_on_executed"] == pytest.approx(0.85)
    assert rep["better_action_in_set_frac"] == pytest.approx(0.5)
