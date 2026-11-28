"""DX regression tests for G2L-1 (G2_learn.md): the count mode search reported ``converged=True`` at an absurd mode
(eta 3.3e11 instead of 48.35) when the prior predictor e exp(eta0) lies ~16 orders of magnitude below the count.

Reference: a float64 bisection of the same scalar stationarity condition (independent of nn.fast)."""
import math

import numpy as np
import pytest
import torch

from cam_live.contracts import OutcomeKind
from cam_live.nn import fast
from cam_live.nn.fast import F64, FastBeliefState, FastPrior, glm_update

K = OutcomeKind


def bisect_root(eta0, s, c, e, disp, n_iter=400):
    """Root of c - e exp(x) - (disp / s) (x - eta0) (decreasing in x) by bisection in float64 on a bracket whose
    end points are checked (r(lo) > 0 > r(hi))."""
    k = disp / s

    def r(x):
        lmu = math.log(e) + x
        mu = math.exp(min(lmu, 709.0))
        return c - mu - k * (x - eta0)

    lo, hi = eta0 - 50.0, eta0 + c / k + 50.0          # r(hi) = -e exp(hi) - 50 k < 0
    while r(lo) <= 0:
        lo -= max(50.0, abs(lo))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if r(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# the four grid problems of probe_fast_stress.py that were wrong while 'converged' (eta0, s, count, exposure, disp)
G2L1_CASES = [(-30.0, 100.0, 1e9, 1e-12, 0.3), (-30.0, 1e6, 1e9, 1e-12, 0.3), (-30.0, 1e6, 1e9, 1e-12, 1.0),
              (-30.0, 1e6, 1e9, 1e-12, 3.0)]


@pytest.mark.parametrize("eta0,s,c,e,disp", G2L1_CASES)
def test_count_far_above_the_prior_predictor_reaches_the_mode(eta0, s, c, e, disp):
    mu = torch.tensor([[eta0]], dtype=F64)
    cov = torch.tensor([[[s]]], dtype=F64)
    phi = torch.ones(1, 1, dtype=F64)
    mu_n, _, info = glm_update(mu, cov, phi, torch.tensor([c], dtype=F64), K.COUNT, torch.tensor([e], dtype=F64),
                               None, torch.tensor([disp], dtype=F64))
    ref = bisect_root(eta0, s, c, e, disp)
    eta = float(info["eta"][0])
    assert abs(ref - 48.354) < 1e-2                       # the reference itself (G2_learn.md: 48.35)
    assert abs(eta - ref) <= 1e-9 * (1.0 + abs(ref)), (eta, ref)
    assert bool(info["converged"][0])
    assert abs(float(mu_n[0, 0]) - ref) <= 1e-9 * (1.0 + abs(ref))


@pytest.mark.parametrize("prior_var,disp", [(100.0, 0.3), (1e6, 1.0)])
def test_stateful_update_reaches_the_mode_or_reports_non_convergence(prior_var, disp):
    """The FastBeliefState path the agent runs (the two counterexamples printed by the probe)."""
    fp = FastPrior(1)
    with torch.no_grad():
        fp.disp_raw.fill_(math.log(disp))
    st = FastBeliefState(fp, change_h=0.0)
    st._prior[K.COUNT]["mu"] = torch.tensor([-30.0], dtype=F64)
    st._prior[K.COUNT]["cov"] = torch.tensor([[prior_var]], dtype=F64)
    st.reset(K.COUNT)
    out = st.update(torch.tensor([1.0], dtype=F64), 1e9, K.COUNT, exposure=1e-12)
    eta = float(st.posterior(K.COUNT)["mu"][0])
    d = float(fp.dispersion().detach()) if hasattr(fp, "dispersion") else disp
    ref = bisect_root(-30.0, prior_var, 1e9, 1e-12, d)
    assert out["converged"] and abs(eta - ref) <= 1e-6 * (1.0 + abs(ref)), (eta, ref, out)
    assert eta < 100.0                                    # was 3.3e11 / 1e15


def test_a_stopping_rule_met_away_from_the_root_is_not_reported_as_converged():
    """Convergence detection itself: with a tolerance so loose that the step rule is met after one iteration, the
    returned point is not the root and must not be called converged (the old rule trusted |step| alone)."""
    eta0, s, c, e, disp = (torch.tensor([v], dtype=F64) for v in (0.0, 1.0, 10.0, 1.0, 1.0))
    x, iters, conv = fast._mode_poisson(eta0, s, c, e, disp, 1, 1e3)
    ref = bisect_root(0.0, 1.0, 10.0, 1.0, 1.0)
    assert abs(float(x[0]) - ref) > 1e-6
    assert not bool(conv[0])
    x, iters, conv = fast._mode_poisson(eta0, s, c, e, disp, fast.NEWTON_MAX_ITER, fast.NEWTON_TOL)
    assert bool(conv[0]) and abs(float(x[0]) - ref) <= 1e-12 * (1 + abs(ref))


def test_whole_extreme_count_grid_converges_to_the_reference():
    """The probe's count grid (2625 problems): every mode equals the bisection root and reports convergence."""
    rows = [(eta0, s, c, e, d) for eta0 in (-30.0, -5.0, 0.0, 5.0, 30.0) for s in (1e-8, 1e-2, 1.0, 1e2, 1e6)
            for c in (0.0, 1.0, 3.0, 1e3, 1e9) for e in (1e-12, 1e-3, 1.0, 1e3, 1e7, 1e12, 1e15) for d in (0.3, 1.0, 3.0)]
    a = np.array(rows)
    t = [torch.tensor(a[:, i], dtype=F64) for i in range(5)]
    x, iters, conv = fast._mode_poisson(*t, fast.NEWTON_MAX_ITER, fast.NEWTON_TOL)
    ref = np.array([bisect_root(*r, n_iter=200) for r in rows])
    err = np.abs(x.numpy() - ref) / (1.0 + np.abs(ref))
    assert err.max() <= 1e-9, rows[int(err.argmax())]
    assert bool(conv.all()), [rows[i] for i in np.where(~conv.numpy())[0][:5]]
