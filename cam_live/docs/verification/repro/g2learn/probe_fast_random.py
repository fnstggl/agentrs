"""G2-learn round 2 probe: the GLM mode search after DX's fix of G2L-1, on random problems far outside the earlier grid.

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_fast_random.py [--n 20000]

Log-uniform random scalar problems: linear predictor eta0 in [-60, 60], prior variance 1e-10 .. 1e7, counts 0 or
1 .. 1e12, exposures 1e-15 .. 1e15, dispersion 0.1 .. 10 (counts); time-to-event times 1e-6 .. 1e10 with units
1e-3 .. 1e5 and random censoring; binary with 1 .. 1e12 trials. Reference: float64 bisection of the same stationarity
condition. Checks: every reported-converged mode equals the reference (rel. 1e-9); how many problems are reported
unconverged (a fail-safe answer, but then the update is still applied: counted separately), and whether any of those
was in fact at the root.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import check, finish, info  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import special  # noqa: E402

from cam_live.contracts import OutcomeKind as K  # noqa: E402
from cam_live.nn.fast import glm_update  # noqa: E402

F64 = torch.float64
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20000
rng = np.random.default_rng(20261128)


def ref_root(family, eta0, s, c, e, disp, iters=3000):
    def g(x):
        if family == "poisson":
            with np.errstate(over="ignore"):
                return c - np.exp(np.minimum(np.log(e) + x, 709.0))
        return np.where(x > 0, (c - e) + e * special.expit(-x), c - e * special.expit(x))

    def f(x):
        return g(x) / disp - (x - eta0) / s
    f0 = f(eta0)
    lo, hi = np.minimum(eta0, eta0 + s * f0), np.maximum(eta0, eta0 + s * f0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        lo = np.where(fm > 0, mid, lo)
        hi = np.where(fm > 0, hi, mid)
        lo = np.where(fm == 0, mid, lo)
        hi = np.where(fm == 0, mid, hi)
    return 0.5 * (lo + hi)


def lu(lo, hi, n):
    return 10.0 ** rng.uniform(lo, hi, n)


def run(kind, eta0, s, y, ex, ce, disp):
    n = len(eta0)
    mu = torch.tensor(eta0, dtype=F64).reshape(n, 1)
    cov = torch.tensor(s, dtype=F64).reshape(n, 1, 1)
    _, _, inf = glm_update(mu, cov, torch.ones(n, 1, dtype=F64), torch.tensor(y, dtype=F64), kind,
                           torch.tensor(ex, dtype=F64), None if ce is None else torch.tensor(ce, dtype=F64),
                           torch.tensor(disp, dtype=F64))
    if kind == K.COUNT:
        xr = ref_root("poisson", eta0, s, y, ex, disp)
    elif kind == K.TIME_TO_EVENT:
        xr = ref_root("poisson", eta0, s, 1.0 - ce, y / ex, np.ones(n))
    else:
        xr = ref_root("binomial", eta0, s, y, ex, np.ones(n))
    eta = inf["eta"].numpy()
    err = np.abs(eta - xr) / (1.0 + np.abs(xr))
    conv = inf["converged"].numpy().astype(bool)
    return err, conv, inf["newton_iters"].numpy(), xr, eta


eta0 = rng.uniform(-60, 60, N)
s = lu(-10, 7, N)
cnt = np.where(rng.random(N) < 0.2, 0.0, np.floor(lu(0, 12, N)))
res = {"count": run(K.COUNT, eta0, s, cnt, lu(-15, 15, N), None, lu(-1, 1, N))}
t = lu(-6, 10, N)
res["time_to_event"] = run(K.TIME_TO_EVENT, eta0, s, t, lu(-3, 5, N), (rng.random(N) < 0.5).astype(float), np.ones(N))
ntr = np.floor(lu(0, 12, N))
succ = np.floor(rng.random(N) * (ntr + 1))
succ = np.where(rng.random(N) < 0.15, 0.0, np.where(rng.random(N) < 0.15, ntr, succ))
res["binary"] = run(K.BINARY, eta0, s, succ, ntr, None, np.ones(N))
for name, (err, conv, it, xr, eta) in res.items():
    bad = conv & (err > 1e-9)
    unconv = ~conv
    at_root = unconv & (err <= 1e-9)
    info(f"{name}: problems / reported unconverged / of those actually at the root / max iterations",
         (N, int(unconv.sum()), int(at_root.sum()), int(it.max())))
    worst = int(np.argmax(np.where(conv, err, 0)))
    check(f"{name}: every mode reported converged equals the float64 bisection root (rel. 1e-9) on {N} random problems",
          not bad.any(), f"{int(bad.sum())} wrong; worst converged rel. error {err[conv].max() if conv.any() else 0:.3g} "
                         f"(eta0 {eta0[worst]:.3g}, s {s[worst]:.3g}, true {xr[worst]:.6g}, reported {eta[worst]:.6g})")
    if unconv.any():
        k = int(np.argmax(np.where(unconv, err, -1)))
        info(f"{name}: worst unconverged problem (eta0, s, true mode, reported, rel. error)",
             (float(eta0[k]), float(s[k]), float(xr[k]), float(eta[k]), float(err[k])))
    check(f"{name}: at most 0.1 % of the random problems are reported unconverged", unconv.mean() <= 1e-3,
          f"{unconv.mean() * 100:.3f} %")
sys.exit(finish())
