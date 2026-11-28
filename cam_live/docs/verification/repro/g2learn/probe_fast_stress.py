"""G2-learn probe: the Bayesian fast layer (NA-07) under extreme observations, after A's fixes of G-nn D1 / D2 / D4 / D5.

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_fast_stress.py

Independent references (numpy / scipy only):
  1. GLM mode search (Poisson family = counts and time-to-event, binomial = binary) on a grid of extreme scalar
     problems (linear predictor -30..30, prior variance 1e-8..1e6, counts up to 1e9, exposures 1e-12..1e15, a year in
     seconds, 1e9 trials, dispersion 0.3..3 for counts): ``glm_update``'s mode against a float64 bisection of the
     same stationarity condition run to the last bit; the convergence flag must be set.
  2. Implicit-function gradient of the mode: autograd d eta*/d mu and d eta*/d cov against the closed-form implicit
     derivative at the reference root.
  3. ``online_nll`` (the ALPaCA training path) on extreme sequences: autograd gradient w.r.t. the prior mean equals
     central finite differences; per-row NLL equals the stateful ``FastBeliefState.update`` surprises.
  4. Inferred-noise Gaussian (NIG): ``predict()`` variance equals the Student-t variance and the Monte-Carlo variance
     of the posterior predictive; correctly standardised Student-t intervals cover 90 % on data from the prior.
  5. Dispersion: binary and time-to-event updates do not depend on the count dispersion; count updates do.
  6. Impossible observations raise and leave the state untouched; possible boundary observations are accepted.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import check, finish, info  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import special, stats  # noqa: E402

from cam_live.contracts import OutcomeKind as K  # noqa: E402
from cam_live.nn.fast import FastBeliefState, FastPrior, glm_update, online_nll  # noqa: E402

torch.set_num_threads(4)
F64 = torch.float64


# ---------------------------------------------------------------------------------------------------- reference
def ref_root(family, eta0, s, c, e, disp, iters=3000):
    """Bisection (vectorised, float64) of g(x)/disp - (x - eta0)/s = 0, g = c - e exp(x) (Poisson family) or
    c - e sigmoid(x) (binomial). The root lies between eta0 and eta0 + s g(eta0)/disp (concavity)."""
    def g(x):
        if family == "poisson":
            with np.errstate(over="ignore"):
                return c - np.exp(np.minimum(np.log(e) + x, 709.0))
        return np.where(x > 0, (c - e) + e * special.expit(-x), c - e * special.expit(x))

    def f(x):
        return g(x) / disp - (x - eta0) / s

    f0 = f(eta0)
    a = eta0.copy()
    b = eta0 + s * f0
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        lo = np.where(fm > 0, mid, lo)
        hi = np.where(fm > 0, hi, mid)
        lo = np.where(fm == 0, mid, lo)
        hi = np.where(fm == 0, mid, hi)
    x = 0.5 * (lo + hi)
    # curvature h = -g'(x)
    if family == "poisson":
        h = np.exp(np.minimum(np.log(e) + x, 709.0))
    else:
        h = e * special.expit(x) * special.expit(-x)
    return x, h, g(x)


def run_family(kind, rows):
    """rows: list of (eta0, s, y, exposure, censored, disp) -> glm_update on [N, 1] features (s = cov)."""
    arr = np.array(rows, dtype=np.float64)
    eta0, s, y, ex, ce, disp = (arr[:, i] for i in range(6))
    N = len(rows)
    mu = torch.tensor(eta0, dtype=F64).reshape(N, 1).requires_grad_(True)
    cov = torch.tensor(s, dtype=F64).reshape(N, 1, 1).requires_grad_(True)
    phi = torch.ones(N, 1, dtype=F64)
    yt = torch.tensor(y, dtype=F64)
    ext = torch.tensor(ex, dtype=F64)
    cet = torch.tensor(ce, dtype=F64) if kind == K.TIME_TO_EVENT else None
    dt = torch.tensor(disp, dtype=F64)
    mu_n, cov_n, inf = glm_update(mu, cov, phi, yt, kind, ext, cet, dt)
    eta = inf["eta"]
    # reference on the same scalar problem
    if kind == K.COUNT:
        fam, c, e, d = "poisson", y, ex, disp
    elif kind == K.TIME_TO_EVENT:
        fam, c, e, d = "poisson", 1.0 - ce, y / ex, np.ones_like(y)
    else:
        fam, c, e, d = "binomial", y, ex, np.ones_like(y)
    xr, hr, gr = ref_root(fam, eta0, s, c, e, d)
    err = np.abs(eta.detach().numpy() - xr) / (1.0 + np.abs(xr))
    conv = inf["converged"].numpy().astype(bool)
    iters = inf["newton_iters"].numpy()
    # implicit derivative at the reference root
    denom = hr / d + 1.0 / s
    d_eta0 = (1.0 / s) / denom
    # at the root (x - eta0) / s = g(x) / disp: take whichever of the two forms is better conditioned in float64
    # (x - eta0 cancels when s is tiny; g(x) = c - e exp(x) cancels when the count is large)
    cond_diff = np.maximum(np.abs(xr), np.abs(eta0)) / np.maximum(np.abs(xr - eta0), 1e-300)
    cond_g = (np.abs(c) + np.abs(c - gr)) / np.maximum(np.abs(gr), 1e-300)
    if fam == "binomial":                         # g is formed without cancellation for the binomial family
        cond_g = np.zeros_like(cond_g)
    num = np.where(cond_diff < cond_g, (xr - eta0) / s, gr / d)
    d_s = (num / s) / denom
    g_mu, g_cov = torch.autograd.grad(eta.sum(), [mu, cov])
    a_eta0 = g_mu.reshape(-1).numpy()
    a_s = g_cov.reshape(-1).numpy()
    rel_g0 = np.abs(a_eta0 - d_eta0) / (np.abs(d_eta0) + 1e-12)
    rel_gs = np.abs(a_s - d_s) / (np.abs(d_s) + 1e-12 * (1 + np.abs(d_s)))
    consist = np.abs(mu_n.detach().numpy()[:, 0] - eta.detach().numpy()) / (1 + np.abs(xr))
    return dict(err=err, conv=conv, iters=iters, rel_g0=rel_g0, rel_gs=rel_gs, consist=consist, xr=xr, eta0=eta0,
                s=s, y=y, ex=ex, ce=ce, disp=disp, d_eta0=d_eta0, d_s=d_s, a_eta0=a_eta0, a_s=a_s)


def report(name, out, tol=1e-9, gtol=1e-6):
    n = len(out["err"])
    worst = int(np.argmax(out["err"]))
    w = {k: float(out[k][worst]) for k in ("eta0", "s", "y", "ex", "ce", "disp", "xr")}
    check(f"{name}: mode equals the float64 reference root on {n} extreme problems (rel. error <= {tol:g})",
          bool((out["err"] <= tol).all()), f"max rel. error {out['err'].max():.3g} at {w}")
    check(f"{name}: every mode search reports convergence", bool(out["conv"].all()),
          f"unconverged {int((~out['conv']).sum())} of {n}; max iterations {int(out['iters'].max())}")
    check(f"{name}: the posterior mean moves to the mode (phi^T mu' = eta*)", bool((out["consist"] <= 1e-9).all()),
          f"max {out['consist'].max():.3g}")
    # gradients: only where the derivative is not vanishingly small relative to float precision
    ok0 = out["rel_g0"] <= gtol
    bad0 = np.where(~ok0)[0]
    check(f"{name}: implicit-function gradient d eta*/d eta0 equals the closed form (rel. {gtol:g})", bool(ok0.all()),
          f"max rel. {out['rel_g0'].max():.3g}" + (f"; worst case {[(float(out['eta0'][i]), float(out['s'][i]), float(out['y'][i]), float(out['ex'][i]), float(out['a_eta0'][i]), float(out['d_eta0'][i])) for i in bad0[:3]]}" if len(bad0) else ""))
    big = np.abs(out["d_s"]) > 1e-10
    okS = (out["rel_gs"] <= gtol) | ~big
    badS = np.where(~okS)[0]
    check(f"{name}: implicit-function gradient d eta*/d s equals the closed form (rel. {gtol:g}, |d|>1e-10)",
          bool(okS.all()), f"max rel. {out['rel_gs'][big].max() if big.any() else 0:.3g}" +
          (f"; worst {[(float(out['eta0'][i]), float(out['s'][i]), float(out['y'][i]), float(out['ex'][i]), float(out['a_s'][i]), float(out['d_s'][i])) for i in badS[:3]]}" if len(badS) else ""))
    info(f"{name}: iterations (median / max)", (float(np.median(out["iters"])), int(out["iters"].max())))


ETA0 = [-30.0, -5.0, 0.0, 5.0, 30.0]
SS = [1e-8, 1e-2, 1.0, 1e2, 1e6]

# 1 + 2. counts
rows = []
for eta0 in ETA0:
    for s in SS:
        for c in [0.0, 1.0, 3.0, 1e3, 1e9]:
            for e in [1e-12, 1e-3, 1.0, 1e3, 1e7, 1e12, 1e15]:
                for disp in [0.3, 1.0, 3.0]:
                    rows.append((eta0, s, c, e, 0.0, disp))
out = run_family(K.COUNT, rows)
report("count", out)
bad = np.where(out["err"] > 1e-6)[0]
info("count: problems whose mode is wrong by > 1e-6 while 'converged' is True (eta0, s, count, exposure, disp, "
     "true mode, reported mode)",
     [(float(out["eta0"][i]), float(out["s"][i]), float(out["y"][i]), float(out["ex"][i]), float(out["disp"][i]),
       round(float(out["xr"][i]), 4), float(out["xr"][i] + np.sign(1) * out["err"][i] * (1 + abs(out["xr"][i]))))
      for i in bad if out["conv"][i]])
# a single reproducible counterexample through the stateful layer the agent runs (FastBeliefState.update)
for (pv, dsp) in ((100.0, 0.3), (1e6, 1.0)):
    fp_ce = FastPrior(1)                          # (FastPrior(prior_var=1e6) itself overflows in _inv_softplus)
    with torch.no_grad():
        fp_ce.disp_raw.fill_(math.log(dsp))
    st_ce = FastBeliefState(fp_ce, change_h=0.0)
    st_ce._prior[K.COUNT]["mu"] = torch.tensor([-30.0], dtype=F64)
    st_ce._prior[K.COUNT]["cov"] = torch.tensor([[pv]], dtype=F64)
    st_ce.reset(K.COUNT)
    r_ce = st_ce.update(torch.tensor([1.0], dtype=F64), 1e9, K.COUNT, exposure=1e-12)
    eta_ce = float(st_ce.posterior(K.COUNT)["mu"][0])
    s_ce = float(st_ce._prior[K.COUNT]["cov"][0, 0])
    xr_ce = ref_root("poisson", np.array([-30.0]), np.array([s_ce]), np.array([1e9]), np.array([1e-12]),
                     np.array([dsp]))[0][0]
    m_ce = float(st_ce.predict(torch.tensor([1.0], dtype=F64), K.COUNT, exposure=1e-12)[0])
    check(f"stateful FastBeliefState.update, count far above the prior predictor (prior eta -30, variance {s_ce:g}, "
          f"dispersion {dsp}, count 1e9 at exposure 1e-12): the mode is reached or non-convergence is reported",
          abs(eta_ce - xr_ce) <= 1e-6 * (1 + abs(xr_ce)) or not r_ce["converged"],
          f"posterior eta {eta_ce:.6g} (true mode {xr_ce:.6g}), converged={r_ce['converged']}, iterations "
          f"{r_ce['newton_iters']}, next predicted count {m_ce:.4g}")

# time-to-event (exposure = time unit; censored flag); disp is ignored by the layer for this kind
rows = []
for eta0 in ETA0:
    for s in SS:
        for t in [1e-6, 1.0, 3600.0, 86400.0, 2.592e6, 3.1536e7, 3.1536e9]:
            for unit in [1.0, 3600.0]:
                for cen in [0.0, 1.0]:
                    rows.append((eta0, s, t, unit, cen, 1.0))
report("time-to-event", run_family(K.TIME_TO_EVENT, rows))

# binary (exposure = trials)
rows = []
for eta0 in ETA0:
    for s in SS:
        for n in [1.0, 10.0, 1e6, 1e9]:
            for c in sorted({0.0, 1.0, math.floor(n / 2), n - 1.0, n}):
                if 0 <= c <= n:
                    rows.append((eta0, s, c, n, 0.0, 1.0))
report("binary", run_family(K.BINARY, rows))

# 3. online_nll on extreme sequences: gradient by finite differences; agreement with the stateful path
rng = np.random.default_rng(3)
Fd = 3


def prior_dict(mu, cov, disp=1.0):
    return {"mu": mu, "cov": cov, "V": cov / 0.25, "a": torch.tensor(2.0, dtype=F64), "b": torch.tensor(0.5, dtype=F64),
            "noise": torch.tensor(0.25, dtype=F64), "dispersion": torch.tensor(disp, dtype=F64)}


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


seqs = {
    "time-to-event censored at a year in seconds": (K.TIME_TO_EVENT, np.array([3.1536e7, 86400.0, 3.1536e7, 600.0, 3.1536e7]),
                                                    np.ones(5), np.array([1.0, 0.0, 1.0, 0.0, 1.0])),
    "count 0..3 with exposure 1e7": (K.COUNT, np.array([3.0, 0.0, 1.0, 0.0, 2.0]), np.full(5, 1e7), None),
    "count 0 with exposure 1e9": (K.COUNT, np.array([0.0, 0.0, 1.0, 0.0, 0.0]), np.full(5, 1e9), None),
    "binary 0 of 1e6 trials": (K.BINARY, np.array([0.0, 1.0, 0.0, 3.0, 0.0]), np.full(5, 1e6), None),
}
for name, (kind, y, ex, ce) in seqs.items():
    phi = torch.tensor(unit(rng.standard_normal((5, Fd))), dtype=F64)
    mu0 = torch.tensor(rng.standard_normal(Fd) * 0.3, dtype=F64)
    cov0 = torch.eye(Fd, dtype=F64)
    yt = torch.tensor(y, dtype=F64)
    ext = torch.tensor(ex, dtype=F64)
    cet = None if ce is None else torch.tensor(ce, dtype=F64)
    mu_p = mu0.clone().requires_grad_(True)
    nll = online_nll(prior_dict(mu_p, cov0), phi, yt, kind, exposure=ext, censored=cet)
    total = nll.sum()
    g = torch.autograd.grad(total, mu_p)[0].numpy()
    fd = np.zeros(Fd)

    def L(dm):
        return float(online_nll(prior_dict(mu0 + dm, cov0), phi, yt, kind, exposure=ext, censored=cet).sum())
    for j in range(Fd):
        h = 1e-4
        dm = torch.zeros(Fd, dtype=F64)
        dm[j] = h
        fd[j] = (-L(2 * dm) + 8 * L(dm) - 8 * L(-dm) + L(-2 * dm)) / (12 * h)     # 4th-order central stencil
    rel = float(np.abs(g - fd).max() / (np.abs(fd).max() + 1e-12))
    check(f"online_nll gradient w.r.t. the prior mean equals finite differences: {name}", rel < 1e-6,
          f"rel. {rel:.2e}; total NLL {float(total):.4f}")
    # stateful path with the same prior
    fp = FastPrior(Fd, kinds=(K.REAL, K.COUNT, K.BINARY, K.TIME_TO_EVENT))
    st = FastBeliefState(fp, forgetting=1.0, change_h=0.0)
    p = st._prior[kind]
    p["mu"] = mu0.clone()
    p["cov"] = cov0.clone()
    st.reset(kind)
    sur = []
    for t in range(5):
        r = st.update(phi[t], float(y[t]), kind, exposure=float(ex[t]),
                      censored=None if ce is None else bool(ce[t]))
        sur.append(r["surprise"])
    d = float(np.abs(np.array(sur) - nll.detach().numpy()).max())
    check(f"stateful FastBeliefState.update surprises equal online_nll per row: {name}", d < 1e-8, f"max |diff| {d:.2e}")

# 4. NIG predictive variance and calibration
pnig = FastPrior(6, noise=0.4, noise_mode="nig")
pg = {k: v.detach().numpy() for k, v in pnig.prior(K.REAL).items()}
cover, ratio_var, mc_rel = [], [], []
rng = np.random.default_rng(11)
for task in range(400):
    s2 = 1.0 / rng.gamma(float(pg["a"]), 1.0 / float(pg["b"]))
    w = rng.multivariate_normal(pg["mu"], s2 * pg["V"])
    Pt = unit(rng.standard_normal((25, 6)))
    yt = Pt @ w + math.sqrt(s2) * rng.standard_normal(25)
    st = FastBeliefState(pnig, forgetting=1.0, change_h=0.0)
    for i in range(25):
        m, e, nz = (float(v) for v in st.predict(torch.tensor(Pt[i]), K.REAL))
        post = st._post[K.REAL]
        a, b = float(post["a"]), float(post["b"])
        q = float(torch.tensor(Pt[i]) @ post["cov"] @ torch.tensor(Pt[i]))
        scale2 = (b / a) * (1 + q)
        nu = 2 * a
        ratio_var.append((e + nz) / (scale2 * nu / (nu - 2)))
        z = (yt[i] - m) / math.sqrt(scale2)
        cover.append(abs(z) < stats.t.ppf(0.95, nu))
        if task < 10 and i in (8, 20):          # dof 2a >= 12: the sample variance has a finite variance
            sw = st.sample_weights(200000, K.REAL, generator=torch.Generator().manual_seed(task * 100 + i)).numpy()
            # predictive draw: w ~ posterior, s2 drawn jointly inside sample_weights; noise with the same s2 is not
            # returned, so compare the epistemic part only: Var(phi^T w) = (b/(a-1)) q
            mc = float(np.var(sw @ Pt[i]))
            mc_rel.append(abs(mc - e) / e)
        st.update(torch.tensor(Pt[i]), float(yt[i]), K.REAL)
rv = np.array(ratio_var)
check("NIG predict(): epistemic + noise equals the Student-t variance (b/(a-1))(1+q)", bool(np.allclose(rv, 1.0, rtol=1e-9)),
      f"ratio min {rv.min():.12f} max {rv.max():.12f}")
check("NIG predict(): epistemic variance equals the Monte-Carlo variance of phi^T w under sample_weights (200000 draws, 3 %)",
      max(mc_rel) < 0.03, f"max rel. diff {max(mc_rel):.3f} over {len(mc_rel)} cases")
cv = float(np.mean(cover))
se = math.sqrt(0.09 / len(cover))
check("NIG calibration on data from the prior: Student-t 90 % interval (scale (b/a)(1+q), dof 2a) covers 0.90 +- 3 se",
      abs(cv - 0.90) < 3 * se, f"coverage {cv:.4f} (n {len(cover)}, se {se:.4f})")

# 5. dispersion tempers counts only
fp = FastPrior(4)
with torch.no_grad():
    fp.disp_raw.fill_(math.log(3.13))
fp1 = FastPrior(4)
rng = np.random.default_rng(5)
phi = torch.tensor(unit(rng.standard_normal((30, 4))), dtype=F64)
for kind, y, ex, ce in ((K.BINARY, 1.0, 1.0, None), (K.TIME_TO_EVENT, 50.0, 10.0, False), (K.COUNT, 7.0, 2.0, None)):
    sa, sb = FastBeliefState(fp, change_h=0.0), FastBeliefState(fp1, change_h=0.0)
    for t in range(30):
        sa.update(phi[t], y, kind, exposure=ex, censored=ce)
        sb.update(phi[t], y, kind, exposure=ex, censored=ce)
    d = float((sa.posterior(kind)["mu"] - sb.posterior(kind)["mu"]).abs().max())
    if kind == K.COUNT:
        check("dispersion 3.13 changes count updates (it is the count dispersion)", d > 1e-3, f"max |dmu| {d:.3g}")
    else:
        check(f"dispersion 3.13 leaves {kind} updates unchanged", d == 0.0, f"max |dmu| {d:.3g}")

# 6. impossible observations
fp = FastPrior(3)
cases = [("negative count", K.COUNT, -1.0, 1.0, None), ("count > 0 at zero exposure", K.COUNT, 2.0, 0.0, None),
         ("negative exposure", K.COUNT, 1.0, -1.0, None), ("successes > trials", K.BINARY, 3.0, 2.0, None),
         ("negative successes", K.BINARY, -1.0, 2.0, None), ("negative time", K.TIME_TO_EVENT, -5.0, 1.0, False),
         ("zero time unit", K.TIME_TO_EVENT, 5.0, 0.0, False), ("censoring flag 0.5", K.TIME_TO_EVENT, 5.0, 1.0, 0.5),
         ("NaN exposure", K.COUNT, 1.0, float("nan"), None), ("infinite outcome", K.REAL, float("inf"), None, None)]
for name, kind, y, ex, ce in cases:
    st = FastBeliefState(fp, change_h=0.0)
    before = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st._post[kind].items()}
    raised = False
    try:
        st.update(torch.tensor([0.6, 0.0, 0.8], dtype=F64), y, kind, exposure=ex, censored=ce)
    except ValueError:
        raised = True
    same = all((torch.equal(before[k], st._post[kind][k]) if torch.is_tensor(before[k]) else before[k] == st._post[kind][k])
               for k in before)
    check(f"impossible observation refused, state untouched: {name}", raised and same, f"raised={raised} unchanged={same}")
ok_cases = [("count 0 at zero exposure", K.COUNT, 0.0, 0.0, None), ("0 successes of 0 trials", K.BINARY, 0.0, 0.0, None),
            ("fractional successes 0.5 of 1", K.BINARY, 0.5, 1.0, None), ("time 0 with event", K.TIME_TO_EVENT, 0.0, 1.0, False)]
for name, kind, y, ex, ce in ok_cases:
    st = FastBeliefState(fp, change_h=0.0)
    try:
        r = st.update(torch.tensor([0.6, 0.0, 0.8], dtype=F64), y, kind, exposure=ex, censored=ce)
        good = math.isfinite(r["surprise"]) and bool(torch.isfinite(st.posterior(kind)["mu"]).all())
        check(f"possible boundary observation accepted with a finite update: {name}", good,
              f"surprise {r['surprise']:.4g}, converged {r.get('converged')}")
    except Exception as e:  # noqa: BLE001
        check(f"possible boundary observation accepted with a finite update: {name}", False, f"{type(e).__name__}: {e}")
# masked rows may carry placeholders (only existing rows are checked)
phi = torch.tensor(unit(np.random.default_rng(1).standard_normal((3, 3))), dtype=F64)
try:
    out = online_nll(prior_dict(torch.zeros(3, dtype=F64), torch.eye(3, dtype=F64)), phi,
                     torch.tensor([1.0, -7.0, 2.0], dtype=F64), K.COUNT, mask=torch.tensor([True, False, True]))
    check("online_nll ignores impossible values in masked rows (0 NLL there)", float(out[1]) == 0.0, f"{out.tolist()}")
except Exception as e:  # noqa: BLE001
    check("online_nll ignores impossible values in masked rows (0 NLL there)", False, f"{type(e).__name__}: {e}")

sys.exit(finish())
