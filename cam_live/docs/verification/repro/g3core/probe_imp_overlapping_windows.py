"""G3-core probe IMP-1: does the ImaginationCalibration's posterior stay calibrated when its verified windows overlap?

``LatentPlanner.refresh_calibration`` (predict/planner.py:239-299) adds one observation per executed decision t with
target y_h(t) = sum_{j=0..h} verified_progress(t + j) — consecutive targets share h of their h + 1 terms — and
``SelectionCalibration.posterior`` (predict/policy.py) treats the observations as independent (Bayesian linear
regression with i.i.d. Gaussian noise). When the regressor m (the chosen candidate's imagined preference) is
autocorrelated across consecutive decisions, the residual correlation inflates the true sampling variance of beta_im
and the posterior sd is too small.

Simulation with the real class (no model): z_t ~ N(0,1) i.i.d. (one-step standardised value of the chosen candidate),
m_t AR(1) with coefficient rho (imagined preference of the chosen candidate), per-decision verified progress
y_t = 0.3 z_t + N(0, 1): imagination carries NO information (true beta_im = 0 for every h). Per run: T decisions,
windows of horizon h fed exactly like refresh_calibration (each decision once, overlapping). Reports the share of runs
whose 95 % posterior interval for beta_im excludes 0, the share with P(beta_im > 0) > 0.975, and the mean
imagination weight lambda_im = beta_im^+ / max(beta_q, 1) drawn by the rule (predict.policy.imagination_weight).
Control: the same data with non-overlapping windows (every (h+1)-th decision).

    PYTHONPATH=<freeze>/cam_live/src python probe_imp_overlapping_windows.py --out out/imp_overlapping_windows.json
"""
import argparse
import json
import math

import torch


def run(h, rho, T, reps, overlap, seed=0):
    from cam_live.predict.policy import ImaginationCalibration, imagination_weight
    g = torch.Generator().manual_seed(seed)
    excl = pos = 0
    lam_means = []
    sds = []
    means = []
    for r in range(reps):
        z = torch.randn(T + h + 1, generator=g, dtype=torch.float64)
        e = torch.randn(T + h + 1, generator=g, dtype=torch.float64)
        m = torch.zeros(T + h + 1, dtype=torch.float64)
        inn = torch.randn(T + h + 1, generator=g, dtype=torch.float64)
        for t in range(T + h + 1):
            m[t] = (rho * m[t - 1] if t else 0.0) + math.sqrt(1 - rho * rho) * inn[t]
        y = 0.3 * z + e
        cal = ImaginationCalibration()
        step = 1 if overlap else h + 1
        for t in range(0, T, step):
            cal.observe(f"d{t}", h, float(z[t]), float(m[t]), float(y[t:t + h + 1].sum()))
        s = cal.summary(h)
        b, sd = s["beta_im"], s["beta_im_sd"]
        means.append(b)
        sds.append(sd)
        excl += int(abs(b) > 1.96 * sd)
        pos += int(b > 1.96 * sd)
        draws = cal.sample(h, 4000, torch.Generator().manual_seed(r))
        lam_means.append(float(imagination_weight(draws).mean()))
    n = reps
    emp_sd = (sum((x - sum(means) / n) ** 2 for x in means) / (n - 1)) ** 0.5
    return {"h": h, "rho_m": rho, "T": T, "reps": reps, "overlapping_windows": overlap,
            "share_95pct_interval_excludes_0": excl / n, "share_confidently_positive": pos / n,
            "mean_posterior_sd": sum(sds) / n, "empirical_sd_of_posterior_mean": emp_sd,
            "sd_ratio_empirical_over_posterior": emp_sd / (sum(sds) / n),
            "mean_lambda_im": sum(lam_means) / n, "q90_lambda_im": sorted(lam_means)[int(0.9 * n)],
            "prior_mean_lambda_im": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from cam_live.predict.policy import ImaginationCalibration, imagination_weight
    prior = float(imagination_weight(ImaginationCalibration().sample(4, 20000, torch.Generator().manual_seed(1))).mean())
    res = []
    for h in (1, 4):
        for rho in (0.0, 0.9):
            for ov in (True, False):
                r = run(h, rho, a.T, a.reps, ov)
                r["prior_mean_lambda_im"] = prior
                res.append(r)
                print(json.dumps(r), flush=True)
    json.dump({"probe": "imp_overlapping_windows", "results": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
