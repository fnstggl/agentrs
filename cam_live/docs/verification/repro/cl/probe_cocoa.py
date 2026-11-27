"""G-cl probe: COCOA finite-action reference (RC-04) and the CAM extension's refusals (RC-05).

Independent of D's code for the ground truth: trajectories are enumerated here, the exact policy gradient is the
autograd gradient of the enumerated J(theta), the contribution coefficients are recomputed from the enumeration as
w(t,s,a,u) = p(A_t=a | S_t=s, U_{t+k}=u for some k>=1, counted with multiplicity) / pi(a|s) - 1 (Meulemans et al.
2023, Def. 3 / Eq. 4), and the COCOA estimator is re-implemented from Eq. 4. D's code is used only as the object under
test (``cocoa_grad``, ``contribution_coefficients``, ``reinforce_grad``, ``check_support``, ``HindsightModel``).

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/cl/probe_cocoa.py [--seeds 200]
Prints [PASS]/[FAIL] per check and a JSON summary; exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch

sys.path.insert(0, "cam_live/src")
from cam_live.diagnostics import tabular as tb                     # noqa: E402  (MDP builders + sampler only)
from cam_live.credit import estimators as est                      # noqa: E402  (under test)
from cam_live.credit.cocoa import HindsightModel, check_support    # noqa: E402  (under test)

DT = torch.float64
FAILS = []
OUT = {}


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------ own enumeration / ground truth
def enumerate_traj(mdp, pi):
    """List of (s[T], a[T], u[T], p) by brute force (independent of tabular.enumerate_batch)."""
    T = mdp.T
    out = []

    def rec(t, s, S, A, U, p):
        if t == T:
            out.append((list(S), list(A), list(U), p))
            return
        for a in range(mdp.A):
            pa = float(pi[s, a])
            if pa <= 0:
                continue
            for s2 in range(mdp.S):
                ps = float(mdp.P[t, s, a, s2])
                if ps <= 0:
                    continue
                rec(t + 1, s2, S + [s], A + [a], U + [int(mdp.outcome[t, s, a, s2])], p * pa * ps)
    for s0 in range(mdp.S):
        if float(mdp.init[s0]) > 0:
            rec(0, s0, [], [], [], float(mdp.init[s0]))
    return out


def exact_grad_own(mdp, theta):
    """d/dtheta sum_traj P_theta(traj) G(traj) by autograd through the enumerated trajectory probabilities."""
    th = theta.clone().requires_grad_(True)
    pi = torch.softmax(th, -1)
    trajs = enumerate_traj(mdp, pi.detach())
    J = th.new_zeros(())
    for S, A, U, p in trajs:
        lp = sum(torch.log(pi[s, a]) for s, a in zip(S, A))
        p_pol = torch.exp(lp)
        env = p / float(p_pol.detach())            # environment factor (init x transitions): independent of theta
        G = float(sum(float(mdp.reward_mean[u]) for u in U))
        J = J + p_pol * env * G
    (g,) = torch.autograd.grad(J, th)
    return g, trajs


def own_coefficients(mdp, pi, trajs):
    """w[t,s,a,u] from the enumeration; plus a mask of (t,s,u) that occur."""
    T, S, A, U = mdp.T, mdp.S, mdp.A, mdp.U
    N = torch.zeros(T, S, A, U, dtype=DT)
    for Ss, As, Us, p in trajs:
        for t in range(T):
            for k in range(t + 1, T):
                N[t, Ss[t], As[t], Us[k]] += p
    Nbar = N.sum(2)                                                 # [T, S, U]
    occ = torch.zeros(T, S, dtype=DT)
    for Ss, As, Us, p in trajs:
        for t in range(T):
            occ[t, Ss[t]] += p
    ok = Nbar > 1e-15
    h = torch.where(ok[:, :, None, :], N / Nbar.clamp_min(1e-300)[:, :, None, :], torch.zeros_like(N))
    w = torch.where(ok[:, :, None, :], h / pi[None, :, :, None] - 1.0, torch.zeros_like(N))
    return w, ok


def own_cocoa_per_traj(mdp, pi, w, S, A, U, R):
    """Eq. 4: sum_t grad log pi(A_t|S_t) R_t + sum_t sum_a grad pi(a|S_t) sum_{k>=1} w(S_t,a,U_{t+k}) R_{t+k}."""
    g = torch.zeros(mdp.S, mdp.A, dtype=DT)
    T = len(S)
    for t in range(T):
        s, a = S[t], A[t]
        e = torch.zeros(mdp.A, dtype=DT); e[a] = 1
        g[s] += (e - pi[s]) * R[t]
        for b in range(mdp.A):
            coef = sum(float(w[t, s, b, U[k]]) * R[k] for k in range(t + 1, T))
            eb = torch.zeros(mdp.A, dtype=DT); eb[b] = 1
            g[s] += pi[s, b] * (eb - pi[s]) * coef
    return g


def batch_from(trajs, mdp, pi):
    s = torch.tensor([t[0] for t in trajs]); a = torch.tensor([t[1] for t in trajs]); u = torch.tensor([t[2] for t in trajs])
    r = mdp.reward_mean[u].to(DT)
    return tb.Batch(s, a, u, r, torch.log(pi[s, a])), torch.tensor([t[3] for t in trajs], dtype=DT)


def own_sample(mdp, pi, n, g):
    T = mdp.T
    s = torch.multinomial(mdp.init.expand(n, -1), 1, generator=g).squeeze(1)
    S_, A_, U_, R_ = [], [], [], []
    for t in range(T):
        a = torch.multinomial(pi[s], 1, generator=g).squeeze(1)
        s2 = torch.multinomial(mdp.P[t, s, a], 1, generator=g).squeeze(1)
        u = mdp.outcome[t, s, a, s2]
        r = mdp.reward_mean[u].clone()
        if mdp.reward_noise > 0:
            r = r + mdp.reward_noise * torch.randn(n, dtype=DT, generator=g)
        S_.append(s); A_.append(a); U_.append(u); R_.append(r)
        s = s2
    st = lambda xs: torch.stack(xs, 1)
    S, A = st(S_), st(A_)
    return tb.Batch(S, A, st(U_), st(R_), torch.log(pi[S, A]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    torch.manual_seed(0)
    mdps = {
        "key_to_door_d3": tb.key_to_door(n_distract=3, apple_prob=0.4, key_cost=0.2),
        "noisy_key_to_door": tb.noisy_key_to_door(n_distract=2),
        "random_tree_s3": tb.random_tree(depth=3, branching=2, n_actions=3, n_outcomes=4, seed=3),
        "random_tree_s7": tb.random_tree(depth=3, branching=2, n_actions=2, n_outcomes=3, seed=7),
    }
    for name, mdp in mdps.items():
        g = torch.Generator().manual_seed(11)
        theta = 0.8 * torch.randn(mdp.S, mdp.A, dtype=DT, generator=g)
        pi = torch.softmax(theta, -1)
        exact, trajs = exact_grad_own(mdp, theta)
        d_exact = tb.exact_gradient(mdp, theta)
        check(f"{name}: D's exact_gradient equals own enumeration", float((d_exact - exact).abs().max()) < 1e-12,
              f"max diff {float((d_exact - exact).abs().max()):.2e}")
        w_own, ok = own_coefficients(mdp, pi, trajs)
        w_d = tb.contribution_coefficients(mdp, theta)
        # compare only on (t, s, u) that occur with positive probability
        reach = ok[:, :, None, :].expand_as(w_own)
        diff = float((w_own - w_d)[reach].abs().max()) if reach.any() else 0.0
        check(f"{name}: D's contribution coefficients equal own p(a|s,u)/pi-1", diff < 1e-10, f"max diff {diff:.2e}")
        batch, probs = batch_from(trajs, mdp, pi)
        # exact expectation of D's estimator and of own re-implementation
        gd = est.cocoa_grad(theta, batch, w_d)
        Eg_d = (probs[:, None, None] * gd).sum(0)
        rb_d = float(((Eg_d - exact) ** 2).sum() / (exact ** 2).sum())
        own = torch.stack([own_cocoa_per_traj(mdp, pi, w_own, t[0], t[1], t[2],
                                              [float(mdp.reward_mean[u]) for u in t[2]]) for t in trajs])
        Eg_own = (probs[:, None, None] * own).sum(0)
        rb_own = float(((Eg_own - exact) ** 2).sum() / (exact ** 2).sum())
        per_traj_diff = float((own - gd).abs().max())
        check(f"{name}: D's cocoa_grad per trajectory equals own Eq.4 implementation", per_traj_diff < 1e-10,
              f"max diff {per_traj_diff:.2e}")
        check(f"{name}: exact expectation of D's COCOA = exact gradient (rel bias)", rb_d < 1e-20, f"{rb_d:.2e}")
        check(f"{name}: exact expectation of own COCOA = exact gradient (rel bias)", rb_own < 1e-20, f"{rb_own:.2e}")
        # exact variance vs REINFORCE (same enumeration)
        gr = est.reinforce_grad(theta, batch)
        var = lambda G: float((probs * ((G - (probs[:, None, None] * G).sum(0)) ** 2).sum((1, 2))).sum())
        v_c, v_r = var(gd), var(gr)
        OUT[name] = {"rel_bias_cocoa_exact": rb_d, "rel_bias_own": rb_own, "var_cocoa": v_c, "var_reinforce": v_r,
                     "n_traj": len(trajs), "grad_norm2": float((exact ** 2).sum())}
        print(f"       {name}: exact variance trace COCOA {v_c:.4g} vs REINFORCE {v_r:.4g}")
        # Monte Carlo with D's estimator on trajectories from an own sampler. z uses the EXACT per-component
        # variance from the enumeration, so E[z^2] = 1 + n * bias^2 / var exactly (no small-sample SE artefacts).
        Eg = (probs[:, None, None] * gd).sum(0).reshape(-1)
        vj = (probs[:, None] * (gd.reshape(len(trajs), -1) - Eg) ** 2).sum(0)
        live = vj > 1e-14
        zsq, means = [], []
        for sd in range(args.seeds):
            gg = torch.Generator().manual_seed(1000 + sd)
            b = own_sample(mdp, pi, args.n, gg)
            per = est.cocoa_grad(theta, b, w_d).reshape(args.n, -1)
            means.append(per.mean(0))
            zsq.append((((per.mean(0) - exact.reshape(-1)) / (vj / args.n).sqrt().clamp_min(1e-300))[live]) ** 2)
        Z = torch.stack(zsq)                                    # [seeds, comps]
        msq = float(Z.mean())
        sd_msq = float(Z.mean(1).std() / math.sqrt(Z.shape[0]))       # seeds are independent, components are not
        grand = torch.stack(means).mean(0)
        zp = ((grand - exact.reshape(-1)) / (vj / (args.n * args.seeds)).sqrt().clamp_min(1e-300))[live]
        OUT[name].update({"mc_mean_sq_z": msq, "mc_mean_sq_z_se": sd_msq, "mc_n_z": int(Z.numel()),
                          "mc_pooled_max_abs_z": float(zp.abs().max()), "mc_pooled_n_components": int(live.sum()),
                          "mc_samples": args.n * args.seeds})
        check(f"{name}: MC mean z^2 (exact variance) over {args.seeds} seeds x {int(live.sum())} comps ~ 1",
              abs(msq - 1) < 4 * sd_msq + 0.02, f"{msq:.3f} +- {sd_msq:.3f}")
        bonf = math.sqrt(2) * torch.special.erfinv(torch.tensor(1 - 0.01 / max(1, int(live.sum())), dtype=DT)).item()
        check(f"{name}: pooled MC mean ({args.n * args.seeds} traj) within Bonferroni 1% band",
              float(zp.abs().max()) < bonf, f"max|z|={float(zp.abs().max()):.2f} < {bonf:.2f}")
    # noisy rewards: MC only (enumeration impossible); coefficients are defined over outcomes, noise is independent
    mdp = tb.key_to_door(n_distract=4, apple_prob=0.5, reward_noise=0.5)
    g = torch.Generator().manual_seed(5)
    theta = 0.5 * torch.randn(mdp.S, mdp.A, dtype=DT, generator=g)
    exact = tb.exact_gradient(mdp, theta)
    w_d = tb.contribution_coefficients(mdp, theta)
    pi = torch.softmax(theta, -1)
    means_c, means_r, vc, vr = [], [], [], []
    for sd in range(args.seeds):
        b = own_sample(mdp, pi, args.n, torch.Generator().manual_seed(5000 + sd))
        pc = est.cocoa_grad(theta, b, w_d).reshape(args.n, -1)
        pr = est.reinforce_grad(theta, b).reshape(args.n, -1)
        means_c.append(pc.mean(0)); means_r.append(pr.mean(0))
        vc.append(float(pc.var(0).sum())); vr.append(float(pr.var(0).sum()))
    for nm, M in (("cocoa", torch.stack(means_c)), ("reinforce", torch.stack(means_r))):
        pooled, se = M.mean(0), M.std(0, unbiased=True) / math.sqrt(M.shape[0])
        live = se > 1e-12
        zp = ((pooled - exact.reshape(-1)) / se.clamp_min(1e-300))[live]
        check(f"key_to_door_d4 noise 0.5: pooled MC z of {nm} (max |z| over {int(live.sum())} comps)",
              float(zp.abs().max()) < 4.0, f"{float(zp.abs().max()):.2f}")
        OUT.setdefault("key_to_door_d4_noisy", {})[f"pooled_max_abs_z_{nm}"] = float(zp.abs().max())
    OUT["key_to_door_d4_noisy"]["var_trace_cocoa"] = sum(vc) / len(vc)
    OUT["key_to_door_d4_noisy"]["var_trace_reinforce"] = sum(vr) / len(vr)
    check("key_to_door_d4 noise 0.5: COCOA variance < REINFORCE variance", sum(vc) < sum(vr),
          f"{sum(vc)/len(vc):.3f} vs {sum(vr)/len(vr):.3f}")

    # --------------------------------------------------------------------------- reproduce D's measured numbers
    # D.json measured: exact_rel_var_reinforce 23.46, exact_rel_var_cocoa 10.52 (noisy_key_to_door sigma0=0.6,
    # theta = 0.5 * randn(seed 0)); key_to_door_key_decision_var_reinforce 8.087 / _cocoa 0.0593 (key_to_door d8,
    # noise 0.3, theta seed 4, 60k samples). Recomputed here with own enumeration / own estimator / own sampler.
    mdp = tb.noisy_key_to_door(sigma0=0.6)
    theta = 0.5 * torch.randn(mdp.S, mdp.A, generator=torch.Generator().manual_seed(0), dtype=DT)
    pi = torch.softmax(theta, -1)
    exact, trajs = exact_grad_own(mdp, theta)
    w_own, _ = own_coefficients(mdp, pi, trajs)
    probs = torch.tensor([t[3] for t in trajs], dtype=DT)
    own = torch.stack([own_cocoa_per_traj(mdp, pi, w_own, t[0], t[1], t[2], [float(mdp.reward_mean[u]) for u in t[2]])
                       for t in trajs])
    def own_reinforce(S, A, U):
        R = [float(mdp.reward_mean[u]) for u in U]
        g = torch.zeros(mdp.S, mdp.A, dtype=DT)
        for t in range(len(S)):
            e = torch.zeros(mdp.A, dtype=DT); e[A[t]] = 1
            g[S[t]] += (e - pi[S[t]]) * sum(R[t:])
        return g
    rf = torch.stack([own_reinforce(t[0], t[1], t[2]) for t in trajs])
    gn = float((exact ** 2).sum())
    rv = lambda G: float((probs * ((G - (probs[:, None, None] * G).sum(0)) ** 2).sum((1, 2))).sum()) / gn
    rv_r, rv_c = rv(rf), rv(own)
    OUT["repro_exact_rel_var_reinforce"] = rv_r
    OUT["repro_exact_rel_var_cocoa"] = rv_c
    check("reproduce exact_rel_var_reinforce = 23.46", abs(rv_r - 23.46) < 0.01, f"{rv_r:.4f}")
    check("reproduce exact_rel_var_cocoa = 10.52", abs(rv_c - 10.52) < 0.01, f"{rv_c:.4f}")
    mdp = tb.key_to_door(n_distract=8, reward_noise=0.3)
    theta = 0.5 * torch.randn(mdp.S, mdp.A, generator=torch.Generator().manual_seed(4), dtype=DT)
    pi = torch.softmax(theta, -1)
    w_d = tb.contribution_coefficients(mdp, theta)
    b = own_sample(mdp, pi, 60_000, torch.Generator().manual_seed(77))
    vr = float(est.reinforce_grad(theta, b)[:, 0].var(0).sum())
    vc = float(est.cocoa_grad(theta, b, w_d)[:, 0].var(0).sum())
    OUT["repro_key_decision_var_reinforce"] = vr
    OUT["repro_key_decision_var_cocoa"] = vc
    check("reproduce key_to_door_key_decision_var_reinforce = 8.087 (MC, own sampler, 3%)", abs(vr / 8.0869 - 1) < 0.03, f"{vr:.4f}")
    check("reproduce key_to_door_key_decision_var_cocoa = 0.0593 (MC, own sampler, 5%)", abs(vc / 0.05928 - 1) < 0.05, f"{vc:.5f}")

    # --------------------------------------------------------------------------- biased-coefficient control
    # The z-test above must be able to fail: perturb w by +0.3 on one action and check that it is detected.
    mdp = mdps["key_to_door_d3"]
    g = torch.Generator().manual_seed(11)
    theta = 0.8 * torch.randn(mdp.S, mdp.A, dtype=DT, generator=g)
    pi = torch.softmax(theta, -1)
    exact = tb.exact_gradient(mdp, theta)
    w_bad = tb.contribution_coefficients(mdp, theta).clone()
    w_bad[0, :, 0, :] += 0.3
    M = []
    for sd in range(args.seeds):
        b = own_sample(mdp, pi, args.n, torch.Generator().manual_seed(1000 + sd))
        M.append(est.cocoa_grad(theta, b, w_bad).reshape(args.n, -1).mean(0))
    M = torch.stack(M)
    se = M.std(0, unbiased=True) / math.sqrt(M.shape[0]); live = se > 1e-12
    zbad = float((((M.mean(0) - exact.reshape(-1)) / se.clamp_min(1e-300))[live]).abs().max())
    check("control: a +0.3 bias in w is detected by the same z-test", zbad > 6, f"max|z|={zbad:.1f}")
    OUT["control_biased_w_max_abs_z"] = zbad

    # --------------------------------------------------------------------------- refusals (RC-05)
    m = torch.ones(1, 3, dtype=torch.bool)
    good = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    cases = {
        "missing": (None, {}),
        "nan": (torch.tensor([[float("nan"), -1.0, -1.0]]), {}),
        "p_above_1": (torch.tensor([[0.1, -3.0, -3.0]]), {}),
        "not_normalised": (torch.log(torch.tensor([[0.5, 0.2, 0.1]])), {}),
        "near_zero": (torch.log(torch.tensor([[0.9995, 0.0004, 0.0001]])), {}),
        "neg_inf": (torch.tensor([[0.0, float("-inf"), float("-inf")]]), {}),
        "open_untracked": (good, {"open_ended": True, "tracked_proposals": False}),
    }
    for nm, (lp, kw) in cases.items():
        rep = check_support(lp, m, **kw)
        check(f"check_support refuses {nm}", rep.status[0] == "unsupported", f"reasons={rep.reasons[0]}")
    rep = check_support(good, m)
    check("check_support accepts a valid logged distribution", rep.status[0] == "supported", str(rep.reasons[0]))
    # the model-level refusal yields NaN, not a number
    hm = HindsightModel(4, 3, 5, d_hidden=16, d_key=8, n_members=3)
    b = torch.randn(2, 4); u = torch.randn(2, 3); cand = torch.randn(2, 3, 5); mm = torch.ones(2, 3, dtype=torch.bool)
    lp2 = torch.stack([good[0], torch.log(torch.tensor([0.9995, 0.0004, 0.0001]))])
    cc = hm.contribution(b, u, cand, mm, lp2, outcome_value=torch.tensor([1.0, 1.0]))
    check("HindsightModel.contribution: unsupported row is NaN", bool(torch.isnan(cc.counterfactual_estimate[1]).all())
          and bool(torch.isfinite(cc.counterfactual_estimate[0]).all()), str(cc.status))
    cc0 = hm.contribution(b, u, cand, mm, None)
    check("HindsightModel.contribution: missing propensities -> all NaN", bool(torch.isnan(cc0.counterfactual_estimate).all()))
    # untrained model claims nothing (w = 0)
    check("untrained hindsight model: w == 0", float(cc.w[0].abs().max()) < 1e-6, f"{float(cc.w[0].abs().max()):.2e}")
    # Fail-open default: a decision whose candidates were generated open-endedly but that carries no
    # diagnostics flag is treated as supported (the flag defaults to closed / tracked).
    rep = check_support(good, m)          # what HindsightProvider passes for diagnostics without "open_ended"
    OUT["open_ended_flag_absent_status"] = rep.status[0]
    print(json.dumps(OUT, indent=1, default=float))
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)


if __name__ == "__main__":
    main()
