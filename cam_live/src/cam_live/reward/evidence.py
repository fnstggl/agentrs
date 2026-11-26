"""cam_live.reward.evidence — reproduce the measured numbers of Workstream C (diagnostics, not live evidence).

    python -m cam_live.reward.evidence --out cam_live/docs/evidence/C [--device cuda] [--reps 200000]

Writes ``exact_meta_gradient.json`` (finite-difference agreement, unbiasedness with z-scores and relative error,
truncation bias, meta-training on key-to-door MDPs, honest comparison with a magnitude-tuned extrinsic reward) and
``meta_cost.json`` (time and memory of the CAM-scale engine per K and unroll mode).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any, Dict

import torch

from ..diagnostics import tabular as X
from . import exact as E
from . import meta as M
from .bandit_lifetime import BanditLifetime

DT = X.DT


def exact_report(reps: int = 200_000) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    mdp = X.key_to_door(n_distract=2, key_cost=0.3, apple_reward=0.2)
    th0 = torch.zeros(mdp.S, mdp.A, dtype=DT)
    psi = 0.3 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=DT, generator=torch.Generator().manual_seed(0))
    fd = {}
    for K in (1, 2, 4, 8):
        g, J = E.exact_meta_gradient(mdp, psi, th0, K, 1.0)
        f = E.finite_difference_gradient(mdp, psi, th0, K, 1.0)
        fd[K] = {"J": J, "grad_norm": float(g.norm()), "rel_err_vs_fd": float((g - f).norm() / g.norm()),
                 "max_abs_err_vs_fd": float((g - f).abs().max()),
                 "grad_abs_max_reward_detached": float(E.exact_meta_gradient(
                     mdp, psi, th0, K, 1.0, detach_reward=True)[0].abs().max()),
                 "grad_abs_max_unroll_stopped": float(E.exact_meta_gradient(
                     mdp, psi, th0, K, 1.0, stop_unroll=True)[0].abs().max())}
    out["finite_difference_agreement"] = {"mdp": mdp.name, "alpha": 1.0, "per_K": fd}

    tree = X.random_tree(depth=2, branching=1, n_actions=2, seed=3)
    t0 = torch.zeros(tree.S, tree.A, dtype=DT)
    tpsi = 0.5 * torch.randn(tree.T, tree.S, tree.A, dtype=DT, generator=torch.Generator().manual_seed(1))
    unb = []
    for K, n in ((1, 2), (2, 2), (4, 2), (8, 1)):
        ex, J = E.enumerated_sampled_gradient(tree, tpsi, t0, K, 0.7, n)
        gen = torch.Generator().manual_seed(100 + K)
        row: Dict[str, Any] = {"K": K, "n_inner": n, "n_query": 2, "reps": reps,
                               "gap_to_exact_update_objective": float(
                                   (ex - E.exact_meta_gradient(tree, tpsi, t0, K, 0.7)[0]).norm() / ex.norm())}
        for label, kw in (("with_score_term", {}), ("without_score_term", {"inner_score": False}),
                          ("self_inclusive_outer_baseline", {"outer_baseline": "mean"})):
            s = E.mc_summary(E.sampled_meta_gradient(tree, tpsi, t0, K, 0.7, n, 2, gen, reps=reps,
                                                     score_baseline=J, **kw), ex)
            row[label] = {k: s[k] for k in ("rel_err", "max_abs_z", "cos", "norm_ratio", "rel_se")}
        unb.append(row)
    out["sampled_estimator_vs_enumerated_truth"] = {"mdp": tree.name, "alpha": 0.7, "rows": unb}

    door = X.key_to_door(n_distract=3, key_cost=0.3, apple_reward=0.2)
    d0 = torch.zeros(door.S, door.A, dtype=DT)
    dpsi = 0.3 * torch.randn(door.T, door.S, door.A, dtype=DT, generator=torch.Generator().manual_seed(0))
    out["truncation_bias"] = {"mdp": door.name, "alpha": 0.5, **{
        f"{obj}_L{L}": E.truncation_bias(door, dpsi, d0, L, 0.5, Ks=(1, 2, 4, 8), objective=obj)
        for obj in ("sum", "final") for L in (8, 16)}}

    train = [X.key_to_door(n_distract=3, key_cost=kc, apple_reward=ar) for kc in (0.1, 0.3, 0.5)
             for ar in (0.1, 0.3)]
    held = X.key_to_door(n_distract=3, key_cost=0.4, apple_reward=0.2)
    K, alpha, steps = 3, 1.0, 8
    mt = E.meta_train(train, K, alpha, 150, lr=0.1, bound=1.0, theta_init_scale=0.3, seed=0)
    r_learn = E.reward_table(mt["psi"], bound=1.0)
    r_ext = E.extrinsic_reward_table(held)
    h0 = torch.zeros(held.S, held.A, dtype=DT)
    matched = E.scale_matched(held, r_learn, r_ext)
    curves = {"extrinsic": E.learning_curve_exact(held, r_ext, h0, steps, alpha),
              "learned_scale_matched": E.learning_curve_exact(held, matched, h0, steps, alpha)}
    for c in (2, 4, 8, 16, 32):
        curves[f"extrinsic_x{c}"] = E.learning_curve_exact(held, c * r_ext, h0, steps, alpha)
    gen = torch.Generator().manual_seed(0)
    sampled = {}
    for n in (4, 16):
        for name, table in (("extrinsic", None), ("learned_scale_matched", matched)):
            m, se = E.learning_curve_sampled(held, h0, steps, alpha, n, 4000, gen, r_table=table)
            sampled[f"{name}_n{n}"] = {"mean": m, "se": se}
        for c in (4, 16, 32):
            m, se = E.learning_curve_sampled(held, h0, steps, alpha * c, n, 4000, gen, r_table=None)
            sampled[f"extrinsic_step_x{c}_n{n}"] = {"mean": m, "se": se}
    best_ext = max((v[K] for k, v in curves.items() if k.startswith("extrinsic")))
    out["meta_training_key_to_door"] = {
        "train_mdps": [m.name + f"_kc{float(-m.reward_mean[1])}_ar{float(m.reward_mean[2])}" for m in train],
        "held_out": "key_cost=0.4, apple_reward=0.2 (never used in meta-training)", "K": K, "alpha": alpha,
        "meta_objective_first10_mean": sum(mt["curve"][:10]) / 10, "meta_objective_last10_mean": sum(mt["curve"][-10:]) / 10,
        "learned_reward_key_pickup": float(r_learn[0, 0, 0]), "learned_reward_no_pickup": float(r_learn[0, 0, 1]),
        "J_ext_after_k_exact_updates": curves, "J_ext_sampled_learners": sampled,
        "J_at_K_learned_minus_extrinsic": curves["learned_scale_matched"][K] - curves["extrinsic"][K],
        "J_at_K_learned_minus_best_magnitude_tuned_extrinsic": curves["learned_scale_matched"][K] - best_ext,
        "sampled_learners_J_at_K": {k: v["mean"][K] for k, v in sampled.items()},
        "learned_beats_extrinsic_at_equal_step_and_magnitude": bool(
            curves["learned_scale_matched"][K] > curves["extrinsic"][K]
            and all(sampled[f"learned_scale_matched_n{n}"]["mean"][K] > sampled[f"extrinsic_n{n}"]["mean"][K]
                    for n in (4, 16))),
        "learned_beats_best_magnitude_tuned_extrinsic_exact_updates": bool(
            curves["learned_scale_matched"][K] > best_ext),
        "learned_beats_every_step_scaled_extrinsic_sampled_updates": {
            f"n{n}": bool(all(sampled[f"learned_scale_matched_n{n}"]["mean"][K]
                              > sampled[f"extrinsic_step_x{c}_n{n}"]["mean"][K] for c in (4, 16, 32)))
            for n in (4, 16)},
        "note": "extrinsic_x<c> multiplies the extrinsic reward (equivalently the step size) by an oracle-chosen "
                "constant; the booleans above are computed from the curves, not asserted."}
    return out


def cost_report(device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"device": device, "dtype": "float32", "configs": []}
    cuda = device.startswith("cuda")
    for label, hidden, depth, rows_n in (("theta_1.9M", 1024, 2, 64), ("theta_10M", 2048, 3, 64),
                                         ("theta_1.9M_rows2048", 1024, 2, 2048)):
        P = BanditLifetime(n_ctx=256, n_arms=16, d_x=768, d_signal=33, hidden=hidden, reward_hidden=512, seed=0,
                           dtype=torch.float32, device=device, depth=depth)
        n_theta = sum(p.numel() for p in P.policy.parameters())
        n_psi = sum(p.numel() for p in P.reward.parameters())
        g = torch.Generator().manual_seed(0)
        sup = [P.batch(torch.randint(P.n_ctx, (rows_n,), generator=g).to(device),
                       torch.randint(P.n_arms, (rows_n,), generator=g).to(device)) for _ in range(8)]
        qc, qa = torch.randint(P.n_ctx, (64,), generator=g).to(device), torch.randint(P.n_arms, (64,), generator=g).to(device)
        query = M.QueryData(batch=P.batch(qc, qa), utility=P.utility[qc, qa])
        mg = M.MetaGradient(M.policy_gradient_inner_loss(P.logp_fn, P.reward_fn), P.logp_fn, inner_lr=0.05)
        rows = []
        ref: Dict[int, torch.Tensor] = {}
        for mode in M.MODES:
            for K in M.SUPPORTED_K:
                times, peak = [], 0.0
                for rep in range(4):
                    if cuda:
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                    t = time.perf_counter()
                    tr = mg.inner_update(P.theta(), P.psi(), sup, K=K, mode=mode)
                    res = mg.meta_gradient(tr, query)
                    if cuda:
                        torch.cuda.synchronize()
                        peak = max(peak, (torch.cuda.max_memory_allocated() - base) / 2 ** 20)
                    if rep:
                        times.append(time.perf_counter() - t)
                    gflat = M.flat(res.grads).double().cpu()
                    del tr, res
                if mode == "unroll":
                    ref[K] = gflat
                rows.append({"mode": mode, "K": K, "seconds_median": statistics.median(times),
                             "peak_extra_memory_mb": peak if cuda else None,
                             "rel_diff_vs_unroll": float((gflat - ref[K]).norm() / ref[K].norm())})
        out["configs"].append({"label": label, "theta_parameters": n_theta, "psi_parameters": n_psi,
                               "support_rows_per_step": rows_n, "query_rows": 64, "rows": rows})
        del P, sup, query, mg
        if cuda:
            torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reps", type=int, default=200_000)
    ap.add_argument("--only", choices=("exact", "cost"), default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)
    cmd = f"python -m cam_live.reward.evidence --out {a.out} --device {a.device} --reps {a.reps}"
    if a.only in (None, "exact"):
        t = time.time()
        rep = exact_report(a.reps)
        rep["command"], rep["seconds"], rep["torch"] = cmd, time.time() - t, torch.__version__
        with open(os.path.join(a.out, "exact_meta_gradient.json"), "w") as f:
            json.dump(rep, f, indent=1)
    if a.only in (None, "cost"):
        if a.device.startswith("cuda"):
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
            if free < 5.0:
                raise SystemExit(f"only {free:.1f} GB free on the shared GPU; rerun later or with --device cpu")
        rep = cost_report(a.device)
        rep["command"], rep["torch"] = cmd, torch.__version__
        with open(os.path.join(a.out, "meta_cost.json"), "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
