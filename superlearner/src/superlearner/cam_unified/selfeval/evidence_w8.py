"""W8 evidence (CPU): numbers behind the W8 tests, written to docs/evidence/dom/W8/*.json.

    python -m superlearner.cam_unified.selfeval.evidence_w8 --out-dir docs/evidence/dom/W8 [--only meta,exf,cand]

* meta  — 2000-repetition z-test of the sampled meta-gradient against the exact tabular meta-gradient (key_to_door
          d=5, K=4, 64 inner / 256 query trajectories): tabular reference with / without the inner-score term and
          the CAM-scale MetaRewardLearner.
* cand  — exact (enumerated) relative bias of the candidate-set COCOA / R-HCA estimators on the tabular candidate
          wrapper, and the key-decision variance as the immediate-reward noise grows.
* exf   — Exp F: inflation reports of a naive reward on all seven integrity worlds; meta-training of R_psi on
          verified utility for easy_useless and noisy_tv with lifetime utilities of fresh learners before / after.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time

import torch
import torch.nn as nn


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _z(G: torch.Tensor, exact: torch.Tensor) -> dict:
    m, se = G.mean(0), G.std(0) / math.sqrt(G.shape[0])
    ex = exact.reshape(-1)
    mask = (ex.abs() > 1e-6) | (m.abs() > 1e-6)
    z = ((m - ex)[mask] / se[mask].clamp_min(1e-12)).abs()
    return {"max_abs_z": float(z.max()), "rel_err": float((m - ex).norm() / ex.norm()),
            "rel_se": float(se.norm() / ex.norm()),
            "cos": float(torch.nn.functional.cosine_similarity(m, ex, 0)), "n_coords": int(mask.sum())}


def meta_regression(rep: int = 2000, K: int = 4, n_inner: int = 64, n_query: int = 256) -> dict:
    from ..credit import exact as X
    from ..credit import meta_exact as M
    from .metagrad import InnerBatch, MetaRewardLearner, QueryBatch
    mdp = X.key_to_door(n_distract=5, key_cost=0.3, apple_reward=0.2, door_reward=1.0)
    th0 = torch.zeros(mdp.S, mdp.A, dtype=X.DT)
    psi = 0.1 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=X.DT, generator=torch.Generator().manual_seed(0))
    ge, J = M.exact_meta_gradient(mdp, psi, th0, K, 1.0)
    out = {"setting": {"mdp": mdp.name, "K": K, "alpha": 1.0, "n_inner": n_inner, "n_query": n_query, "rep": rep,
                       "exact_norm": float(ge.norm()), "J_exact": J}}
    for name, kw in (("tabular_with_inner_score", dict(score_baseline=J)),
                     ("tabular_without_inner_score", dict(inner_score=False))):
        t = time.time()
        gen = torch.Generator().manual_seed(7)
        G = torch.stack([M.sampled_meta_gradient(mdp, psi, th0, K, 1.0, n_inner, n_query, gen, **kw)
                         for _ in range(rep)]).reshape(rep, -1)
        out[name] = {**_z(G, ge), "wall_s": time.time() - t}

    class Pol(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.zeros(mdp.S, mdp.A, dtype=X.DT))

        def forward(self, s):
            return self.theta[s]

    class Rew(nn.Module):
        def __init__(self):
            super().__init__()
            self.psi = nn.Parameter(psi.clone())

        def forward(self, t, s, a):
            return self.psi[t, s, a]

    pol, rew = Pol(), Rew()
    gen = torch.Generator().manual_seed(11)
    t0 = time.time()
    est = []
    for _ in range(rep):
        mrl = MetaRewardLearner(pol, rew, inner_lr=mdp.T, K=K, normalize_inner_adv=False)
        fast = mrl.start()
        for _k in range(K):
            b = X.sample(mdp, fast["theta"].detach(), n_inner, gen)
            tt = torch.arange(mdp.T)[None, :].expand_as(b.s)
            fast, _ = mrl.inner_step(fast, InnerBatch({"s": b.s}, b.a, torch.ones_like(b.s, dtype=torch.bool),
                                                      {"t": tt, "s": b.s, "a": b.a}))
        q = X.sample(mdp, fast["theta"].detach(), n_query, gen)
        rtg = torch.flip(torch.cumsum(torch.flip(q.r, [1]), 1), [1])
        (g,) = mrl.meta_gradient(fast, QueryBatch({"s": q.s}, q.a, torch.ones_like(q.s, dtype=torch.bool), rtg),
                                 score_baseline=J)
        est.append(-g * mdp.T)
    out["cam_meta_reward_learner"] = {**_z(torch.stack(est).reshape(rep, -1), ge), "wall_s": time.time() - t0}
    return out


def candidates() -> dict:
    from ..credit import candidates as C
    from ..credit import exact as X
    out = {}
    for s0 in (0.0, 0.6, 1.5):
        base = X.noisy_key_to_door(sigma0=s0, n_distract=1)
        cm = C.wrap(base, K=2, seed=3)
        theta = 0.5 * torch.randn(base.S, base.A, dtype=X.DT, generator=torch.Generator().manual_seed(0))
        batch, probs = X.enumerate_batch(cm.mdp, cm.aug_theta(theta))
        w, rho = C.exact_coefficients(cm, theta)
        wp, rp = C.batch_coefficients(w, batch), C.batch_coefficients(rho, batch)
        ex = cm.exact_gradient(theta)
        row = {"n_trajectories": int(batch.N), "exact_grad_norm": float(ex.norm())}
        fns = {"cocoa": lambda st: C.cocoa_surrogate(st["sel_logits"], st["cand_mask"], st["chosen"], batch.r, wp),
               "rhca": lambda st: C.rhca_surrogate(st["sel_logits"], st["cand_mask"], batch.r, rp),
               "reinforce": lambda st: C.cocoa_surrogate(st["sel_logits"], st["cand_mask"], st["chosen"],
                                                         torch.flip(torch.cumsum(torch.flip(batch.r, [1]), 1), [1]),
                                                         torch.zeros_like(wp))}
        for name, fn in fns.items():
            th = theta.detach().clone().requires_grad_(True)
            st = cm.steps(batch, th)
            (g,) = torch.autograd.grad((probs * fn(st)).sum(), th)
            st2 = cm.steps(batch, theta.detach())
            lg = st2["sel_logits"].detach().clone().requires_grad_(True)
            fn({**st2, "sel_logits": lg}).sum().backward()
            g0 = torch.zeros(batch.N, base.A, dtype=X.DT).scatter_add_(1, st2["cand"][:, 0, :], lg.grad[:, 0, :])
            m = (probs[:, None] * g0).sum(0)
            row[name] = {"rel_bias_exact": float((g - ex).norm() / ex.norm()),
                         "key_decision_variance": float((probs * ((g0 - m) ** 2).sum(1)).sum())}
        out[f"sigma0={s0}"] = row
    return out


def exf(meta_steps: int = 60) -> dict:
    from ..envs.integrity import VARIANTS
    from . import integrity_harness as H
    naive = torch.tensor([1.0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    out = {"features": list(H.FEATURES), "psi0": naive.tolist(), "inflation_at_psi0": {}}
    for v in VARIANTS:
        out["inflation_at_psi0"][v] = H.inflation_report(v, naive, list(range(200, 216)))
    for v in ("easy_useless", "noisy_tv"):
        t = time.time()
        u0 = H.lifetime_utility(v, naive, lifetimes=8, seed=7)
        mt = H.meta_train(v, naive, meta_steps=meta_steps, seed=0)
        u1 = H.lifetime_utility(v, mt["psi_final"], lifetimes=8, seed=7)
        out[f"meta_{v}"] = {"meta_steps": meta_steps, "lifetime_utility_psi0": u0, "lifetime_utility_psi_final": u1,
                            "psi_final": mt["psi_final"].tolist(), "psi_version": mt["psi_version"],
                            "query_utility_curve": [h["query_utility"] for h in mt["history"]],
                            "inflation_at_psi_final": H.inflation_report(v, mt["psi_final"], list(range(200, 216))),
                            "wall_s": time.time() - t}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/evidence/dom/W8")
    ap.add_argument("--only", default="meta,cand,exf")
    ap.add_argument("--rep", type=int, default=2000)
    ap.add_argument("--meta-steps", type=int, default=60)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = {"meta": lambda: meta_regression(a.rep), "cand": candidates, "exf": lambda: exf(a.meta_steps)}
    names = {"meta": "meta_gradient_unbiased_2000rep.json", "cand": "candidate_credit_exact.json",
             "exf": "exp_f_integrity.json"}
    for k in a.only.split(","):
        t = time.time()
        res = jobs[k]()
        res = {"commit": _commit(), "command": f"python -m superlearner.cam_unified.selfeval.evidence_w8 --only {k}",
               "wall_s": time.time() - t, **res}
        with open(os.path.join(a.out_dir, names[k]), "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(k, json.dumps(res, default=str)[:1500])


if __name__ == "__main__":
    main()
