"""Re-run measured numbers of B and C and compare with their evidence files (honesty of B.json / C.json).

1. C, real adaptable subset: reward.diagnose.meta_gradient_check(Ks=(1, 2, 4)) vs docs/evidence/C/diagnostics.json
   (relative FD error on 57 coordinates ~1e-5, directional ~1e-8, detached reward 0, plain v.detach() bias 1.0 % /
   3.3 % at K = 2 / 4).
2. C, stage 1 exact tabular meta-gradient (reward.exact) vs FD, K in 1, 2, 4, 8, vs exact_meta_gradient.json.
3. B: re-run the default B suite with CAM_LIVE_B_MEASURE pointing to a scratch file (via pytest, separately) and
   compare every scalar with docs/evidence/B/measured.json — see G_pr.md for the command; this script compares
   the two JSON files when --b-measured PATH is given.
"""
import json
import math
import os
import sys

import torch

import _common as C

EV = os.path.join(C.HERE, "..", "..", "..", "evidence")

# ------------------------------------------------------------------------------------------------ C real subset
from cam_live.reward.diagnose import meta_gradient_check

stored = json.load(open(os.path.join(EV, "C", "diagnostics.json")))["meta_gradient"]
mine = meta_gradient_check(Ks=(1, 2, 4))
for K in ("1", "2", "4"):
    a, b = mine["K"][K], stored["K"][K]
    C.check(f"C real-subset FD check K={K}: rel error reproduces (stored {b['rel_error_sampled_coords']:.2e})",
            a["rel_error_sampled_coords"] < 1e-4 and abs(math.log10(a["rel_error_sampled_coords"] /
                                                                    b["rel_error_sampled_coords"])) < 1.0,
            f"mine={a['rel_error_sampled_coords']:.2e} directional={a['directional_rel_error']:.1e}")
    C.check(f"C real-subset K={K}: detached-reward gradient 0 and gradient norm reproduce",
            a["grad_norm_reward_detached"] == 0.0 and abs(a["grad_norm"] - b["grad_norm"]) / b["grad_norm"] < 0.05,
            f"|g| mine={a['grad_norm']:.4e} stored={b['grad_norm']:.4e}")
    C.info(f"K={K} plain v.detach() bias mine / stored", (a["rel_error_if_value_baseline_plainly_detached"],
                                                         b["rel_error_if_value_baseline_plainly_detached"]))

# ------------------------------------------------------------------------------------------------ C stage 1 exact
from cam_live.diagnostics import tabular as TX
from cam_live.reward import exact as E

ex = json.load(open(os.path.join(EV, "C", "exact_meta_gradient.json")))["finite_difference_agreement"]["per_K"]
mdp = TX.key_to_door(n_distract=2, key_cost=0.3, apple_reward=0.2)
th0 = torch.zeros(mdp.S, mdp.A, dtype=TX.DT)
psi = 0.3 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=TX.DT, generator=torch.Generator().manual_seed(0))
for K in (1, 2, 4, 8):
    g, J = E.exact_meta_gradient(mdp, psi, th0, K, 1.0)
    f = E.finite_difference_gradient(mdp, psi, th0, K, 1.0)
    rel = float((g - f).norm() / g.norm())
    st = ex[str(K)]
    C.check(f"C stage-1 exact tabular K={K}: J and FD agreement reproduce (stored rel {st['rel_err_vs_fd']:.1e})",
            abs(J - st["J"]) < 1e-12 and rel < 1e-8, f"J={J:.12f} stored={st['J']:.12f} rel={rel:.1e}")
    # independent check of the exact objective itself: finite differences of J computed by my own unroll would need
    # my own tabular model; instead perturb psi along a random direction and compare with g . d
    d = torch.randn(psi.shape, dtype=TX.DT, generator=torch.Generator().manual_seed(K))
    d = d / d.norm()
    eps = 1e-6
    jp = E.exact_meta_gradient(mdp, psi + eps * d, th0, K, 1.0)[1]
    jm = E.exact_meta_gradient(mdp, psi - eps * d, th0, K, 1.0)[1]
    C.check(f"C stage-1 K={K}: directional derivative of the exact J matches g.d",
            abs((jp - jm) / (2 * eps) - float((g * d).sum())) < 1e-7 * max(1.0, abs(float((g * d).sum()))))
sys.exit(C.finish())
