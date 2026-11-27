"""Evidence script (Workstream B): does a bootstrapped ensemble need randomised prior functions to report higher
epistemic uncertainty off-distribution?  Run:

    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/B/ablation_prior.py \
        > cam_live/docs/evidence/B/ablation_prior.txt

Prints, per (prior_scale, n_train, seed): mean epistemic std in-distribution, off-distribution (inputs shifted by
+4 standard deviations) and RMSE of the ensemble mean on in-distribution test points. CPU, ~1 minute."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
import torch  # noqa: E402

from cam_live.predict.world import (K_REAL, WorldConfig, WorldHeads, WorldTargets, bootstrap_weights,  # noqa: E402
                                    world_nll)

torch.set_num_threads(4)
DM = 12


def run(n, seed, prior, members=5, steps=400):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(DM, generator=g) / math.sqrt(DM)
    e_q = torch.randn(1, DM, generator=g)
    z = torch.randn(n, 1, DM, generator=g)
    y = torch.sin(z[:, 0] @ w) * 2 + 0.3 * torch.randn(n, generator=g)
    torch.manual_seed(seed)
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=64, n_members=members, n_hazard_bins=4, max_categories=4,
                                   prior_scale=prior, seed=seed))
    opt = torch.optim.Adam(heads.parameters(), lr=3e-3)
    boot = bootstrap_weights([f"r{i}" for i in range(n)], members, seed)
    hour = lambda k: torch.full((k, 1), 3600.0, dtype=torch.float64)  # noqa: E731
    kind = lambda k: torch.full((k, 1), K_REAL)  # noqa: E731
    for _ in range(steps):
        idx = torch.randint(0, n, (min(n, 128),), generator=g)
        k = len(idx)
        out = heads(z[idx], e_q.expand(k, 1, DM), kind(k), hour(k))
        tgt = WorldTargets(y[idx, None], torch.ones(k, 1, dtype=torch.bool), torch.zeros(k, 1, dtype=torch.bool))
        loss, _ = world_nll(out, tgt, member_weights=boot[:, idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z_in = torch.randn(512, 1, DM, generator=g)
        z_off = torch.randn(512, 1, DM, generator=g) + 4.0
        p_in = heads(z_in, e_q.expand(512, 1, DM), kind(512), hour(512)).predictive()
        p_off = heads(z_off, e_q.expand(512, 1, DM), kind(512), hour(512)).predictive()
        rmse = float(((p_in["mean"][:, 0, 0] - torch.sin(z_in[:, 0] @ w) * 2) ** 2).mean().sqrt())
    return float(p_in["epistemic_std"].mean()), float(p_off["epistemic_std"].mean()), rmse


if __name__ == "__main__":
    print("prior_scale n_train seed epistemic_in epistemic_off rmse_in")
    for prior in (0.0, 1.0):
        for n in (16, 64, 256, 2048):
            for seed in range(3):
                e_in, e_off, rmse = run(n, seed, prior)
                print(f"{prior:.1f} {n} {seed} {e_in:.4f} {e_off:.4f} {rmse:.4f}")
