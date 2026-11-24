"""Fast within-episode adaptation: a meta-learned Bayesian linear regression of reward on learned per-lever features.

Inside one episode the regression is fitted in closed form from the agent's own (setting, reward) pairs, so it adapts
after a handful of ticks without any gradient step. Across episodes the feature map, prior precisions, noise level and
forgetting rate are trained by the posterior predictive likelihood of each reward given only earlier pairs from the same
episode (the ALPaCA recipe, Harrison et al. 2018). The feature map is shared by all lever slots, so nothing is tied to a
slot position.

Units. The prior variance and the noise variance are both proportional to the episode's reward variance ``s^2``, so the
posterior mean does not depend on ``s`` and the sufficient statistics can be kept in raw reward units. Only predictive
variances are multiplied by ``s^2``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EpisodeRegression(nn.Module):
    def __init__(self, n_levers: int = 12, k: int = 4, hidden: int = 32):
        super().__init__()
        self.n_levers, self.k = n_levers, k
        self.dim = n_levers * k + 1
        self.psi = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, k))
        self.log_prec = nn.Parameter(torch.tensor(0.0))          # weights of the lever features
        self.log_prec_bias = nn.Parameter(torch.tensor(-7.0))    # the reward level: almost unconstrained
        self.log_noise = nn.Parameter(torch.tensor(math.log(0.5)))
        self.decay_logit = nn.Parameter(torch.tensor(4.0))       # forgetting per pair, sigmoid(4) = 0.982

    # ------------------------------------------------------------------ features and statistics
    def phi(self, x: Tensor, x0: Tensor, mask: Tensor) -> Tensor:
        """``x``, ``x0``, ``mask``: ``[..., L]`` -> features ``[..., D]`` (last entry is the bias)."""
        inp = torch.stack([x, x - x0, (x - x0) ** 2], -1)
        f = self.psi(inp) * mask.unsqueeze(-1)
        return torch.cat([f.flatten(-2), torch.ones_like(x[..., :1])], -1)

    @property
    def decay(self) -> Tensor:
        return torch.sigmoid(self.decay_logit)

    def empty(self, n: int, device) -> tuple[Tensor, Tensor]:
        return torch.zeros(n, self.dim, self.dim, device=device), torch.zeros(n, self.dim, device=device)

    def add(self, A: Tensor, b: Tensor, phi: Tensor, y: Tensor, live: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Fold one pair per row into the decayed sufficient statistics. ``live`` rows only."""
        d = self.decay
        A2 = d * A + phi.unsqueeze(-1) * phi.unsqueeze(-2)
        b2 = d * b + phi * y.unsqueeze(-1)
        if live is None:
            return A2, b2
        m = live.reshape(-1, 1, 1).to(A.dtype)
        return m * A2 + (1 - m) * A, m.squeeze(-1) * b2 + (1 - m.squeeze(-1)) * b

    def _precision(self, A: Tensor) -> Tensor:
        prior = torch.full((self.dim,), 1.0, device=A.device, dtype=A.dtype) * self.log_prec.exp()
        prior = torch.cat([prior[:-1], self.log_prec_bias.exp().reshape(1)])
        return torch.diag_embed(prior.expand(*A.shape[:-2], -1)) + A / self.log_noise.exp()

    def posterior(self, A: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        """Mean ``[..., D]`` and Cholesky factor of the precision ``[..., D, D]``."""
        L = torch.linalg.cholesky(self._precision(A))
        mean = torch.cholesky_solve((b / self.log_noise.exp()).unsqueeze(-1), L).squeeze(-1)
        return mean, L

    def predict(self, A: Tensor, b: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
        """Posterior predictive mean and variance (in units of ``s^2``) for features ``phi`` ``[B, ..., D]``."""
        mean, L = self.posterior(A, b)
        shape = phi.shape
        flat = phi.reshape(shape[0], -1, shape[-1])
        mu = (flat * mean.unsqueeze(1)).sum(-1)
        sol = torch.cholesky_solve(flat.transpose(1, 2), L)            # P^-1 phi^T
        var = (flat * sol.transpose(1, 2)).sum(-1) + self.log_noise.exp()
        return mu.reshape(shape[:-1]), var.reshape(shape[:-1])

    def sample_weights(self, A: Tensor, b: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """One posterior draw of the weights per row (Thompson sampling)."""
        mean, L = self.posterior(A, b)
        eps = torch.randn(mean.shape, device=mean.device, generator=generator)
        # P = L L^T, so w = mean + L^{-T} eps has covariance P^{-1}
        return mean + torch.linalg.solve_triangular(L.transpose(-1, -2), eps.unsqueeze(-1), upper=True).squeeze(-1)

    # ------------------------------------------------------------------ meta-training
    def prefix_nll(self, x: Tensor, r: Tensor, sd: Tensor, x0: Tensor, mask: Tensor, valid: Tensor,
                   n_points: int = 16, min_pairs: int = 3, return_predictions: bool = False) -> tuple[Tensor, dict]:
        """Posterior predictive negative log-likelihood of reward ``t`` given pairs ``< t`` of the same episode.

        ``x [B, H, L]`` settings in force, ``r [B, H]`` reward, ``sd [B, H]`` the episode's reward spread known at
        that tick, ``x0 [B, L]`` starting settings, ``mask [B, L]`` available levers, ``valid [B, H]`` real ticks.
        Also returns the same likelihood for a predictor that knows only the running mean (the natural baseline).
        """
        B, H, _ = x.shape
        phi = self.phi(x, x0.unsqueeze(1).expand_as(x), mask.unsqueeze(1).expand_as(x))
        A, b = self.empty(B, x.device)
        picks = set(torch.linspace(min_pairs, H - 1, min(n_points, max(1, H - min_pairs))).long().tolist())
        nll, base, se, se_base, cnt, mus, ts = [], [], [], [], [], [], []
        run_sum = torch.zeros(B, device=x.device)
        run_n = torch.zeros(B, device=x.device)
        for t in range(H):
            if t in picks:
                mu, var = self.predict(A, b, phi[:, t].unsqueeze(1))
                mu, var = mu[:, 0], var[:, 0]
                mus.append(mu)
                ts.append(t)
                s2 = sd[:, t].clamp(min=1e-4) ** 2
                ok = valid[:, t] * (run_n >= min_pairs).float()
                v = var * s2
                nll.append(0.5 * (torch.log(2 * math.pi * v) + (r[:, t] - mu) ** 2 / v) * ok)
                m0 = run_sum / run_n.clamp(min=1)
                base.append(0.5 * (torch.log(2 * math.pi * s2) + (r[:, t] - m0) ** 2 / s2) * ok)
                se.append(((r[:, t] - mu) ** 2 / s2) * ok)
                se_base.append(((r[:, t] - m0) ** 2 / s2) * ok)
                cnt.append(ok)
            A, b = self.add(A, b, phi[:, t], r[:, t], valid[:, t] > 0)
            run_sum = run_sum + r[:, t] * valid[:, t]
            run_n = run_n + valid[:, t]
        n = torch.stack(cnt).sum().clamp(min=1.0)
        loss = torch.stack(nll).sum() / n
        info = {"adapt/nll": loss.detach(), "adapt/nll_mean_only": torch.stack(base).sum().detach() / n,
                "adapt/r2_vs_mean": 1 - torch.stack(se).sum().detach() / torch.stack(se_base).sum().clamp(min=1e-8),
                "adapt/decay": self.decay.detach(), "adapt/noise": self.log_noise.exp().detach()}
        if return_predictions:
            info["ticks"], info["mu"] = ts, torch.stack(mus, 1)
        return loss, info
