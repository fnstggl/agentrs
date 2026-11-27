"""Restricted candidate-policy credit estimators over RECORDED candidate sets (research note §7.2; manifest CR-06,
CR-07; plan §3 W8 "COCOA candidate estimator").

At decision t CAM records the candidate set C_t = (c_{t,1..K_t}) (independent proposals), the selection distribution
sigma_theta(. | b_t, C_t) over its slots and the chosen slot j_t. Treating s~_t = (b_t, C_t) as the state of a
finite-action decision process whose action is the slot j, the COCOA and R-HCA estimators apply unchanged:

    COCOA:  g = sum_t [ grad log sigma(j_t|s~_t) R_t + sum_j grad sigma(j|s~_t) sum_{k>t}  w(s~_t, j, U_k) G_{t,k} R_k ]
    R-HCA:  g = sum_t   sum_j grad sigma(j|s~_t) sum_{k>=t} rho(s~_t, j, U_k) G_{t,k} R_k

with w = h/sigma - 1 (outcome occurrences k >= 1) and rho = h/sigma (k >= 0, time-abstracted), R the VERIFIED
utility increment placed at the decision index where the outcome occurred, G_{t,k} = Gamma weights (1 = undiscounted).
Both are exact (unbiased) for the gradient w.r.t. the selector parameters CONDITIONAL on the recorded candidate sets
(C_t is drawn before sigma acts and is part of the state); they assign no credit to actions never proposed (a
coverage limit, reported by `coverage_report`). Extensions beyond the papers, declared: candidate-set states,
SMDP Gamma weights (exact only with Gamma-weighted hindsight, COCOA App. J; only Gamma = 1 is exact-tested here).

Surrogates (`cocoa_surrogate`, `rhca_surrogate`) return a per-trajectory scalar whose autograd gradient w.r.t. the
selector logits' parameters is the estimator; coefficients are stop-gradient. `pair_coefficients` builds the
[N, T, K, T] coefficient tensor from a learned `credit.heads.HindsightModel`.

`CandidateMDP` / `wrap` is the exact tabular wrapper used by the tests: a TabularMDP whose states are (s, C) with C a
K-tuple of base actions drawn i.i.d. from a proposal q(.|s), whose actions are slots, and whose selector is
sigma(j | s, C) = softmax_j theta[s, C_j] over base logits theta [S, A]: exact J(theta), its exact gradient, exact
COCOA coefficients and exact R-HCA ratios on the wrapped process are available from `credit.exact`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from . import exact as X


# ================================================================================================ surrogates

def _gamma(gamma: Optional[Tensor], N: int, T: int, ref: Tensor) -> Tensor:
    if gamma is None:
        return torch.ones(N, T, T, dtype=ref.dtype, device=ref.device)
    return gamma.to(ref.dtype)


def gamma_from_times(tau: Tensor, kappa: float) -> Tensor:
    """G[n, t, k] = exp(-kappa (tau_k - tau_t)) for k >= t (0 below the diagonal); tau [N, T] decision times."""
    d = tau[:, None, :] - tau[:, :, None]
    return torch.where(d >= 0, torch.exp(-kappa * d.clamp_min(0)), torch.zeros_like(d))


def cocoa_surrogate(sel_logits: Tensor, cand_mask: Tensor, chosen: Tensor, R: Tensor, w: Tensor,
                    gamma: Optional[Tensor] = None, step_mask: Optional[Tensor] = None) -> Tensor:
    """Per-trajectory COCOA surrogate [N]. sel_logits [N,T,K] (graph); cand_mask [N,T,K]; chosen [N,T]; R [N,T]
    verified utility increments; w [N,T,K,T]: w[n,t,j,k] = w(s~_t, j, U_k) (only k > t is used); gamma [N,T,T]."""
    N, T, K = sel_logits.shape
    lg = sel_logits.masked_fill(~cand_mask.bool(), float("-inf"))
    logp = torch.log_softmax(lg, -1)
    p = logp.exp()
    sm = step_mask.to(R.dtype) if step_mask is not None else torch.ones_like(R)
    Rm = (R * sm).detach()
    lp_taken = logp.gather(-1, chosen.long()[..., None]).squeeze(-1)
    future = (torch.arange(T, device=R.device)[None, :] > torch.arange(T, device=R.device)[:, None]).to(R.dtype)
    G = _gamma(gamma, N, T, R) * future[None]
    coef = torch.einsum("ntjk,ntk,nk->ntj", w.to(R.dtype), G, Rm).detach()
    imm = torch.where(sm > 0, lp_taken, torch.zeros_like(lp_taken)) * Rm
    cf = (torch.where(cand_mask.bool(), p, torch.zeros_like(p)) * coef).sum(-1) * sm
    return (imm + cf).sum(-1)


def rhca_surrogate(sel_logits: Tensor, cand_mask: Tensor, R: Tensor, rho: Tensor, gamma: Optional[Tensor] = None,
                   step_mask: Optional[Tensor] = None) -> Tensor:
    """Per-trajectory R-HCA surrogate [N]: sum_t sum_j sigma(j|s~_t) sg(sum_{k>=t} rho[t,j,k] G_{t,k} R_k)."""
    N, T, K = sel_logits.shape
    lg = sel_logits.masked_fill(~cand_mask.bool(), float("-inf"))
    p = torch.softmax(lg, -1)
    sm = step_mask.to(R.dtype) if step_mask is not None else torch.ones_like(R)
    Rm = (R * sm).detach()
    incl = (torch.arange(T, device=R.device)[None, :] >= torch.arange(T, device=R.device)[:, None]).to(R.dtype)
    G = _gamma(gamma, N, T, R) * incl[None]
    coef = torch.einsum("ntjk,ntk,nk->ntj", rho.to(R.dtype), G, Rm).detach()
    return ((torch.where(cand_mask.bool(), p, torch.zeros_like(p)) * coef).sum(-1) * sm).sum(-1)


def pair_coefficients(hindsight, b_t: Tensor, u: Tensor, cand: Tensor, cand_mask: Tensor, logp_sel: Tensor,
                      e_g: Optional[Tensor] = None, t_days: Optional[Tensor] = None, method: str = "cocoa",
                      clip: float = 10.0) -> Tensor:
    """Learned coefficients [N,T,K,T] from a HindsightModel h(j | b_t, C_t, u_k, g, elapsed, log sigma):
    COCOA w = h/sigma - 1 (k > t), R-HCA rho = h/sigma (k >= t); clipped at `clip` (declared bias). Inputs:
    b_t [N,T,D] behaviour-time state features (detached), u [N,T,du] outcome encodings at each step, cand [N,T,K,D],
    cand_mask [N,T,K], logp_sel [N,T,K] behaviour selection log-probs, t_days [N,T] decision times."""
    N, T, K = cand_mask.shape
    out = torch.zeros(N, T, K, T, device=b_t.device)
    lo = 1 if method == "cocoa" else 0
    with torch.no_grad():
        for t in range(T):
            for k in range(t + lo, T):
                el = torch.stack([torch.log1p((t_days[:, k] - t_days[:, t]).clamp_min(0)) if t_days is not None
                                  else torch.zeros(N, device=b_t.device),
                                  torch.full((N,), float(k - t), device=b_t.device).log1p()], -1)
                lg = hindsight.logits(b_t[:, t].detach(), u[:, k].detach(), e_g, cand[:, t].detach(), cand_mask[:, t],
                                      logp_sel[:, t].detach(), el)
                h = torch.softmax(lg.float(), -1)
                ratio = (h / logp_sel[:, t].float().exp().clamp_min(1e-8)).clamp(0.0, clip)
                ratio = torch.where(cand_mask[:, t].bool(), ratio, torch.zeros_like(ratio))
                out[:, t, :, k] = ratio - 1.0 if method == "cocoa" else ratio
    if method == "cocoa":
        out = torch.where(cand_mask[..., None].bool(), out.clamp_min(-1.0), torch.zeros_like(out))
    return out


def coverage_report(sel_probs: Tensor, chosen: Tensor, cand_mask: Tensor, h_probs: Optional[Tensor] = None,
                    proposal_logp: Optional[Tensor] = None, better_in_set: Optional[Tensor] = None,
                    step_mask: Optional[Tensor] = None) -> dict:
    """Candidate coverage (research note §7.2): K_t histogram, selection entropy, hindsight mass on the executed
    candidate (h_probs [N,T,K] at the realised outcome), proposal entropy proxy (mean -log q of the candidates) and
    the fraction of decisions whose candidate set held a later-verified better action (`better_in_set` [N,T] bool)."""
    sm = step_mask.bool() if step_mask is not None else torch.ones(chosen.shape, dtype=torch.bool,
                                                                    device=chosen.device)
    Kt = cand_mask.long().sum(-1)[sm]
    p = sel_probs.float()
    ent = -(p * p.clamp_min(1e-12).log() * cand_mask.float()).sum(-1)[sm]
    rep = {"n": int(sm.sum()), "K_hist": {int(k): int((Kt == k).sum()) for k in torch.unique(Kt).tolist()},
           "selection_entropy": float(ent.mean()) if ent.numel() else 0.0}
    if h_probs is not None:
        hm = h_probs.float().gather(-1, chosen.long()[..., None]).squeeze(-1)[sm]
        rep["hindsight_mass_on_executed"] = float(hm.mean()) if hm.numel() else 0.0
    if proposal_logp is not None:
        lq = (-proposal_logp.float() * cand_mask.float()).sum(-1) / cand_mask.float().sum(-1).clamp_min(1)
        rep["proposal_neg_logq"] = float(lq[sm].mean())
    if better_in_set is not None:
        rep["better_action_in_set_frac"] = float(better_in_set.float()[sm].mean())
    return rep


# ================================================================================================ tabular wrapper

@dataclass
class CandidateMDP:
    mdp: X.TabularMDP          # the wrapped process: states (s, C), actions = slots
    base: X.TabularMDP
    K: int
    tuples: list               # index -> K-tuple of base actions
    proposal: Tensor           # [S, A] q(a | s)

    @property
    def n_tuples(self) -> int:
        return len(self.tuples)

    def decode(self, s_aug: Tensor) -> tuple:
        """augmented state index -> (base state, tuple index)."""
        return s_aug // self.n_tuples, s_aug % self.n_tuples

    def cand_table(self) -> Tensor:
        return torch.tensor(self.tuples, dtype=torch.long)                 # [n_tuples, K]

    def aug_theta(self, theta: Tensor) -> Tensor:
        """theta [S, A] base logits -> selector logits [S*n_tuples, K]: theta[s, C_j]."""
        S = self.base.S
        C = self.cand_table()
        return theta[:, C].reshape(S * self.n_tuples, self.K)

    def expected_return(self, theta: Tensor) -> Tensor:
        return X.expected_return(self.mdp, self.aug_theta(theta))

    def exact_gradient(self, theta: Tensor) -> Tensor:
        th = theta.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(self.expected_return(th), th)
        return g

    def steps(self, batch: X.Batch, theta: Tensor) -> dict:
        """Per-step recorded quantities of wrapped-process trajectories: sel_logits [N,T,K] (graph to theta),
        candidates [N,T,K] base actions, mask, chosen slot."""
        s, ci = self.decode(batch.s)
        C = self.cand_table()[ci]                                          # [N, T, K]
        logits = theta[s[..., None].expand_as(C), C]
        return {"sel_logits": logits, "cand": C, "cand_mask": torch.ones_like(C, dtype=torch.bool),
                "chosen": batch.a, "base_s": s}


def wrap(base: X.TabularMDP, K: int, proposal: Optional[Tensor] = None, seed: int = 0) -> CandidateMDP:
    """Candidate-set wrapper of `base` (see module docstring). proposal [S, A] (default: random, all actions > 0)."""
    S, A, T = base.S, base.A, base.T
    if proposal is None:
        g = torch.Generator().manual_seed(seed)
        proposal = torch.rand(S, A, generator=g, dtype=X.DT) + 0.25
        proposal = proposal / proposal.sum(-1, keepdim=True)
    tuples = list(itertools.product(range(A), repeat=K))
    nt = len(tuples)
    Ct = torch.tensor(tuples, dtype=torch.long)
    Q = proposal[:, Ct].prod(-1)                                           # [S, nt] q(C | s)
    Sa = S * nt
    P = torch.zeros(T, Sa, K, Sa, dtype=X.DT)
    out = torch.zeros(T, Sa, K, Sa, dtype=torch.long)
    for t in range(T):
        for s in range(S):
            for c in range(nt):
                sa = s * nt + c
                for j in range(K):
                    a = tuples[c][j]
                    nxt = base.P[t, s, a]                                  # [S]
                    P[t, sa, j] = (nxt[:, None] * Q).reshape(Sa)
                    out[t, sa, j] = base.outcome[t, s, a][:, None].expand(S, nt).reshape(Sa)
    init = (base.init[:, None] * Q).reshape(Sa)
    m = X.TabularMDP(P, out, base.reward_mean.clone(), init, base.reward_noise, name=f"cand{K}({base.name})")
    m.validate()
    return CandidateMDP(m, base, K, tuples, proposal)


def exact_coefficients(cm: CandidateMDP, theta: Tensor) -> tuple:
    """Exact COCOA w [T, S~, K, U] and R-HCA rho [T, S~, K, U] of the wrapped process at selector theta [S, A]."""
    ta = cm.aug_theta(theta.detach())
    return X.contribution_coefficients(cm.mdp, ta), X.rhca_ratios(cm.mdp, ta)


def batch_coefficients(table: Tensor, batch: X.Batch) -> Tensor:
    """table [T, S~, K, U] -> per-trajectory pair coefficients [N, T, K, T]: c[n,t,j,k] = table[t, S_t, j, U_k]."""
    N, T = batch.s.shape
    tt = torch.arange(T)
    ct = table[tt[None, :], batch.s]                                       # [N, T, K, U]
    return torch.gather(ct, 3, batch.u[:, None, None, :].expand(N, T, ct.shape[2], T))
