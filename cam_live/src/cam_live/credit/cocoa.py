"""cam_live.credit.cocoa — counterfactual contribution analysis over probability-tracked candidate sets (spec 9.3;
RC-04, RC-05).

The exact finite-action reference is ``credit.estimators.cocoa_grad`` with ``tabular.contribution_coefficients``.
This module is the CAM extension: a learned hindsight model

    h(a | b, u, elapsed)  ~  p_pi(A_t = a | belief b_t, later outcome u, time between them)

over the *recorded* candidate set of a decision, and contribution coefficients  w = h / pi - 1  from the logged
behaviour probabilities.

* **Policy-logit prior.** logits = log pi(a | b) + f(b, u, elapsed, a) with the last layer of f zero-initialised:
  before any training h = pi and w = 0 (no credit is claimed), and the model only has to learn how an outcome
  shifts the action distribution (COCOA Prop. 5: conditioning on the policy logits is free and tracks a moving
  policy).
* **Elapsed time** between decision and outcome is an input (time-indexed hindsight), so the same kind of outcome
  can mean different things at different delays.
* **Uncertainty** comes from a bootstrapped ensemble and, when the caller supplies it, from the support
  (effective sample size per candidate).
* **Refusal.** :func:`check_support` returns ``UNSUPPORTED`` — and the estimate is NaN, not a number — when
  propensities are missing or invalid, when a candidate had near-zero logged probability, or when the action space
  was open-ended without probability-tracked proposals.
* **Three separate fields.** ``learned_association`` (what the outcome says about which action was taken),
  ``counterfactual_estimate`` (w times the outcome value: would the outcome have been more or less likely under
  that action) and ``experimental_effect`` (filled only from randomised arms, never by the model).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .simple import ArmDifference

NAME = "cocoa"
SUPPORTED, UNSUPPORTED = "supported", "unsupported"
R_MISSING = "propensities_missing"
R_INVALID = "propensities_invalid"
R_NEAR_ZERO = "near_zero_propensity"
R_OPEN = "open_ended_without_tracked_proposals"
R_EMPTY = "no_candidates"
R_ESS = "insufficient_support"


@dataclass
class SupportReport:
    status: List[str]                     # per decision: SUPPORTED | UNSUPPORTED
    reasons: List[List[str]]
    candidate_ok: Tensor                  # [B, N] bool: candidates whose logged probability supports a ratio

    @property
    def supported(self) -> Tensor:
        return torch.tensor([s == SUPPORTED for s in self.status], dtype=torch.bool)


def _per_row(v: Union[bool, Sequence[bool], Tensor], B: int) -> List[bool]:
    if isinstance(v, Tensor):
        return [bool(x) for x in v.reshape(-1).tolist()] if v.numel() > 1 else [bool(v)] * B
    if isinstance(v, (list, tuple)):
        return [bool(x) for x in v]
    return [bool(v)] * B


def check_support(logp_policy: Optional[Tensor], cand_mask: Tensor, *, min_prob: float = 1e-3,
                  open_ended: Union[bool, Sequence[bool], Tensor] = False,
                  tracked_proposals: Union[bool, Sequence[bool], Tensor] = True, norm_tol: float = 1e-3,
                  n_eff: Optional[Tensor] = None, min_ess: float = 0.0) -> SupportReport:
    """Can a contribution coefficient h / pi - 1 be stated for each decision?

    logp_policy  [B, N] logged behaviour log-probabilities of the candidates (``None`` = not recorded)
    cand_mask    [B, N] True for real candidates
    open_ended / tracked_proposals   the action space was open-ended (free-form proposals); the estimate is only
                 defined if every proposal that could have been executed was recorded with its probability
    n_eff / min_ess   optional effective sample size per candidate and the minimum required
    """
    m = cand_mask.bool().cpu()                       # bookkeeping on the CPU, whatever device the latents live on
    if logp_policy is not None:
        logp_policy = logp_policy.detach().cpu()
    if n_eff is not None:
        n_eff = n_eff.detach().cpu()
    B, N = m.shape
    oe, tr = _per_row(open_ended, B), _per_row(tracked_proposals, B)
    status, reasons = [], []
    cand_ok = torch.zeros(B, N, dtype=torch.bool)
    for i in range(B):
        why: List[str] = []
        if not bool(m[i].any()):
            why.append(R_EMPTY)
        if oe[i] and not tr[i]:
            why.append(R_OPEN)
        if logp_policy is None:
            why.append(R_MISSING)
        elif m[i].any():
            lp = logp_policy[i][m[i]].double()
            p = lp.exp()
            if bool(torch.isnan(lp).any()) or bool((lp > 1e-6).any()):
                why.append(R_INVALID)           # NaN, or a probability above 1
            elif abs(float(p.sum()) - 1.0) > norm_tol:
                why.append(R_INVALID)           # not a distribution over the recorded candidates
            else:
                ok = p >= min_prob
                if n_eff is not None and min_ess > 0:
                    enough = n_eff[i][m[i]].double() >= min_ess
                    if not bool(enough.all()):
                        why.append(R_ESS)
                    ok = ok & enough
                cand_ok[i, m[i]] = ok
                if not bool((p >= min_prob).all()):
                    why.append(R_NEAR_ZERO)
        status.append(UNSUPPORTED if why else SUPPORTED)
        reasons.append(why)
    return SupportReport(status, reasons, cand_ok)


def effective_sample_size(weights: Tensor) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2 of non-negative weights."""
    w = weights.double()
    s2 = float((w ** 2).sum())
    return float(w.sum()) ** 2 / s2 if s2 > 0 else 0.0


def candidate_ess(keys: Sequence[Sequence[str]], chosen: Sequence[int], probs: Sequence[Sequence[float]]
                  ) -> Dict[str, float]:
    """Support per candidate key over logged decisions: the Kish effective sample size of the inverse-propensity
    weights 1 / pi_i(a) of the decisions in which candidate ``a`` was the executed one. ``keys[i][j]`` identifies
    candidate j of decision i (e.g. the rendered action)."""
    w: Dict[str, List[float]] = {}
    for ks, c, ps in zip(keys, chosen, probs):
        for k in ks:
            w.setdefault(k, [])
        if 0 <= c < len(ks) and c < len(ps) and ps[c] > 0:
            w[ks[c]].append(1.0 / ps[c])
    return {k: effective_sample_size(torch.tensor(v, dtype=torch.float64)) if v else 0.0 for k, v in w.items()}


@dataclass
class CounterfactualCredit:
    """Output of :meth:`HindsightModel.contribution`. The three kinds of statement are separate fields."""
    status: List[str]
    reasons: List[List[str]]
    #: learned association [B, N]: h(a | b, u) - pi(a | b). Correlational: how observing the outcome changes the
    #: belief about which action was taken. NaN when propensities are missing / invalid.
    learned_association: Tensor
    #: counterfactual estimate [B, N]: w(a, u) (times ``outcome_value`` when given). NaN wherever UNSUPPORTED.
    counterfactual_estimate: Tensor
    #: experimentally supported effect per decision row: only ever set by :meth:`attach_experiment`
    experimental_effect: Dict[int, ArmDifference] = field(default_factory=dict)
    w: Optional[Tensor] = None                 # [B, N] clipped coefficients (NaN where unsupported)
    w_std: Optional[Tensor] = None             # [B, N] ensemble (+ support) standard deviation of w
    h: Optional[Tensor] = None                 # [B, N] ensemble-mean hindsight probabilities
    clipped: Optional[Tensor] = None           # [B, N] bool: the clip changed the value (a declared bias)
    candidate_supported: Optional[Tensor] = None

    @property
    def supported(self) -> Tensor:
        return torch.tensor([s == SUPPORTED for s in self.status], dtype=torch.bool)

    def attach_experiment(self, row: int, effect: ArmDifference) -> None:
        """Record a randomised-experiment result for a decision. Non-randomised comparisons are refused: they are
        associations, not experimentally supported effects."""
        if not effect.randomized:
            raise ValueError("experimental_effect accepts randomised arm differences only")
        self.experimental_effect[int(row)] = effect


class _Member(nn.Module):
    def __init__(self, d_ctx: int, d_a: int, d_hidden: int, d_key: int):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d_ctx, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_hidden), nn.SiLU())
        self.q_out = nn.Linear(d_hidden, d_key)
        self.k = nn.Sequential(nn.Linear(d_a, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_key))
        nn.init.zeros_(self.q_out.weight)                 # f = 0 at initialisation  =>  h = pi, w = 0
        nn.init.zeros_(self.q_out.bias)
        self.d_key = d_key

    def forward(self, ctx: Tensor, cand: Tensor) -> Tensor:
        q = self.q_out(self.q(ctx))
        return torch.einsum("bd,bnd->bn", q, self.k(cand)) / math.sqrt(self.d_key)


class HindsightModel(nn.Module):
    """h(a | b, u, elapsed) over the recorded candidates of a decision, as a bootstrapped ensemble.

    d_b, d_u, d_a   widths of the belief latent, the outcome representation and the candidate-action embeddings
    n_members       ensemble size (uncertainty = spread across members)
    n_elapsed       number of elapsed-time inputs (default 2: seconds and decisions between decision and outcome);
                    they enter as log1p(x / unit)
    """

    def __init__(self, d_b: int, d_u: int, d_a: int, d_hidden: int = 128, d_key: int = 64, n_members: int = 4,
                 n_elapsed: int = 2, elapsed_units: Sequence[float] = (60.0, 1.0), seed: int = 0):
        super().__init__()
        self.n_elapsed = int(n_elapsed)
        self.register_buffer("elapsed_units", torch.tensor(list(elapsed_units)[:n_elapsed] or [1.0], dtype=torch.float32))
        g = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.members = nn.ModuleList([_Member(d_b + d_u + self.n_elapsed, d_a, d_hidden, d_key) for _ in range(n_members)])
        torch.random.set_rng_state(g)

    # ------------------------------------------------------------------------------------------------ forward
    def _ctx(self, b: Tensor, u: Tensor, elapsed: Optional[Tensor]) -> Tensor:
        B = b.shape[0]
        if self.n_elapsed == 0:
            return torch.cat([b.float(), u.float()], -1)
        if elapsed is None:
            el = b.new_zeros(B, self.n_elapsed).float()
        else:
            el = elapsed.float().reshape(B, -1)
            if el.shape[1] != self.n_elapsed:
                raise ValueError(f"elapsed has {el.shape[1]} columns, the model expects {self.n_elapsed}")
            el = torch.log1p(el.clamp_min(0) / self.elapsed_units)
        return torch.cat([b.float(), u.float(), el], -1)

    def member_logits(self, b: Tensor, u: Tensor, cand: Tensor, cand_mask: Tensor, logp_policy: Tensor,
                      elapsed: Optional[Tensor] = None) -> Tensor:
        """[K, B, N] logits of every ensemble member: logged policy log-probabilities + learned residual.
        b [B, d_b] (decision-time belief, detached by the caller), u [B, d_u], cand [B, N, d_a], cand_mask [B, N],
        logp_policy [B, N], elapsed [B, n_elapsed] raw (seconds, decisions)."""
        ctx = self._ctx(b, u, elapsed)
        m = cand_mask.bool()
        prior = logp_policy.float().masked_fill(~m, 0.0).clamp_min(-30.0)
        out = torch.stack([prior + mem(ctx, cand.float()) for mem in self.members])
        return out.masked_fill(~m[None], float("-inf"))

    def forward(self, b: Tensor, u: Tensor, cand: Tensor, cand_mask: Tensor, logp_policy: Tensor,
                elapsed: Optional[Tensor] = None) -> Tensor:
        """Ensemble-mean hindsight probabilities h [B, N]."""
        return torch.softmax(self.member_logits(b, u, cand, cand_mask, logp_policy, elapsed), -1).mean(0)

    # ------------------------------------------------------------------------------------------------ training
    def loss(self, b: Tensor, u: Tensor, cand: Tensor, cand_mask: Tensor, taken: Tensor, logp_policy: Tensor,
             elapsed: Optional[Tensor] = None, generator: Optional[torch.Generator] = None,
             bootstrap: bool = True) -> Tuple[Tensor, Dict[str, float]]:
        """Cross-entropy of the executed candidate given the later outcome, for every member on its own bootstrap
        resample of the batch (Poisson(1) weights). Train on legitimate retrospective pairs only: a decision and an
        outcome that was observed after it."""
        lg = self.member_logits(b, u, cand, cand_mask, logp_policy, elapsed)          # [K, B, N]
        K, B, _ = lg.shape
        ce = F.cross_entropy(lg.reshape(K * B, -1), taken.long().repeat(K), reduction="none").view(K, B)
        if bootstrap and K > 1:
            wt = torch.poisson(torch.ones(K, B), generator=generator).to(ce.device)
        else:
            wt = torch.ones(K, B, device=ce.device)
        loss = ((ce * wt).sum(1) / wt.sum(1).clamp_min(1.0)).sum()
        with torch.no_grad():
            base = F.cross_entropy(logp_policy.float().masked_fill(~cand_mask.bool(), float("-inf")), taken.long())
        return loss, {"ce": float(ce.mean().detach()), "ce_policy_prior": float(base)}

    # ------------------------------------------------------------------------------------------------ credit
    @torch.no_grad()
    def contribution(self, b: Tensor, u: Tensor, cand: Tensor, cand_mask: Tensor, logp_policy: Optional[Tensor],
                     elapsed: Optional[Tensor] = None, *, outcome_value: Optional[Tensor] = None, clip: float = 10.0,
                     min_prob: float = 1e-3, open_ended: Union[bool, Sequence[bool], Tensor] = False,
                     tracked_proposals: Union[bool, Sequence[bool], Tensor] = True, n_eff: Optional[Tensor] = None,
                     min_ess: float = 0.0) -> CounterfactualCredit:
        """Contribution coefficients w = h / pi - 1 (clipped to [-1, clip]) for every recorded candidate.

        Where :func:`check_support` says UNSUPPORTED the coefficient and the counterfactual estimate are NaN.
        ``outcome_value`` [B] turns the coefficient into credit in outcome units. ``n_eff`` [B, N] adds the
        support-based standard error sqrt(h (1 - h) / n_eff) / pi to the ensemble spread."""
        m = cand_mask.bool()
        B, N = m.shape
        rep = check_support(logp_policy, m, min_prob=min_prob, open_ended=open_ended,
                            tracked_proposals=tracked_proposals, n_eff=n_eff, min_ess=min_ess)
        nan = torch.full((B, N), float("nan"), dtype=torch.float64)
        if logp_policy is None:
            return CounterfactualCredit(rep.status, rep.reasons, nan.clone(), nan.clone(), candidate_supported=rep.candidate_ok)
        valid_p = torch.tensor([not ({R_MISSING, R_INVALID} & set(r)) for r in rep.reasons])
        safe_lp = torch.where(valid_p[:, None].to(m.device) & m, logp_policy.float(), torch.zeros_like(logp_policy.float()))
        was = self.training
        self.eval()
        try:
            hk = torch.softmax(self.member_logits(b, u, cand, m, safe_lp, elapsed), -1).double().cpu()     # [K, B, N]
        finally:
            self.train(was)
        m, pi = m.cpu(), safe_lp.double().exp().cpu()
        h = hk.mean(0)
        wk = hk / pi.clamp_min(1e-12) - 1.0
        w_raw = wk.mean(0)
        w = w_raw.clamp(-1.0, clip)
        w_std = wk.std(0, unbiased=True) if hk.shape[0] > 1 else torch.zeros_like(w)
        if n_eff is not None:
            se = torch.sqrt((h * (1 - h)).clamp_min(0) / n_eff.double().cpu().clamp_min(1e-9)) / pi.clamp_min(1e-12)
            w_std = torch.sqrt(w_std ** 2 + se ** 2)
        sup = rep.supported[:, None] & m
        assoc = torch.where(valid_p[:, None] & m, h - pi, nan)
        w_out = torch.where(sup, w, nan)
        est = w_out if outcome_value is None else w_out * outcome_value.double().cpu()[:, None]
        return CounterfactualCredit(rep.status, rep.reasons, assoc, est, w=w_out, w_std=torch.where(sup, w_std, nan),
                                    h=torch.where(valid_p[:, None] & m, h, nan), clipped=(w != w_raw) & sup,
                                    candidate_supported=rep.candidate_ok)


def coefficients_table(model: HindsightModel, b: Tensor, u: Tensor, cand: Tensor, logp: Tensor,
                       elapsed: Optional[Tensor] = None, clip: float = 1e9) -> Tensor:
    """w for every (context row, outcome row, candidate): contexts b [S, d_b] with policies logp [S, A], outcomes
    u [U, d_u], shared candidates cand [A, d_a] -> [S, A, U]. For comparing a learned model with exact tabular
    coefficients and for plugging it into ``estimators.cocoa_grad``."""
    S, U, A = b.shape[0], u.shape[0], cand.shape[0]
    bb = b[:, None, :].expand(S, U, -1).reshape(S * U, -1)
    uu = u[None, :, :].expand(S, U, -1).reshape(S * U, -1)
    cc = cand[None].expand(S * U, A, -1)
    lp = logp[:, None, :].expand(S, U, A).reshape(S * U, A)
    el = None if elapsed is None else elapsed[None].expand(S * U, -1)
    out = model.contribution(bb, uu, cc, torch.ones(S * U, A, dtype=torch.bool), lp, el, clip=clip, min_prob=0.0)
    return out.w.view(S, U, A).permute(0, 2, 1)


def arm_contrast(w: Tensor, outcome_value: Tensor, treatment: int, control: int) -> Optional[Tuple[float, float, int]]:
    """COCOA's estimate of the effect of candidate ``treatment`` versus ``control`` from logged decisions: the mean
    over decisions (whatever action was actually taken) of [w(treatment, u) - w(control, u)] * outcome value, with
    its standard error and the number of supported decisions. ``w`` [B, N] may contain NaN rows (unsupported):
    they are left out, and ``None`` is returned when fewer than two decisions are supported."""
    d = (w[:, treatment] - w[:, control]).double() * outcome_value.double().cpu()
    d = d[torch.isfinite(d)]
    if d.numel() < 2:
        return None
    return float(d.mean()), float(d.std(unbiased=True) / math.sqrt(d.numel())), int(d.numel())
