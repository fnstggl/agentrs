"""cam_live.predict.policy — candidate-set policy  pi(a_i | b, g, C_t)  and exploration rules (spec Part 5).

``CandidatePolicy`` scores a *variable* set of candidate actions from their decision latents ``z`` [B, N, D]
(belief + action + goal, produced by the shared trunk) **and** directly from the belief ``b`` [B, D] through a
multiplicative interaction, so the action distribution cannot be a function of the candidates alone
(:func:`latent_dependence` measures it; an earlier system had KL 4e-9 here — the language model decided
everything). It also returns a state value V [B] and a per-candidate value Q [B, N].

:func:`select` implements the exploration rules — policy sampling, Thompson sampling, UCB, greedy, information-gain
weighting and explicit randomised assignment over experiment arms. Every rule first builds the full probability
vector and then samples from exactly that vector, so the recorded propensities are exact, and an epsilon floor
keeps every feasible candidate in the support (needed by importance weighting and hindsight credit).

``functional_forward`` evaluates the policy with an explicit parameter dictionary (``torch.func.functional_call``)
so the reward workstream can differentiate through policy updates.

Nothing here consumes or produces a reward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

RULE_POLICY = "policy"
RULE_THOMPSON = "thompson"
RULE_UCB = "ucb"
RULE_GREEDY = "greedy"
RULE_INFO_GAIN = "voi"
RULE_RANDOMIZED = "randomized"
RULES = (RULE_POLICY, RULE_THOMPSON, RULE_UCB, RULE_GREEDY, RULE_INFO_GAIN, RULE_RANDOMIZED)


@dataclass
class PolicyConfig:
    d_model: int
    hidden: int = 192
    out_std: float = 0.05            # small random output init (never zero)


def masked_log_softmax(logits: Tensor, a_mask: Tensor) -> Tensor:
    if not bool(a_mask.any(-1).all()):
        raise ValueError("every decision needs at least one feasible candidate")
    neg = torch.finfo(logits.dtype).min
    return torch.log_softmax(torch.where(a_mask, logits, torch.full_like(logits, neg)), dim=-1)


class PolicyOutput:
    def __init__(self, logits: Tensor, a_mask: Tensor, value: Tensor, q: Tensor):
        self.a_mask = a_mask
        self.raw_logits = logits                                   # [B, N] unmasked scores
        self.log_probs = masked_log_softmax(logits, a_mask)        # [B, N]; ~ -inf at infeasible candidates
        self.value = value                                         # [B]
        self.q = torch.where(a_mask, q, torch.zeros_like(q))       # [B, N]

    @property
    def logits(self) -> Tensor:
        return torch.where(self.a_mask, self.raw_logits, torch.full_like(self.raw_logits, float("-inf")))

    @property
    def probs(self) -> Tensor:
        return torch.where(self.a_mask, torch.exp(self.log_probs), torch.zeros_like(self.log_probs))

    def log_prob(self, chosen: Tensor) -> Tensor:
        return self.log_probs.gather(-1, chosen.view(-1, 1)).squeeze(-1)

    def entropy(self) -> Tensor:
        p = self.probs
        return -(p * torch.where(self.a_mask, self.log_probs, torch.zeros_like(p))).sum(-1)


class CandidatePolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.in_z = nn.Linear(cfg.d_model, h)
        self.in_b = nn.Linear(cfg.d_model, h)
        self.in_zb = nn.Linear(cfg.d_model, h, bias=False)         # z (x) b interaction: the direct latent path
        self.in_bb = nn.Linear(cfg.d_model, h, bias=False)
        self.pool = nn.Linear(h, h)                                # summary of the alternatives on offer
        self.norm = nn.LayerNorm(h)
        self.mid = nn.Linear(h, h)
        self.logit_head = nn.Linear(h, 1)
        self.q_head = nn.Linear(h, 1)
        self.v_mid = nn.Linear(h, h)
        self.v_head = nn.Linear(h, 1)
        for head in (self.logit_head, self.q_head, self.v_head):
            nn.init.normal_(head.weight, std=cfg.out_std)
            nn.init.zeros_(head.bias)

    def forward(self, z: Tensor, a_mask: Tensor, b: Tensor) -> PolicyOutput:
        """z [B, N, D] decision latents, a_mask [B, N] (True = feasible candidate), b [B, D] belief."""
        hz = self.in_z(z)
        m = a_mask.unsqueeze(-1).to(hz.dtype)
        pooled = self.pool((hz * m).sum(1) / m.sum(1).clamp_min(1.0))                # [B, H], padding-invariant
        hb = self.in_b(b)
        inter = self.in_zb(z) * self.in_bb(b).unsqueeze(1) / math.sqrt(self.cfg.hidden)
        x = F.silu(self.mid(F.silu(self.norm(hz + hb.unsqueeze(1) + pooled.unsqueeze(1) + inter))))
        logits = self.logit_head(x).squeeze(-1)
        q = self.q_head(x).squeeze(-1)
        value = self.v_head(F.silu(self.v_mid(hb + pooled))).squeeze(-1)
        return PolicyOutput(logits, a_mask, value, q)

    def functional_forward(self, params: Dict[str, Tensor], z: Tensor, a_mask: Tensor, b: Tensor) -> PolicyOutput:
        """Same computation with explicit parameters (``dict(policy.named_parameters())``-shaped). Used to
        evaluate ``pi_{theta'}`` after a differentiable inner update without touching the module's own weights."""
        return torch.func.functional_call(self, params, (z, a_mask, b))

    def named_param_dict(self) -> Dict[str, Tensor]:
        return dict(self.named_parameters())

    ACTOR_PREFIXES = ("in_z.", "in_b.", "in_zb.", "in_bb.", "pool.", "norm.", "mid.", "logit_head.")

    def actor_param_dict(self) -> Dict[str, Tensor]:
        """The parameters the action distribution depends on (everything except the V and Q read-outs). A
        differentiable inner update of ``log pi`` touches exactly these; pass ``{**all, **updated_actor}`` to
        :meth:`functional_forward`."""
        return {k: v for k, v in self.named_parameters() if k.startswith(self.ACTOR_PREFIXES)}


# ----------------------------------------------------------------------------------------------------------------
# exploration rules with exact propensities
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Selection:
    index: Tensor                    # [B] chosen candidate
    probs: Tensor                    # [B, N] the distribution the choice was sampled from
    rule: str
    eps_floor: float
    diagnostics: Dict[str, Tensor] = field(default_factory=dict)

    @property
    def propensity(self) -> Tensor:
        return self.probs.gather(-1, self.index.view(-1, 1)).squeeze(-1)

    def probabilities(self, row: int = 0, n: Optional[int] = None) -> List[float]:
        """Plain list for ``contracts.Decision.probabilities`` (first ``n`` candidates of ``row``)."""
        p = self.probs[row].detach().to(torch.float64)
        return [float(x) for x in (p if n is None else p[:n])]


def apply_floor(p: Tensor, support: Tensor, eps: float) -> Tensor:
    """Mix with the uniform distribution over ``support``: every supported candidate keeps probability
    >= eps / |support|; unsupported candidates stay at exactly 0; rows sum to 1."""
    if not 0.0 <= eps < 1.0:
        raise ValueError("eps_floor must be in [0, 1)")
    sup = support.to(p.dtype)
    n = sup.sum(-1, keepdim=True)
    if bool((n == 0).any()):
        raise ValueError("every decision needs at least one candidate in the support")
    p = torch.where(support, p, torch.zeros_like(p))
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-300)
    out = (1.0 - eps) * p + eps * sup / n
    return out / out.sum(-1, keepdim=True)


def _one_hot_argmax(score: Tensor, support: Tensor) -> Tensor:
    s = torch.where(support, score, torch.full_like(score, float("-inf")))
    return F.one_hot(s.argmax(-1), score.shape[-1]).to(torch.float64)


def selection_probs(rule: str, *, a_mask: Tensor, logits: Optional[Tensor] = None, q_mean: Optional[Tensor] = None,
                    q_std: Optional[Tensor] = None, q_samples: Optional[Tensor] = None,
                    info_gain: Optional[Tensor] = None, arms: Optional[Tensor] = None,
                    arm_probs: Optional[Tensor] = None, eps_floor: float = 0.02, temperature: float = 1.0,
                    ucb_c: float = 1.0, ig_weight: float = 1.0) -> Tensor:
    """[B, N] float64 probability of selecting each candidate under ``rule``.

    policy      softmax(logits / temperature)
    thompson    share of posterior samples ``q_samples`` [S, B, N] in which the candidate is the best one
    ucb         all mass on argmax(q_mean + ucb_c * q_std)
    greedy      all mass on argmax(q_mean)  (``logits`` when ``q_mean`` is not given)
    voi         softmax((q_mean + ig_weight * info_gain) / temperature): decision-relevant information gain
                (``predict.voi``) traded against the learned value
    randomized  explicit assignment over experiment ``arms`` [B, N] bool with ``arm_probs`` (uniform by default);
                candidates that are not arms get probability 0

    Each is then mixed with ``eps_floor`` of the uniform distribution over its support (feasible candidates, or
    the arms for ``randomized``).
    """
    support = a_mask
    if rule == RULE_POLICY:
        if logits is None:
            raise ValueError("rule 'policy' needs logits")
        lg = torch.where(a_mask, logits.to(torch.float64) / temperature, torch.full_like(logits, float("-inf"),
                                                                                         dtype=torch.float64))
        p = torch.softmax(lg, -1)
    elif rule == RULE_THOMPSON:
        if q_samples is None:
            raise ValueError("rule 'thompson' needs q_samples [S, B, N]")
        s = torch.where(a_mask.unsqueeze(0), q_samples, torch.full_like(q_samples, float("-inf")))
        p = F.one_hot(s.argmax(-1), q_samples.shape[-1]).to(torch.float64).mean(0)
    elif rule == RULE_UCB:
        if q_mean is None or q_std is None:
            raise ValueError("rule 'ucb' needs q_mean and q_std")
        p = _one_hot_argmax(q_mean + ucb_c * q_std, a_mask)
    elif rule == RULE_GREEDY:
        score = q_mean if q_mean is not None else logits
        if score is None:
            raise ValueError("rule 'greedy' needs q_mean or logits")
        p = _one_hot_argmax(score, a_mask)
    elif rule == RULE_INFO_GAIN:
        if info_gain is None:
            raise ValueError("rule 'voi' needs info_gain")
        base = torch.zeros_like(info_gain) if q_mean is None else q_mean
        sc = ((base + ig_weight * info_gain) / temperature).to(torch.float64)
        p = torch.softmax(torch.where(a_mask, sc, torch.full_like(sc, float("-inf"))), -1)
    elif rule == RULE_RANDOMIZED:
        if arms is None:
            raise ValueError("rule 'randomized' needs arms")
        support = arms & a_mask
        if bool((arms & ~a_mask).any()):
            raise ValueError("an experiment arm must be a feasible candidate")
        p = support.to(torch.float64) if arm_probs is None else arm_probs.to(torch.float64)
        if arm_probs is not None and bool(((arm_probs > 0) & ~support).any()):
            raise ValueError("arm_probs puts mass outside the arms")
    else:
        raise ValueError(f"unknown selection rule {rule!r}; choose from {RULES}")
    return apply_floor(p.to(torch.float64), support, eps_floor)


def select(rule: str, *, a_mask: Tensor, generator: Optional[torch.Generator] = None, **kw) -> Selection:
    """Choose one candidate per decision and return the exact probability vector it was drawn from."""
    probs = selection_probs(rule, a_mask=a_mask, **kw)
    idx = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(-1).to(probs.device)
    diag = {"entropy": -(probs * torch.log(probs.clamp_min(1e-300))).sum(-1),
            "n_support": (probs > 0).sum(-1)}
    return Selection(idx, probs, rule, float(kw.get("eps_floor", 0.02)), diag)


# ----------------------------------------------------------------------------------------------------------------
# does the policy depend on CAM's latent?
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def latent_dependence(policy: CandidatePolicy, z: Tensor, a_mask: Tensor, b: Tensor, *,
                      z_fn: Optional[Callable[[Tensor], Tensor]] = None, perm: Optional[Tensor] = None
                      ) -> Dict[str, float]:
    """Mean KL(pi(.|b) || pi(.|b permuted across the batch)) and total variation, candidates held fixed.

    ``z_fn(b) -> z`` recomputes the decision latents for the permuted beliefs (pass the model's
    ``decision_latent`` closed over e_a and e_g); without it only the policy's direct belief path is tested.
    A value near zero means the choice is decided by the candidates alone."""
    if perm is None:
        perm = torch.roll(torch.arange(b.shape[0], device=b.device), 1)
    base = policy(z if z_fn is None else z_fn(b), a_mask, b)
    bp = b[perm]
    other = policy(z if z_fn is None else z_fn(bp), a_mask, bp)
    p, q = base.probs.double(), other.probs.double()
    kl = (p * (torch.log(p.clamp_min(1e-300)) - torch.log(q.clamp_min(1e-300)))).sum(-1)
    return {"kl": float(kl.mean()), "tv": float(0.5 * (p - q).abs().sum(-1).mean()),
            "argmax_changed": float((p.argmax(-1) != q.argmax(-1)).double().mean())}
