"""Decision heads of workstream D: value V(R, G), hierarchical options and the BUDGET token.

* `ValueHead` — distributional critic over symlog two-hot bins (DreamerV3), so one head works for rewards measured in
  cents of revenue, 0/1 success bonuses or learning-progress nats without retuning. Attention pooling from a learned
  query over the reasoned state ``R [B,S,D]``, the goal tokens and any extra context (BUDGET, OPTION tokens).
* `OptionModule` — ``n_options`` learned option embeddings (latent intents, spec 9.4); a categorical option policy
  (pointer attention from R to the option embeddings); a termination head beta(R, o); OPTION tokens for the core's
  working context; exact log-probabilities of every option decision (select / continue / terminate+reselect), which
  enter the action's behaviour log-probability so PPO / V-trace train the option policy and the termination head with
  the same advantages as the primitive actions (semi-Markov credit: `returns.option_targets`).
* `BudgetEncoder` — remaining simulated time, funds and decisions (and fractions of their totals when known) as one
  BUDGET token, so the same policy can trade exploration against the remaining horizon (spec 8.2).

symlog / two-hot helpers are copy-adapted from ``superlearner/src/superlearner/unified/dist.py`` (CAM 6.1, tested
against DreamerV3 constants in ``superlearner/tests/unified/test_returns.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import TokenKind, TokenSet


# ----------------------------------------------------------------------------------------------- symlog two-hot

def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def twohot_encode(y: Tensor, bins: Tensor) -> Tensor:
    """Spread ``y`` (already in bin space) over its two neighbouring bins: ``y.shape + [n_bins]``."""
    y = y.clamp(bins[0], bins[-1])
    above = torch.searchsorted(bins, y.contiguous(), right=False).clamp(1, len(bins) - 1)
    below = above - 1
    span = bins[above] - bins[below]
    w_above = (y - bins[below]) / span
    target = torch.zeros(*y.shape, len(bins), device=y.device, dtype=y.dtype)
    target.scatter_(-1, below.unsqueeze(-1), (1 - w_above).unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
    return target


def twohot_mean(logits: Tensor, bins: Tensor) -> Tensor:
    """Expected value in the original scale; symmetric summation gives exactly 0 for uniform logits."""
    probs = F.softmax(logits.float(), -1)
    n = len(bins)
    m = (n - 1) // 2
    p1, p2, p3 = probs[..., :m], probs[..., m:m + 1], probs[..., m + 1:]
    b1, b2, b3 = bins[:m], bins[m:m + 1], bins[m + 1:]
    wavg = (p2 * b2).sum(-1) + ((p1.flip(-1) * b1.flip(-1)) + (p3 * b3)).sum(-1)
    return symexp(wavg)


# ----------------------------------------------------------------------------------------------- building blocks

class _MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, zero_out: bool = False):
        super().__init__()
        self.norm = nn.RMSNorm(d_in)
        self.fc1 = nn.Linear(d_in, 2 * d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        if zero_out:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.fc1(self.norm(x)).chunk(2, -1)
        return self.fc2(F.silu(a) * b)          # SwiGLU


class AttentionPool(nn.Module):
    """``n_queries`` learned queries attend over ``[R; extra tokens]`` with kind embeddings and padding masks.

    R [B,S,D] (always valid); tokens: list of optional TokenSets [B,N_i,D]. Returns [B, n_queries, D]. A learned
    always-valid null key keeps rows with no valid context well defined."""

    def __init__(self, d_model: int, n_heads: int = 4, n_queries: int = 1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.null = nn.Parameter(torch.zeros(1, d_model))
        self.kind = nn.Embedding(int(TokenKind.N_KINDS), d_model)
        nn.init.normal_(self.kind.weight, std=0.02)
        self.norm = nn.RMSNorm(d_model)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.qp = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, R: Tensor, tokens: list | tuple = (), state_kind: int = TokenKind.STATE) -> Tensor:
        B, S, D = R.shape
        xs = [R + self.kind.weight[int(state_kind)], self.null.expand(B, 1, D)]
        ms = [torch.ones(B, S + 1, dtype=torch.bool, device=R.device)]
        for ts in tokens:
            if ts is None or ts.N == 0:
                continue
            if ts.B != B:
                raise ValueError(f"context batch {ts.B} != state batch {B}")
            xs.append(ts.x.to(R.dtype) + self.kind(ts.kind))
            ms.append(ts.mask.bool())
        x = self.norm(torch.cat(xs, 1))
        m = torch.cat(ms, 1)
        k, v = self.kv(x).chunk(2, -1)
        q = self.qp(self.q).unsqueeze(0).expand(B, -1, -1)
        nq = q.shape[1]

        def heads(t):
            return t.view(B, -1, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=m[:, None, None, :])
        o = o.transpose(1, 2).reshape(B, nq, D)
        return self.out(o) + self.q.unsqueeze(0)


# ----------------------------------------------------------------------------------------------- value

@dataclass
class ValueOutput:
    value: Tensor     # [B] expected return in the original reward scale
    logits: Tensor    # [B, n_bins] distribution over symlog bins


class ValueHead(nn.Module):
    """Distributional value V(R, G[, extra]) over symlog two-hot bins.

    forward(R [B,S,D], goal TokenSet [B,Ng,D] | None, extra list[TokenSet] | None) -> ValueOutput(value [B],
    logits [B,n_bins]). ``loss(logits [..., n_bins], target [...])`` is the two-hot cross-entropy of
    ``symlog(target)`` (per element). The output layer starts at zero, so the initial value is exactly 0."""

    def __init__(self, d_model: int, n_heads: int = 4, n_bins: int = 255, low: float = -20.0, high: float = 20.0,
                 hidden: Optional[int] = None):
        super().__init__()
        if n_bins % 2 == 0:
            raise ValueError("n_bins must be odd (symmetric bins around 0)")
        self.pool = AttentionPool(d_model, n_heads, 1)
        self.mlp = _MLP(d_model, hidden or d_model, n_bins, zero_out=True)
        self.register_buffer("bins", torch.linspace(low, high, n_bins), persistent=False)

    def forward(self, R: Tensor, goal: Optional[TokenSet] = None, extra: Optional[list] = None,
                state_kind: int = TokenKind.STATE) -> ValueOutput:
        h = self.pool(R, [goal] + list(extra or []), state_kind=state_kind)[:, 0]
        logits = self.mlp(h).float()
        return ValueOutput(twohot_mean(logits, self.bins), logits)

    def value_from_logits(self, logits: Tensor) -> Tensor:
        return twohot_mean(logits, self.bins)

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        tgt = twohot_encode(symlog(target.detach().float()), self.bins)
        return -(tgt * F.log_softmax(logits.float(), -1)).sum(-1)


# ----------------------------------------------------------------------------------------------- options

@dataclass
class OptionStep:
    option: Tensor        # [B] long: option active for this decision
    new: Tensor           # [B] bool: a (re)selection happened at this decision
    terminated: Tensor    # [B] bool: the previous option was terminated (sampled or forced)
    logp: Tensor          # [B] log-probability of the option decision (select / continue / terminate+reselect)
    entropy: Tensor       # [B] entropy of the option decision
    beta: Tensor          # [B] termination probability of the previous option (1 if none / forced)
    steps: Tensor         # [B] long: decisions the active option has run so far, including this one


class OptionModule(nn.Module):
    """Learned temporally extended options (spec 9.4).

    Option embeddings E [n_options, D] are latent intents; nothing in the code names what an option does, so useful
    options can only emerge from experience. Per decision, given the reasoned state R [B,S,D] and goal tokens:

    * if no option is active (``prev = -1``) or the active one reached ``max_steps``: select ``o ~ pi(o | R, G)``;
    * else terminate with probability ``beta(R, o_prev)``: on termination select ``o ~ pi(o | R, G)`` (possibly the
      same id), otherwise continue ``o_prev``.

    ``log_prob`` recomputes the exact probability of the recorded decision (differentiable), so it can be summed into
    the action log-probability for PPO / V-trace. ``token`` returns the OPTION token [B,1,D] (embedding plus a code of
    the option's elapsed fraction) for the core's working context; ``mask`` is False when no option is active."""

    def __init__(self, d_model: int, n_options: int = 16, max_steps: int = 32, n_heads: int = 4):
        super().__init__()
        self.n, self.max_steps, self.D = n_options, max_steps, d_model
        self.emb = nn.Parameter(torch.randn(n_options, d_model) / math.sqrt(d_model))
        self.pool = AttentionPool(d_model, n_heads, 2)
        self.sel_q = nn.Linear(d_model, d_model)
        self.sel_b = nn.Parameter(torch.zeros(n_options))
        self.term = _MLP(2 * d_model, d_model, 1)
        self.elapsed = nn.Linear(2, d_model)
        with torch.no_grad():                     # start with long options (beta ~ 0.12) rather than coin flips
            self.term.fc2.bias.fill_(-2.0)

    # -- pieces
    def _pooled(self, R: Tensor, goal: Optional[TokenSet]):
        h = self.pool(R, [goal])
        return h[:, 0], h[:, 1]

    def select_logits(self, R: Tensor, goal: Optional[TokenSet] = None) -> Tensor:
        """[B, n_options] logits of the option policy (pointer attention onto the option embeddings)."""
        hs, _ = self._pooled(R, goal)
        return (self.sel_q(hs) @ self.emb.t()) / math.sqrt(self.D) + self.sel_b

    def beta(self, R: Tensor, option: Tensor, goal: Optional[TokenSet] = None) -> Tensor:
        """[B] termination probability of option ``option`` ([B] long; -1 rows get 1)."""
        _, hb = self._pooled(R, goal)
        e = self.emb[option.clamp_min(0)]
        b = torch.sigmoid(self.term(torch.cat([hb, e], -1)).squeeze(-1).float())
        return torch.where(option >= 0, b, torch.ones_like(b))

    def token(self, option: Tensor, steps: Optional[Tensor] = None) -> TokenSet:
        """OPTION token [B,1,D]; masked out where ``option < 0``."""
        B = option.shape[0]
        steps = torch.zeros_like(option) if steps is None else steps
        frac = (steps.float() / float(self.max_steps)).clamp(0, 1)
        code = self.elapsed(torch.stack([frac, torch.log1p(steps.float().clamp_min(0))], -1))
        x = (self.emb[option.clamp_min(0)] + code).unsqueeze(1)
        mask = (option >= 0).unsqueeze(1)
        return TokenSet(x, mask, torch.full((B, 1), int(TokenKind.OPTION), dtype=torch.long, device=option.device),
                        torch.zeros(B, 1, device=option.device), torch.full((B, 1), -1, dtype=torch.long,
                                                                             device=option.device),
                        [[f"option:{int(o)}"] for o in option.tolist()])

    # -- decisions
    def _decision_terms(self, R, goal, prev, steps_prev):
        logits = self.select_logits(R, goal).float()
        logp_sel = F.log_softmax(logits, -1)
        ent_sel = -(logp_sel.exp() * logp_sel).sum(-1)
        forced = (prev < 0) | (steps_prev >= self.max_steps)
        beta = torch.where(forced, torch.ones(prev.shape, device=R.device), self.beta(R, prev, goal))
        return logp_sel, ent_sel, forced, beta

    def step(self, R: Tensor, goal: Optional[TokenSet], prev: Tensor, steps_prev: Tensor, greedy: bool = False,
             generator: Optional[torch.Generator] = None) -> OptionStep:
        """Sample (or take the mode of) the option decision for one decision step.
        prev [B] long (-1 = none); steps_prev [B] long decisions the previous option has run."""
        logp_sel, ent_sel, forced, beta = self._decision_terms(R, goal, prev, steps_prev)
        if greedy:
            term = forced | (beta > 0.5)
            choice = logp_sel.argmax(-1)
        else:
            u = torch.rand(prev.shape, device=R.device, generator=generator)
            term = forced | (u < beta)
            choice = torch.multinomial(logp_sel.exp(), 1, generator=generator).squeeze(-1)
        option = torch.where(term, choice, prev)
        logp, ent = self._logp_entropy(logp_sel, ent_sel, forced, beta, option, term)
        steps = torch.where(term, torch.ones_like(steps_prev), steps_prev + 1)
        return OptionStep(option, term, term & (prev >= 0), logp, ent, beta, steps)

    def _logp_entropy(self, logp_sel, ent_sel, forced, beta, option, new):
        eps = 1e-6
        lsel = logp_sel.gather(-1, option.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        b = beta.clamp(eps, 1 - eps)
        lp_free = torch.where(new, torch.log(b) + lsel, torch.log1p(-b))
        logp = torch.where(forced, lsel, lp_free)
        h_bern = -(b * torch.log(b) + (1 - b) * torch.log1p(-b))
        ent = torch.where(forced, ent_sel, h_bern + b * ent_sel)
        return logp, ent

    def log_prob(self, R: Tensor, goal: Optional[TokenSet], prev: Tensor, steps_prev: Tensor, option: Tensor,
                 new: Tensor) -> tuple[Tensor, Tensor]:
        """Differentiable (logp [B], entropy [B]) of the recorded option decision (``option``, ``new``)."""
        logp_sel, ent_sel, forced, beta = self._decision_terms(R, goal, prev, steps_prev)
        new = new.bool() | forced
        if ((~new) & (option != prev)).any():
            raise ValueError("a continued option must keep the previous option id")
        return self._logp_entropy(logp_sel, ent_sel, forced, beta, option, new)


def deliberation_penalty(terminated: Tensor, cost: float) -> Tensor:
    """Reward shaping ``-cost`` per voluntary option termination (deliberation cost, Harb et al. 2018): encourages
    options that stay committed unless switching is worth it. terminated [B,T] bool -> [B,T] float."""
    return -float(cost) * terminated.float()


# ----------------------------------------------------------------------------------------------- budget

@dataclass
class BudgetState:
    """Remaining resources per environment. NaN = unknown. Shapes [B]."""
    days_left: Tensor
    funds_left: Tensor          # in the world's money unit as given (Crucible: cents)
    decisions_left: Tensor
    days_total: Tensor
    funds_total: Tensor
    decisions_total: Tensor

    @staticmethod
    def make(B: int, device=None, **kw) -> "BudgetState":
        nan = torch.full((B,), float("nan"), device=device)
        vals = {}
        for f in ("days_left", "funds_left", "decisions_left", "days_total", "funds_total", "decisions_total"):
            v = kw.get(f)
            vals[f] = nan.clone() if v is None else torch.as_tensor(v, dtype=torch.float32, device=device).expand(B).clone()
        return BudgetState(**vals)

    def to(self, device) -> "BudgetState":
        return BudgetState(*(getattr(self, f).to(device) for f in
                             ("days_left", "funds_left", "decisions_left", "days_total", "funds_total",
                              "decisions_total")))

    def stack(self) -> Tensor:
        """[B, 6] raw values (NaN kept)."""
        return torch.stack([self.days_left, self.funds_left, self.decisions_left, self.days_total, self.funds_total,
                            self.decisions_total], -1)


class BudgetEncoder(nn.Module):
    """BUDGET token [B,1,D] from a `BudgetState`.

    Per quantity (time, funds, decisions): presence flag, symlog(remaining), remaining fraction of the total (when the
    total is known), and ``n_fourier`` sin/cos features of symlog(remaining) — robust to the very different scales of
    days, cents and decision counts."""

    N_Q = 3

    def __init__(self, d_model: int, n_fourier: int = 8):
        super().__init__()
        self.nf = n_fourier
        self.register_buffer("freq", 2.0 ** torch.arange(n_fourier).float() * 0.25, persistent=False)
        self.d_in = self.N_Q * (3 + 2 * n_fourier + 1)
        self.mlp = _MLP(self.d_in, d_model, d_model)

    def features(self, b: BudgetState) -> Tensor:
        raw = b.stack().float()                          # [B,6]
        left, total = raw[:, :3], raw[:, 3:]
        has_left = torch.isfinite(left)
        has_tot = torch.isfinite(total) & (total > 0)
        l0 = torch.where(has_left, left, torch.zeros_like(left))
        sl = symlog(l0)
        frac = torch.where(has_left & has_tot, (l0 / torch.where(has_tot, total, torch.ones_like(total))).clamp(0, 1),
                           torch.zeros_like(l0))
        ang = sl.unsqueeze(-1) * self.freq                # [B,3,nf]
        four = torch.cat([ang.sin(), ang.cos()], -1) * has_left.unsqueeze(-1)
        feats = torch.cat([has_left.float().unsqueeze(-1), sl.unsqueeze(-1), frac.unsqueeze(-1),
                           has_tot.float().unsqueeze(-1), four], -1)    # [B,3,4+2nf]
        return feats.flatten(1)

    def forward(self, b: BudgetState) -> TokenSet:
        f = self.features(b)
        x = self.mlp(f).unsqueeze(1)
        B = x.shape[0]
        dev = x.device
        return TokenSet(x, torch.ones(B, 1, dtype=torch.bool, device=dev),
                        torch.full((B, 1), int(TokenKind.BUDGET), dtype=torch.long, device=dev),
                        torch.zeros(B, 1, device=dev), torch.full((B, 1), -1, dtype=torch.long, device=dev),
                        [["budget"] for _ in range(B)])
