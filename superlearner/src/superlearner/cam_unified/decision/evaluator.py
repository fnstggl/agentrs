"""Pre-execution evaluation of CAM's CONCRETE candidate actions (DOM Mechanism B; plan 2026-11-26 §2, W5).

For every candidate k of every row (its executable tool + arguments, tokenised as ACTION tokens A_k):

    Z^_k   = WM.prior(R_t, A_k, B_t, dt_k)                     (the world model's action-conditioned next latent)
    r^_k   = EvalHead(pool[Z^_k; A_k; I_t])                     (learned; zero-initialised output: 0 at init)
    Q_k    = r^_k + Gamma_k * ( w_phi * sg Phi(Z^_k, G) + w_v * sg V_int(Z^_k, I_t) ),   Gamma_k = exp(-kappa dt_k)

Phi is the reward model's potential (`RewardModelP.potential`, ProspectEvaluator P(success | R, G) trained only on
verified labels; zero without a reward model) and V_int the intrinsic critic V(., I). Both enter Q stop-gradient:
the actor's selection loss trains the evaluator head and, through Z^_k, the world model's transition and the core;
it never moves psi, Phi or the critic. `eval_loss` is the evaluator's own regression (L_eval: Q of the executed
action toward the learner's n-step target r_psi + Gamma V_int(next)).

`exact_rows=True` (default) evaluates each row's K candidates as their own batch (padded to that row's longest
candidate only), so a row's Q does not depend on the minibatch it is evaluated in: the selection log-probability
recomputed by `policy_evaluate` equals the recorded one bit for bit (tests/cam_unified/dom/test_selection_replay.py).
`exact_rows=False` evaluates all B*K rows in one prior pass (faster at 1B; float-level drift between batchings).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..config import CamConfig
from ..interfaces import TokenSet
from ..learn.heads import AttentionPool, _MLP
from .intent import intent_tokens


def _rep(ts: Optional[TokenSet], idx: Tensor) -> Optional[TokenSet]:
    return None if ts is None else ts.select(idx)


def _value(out) -> Tensor:
    return (out.value if hasattr(out, "value") else out).float()


def row_cut(ts: Optional[TokenSet], b: int) -> Optional[TokenSet]:
    """Row b of a TokenSet cut to its own length (up to its last real token), so a row's evaluation does not depend
    on the padding of the batch it came from (same rule as semantic.actions.row_tokens)."""
    if ts is None:
        return None
    m = ts.mask[b]
    n = int(torch.nonzero(m).max()) + 1 if bool(m.any()) else 0
    return TokenSet(ts.x[b:b + 1, :n], ts.mask[b:b + 1, :n], ts.kind[b:b + 1, :n], ts.time[b:b + 1, :n],
                    ts.entity[b:b + 1, :n], None if ts.refs is None else [list(ts.refs[b][:n])],
                    None if ts.entity2 is None else ts.entity2[b:b + 1, :n])


class CandidateEvaluator(nn.Module):
    """Q of concrete candidate actions. Parameters: the attention pool and the residual head r^ (theta_eval)."""

    def __init__(self, cfg: CamConfig, n_heads: int = 4, hidden: Optional[int] = None, w_phi: float = 1.0,
                 w_v: float = 1.0):
        super().__init__()
        D = cfg.d_model
        self.d_model = D
        self.kappa = float(cfg.kappa)
        self.tau = float(getattr(cfg, "select_tau", 1.0))
        self.veto_q = getattr(cfg, "veto_q", None)
        self.w_phi = float(w_phi)
        self.w_v = float(w_v)
        self.pool = AttentionPool(D, n_heads if D % n_heads == 0 else 1, 1)
        self.head = _MLP(D, hidden or D, 1, zero_out=True)
        self.exact_rows = True
        self.calls = {"score": 0, "rows": 0, "prior_passes": 0}

    # ------------------------------------------------------------------ pieces
    def gamma(self, dt: Tensor) -> Tensor:
        return torch.exp(-self.kappa * dt.float().clamp_min(0))

    def residual(self, Z_hat: Tensor, act: TokenSet, intent: Optional[Tensor]) -> Tensor:
        extra = [act] + ([intent_tokens(intent)] if intent is not None else [])
        h = self.pool(Z_hat, extra)[:, 0]
        return self.head(h).squeeze(-1).float()

    def prior_rows(self, R: Tensor, belief: Optional[TokenSet], act: TokenSet, dt: Tensor, world_model) -> Tensor:
        """Z^ [n,S,D] = WM.prior(R, A, B, dt) (deterministic: the prior mode) for n candidate rows."""
        self.calls["prior_passes"] += 1
        Z_hat, _ = world_model.prior(R, act, belief, dt, sample=False)
        return Z_hat.float()

    @staticmethod
    def potential_rows(Z_hat: Tensor, goal: Optional[TokenSet], potential: Optional[Callable]) -> Tensor:
        """Phi(Z^, G) [n], stop-gradient (0 without a reward model)."""
        if potential is None:
            return torch.zeros(Z_hat.shape[0], device=Z_hat.device)
        with torch.no_grad():
            return potential(Z_hat.detach(), goal).float().reshape(-1).detach()

    def q_rows(self, Z_hat: Tensor, act: TokenSet, dt: Tensor, intent: Optional[Tensor], value_head,
               phi: Tensor) -> tuple:
        """Q [n] = r^ + Gamma (w_phi sg Phi + w_v sg V(Z^, I)) and its parts, from given prior outputs."""
        r_hat = self.residual(Z_hat, act, intent)
        g = self.gamma(dt)
        if value_head is not None:
            ex = [intent_tokens(intent)] if intent is not None else []
            v = _value(value_head(Z_hat, None, ex)).detach()
        else:
            v = torch.zeros_like(r_hat)
        q = r_hat + g * (self.w_phi * phi.float() + self.w_v * v)
        return q, {"r_hat": r_hat, "phi": phi.float(), "v": v, "gamma": g}

    def _group(self, R, belief, goal, act, dt, intent, world_model, value_head, potential):
        """One prior pass over n candidate rows (R, belief, goal, intent already expanded to the rows)."""
        Z_hat = self.prior_rows(R, belief, act, dt, world_model)
        phi = self.potential_rows(Z_hat, goal, potential)
        q, parts = self.q_rows(Z_hat, act, dt, intent, value_head, phi)
        parts["Z_hat"] = Z_hat
        return q, parts

    # ------------------------------------------------------------------ score
    def score(self, R: Tensor, belief: Optional[TokenSet], goal: Optional[TokenSet], cand_tokens, dt: Tensor, *,
              world_model, value_head=None, intent: Optional[Tensor] = None,
              potential: Optional[Callable] = None, K: Optional[int] = None,
              exact_rows: Optional[bool] = None) -> tuple:
        """Q [B, K] and preds {Z_hat [B,K,S,D], r_hat, phi, v, gamma, dt: [B, K]}.

        R [B,S,D]; belief / goal TokenSets [B, .]; intent [B, n, D] or None; dt [B, K] (or [B*K]) days.
        cand_tokens: a list of B TokenSets [K, Na_b, D] (row b's candidates, padded within the row; required for
        exact_rows) or one TokenSet [B*K, Na, D] (row-major: candidate k of row b at b*K + k)."""
        exact = self.exact_rows if exact_rows is None else exact_rows
        B = R.shape[0]
        per_row = isinstance(cand_tokens, (list, tuple))
        if per_row:
            K = cand_tokens[0].B if K is None else K
            if len(cand_tokens) != B or any(c.B != K for c in cand_tokens):
                raise ValueError("cand_tokens: need B TokenSets of K rows each")
        elif K is None:
            K = cand_tokens.B // B
        if not per_row and cand_tokens.B != B * K:
            raise ValueError(f"cand_tokens has {cand_tokens.B} rows, expected B*K = {B * K}")
        dt = dt.reshape(B, K).float()
        self.calls["score"] += 1
        self.calls["rows"] += B * K
        if exact:
            if not per_row:
                raise ValueError("exact_rows needs per-row candidate tokens (a list of B TokenSets)")
            outs = []
            z = torch.zeros(K, dtype=torch.long, device=R.device)
            for b in range(B):
                idx = torch.full((K,), b, dtype=torch.long, device=R.device)
                outs.append(self._group(R[idx], _rep(row_cut(belief, b), z), _rep(row_cut(goal, b), z),
                                        cand_tokens[b], dt[b], None if intent is None else intent[idx],
                                        world_model, value_head, potential))
            Q = torch.stack([o[0] for o in outs], 0)
            preds = {k: torch.stack([o[1][k] for o in outs], 0) for k in outs[0][1]}
        else:
            if per_row:
                from ..model import cat_rows
                act = cat_rows(list(cand_tokens))
            else:
                act = cand_tokens
            idx = torch.arange(B, device=R.device).repeat_interleave(K)
            q, pr = self._group(R[idx], _rep(belief, idx), _rep(goal, idx), act, dt.reshape(-1),
                                None if intent is None else intent[idx], world_model, value_head, potential)
            Q = q.view(B, K)
            preds = {k: v.view(B, K, *v.shape[1:]) for k, v in pr.items()}
        preds["dt"] = dt
        return Q, preds

    # ------------------------------------------------------------------ the evaluator's own objective
    @staticmethod
    def eval_loss(q: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
        """Huber regression of Q (executed action) toward the learner's detached n-step target [B] -> [B]."""
        return nn.functional.huber_loss(q.float(), target.detach().float(), reduction="none", delta=delta)


def score(model, R: Tensor, B_t: Optional[TokenSet], G: Optional[TokenSet], cand_tokens, dt: Tensor,
          intent: Optional[Tensor] = None, **kw) -> tuple:
    """Plan §3 W5 signature: Q [B,K], preds = model.decision_evaluator.score(...) with the model's world model,
    value head and the attached reward model's potential (model.reward_model; None -> Phi = 0)."""
    ev = model.decision_evaluator
    rm = getattr(model, "reward_model", None)
    pot = getattr(rm, "potential", None) if rm is not None else None
    return ev.score(R, B_t, G, cand_tokens, dt, world_model=model.world_model, value_head=model.value_head,
                    intent=intent, potential=pot, **kw)


__all__ = ["CandidateEvaluator", "score"]
