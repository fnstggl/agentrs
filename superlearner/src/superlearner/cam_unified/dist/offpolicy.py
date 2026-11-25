"""Policy-lag accounting and off-policy routing for asynchronous actor data (spec 14.4, manifest F03).

Versions are learner *parameter* versions: the learner increments its version after every optimiser update and
publishes some of them, so ``lag = learner_version − record.model_version`` is exactly the number of updates the
behaviour policy is behind the parameters being trained. Routing:

* ``lag == 0``                      → ``on_policy``  (π = μ; PPO-style ratios are valid);
* ``0 < lag <= max_lag``            → ``off_policy`` (must be corrected: V-trace with clipped importance weights
                                      computed from the *recorded* behaviour log-probability);
* ``lag > max_lag`` (or no logp)    → ``discarded`` from training (it stays in the archive; counted, never silent).

`vtrace` is the duration-aware V-trace of Espeholt et al. (2018) — copy-adapted from the tested
``superlearner/unified/learning/returns.py`` — with the semi-Markov discount γ_t = exp(−κ·Δt_t) (Δt in simulated
days, `semi_markov_discount`). Workstream D's ``learn/vtrace.py`` is the CAM objective's implementation; this one
serves the distributed stack and its benchmarks, and the two must agree (`vtrace` reduces to the Monte Carlo
return at zero log-ratio and λ = 1, tested).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


def semi_markov_discount(dt_days: Tensor, kappa: float) -> Tensor:
    """γ = exp(−κ Δt); Δt in simulated days, Δt >= 0."""
    if (dt_days < 0).any():
        raise ValueError("elapsed simulated time must be non-negative")
    return torch.exp(-kappa * dt_days)


def vtrace(rewards: Tensor, values: Tensor, bootstrap: Tensor, discounts: Tensor, log_rho: Tensor,
           rho_bar: float = 1.0, c_bar: float = 1.0, lam: float = 1.0) -> tuple:
    """V-trace targets.

    rewards, values, discounts, log_rho: [B, T] (discounts already include termination: 0 after a terminal step);
    bootstrap: [B] value of the state after the last step. log_rho = log π(a_t|x_t) − log μ(a_t|x_t).
    Returns (vs [B, T] value targets, pg_adv [B, T] policy-gradient advantages ρ_t (r_t + γ_t v_{t+1} − V(x_t)),
    rho [B, T] clipped weights). All outputs are detached.
    """
    with torch.no_grad():
        rho_raw = log_rho.exp()
        rho = rho_raw.clamp(max=rho_bar)
        c = lam * rho_raw.clamp(max=c_bar)
        v_next = torch.cat([values[:, 1:], bootstrap[:, None]], 1)
        delta = rho * (rewards + discounts * v_next - values)
        acc = torch.zeros_like(bootstrap)
        vs_minus_v = torch.zeros_like(values)
        for t in reversed(range(rewards.shape[1])):
            acc = delta[:, t] + discounts[:, t] * c[:, t] * acc
            vs_minus_v[:, t] = acc
        vs = values + vs_minus_v
        vs_next = torch.cat([vs[:, 1:], bootstrap[:, None]], 1)
        pg_adv = rho * (rewards + discounts * vs_next - values)
    return vs, pg_adv, rho


@dataclass
class LagReport:
    learner_version: int
    on_policy: list = field(default_factory=list)      # records with lag 0
    off_policy: list = field(default_factory=list)     # 0 < lag <= max_lag
    discarded: list = field(default_factory=list)      # lag > max_lag, missing behaviour logp, or future version
    lag_hist: dict = field(default_factory=dict)       # lag -> count (all routed records)
    reasons: dict = field(default_factory=dict)        # discard reason -> count

    def summary(self) -> dict:
        lags = [l for l, n in self.lag_hist.items() for _ in range(n)]
        lags.sort()
        q = (lambda p: lags[min(len(lags) - 1, int(p * len(lags)))] if lags else None)
        return {"learner_version": self.learner_version, "n_on_policy": len(self.on_policy),
                "n_off_policy": len(self.off_policy), "n_discarded": len(self.discarded),
                "lag_mean": (sum(lags) / len(lags)) if lags else None, "lag_p50": q(0.5), "lag_p90": q(0.9),
                "lag_max": lags[-1] if lags else None, "lag_hist": {str(k): v for k, v in sorted(self.lag_hist.items())},
                "discard_reasons": dict(self.reasons)}


class LagRouter:
    """Routes EventRecords by policy lag. ``mode="vtrace"`` keeps 0 < lag <= max_lag as off-policy data;
    ``mode="discard"`` keeps only lag 0 (strictly on-policy training)."""

    def __init__(self, max_lag: int = 8, mode: str = "vtrace"):
        if mode not in ("vtrace", "discard"):
            raise ValueError(mode)
        self.max_lag = int(max_lag)
        self.mode = mode
        self.total = Counter()
        self.total_hist = Counter()

    def route(self, records: list, learner_version: int, needs_logp: bool = True) -> LagReport:
        """Route individual decision EventRecords (records without an action — delayed outcomes, world events —
        follow their decision and are not routed here)."""
        return self._route([r for r in records if r.action is not None], learner_version,
                           version=lambda r: int(r.model_version),
                           has_logp=lambda r: (not needs_logp) or r.behavior_logp is not None, weight=lambda r: 1)

    def route_trajectories(self, trajs: list, learner_version: int, needs_logp: bool = True) -> LagReport:
        """Route whole trajectories (objects with ``events`` and ``behavior_version``); counts are in decisions.
        A trajectory is trainable only if every decision has a behaviour log-probability."""
        return self._route(trajs, learner_version, version=lambda t: int(t.behavior_version),
                           has_logp=lambda t: (not needs_logp) or all(e.behavior_logp is not None for e in t.events),
                           weight=lambda t: len(t.events))

    def _route(self, items, learner_version, version, has_logp, weight) -> LagReport:
        rep = LagReport(learner_version)
        hist, reasons = Counter(), Counter()
        for r in items:
            lag = learner_version - version(r)
            w = weight(r)
            hist[lag] += w
            if lag < 0:
                reasons["future_version"] += w
                rep.discarded.append(r)
            elif not has_logp(r):
                reasons["no_behavior_logp"] += w
                rep.discarded.append(r)
            elif lag == 0:
                rep.on_policy.append(r)
            elif self.mode == "vtrace" and lag <= self.max_lag:
                rep.off_policy.append(r)
            else:
                reasons["lag_over_max" if self.mode == "vtrace" else "stale_in_discard_mode"] += w
                rep.discarded.append(r)
        rep.lag_hist = dict(hist)
        rep.reasons = dict(reasons)
        self.total.update({"on_policy": sum(weight(r) for r in rep.on_policy),
                           "off_policy": sum(weight(r) for r in rep.off_policy),
                           "discarded": sum(weight(r) for r in rep.discarded)})
        self.total.update({f"discard:{k}": v for k, v in reasons.items()})
        self.total_hist.update(hist)
        return rep

    def totals(self) -> dict:
        return {"counts": dict(self.total), "lag_hist": {str(k): v for k, v in sorted(self.total_hist.items())},
                "max_lag": self.max_lag, "mode": self.mode}


def importance_log_ratio(target_logp: Tensor, behavior_logp: Tensor, lag: Tensor,
                         on_policy_exact: bool = True) -> Tensor:
    """log ρ = log π − log μ. With ``on_policy_exact`` rows at lag 0 use 0 (π = μ by construction; avoids
    re-scoring noise such as bf16 padding effects being read as off-policyness)."""
    lr = target_logp - behavior_logp
    if on_policy_exact:
        lr = torch.where(lag == 0, torch.zeros_like(lr), lr)
    return lr


def lag_summary(lags: list) -> Optional[dict]:
    if not lags:
        return None
    s = sorted(lags)
    return {"mean": sum(s) / len(s), "p50": s[len(s) // 2], "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
            "max": s[-1]}
