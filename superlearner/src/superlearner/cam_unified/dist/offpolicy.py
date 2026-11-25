"""Policy-lag accounting and off-policy routing for asynchronous actor data (spec 14.4, manifest F03).

Versions are learner *parameter* versions: the learner increments its version after every optimiser update and
publishes some of them, so ``lag = learner_version − record.model_version`` is exactly the number of updates the
behaviour policy is behind the parameters being trained. Lag is a property of each **decision**, not of a
trajectory: with decision-boundary refresh one trajectory holds decisions of several versions (versions never
decrease along a trajectory, so lags never increase). Routing, per decision:

* ``lag == 0``                      → on-policy;
* ``0 < lag <= max_lag``            → off-policy (corrected: V-trace with clipped importance weights computed from the
                                      *recorded* behaviour log-probability);
* ``lag > max_lag``, a future version, or no recorded behaviour log-probability → not trained (the decision stays in
  the trajectory as recurrent context only, is counted per reason and logged; it stays in the archive).

A trajectory (or trajectory segment) is ``discarded`` only when none of its decisions is trainable.

**Exactly on-policy.** A lag-0 decision is treated as exactly on-policy (log ρ = 0, which avoids reading re-scoring
noise as off-policyness) only when (1) the actor's latent state at that decision was built entirely under the same
weights version (``extra["state_version"] == model_version``: no decision-boundary refresh without latent rebuild
since the state was built), (2) the behaviour policy was stochastic (``extra["deterministic"]`` false — a greedy
argmax has μ(a) = 1, which is not π), and (3) the learner recomputes the policy from the start of the episode (a
segment starting mid-episode re-scores from a burn-in prefix only). Every other trained decision uses the importance
ratio π/μ from the recorded behaviour log-probability, also at lag 0.

**Validated max lag.** ``LagRouter(max_lag=None)`` uses the largest lag that a lag sweep (`dist/lagsweep.py`:
learning per update with an injected fixed policy lag, V-trace vs. uncorrected) showed to still learn within
tolerance of on-policy training; the sweep result is stored in ``dist/lag_validation.json``. The validation belongs to
the model and task family it was run on (the toy network here): a new learner model must re-run the sweep.

`vtrace` is the duration-aware V-trace of Espeholt et al. (2018) — copy-adapted from the tested
``superlearner/unified/learning/returns.py`` — with the semi-Markov discount γ_t = exp(−κ·Δt_t) (Δt in simulated
days, `semi_markov_discount`). Workstream D's ``learn/vtrace.py`` is the CAM objective's implementation; this one
serves the distributed stack and its benchmarks, and the two must agree (`vtrace` reduces to the Monte Carlo
return at zero log-ratio and λ = 1, tested). Steps whose log-ratio is `CUT_LOG_RHO` get ρ = c = 0: their value
target is their own value estimate, which is how a non-trained step at the end of a segment serves as the
bootstrap value of the steps before it.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

CUT_LOG_RHO = -1.0e4                      # exp() == 0: ρ = c = 0 (bootstrap / non-trained step)
VALIDATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lag_validation.json")


def validated_max_lag(path: str = VALIDATION_FILE) -> tuple:
    """-> (max_lag, provenance dict) from the stored lag sweep. Raises FileNotFoundError if no sweep was stored."""
    with open(path) as fh:
        d = json.load(fh)
    return int(d["validated_max_lag"]), {"source": os.path.basename(path), "criterion": d.get("criterion"),
                                         "model": d.get("model"), "commit_note": d.get("note")}


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
    on_policy: list = field(default_factory=list)      # items whose trained decisions all have lag 0
    off_policy: list = field(default_factory=list)     # items with at least one trained decision at lag > 0
    discarded: list = field(default_factory=list)      # items with no trainable decision
    lag_hist: dict = field(default_factory=dict)       # lag -> decisions routed (trained or not)
    reasons: dict = field(default_factory=dict)        # reason a decision was not trained -> count
    n_on: int = 0                                      # trained decisions at lag 0
    n_off: int = 0                                     # trained decisions at lag > 0 (V-trace corrected)
    n_dropped: int = 0                                 # routed decisions not trained (any reason)
    n_exact: int = 0                                   # trained decisions treated as exactly on-policy

    def summary(self) -> dict:
        lags = sorted(l for l, n in self.lag_hist.items() for _ in range(n))
        q = (lambda p: lags[min(len(lags) - 1, int(p * len(lags)))] if lags else None)
        return {"learner_version": self.learner_version, "n_on_policy": self.n_on, "n_off_policy": self.n_off,
                "n_discarded": self.n_dropped, "n_exact_on_policy": self.n_exact,
                "n_traj_on_policy": len(self.on_policy), "n_traj_off_policy": len(self.off_policy),
                "n_traj_discarded": len(self.discarded),
                "lag_mean": (sum(lags) / len(lags)) if lags else None, "lag_p50": q(0.5), "lag_p90": q(0.9),
                "lag_max": lags[-1] if lags else None, "lag_hist": {str(k): v for k, v in sorted(self.lag_hist.items())},
                "discard_reasons": dict(self.reasons)}


class LagRouter:
    """Routes decisions by policy lag. ``mode="vtrace"`` trains 0 <= lag <= max_lag (lag > 0 V-trace corrected);
    ``mode="discard"`` trains only lag 0 (strictly on-policy training). ``max_lag=None`` loads the validated value
    (`validated_max_lag`)."""

    def __init__(self, max_lag: Optional[int] = None, mode: str = "vtrace", allow_deterministic: bool = True):
        if mode not in ("vtrace", "discard"):
            raise ValueError(mode)
        if max_lag is None:
            self.max_lag, self.max_lag_source = validated_max_lag()
        else:
            self.max_lag, self.max_lag_source = int(max_lag), {"source": "explicit"}
        self.mode = mode
        self.allow_deterministic = allow_deterministic
        self.total = Counter()
        self.total_hist = Counter()

    def _why_not(self, lag: int, logp, deterministic: bool) -> Optional[str]:
        if lag < 0:
            return "future_version"
        if logp is None:
            return "no_behavior_logp"
        if deterministic and not self.allow_deterministic:
            return "deterministic_behavior"
        if lag == 0:
            return None
        if self.mode == "vtrace":
            return None if lag <= self.max_lag else "lag_over_max"
        return "stale_in_discard_mode"

    def route(self, records: list, learner_version: int) -> LagReport:
        """Route individual decision EventRecords (records without an action — delayed outcomes, world events —
        follow their decision and are not routed here). ``on_policy``/``off_policy``/``discarded`` hold records."""
        rep = LagReport(learner_version)
        hist, reasons = Counter(), Counter()
        for r in records:
            if r.action is None:
                continue
            lag = learner_version - int(r.model_version)
            hist[lag] += 1
            why = self._why_not(lag, r.behavior_logp, bool((r.extra or {}).get("deterministic")))
            if why is not None:
                reasons[why] += 1
                rep.discarded.append(r)
                rep.n_dropped += 1
            elif lag == 0:
                rep.on_policy.append(r)
                rep.n_on += 1
            else:
                rep.off_policy.append(r)
                rep.n_off += 1
        return self._account(rep, hist, reasons)

    def route_trajectories(self, trajs: list, learner_version: int) -> LagReport:
        """Route the decisions of trajectories (objects with ``events``, ``train_mask`` — decisions the assembler
        offers for training; context/bootstrap decisions are False — and ``from_start``). Sets on each trajectory:
        ``lags`` [T] per-decision lag, ``train_mask`` [T] (narrowed to trainable decisions), ``exact`` [T] (exactly
        on-policy, see the module docstring), ``dropped`` [(event_id, reason)], ``lag`` (max lag over trained
        decisions) and ``route``. Counts are in decisions."""
        rep = LagReport(learner_version)
        hist, reasons = Counter(), Counter()
        for tr in trajs:
            evs = tr.events
            lags = [learner_version - int(e.model_version) for e in evs]
            mask = list(tr.train_mask)
            exact = [False] * len(evs)
            dropped = []
            v_first = int(evs[0].model_version) if evs else 0
            uniform = all(int(e.model_version) == v_first for e in evs)
            for i, e in enumerate(evs):
                if not mask[i]:
                    continue
                hist[lags[i]] += 1
                det = bool((e.extra or {}).get("deterministic"))
                why = self._why_not(lags[i], e.behavior_logp, det)
                if why is not None:
                    mask[i] = False
                    reasons[why] += 1
                    dropped.append((e.event_id, why))
                    rep.n_dropped += 1
                    continue
                if lags[i] == 0:
                    rep.n_on += 1
                    sv = (e.extra or {}).get("state_version")
                    consistent = (int(sv) == int(e.model_version)) if sv is not None else uniform
                    if consistent and not det and getattr(tr, "from_start", False):
                        exact[i] = True
                        rep.n_exact += 1
                else:
                    rep.n_off += 1
            tr.lags, tr.train_mask, tr.exact, tr.dropped = lags, mask, exact, dropped
            trained_lags = [l for l, m in zip(lags, mask) if m]
            tr.lag = max(trained_lags) if trained_lags else (max(lags) if lags else 0)
            if not trained_lags:
                tr.route = "discarded"
                rep.discarded.append(tr)
            elif tr.lag == 0:
                tr.route = "on_policy"
                rep.on_policy.append(tr)
            else:
                tr.route = "off_policy"
                rep.off_policy.append(tr)
        return self._account(rep, hist, reasons)

    def _account(self, rep: LagReport, hist: Counter, reasons: Counter) -> LagReport:
        rep.lag_hist = dict(hist)
        rep.reasons = dict(reasons)
        self.total.update({"on_policy": rep.n_on, "off_policy": rep.n_off, "discarded": rep.n_dropped,
                           "exact_on_policy": rep.n_exact})
        self.total.update({f"discard:{k}": v for k, v in reasons.items()})
        self.total_hist.update(hist)
        return rep

    def totals(self) -> dict:
        return {"counts": {k: v for k, v in self.total.items() if v}, "max_lag": self.max_lag,
                "max_lag_source": self.max_lag_source, "mode": self.mode,
                "lag_hist": {str(k): v for k, v in sorted(self.total_hist.items())}}


def importance_log_ratio(target_logp: Tensor, behavior_logp: Tensor, lag: Optional[Tensor] = None,
                         on_policy_exact: bool = True, exact: Optional[Tensor] = None,
                         train: Optional[Tensor] = None) -> Tensor:
    """log ρ = log π − log μ, [B, T].

    ``exact`` [B, T] bool marks decisions that are exactly on-policy (log ρ = 0); without it, ``on_policy_exact``
    treats every ``lag == 0`` row as exact (only correct when the behaviour latent state and policy match the
    learner's recomputation, see the module docstring). ``train`` [B, T] bool: rows that are not trained get
    `CUT_LOG_RHO` (ρ = c = 0: bootstrap/context steps)."""
    lr = target_logp - behavior_logp
    if exact is not None:
        lr = torch.where(exact, torch.zeros_like(lr), lr)
    elif on_policy_exact and lag is not None:
        lr = torch.where(lag == 0, torch.zeros_like(lr), lr)
    if train is not None:
        lr = torch.where(train, lr, torch.full_like(lr, CUT_LOG_RHO))
    return lr

def lag_summary(lags: list) -> Optional[dict]:
    if not lags:
        return None
    s = sorted(lags)
    return {"mean": sum(s) / len(s), "p50": s[len(s) // 2], "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
            "max": s[-1]}
