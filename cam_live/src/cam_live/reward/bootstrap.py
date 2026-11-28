"""cam_live.reward.bootstrap — the cold-start learning signal and its hand-over to the learned reward (spec 7.6).

Everything produced here is labelled ``bootstrap``. It is a fixed, hand-written stabiliser used while R_psi has not
yet earned trust; it is never described or logged as meta-learned.

    learning progress    improvement of the prediction loss on the *same* targets (loss_before - loss_after after
                         a model update). Raw prediction error is never rewarded.
    grounded delta V     change of the evaluator's success estimate (the evaluator is trained on verified outcomes)
    exploration          relative reduction of *epistemic* uncertainty, closed by the noisy-TV gate once that
                         uncertainty stops falling; aleatoric noise earns nothing
    verified outcomes    sparse verifier measurements that were already observed
    information value    (ST-3) the realised information value of the decision's result: the world model's loss on the
                         next decisions' matured outcomes without minus with the decision's context items (agent
                         annotation; measured once those outcomes matured, so present in inner updates, not at
                         reward time)

``RewardMixer`` computes r = mix * r_bootstrap + (1 - mix) * r_learned with mix starting at 1.0 (the inner update
standardises both before mixing, ``reward.inner``). The mix is lowered only when held-out verified query evidence
shows the learned reward is not worse than the bootstrap by more than a margin (non-inferiority), raised when it
shows harm, and held when the evidence is too thin or inconclusive. Every use and every promotion decision is logged
with what each component produced.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

from .integrity import NoisyTVGate, bound_reward

BOOTSTRAP_LABEL = "bootstrap"
LEARNED_LABEL = "learned"
MIXED_LABEL = "mixed"
BOOTSTRAP_VERSION = "bootstrap-2"          # 2: + realised information value (ST-3)

__all__ = ["BOOTSTRAP_LABEL", "LEARNED_LABEL", "MIXED_LABEL", "BOOTSTRAP_VERSION", "learning_progress",
           "LearningProgressTracker", "BootstrapWeights", "BootstrapOutput", "BootstrapReward", "MixedReward",
           "PromotionDecision", "RewardMixer", "paired_policy_value_diffs"]


# ========================================================================================== learning progress
def learning_progress(loss_before: Tensor, loss_after: Tensor) -> Tensor:
    """Per-target improvement of the prediction loss on the same targets: positive = the model got better at
    predicting them. A target that stays unpredictable (pure noise) has zero progress however large its error."""
    if loss_before.shape != loss_after.shape:
        raise ValueError("learning progress needs losses on the same targets")
    return loss_before - loss_after


class LearningProgressTracker:
    """Pairs the loss of the model *before* an update with its loss on the same target set *after* it."""

    def __init__(self, max_pending: int = 4096):
        self._before: "collections.OrderedDict[str, float]" = collections.OrderedDict()
        self.max_pending = max_pending

    def record_before(self, key: str, loss: float) -> None:
        self._before[key] = float(loss)
        self._before.move_to_end(key)
        while len(self._before) > self.max_pending:
            self._before.popitem(last=False)

    def progress(self, key: str, loss_after: float) -> Optional[float]:
        """loss_before - loss_after for target set ``key``; None when no earlier loss on those targets exists."""
        before = self._before.pop(key, None)
        return None if before is None else before - float(loss_after)

    def state_dict(self) -> Dict[str, Any]:
        return {"before": list(self._before.items())}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self._before = collections.OrderedDict((k, float(v)) for k, v in d["before"])


# ============================================================================================ bootstrap reward
@dataclass
class BootstrapWeights:
    learning_progress: float = 1.0
    delta_v: float = 1.0
    exploration: float = 0.5
    verified: float = 2.0
    information_value: float = 1.0    # ST-3: tanh of the realised information value (nats), on the scale of learning
                                      # progress (also tanh of a loss difference in nats)


@dataclass
class BootstrapOutput:
    total: Tensor                                  # [B] bounded
    components: Dict[str, Tensor]                  # name -> [B] weighted contribution before the bound
    label: str = BOOTSTRAP_LABEL
    version: str = BOOTSTRAP_VERSION


class BootstrapReward:
    def __init__(self, weights: Optional[BootstrapWeights] = None, bound: float = 1.0,
                 gate: Optional[NoisyTVGate] = None):
        self.w = weights or BootstrapWeights()
        self.bound = float(bound)
        self.gate = gate or NoisyTVGate()

    def __call__(self, *, learning_progress: Optional[Tensor] = None, delta_v: Optional[Tensor] = None,
                 epistemic_before: Optional[Tensor] = None, epistemic_after: Optional[Tensor] = None,
                 verified_delta: Optional[Tensor] = None, verified_mask: Optional[Tensor] = None,
                 context_keys: Optional[Sequence[str]] = None, gate_update: bool = True,
                 information_value: Optional[Tensor] = None) -> BootstrapOutput:
        """All inputs [B] (None = not available, contributes 0). ``context_keys`` names the context of each row
        for the noisy-TV gate (rows without a key are not gated). ``gate_update=False`` reads the gate without
        advancing it (replay of a step the gate has already seen; the gate moves once per real step)."""
        ref = next(t for t in (learning_progress, delta_v, epistemic_before, verified_delta, information_value)
                   if t is not None)
        zeros = torch.zeros_like(ref, dtype=ref.dtype)
        comp: Dict[str, Tensor] = {}
        comp["learning_progress"] = self.w.learning_progress * (
            zeros if learning_progress is None else torch.tanh(learning_progress.to(ref.dtype)))
        comp["delta_v"] = self.w.delta_v * (zeros if delta_v is None else delta_v.to(ref.dtype))
        # ST-3: what an action's result changed in CAM's later predictions (realised information value, nats): the
        # world model's loss on the next decisions' matured outcomes without minus with the decision's context items
        comp["information_value"] = self.w.information_value * (
            zeros if information_value is None else torch.tanh(information_value.to(ref.dtype)))
        if epistemic_before is None or epistemic_after is None:
            comp["exploration"] = zeros
        else:
            eb, ea = epistemic_before.to(ref.dtype), epistemic_after.to(ref.dtype)
            drop = ((eb - ea).clamp_min(0.0) / eb.clamp_min(1e-12)).clamp(0.0, 1.0)      # epistemic only
            if context_keys is not None:
                g = torch.tensor([1.0 if k is None else (self.gate.update(k, float(e)) if gate_update
                                                         else self.gate.gate(k))
                                  for k, e in zip(context_keys, ea.tolist())], dtype=ref.dtype, device=ref.device)
                drop = drop * g
            comp["exploration"] = self.w.exploration * drop
        if verified_delta is None:
            comp["verified"] = zeros
        else:
            v = verified_delta.to(ref.dtype)
            v = torch.sign(v) * torch.log1p(v.abs())
            comp["verified"] = self.w.verified * (v if verified_mask is None else v * verified_mask.to(ref.dtype))
        total = bound_reward(sum(comp.values()), self.bound)
        return BootstrapOutput(total=total, components=comp)


# ====================================================================================================== mixer
@dataclass
class MixedReward:
    total: Tensor                 # [B] the reward the learner uses
    bootstrap: Tensor             # [B] raw bootstrap reward
    learned: Tensor               # [B] raw learned reward (detached: mixing is not a meta-gradient path)
    mix: float                    # weight of the bootstrap reward
    label: str                    # "bootstrap" (mix == 1), "learned" (mix == 0) or "mixed"
    reward_version: str
    bootstrap_version: str = BOOTSTRAP_VERSION

    def record_fields(self, i: int) -> Dict[str, Any]:
        """Fields of ``contracts.ExperienceRecord`` for row i."""
        return {"intrinsic_reward": float(self.total[i]), "bootstrap_reward": float(self.bootstrap[i]),
                "reward_version": f"{self.label}:mix={self.mix:.3f}:{self.reward_version}"}


@dataclass
class PromotionDecision:
    action: str                   # "lower" | "raise" | "hold"
    reason: str
    mix_before: float
    mix_after: float
    n_independent: int
    mean_diff: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    margin: float = 0.0


# one-sided 95 % Student-t quantiles for df = 1..30
_T95 = (6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771, 1.761, 1.753,
        1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697)


def _t95(df: int) -> float:
    return _T95[df - 1] if df <= 30 else 1.645 + 1.56 / df


class RewardMixer:
    """r = mix * r_bootstrap + (1 - mix) * r_learned, with an evidence-gated promotion rule."""

    def __init__(self, mix: float = 1.0, step: float = 0.1, min_mix: float = 0.0, min_independent: int = 8,
                 tolerance: float = 0.0, log_size: int = 10_000, rel_margin: float = 0.0):
        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        self.mix, self.step, self.min_mix = float(mix), float(step), float(min_mix)
        self.min_independent, self.tolerance = int(min_independent), float(tolerance)
        self.rel_margin = float(rel_margin)
        self.log: Deque[Dict[str, Any]] = collections.deque(maxlen=log_size)
        self.decisions: List[PromotionDecision] = []
        self.meta_updates_applied = 0     # outer updates of psi that were applied (MetaLearner.meta_update)
        self.promotions = 0               # "lower" decisions taken after at least one applied meta-update

    @property
    def promoted(self) -> bool:
        """R_psi has been meta-updated at least once *and* promoted on held-out verified evidence since."""
        return self.meta_updates_applied > 0 and self.promotions > 0

    @property
    def label(self) -> str:
        """``bootstrap`` until R_psi has been meta-updated and promoted (spec 7.6: a randomly initialised or merely
        configured reward is not described as learned — ``mix`` is reported separately in every record); then
        ``learned`` at mix 0 and ``mixed`` in between."""
        if self.mix >= 1.0 or not self.promoted:
            return BOOTSTRAP_LABEL
        return LEARNED_LABEL if self.mix <= 0.0 else MIXED_LABEL

    def note_meta_update(self) -> None:
        """Called by the meta-learner after an outer update of psi was applied."""
        self.meta_updates_applied += 1

    def combine(self, r_bootstrap: Tensor, r_learned: Tensor, *, reward_version: str,
                bootstrap_components: Optional[Mapping[str, Tensor]] = None,
                exp_ids: Optional[Sequence[str]] = None) -> MixedReward:
        rl = r_learned.detach().to(r_bootstrap.dtype)
        rb = r_bootstrap.detach()
        total = self.mix * rb + (1.0 - self.mix) * rl
        out = MixedReward(total=total, bootstrap=rb, learned=rl, mix=self.mix, label=self.label,
                          reward_version=reward_version)
        entry: Dict[str, Any] = {
            "label": out.label, "mix": self.mix, "reward_version": reward_version,
            "bootstrap_version": BOOTSTRAP_VERSION, "n": int(total.numel()),
            "from_bootstrap": (self.mix * rb).tolist(), "from_learned": ((1.0 - self.mix) * rl).tolist(),
            "total": total.tolist(), "exp_ids": list(exp_ids) if exp_ids is not None else []}
        if bootstrap_components is not None:
            entry["bootstrap_components"] = {k: v.detach().tolist() for k, v in bootstrap_components.items()}
        self.log.append(entry)
        return out

    def update(self, diffs: Sequence[float], *, scale: float = 0.0) -> PromotionDecision:
        """One promotion step from held-out verified query evidence. ``diffs[g]`` is, for independent outcome
        group g that was held out of the meta-update, (verified utility attributed to the policy updated with the
        learned reward) - (the same for the policy updated with the bootstrap reward); see
        :func:`paired_policy_value_diffs`. One number per *independent* group, never per row. ``scale`` is the
        typical size of a verified utility (e.g. the RMS of the held-out utilities): the non-inferiority margin is
        ``delta = tolerance + rel_margin * scale``.

        lower  : enough groups and the one-sided 95 % lower bound of the mean difference >= -delta (the learned
                 reward is shown not to be worse by more than the margin: promotion)
        raise  : enough groups and the one-sided 95 % upper bound < 0 (the learned reward is shown to hurt: revert)
        hold   : fewer than ``min_independent`` groups, or the evidence is inconclusive — change nothing"""
        d = [float(x) for x in diffs]
        n = len(d)
        before = self.mix
        delta = self.tolerance + self.rel_margin * max(float(scale), 0.0)
        if n < max(2, self.min_independent) or any(not math.isfinite(x) for x in d):
            dec = PromotionDecision("hold", f"insufficient_independent_evidence:{n}<{max(2, self.min_independent)}"
                                    if n < max(2, self.min_independent) else "non_finite_evidence",
                                    before, before, n)
        else:
            mean = sum(d) / n
            se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n)
            t = _t95(n - 1)
            lo, hi = mean - t * se, mean + t * se
            if lo >= -delta:
                self.mix = max(self.min_mix, self.mix - self.step)
                dec = PromotionDecision("lower", "held_out_evidence_shows_no_harm", before, self.mix, n, mean, lo, hi)
                if self.meta_updates_applied > 0:
                    self.promotions += 1
            elif hi < 0.0:
                self.mix = min(1.0, self.mix + self.step)
                dec = PromotionDecision("raise", "held_out_evidence_shows_harm", before, self.mix, n, mean, lo, hi)
            else:
                dec = PromotionDecision("hold", "inconclusive_evidence", before, before, n, mean, lo, hi)
        dec.margin = delta
        self.decisions.append(dec)
        self.log.append({"label": "promotion", "action": dec.action, "reason": dec.reason,
                         "mix_before": dec.mix_before, "mix_after": dec.mix_after, "n_independent": n,
                         "mean_diff": dec.mean_diff, "lower_bound": dec.lower_bound, "upper_bound": dec.upper_bound,
                         "margin": delta})
        return dec

    def apply(self, dec: Any) -> PromotionDecision:
        """Apply a decision of the prospective validity evidence (``reward.validity.ValidityDecision``): set the mix
        it names, record and log it like :meth:`update`. A "lower" after at least one applied meta-update counts as
        a promotion of the learned reward (label ``mixed`` / ``learned``)."""
        before = self.mix
        action, reason = dec.action, dec.reason
        if action in ("lower", "raise") and self.step <= 0.0:          # a fixed mix (step 0) never moves
            action, reason = "hold", f"fixed_mix:{dec.action}"
        after = max(self.min_mix, min(1.0, float(dec.mix_after))) if action in ("lower", "raise") else before
        self.mix = after
        out = PromotionDecision(action, reason, before, after, int(dec.n_pairs), float(dec.mean),
                                float(dec.lower), float(dec.upper), 0.0)
        if action == "lower" and self.meta_updates_applied > 0:
            self.promotions += 1
        if action != "hold" or not self.decisions or self.decisions[-1].action != "hold":
            self.decisions.append(out)
        else:                                          # consecutive holds: keep the latest, not one per pair
            self.decisions[-1] = out
        if len(self.decisions) > 4096:
            del self.decisions[:len(self.decisions) - 4096]
        self.log.append({"label": "promotion", "rule": "validity", "action": out.action, "reason": out.reason,
                         "mix_before": before, "mix_after": after, "n_pairs": int(dec.n_pairs),
                         "n_informative": int(dec.n_informative), "comparison": dec.comparison,
                         "mean_diff": out.mean_diff, "lower_bound": out.lower_bound, "upper_bound": out.upper_bound})
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"mix": self.mix, "step": self.step, "min_mix": self.min_mix, "rel_margin": self.rel_margin,
                "min_independent": self.min_independent, "tolerance": self.tolerance,
                "meta_updates_applied": self.meta_updates_applied, "promotions": self.promotions}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self.mix, self.step, self.min_mix = float(d["mix"]), float(d["step"]), float(d["min_mix"])
        self.min_independent, self.tolerance = int(d["min_independent"]), float(d["tolerance"])
        self.rel_margin = float(d.get("rel_margin", 0.0))
        self.meta_updates_applied = int(d.get("meta_updates_applied", 0))
        self.promotions = int(d.get("promotions", 0))


def paired_policy_value_diffs(logp_learned: Tensor, logp_bootstrap: Tensor, behaviour_logp: Tensor,
                              utility: Tensor, groups: Tensor, weight_clip: float = 10.0) -> List[float]:
    """Per independent group: clipped importance-weighted verified utility of the learned-reward policy minus that
    of the bootstrap-reward policy, on held-out query rows logged with propensities. Input of
    :meth:`RewardMixer.update`. Rows of one group are averaged, so a group counts once."""
    rho_l = torch.exp(logp_learned.detach() - behaviour_logp).clamp(max=weight_clip)
    rho_b = torch.exp(logp_bootstrap.detach() - behaviour_logp).clamp(max=weight_clip)
    d = (rho_l - rho_b) * utility.detach().to(rho_l.dtype)
    _, inv = torch.unique(groups, return_inverse=True)
    n = int(inv.max()) + 1
    s = torch.zeros(n, dtype=d.dtype).index_add(0, inv, d)
    c = torch.zeros(n, dtype=d.dtype).index_add(0, inv, torch.ones_like(d))
    return (s / c).tolist()
