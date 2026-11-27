"""cam_live.reward.metalearner — meta-learning of R_psi on the real model, inside one lifetime (spec 7.3 - 7.6).

    reward parameters psi -> inner neural update -> changed policy -> later verified outcomes -> outer update of psi

``MetaLearner`` owns the reward optimizer and is the only code that changes the ``reward`` parameter group.

* **Real policy updates.** :meth:`policy_update` applies theta <- theta - alpha * grad L_inner(theta; psi, D_support)
  to the live adaptable subset (policy head + decision trunk) with ``reward.inner.PolicyImprovementLoss`` — the very
  function the unroll differentiates — under a ``PsiGuard``. Before the update it snapshots theta and records the
  support batch (ids, cached latents, reward-input statistics) and the inner hyper-parameters as an *update event*.
  A learner that steps theta with its own optimizer calls :meth:`note_policy_update` first instead.
* **Windows.** Every decision is registered once (:meth:`internal_rewards`); decisions taken after an update event
  are its query rows. ``reward.windows.WindowBuilder`` enforces temporal separation, eligibility, propensities and
  per-group independence; rows whose own verified measurements are still pending are not used.
* **Verified utility only.** Utilities come from the ``integrity.OutcomeLedger``: signed objective outcomes
  (:meth:`admit_objective`) and authenticated verifier measurements (:meth:`admit_measurements`). The number is
  derived there from the signed record (``integrity.objective_utility`` / ``measurement_utility``); a caller-stated
  utility that disagrees is refused and logged. Only objectives with a recorded operator origin supervise
  (:meth:`register_objective`, from ``TaskSpec.origin``; unknown origin is excluded). Unsigned, fabricated,
  duplicated, imagined or self-authored outcomes are rejected there and never reach the objective.
* **Meta-update** (:meth:`meta_update`). With at least ``min_independent`` independent verified outcomes it
  recomputes theta_K(psi) functionally from each event's snapshot on the cached latents, scores the logged query
  decisions under pi_{theta_K} — on-policy only when the logged policy *is* the recomputed one (the update was
  applied by :meth:`policy_update` with the same bootstrap mix and the current psi, consecutive events chain
  exactly, and the logged propensities agree within ``on_policy_tol``), otherwise clipped importance weighting with
  the support checks of ``reward.meta`` — backpropagates to psi, applies one clipped step of the reward optimizer,
  feeds held-out groups to the promotion rule of the bootstrap mix, and returns a ``contracts.MetaUpdateRecord``.
  A skipped update returns a record too, with the reason, and leaves psi bit-identical.
* **Baseline.** The baseline of a query row never depends on that row's action or outcome: it is the mean
  utility of rows outside the row's *baseline group* — its own independent-outcome group with ``utility="direct"``,
  its whole return-to-go chain (lifetime and held-out partition) with ``utility="return_to_go"``, because the
  return-to-go of every earlier row of the chain contains this row's measured outcome — and, when no such row
  exists, the running mean of verified utility as it was when the decision was registered (prefix-only).
* **Cold start.** While the mix is above 0 the live policy is (partly) driven by the labelled bootstrap reward. The
  differentiated update uses ``meta_mix`` (default 0: the pure learned reward), i.e. it is a *shadow* update from the
  same snapshot, evaluated by importance weighting on what the live policy actually did. The mix moves only through
  ``RewardMixer.update`` on held-out verified evidence comparing the learned-reward update with the bootstrap one.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor, nn

from ..contracts import (Clock, ExperienceRecord, ExperienceStore, LearningUpdateRecord, MetaUpdateRecord,
                         OutcomeObservation, OutcomeQuery, OutcomeStatus, Trust, VerifiedOutcome, new_id)
from ..learning_api import matured_outcomes
from ..nn.api import GROUP_HEADS, GROUP_REWARD
from .assemble import SignalAssembler
from .bootstrap import BootstrapReward, RewardMixer, paired_policy_value_diffs
from .inner import (POLICY_PREFIX, TRUNK_PREFIX, GroupClip, InnerBatch, InnerConfig, PolicyImprovementLoss,
                    verified_progress)
from .integrity import (SUPERVISING_ORIGINS, AdmitResult, IntegrityError, OutcomeLedger, PsiGuard,
                        assert_optimizers_disjoint, assert_reward_optimizer_exclusive, params_hash, screen_verified)
from .intrinsic import IntrinsicReward
from .meta import ESTIMATOR_IMPORTANCE, ESTIMATOR_ON_POLICY, MetaGradient, MetaGradResult, Params, QueryData
from .windows import WindowBuilder, held_out_groups, independence_key

__all__ = ["MetaConfig", "UpdateEvent", "MetaLearner"]


@dataclass
class MetaConfig:
    inner_lr: float = 1e-2               # alpha
    inner_steps: int = 2                 # K: inner steps per policy update
    lr_reward: float = 1e-4              # beta (Adam)
    clip_reward: float = 0.5             # norm clip of the meta-gradient
    clip_policy: float = 1.0             # in-graph norm clip of the inner gradient, policy head
    clip_trunk: float = 1.0              # ... decision trunk
    min_independent: int = 8             # independent verified outcomes needed for one meta-update
    mode: str = "unroll"                 # "unroll" | "checkpoint" | "first_order"
    unroll_events: int = 1               # consecutive update events one unroll covers (K = their steps together)
    on_policy_tol: float = 1e-3          # consistency check (max |pi_thetaK(a) - logged propensity|) on top of the
                                         # structural condition for the on-policy estimator (see module doc)
    min_meta_grad_norm: float = 0.0      # an update is applied only if |dJ/dpsi| is finite and strictly above this
    weight_clip: float = 10.0
    min_ess: float = 2.0
    min_ess_fraction: float = 0.05
    min_propensity: float = 1e-6
    heldout_fraction: float = 0.25       # independent groups reserved for promotion evidence
    meta_mix: Optional[float] = 0.0      # bootstrap weight inside the differentiated update (None: the live mix)
    utility: str = "return_to_go"        # "return_to_go" | "direct" credit of verified measurements
    max_events: int = 32
    provenance: Optional[str] = None     # None: provenance of the query rows (never mixed)
    min_gap_s: float = 0.0
    mix: float = 1.0                     # initial bootstrap mix
    mixer_step: float = 0.25             # mix change per promotion decision (spec 7.6 "gradually ... promote")
    mixer_min_independent: int = 4       # held-out independent groups per promotion decision (Student-t bound)
    mixer_tolerance: float = 0.0
    mixer_rel_margin: float = 0.05       # non-inferiority margin, in units of the held-out verified utility RMS
    reward_hidden: int = 256
    reward_life: int = 64
    reward_bound: float = 1.0
    policy_hidden: int = 192
    target_kl: float = 0.05              # trust region of the live policy step (0: fixed inner_lr; INT2-7)
    max_inner_lr: float = 2.0            # largest live step the trust region may choose
    replay_matured: int = 16             # earlier decisions with final verified outcomes added to a support batch

    @classmethod
    def from_cfg(cls, cfg: Any, **over: Any) -> "MetaConfig":
        o = cfg.optim
        kw: Dict[str, Any] = dict(inner_lr=o.inner_lr, inner_steps=o.inner_steps, lr_reward=o.lr_reward,
                                  clip_reward=o.clip_reward, clip_policy=o.clip_heads, clip_trunk=o.clip_experiential,
                                  min_independent=o.meta_min_independent_outcomes)
        names = {f.name for f in dataclasses.fields(cls)}
        kw.update({k: v for k, v in cfg.section("reward").get("meta", {}).items() if k in names})
        kw.update(over)
        return cls(**kw)


@dataclass
class UpdateEvent:
    """One real policy update: what is needed to recompute it as a function of psi."""
    event_id: int
    ts: float
    exp_ids: List[str]                   # support batch
    inner_lr: float
    steps: int
    mix: float                           # bootstrap mix in force
    reward_version: str
    lifetime: int
    theta0: Params                       # adaptable subset before the update
    ib: Optional[InnerBatch] = None      # cached support latents and reward inputs (rebuilt when missing)
    query_ids: List[str] = field(default_factory=list)      # decisions taken after this update, in order
    external: bool = False               # the update itself was applied by another optimizer
    theta0_hash: str = ""                # params_hash of theta0 / of the live subset after the update
    theta1_hash: str = ""                # ("" for external updates and events restored from a checkpoint)


@dataclass
class _Episode:
    event_index: int
    events: List[UpdateEvent]
    support_window: str
    query_window: str
    ibq: InnerBatch                      # policy-side latents of the ready query rows
    exp_ids: List[str]
    utility: List[float]
    has_u: List[bool]
    group: List[str]
    held: List[bool]
    outcome_groups: Set[str]
    discarded: List[str]
    indep: List[str] = field(default_factory=list)          # independent-outcome group of each row
    prefix_base: List[float] = field(default_factory=list)  # running mean utility when the decision was registered


def _default_measurement_utility(rec: ExperienceRecord, q: OutcomeQuery, o: OutcomeObservation) -> Optional[float]:
    """Goal-relevant verifier measurements count with their measured value; others are not utility."""
    return float(o.value) if (q.goal_relevant and o.value is not None) else None


class MetaLearner:
    def __init__(self, model: Any, store: ExperienceStore, cfg: Any, *, policy: Optional[nn.Module] = None,
                 reward: Optional[IntrinsicReward] = None,
                 check_signature: Optional[Callable[[VerifiedOutcome], bool]] = None,
                 check_measurement: Optional[Callable[[OutcomeObservation], bool]] = None,
                 ledger: Optional[OutcomeLedger] = None, ledger_path: Optional[str] = None,
                 clock: Optional[Clock] = None, assembler: Optional[SignalAssembler] = None,
                 bootstrap: Optional[BootstrapReward] = None, mixer: Optional[RewardMixer] = None,
                 inner_cfg: Optional[InnerConfig] = None, meta_cfg: Optional[MetaConfig] = None,
                 measurement_utility: Optional[Callable[[ExperienceRecord, OutcomeQuery, OutcomeObservation],
                                                        Optional[float]]] = None,
                 policy_head: str = "predict_policy", reward_head: str = "reward",
                 evaluator_head: str = "predict_evaluator"):
        self.model, self.store, self.cfg, self.clock = model, store, cfg, clock
        mc = self.mc = meta_cfg or MetaConfig.from_cfg(cfg)
        if policy is None:
            try:
                policy = model.head(policy_head)
            except KeyError:
                from ..predict.policy import CandidatePolicy, PolicyConfig
                policy = model.register_head(policy_head, CandidatePolicy(PolicyConfig(
                    d_model=model.d_model, hidden=mc.policy_hidden)), GROUP_HEADS)
        if reward is None:
            try:
                reward = model.head(reward_head)
            except KeyError:
                reward = model.register_head(reward_head, IntrinsicReward(
                    model.d_model, d_hidden=mc.reward_hidden, d_life=mc.reward_life, bound=mc.reward_bound),
                    GROUP_REWARD)
        self.policy, self.reward = policy, reward
        from ..predict.policy import selection_calibration
        self.selection = selection_calibration(policy)          # learned action value of the selection rule
        self._sel_pending: List[str] = []
        self._reg_at: Dict[str, int] = {}                          # exp_id -> registration count when registered
        self._n_registered = 0
        if assembler is None:
            try:                                             # B's evaluator, when it is registered and context-free
                ev = model.head(evaluator_head)
                assembler = SignalAssembler(evaluator=ev if getattr(getattr(ev, "cfg", None), "d_ctx", 1) == 0 else None)
            except KeyError:
                assembler = SignalAssembler()
        self.mixer = mixer or RewardMixer(mix=mc.mix, step=mc.mixer_step, min_independent=mc.mixer_min_independent,
                                          tolerance=mc.mixer_tolerance, rel_margin=mc.mixer_rel_margin)
        self._life = 0
        self._life_of: Dict[str, int] = {}
        self._life_states: Dict[str, Tensor] = {}
        self._by_objective: Dict[str, List[str]] = {}
        self._objective_seen: Dict[str, Tuple[int, float]] = {}       # objective -> (rows credited, last value)
        self.inner = PolicyImprovementLoss(model, policy, reward, mixer=self.mixer, bootstrap=bootstrap,
                                           assembler=assembler, cfg=inner_cfg or InnerConfig.from_cfg(cfg),
                                           lifetime_of=lambda i: self._life_of.get(i, 0),
                                           life_states=self._life_states)
        self.subset = self.inner.subset
        self.engine = MetaGradient(self.inner.inner_loss, self.inner.logp, inner_lr=mc.inner_lr, K=mc.inner_steps,
                                   mode=mc.mode, weight_clip=mc.weight_clip, min_ess=mc.min_ess,
                                   min_ess_fraction=mc.min_ess_fraction, min_propensity=mc.min_propensity,
                                   min_independent=1,
                                   grad_transform=GroupClip({POLICY_PREFIX: mc.clip_policy, TRUNK_PREFIX: mc.clip_trunk}))
        self.optimizer = torch.optim.Adam(reward.meta_parameters(), lr=mc.lr_reward)
        assert_reward_optimizer_exclusive(self.optimizer, reward)
        if any(id(p) in {id(q) for q in reward.parameters()} for p in self.subset.live().values()):
            raise IntegrityError("the adaptable subset overlaps the reward parameters")
        self.ledger = ledger or OutcomeLedger(check_signature or (lambda v: False), check_measurement,
                                              path=ledger_path)
        self.measurement_utility = measurement_utility or _default_measurement_utility
        self.events: List[UpdateEvent] = []
        self._n_events = 0
        self._used: Set[str] = set()
        self._tried: Set[str] = set()
        self._promo: List[float] = []
        self._promo_scale: List[float] = []
        self._u_ema: Optional[float] = None
        self._base_at: Dict[str, float] = {}                         # exp_id -> prefix baseline at registration
        self.records: List[MetaUpdateRecord] = []
        self.stats: Dict[str, float] = {"evicted_query_rows": 0.0, "discarded_rows": 0.0, "policy_updates": 0.0,
                                        "meta_applied": 0.0, "meta_skipped": 0.0}

    # ------------------------------------------------------------------------------------------- bookkeeping
    def _now(self, fallback: float = 0.0) -> float:
        return float(self.clock.now()) if self.clock is not None else float(fallback)

    def check_learner_optimizers(self, optimizers: Sequence[torch.optim.Optimizer]) -> None:
        """Call once at start-up with every optimizer of the continual learner: none may hold a reward parameter."""
        assert_optimizers_disjoint(optimizers, self.reward)

    def new_lifetime(self) -> None:
        """A new independent lifetime starts: reset the lifetime context of R_psi (the only reset there is)."""
        self._life += 1
        self.reward.lifetime.new_lifetime()

    def observe(self, records: Sequence[ExperienceRecord]) -> List[ExperienceRecord]:
        """Register executed decisions once, in order. Returns the ones that were new."""
        new = []
        for rec in records:
            if rec.exp_id in self._life_of:
                continue
            self._life_of[rec.exp_id] = self._life
            self._base_at[rec.exp_id] = self._u_ema if self._u_ema is not None else 0.0
            self._by_objective.setdefault(rec.objective_id, []).append(rec.exp_id)
            if self.events:
                self.events[-1].query_ids.append(rec.exp_id)
            self._sel_pending.append(rec.exp_id)
            self._reg_at[rec.exp_id] = self._n_registered
            self._n_registered += 1
            new.append(rec)
        if len(self._sel_pending) > 4096:
            for i in self._sel_pending[:len(self._sel_pending) - 4096]:
                self._reg_at.pop(i, None)
            del self._sel_pending[:len(self._sel_pending) - 4096]
        return new

    # ------------------------------------------------------------------------------------------- selection evidence
    @staticmethod
    def verified_progress(rec: ExperienceRecord) -> Optional[float]:
        """``reward.inner.verified_progress``: the final, signed verified progress attributed to one decision."""
        return verified_progress(rec)

    def refresh_selection(self) -> int:
        """Feed the selection rule's calibration (``predict.policy.SelectionCalibration``) with every registered
        decision whose verified progress has become final since the last call. Returns how many were added."""
        if not self._sel_pending:
            return 0
        many = getattr(self.store, "get_many", None)
        recs = many(self._sel_pending) if many is not None else [self.store.get(i) for i in self._sel_pending]
        by_id = {r.exp_id: r for r in recs if r is not None}
        n, keep = 0, []
        for i in self._sel_pending:
            rec = by_id.get(i)
            if rec is None:
                continue
            y = self.verified_progress(rec)
            if y is None:
                # a decision whose goal-relevant answers are all final but that carries no annotation 64 decisions
                # later never will (no goal-relevant question, or no signed reading): stop asking
                if self._resolved(rec) and self._n_registered - self._reg_at.get(i, self._n_registered) > 64:
                    self._reg_at.pop(i, None)
                    continue
                keep.append(i)
                continue
            self._reg_at.pop(i, None)
            n += int(self.selection.observe_decision(i, rec.decision, y))
        self._sel_pending = keep
        return n

    def internal_rewards(self, records: Sequence[ExperienceRecord], *, write: bool = True) -> Dict[str, Any]:
        """Step 13 of the loop: reward every new decision once (learned and bootstrap parts logged separately)."""
        new = self.observe(records)
        self.refresh_selection()
        mixed = self.inner.reward_step(self.store, new, write=write)
        tot = [float(m.total[0]) for m in mixed]
        return {"n": len(new), "mix": self.mixer.mix, "label": self.mixer.label,
                "reward_version": self.reward.version, "rewards": tot,
                "bootstrap": [float(m.bootstrap[0]) for m in mixed], "learned": [float(m.learned[0]) for m in mixed]}

    # ------------------------------------------------------------------------------------------- inner update
    def _push_event(self, theta0: Params, ib: InnerBatch, recs: Sequence[ExperienceRecord], lr: float, steps: int,
                    external: bool) -> UpdateEvent:
        ts = max((r.decision.ts + max(r.duration_s, 0.0) for r in recs), default=0.0)
        ev = UpdateEvent(event_id=self._n_events, ts=self._now(ts), exp_ids=[r.exp_id for r in recs],
                         inner_lr=float(lr), steps=int(steps), mix=float(self.mixer.mix),
                         reward_version=self.reward.version, lifetime=self._life, theta0=theta0, ib=ib,
                         external=external)
        self._n_events += 1
        self.events.append(ev)
        while len(self.events) > self.mc.max_events:
            old = self.events.pop(0)
            lost = [i for i in old.query_ids if i not in self._used]
            self.stats["evicted_query_rows"] += len(lost)
            self._used.update(lost)
        return ev

    def note_policy_update(self, records: Sequence[ExperienceRecord], *, inner_lr: Optional[float] = None,
                           steps: Optional[int] = None) -> Optional[UpdateEvent]:
        """For a learner that applies the policy-improvement loss with its own optimizer: call this *before* its
        step. Snapshots theta and records the support batch; the meta-update then evaluates a shadow SGD update."""
        recs = [r for r in records if self.inner.eligible(r)]
        if not recs:
            return None
        ib = self.inner.build_batch(self.store, recs)
        return self._push_event(self.subset.snapshot(), ib, recs, inner_lr or self.mc.inner_lr,
                                steps or self.mc.inner_steps, external=True)

    def _replay_matured(self, recs: Sequence[ExperienceRecord]) -> List[ExperienceRecord]:
        """Up to ``replay_matured`` earlier registered decisions (newest first) whose verified progress is final and
        that are not in ``recs``: the support batch then carries verified outcomes, which arrive long after the
        decisions that produced them (spec 10.1 replay of delayed outcomes)."""
        n = int(self.mc.replay_matured)
        if n <= 0:
            return []
        have = {r.exp_id for r in recs}
        ids = [i for i in reversed(list(self._life_of)) if i not in have][:max(4 * n, 64)]
        many = getattr(self.store, "get_many", None)
        cand = many(ids) if many is not None else [self.store.get(i) for i in ids]
        by_id = {r.exp_id: r for r in cand if r is not None}
        out = []
        for i in ids:
            r = by_id.get(i)
            if r is not None and self.inner.eligible(r) and verified_progress(r) is not None:
                out.append(r)
                if len(out) >= n:
                    break
        return out

    def _kl(self, theta0: Params, theta: Params, ib: InnerBatch) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            out0, out1 = self.subset.forward(theta0, ib), self.subset.forward(theta, ib)
            kl = (out0.probs * (out0.log_probs - out1.log_probs).masked_fill(~ib.a_mask, 0.0)).sum(-1)
        return kl, out0.probs, out1.probs

    def policy_update(self, records: Sequence[ExperienceRecord], *, steps: Optional[int] = None,
                      inner_lr: Optional[float] = None, trigger: str = "policy_improvement"
                      ) -> Optional[LearningUpdateRecord]:
        """The real inner update of the live policy: K functional SGD steps of L_inner on ``records`` (plus up to
        ``replay_matured`` earlier decisions with final verified outcomes).

        Step size: with ``target_kl > 0`` (and no explicit ``inner_lr``) the step is scaled so that the mean KL
        between the policy before and after the update on the support batch is about ``target_kl`` (a trust region
        in policy space, step in [inner_lr, max_inner_lr]; KL grows ~ step^2 for small steps). The step chosen is
        recorded in the update event, so the meta-gradient differentiates exactly the SGD update that was applied
        (the step is a constant of that function, like any other hyper-parameter)."""
        recs = [r for r in records if self.inner.eligible(r)]
        if not recs:
            return None
        recs = recs + self._replay_matured(recs)
        K, lr = int(steps or self.mc.inner_steps), float(inner_lr or self.mc.inner_lr)
        ib = self.inner.build_batch(self.store, recs)
        theta0 = self.subset.snapshot()
        psi = {k: v.detach() for k, v in self.reward.psi().items()}
        h_reward = params_hash(self.reward)
        tried: List[Tuple[float, float]] = []
        with PsiGuard(self.reward, "policy update"):
            trace = self.engine.inner_update(theta0, psi, [ib], lr, K, create_graph=False)
            theta = trace.detached()
            kl, p0, p1 = self._kl(theta0, theta, ib)
            tried.append((lr, float(kl.mean())))
            if inner_lr is None and self.mc.target_kl > 0:
                tgt = float(self.mc.target_kl)
                lo, hi = float(self.mc.inner_lr), max(float(self.mc.max_inner_lr), float(self.mc.inner_lr))
                below, above = None, None                  # largest step under / smallest step over the target
                for _ in range(6):
                    m = float(kl.mean())
                    if not math.isfinite(m) or m <= 0 or tgt / 1.5 <= m <= 1.5 * tgt:
                        break
                    if m < tgt:
                        below = lr if below is None else max(below, lr)
                    else:
                        above = lr if above is None else min(above, lr)
                    new = math.sqrt(below * above) if (below is not None and above is not None) else \
                        min(hi, max(lo, lr * math.sqrt(tgt / m)))
                    if abs(new - lr) <= 0.02 * lr:
                        break
                    lr = new
                    trace = self.engine.inner_update(theta0, psi, [ib], lr, K, create_graph=False)
                    theta = trace.detached()
                    kl, p0, p1 = self._kl(theta0, theta, ib)
                    tried.append((lr, float(kl.mean())))
            with torch.no_grad():
                _, aux = self.inner.functional(theta0, psi, ib, detach_reward=True, return_aux=True)
            self.subset.write(theta)
        ev = self._push_event(theta0, ib, recs, lr, K, external=False)
        ev.theta0_hash, ev.theta1_hash = params_hash(theta0), params_hash(theta)
        self.stats["policy_updates"] += 1
        delta = {"heads": 0.0, "experiential": 0.0}
        for n in theta:
            g = "heads" if n.startswith(POLICY_PREFIX) else "experiential"
            delta[g] += float(((theta[n] - theta0[n]) ** 2).sum())
        delta = {g: math.sqrt(v) for g, v in delta.items()}
        r = aux["reward"]
        return LearningUpdateRecord(
            update_id=new_id("upd"), ts=ev.ts, trigger=trigger, timescale="frequent",
            losses={self.inner.name: trace.inner_losses[0], f"{self.inner.name}_last": trace.inner_losses[-1]},
            param_groups=[g for g, v in delta.items() if v > 0],
            grad_norms={g: v / (lr * K) for g, v in delta.items()}, weight_delta=delta,
            hash_before={"adaptable": ev.theta0_hash, "reward": h_reward},
            hash_after={"adaptable": ev.theta1_hash, "reward": params_hash(self.reward)},
            n_experiences=len(recs), exp_ids=[r_.exp_id for r_ in recs],
            intrinsic_reward_stats={"mean": float(r.mean()), "std": float(r.std()) if r.numel() > 1 else 0.0,
                                    "min": float(r.min()), "max": float(r.max()), "mix": ib.mix,
                                    "bootstrap_mean": float(ib.r_boot.mean()),
                                    "learned_mean": float(aux["learned"].mean())},
            prediction_change={"policy_prob_mean_abs_change": float((p1 - p0).abs().sum(-1).mean()),
                               "policy_prob_max_abs_change": float((p1 - p0).abs().max()),
                               "policy_kl": float(kl.mean())},
            diagnostics={"step_search": [list(t) for t in tried], "target_kl": float(self.mc.target_kl),
                         "n_replayed_matured": len(recs) - len([r for r in records if self.inner.eligible(r)]),
                         "goal_verified_rows": aux["goal_verified"], "goal_predicted_rows": aux["goal_predicted"],
                         "mix": ib.mix},
            notes=f"inner SGD K={K} lr={lr:g}; grad_norms = update norm / (lr*K); reward_version={ev.reward_version}")

    # ------------------------------------------------------------------------------------------- verified utility
    def register_objective(self, task: Any, origin: Optional[str] = None) -> Optional[str]:
        """Record the origin of an objective (``OutcomeLedger.register_objective``): call it with the ``TaskSpec``
        whenever the runtime assigns one (its ``origin`` is used; ``Trust.SELF`` for goal-free goals), or with an
        objective id and an explicit ``origin``. Outcomes of objectives never registered do not supervise."""
        return self.ledger.register_objective(task, origin)

    def admit_objective(self, verified: VerifiedOutcome, *, exp_ids: Optional[Sequence[str]] = None,
                        utility: Optional[float] = None, group: str = "",
                        objective_origin: str = Trust.OPERATOR, task: Any = None) -> AdmitResult:
        """Admit a signed objective outcome. It is credited to the decisions of that objective registered since its
        previous admitted evaluation (``exp_ids`` may only name decisions registered for that objective), with the
        utility the ledger derives from the signed record: the change of ``integrity.objective_utility`` since the
        previous admitted evaluation, so repeated evaluations never count the same progress twice. ``utility``, if
        given, is the caller's expectation and must agree (else refused: ``utility_mismatch``). ``task``: the
        objective's ``TaskSpec``, registered first (same as :meth:`register_objective`). ``objective_origin`` can
        only exclude (e.g. ``Trust.SELF``); the origin that admits is the recorded one."""
        if task is not None:
            if str(getattr(task, "objective_id", "")) != verified.objective_id:
                raise ValueError("task and verified outcome name different objectives")
            self.register_objective(task)
        ids_all = self._by_objective.get(verified.objective_id, [])
        n_prev, _ = self._objective_seen.get(verified.objective_id, (0, 0.0))
        ids = list(exp_ids) if exp_ids is not None else ids_all[n_prev:]
        known = set(ids_all)
        stray = [i for i in ids if i not in known]
        if stray and objective_origin in SUPERVISING_ORIGINS:
            ok, _ = screen_verified(verified, self.ledger.check_signature)
            if ok:                                   # a genuine outcome credited to foreign decisions: refuse, log
                return self.ledger.refuse("exp_ids_not_decisions_of_objective", self.ledger.objective_key(verified),
                                          kind="objective", objective_id=verified.objective_id,
                                          observed_at=verified.evaluated_at, signature=verified.signature,
                                          content=verified.content_hash(), claimed=utility, exp_ids=stray)
        res = self.ledger.admit_verified(verified, utility=utility, exp_ids=ids, objective_origin=objective_origin,
                                         group=group or f"obj:{verified.objective_id}#{n_prev}")
        if res.accepted:
            self._objective_seen[verified.objective_id] = (len(ids_all), float(verified.value))
        return res

    def admit_measurements(self, records: Sequence[ExperienceRecord]) -> int:
        """Offer the verified measurements of ``records`` to the ledger (each observation once). Without a
        ``check_measurement`` every one is rejected: an unauthenticated ``verified`` flag is not evidence.
        ``measurement_utility`` decides *which* measurements are utility (None: not utility); the number stored is
        the ledger's (``integrity.measurement_utility``), and a hook value that disagrees is refused."""
        n = 0
        for rec in records:
            for q, o in matured_outcomes(rec):
                key = o.dedup_key or f"{o.query_id}|{o.observed_at!r}|{o.value!r}"
                if not o.verified or key in self._tried:
                    continue
                self._tried.add(key)
                u = self.measurement_utility(rec, q, o)
                if u is None:
                    continue
                res = self.ledger.admit_measurement(o, objective_id=rec.objective_id, utility=u,
                                                    group=independence_key(rec), exp_ids=[rec.exp_id])
                n += int(res.accepted)
        return n

    # ------------------------------------------------------------------------------------------- episodes
    @staticmethod
    def _resolved(rec: ExperienceRecord) -> bool:
        qs = {q.query_id: q for q in list(rec.pending_queries)
              + (list(rec.result.expected_outcomes) if rec.result is not None else [])}
        done = {o.query_id for o in rec.outcomes
                if o.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED, OutcomeStatus.UNOBSERVABLE)}
        return all((not q.goal_relevant) or qid in done for qid, q in qs.items())

    def _is_held(self, group: str) -> bool:
        f = self.mc.heldout_fraction
        return f > 0 and group in held_out_groups([group], fraction=f, salt="promotion")

    def _steps_of(self, evs: Sequence[UpdateEvent]) -> Tuple[List[InnerBatch], List[bool]]:
        steps, flags = [], []
        for m, e in enumerate(evs):
            if e.ib is None:
                recs = [r for r in (self.store.get(i) for i in e.exp_ids) if r is not None]
                e.ib = self.inner.build_batch(self.store, recs)
            ib = e.ib.with_mix(e.mix if self.mc.meta_mix is None else self.mc.meta_mix)
            sampled_after_prev = m > 0 and set(e.exp_ids) <= set(evs[m - 1].query_ids)
            for k in range(e.steps):
                steps.append(ib)
                flags.append(sampled_after_prev and k == 0)
        return steps, flags

    def _episode(self, ei: int) -> Optional[_Episode]:
        ev = self.events[ei]
        q_ids = [i for i in ev.query_ids if i not in self._used]
        if not q_ids:
            return None
        q_recs = [r for r in (self.store.get(i) for i in q_ids) if r is not None]
        if not q_recs:
            return None
        self.admit_measurements(q_recs)
        first = max(0, ei - max(1, self.mc.unroll_events) + 1)
        evs = self.events[first:ei + 1]
        if any(abs(e.inner_lr - ev.inner_lr) > 0 for e in evs):
            evs = [ev]
        sup = [r for r in (self.store.get(i) for e in evs for i in e.exp_ids) if r is not None]
        builder = WindowBuilder(min_independent_outcomes=0, min_gap_s=self.mc.min_gap_s,
                                provenance=self.mc.provenance or q_recs[0].provenance, require_propensities=True,
                                outcome_filter=lambda rec, o: self.ledger.has_observation(o))
        pair = builder.cut(sup + q_recs, boundary_ts=min(r.decision.ts for r in q_recs))
        q_set, by_id = set(q_ids), {r.exp_id: r for r in q_recs}
        rows = [(i, g) for i, g in zip(pair.query.exp_ids, pair.query.row_groups) if i in q_set]
        kept = {i for i, _ in rows}
        discarded = [i for i in q_ids if i not in kept and i in by_id]
        rtg = self.mc.utility == "return_to_go"
        ready: List[Tuple[ExperienceRecord, str]] = []
        for i, g in rows:
            if not self._resolved(by_id[i]):
                if rtg:
                    break
                continue
            ready.append((by_id[i], g))
        if not ready:
            return _Episode(ei, evs, pair.support.window_id, pair.query.window_id, None, [], [], [], [], [], set(),
                            discarded) if discarded else None
        obj = [[e for e in self.ledger.by_exp(r.exp_id) if e.kind == "objective"] for r, _ in ready]
        mea = [[e for e in self.ledger.by_exp(r.exp_id) if e.kind == "measurement"] for r, _ in ready]
        indep = [o[0].group if o else g for (_, g), o in zip(ready, obj)]
        held = [self._is_held(g) for g in indep]
        ts = [r.decision.ts for r, _ in ready]
        life = [self._life_of.get(r.exp_id, 0) for r, _ in ready]
        beta = self.inner.cfg.beta
        n = len(ready)
        U, has_u = [], []
        for a in range(n):
            js = [j for j in range(a, n) if held[j] == held[a] and life[j] == life[a]] if rtg else [a]
            val = sum(math.exp(-beta * max(0.0, ts[j] - ts[a])) * sum(e.utility for e in mea[j]) for j in js)
            U.append(sum(e.utility for e in obj[a]) + val)
            has_u.append(bool(obj[a]) or any(bool(mea[j]) for j in js))
        # baseline group: every row whose utility can contain this row's outcome is in it (return-to-go chain)
        group = [f"rtg:life={life[a]}:held={int(held[a])}" for a in range(n)] if rtg else list(indep)
        ibq = self.inner.build_batch(self.store, [r for r, _ in ready], policy_only=True)
        pos = {i: k for k, i in enumerate(ibq.exp_ids)}
        order = [pos[r.exp_id] for r, _ in ready]
        ibq = ibq.select(order)
        return _Episode(ei, evs, pair.support.window_id, pair.query.window_id, ibq, [r.exp_id for r, _ in ready],
                        U, has_u, group, held, {indep[a] for a in range(n) if obj[a] or mea[a]}, discarded,
                        indep=indep, prefix_base=[self._base_at.get(r.exp_id, 0.0) for r, _ in ready])

    def _collect(self) -> Tuple[List[_Episode], Set[str], Set[str]]:
        eps = [e for e in (self._episode(i) for i in range(len(self.events))) if e is not None]
        groups = set().union(*[e.outcome_groups for e in eps]) if eps else set()
        held = {g for g in groups if self._is_held(g)}
        return eps, groups - held, held

    def ready(self) -> Tuple[bool, str]:
        """Is there enough independent verified query evidence for a meta-update right now?"""
        if not self.events:
            return False, "no_policy_update_recorded"
        _, used, _ = self._collect()
        if len(used) < self.mc.min_independent:
            return False, f"insufficient_independent_outcomes:{len(used)}<{self.mc.min_independent}"
        return True, ""

    # ------------------------------------------------------------------------------------------- outer update
    def _skip(self, reason: str, now: float, eps: Sequence[_Episode], n_groups: int,
              diag: Optional[Dict[str, float]] = None, **fields: Any) -> MetaUpdateRecord:
        v = self.reward.version
        rec = MetaUpdateRecord(meta_id=new_id("meta"), ts=now, reward_version_before=v, reward_version_after=v,
                               n_support=sum(len(e.events[-1].exp_ids) for e in eps if e.events),
                               n_query=sum(sum(e.has_u) for e in eps), n_independent_outcomes=n_groups,
                               inner_steps=self.mc.inner_steps, inner_lr=self.mc.inner_lr,
                               bootstrap_mix=self.mixer.mix, applied=False, skipped_reason=reason,
                               query_window=f"@{self.ledger.head}", diagnostics=dict(diag or {}), **fields)
        self.records.append(rec)
        self.stats["meta_skipped"] += 1
        return rec

    def _discard(self, eps: Sequence[_Episode]) -> None:
        for e in eps:
            if e.discarded:
                self._used.update(e.discarded)
                self.stats["discarded_rows"] += len(e.discarded)

    def meta_update(self, now: Optional[float] = None) -> MetaUpdateRecord:
        """One outer step of psi if the verified query evidence allows it; otherwise a skipped record."""
        now = self._now() if now is None else float(now)
        self.refresh_selection()
        ok, bad = self.ledger.verify_chain()
        if not ok:
            raise IntegrityError(f"the outcome ledger fails its hash chain at entry {bad}")
        if not self.events:
            return self._skip("no_policy_update_recorded", now, [], 0)
        eps, used_groups, held_groups = self._collect()
        self._discard(eps)
        eps = [e for e in eps if e.ibq is not None]
        if len(used_groups) < self.mc.min_independent:
            return self._skip(f"insufficient_independent_outcomes:{len(used_groups)}<{self.mc.min_independent}",
                              now, eps, len(used_groups), {"n_heldout_groups": float(len(held_groups))})
        v_before, h_before = self.reward.version, params_hash(self.reward)
        psi = self.reward.psi()
        dev = next(iter(psi.values())).device
        # rows that carry a verified utility and are not reserved for promotion evidence
        rows: List[Tuple[int, int]] = [(k, a) for k, e in enumerate(eps) for a in range(len(e.exp_ids))
                                       if e.has_u[a] and not e.held[a]]
        N = len(rows)
        tot = sum(eps[k].utility[a] for k, a in rows)
        base_of = self.outer_baselines(eps, rows)
        grads = {n: torch.zeros_like(p) for n, p in psi.items()}
        n_sup, n_used, ess, max_w, clipped, obj = 0, 0, float("inf"), 1.0, 0.0, 0.0
        estimators: Set[str] = set()
        unsupported: Dict[str, int] = {}
        used_ids: List[str] = []
        K_used, lr_used, d_policy = 0, self.mc.inner_lr, []
        adv: List[float] = []                                  # U - b of every supported outer row
        u_rows: List[float] = []
        traces: Dict[int, Tuple[List[InnerBatch], Params]] = {}
        with self.reward.meta_window():
            for k, e in enumerate(eps):
                idx = [a for a in range(len(e.exp_ids)) if e.has_u[a] and not e.held[a]]
                used_ids.extend(e.exp_ids[a] for a in range(len(e.exp_ids)) if e.has_u[a])
                steps, flags = self._steps_of(e.events)
                theta0 = {n: v.to(dev) for n, v in e.events[0].theta0.items()}
                traces[k] = (steps, theta0)
                if not idx:
                    continue
                lr = e.events[-1].inner_lr
                trace = self.engine.inner_update(theta0, psi, steps, lr, len(steps), create_graph=True,
                                                 on_policy=flags, mode=self.mc.mode)
                ibq = e.ibq.select(idx)
                with torch.no_grad():
                    lpK = self.inner.logp(trace.detached(), ibq)
                    lp0 = self.inner.logp(theta0, ibq)
                blp = ibq.behaviour_logp
                current = False
                if all(not ev.external and ev.theta1_hash for ev in e.events):   # replay the live update exactly
                    replay = self.engine.inner_update(theta0, {n_: v_.detach() for n_, v_ in psi.items()}, steps, lr,
                                                      len(steps), create_graph=False).detached()
                    current = self.logged_policy_is_current(e.events, replay)
                match = current and bool(torch.isfinite(blp).all()) and \
                    float((lpK.exp().to(torch.float64) - blp.exp()).abs().max()) <= self.mc.on_policy_tol
                U = torch.tensor([e.utility[a] for a in idx], dtype=torch.float32, device=dev)
                base = torch.tensor([base_of[(k, a)] for a in idx], dtype=torch.float32, device=dev)
                ind = e.indep or e.group
                gkeys = sorted({ind[a] for a in idx})
                gi = torch.tensor([gkeys.index(ind[a]) for a in idx], dtype=torch.long, device=dev)
                q = QueryData(batch=ibq, utility=U, baseline=base, behaviour_logp=blp.to(torch.float32),
                              weights=torch.full((len(idx),), 1.0 / N, device=dev), groups=gi,
                              verified=torch.ones(len(idx), dtype=torch.bool, device=dev))
                res = self.engine.meta_gradient(trace, q, "on_policy" if match else "importance")
                if not res.supported:
                    unsupported[res.reason.split(":")[0]] = unsupported.get(res.reason.split(":")[0], 0) + 1
                    continue
                for n_, g_ in res.grads.items():
                    grads[n_] += g_
                adv.extend((U.to(torch.float64) - base.to(torch.float64)).tolist())
                u_rows.extend(U.tolist())
                estimators.add(res.estimator)
                n_sup += len(steps[0].exp_ids)
                n_used += len(idx)
                ess, max_w = min(ess, res.ess), max(max_w, res.max_weight)
                clipped += res.clipped_fraction * len(idx)
                obj += res.objective
                K_used, lr_used = max(K_used, trace.K), lr
                d_policy.append(float((lpK.exp() - lp0.exp()).abs().mean()))
        if n_used == 0:
            self._used.update(used_ids)
            why = max(unsupported, key=unsupported.get) if unsupported else "no_query_rows_with_verified_utility"
            return self._skip(f"no_supported_episode:{why}", now, eps, len(used_groups),
                              {f"unsupported.{r}": float(c) for r, c in unsupported.items()})
        norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        est = ESTIMATOR_ON_POLICY if estimators == {ESTIMATOR_ON_POLICY} else \
            (ESTIMATOR_IMPORTANCE if estimators == {ESTIMATOR_IMPORTANCE} else "mixed_on_policy_and_importance")
        max_adv = max(abs(a) for a in adv)
        u_scale = max(1.0, max(abs(u) for u in u_rows))
        why_diag = {"meta_grad_norm": norm if math.isfinite(norm) else -1.0, "n_query_rows": float(n_used),
                    "utility_min": min(u_rows), "utility_max": max(u_rows),
                    "n_distinct_utilities": float(len({round(u, 12) for u in u_rows})),
                    "max_abs_advantage": max_adv, "n_heldout_groups": float(len(held_groups)),
                    **{f"unsupported.{r}": float(c) for r, c in unsupported.items()}}
        refuse = ""
        if not math.isfinite(norm):
            refuse = "non_finite_meta_gradient"
        elif max_adv <= 1e-9 * u_scale:
            refuse = (f"zero_advantage: all {n_used} query utilities equal their baselines "
                      f"({len({round(u, 12) for u in u_rows})} distinct utility value(s), "
                      f"utility in [{min(u_rows):.4g}, {max(u_rows):.4g}])")
        elif norm <= self.mc.min_meta_grad_norm:
            refuse = f"zero_meta_gradient:{norm:.3e}<={self.mc.min_meta_grad_norm:g}"
        if refuse:
            # zero advantage: the rows stay available and wait for contrast from later verified outcomes; a zero or
            # non-finite gradient with contrast (psi cannot influence these updates) consumes them
            if not refuse.startswith("zero_advantage"):
                self._used.update(used_ids)
            return self._skip(refuse, now, eps, len(used_groups), why_diag, meta_grad_norm=why_diag["meta_grad_norm"],
                              estimator=est, outer_objective=obj)
        self._used.update(used_ids)
        result = MetaGradResult(True, "", est, grads=grads, grad_norm=norm, objective=obj, n=n_used,
                                n_groups=len(used_groups), ess=ess, max_weight=max_w,
                                clipped_fraction=clipped / n_used, K=K_used, inner_lr=lr_used, mode=self.mc.mode)
        opt_state = copy.deepcopy(self.optimizer.state_dict())
        info = MetaGradient.apply(self.optimizer, self.reward, result, clip=self.mc.clip_reward)
        if not info.get("applied", 0.0) or params_hash(self.reward) == h_before:
            self.optimizer.load_state_dict(opt_state)            # nothing moved: no optimizer state either
            for p in self.reward.parameters():
                p.grad = None
            return self._skip("psi_unchanged_by_the_step" if info.get("applied", 0.0)
                              else f"not_applied:{info.get('reason', '')}", now, eps, len(used_groups),
                              {**why_diag, "clip_scale": float(info.get("clip_scale", 0.0))},
                              meta_grad_norm=norm, estimator=est, outer_objective=obj)
        self.mixer.note_meta_update()
        mean_u = tot / max(N, 1)
        self._u_ema = mean_u if self._u_ema is None else 0.9 * self._u_ema + 0.1 * mean_u
        promo = self._promotion(eps, traces, dev)
        self.stats["meta_applied"] += 1
        tensors_nonzero = sum(1 for g in grads.values() if bool((g != 0).any()))
        diag = {"ess": ess if math.isfinite(ess) else 0.0, "clipped_fraction": clipped / n_used,
                "clip_scale": info["clip_scale"], "n_episodes": float(len(eps)),
                "n_heldout_groups": float(len(held_groups)), "utility_mean": mean_u,
                "policy_prob_change_on_query": sum(d_policy) / len(d_policy),
                "psi_tensors_with_gradient": float(tensors_nonzero), "psi_tensors": float(len(grads)),
                "psi_changed": float(params_hash(self.reward) != h_before), "ledger_entries": float(len(self.ledger)),
                "ledger_rejected": float(len(self.ledger.rejected)), **promo,
                **{f"unsupported.{r}": float(c) for r, c in unsupported.items()}}
        wid = lambda ids: ids[0] if len(ids) == 1 else "w_" + hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]
        rec = MetaUpdateRecord(
            meta_id=new_id("meta"), ts=now, reward_version_before=v_before, reward_version_after=self.reward.version,
            support_window=wid([e.support_window for e in eps]),
            query_window=f"{wid([e.query_window for e in eps])}@{self.ledger.head}",
            n_support=n_sup, n_query=n_used, n_independent_outcomes=len(used_groups), inner_steps=K_used,
            inner_lr=lr_used, outer_objective=obj, meta_grad_norm=norm, estimator=est, max_importance_weight=max_w,
            bootstrap_mix=self.mixer.mix, applied=True, skipped_reason="", diagnostics=diag)
        self.records.append(rec)
        return rec

    @staticmethod
    def outer_baselines(eps: Sequence[_Episode], rows: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """Baseline b of each outer row (k, a): the mean utility of the rows in ``rows`` outside the row's baseline
        group (``_Episode.group``: the independent-outcome group, or with return-to-go the whole chain), else the
        row's prefix baseline (running mean verified utility when its decision was registered). Neither depends
        on the row's own action or outcome, so E[grad log pi(a) * b] = 0 and the estimator stays unbiased."""
        tot = sum(eps[k].utility[a] for k, a in rows)
        g_sum: Dict[str, float] = {}
        g_cnt: Dict[str, int] = {}
        for k, a in rows:
            g = eps[k].group[a]
            g_sum[g] = g_sum.get(g, 0.0) + eps[k].utility[a]
            g_cnt[g] = g_cnt.get(g, 0) + 1
        out: Dict[Tuple[int, int], float] = {}
        for k, a in rows:
            g = eps[k].group[a]
            rest = len(rows) - g_cnt[g]
            pb = eps[k].prefix_base[a] if eps[k].prefix_base else 0.0
            out[(k, a)] = (tot - g_sum[g]) / rest if rest > 0 else pb
        return out

    @staticmethod
    def logged_policy_is_current(events: Sequence[UpdateEvent], theta_K: Optional[Params] = None) -> bool:
        """Were the query decisions after ``events[-1]`` sampled from exactly the policy the meta-update recomputes?
        Only if every event was applied here by :meth:`policy_update` (not by an external optimizer) and
        ``theta_K`` — the same functional SGD replayed without a graph from the event's snapshot with the current
        psi, input statistics and the mix of the differentiated update — is bit-identical to the live parameters
        that update left (``theta1_hash``). That fails whenever psi, the reward's input statistics or the bootstrap
        mix differ from those the live update used (e.g. the shadow update of the cold start), or the subset was
        changed between chained events: the logged policy is then a different one and the estimator is importance
        weighting. (The differentiable unroll itself is not compared: building the graph changes float rounding.)"""
        if not events or any(e.external or not e.theta1_hash for e in events):
            return False
        return theta_K is not None and params_hash(theta_K) == events[-1].theta1_hash

    def _promotion(self, eps: Sequence[_Episode], traces: Dict[int, Tuple[List[InnerBatch], Params]],
                   dev: torch.device) -> Dict[str, float]:
        """Held-out evidence for the bootstrap mix: per held-out group, the clipped importance-weighted verified
        utility of the policy updated with the learned reward (new psi) minus that of the policy updated with the
        bootstrap reward, both from the same snapshot on the same support data."""
        psi = {k: v.detach() for k, v in self.reward.psi().items()}
        lp_l, lp_b, blp, U, grp = [], [], [], [], []
        for k, e in enumerate(eps):
            idx = [a for a in range(len(e.exp_ids)) if e.has_u[a] and e.held[a]]
            if not idx or k not in traces:
                continue
            steps, theta0 = traces[k]
            lr = e.events[-1].inner_lr
            ibq = e.ibq.select(idx)
            if not bool(torch.isfinite(ibq.behaviour_logp).all()):
                continue
            with self.reward.meta_window():
                th_l = self.engine.inner_update(theta0, psi, [s.with_mix(0.0) for s in steps], lr, len(steps),
                                                create_graph=False).detached()
                th_b = self.engine.inner_update(theta0, psi, [s.with_mix(1.0) for s in steps], lr, len(steps),
                                                create_graph=False).detached()
            with torch.no_grad():
                lp_l.append(self.inner.logp(th_l, ibq).to(torch.float64).cpu())
                lp_b.append(self.inner.logp(th_b, ibq).to(torch.float64).cpu())
            blp.append(ibq.behaviour_logp.cpu())
            U.append(torch.tensor([e.utility[a] for a in idx], dtype=torch.float64))
            grp.extend((e.indep or e.group)[a] for a in idx)
        out = {"promotion": 0.0, "promotion_buffer": float(len(self._promo))}
        if not grp:
            return out
        keys = sorted(set(grp))
        gi = torch.tensor([keys.index(g) for g in grp], dtype=torch.long)
        self._promo.extend(paired_policy_value_diffs(torch.cat(lp_l), torch.cat(lp_b), torch.cat(blp), torch.cat(U),
                                                     gi, weight_clip=self.mc.weight_clip))
        u_all = torch.cat(U)
        self._promo_scale.extend(float(x) for x in u_all)
        if len(self._promo) >= max(2, self.mixer.min_independent):
            scale = math.sqrt(sum(x * x for x in self._promo_scale) / max(len(self._promo_scale), 1))
            dec = self.mixer.update(self._promo, scale=scale)
            self._promo, self._promo_scale = [], []
            out.update({"promotion": {"lower": 1.0, "raise": -1.0}.get(dec.action, 0.0),
                        "promotion_mean_diff": dec.mean_diff, "promotion_lower_bound": dec.lower_bound,
                        "promotion_upper_bound": dec.upper_bound, "promotion_margin": dec.margin,
                        "promotion_n": float(dec.n_independent), "mix_after_promotion": dec.mix_after})
        out["promotion_buffer"] = float(len(self._promo))
        return out

    # ------------------------------------------------------------------------------------------- persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"mixer": self.mixer.state_dict(), "life": self._life, "life_of": dict(self._life_of),
                "life_states": {k: v.detach().cpu() for k, v in self._life_states.items()},
                "by_objective": {k: list(v) for k, v in self._by_objective.items()},
                "objective_seen": {k: list(v) for k, v in self._objective_seen.items()},
                "used": sorted(self._used), "tried": sorted(self._tried), "promo": list(self._promo),
                "promo_scale": list(self._promo_scale),
                "base_at": dict(self._base_at), "objective_origins": self.ledger.origins(),
                "u_ema": self._u_ema, "n_events": self._n_events, "optimizer": self.optimizer.state_dict(),
                "ledger_head": self.ledger.head, "stats": dict(self.stats),
                "gate": self.inner.bootstrap.gate.state_dict(),
                "selection": self.selection.state_dict(), "selection_pending": list(self._sel_pending),
                "selection_registered": [self._n_registered, dict(self._reg_at)],
                "records": [r.to_dict() for r in self.records[-256:]],
                "events": [{"event_id": e.event_id, "ts": e.ts, "exp_ids": list(e.exp_ids), "inner_lr": e.inner_lr,
                            "steps": e.steps, "mix": e.mix, "reward_version": e.reward_version,
                            "lifetime": e.lifetime, "query_ids": list(e.query_ids), "external": e.external,
                            "theta0_hash": e.theta0_hash, "theta1_hash": e.theta1_hash,
                            "theta0": {n: v.detach().cpu() for n, v in e.theta0.items()}} for e in self.events]}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        """Restore the bookkeeping. Cached support latents are rebuilt from the store on first use; the ledger is
        reloaded from its own file and must still contain the head recorded here."""
        head = sd.get("ledger_head", OutcomeLedger.GENESIS)
        if head != OutcomeLedger.GENESIS and head not in {e.hash for e in self.ledger.entries()}:
            raise IntegrityError("the outcome ledger does not contain the head recorded in the checkpoint")
        self.ledger.confirm_head(head)               # a ledger file from before anchors: adopted via this head
        dev = next(self.reward.parameters()).device
        self.mixer.load_state_dict(sd["mixer"])
        self._life = int(sd["life"])
        self._life_of.clear()
        self._life_of.update(sd["life_of"])
        self._life_states.clear()
        self._life_states.update({k: v.to(dev) for k, v in sd["life_states"].items()})
        self._by_objective = {k: list(v) for k, v in sd["by_objective"].items()}
        self._objective_seen = {k: (int(v[0]), float(v[1])) for k, v in sd["objective_seen"].items()}
        self._used, self._tried = set(sd["used"]), set(sd["tried"])
        self._promo, self._u_ema, self._n_events = list(sd["promo"]), sd["u_ema"], int(sd["n_events"])
        self._promo_scale = [float(x) for x in sd.get("promo_scale", [])]
        self._base_at = {k: float(v) for k, v in sd.get("base_at", {}).items()}
        self.ledger.load_origins(sd.get("objective_origins", {}))
        self.optimizer.load_state_dict(sd["optimizer"])
        self.stats.update(sd.get("stats", {}))
        self.inner.bootstrap.gate.load_state_dict(sd["gate"])
        if "selection" in sd:                       # checkpoints from before the selection calibration: empty
            self.selection.load_state_dict(sd["selection"])
            self._sel_pending = list(sd.get("selection_pending", []))
            n_reg, reg_at = sd.get("selection_registered", [0, {}])
            self._n_registered, self._reg_at = int(n_reg), {k: int(v) for k, v in reg_at.items()}
        self.records = [MetaUpdateRecord.from_dict(d) for d in sd.get("records", [])]
        self.events = [UpdateEvent(event_id=e["event_id"], ts=e["ts"], exp_ids=list(e["exp_ids"]),
                                   inner_lr=e["inner_lr"], steps=e["steps"], mix=e["mix"],
                                   reward_version=e["reward_version"], lifetime=e["lifetime"],
                                   theta0={n: v.to(dev) for n, v in e["theta0"].items()}, ib=None,
                                   query_ids=list(e["query_ids"]), external=e["external"],
                                   theta0_hash=e.get("theta0_hash", ""), theta1_hash=e.get("theta1_hash", ""))
                       for e in sd["events"]]
