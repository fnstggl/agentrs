"""cam_live.reward.assemble — where every entry of the reward signal vector comes from (spec 7.2, Part 8).

``SignalAssembler.assemble`` builds the named vector of ``reward.signals`` for a batch of stored decisions. Every
value is something that had been observed, measured or predicted *by the time the reward is computed* — the moment
H_{t+1} was formed (``reward_time`` = the later of decision time + duration and the newest item of
``post_item_refs``). Nothing
that matured later is read, and nothing is invented: a signal with no source is masked (value 0, mask 0).

Source of each signal, in priority order (first available wins):

    any signal               ``decision.diagnostics["reward_signals"][name]`` — written by the agent when it computed
                             the quantity at decision / reconciliation time (change request C-4 workaround)
    success_before           with an evaluator: the evaluator head on ``b`` (the same function as success_after);
                             else the state value recorded before acting (``CandidateEvaluation.extras["v_state"]``)
                             — never the candidate's Q(b, a) (``p_success``), which differs from V by an
                             action-dependent offset (V(b') - Q(b, a) paid the actions rated low, S-5)
    success_after            evaluator head on ``b_next`` (``predict.evaluator.delta_v``) when an evaluator is given
    delta_v                  success_after - success_before (derived only when both exist)
    action_quality           ``predict.evaluator.action_quality`` on the world model's predicted goal value net of cost
                             (``extras["q_goal"] - extras["net_cost"]``) of every candidate (else the recorded
                             ``CandidateEvaluation.value``) and the logged probabilities
    info_gain                chosen ``CandidateEvaluation.extras["info_gain"]`` (fast-posterior KL, ``FastBeliefState
                             .update()["kl"]``)
    voi                      chosen ``CandidateEvaluation.voi`` (predict.voi, estimated before acting)
    cost_money / cost_seconds / cost_compute_s      ``ActionResult.cost`` (measured)
    elapsed_s                decision time minus the time of the newest ACTION item in H_t (previous decision)
    prediction_error_std     |outcome - mean| / std of the chosen candidate's *prospective* predictions
                             (``integrity.is_prospective``) for outcomes observed by reward time
    epistemic_before         chosen ``CandidateEvaluation.uncertainty``
    epistemic_after          chosen ``CandidateEvaluation.extras["epistemic_after"]``
    verified_progress_delta  sum of the quantities of OUTCOME items with verifier trust that are in H_{t+1} but not
                             in H_t and answer one of *this decision's* goal-relevant questions (item ref
                             ``<query_id>:out``); the agent's signed per-decision annotation takes precedence
    context_novelty          stored value only
    status.* / failure.*     ``ActionResult.status`` / ``ActionResult.failure_class``

Extra, for the labelled bootstrap reward only: ``learning_progress`` =
``decision.diagnostics["reward_signals"]["learning_progress"]`` (prediction-loss improvement on the same targets,
``bootstrap.LearningProgressTracker``).

The verified return that supervises the outer objective is never an input here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..contracts import ExperienceRecord, ExperienceStore, ItemKind, Trust
from ..learning_api import DecisionBatch, matured_outcomes
from .integrity import is_prospective
from .signals import SCALAR_NAMES, RewardSignals

STORED_KEY = "reward_signals"

__all__ = ["STORED_KEY", "SIGNAL_SOURCES", "AssembledSignals", "SignalAssembler", "record_signals",
           "reward_time", "context_key"]

SIGNAL_SOURCES: Dict[str, str] = {
    "success_before": "stored | evaluator(b) with an evaluator | CandidateEvaluation.extras['v_state'] (state value)",
    "success_after": "stored | evaluator(b_next)",
    "delta_v": "stored | success_after - success_before",
    "action_quality": "stored | predict.evaluator.action_quality(q_goal - net_cost (else CandidateEvaluation.value), "
                      "logged probabilities)",
    "info_gain": "stored | CandidateEvaluation.extras['info_gain'] of the chosen candidate",
    "voi": "stored | CandidateEvaluation.voi of the chosen candidate",
    "cost_money": "stored | ActionResult.cost.money",
    "cost_seconds": "stored | ActionResult.cost.seconds",
    "cost_compute_s": "stored | ActionResult.cost.compute_s",
    "elapsed_s": "stored | decision.ts - ts of the newest ACTION item in H_t",
    "prediction_error_std": "stored | prospective predictions of the chosen candidate vs outcomes observed by reward time",
    "epistemic_before": "stored | CandidateEvaluation.uncertainty of the chosen candidate",
    "epistemic_after": "stored | CandidateEvaluation.extras['epistemic_after'] of the chosen candidate",
    "verified_progress_delta": "stored | verifier-trust OUTCOME items in H_{t+1}, not in H_t, answering this "
                               "decision's own goal-relevant questions",
    "context_novelty": "stored",
    "status.*": "ActionResult.status",
    "failure.*": "ActionResult.failure_class",
}


def context_key(rec: ExperienceRecord) -> str:
    """Context name used by the noisy-TV gate: the operation and what it acted on."""
    a = rec.decision.chosen
    return f"{a.operation}|{a.target}"


def reward_time(rec: ExperienceRecord, post_items: Sequence[Any]) -> float:
    """When H_{t+1} was formed: nothing observed later may enter the reward of this decision."""
    end = float(rec.decision.ts) + max(float(rec.duration_s), 0.0)
    return max([end] + [float(i.ts) for i in post_items])


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _goal_query_ids(rec: ExperienceRecord) -> set:
    qs = list(rec.queries) + list(rec.pending_queries) + (list(rec.result.expected_outcomes) if rec.result else [])
    return {q.query_id for q in qs if q.goal_relevant}


def record_signals(rec: ExperienceRecord, store: ExperienceStore) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """(signals, extras, reward_time) of one stored decision from the record and the item store alone."""
    d = rec.decision
    out: Dict[str, float] = {}
    ev = d.candidates[d.chosen_index] if d.candidates and 0 <= d.chosen_index < len(d.candidates) else None
    ctx_items = store.items(rec.context_item_refs)
    ctx_refs = set(rec.context_item_refs)
    post_items = store.items(rec.post_item_refs) if rec.post_item_refs else []
    t_r = reward_time(rec, post_items)
    if ev is not None:
        # success_before is the *state* value V(b) the evaluator gave when the decision was made (extras["v_state"]),
        # the same function success_after is evaluated with: the candidate's Q(b, a) (p_success) differs from V by an
        # action-dependent offset, and V(b') - Q(b, a) paid systematically for the actions the evaluator rates low
        for name, val in (("success_before", ev.extras.get("v_state")), ("voi", ev.voi),
                          ("epistemic_before", ev.uncertainty), ("info_gain", ev.extras.get("info_gain")),
                          ("epistemic_after", ev.extras.get("epistemic_after"))):
            v = _finite(val)
            if v is not None:
                out[name] = v
    goal = [_finite((c.extras or {}).get("q_goal")) for c in d.candidates]
    vals = [_finite(c.value) for c in d.candidates]
    if goal and all(v is not None for v in goal):          # the world model's predicted goal value, net of cost
        vals = [g - float((c.extras or {}).get("net_cost", 0.0) or 0.0) for g, c in zip(goal, d.candidates)]
    if vals and all(v is not None for v in vals) and len(d.probabilities) == len(vals):
        from ..predict.evaluator import action_quality
        q = torch.tensor([vals], dtype=torch.float64)
        p = torch.tensor([d.probabilities], dtype=torch.float64)
        out["action_quality"] = float(action_quality(q, p, torch.tensor([d.chosen_index]))["quality"][0])
    if rec.result is not None:
        c = rec.result.cost
        out["cost_money"], out["cost_seconds"], out["cost_compute_s"] = float(c.money), float(c.seconds), float(c.compute_s)
    prev = [float(i.ts) for i in ctx_items if i.kind == ItemKind.ACTION and float(i.ts) <= float(d.ts)]
    if prev:
        out["elapsed_s"] = float(d.ts) - max(prev)
    if ev is not None and ev.predictions:
        preds = {p.query.query_id: p for p in ev.predictions}
        errs = []
        for q, o in matured_outcomes(rec):
            p = preds.get(q.query_id)
            if p is None or o.value is None or o.observed_at > t_r or not is_prospective(p, o):
                continue
            mean, std = _finite(p.dist.mean), _finite(p.dist.std)
            if mean is not None and std is not None:
                errs.append(abs(float(o.value) - mean) / max(std, 1e-9))
        if errs:
            out["prediction_error_std"] = sum(errs) / len(errs)
    if rec.post_item_refs:
        # only this decision's own goal-relevant questions: a verifier reading of another decision's outcome that
        # happened to arrive during this step (e.g. while waiting) is not progress made by this decision
        own = _goal_query_ids(rec)
        new = [i for i in post_items if i.ref not in ctx_refs and i.kind == ItemKind.OUTCOME
               and i.trust == Trust.VERIFIER and str(i.ref).rsplit(":", 1)[0] in own]
        out["verified_progress_delta"] = float(sum(q.value for i in new for q in i.quantities))
    stored = d.diagnostics.get(STORED_KEY) if isinstance(d.diagnostics, dict) else None
    extras: Dict[str, float] = {}
    if isinstance(stored, dict):
        for k, v in stored.items():
            f = _finite(v[0] if isinstance(v, (list, tuple)) and v else v)
            if f is None:
                continue
            if k in SCALAR_NAMES:
                out[k] = f
            elif k == "learning_progress":
                extras[k] = f
    return out, extras, t_r


@dataclass
class AssembledSignals:
    signals: RewardSignals            # raw values (float64, CPU), one row per decision
    tensor: Tensor                    # [B, N_SIGNALS] transformed, on the model device
    mask: Tensor                      # [B, N_SIGNALS] 1 = available
    learning_progress: Tensor         # [B] float64, NaN where no value was recorded (bootstrap input only)
    reward_time: Tensor               # [B] float64
    keys: List[str]                   # noisy-TV context key of each row


class SignalAssembler:
    """``evaluator``: a ``predict.evaluator.GoalEvaluator`` (or anything with the same ``forward``); used under
    ``no_grad`` to fill success_before / success_after when the record does not carry them. ``horizon_fn(record)``
    returns the seconds remaining for the objective before and after the step (default: ``default_horizon_s`` and
    that minus the step duration)."""

    def __init__(self, *, evaluator: Optional[Any] = None,
                 horizon_fn: Optional[Callable[[ExperienceRecord], Tuple[float, float]]] = None,
                 default_horizon_s: float = 86400.0):
        self.evaluator, self.horizon_fn, self.default_horizon_s = evaluator, horizon_fn, float(default_horizon_s)

    def _horizons(self, rec: ExperienceRecord) -> Tuple[float, float]:
        if self.horizon_fn is not None:
            h0, h1 = self.horizon_fn(rec)
            return float(h0), float(h1)
        return self.default_horizon_s, max(0.0, self.default_horizon_s - max(float(rec.duration_s), 0.0))

    def assemble(self, records: Sequence[ExperienceRecord], store: ExperienceStore, *,
                 b: Optional[Tensor] = None, b_next: Optional[Tensor] = None, e_g: Optional[Tensor] = None,
                 has_next: Optional[Tensor] = None, device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32) -> AssembledSignals:
        rows, lps, times = [], [], []
        for rec in records:
            sig, extras, t_r = record_signals(rec, store)
            rows.append(sig)
            lps.append(extras.get("learning_progress", float("nan")))
            times.append(t_r)
        if self.evaluator is not None and b is not None and e_g is not None and records:
            hz = torch.tensor([self._horizons(r) for r in records], dtype=torch.float64, device=b.device)
            with torch.no_grad():
                p0 = self.evaluator(b.detach(), e_g.detach(), hz[:, 0]).p_success.detach().cpu()
                p1 = None if b_next is None else \
                    self.evaluator(b_next.detach(), e_g.detach(), hz[:, 1]).p_success.detach().cpu()
            for i, (rec, sig) in enumerate(zip(records, rows)):
                stored = rec.decision.diagnostics.get(STORED_KEY) if isinstance(rec.decision.diagnostics, dict) else None
                stored = stored if isinstance(stored, dict) else {}
                # both ends of delta V from the same function V(., g, horizon) (a stored value still wins)
                if p1 is not None and (has_next is None or bool(has_next[i])) and "success_after" not in stored:
                    sig["success_after"] = float(p1[i])
                    if "success_before" not in stored:
                        sig["success_before"] = float(p0[i])
                else:
                    sig.setdefault("success_before", float(p0[i]))
        items = [RewardSignals.from_step(result=rec.result, **sig) if (sig or rec.result is not None)
                 else RewardSignals(status=[""], failure_class=[""]) for rec, sig in zip(records, rows)]
        sig = RewardSignals.stack(items)
        return AssembledSignals(signals=sig, tensor=sig.to_tensor(dtype, device), mask=sig.mask(dtype, device),
                                learning_progress=torch.tensor(lps, dtype=torch.float64),
                                reward_time=torch.tensor(times, dtype=torch.float64),
                                keys=[context_key(r) for r in records])

    def assemble_batch(self, batch: DecisionBatch, store: ExperienceStore, *, b_next: Optional[Tensor] = None,
                       has_next: Optional[Tensor] = None) -> AssembledSignals:
        return self.assemble(batch.records, store, b=batch.b, b_next=b_next if b_next is not None else batch.b_next,
                             e_g=batch.e_g, has_next=has_next if has_next is not None else batch.has_next,
                             device=batch.b.device)
