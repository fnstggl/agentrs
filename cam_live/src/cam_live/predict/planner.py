"""cam_live.predict.planner — latent imagination with the transition head on the real model (spec 4.3).

``LatentPlanner.plan`` compares alternative feasible first actions by rolling CAM's belief forward in latent space:
the first step applies the candidate itself, further steps follow the candidate policy's most probable action at
the imagined belief. Uncertainty is propagated with particles (a random ensemble member plus its aleatoric noise at
every step) and read out with the goal evaluator. No text is decoded or produced.

The number of imagined steps is limited by a :class:`HorizonController` fed by :meth:`LatentPlanner.measure`: the
open-loop k-step error of the transition head on *held-out real* transitions. It starts at one step and grows only
while the measured error stays under the threshold.

Everything returned is an :class:`ImaginedEvaluation` with ``imagined = True``. The outcome APIs of this package
(``PredictionRegistry.resolve``, ``ExperimentRegistry.add_outcome``, ``world.build_targets``,
``Predictor.observe_outcomes``, ``EvaluatorLoss.add_outcome``) refuse such objects, so an imagined value cannot
become an outcome, a training target for an outcome head, or evidence that an objective was achieved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from ..contracts import (ActionSpec, ExperienceItem, ExperienceRecord, ExperienceStore, OutcomeKind, OutcomeQuery,
                         PredictionRecord, new_id)
from ..learning_api import encode_decisions
from ..nn.api import LatentModelAPI
from . import dists as D
from .heads import PredictConfig, PredictHeads, attach_heads
from .imagination import HorizonController, imagined_prediction, multistep_error


@dataclass(frozen=True)
class ImaginedEvaluation:
    """Model-estimated prospects after imagining a candidate. Never an outcome."""
    action_id: str
    index: int
    value_mean: float                # evaluator success probability at the imagined belief, mean over particles
    value_std: float                 # spread over particles (epistemic + aleatoric, compounded over the steps)
    epistemic_std: float             # mean ensemble disagreement of the one-step models along the rollout
    latent_std: float                # mean per-dimension spread of the final imagined belief
    horizon_steps: int
    elapsed_s: float
    feasible: bool = True
    imagined: bool = True


class LatentPlanner:
    def __init__(self, model: LatentModelAPI, cfg: Any = None, *, heads: Optional[PredictHeads] = None,
                 controller: Optional[HorizonController] = None):
        self.model = model
        pcfg = cfg if isinstance(cfg, PredictConfig) else PredictConfig.from_cfg(cfg or getattr(model, "cfg", None))
        self.heads = heads or attach_heads(model, pcfg)
        self.cfg = self.heads.cfg
        self.controller = controller or HorizonController(max_horizon=self.cfg.max_imagination_horizon,
                                                          threshold=self.cfg.imagination_threshold)

    # ---- measured multi-step accuracy on real transitions -----------------------------------------------------
    @torch.no_grad()
    def measure(self, store: ExperienceStore, sequences: Sequence[Sequence[ExperienceRecord]],
                max_k: Optional[int] = None) -> Dict[str, Any]:
        """Update the horizon controller with the open-loop error on real sequences of consecutive decisions
        (each inner list: one lifetime's records in order; use held-out ones). Returns the mean error per k,
        the number of sequences used and the horizon now allowed."""
        max_k = max_k or self.controller.max_horizon
        errs: List[Tensor] = []
        for recs in sequences:
            recs = [r for r in recs if r.post_item_refs]
            if len(recs) < 2:
                continue
            batch = encode_decisions(self.model, store, recs, with_next=True, grad=False)
            b_seq = torch.cat([batch.b, batch.b_next[-1:]], 0).unsqueeze(0)                 # [1, T+1, D]
            e = multistep_error(self.heads.transition, self.model.decision_latent, b_seq,
                                batch.e_a_chosen.unsqueeze(0), batch.e_g[:1], batch.duration_s.unsqueeze(0), max_k)
            errs.append(e)
            self.controller.update(e)
        if not errs:
            return {"n_sequences": 0, "errors": [], "allowed": self.controller.allowed()}
        mean = torch.nanmean(torch.stack(errs), 0)
        return {"n_sequences": len(errs), "errors": [float(x) for x in mean], "allowed": self.controller.allowed(),
                "threshold": self.controller.threshold}

    # ---- planning ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def plan(self, context_items: Sequence[ExperienceItem], goal_text: str, candidates: Sequence[ActionSpec], *,
             now: float, deadline: Optional[float] = None, objective_scale: float = 1.0,
             horizon: Optional[int] = None, n_particles: Optional[int] = None,
             validate: Optional[Callable[[ActionSpec], List[str]]] = None,
             generator: Optional[torch.Generator] = None) -> List[ImaginedEvaluation]:
        cands = list(candidates)
        if not cands:
            return []
        model, heads, cfg = self.model, self.heads, self.cfg
        steps = self.controller.allowed() if horizon is None else max(1, min(int(horizon), self.controller.allowed()))
        p = max(1, n_particles or cfg.imagination_particles)
        b0 = model.core_forward(model.tokenize([list(context_items)]), None).b              # [1, D]
        e_a, _ = model.encode_actions([cands])                                              # [1, N, D]
        e_g = model.encode_goals([goal_text])
        n, d = e_a.shape[1], b0.shape[-1]
        feas = torch.tensor([not (validate(a) if validate is not None else []) for a in cands], device=b0.device)
        if not bool(feas.any()):
            feas = torch.ones_like(feas)
        dt = torch.tensor([max(float(a.est_cost.seconds), 1.0) for a in cands], dtype=torch.float64,
                          device=b0.device)
        r = n * p
        parts = b0.expand(r, d).clone()                                                     # row = candidate * P + particle
        eg = e_g.expand(r, d)
        ea_all = e_a.expand(r, n, d)
        idx = torch.arange(n, device=b0.device).repeat_interleave(p)                        # first action of each row
        ar = torch.arange(r, device=b0.device)
        elapsed = torch.zeros(r, dtype=torch.float64, device=b0.device)
        epi = torch.zeros(r, device=b0.device)
        for s in range(steps):
            z_all = model.decision_latent(parts, ea_all, eg)                                # [R, N, D]
            if s > 0:                                                                       # continuation policy
                pol = heads.policy(z_all, feas.unsqueeze(0).expand(r, n), parts)
                idx = pol.log_probs.argmax(-1)
            pred = heads.transition(z_all[ar, idx], dt[idx], parts)
            parts = pred.sample(generator)
            epi = epi + pred.epistemic_var.sqrt().mean(-1)
            elapsed = elapsed + dt[idx]
        left = (deadline - now) if deadline is not None else cfg.default_horizon_s
        hz = (left - elapsed).clamp_min(1.0)
        val = heads.evaluator(parts, eg, hz, scale=torch.full((r,), max(float(objective_scale), 1.0),
                                                              device=b0.device)).p_success
        val, view = val.reshape(n, p), parts.reshape(n, p, d)
        out: List[ImaginedEvaluation] = []
        for i, a in enumerate(cands):
            out.append(ImaginedEvaluation(
                action_id=a.action_id, index=i, value_mean=float(val[i].mean()),
                value_std=float(val[i].std()) if p > 1 else 0.0,
                epistemic_std=float(epi.reshape(n, p)[i].mean() / steps),
                latent_std=float(view[i].std(0).mean()) if p > 1 else 0.0, horizon_steps=steps,
                elapsed_s=float(elapsed.reshape(n, p)[i].mean()), feasible=bool(feas[i])))
        return out

    def success_prediction(self, ev: ImaginedEvaluation, objective_id: str, made_at: float, horizon_s: float,
                           decision_id: str = "") -> PredictionRecord:
        """An imagined success forecast as a ``PredictionRecord`` (``imagined_horizon_steps >= 1``). It may be
        registered and later scored against reality; it is never an outcome."""
        q = OutcomeQuery(query_id=new_id("q"), metric=f"objective.{objective_id}.achieved", kind=OutcomeKind.BINARY,
                         horizon_s=float(horizon_s), issued_at=float(made_at), decision_id=decision_id,
                         description="the objective is verified as achieved", goal_relevant=True)
        p = min(max(ev.value_mean, 1e-9), 1 - 1e-9)
        dist = D.Bernoulli.from_prob(torch.tensor(p, dtype=torch.float64)).to_record(epistemic_std=ev.value_std)
        return imagined_prediction(q, dist, made_at, ev.horizon_steps, decision_id=decision_id,
                                   action_id=ev.action_id)
