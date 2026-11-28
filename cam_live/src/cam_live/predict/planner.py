"""cam_live.predict.planner — multi-step latent imagination in decision-making (spec 4.3, amendment 2026-11-28 item 2).

``LatentPlanner.plan`` compares alternative feasible first actions by imagining, in CAM's learned latent world model,
how each of them would continue: the first imagined transition applies the candidate itself
(``LatentTransition`` at the real belief), every further step takes, *in each imagined particle*, the candidate
whose value at the imagined belief is highest (``imagination_continuation = "greedy"``; ``"policy"``: the trainable
policy's most probable candidate) and is followed by another imagined transition. Values are read with the same
heads that value the real decision: the world model's net goal-relevant value of every candidate at the imagined
belief (``Predictor.goal_values``: same questions, scalers, exposures and fast posterior as the real evaluation),
or, when the decision asks no goal-relevant question, the evaluator's success probability of the candidates.

    continuation value   C_i^p = sum_{k=1..H} v_p(b_k^{i,p}, a_k^{i,p})      (goal values: additive progress)
                         C_i^p = max_j v_p(b_H^{i,p}, a_j)                   (success probabilities: terminal)

where particle p follows transition ensemble member p mod M_T and reads world-model ensemble member p mod M_W
(a coherent sampled model: epistemic uncertainty; the transition's aleatoric noise is added only with
``imagination_aleatoric``), and an imagined step whose action would end after the objective's deadline adds
nothing. ``H`` is the horizon the :class:`HorizonController` allows: it starts at one imagined transition (a two-
action look-ahead) and grows only while the measured error of imagination on held-out *real* sequences stays under
the threshold — by default the error that matters for a decision (``imagination_criterion = "value"``): the
candidates' values read at the k-step imagined belief vs at the real belief, relative to the differences between
those candidates (:func:`value_consistency_error`); the latent error of the belief itself is measured and recorded
too.

Per candidate the planner also reports the knowledge gradient of resolving its imagined long-term value
(``info_value``: E[max(V_i, best other)] - max_j E[V_j] over the particles — how much the decision would gain if
the imagined uncertainty about this candidate were removed, i.e. the value of an experiment on it), the imagined
next action and how many particles agree on it, and the evaluator's success probability at the final imagined
belief.

How much these imagined consequences may influence the choice is *learned from verified outcomes*:
:meth:`LatentPlanner.measure` feeds the :class:`cam_live.predict.policy.ImaginationCalibration` of the transition
head with windows of verified progress (the decision and the H decisions after it) against what the decision
recorded beforehand, and rule ``thompson_policy`` weights the imagined preference by draws from that posterior
(``DecisionMaker.thompson_policy``). Unsupported or poorly calibrated imagination therefore keeps no systematic
weight.

Everything returned is an :class:`ImaginedEvaluation` with ``imagined = True``. The outcome APIs of this package
(``PredictionRegistry.resolve``, ``ExperimentRegistry.add_outcome``, ``world.build_targets``,
``Predictor.observe_outcomes``, ``EvaluatorLoss.add_outcome``) refuse such objects, so an imagined value cannot
become an outcome, a training target for an outcome head, or evidence that an objective was achieved. No text is
decoded or produced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..contracts import (ActionSpec, ExperienceItem, ExperienceRecord, ExperienceStore, OutcomeKind, OutcomeQuery,
                         PredictionRecord, new_id)
from ..learning_api import encode_decisions
from ..nn.api import LatentModelAPI
from . import dists as D
from .heads import PredictConfig, PredictHeads, attach_heads, canonical_query, world_batch
from .imagination import HorizonController, imagined_prediction, multistep_error, value_consistency_error
from .policy import IM_HORIZON, IM_MEAN, IM_SAMPLE, ImaginationCalibration, imagination_calibration

F64 = torch.float64
MAX_PENDING = 4096
MAX_CLOSED = 256


@dataclass(frozen=True)
class ImaginedEvaluation:
    """Model-estimated prospects after imagining a candidate. Never an outcome."""
    action_id: str
    index: int
    value_mean: float                # evaluator success probability at the final imagined belief, mean over particles
    value_std: float                 # its spread over particles
    epistemic_std: float             # mean ensemble disagreement of the one-step transition along the rollout
    latent_std: float                # mean per-dimension spread of the final imagined belief over particles
    horizon_steps: int               # imagined transitions
    elapsed_s: float                 # imagined time of the whole imagined sequence, mean over particles
    feasible: bool = True
    cont_mean: float = 0.0           # imagined continuation value C_i (value units of the decision), mean over particles
    cont_std: float = 0.0
    cont_samples: Tuple[float, ...] = ()
    info_value: float = 0.0          # knowledge gradient of the candidate's imagined long-term value
    next_index: int = -1             # imagined action after this candidate (most frequent over particles)
    next_share: float = 0.0          # share of particles that imagine that next action
    value_source: str = ""           # "q_goal" (world-model goal values) | "p_success" (evaluator)
    imagined: bool = True

    def extras(self) -> Dict[str, float]:
        """The numbers rule ``thompson_policy`` reads (``predict.policy.imagination_inputs``) and the decision
        records, as ``CandidateEvaluation.extras`` entries."""
        out = {"imagined_value_mean": float(self.value_mean), "imagined_value_std": float(self.value_std),
               "imagined_horizon_steps": float(self.horizon_steps), IM_HORIZON: float(self.horizon_steps),
               IM_MEAN: float(self.cont_mean), "im_cont_std": float(self.cont_std),
               "im_info_value": float(self.info_value), "im_next": float(self.next_index),
               "im_next_share": float(self.next_share), "im_epistemic_std": float(self.epistemic_std),
               "im_elapsed_s": float(self.elapsed_s), "im_value_goal": 1.0 if self.value_source == "q_goal" else 0.0}
        out.update({f"{IM_SAMPLE}{p}": float(v) for p, v in enumerate(self.cont_samples)})
        return out


def _progress(rec: ExperienceRecord) -> Tuple[Optional[float], bool]:
    """(verified progress attributed to the decision, whether that is final). A decision without any goal-relevant
    question has nothing attributable: (0, True)."""
    from ..reward.inner import verified_progress
    v = verified_progress(rec)
    if v is not None:
        return v, True
    qs = list(rec.pending_queries) + (list(rec.result.expected_outcomes) if rec.result is not None else [])
    if not any(q.goal_relevant for q in qs):
        return 0.0, True
    return None, False


class LatentPlanner:
    def __init__(self, model: LatentModelAPI, cfg: Any = None, *, heads: Optional[PredictHeads] = None,
                 controller: Optional[HorizonController] = None):
        self.model = model
        pcfg = cfg if isinstance(cfg, PredictConfig) else PredictConfig.from_cfg(cfg or getattr(model, "cfg", None))
        self.heads = heads or attach_heads(model, pcfg)
        self.cfg = self.heads.cfg
        self.controller = controller or HorizonController(max_horizon=self.cfg.max_imagination_horizon,
                                                          threshold=self.cfg.imagination_threshold,
                                                          criterion=self.cfg.imagination_criterion)
        self.calibration: ImaginationCalibration = imagination_calibration(self.heads.transition)
        if self.cfg.imagination_prior_sd != self.calibration.prior_sd_im and not len(self.calibration):
            self.calibration.prior_sd_q = self.calibration.prior_sd_im = float(self.cfg.imagination_prior_sd)
        self.restore_info: Dict[str, Any] = {}

    # ---- persistence ----------------------------------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        """Controller state (the keys ``HorizonController.load_state`` reads, so older loaders still work) plus the
        imagination calibration."""
        s = self.controller.state()
        s["imagination_calibration"] = self.calibration.state_dict()
        return s

    def load_state(self, s: Dict[str, Any]) -> None:
        """Reads both this format and a bare controller state of a checkpoint written before the imagination
        calibration existed (the calibration then starts from its prior; recorded in :attr:`restore_info`)."""
        self.controller.load_state(s)
        cal = s.get("imagination_calibration")
        if cal:
            self.calibration.load_state_dict(cal)
            self.restore_info = {"imagination_calibration": "restored", "n": len(self.calibration)}
        else:
            self.restore_info = {"imagination_calibration": "fresh (the checkpoint predates it)",
                                 "value_criterion": "fresh" if not s.get("value_count") else "restored"}

    # ---- measured multi-step accuracy on real transitions -----------------------------------------------------
    def _value_fn(self, recs: Sequence[ExperienceRecord], e_g: Tensor) -> Callable[[Tensor, int], Optional[Tensor]]:
        """q_fn(b [1, D], t) of :func:`value_consistency_error`: the world model's (heads only: no within-lifetime
        fast belief, the same for the imagined and the real belief) net goal values of decision t's candidates."""
        model, heads = self.model, self.heads
        cache: Dict[int, Any] = {}

        def rows_of(t: int) -> Optional[Tuple[Tensor, List[List[Tuple[OutcomeQuery, None]]], Tensor]]:
            if t in cache:
                return cache[t]
            rec = recs[t]
            cands = list(rec.decision.candidates)
            rows, keep = [], []
            for i, c in enumerate(cands):
                gq = [canonical_query(p.query, [c.action.action_id]) for p in c.predictions if p.query.goal_relevant]
                if gq and c.feasible:
                    top = max(q.horizon_s for q in gq)
                    rows.append([(q, None) for q in gq if q.horizon_s == top])
                    keep.append(i)
            if len(keep) < 2:
                cache[t] = None
                return None
            e_a, _ = model.encode_actions([[cands[i].action for i in keep]])
            cost = torch.tensor([float((cands[i].extras or {}).get("net_cost", 0.0) or 0.0) for i in keep], dtype=F64)
            cache[t] = (e_a, rows, cost)
            return cache[t]

        def q_fn(b: Tensor, t: int) -> Optional[Tensor]:
            r = rows_of(t)
            if r is None:
                return None
            e_a, rows, cost = r
            z = model.decision_latent(b, e_a, e_g)[0]                                        # [N, D]
            wb = world_batch(model, heads, z, rows)
            o = wb.out
            cnt = torch.exp(o.log_rate[:, :, 0].double()).mean(0) * wb.meta.exposure.double()
            real = o.loc[:, :, 0].double().mean(0) * wb.real_scale.double() + wb.real_loc.double()
            p = torch.sigmoid(o.logit[:, :, 0].double()).mean(0)
            from .world import K_BINARY, K_COUNT, K_REAL
            kid = wb.kind_ids
            val = torch.where(kid == K_COUNT, cnt, torch.where(kid == K_REAL, real,
                                                               torch.where(kid == K_BINARY, p, torch.zeros_like(p))))
            return (val * wb.q_mask.double()).sum(-1).cpu() - cost
        return q_fn

    @torch.no_grad()
    def measure(self, store: ExperienceStore, sequences: Sequence[Sequence[ExperienceRecord]],
                max_k: Optional[int] = None) -> Dict[str, Any]:
        """Update the horizon controller with the open-loop errors of imagination on real sequences of consecutive
        decisions (each inner list: one lifetime's records in order; use held-out ones): the latent error
        (:func:`multistep_error`) and the value error (:func:`value_consistency_error`). Then feed the imagination
        calibration with every window of verified progress that has become final (:meth:`refresh_calibration`).
        Returns the mean errors per k, the number of sequences used and the horizon now allowed."""
        max_k = max_k or self.controller.max_horizon
        errs: List[Tensor] = []
        vnum = torch.zeros(max_k, dtype=F64)
        vden = torch.zeros(max_k, dtype=F64)
        vper = torch.zeros(max_k, dtype=F64)
        for recs in sequences:
            recs = [r for r in recs if r.post_item_refs]
            if len(recs) < 2:
                continue
            batch = encode_decisions(self.model, store, recs, with_next=True, grad=False)
            b_seq = torch.cat([batch.b, batch.b_next[-1:]], 0).unsqueeze(0)                 # [1, T+1, D]
            e_a = batch.e_a_chosen.unsqueeze(0)
            dt = batch.duration_s.unsqueeze(0)
            e = multistep_error(self.heads.transition, self.model.decision_latent, b_seq, e_a, batch.e_g[:1], dt, max_k)
            errs.append(e)
            self.controller.update(e)
            n_, d_, p_ = value_consistency_error(self.heads.transition, self.model.decision_latent,
                                                 self._value_fn(recs, batch.e_g[:1]), b_seq, e_a, batch.e_g[:1], dt,
                                                 max_k)
            vnum += n_
            vden += d_
            vper += p_
            self.controller.update_value([float(a / b) if b > 0 else float("nan") for a, b in zip(n_, d_)])
        cal = self.refresh_calibration(store, [list(s) for s in sequences])
        if not errs:
            return {"n_sequences": 0, "errors": [], "value_errors": [], "allowed": self.controller.allowed(),
                    "calibration": cal}
        mean = torch.nanmean(torch.stack(errs), 0)
        verr = [float(a / b) if b > 0 else float("nan") for a, b in zip(vnum, vden)]
        vpe = [float(a / b) if b > 0 else float("nan") for a, b in zip(vper, vden)]
        return {"n_sequences": len(errs), "errors": [float(x) for x in mean], "value_errors": verr,
                "value_errors_persistence": vpe,
                "allowed": self.controller.allowed(), "allowed_by": self.controller.state()["allowed_by"],
                "criterion": self.controller.criterion, "threshold": self.controller.threshold, "calibration": cal}

    # ---- verified windows -> imagination calibration ----------------------------------------------------------
    def refresh_calibration(self, store: ExperienceStore,
                            sequences: Sequence[Sequence[ExperienceRecord]] = ()) -> Dict[str, Any]:
        """Register the decisions of ``sequences`` that recorded imagined values, then add every registered decision
        whose verified window is final: its own verified progress plus that of the h decisions that followed it in
        the same objective (h = its imagined horizon). A window that runs past the end of an objective is final once
        a later objective has been seen (nothing can follow); a decision whose own goal-relevant answers never get a
        signed value is dropped then. Returns counts."""
        cal = self.calibration
        objs = []
        for seq in sequences:
            for r in seq:
                if r.objective_id and r.objective_id not in objs:
                    objs.append(r.objective_id)
                if r.exp_id in cal.pending or r.result is None:
                    continue
                if ImaginationCalibration.features(r.decision) is not None and \
                        not any(r.exp_id in c._seen for c in cal.by_h.values()):
                    cal.pending[r.exp_id] = r.objective_id
        if objs:                                    # every objective seen before the newest one is closed
            newest = objs[-1]
            for o in set(cal.pending.values()) | set(objs[:-1]):
                if o != newest and o not in cal.closed:
                    cal.closed.append(o)
            del cal.closed[:max(0, len(cal.closed) - MAX_CLOSED)]
        added = dropped = 0
        by_obj: Dict[str, List[str]] = {}
        for i, o in cal.pending.items():
            by_obj.setdefault(o, []).append(i)
        for o, ids in by_obj.items():
            traj = store.trajectory(o) if o else []
            pos = {r.exp_id: k for k, r in enumerate(traj)}
            closed = o in cal.closed
            for i in ids:
                k = pos.get(i)
                if k is None:
                    if closed:
                        cal.pending.pop(i, None)
                        dropped += 1
                    continue
                rec = traj[k]
                f = ImaginationCalibration.features(rec.decision)
                if f is None:
                    cal.pending.pop(i, None)
                    dropped += 1
                    continue
                z, m, h = f
                window = traj[k:k + h + 1]
                if len(window) < h + 1 and not closed:
                    continue
                vals = [_progress(r) for r in window]
                if all(ok for _, ok in vals):
                    cal.observe(i, h, z, m, float(sum(v for v, _ in vals)))
                    cal.pending.pop(i, None)
                    added += 1
                elif closed:
                    cal.pending.pop(i, None)
                    dropped += 1
        if len(cal.pending) > MAX_PENDING:
            for i in list(cal.pending)[:len(cal.pending) - MAX_PENDING]:
                cal.pending.pop(i, None)
        return {"added": added, "dropped": dropped, "pending": len(cal.pending), "n": len(cal)}

    # ---- planning ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def plan(self, context_items: Sequence[ExperienceItem], goal_text: str, candidates: Sequence[ActionSpec], *,
             now: float, deadline: Optional[float] = None, objective_scale: float = 1.0,
             horizon: Optional[int] = None, n_particles: Optional[int] = None,
             validate: Optional[Callable[[ActionSpec], List[str]]] = None,
             generator: Optional[torch.Generator] = None, predictor: Any = None,
             evaluations: Optional[Sequence[Any]] = None) -> List[ImaginedEvaluation]:
        """Imagine every candidate's continuation (module docstring). With ``predictor`` (the
        :class:`cam_live.predict.predictor.Predictor` whose latest evaluation is of these candidates in this
        context) its belief, latents and world view are reused and the continuation is valued in its goal units;
        ``evaluations`` (that evaluation's ``CandidateEvaluation`` list) give feasibility and costs. Without a
        predictor the belief is encoded here and values are evaluator success probabilities."""
        cands = list(candidates)
        if not cands:
            return []
        model, heads, cfg = self.model, self.heads, self.cfg
        steps = self.controller.allowed() if horizon is None else max(1, min(int(horizon), self.controller.allowed()))
        last = getattr(predictor, "last", None) or {}
        reuse = bool(last) and last.get("z") is not None and last["z"].shape[1] == len(cands)
        if reuse:
            b0, e_a, e_g, z0 = last["b"], last["e_a"], last["e_g"], last["z"]
        else:
            b0 = model.core_forward(model.tokenize([list(context_items)]), None).b              # [1, D]
            e_a, _ = model.encode_actions([cands])                                              # [1, N, D]
            e_g = model.encode_goals([goal_text])
            z0 = model.decision_latent(b0, e_a, e_g)
        dev = b0.device
        n, d = e_a.shape[1], b0.shape[-1]
        if evaluations is not None and len(evaluations) == n:
            feas = torch.tensor([bool(e.feasible) for e in evaluations], device=dev)
        else:
            feas = torch.tensor([not (validate(a) if validate is not None else []) for a in cands], device=dev)
        if not bool(feas.any()):
            feas = torch.ones_like(feas)
        dt = torch.tensor([max(float(a.est_cost.seconds), 1.0) for a in cands], dtype=F64, device=dev)
        goal_mode = reuse and predictor is not None and last.get("world") is not None
        m_t = heads.transition.n_members
        m_w = heads.world.cfg.n_members
        p = max(1, int(n_particles or cfg.imagination_particles))
        if not cfg.imagination_aleatoric:
            p = min(p, m_t * m_w)                     # without aleatoric noise a particle is a (member, member) pair
        r = n * p                                     # row = first candidate * P + particle
        part = torch.arange(r, device=dev) % p
        mem_t0, mem_w0 = part % m_t, part % m_w
        left = float((deadline - now) if deadline is not None else cfg.default_horizon_s)
        f_scale = max(float(objective_scale), 1.0)
        feas_idx = torch.nonzero(feas).squeeze(-1)
        n_feas = int(feas_idx.numel())
        max_rows = int(cfg.imagination_max_rows) or (32768 if dev.type == "cuda" else 8192)
        max_rows = max(max_rows, n)
        if evaluations is not None and len(evaluations) == n:
            cost = torch.tensor([float((e.extras or {}).get("net_cost", 0.0) or 0.0) for e in evaluations],
                                dtype=F64, device=dev)
        else:
            cost = torch.zeros(n, dtype=F64, device=dev)
        self.last_search = {"levels": []}

        def step(z: Tensor, dts: Tensor, b: Tensor, mt: Tensor) -> Tuple[Tensor, Tensor]:
            out = heads.transition(z, dts, b)
            if cfg.imagination_aleatoric:
                nxt = out.sample(generator, member=mt)
            else:
                nxt = out.member_mean[mt.to(out.member_mean.device), torch.arange(z.shape[0], device=dev)]
            return nxt, out.epistemic_var.sqrt().mean(-1)

        def values(z_all: Tensor, hz: Tensor, mw: Tensor) -> Tensor:                         # [rows, N] float64
            if goal_mode:
                return predictor.goal_values(z_all, member=mw).to(dev)
            rows = z_all.shape[0]
            ev = heads.evaluator.forward_candidates(z_all, e_g.expand(rows, d), hz.clamp_min(1.0),
                                                    scale=torch.full((rows,), f_scale, device=dev))
            return ev.p_success.double() - cost[None]

        def search(parts: Tensor, elapsed: Tensor, mt: Tensor, mw: Tensor, depth: int):
            """Best imagined continuation of every row over ``depth`` more actions: (value [rows], first action of
            it [rows], summed transition epistemic std [rows], elapsed time at its end [rows], belief at its end)."""
            rows = parts.shape[0]
            ar = torch.arange(rows, device=dev)
            z_all = model.decision_latent(parts, e_a.expand(rows, n, d), e_g.expand(rows, d))       # [rows, N, D]
            v = values(z_all, (left - elapsed).to(F64), mw)
            fm = feas[None].expand(rows, n)
            ninf = torch.full_like(v, float("-inf"))
            if goal_mode:                             # additive progress; an action ending after the deadline adds 0
                fits = (elapsed[:, None] + dt[None, :]) <= max(left, 0.0) if deadline is not None else fm
                gain = torch.where(fits, v, torch.zeros_like(v))
            else:                                     # success probabilities: read at the end only
                gain = torch.zeros_like(v)
            if depth <= 1:
                tot = torch.where(fm, gain if goal_mode else v, ninf)
                self.last_search["levels"].append({"rows": rows, "children": 1})
                return (tot.max(-1).values, tot.argmax(-1), torch.zeros(rows, device=dev), elapsed, parts)
            mode = cfg.imagination_continuation
            if mode == "policy":
                kids = heads.policy(z_all, fm, parts).log_probs.argmax(-1, keepdim=True)
            elif mode == "greedy":
                kids = torch.where(fm, v, ninf).argmax(-1, keepdim=True)
            else:                                     # search: every feasible action while the row budget allows,
                w = max(1, min(n_feas, max_rows // max(rows * n, 1)))     # else the greedy one + a random subset
                if w >= n_feas:
                    kids = feas_idx[None].expand(rows, n_feas)
                else:
                    u = torch.rand(rows, n, generator=generator, dtype=torch.float64).to(dev)
                    u = torch.where(fm, u, torch.full_like(u, -1.0))
                    u[ar, torch.where(fm, v, ninf).argmax(-1)] = 2.0
                    kids = u.topk(w, -1).indices
            wk = kids.shape[1]
            self.last_search["levels"].append({"rows": rows, "children": wk})
            rr = ar.repeat_interleave(wk)
            act = kids.reshape(-1)
            nb, e1 = step(z_all[rr, act], dt[act], parts[rr], mt[rr])
            sub_v, _, sub_e, sub_el, sub_b = search(nb, elapsed[rr] + dt[act], mt[rr], mw[rr], depth - 1)
            tot = (gain[rr, act] + sub_v).reshape(rows, wk)
            j = tot.argmax(-1)
            pick = ar * wk + j
            return tot.max(-1).values, kids[ar, j], (e1 + sub_e)[pick], sub_el[pick], sub_b[pick]

        first = torch.arange(n, device=dev).repeat_interleave(p)
        parts, epi0 = step(z0[0][first], dt[first], b0.expand(r, d), mem_t0)
        cont, nxt_idx, epi_rest, elapsed, parts = search(parts, dt[first].clone(), mem_t0, mem_w0, steps)
        epi = epi0 + epi_rest
        hz = (left - elapsed).clamp_min(1.0)
        val = heads.evaluator(parts, e_g.expand(r, d), hz, scale=torch.full((r,), f_scale, device=dev)).p_success.double()
        val, view, contv = val.reshape(n, p), parts.reshape(n, p, d), cont.reshape(n, p)
        epiv, elv, nxv = epi.reshape(n, p), elapsed.reshape(n, p), nxt_idx.reshape(n, p)
        # information value of each candidate's imagined long-term value (one-step value + continuation) under the
        # joint particles: the expected amount by which it turns out better than the incumbent (for the incumbent:
        # by which the best other candidate does) — what an experiment that settled it would be worth
        if goal_mode:
            q1 = predictor.goal_values(z0, member=None)[0].to(dev)                           # [N] mean one-step value
        else:
            q1 = torch.zeros(n, dtype=F64, device=dev)
        long = torch.where(feas[:, None], q1[:, None] + contv, torch.full_like(contv, float("-inf")))   # [N, P]
        inc = int(long.mean(1).argmax())
        rival = torch.cat([long[:inc], long[inc + 1:]]).max(0).values if n > 1 else long[inc]
        out: List[ImaginedEvaluation] = []
        for i, a in enumerate(cands):
            if not bool(feas[i]) or n < 2:
                kg = 0.0
            else:
                kg = float(((rival if i == inc else long[i]) - long[inc]).clamp_min(0.0).mean())
            nx = nxv[i]
            mode = int(torch.mode(nx).values) if nx.numel() else -1
            out.append(ImaginedEvaluation(
                action_id=a.action_id, index=i, value_mean=float(val[i].mean()),
                value_std=float(val[i].std()) if p > 1 else 0.0,
                epistemic_std=float(epiv[i].mean() / steps),
                latent_std=float(view[i].std(0).mean()) if p > 1 else 0.0, horizon_steps=steps,
                elapsed_s=float(elv[i].mean()), feasible=bool(feas[i]),
                cont_mean=float(contv[i].mean()), cont_std=float(contv[i].std()) if p > 1 else 0.0,
                cont_samples=tuple(float(x) for x in contv[i]), info_value=kg, next_index=mode,
                next_share=float((nx == mode).double().mean()), value_source="q_goal" if goal_mode else "p_success"))
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
