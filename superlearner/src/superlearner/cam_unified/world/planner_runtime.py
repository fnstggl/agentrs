"""Built-in belief-aware latent imagination planner of CamUnifiedModel (moved out of model.py by the seam commit,
plan 2026-11-26 §3; owner W6).

`CamUnifiedModel` constructs `LatentImaginationPlanner()` when no planner component is given and calls
`plan(model, PlanRequest) -> PlanResult` from `decide` (cfg.use_imagination). `model.LatentImaginationPlanner` is
the same class (re-exported).

Import rule: model.py imports this module at import time, so this module never imports `..model` at module level;
model-level helpers and record types are imported inside the functions that use them.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch

from ..core import belief_runtime as _belief
from ..interfaces import ActionSource, ImaginedStep, ImaginedTrajectory, TokenSet
from ..outcome_spec import CORE_INDEX, select_outcomes, with_targets

if TYPE_CHECKING:                                    # pragma: no cover
    from ..model import CamUnifiedModel, PlanRequest, PlanResult


class LatentImaginationPlanner:
    """Belief-aware multistep imagination over CAM-generated candidates (spec 7.1-7.2), composed only from the
    contract protocols: n candidate first actions from the ActionGenerator; for each, `n_samples` stochastic futures
    of `horizon` decisions: Z_hat = WorldModel.prior(Z, EncAction(a), B, dt) (stochastic latent), outcomes sampled from
    the particle-conditioned outcome head combined with the Kalman/NIG last layer on bound continuous outcomes, an
    *independent clone* of the FastState updated with the imagined outcomes (BeliefModule.update and the Kalman
    update), the reasoner re-run on the imagined latent, and the next action sampled from the policy conditioned on
    the revised belief. Return = sum_h gamma_h * reward_hat_h + gamma_H * V(R_H) (gamma = exp(-kappa dt)) minus
    unc_coef * (particle disagreement on reward); the real action is the argmax candidate (recorded as
    selection="argmax" with one-hot probs; its logp stays the generator's proposal log-probability). Nothing here
    touches the environment, the archive or the real FastState. Workstream C's planner replaces it via the registry."""

    def plan(self, model: "CamUnifiedModel", req: "PlanRequest") -> "PlanResult":
        from ..model import PlanResult, ShapeError, dist_mean, dist_sample, dist_var, gen_on, randn_like_gen
        cfg = model.cfg
        B, n, H, ns = req.R.shape[0], req.n_candidates, req.horizon, req.n_samples
        dev = req.R.device
        rep = torch.arange(B, device=dev).repeat_interleave(n)                           # candidate rows [B*n]
        ag = model.action_generator
        if req.candidates is not None:
            if len(req.candidates) != B * n:
                raise ShapeError(f"PlanRequest.candidates: {len(req.candidates)} actions for B*n={B * n}")
            cands = list(req.candidates)
        else:
            model._count("action_generator.sample")
            cands = ag.sample(req.R[rep], req.goal_in.select(rep), req.belief.select(rep), req.tools.select(rep),
                              req.obs.select(rep), [req.tool_lists[int(i)] for i in rep],
                              [req.obs_lists[int(i)] for i in rep], temperature=1.0, greedy=False)
        rows = rep.repeat_interleave(ns)                                                  # rollout rows [B*n*ns]
        cand_of_row = torch.arange(B * n, device=dev).repeat_interleave(ns)
        Bb = rows.shape[0]
        Z = req.Z[rows]
        fast = req.fast.select(rows) if (cfg.use_fast_adaptation and req.fast is not None) else None
        kal = model._kalman_on() and fast is not None
        goal_in, goal = req.goal_in.select(rows), req.goal.select(rows)
        tools_t, obs_t = req.tools.select(rows), req.obs.select(rows)
        tool_lists = [req.tool_lists[int(i)] for i in rows]
        obs_lists = [req.obs_lists[int(i)] for i in rows]
        mem = None if req.memory is None else req.memory.select(rows)
        prosp = select_outcomes(req.prospective, rows.tolist())
        r_idx = CORE_INDEX["reward"]
        acts = [cands[int(c)] for c in cand_of_row]
        ret = torch.zeros(Bb, device=dev)
        pen = torch.zeros(Bb, device=dev)
        disc = torch.ones(Bb, device=dev)
        steps, first_out = [], None
        R = req.R[rows]
        # Mechanism C in imagination (W5 change request): branch rows -> their source rows of S_t
        sem = getattr(req, "sem", None)
        p_kw = {"sem": sem, "sem_rows": rows} if sem is not None else {}
        r_kw = {}
        if sem is not None:
            import inspect as _inspect
            rp = _inspect.signature(model.reasoner.forward).parameters
            if "sem" in rp:
                r_kw = {"sem": sem.select(rows) if hasattr(sem, "select") else sem}
        for h in range(H):
            act_tok = model.tokenizer.action(acts)
            dt = model.expected_dt(Z, act_tok, acts, tool_lists)
            belief = model.beliefs.tokens(fast) if fast is not None else model.empty_tokens(Bb)
            model._count("world_model.prior")
            Z_hat, _ = model.world_model.prior(Z, act_tok, belief, dt, sample=True, **p_kw)
            feat = model.outcome_head.features(Z_hat, prosp)
            if fast is not None:
                d = model.outcome_head.dist(feat, prosp, cond=fast.particles)
                mk = dist_mean(d).float()                                               # [Bb,K,Q]
                w = torch.softmax(fast.logw.float(), -1)
                k = torch.multinomial(w.cpu(), 1, generator=req.generator).view(Bb).to(dev)
                ys = dist_sample(d, gen_on(req.generator, dev)).float()[torch.arange(Bb, device=dev), k]   # [Bb,Q]
                base = (w.unsqueeze(-1) * mk).sum(1)
                var = (w.unsqueeze(-1) * (dist_var(d).float() + mk * mk)).sum(1) - base * base
                disagree = ((w * (mk[:, :, r_idx] - base[:, r_idx:r_idx + 1]) ** 2).sum(1)).sqrt()
                mean = base
                kbase = base
                if kal and _belief.fusion_on(model):
                    # gated fusion (DOM): sample the two-expert predictive; Kalman residuals against the
                    # particle-free head mean
                    prosp_c = model.channels_of(fast, prosp)
                    fp = _belief.fused_predictive(model, fast, feat, prosp_c)
                    mean, var = fp.mean(), fp.var()
                    eps = randn_like_gen(mean, req.generator)
                    u = randn_like_gen(mean, req.generator)
                    take_k = fp.fused & (torch.special.ndtr(u) < fp.gate)
                    ys = torch.where(take_k, fp.kal_mean + fp.kal_var.clamp_min(0).sqrt() * eps, ys)
                    kbase = _belief.kalman_base(model, feat, prosp, fast)
                    model._count("imagination.bayes.predict")
                elif kal:
                    prosp_c = model.channels_of(fast, prosp)
                    cm, cv, _, bound = model.kalman_predict(fast, feat, prosp_c)
                    mean, var = model.combine(base, var.clamp_min(0), cm, cv, bound)
                    eps = randn_like_gen(mean, req.generator)
                    ys = torch.where(bound, mean + var.clamp_min(0).sqrt() * eps, ys)
                    model._count("imagination.bayes.predict")
            else:
                d = model.outcome_head.dist(feat, prosp)
                mean, var = dist_mean(d).float(), dist_var(d).float()
                ys = dist_sample(d, gen_on(req.generator, dev)).float()
                disagree = var[:, r_idx].clamp_min(0).sqrt()
            ret = ret + disc * mean[:, r_idx]
            pen = pen + disc * disagree
            disc = disc * torch.exp(-cfg.kappa * dt)
            if h == 0:
                first_out = mean
            imagined = with_targets(prosp, ys, prosp.q.mask.clone())
            if fast is not None:
                model._count("imagination.beliefs.update")
                fast_n, _ = model.beliefs.update(fast, Z, act_tok, model.belief_view(imagined), model.outcome_head)
                if kal:
                    model._count("imagination.bayes.update")
                    fast_n = model.kalman_update(fast_n, feat, model.belief_view(model.channels_of(fast, imagined)),
                                                 kbase)
                fast = fast_n
            Z = Z_hat
            belief = model.beliefs.tokens(fast) if fast is not None else model.empty_tokens(Bb)
            if cfg.use_reasoning:
                R, _ = model.reasoner(Z, TokenSet.cat(belief, mem, goal), model.reason_iters, **r_kw)
            else:
                R = Z
            v = model._value(model.value_head(R, goal))
            if fast is not None:
                wn = torch.softmax(fast.logw.float(), -1)
                ess = 1.0 / (wn * wn).sum(-1)
            else:
                ess = torch.ones(Bb, device=dev)
            steps.append(ImaginedStep(Z.detach(), list(acts), act_tok, mean.detach(), var.detach(), v.detach(),
                                      mean[:, r_idx].detach(), ess, disagree.detach()))
            if h + 1 < H:
                acts = ag.sample(R, goal_in, belief, tools_t, obs_t, tool_lists, obs_lists, temperature=1.0)
        ret = ret + disc * steps[-1].value
        score_row = ret - model.unc_coef * pen
        scores = score_row.view(B, n, ns).mean(-1)                                       # [B,n]
        choice = scores.argmax(-1)
        actions, cand_out, imag_out = [], [], []
        for b in range(B):
            c = int(choice[b])
            a = copy.deepcopy(cands[b * n + c])
            a.source = ActionSource.CAM_PLANNER
            a.logp_parts["select"] = 0.0                       # deterministic given the imagined scores
            a.proposal = {"n": n, "selection": "argmax", "probs": [1.0 if i == c else 0.0 for i in range(n)],
                          "chosen": c, "scores": [float(s) for s in scores[b]],
                          "generator_logp": [float(cands[b * n + i].logp) for i in range(n)]}
            actions.append(a)
            cand_out.append([{"action": cands[b * n + i].to_dict(), "score": float(scores[b, i]),
                              "logp": cands[b * n + i].logp, "source": ActionSource.CAM_PLANNER.value}
                             for i in range(n)])
            items = []
            for i in range(n):
                sl = slice((b * n + i) * ns, (b * n + i + 1) * ns)
                keys = req.prospective.keys[b] if req.prospective.keys else []
                items.append({"candidate": i, "horizon": H, "value": float(scores[b, i]),
                              "return": float(ret[sl].mean()), "model_error": float(pen[sl].mean()),
                              "outcomes": {k: float(first_out[sl, q].mean()) for q, k in enumerate(keys[:8])
                                           if k is not None},
                              "next_tools": [steps[1].action[j].tool for j in range(sl.start, sl.stop)]
                              if H > 1 else []})
            imag_out.append(items)
        traj = ImaginedTrajectory(steps, ret.detach(), rows.detach(), cand_of_row.remainder(n).detach())
        return PlanResult(actions, cand_out, imag_out, scores.detach(), traj)
