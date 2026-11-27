"""Decision-relevant value of information from CAM's own world model and beliefs (meta-reward spec 8.2; plan
2026-11-26 §1 World "VOI: IMPLEMENT", §3 W6 `world.voi.decision_voi`).

    VOI(a) = gamma(a) * E_{y ~ p(y | b, a)} [ max_{a'} Q(b'(a, y), a') ] - max_{a'} Q(b, a') - cost(a)

for every candidate information action a of every environment row, where

* p(y | b, a) is the full outcome predictive the agent acts with: the world-model prior Z^ = WM.prior(R_t, A(a),
  B_t, dt) (mode latent) read by the outcome head, mixed over the belief particles and fused with the Kalman/NIG
  layer (`core.belief_runtime.fused_predictive`, gated fusion on the DOM path);
* b'(a, y) is an **independent clone** of the FastState updated with the imagined outcome exactly as the runtime
  updates beliefs (`BeliefModule.update` with the world-model likelihood features, then the Kalman/NIG update); the
  real FastState is never touched;
* Q(b, a') scores a fixed continuation candidate set: the default `outcome_q` is the one-step imagined utility
  sum_key w_key E[y_key | b, Z^'] of the candidate's predicted observable outcomes (Z^' = WM.prior(R', A(a'), B, dt)
  from the state after the information action, R' = Reason(Z^, [B'; G]) on the DOM path), so the information
  action's own state change (time, resources) is accounted for; `value_q` uses the value head instead, and any
  `q_fn(model, R, fast, goal, cand_tokens) -> [N, n_c]` (e.g. W5's decision evaluator r_psi + Phi + gamma V) plugs in.

The estimator is `selfeval.voi.voi_monte_carlo` evaluated batched over rows, actions and outcome samples, with
common random numbers across the information actions (identical particle / gate / noise draws, so differences
between actions are not sampling noise). `decision_voi_reference` runs `voi_monte_carlo` itself row by row with the
same draws (the batched estimate equals it; tested). Everything runs without gradients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor

from ..interfaces import FastState, OutcomeBatch, TokenSet
from ..selfeval.voi import voi_monte_carlo


@dataclass
class VOIState:
    """Decision-time inputs of `decision_voi` ([B] real environment rows).

    R: the world-model prior input [B,S,D] (the reasoned latent R_t on the DOM path, Z_t otherwise); fast: the real
    FastState (cloned, never modified); goal: GOAL tokens [B,Ng,D] or None; prospective: the outcome queries whose
    imagined values update the belief (OutcomeBatch with keys; e.g. `CamUnifiedModel.prospective_batch`); dt: [B]
    simulated days of the information action (None = the world model's predicted duration); dt_next: [B] days of the
    continuation step (None = dt)."""
    R: Tensor
    fast: FastState
    goal: Optional[TokenSet]
    prospective: OutcomeBatch
    dt: Optional[Tensor] = None
    dt_next: Optional[Tensor] = None


def _rep_tokens(ts: Optional[TokenSet], idx: Tensor) -> Optional[TokenSet]:
    return None if ts is None else ts.select(idx)


def _rep_outcomes(ob: OutcomeBatch, idx: Tensor) -> OutcomeBatch:
    il = [int(i) for i in idx]
    return OutcomeBatch(ob.q.select(idx), ob.lik[idx], ob.n_classes[idx], ob.y[idx], ob.censored[idx],
                        ob.y_mask[idx], ob.channel[idx], None if ob.keys is None else [list(ob.keys[i]) for i in il])


def _reason(model, Z: Tensor, belief: TokenSet, goal: Optional[TokenSet]) -> Tensor:
    from .train_seq import reason_fn_of
    fn = reason_fn_of(model)
    return Z if fn is None else fn(Z, belief, goal)


def outcome_q(weights: dict, transform: Optional[Callable] = None) -> Callable:
    """Q(b, a') = sum_key w_key * u(E[y_key | b, Z^'(a')]) over the candidate's predicted outcomes (fused predictive),
    Z^' = WM.prior(R, A(a'), tokens(b), dt). ``weights`` maps outcome keys (exact) or key prefixes ending in "*" to
    utility weights (observable outcomes only in observation_only mode, e.g. {"num:points@me[points]": 1.0}).
    ``transform`` u (default identity): maps the predictive mean to utility units, e.g. `world._util.symexp` for
    numeric outcomes predicted in symlog space (the payout of the predictive mean; a plug-in, not E[symexp(y)])."""
    def q_fn(model, R: Tensor, fast: FastState, goal, cands: list, ob: OutcomeBatch, dt) -> Tensor:
        from ..core.belief_runtime import predictive_moments
        N = R.shape[0]
        n_c = len(cands[0])
        rows = torch.arange(N, device=R.device).repeat_interleave(n_c)
        flat = [a for c in cands for a in c]
        A = model.tokenizer.action(flat)
        btok = model.beliefs.tokens(fast.select(rows))
        Zc, _ = model.world_model.prior(R[rows], A, btok, None if dt is None else dt[rows], sample=False)
        obr = _rep_outcomes(ob, rows)
        feat = model.outcome_head.features(Zc, obr)
        mean, _ = predictive_moments(model, fast.select(rows), feat, obr)
        if transform is not None:
            mean = transform(mean)
        w = torch.zeros_like(mean)
        for i, keys in enumerate(obr.keys or []):
            for q, k in enumerate(keys):
                if k is None:
                    continue
                for pat, v in weights.items():
                    if k == pat or (pat.endswith("*") and k.startswith(pat[:-1])):
                        w[i, q] = float(v)
                        break
        return (w * mean).sum(-1).view(N, n_c)
    return q_fn


def value_q(model, R: Tensor, fast: FastState, goal, cands: list, ob: OutcomeBatch, dt) -> Tensor:
    """Q(b, a') = V(Reason(Z^'(a'), [B; G]), G): the value head on the reasoned state after the candidate."""
    N = R.shape[0]
    n_c = len(cands[0])
    rows = torch.arange(N, device=R.device).repeat_interleave(n_c)
    A = model.tokenizer.action([a for c in cands for a in c])
    fr = fast.select(rows)
    btok = model.beliefs.tokens(fr)
    g = _rep_tokens(goal, rows)
    Zc, _ = model.world_model.prior(R[rows], A, btok, None if dt is None else dt[rows], sample=False)
    Rc = _reason(model, Zc, btok, g)
    v = model.value_head(Rc, g if g is not None else model.empty_tokens(Rc.shape[0]))
    v = (v.value if hasattr(v, "value") else v).float()
    return v.view(N, n_c)


def sample_imagined_outcomes(model, fast: FastState, feat: Tensor, ob: OutcomeBatch, n_y: int, seed: int
                             ) -> Tensor:
    """n_y outcome draws per row [B, n_y, Q] from the full predictive (particle k ~ w, y ~ p(y | m_k); on fused
    Kalman queries the Kalman expert with probability g). The draws are a deterministic function of ``seed``, so two
    calls with the same seed use common random numbers."""
    from ..core.belief_runtime import fused_predictive
    dev = feat.device
    B, Q = ob.lik.shape
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    fp = fused_predictive(model, fast, feat, ob)
    K = fast.logw.shape[1]
    w = torch.softmax(fast.logw.float(), -1)
    cdf = torch.cumsum(w, -1)
    out = []
    for _ in range(n_y):
        yk = fp.dist.sample(gen).float()                                    # [B,K,Q]
        u = torch.rand(B, 1, generator=gen, device=dev)
        k = torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp(max=K - 1).view(B)
        y = yk[torch.arange(B, device=dev), k]
        ug = torch.rand(B, Q, generator=gen, device=dev)
        eps = torch.randn(B, Q, generator=gen, device=dev)
        if fp.gate is not None:
            take = fp.fused & (ug < fp.gate)
            y = torch.where(take, fp.kal_mean + fp.kal_var.clamp_min(0).sqrt() * eps, y)
        out.append(y)
    return torch.stack(out, 1)


def _updated(model, R: Tensor, fast: FastState, A: TokenSet, Zh: Tensor, ob: OutcomeBatch, ys: Tensor) -> FastState:
    """Clone of ``fast`` updated with imagined outcomes ys [N,Q] of the action A (exactly the runtime's update:
    world-model likelihood features of Zh, particle update, Kalman/NIG update against `kalman_base`)."""
    from ..core.belief_runtime import kalman_base
    from ..outcome_spec import with_targets
    f0 = fast.clone()
    imagined = model.belief_view(with_targets(ob, ys, ob.q.mask.clone()))
    feat = model.outcome_head.features(Zh, imagined)
    f1, _ = model.beliefs.update(f0, R, A, imagined, model.outcome_head, feat=feat)
    if model._kalman_on():
        imc = model.channels_of(f0, imagined)
        f1 = model.kalman_update(f1, feat, imc, kalman_base(model, feat, imc, f0))
    return f1


@torch.no_grad()
def decision_voi(model, state: VOIState, info_actions: Sequence[Sequence], n_y: int = 8, *,
                 next_actions: Sequence[Sequence], q_fn: Optional[Callable] = None, cost: Optional[Tensor] = None,
                 gamma: Optional[Tensor] = None, seed: int = 0) -> dict:
    """VOI of every information action [B, n_a] (module docstring). ``info_actions`` [B] lists of n_a Actions;
    ``next_actions`` [B] lists of n_c continuation candidates (scored by ``q_fn``, default `value_q`); ``cost`` /
    ``gamma`` optional [B, n_a] (resource cost in utility units, discount of the post-information value).
    Returns {"voi" [B,n_a], "base" [B] = max_a' Q(b, a'), "q_after" [B,n_a,n_y] = max_a' Q(b'(a, y_j), a'),
    "y" [B,n_a,n_y,Q] the imagined outcomes, "n_y", "q_base" [B,n_c]}. The real FastState is bit-identical
    afterwards."""
    q_fn = q_fn or value_q
    R, fast, ob = state.R.float(), state.fast, state.prospective
    B = R.shape[0]
    dev = R.device
    n_a = len(info_actions[0])
    if any(len(a) != n_a for a in info_actions) or len(info_actions) != B:
        raise ValueError("info_actions must hold the same number of actions for every row")
    btok = model.beliefs.tokens(fast)
    dt_next = state.dt_next if state.dt_next is not None else state.dt
    with model._autocast():
        q_base = q_fn(model, R, fast, state.goal, list(next_actions), ob, dt_next).float()       # [B,n_c]
        base = q_base.max(-1).values
        voi, q_after, ys_all = [], [], []
        rep = torch.arange(B, device=dev).repeat_interleave(n_y)
        for i in range(n_a):
            A = model.tokenizer.action([info_actions[b][i] for b in range(B)])
            Zh, _ = model.world_model.prior(R, A, btok, state.dt, sample=False)
            feat = model.outcome_head.features(Zh, ob)
            ys = sample_imagined_outcomes(model, fast, feat, model.channels_of(fast, ob), n_y, seed)  # [B,n_y,Q]
            fr = fast.select(rep)
            obr = _rep_outcomes(ob, rep)
            Ar = A.select(rep)
            Zr = Zh[rep]
            f1 = _updated(model, R[rep], fr, Ar, Zr, obr, ys.reshape(B * n_y, -1))
            Rn = _reason(model, Zr, model.beliefs.tokens(f1), _rep_tokens(state.goal, rep))
            nxt = [next_actions[int(b)] for b in rep]
            qa = q_fn(model, Rn, f1, _rep_tokens(state.goal, rep), nxt, obr,
                      None if dt_next is None else dt_next[rep]).float().max(-1).values.view(B, n_y)
            g = 1.0 if gamma is None else gamma[:, i].float()
            c = 0.0 if cost is None else cost[:, i].float()
            voi.append(g * qa.mean(-1) - base - c)
            q_after.append(qa)
            ys_all.append(ys)
    return {"voi": torch.stack(voi, 1), "base": base, "q_after": torch.stack(q_after, 1),
            "y": torch.stack(ys_all, 1), "n_y": n_y, "q_base": q_base}


@torch.no_grad()
def decision_voi_reference(model, state: VOIState, info_actions: Sequence[Sequence], n_y: int = 8, *,
                           next_actions: Sequence[Sequence], q_fn: Optional[Callable] = None,
                           cost: Optional[Tensor] = None, gamma: Optional[Tensor] = None, seed: int = 0) -> Tensor:
    """`selfeval.voi.voi_monte_carlo` row by row with CAM callables (cloned single-row belief updates, the same
    imagined draws as `decision_voi` via the same seed). Returns VOI [B, n_a] (slow; the reference for tests)."""
    q_fn = q_fn or value_q
    R, fast, ob = state.R.float(), state.fast, state.prospective
    B = R.shape[0]
    dev = R.device
    n_a = len(info_actions[0])
    btok = model.beliefs.tokens(fast)
    dt_next = state.dt_next if state.dt_next is not None else state.dt
    draws, Zhs, As = [], [], []
    with model._autocast():
        for i in range(n_a):
            A = model.tokenizer.action([info_actions[b][i] for b in range(B)])
            Zh, _ = model.world_model.prior(R, A, btok, state.dt, sample=False)
            feat = model.outcome_head.features(Zh, ob)
            draws.append(sample_imagined_outcomes(model, fast, feat, model.channels_of(fast, ob), n_y, seed))
            Zhs.append(Zh)
            As.append(A)
        out = torch.zeros(B, n_a, device=dev)
        for b in range(B):
            idx = torch.tensor([b], device=dev)
            fb = fast.select(idx)
            ob_b = _rep_outcomes(ob, idx)
            g_b = _rep_tokens(state.goal, idx)
            dtn = None if dt_next is None else dt_next[idx]

            def sample_outcomes(i, n, b=b):
                return [draws[i][b, j] for j in range(n)]

            def update(i, y, b=b, idx=idx, fb=fb, ob_b=ob_b, g_b=g_b):
                f1 = _updated(model, R[idx], fb, As[i].select(idx), Zhs[i][idx], ob_b, y.view(1, -1))
                return ("after", _reason(model, Zhs[i][idx], model.beliefs.tokens(f1), g_b), f1)

            def q_values(belief, idx=idx, fb=fb, ob_b=ob_b, g_b=g_b, dtn=dtn, b=b):
                Rb, fbb = (R[idx], fb) if belief == "base" else (belief[1], belief[2])
                return q_fn(model, Rb, fbb, g_b, [list(next_actions[b])], ob_b, dtn).float()[0]

            costs = None if cost is None else (lambda i, b=b: float(cost[b, i]))
            gammas = None if gamma is None else (lambda i, b=b: float(gamma[b, i]))
            out[b] = voi_monte_carlo(list(range(n_a)), sample_outcomes, update, q_values, "base", n=n_y,
                                     cost=costs, gamma=gammas).to(dev)
    return out
