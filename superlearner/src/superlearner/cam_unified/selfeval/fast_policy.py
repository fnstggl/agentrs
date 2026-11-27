"""The CAM decision path as a functional policy over the fast set phi, for the psi meta step (plan §2 "Inner":
functional K-step updates of phi on DETACHED `policy_evaluate` features; core, Qwen, world model and beliefs are never
inner-updated).

* `decision_features(model, inp)`: the actor-visible features of stored decisions (StepInputs), computed exactly as
  `CamUnifiedModel.policy_evaluate` does (acting mode, copied registries) but under no_grad: R [B,S,D], the action
  generator's conditioning [G; OPTION; BUDGET], beliefs, tool and observation tokens, plus the goal tokens G (the
  input of Phi and rho_psi). Nothing label-derived is in them (they are functions of the stored actor inputs).
* `cam_logp_fn`: log pi_phi(a | features) = action_generator.log_prob with phi substituted functionally
  (`torch.func.functional_call` over `metagrad._Call`), so the inner step is differentiable w.r.t. phi and the outer
  loss w.r.t. psi through phi_K(psi).
* `window_batch` / `query_batch`: InnerBatch / QueryBatch over a window of the agent's fresh on-policy buffer
  (agent.fresh: FreshStep per round); rows = environments, columns = rounds.
* `belief_change(fast_prev, fast_new)`: actor-visible information gain of a belief update — KL(w_new || w_prev) of
  the particle weights plus the Kalman entropy reduction 0.5 * sum(log det P_prev - log det P_new) — used as IG_t
  until W6's `core.belief_runtime.belief_change` exists (preferred when importable).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..interfaces import TokenSet
from .metagrad import InnerBatch, QueryBatch, _Call
from .reward_model import detach_ts


def _cat_ts(sets: Sequence[TokenSet]) -> TokenSet:
    from ..model import cat_rows
    return cat_rows(list(sets))


def belief_change(fast_prev, fast_new) -> dict:
    """{"kl_weights" [B], "kalman_ig" [B], "info_gain" [B]} between two FastStates (None -> zeros)."""
    try:
        from ..core.belief_runtime import belief_change as _bc        # W6 (plan §3 W6 exposes it)
        out = dict(_bc(fast_prev, fast_new))
        if "info_gain" not in out:
            out["info_gain"] = out.get("kalman_ig", 0) + out.get("kl_weights", 0)
        return out
    except ImportError:
        pass
    if fast_prev is None or fast_new is None:
        return {"kl_weights": None, "kalman_ig": None, "info_gain": None}
    lw_n = fast_new.logw.float()
    lw_p = fast_prev.logw.float().to(lw_n.device)
    kl = (lw_n.exp() * (lw_n - lw_p)).sum(-1).clamp_min(0.0)
    kig = torch.zeros_like(kl)
    try:
        Pp = fast_prev.bayes_P.double()
        Pn = fast_new.bayes_P.double().to(Pp.device)
        if Pp.numel() and Pp.shape == Pn.shape:
            ld_p = torch.linalg.slogdet(Pp)[1]
            ld_n = torch.linalg.slogdet(Pn)[1]
            kig = (0.5 * (ld_p - ld_n)).sum(-1).float().to(kl.device).clamp_min(0.0)
    except (RuntimeError, AttributeError):
        pass
    return {"kl_weights": kl, "kalman_ig": kig, "info_gain": kl + kig}


@torch.no_grad()
def decision_features(model, inp) -> dict:
    """Detached decision features of StepInputs `inp` (see module docstring)."""
    from ..model import acting_mode, detach_tokens
    regs = copy.deepcopy(inp.registries)
    live = type(inp)(**{**inp.__dict__, "registries": regs})
    with acting_mode(model), model._autocast():
        mem = None
        if model.cfg.use_memory and any(inp.memory_ids):
            mem = model.memory_tokens(inp.memory_ids, inp.now)
        enc = model.encode_step(live, memory=mem)
        opt_tok = model._option_token(inp.option, inp.option_steps) if model.option_policy is not None else None
        goal_in = model._goal_inputs(enc, opt_tok, inp.budget)
        T = model.tokenizer.tools(inp.tools)
    return {"R": enc.R.detach().float(), "goal_in": detach_tokens(goal_in), "belief": detach_tokens(enc.belief),
            "tools": detach_tokens(T), "obs": detach_tokens(enc.obs), "goal": detach_tokens(enc.goal),
            "tool_lists": list(inp.tools), "obs_lists": list(inp.obs_lists)}


def _flat(feats: Sequence[dict]) -> dict:
    out = {"R": torch.cat([f["R"] for f in feats], 0)}
    for k in ("goal_in", "belief", "tools", "obs", "goal"):
        out[k] = _cat_ts([f[k] for f in feats])
    out["tool_lists"] = [t for f in feats for t in f["tool_lists"]]
    out["obs_lists"] = [o for f in feats for o in f["obs_lists"]]
    return out


def _dom_logp(call: _Call, fast_prefixed: dict, dom: list) -> tuple:
    """DOM decision path (W5 Mechanism B): flat (logp, entropy) of the recorded actions under phi through
    `policy_from_features` on each step's detached `policy_features` (intent -> generator -> evaluator Q of the
    recorded candidates -> selection), with phi substituted functionally."""
    lps, ents = [], []
    with sdpa_kernel([SDPBackend.MATH]):
        for feats, acts in dom:
            out = functional_call(call, fast_prefixed, args=(call.m.policy_from_features, feats, list(acts)),
                                  strict=False)
            lps.append(out["logp"].float())
            ents.append(out["entropy"].float())
    return torch.cat(lps, 0), torch.cat(ents, 0)


def cam_logp_fn(call: _Call, fast_prefixed: dict, inputs: dict, actions) -> tuple:
    """(logp [N,T], entropy [N,T]) of the recorded actions under phi (fast_prefixed: "m.<name>" -> tensor)."""
    if inputs.get("dom"):
        lp, ent = _dom_logp(call, fast_prefixed, inputs["dom"])
        return _grid(lp, inputs), _grid(ent, inputs)
    f = inputs["flat"]
    ag = call.m.action_generator
    # the fused / memory-efficient SDPA kernels have no double backward: the meta step runs attention on the math
    # backend (same function, second-order differentiable)
    with sdpa_kernel([SDPBackend.MATH]):
        lp, ent = functional_call(call, fast_prefixed, args=(ag.log_prob, f["R"], f["goal_in"], f["belief"],
                                                             f["tools"], f["obs"], f["tool_lists"], f["obs_lists"],
                                                             list(actions)), strict=False)
    return _grid(lp.float(), inputs), _grid(ent.float(), inputs)


def _grid(v: Tensor, inputs: dict) -> Tensor:
    idx = inputs["index"]                                    # [N, T] long into the flat rows (-1 = none)
    g = v[idx.clamp_min(0).to(v.device)]
    return torch.where(idx.to(v.device) >= 0, g, torch.zeros_like(g))


def cam_reward_fn(reward_model):
    """reward_inputs -> r_psi [N,T] (graph to psi; Phi and features detached)."""
    def fn(flat_R, flat_R_next, flat_goal, flat_dt, flat_ig, flat_done, index):
        r = reward_model.reward_psi(flat_R, flat_R_next, flat_goal, flat_dt, info_gain=flat_ig, done=flat_done)
        return _grid(r, {"index": index})
    return fn


@dataclass
class Window:
    """A rectangular view of FreshSteps: rows = environment ids, columns = rounds."""
    env_ids: list
    index: Tensor            # [N, T] flat row of (env n, round t) or -1
    mask: Tensor             # [N, T] bool
    actions: list            # flat
    flat: dict               # flat features
    dt: Tensor               # flat [M]
    done: Tensor             # flat [M]
    info_gain: Tensor        # flat [M]
    next_index: Tensor       # flat [M] row of the same env's next decision (-1 = none in the window)
    event_ids: list          # flat
    dom: Optional[list] = None   # DOM path: [(model.policy_features(step inputs, step actions), step actions)]


def window(model, steps: Sequence) -> Window:
    """FreshSteps (agent.fresh) -> Window with detached features of every decision."""
    feats = [decision_features(model, s.inputs) for s in steps]
    dom = None
    if bool(getattr(model, "dom_decision", False)) and callable(getattr(model, "policy_features", None)):
        # W5: the DOM policy of a stored decision is intent -> generator -> candidate evaluator -> selection; its
        # fast-set part is recomputed from detached features (`policy_features` / `policy_from_features`)
        with torch.no_grad():
            dom = [(model.policy_features(s.inputs, list(s.actions)), list(s.actions)) for s in steps]
    env_ids = []
    for s in steps:
        for e in (s.env_ids or [str(r) for r in s.rows]):
            if e not in env_ids:
                env_ids.append(e)
    N, T = len(env_ids), len(steps)
    index = torch.full((N, T), -1, dtype=torch.long)
    actions, dts, dones, igs, eids = [], [], [], [], []
    m = 0
    prev_fast = {}
    for t, s in enumerate(steps):
        es = s.env_ids or [str(r) for r in s.rows]
        for j, e in enumerate(es):
            index[env_ids.index(e), t] = m
            actions.append(s.actions[j])
            dts.append(float(s.dt[j]))
            dones.append(float(bool(s.done[j])))
            eids.append(None if s.event_ids is None else s.event_ids[j])
            m += 1
    # info gain of the belief update that followed decision (n, t): fast state at t+1 vs at t
    flat_fast = []
    for t, s in enumerate(steps):
        fs = s.inputs.fast
        for j in range(len(s.rows)):
            flat_fast.append((fs, j))
    nxt = torch.full((m,), -1, dtype=torch.long)
    ig = torch.zeros(m)
    for n in range(N):
        cols = [int(index[n, t]) for t in range(T) if index[n, t] >= 0]
        for a, b in zip(cols[:-1], cols[1:]):
            nxt[a] = b
            fa, ja = flat_fast[a]
            fb, jb = flat_fast[b]
            if fa is not None and fb is not None:
                bc = belief_change(fa.select([ja]), fb.select([jb]))
                if bc.get("info_gain") is not None:
                    ig[a] = float(torch.as_tensor(bc["info_gain"]).reshape(-1)[0])
    dev = feats[0]["R"].device
    index = index.to(dev)
    return Window(env_ids, index, index >= 0, actions, _flat(feats), torch.tensor(dts, device=dev),
                  torch.tensor(dones, device=dev), ig.to(dev), nxt.to(dev), eids, dom)


def _reward_inputs(w: Window) -> dict:
    R = w.flat["R"]
    nxt = w.next_index.to(R.device)
    Rn = torch.where((nxt >= 0)[:, None, None], R[nxt.clamp_min(0)], R)
    done = torch.where(nxt >= 0, w.done, torch.ones_like(w.done))     # last decision of the window: Phi(next) = 0
    return {"flat_R": R, "flat_R_next": Rn, "flat_goal": w.flat["goal"], "flat_dt": w.dt, "flat_ig": w.info_gain,
            "flat_done": done, "index": w.index}


def window_batch(w: Window, kappa: float = 0.0) -> InnerBatch:
    """InnerBatch over the window; per-step discount gamma = exp(-kappa * dt) (SMDP)."""
    g = _grid(torch.exp(-float(kappa) * w.dt.clamp_min(0.0)), {"index": w.index})
    return InnerBatch(policy_inputs={"flat": w.flat, "index": w.index, "dom": w.dom}, actions=w.actions, mask=w.mask,
                      reward_inputs=_reward_inputs(w), gamma=g.to(w.dt.device))


def query_batch(w: Window, utility: dict, baseline: Optional[dict] = None) -> QueryBatch:
    """utility: env_id -> verified utility of the query episode (rows without one are masked out)."""
    keep = [n for n, e in enumerate(w.env_ids) if e in utility and utility[e] is not None]
    if not keep:
        raise ValueError("query_batch: no verified utility for any row")
    idx = w.index[keep]
    U = torch.tensor([float(utility[w.env_ids[n]]) for n in keep])
    b = None if baseline is None else torch.tensor([float(baseline[w.env_ids[n]]) for n in keep])
    return QueryBatch(policy_inputs={"flat": w.flat, "index": idx, "dom": w.dom}, actions=w.actions, mask=idx >= 0,
                      utility=U,
                      baseline=b)
