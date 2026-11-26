"""Integrated learning objectives of CamUnifiedAgent and the loss assembly of one update (moved out of agent.py by the
seam commit, plan 2026-11-26 §3; owner W7).

    objective_world / objective_adapt / objective_memory / objective_policy (agent, UpdateBatch) -> {"losses",
        "metrics", ["consumed", "consumed_events"]}                    the default objectives (DEFAULT_OBJECTIVES)
    gae_semi_markov(entries, kappa, lam, lp_coef)                      GAE under the semi-Markov discount
    collect_losses(agent, batch) -> (losses, metrics, consumed, consumed_events)
    cap_losses(agent, losses, metrics) -> raw_losses                   per-objective magnitude cap (agent.loss_cap)
    trainer_step(agent, losses, consumed) -> dict                      the injected trainer's step

`CamUnifiedAgent.update()` remains the caller (it samples / tokenises replay, builds the UpdateBatch, calls
`collect_losses`, `cap_losses` and `trainer_step`, then does the provenance bookkeeping); agent.py re-exports every
name here, so `from superlearner.cam_unified.agent import objective_policy` keeps working.

`IntegratedLearner(model, reward_model, label_store, cfg, agent=...)` (W7, plan 2026-11-26 §2 "canonical update") is
the LearnerHook of the canonical loop: `on_step(transition)` every round, `step()` every cfg.update_every rounds,
`meta_step()` every cfg.meta_every updates (delegated to W8's meta trainer). One `step` is:

1. batch: replay windows (world / adaptation) and the fresh decision buffer of the bound agent (or an explicit
   `UpdateBatch`);
2. reward: per decision, by cfg.reward_mode (Exp D conditions, labelled in every UpdateInfo):
   "observation_only" (D0, default of the dom presets) r_t = RewardModelP.reward(R_t, R_{t+1}, G, dt, info_gain,
   cost, ...) - detached, computed under the current psi and theta; no environment reward is read anywhere, and a
   nonzero environment reward in a transition, the fresh buffer or a replay window raises RewardIntegrityError;
   "sparse_verified" (D2, non-compliant labelled baseline) beta_ver * U_ver of the joined LabelStore label on the
   episode's terminal decision; "legacy" (D1, non-compliant reference) the evaluator's verified increment +
   lp_coef * LP (the pre-seam actor reward);
3. routing: each environment's chronological chain is split into segments of equal policy lag; lag-0 segments get
   GAE from the behaviour-time values and PPO-clip, lag>0 segments (survivors of the previous update) get V-trace
   with the current values (`ActorCriticObjective`; PPO ratios are never applied to stale data);
4. losses (plan §2 loss / parameter table): world (W6 world_loss), adapt (W6 adapt_loss), actor and value (V_int),
   pred (RewardModelP.losses on the LabelStore labels), extra objectives (e.g. W5's evaluator loss);
5. per-objective gradients into every named DOM-08 group (`grad_probe`), probe-calibrated loss balancing on the core
   (`balance.LossBalancer`: world share <= 0.45, actor share >= 0.1), per-group clipping and role optimisers
   (`trainer.CombinedTrainer`: slow = model + predictors, meta = psi, which no slow loss may touch);
6. further PPO epochs on the same batch (actor + value only) until cfg.ppo_epochs or KL > 1.5 * cfg.target_kl;
7. the agent's bookkeeping (version bump, trace / archive provenance, survivors for the next update's bootstrap).

Import rule: agent.py imports this module at import time, so this module never imports `..agent` at module level.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
import random
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import torch

from ..interfaces import TokenSet
from ..model import StepInputs, acting_mode

if TYPE_CHECKING:                                    # pragma: no cover
    from ..agent import CamUnifiedAgent, UpdateBatch

SUPPORT_SIZES = (0, 1, 2, 4, 8, 16, 32, 64)


# ============================================================================ default objectives (replaceable)

def objective_world(agent: "CamUnifiedAgent", batch: "UpdateBatch") -> dict:
    """L_world from WorldModelP.loss(SequenceBatch[, beliefs, memory]): expects key "world"; scalars -> metrics."""
    if batch.seq is None:
        return {}
    out = agent.model.world_loss(batch.seq, batch.filtered())
    if "world" not in out:
        raise KeyError(f"WorldModelP.loss must return the total under 'world'; got keys {sorted(out)}")
    mets = {f"world/{k}": float(v.detach()) for k, v in out.items()
            if torch.is_tensor(v) and v.numel() == 1 and k != "world"}
    return {"losses": {"world": out["world"]}, "metrics": mets}


def objective_adapt(agent: "CamUnifiedAgent", batch: "UpdateBatch") -> dict:
    """L_adapt with a support size k drawn uniformly from SUPPORT_SIZES restricted to k < T (at least one query
    step remains), on the longer adaptation sequences when available. The drawn k are logged (agent.adapt_k_hist)."""
    cfg = agent.config
    seq = batch.adapt_seq if batch.adapt_seq is not None else batch.seq
    if seq is None or not cfg.use_fast_adaptation:
        return {}
    T = len(seq.steps)
    ks = [k for k in SUPPORT_SIZES if k < T]
    k = agent.py_rng.choice(ks)
    agent.adapt_k_hist[k] += 1
    loss, mets = agent.model.adapt_loss(seq, batch.adapt_filtered(), k)
    return {"losses": {"adapt": loss}, "metrics": {f"adapt/{n}": v for n, v in mets.items()}}


def objective_memory(agent: "CamUnifiedAgent", batch: "UpdateBatch") -> dict:
    """L_memory. The retriever's own sequence objective (RetrieverP.loss(seq, Zs)) when it defines one; otherwise,
    for workstream E's Retriever, its compression objective on the replayed archived sequences: each replayed slice
    is encoded as a memory segment by the *online* event and segment encoders and every event token must be
    identified among all event tokens of the batch from its segment's latent memory tokens (contrastive
    reconstruction, `Retriever.compression_loss`), so the memory tokens keep the information of their events. The
    keys the index serves come from the frozen target snapshot; `CamUnifiedAgent.memory_sync_every` moves the
    target to the trained encoders and migrates the index (E's sync_target + Consolidator.reencode_step)."""
    r = agent.model.retriever
    if batch.seq is None or not agent.config.use_memory or r is None:
        return {}
    if hasattr(r, "loss"):
        return {"losses": {"memory": r.loss(batch.seq, batch.filtered().Zs)}}
    if not (hasattr(r, "compression_loss") and hasattr(r, "encode_records")) or not batch.records:
        return {}
    groups = [list(g) for g in batch.records if len(g) >= 2]
    if not groups:
        return {}
    L = max(len(g) for g in groups)
    dev = agent.device
    ev = r.encode_records([x for g in groups for x in g], target=False).to(dev)
    n, D = len(groups), ev.shape[-1]
    x = torch.zeros(n, L, D, device=dev)
    mask = torch.zeros(n, L, dtype=torch.bool, device=dev)
    dt = torch.zeros(n, L, device=dev)
    p = 0
    for i, g in enumerate(groups):
        t_end = max(float(e.t_sim) for e in g)
        x[i, :len(g)] = ev[p:p + len(g)]
        mask[i, :len(g)] = True
        dt[i, :len(g)] = torch.tensor([float(e.t_sim) - t_end for e in g], device=dev)
        p += len(g)
    loss = r.compression_loss(x, mask, dt)
    return {"losses": {"memory": loss}, "metrics": {"memory/compression": float(loss.detach()),
                                                    "memory/segments": n, "memory/events": int(mask.sum()),
                                                    "memory/chance": math.log(max(1, int(mask.sum())))}}


def gae_semi_markov(entries: list, kappa: float, lam: float, lp_coef: float) -> dict:
    """GAE over one environment's chronological fresh entries [(FreshStep, j)], one episode at a time.
    gamma_t = exp(-kappa * dt_t) (semi-Markov); r_t = reward + lp_coef * LP_t. The last entry of the current episode
    (no successor yet, not done) only bootstraps; a non-final episode's undone last entry (forced reset) is dropped.
    Returns {(id(step), j): (advantage, return)}."""
    out = {}
    episodes: list = []
    for st, j in entries:
        if not episodes or episodes[-1][0] != st.episode_ids[j]:
            episodes.append((st.episode_ids[j], []))
        episodes[-1][1].append((st, j))
    for _, seq in episodes:
        gae_next, v_next = 0.0, 0.0
        for idx in range(len(seq) - 1, -1, -1):
            st, j = seq[idx]
            done = bool(st.done[j])
            v = float(st.value_old[j])
            if idx == len(seq) - 1 and not done:
                gae_next, v_next = 0.0, v       # bootstrap-only (or truncated) entry
                continue
            gamma = math.exp(-kappa * float(st.dt[j]))
            r = float(st.reward[j]) + lp_coef * float(st.lp[j])
            nonterm = 0.0 if done else 1.0
            delta = r + gamma * v_next * nonterm - v
            gae = delta + gamma * lam * nonterm * gae_next
            out[(id(st), j)] = (gae, gae + v)
            gae_next, v_next = gae, v
    return out


def objective_policy(agent: "CamUnifiedAgent", batch: "UpdateBatch") -> dict:
    """Actor + value on the fresh on-policy buffer (spec 12.3), with the reasoner's compute cost (architecture §4) and
    the entropy bonus inside the actor term. Behaviour log-probabilities are recomputed through the full step (tokenizer -> core -> reasoner ->
    option policy -> action generator) from stored inputs, so gradients reach every one of them.
    Routing by behaviour provenance: decoder-sampled decisions use PPO-clip ratios against their exact behaviour logp;
    planner-selected decisions (argmax over imagined scores; the selection probability is not tractable) use
    advantage-weighted regression, -exp(A / beta) log pi(a), which needs no behaviour probability."""
    cfg = agent.config
    per_env: dict = {}
    for st in batch.fresh:
        for j, r in enumerate(st.rows):
            per_env.setdefault(r, []).append((st, j))
    lp_coef = cfg.lp_coef if cfg.use_learned_exploration else 0.0
    targets = {}
    for r, entries in per_env.items():
        targets.update(gae_semi_markov(entries, cfg.kappa, cfg.gae_lambda, lp_coef))
    chosen = [(st, [j for j in range(len(st.rows)) if (id(st), j) in targets]) for st in batch.fresh]
    chosen = [(st, js) for st, js in chosen if js]
    if not chosen:
        return {}
    subs = [st.subset(js) for st, js in chosen]
    inp = StepInputs.cat([s.inputs for s in subs])
    actions = [a for s in subs for a in s.actions]
    dev = agent.device
    adv = torch.tensor([targets[(id(st), j)][0] for st, js in chosen for j in js], device=dev)
    ret = torch.tensor([targets[(id(st), j)][1] for st, js in chosen for j in js], device=dev)
    logp_old = torch.cat([s.logp_old for s in subs]).to(dev)
    planner = torch.cat([s.planner for s in subs]).to(dev).bool()
    lag = torch.tensor([agent.version - s.version for s in subs for _ in s.rows], dtype=torch.float, device=dev)
    out = agent.model.policy_evaluate(inp, actions)
    logp, ent, value = out["logp"], out["entropy"], out["value"]
    a_n = (adv - adv.mean()) / (adv.std() + 1e-8) if adv.numel() > 1 else adv
    ratio = torch.exp(logp - logp_old)
    clipped = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip)
    ppo = -torch.min(ratio * a_n, clipped * a_n)
    w = torch.exp(torch.clamp(a_n / agent.awr_beta, max=math.log(agent.awr_max_weight))).detach()
    awr = -w * logp
    actor = torch.where(planner, awr, ppo).mean() - cfg.entropy_coef * ent.mean()
    vh = agent.model.value_head
    if out.get("value_logits") is not None and hasattr(vh, "loss"):
        vloss = vh.loss(out["value_logits"], ret).mean()
    else:
        vloss = ((value - ret) ** 2).mean()
    reason_cost = None
    if out.get("ponder") is not None and cfg.use_reasoning and cfg.adaptive_reasoning:
        # the reasoner's compute penalty is part of the policy objective (keys stay those of cfg.loss_weights,
        # which D's CombinedTrainer requires); reported separately as policy/reason_compute
        reason_cost = cfg.reason_compute_cost * out["ponder"].mean()
        actor = actor + reason_cost
    losses = {"actor": actor, "value": vloss}
    dev_lp = (logp.detach() - logp_old).abs()
    lag0 = lag == 0
    ppo_rows = ~planner
    mets = {"policy/n": int(adv.numel()), "policy/n_ppo": int(ppo_rows.sum()), "policy/n_awr": int(planner.sum()),
            "policy/ratio_mean": float(ratio.detach()[ppo_rows].mean()) if bool(ppo_rows.any()) else float("nan"),
            "policy/ratio_maxdev": float((ratio.detach()[ppo_rows] - 1).abs().max()) if bool(ppo_rows.any())
            else float("nan"),
            "policy/logp_maxdev_lag0": float(dev_lp[lag0].max()) if bool(lag0.any()) else float("nan"),
            "policy/awr_w_mean": float(w[planner].mean()) if bool(planner.any()) else float("nan"),
            "policy/approx_kl": float((logp_old - logp.detach()).mean()), "policy/entropy": float(ent.detach().mean()),
            "policy/adv_mean": float(adv.mean()), "policy/return_mean": float(ret.mean()),
            "policy/max_version_lag": float(lag.max()) if lag.numel() else 0.0,
            "policy/memory_rebuilt": bool(out.get("memory_rebuilt"))}
    if reason_cost is not None:
        mets["policy/reason_compute"] = float(reason_cost.detach())
        mets["policy/ponder_mean"] = float(out["ponder"].detach().mean())
    return {"losses": losses, "metrics": mets, "consumed": [d for s in subs for d in s.decision_ids],
            "consumed_events": [(e, v) for s in subs for e, v in zip(s.event_ids or [], s.env_ids or [])]}


DEFAULT_OBJECTIVES = {"world": objective_world, "adapt": objective_adapt, "memory": objective_memory,
                      "policy": objective_policy}


# ============================================================================ loss assembly of one update

def collect_losses(agent: "CamUnifiedAgent", batch: "UpdateBatch") -> tuple:
    """Run every objective of `agent.objectives` on `batch`. A non-finite loss raises; a loss without gradient (e.g.
    a replay window too short for the objective) is skipped and counted under metrics["skipped/<key>"].
    Returns (losses {key: tensor}, metrics, consumed decision ids, consumed (event id, env id) pairs)."""
    losses, metrics, consumed, consumed_events = {}, {}, [], []
    for name, fn in agent.objectives.items():
        out = fn(agent, batch)
        if not out:
            continue
        for k, v in out.get("losses", {}).items():
            if not torch.is_tensor(v) or not torch.isfinite(v).all():
                raise FloatingPointError(f"objective {name!r} produced a non-finite loss {k!r}: {v}")
            if not v.requires_grad:          # e.g. a replay window too short for the objective: no signal
                metrics[f"skipped/{k}"] = metrics.get(f"skipped/{k}", 0) + 1
                continue
            losses[k] = v
        metrics.update(out.get("metrics", {}))
        consumed += out.get("consumed", [])
        consumed_events += out.get("consumed_events", [])
    return losses, metrics, consumed, consumed_events


def cap_losses(agent: "CamUnifiedAgent", losses: dict, metrics: dict) -> dict:
    """Magnitude cap per objective (agent.loss_cap; None = off): an objective whose running magnitude m_k (EMA of
    |L_k|, agent.loss_mag) exceeds the cap enters the combined loss as L_k * cap / m_k (in place in `losses`; scales
    logged as metrics["balance/<key>"]). Returns the raw (uncapped) loss values {key: float}."""
    raw_losses = {k: float(v.detach()) for k, v in losses.items()}
    if agent.loss_cap is not None:
        for k, v in list(losses.items()):
            m = abs(raw_losses[k])
            prev = agent.loss_mag.get(k)
            agent.loss_mag[k] = m if prev is None else agent.loss_cap_beta * prev + (1 - agent.loss_cap_beta) * m
            sc = min(1.0, float(agent.loss_cap) / max(agent.loss_mag[k], 1e-12))
            metrics[f"balance/{k}"] = sc
            if sc < 1.0:
                losses[k] = v * sc
    return raw_losses


def trainer_step(agent: "CamUnifiedAgent", losses: dict, consumed: list) -> dict:
    """agent.trainer.step(losses[, model=]) -> dict (an object result is converted with to_dict())."""
    fn = agent.trainer.step
    names = set()
    try:
        names = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        pass
    kw = {}
    if "model" in names:
        kw["model"] = agent.model
    # `consumed` is not passed: D's CombinedTrainer mutates record objects there, while this agent links the
    # consumed decisions itself (recorder.link_updates with the trainer's update id, update_log)
    out = fn(losses, **kw)
    if hasattr(out, "to_dict"):
        out = out.to_dict()
    return dict(out or {})


# ============================================================================ the integrated learner (W7)

# Exp D reward conditions (meta-reward spec, Experiment D) selectable through cfg.reward_mode
REWARD_CONDITIONS = {
    "observation_only": ("D0", "meta-learned intrinsic reward R_psi, observation-only (compliant default)"),
    "sparse_verified": ("D2", "sparse verified terminal utility beta_ver * U_ver (non-compliant labelled baseline)"),
    "legacy": ("D1", "legacy shaped evaluator reward: verified increment + lp_coef * LP (non-compliant reference)"),
}
OBSERVATION_ONLY = "observation_only"


class RewardIntegrityError(RuntimeError):
    """An environment reward reached the learner (or the agent) in reward_mode "observation_only"."""


@dataclass
class _Node:
    """One recorded decision of the learner batch (a row of a FreshStep)."""
    st: object                 # agent.FreshStep
    j: int
    env_row: int
    episode: str
    lag: int
    done: bool
    planner: bool


@dataclass
class PolicyBatch:
    """The decision part of one update: nodes (recomputed every epoch), segments (Trajectory rows) and targets."""
    nodes: list
    inputs: StepInputs
    actions: list
    traj: object               # learn.trajectory.Trajectory over the segments
    col: torch.Tensor          # [S, T] node index of each step (0 at padding)
    boot: torch.Tensor         # [S] node index of the bootstrap state (-1: terminal / none)
    reward_info: dict
    prep: object = None
    consumed: list = field(default_factory=list)
    consumed_events: list = field(default_factory=list)
    # plan §2 L_eval: one-step target of the decision evaluator Q(R, A_exec) for every trainable node,
    # r_t + Gamma_t * (1 - done_t) * V_int(next) with behaviour-time values (detached; NaN = not trainable)
    eval_target: Optional[torch.Tensor] = None


@dataclass
class LabelBatch:
    """What `RewardModelP.losses` receives from the learner (until W8's label_join supplies its own batch):
    every verified label in the LabelStore (`labels`) and the ones new since the previous update (`new`), the fresh
    decisions of this update (`events`: event / decision / env / agent episode id, the learner's episode ordinal,
    done, time), their detached reasoned latents `R` [N,S,D] and goal tokens (`goal`), and `encode(idx)` -> (R, goal)
    recomputed WITH gradient for the given event indices (so predictor losses reach the core)."""
    labels: list
    new: list
    events: list
    R: Optional[torch.Tensor] = None
    goal: Optional[TokenSet] = None
    encode: Optional[Callable] = None


class _ToolEntropyCapture:
    """Records the per-part entropies of `ActionGenerator.log_prob` during `CamUnifiedModel.policy_evaluate`, which
    returns only the summed entropy. Installs an instance-level `log_prob` that returns exactly what the class method
    returns (the sum of `log_prob_parts`) and keeps the parts; only for the semantic ActionGenerator, whose log_prob is
    that sum. (Change request to W5: return "entropy_parts" from policy_evaluate, which makes this unnecessary.)"""

    def __init__(self, ag):
        self.ag = ag
        self.parts = None
        self.on = (ag is not None and hasattr(ag, "log_prob_parts")
                   and getattr(type(ag).log_prob, "__qualname__", "") == "ActionGenerator.log_prob")

    def __enter__(self):
        if self.on:
            ag, cap = self.ag, self

            def log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, actions, *, exact_rows=None,
                         temperature=None):
                lp, en = ag.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, actions,
                                           exact_rows=exact_rows, temperature=temperature)
                cap.parts = en
                return sum(lp.values()), sum(en.values())
            ag.log_prob = log_prob
        return self

    def __exit__(self, *exc):
        if self.on and "log_prob" in self.ag.__dict__:
            del self.ag.__dict__["log_prob"]
        return False


def info_gain(f0, f1) -> Optional[torch.Tensor]:
    """Actor-visible information gain [B] between the belief states before two consecutive decisions. Uses W6's
    `core.belief_runtime.belief_change(fast_prev, fast_new)` (kalman_ig + KL of the mixture weights) when it exists;
    until then a fallback on the same FastState fields: KL(w_{t+1} || w_t) of the particle weights when the particle
    set is unchanged (0 after a resample), plus the Kalman entropy reduction 0.5 * (logdet P_t - logdet P_{t+1})
    summed over channels."""
    if f0 is None or f1 is None:
        return None
    from ..core import belief_runtime as br
    fn = getattr(br, "belief_change", None)
    if callable(fn):
        out = fn(f0, f1)
        if isinstance(out, dict):
            parts = [out[k] for k in ("kalman_ig", "kl_weights") if out.get(k) is not None]
            return sum(p.float() for p in parts) if parts else None
        return out.float() if torch.is_tensor(out) else None
    with torch.no_grad():
        same = (f0.particles.shape == f1.particles.shape) and torch.isclose(
            f0.particles.float(), f1.particles.float()).flatten(1).all(1)
        lw0, lw1 = f0.logw.float(), f1.logw.float()
        kl = (lw1.exp() * (lw1 - lw0)).sum(-1).clamp_min(0)
        kl = torch.where(same, kl, torch.zeros_like(kl))
        ig = kl
        if f0.bayes_P is not None and f1.bayes_P is not None and f0.bayes_P.numel():
            P0, P1 = f0.bayes_P.float(), f1.bayes_P.float()
            eye = 1e-6 * torch.eye(P0.shape[-1], device=P0.device)
            d0 = torch.linalg.slogdet(P0 + eye).logabsdet
            d1 = torch.linalg.slogdet(P1 + eye).logabsdet
            ig = ig + (0.5 * (d0 - d1)).clamp_min(0).flatten(1).sum(1)
        return ig


class IntegratedLearner:
    """The integrated CAM update (module docstring). LearnerHook: `on_step(transition)`, `step(batch=None) -> dict`,
    `meta_step() -> dict`, `state_dict()` / `load_state_dict()`.

    Args:
      model: CamUnifiedModel. reward_model: RewardModelP (default: selfeval.reward_model.RewardModel, the seam stub).
      label_store: envs.labels.LabelStore (trainer-only; the agent never holds it). cfg: CamConfig (default model.cfg).
      agent: the acting CamUnifiedAgent whose fresh buffer / replay / archive this learner consumes (required for
        `step()` without a batch); trainer: a CombinedTrainer over named groups (built from `grad_probe.named_groups`
        when None: one group per DOM-08 group, lr_core for core groups with warmup, lr_heads elsewhere, psi in the
        meta role at lr_psi, per-group clipping per cfg.clip_mode).
      balancer: LossBalancer (default core-share ranges: world (0.2, cfg.world_core_share_max = 0.45 in dom), actor
        (0.1, 0.4), pred (0.1, 0.4)); `balance=False` turns balancing off (the "no loss balancing" ablation).
      probe_warmup / probe_every: the per-objective named-group probe runs on updates 1..probe_warmup and then every
        probe_every updates (default cfg.balance_every or 25); balancing calibrates at the first probe, refines with
        the clipped rule on every warmup probe and then every cfg.balance_every updates.
      epochs / target_kl: policy epochs (cfg.ppo_epochs) and early stop at KL > 1.5 * target_kl (cfg.target_kl).
      extra_objectives: {name: fn(learner, batch, policy_batch) -> {loss_name: tensor}} (e.g. W5's evaluator loss);
        loss weight cfg.loss_weights.get(name, 1.0).
      meta_trainer: W8's meta-reward trainer; `meta_step()` calls `meta_trainer.meta_step(self)`.
      label_batch_fn: fn(learner, batch, policy_batch) -> label batch for reward_model.losses (default `LabelBatch`).
      sparse_beta: beta_ver of the sparse verified condition; episode_base: ordinal of an environment's first episode
        in the label source's numbering (ExperienceOperator: 1, its reset before the first decision counts).
    """

    def __init__(self, model, reward_model=None, label_store=None, cfg=None, *, agent=None, trainer=None,
                 balancer=None, balance: bool = True, reward_mode: Optional[str] = None,
                 epochs: Optional[int] = None, target_kl: Optional[float] = None, max_policy_lag: int = 0,
                 probe_warmup: int = 8, probe_every: Optional[int] = None, log_path: Optional[str] = None,
                 extra_objectives: Optional[dict] = None, meta_trainer=None,
                 label_batch_fn: Optional[Callable] = None, sparse_beta: float = 1.0, episode_base: int = 1,
                 seed: int = 0, amp: Optional[bool] = None, trainer_kw: Optional[dict] = None,
                 objective_kw: Optional[dict] = None):
        from ..selfeval.reward_model import RewardModel
        from .balance import LossBalancer
        from .grad_probe import named_groups
        from .ppo import ActorCriticObjective
        from .trainer import CombinedTrainer
        self.model = model
        self.cfg = cfg if cfg is not None else model.cfg
        cfg = self.cfg
        self.reward_model = reward_model if reward_model is not None else RewardModel(cfg)
        self.label_store = label_store
        self.agent = agent
        self.reward_mode = str(reward_mode or getattr(cfg, "reward_mode", "legacy"))
        if self.reward_mode not in REWARD_CONDITIONS:
            raise ValueError(f"reward_mode {self.reward_mode!r}; known: {sorted(REWARD_CONDITIONS)}")
        rm = self.reward_model
        dev = model.device
        if isinstance(rm, torch.nn.Module) and any(True for _ in rm.parameters()):
            rm.to(dev)
        self.groups = named_groups(model, rm if isinstance(rm, torch.nn.Module) else None)
        live = OrderedDict((g, ps) for g, ps in self.groups.items() if ps)
        if trainer is None:
            roles = {"slow": [g for g in live if g != "psi"]}
            if "psi" in live:
                roles["meta"] = ["psi"]
            kw = dict(stage="continual", amp=(dev.type == "cuda") if amp is None else amp, grad_probe_every=0,
                      groups={g: [p for _, p in ps] for g, ps in live.items()}, roles=roles,
                      loss_weights={"value": float(cfg.loss_weights.get("value", cfg.value_coef))})
            kw.update(trainer_kw or {})
            trainer = CombinedTrainer(cfg, **kw)
        self.trainer = trainer
        if balancer is None and balance:
            wmax = float(cfg.world_core_share_max) if cfg.world_core_share_max < 1 else 0.45
            tgt = {"world": (min(0.20, wmax / 2), wmax), "actor": (0.10, 0.40), "pred": (0.10, 0.40)}
            balancer = LossBalancer(tgt)
        self.balancer = balancer
        self.epochs = int(epochs if epochs is not None else max(1, int(getattr(cfg, "ppo_epochs", 1) or 1)))
        tk = target_kl if target_kl is not None else (getattr(cfg, "target_kl", 0.0) or None)
        self.target_kl = None if not tk else float(tk)
        okw = dict(max_policy_lag=int(max_policy_lag), target_kl=self.target_kl)
        okw.update(objective_kw or {})
        self.objective = ActorCriticObjective.from_config(cfg, **okw)
        self.probe_warmup = int(probe_warmup)
        be = int(getattr(cfg, "balance_every", 0) or 0)
        self.balance_every = be
        self.probe_every = int(probe_every if probe_every is not None else (be or 25))
        self.log_path = log_path
        self.extra_objectives = dict(extra_objectives or {})
        self.meta_trainer = meta_trainer
        self.label_batch_fn = label_batch_fn
        self.sparse_beta = float(sparse_beta)
        self.episode_base = int(episode_base)
        self.py_rng = random.Random(seed)
        self.n_updates = 0
        self.n_meta = 0
        self.adapt_k_hist: Counter = Counter()
        self.cost_by_event: dict = {}
        self.episode_ordinal: dict = {}          # (env_id, agent episode id) -> label-source episode number
        self._episodes_seen: Counter = Counter()  # env_id -> episodes registered
        self._labels_seen = 0
        self._probe_cache: dict = {}
        self.integrity = {"transitions": 0, "env_reward_nonzero": 0, "fresh_reward_nonzero": 0,
                          "replay_reward_nonzero": 0}
        self.history: list = []

    # ------------------------------------------------------------------ LearnerHook
    def attach(self, agent) -> "IntegratedLearner":
        self.agent = agent
        return self

    def on_step(self, transition) -> None:
        """Per round, after agent.observe: integrity check of the reward channel, actor-visible costs by event id,
        agent episode ids -> label-source episode ordinals (for the label join). Never changes the agent."""
        self.integrity["transitions"] += 1
        rew = getattr(transition, "rewards", None)
        if self.reward_mode == OBSERVATION_ONLY and rew is not None:
            bad = [float(r) for r in rew if r is not None and float(r) != 0.0]
            if bad:
                self.integrity["env_reward_nonzero"] += len(bad)
                raise RewardIntegrityError(f"environment rewards {bad[:4]} reached the learner in observation_only")
        for rec in getattr(transition, "records", None) or []:
            if rec is None:
                continue
            res = getattr(rec, "result", None)
            act = getattr(rec, "action", None)
            if res is not None:
                c = float(getattr(res, "cost", 0.0) or 0.0)
                self.cost_by_event[rec.event_id] = c
                if act is not None and getattr(act, "action_id", None):
                    self.cost_by_event[act.action_id] = c
            if act is not None:
                self._register_episode(rec.env_id, rec.episode_id)
        if len(self.cost_by_event) > 200_000:            # bounded: costs are consumed within one or two updates
            for k in list(self.cost_by_event)[:100_000]:
                del self.cost_by_event[k]

    def _register_episode(self, env_id, episode_id) -> int:
        key = (str(env_id), str(episode_id))
        if key not in self.episode_ordinal:
            self.episode_ordinal[key] = self.episode_base + self._episodes_seen[str(env_id)]
            self._episodes_seen[str(env_id)] += 1
        return self.episode_ordinal[key]

    # ------------------------------------------------------------------ batch
    def collect(self):
        """The UpdateBatch of the bound agent (replay windows + the fresh decision buffer), as agent.update builds
        it. Returns (batch, extras) or (None, reason)."""
        from ..agent import UpdateBatch
        a = self.agent
        if a is None:
            raise RuntimeError("IntegratedLearner.step() without a batch needs a bound agent (attach(agent))")
        cfg = self.cfg
        if not cfg.continual_updates:
            return None, "continual_updates=False"
        seq = aux = aseq = aaux = None
        seqs, aseqs = [], []
        consolidated = None
        if cfg.use_memory and a.consolidator is not None and hasattr(a.consolidator, "step"):
            consolidated = a.consolidator.step()
            if hasattr(a.consolidator, "ensure_current"):
                a.consolidator.ensure_current()
        if a.replay is not None and a.archive is not None and a.archive.count() > 0:
            if hasattr(a.replay, "sync"):
                a.replay.sync()
            seqs = a._sample(a.n_seqs, a.seq_len)
            if seqs:
                seq, aux = a.tokenise_sequences(seqs)
            aseqs = a._sample(a.adapt_n_seqs, a.adapt_seq_len) if cfg.use_fast_adaptation else []
            if aseqs:
                aseq, aaux = a.tokenise_sequences(aseqs)
        batch = UpdateBatch(seq, aux, list(a.fresh), self.model, aseq, aaux, records=list(seqs) or None)
        return batch, {"seqs": seqs, "aseqs": aseqs, "consolidated": consolidated}

    # ------------------------------------------------------------------ decision nodes and rewards
    def _nodes(self, fresh: list) -> tuple:
        """Chronological chains per environment row -> nodes and lag-homogeneous segments.
        Returns (nodes, segments [(node indices, boot node index | -1, terminal)], dropped counts)."""
        cur = int(self.model.version)
        per_env: "OrderedDict[int, list]" = OrderedDict()
        for st in fresh:
            for j, r in enumerate(st.rows):
                per_env.setdefault(int(r), []).append((st, j))
        nodes, segs = [], []
        drops = {"forced_reset": 0, "planner": 0}
        for r, entries in per_env.items():
            eps: list = []
            for st, j in entries:
                if not eps or eps[-1][0] != st.episode_ids[j]:
                    eps.append((st.episode_ids[j], []))
                eps[-1][1].append((st, j))
            for e_i, (ep, ents) in enumerate(eps):
                ids = []
                for st, j in ents:
                    planner = bool(st.planner[j]) if st.planner is not None else False
                    nodes.append(_Node(st, j, r, str(ep), cur - int(st.version), bool(st.done[j]), planner))
                    ids.append(len(nodes) - 1)
                last_done = nodes[ids[-1]].done
                if not last_done and e_i < len(eps) - 1:
                    drops["forced_reset"] += 1           # a reset ended the episode without a terminal: no target
                trainable = [k for k in range(len(ids)) if k < len(ids) - 1 or last_done]
                run: list = []

                def close(run_):
                    if not run_:
                        return
                    k_last = run_[-1]
                    boot = ids[k_last + 1] if k_last + 1 < len(ids) else -1
                    segs.append(([ids[k] for k in run_], boot, nodes[ids[k_last]].done))
                for k in trainable:
                    n = nodes[ids[k]]
                    if n.planner:                        # no exact behaviour probability: a chain break
                        drops["planner"] += 1
                        close(run)
                        run = []
                        continue
                    if run and (nodes[ids[run[-1]]].lag > 0) != (n.lag > 0):
                        close(run)
                        run = []
                    run.append(k)
                close(run)
        return nodes, segs, drops

    def _encode_all(self, inputs: StepInputs, grad: bool = False):
        """(R [N,S,D], goal TokenSet, belief TokenSet) of stored decisions recomputed as when acting."""
        m = self.model
        live = StepInputs(**{**inputs.__dict__, "registries": copy.deepcopy(inputs.registries)})
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, acting_mode(m), m._autocast():
            mem = None
            if self.cfg.use_memory and any(inputs.memory_ids):
                mem = m.memory_tokens(inputs.memory_ids, inputs.now)
            enc = m.encode_step(live, memory=mem)
        return enc.R.float(), enc.goal, enc.belief

    def _rewards(self, nodes: list, segs: list, inputs: StepInputs) -> tuple:
        """Actor reward [N] of every node (by reward mode) + diagnostics. Only trainable nodes' values are used."""
        N = len(nodes)
        dev = self.model.device
        r = torch.zeros(N, device=dev)
        info = {"mode": self.reward_mode, "condition": REWARD_CONDITIONS[self.reward_mode][0],
                "label": REWARD_CONDITIONS[self.reward_mode][1]}
        nxt = torch.full((N,), -1, dtype=torch.long)
        for ids, boot, _ in segs:
            chain = ids + ([boot] if boot >= 0 else [])
            for a, b in zip(chain, chain[1:]):
                nxt[a] = b
        trainable = sorted({i for ids, _, _ in segs for i in ids})
        if self.reward_mode == OBSERVATION_ONLY:
            bad = sum(int((n.st.reward[n.j] != 0).item()) for n in nodes)
            if bad:
                self.integrity["fresh_reward_nonzero"] += bad
                raise RewardIntegrityError(f"{bad} fresh decisions carry a nonzero environment reward in "
                                           "observation_only (the canonical loop must pass rewards=None)")
            R, goal, belief = self._encode_all(inputs, grad=False)
            idx = torch.tensor(trainable, dtype=torch.long, device=dev)
            nx = nxt[trainable].to(dev)
            term = nx < 0
            R_next = torch.where(term.view(-1, 1, 1), R[idx], R[nx.clamp_min(0)])
            dt = torch.tensor([float(nodes[i].st.dt[nodes[i].j]) for i in trainable], device=dev)
            ig = []
            for i in trainable:
                f0 = inputs.fast.select([i]) if inputs.fast is not None else None
                k = int(nxt[i])
                f1 = inputs.fast.select([k]) if (inputs.fast is not None and k >= 0) else None
                g = info_gain(f0, f1)
                ig.append(torch.zeros((), device=dev) if g is None else g.reshape(-1)[0].to(dev).float())
            ig = torch.stack(ig) if ig else torch.zeros(0, device=dev)
            cost = torch.tensor([float(self.cost_by_event.get(nodes[i].st.event_ids[nodes[i].j], 0.0))
                                 if nodes[i].st.event_ids is not None else 0.0 for i in trainable], device=dev)
            lp = torch.tensor([float(nodes[i].st.lp[nodes[i].j]) if nodes[i].st.lp is not None else 0.0
                               for i in trainable], device=dev)
            nidx = torch.where(term, idx, nx.clamp_min(0))
            sig = {"terminal": term, "lp": lp}
            tok = getattr(self.model, "tokenizer", None)
            if tok is not None and hasattr(tok, "action"):     # executed-action tokens (e_a of IntrinsicReward)
                with torch.no_grad():
                    sig["action"] = tok.action([nodes[i].st.actions[nodes[i].j] for i in trainable])
            rr = self.reward_model.reward(R[idx], R_next, goal.select(idx) if hasattr(goal, "select") else goal, dt,
                                          info_gain=ig, cost=cost,
                                          belief=belief.select(idx) if hasattr(belief, "select") else None,
                                          belief_next=belief.select(nidx) if hasattr(belief, "select") else None,
                                          intent=None, signals=sig)
            if torch.is_tensor(rr) and rr.requires_grad:
                raise RewardIntegrityError("RewardModelP.reward must be detached (no actor loss reaches psi)")
            rr = torch.as_tensor(rr, device=dev).float().reshape(-1)
            if rr.numel() != len(trainable):
                raise ValueError(f"reward model returned {rr.numel()} rewards for {len(trainable)} decisions")
            r[idx] = rr
            info.update({"info_gain_mean": float(ig.mean()) if ig.numel() else 0.0,
                         "cost_mean": float(cost.mean()) if cost.numel() else 0.0,
                         "psi_version": int(getattr(self.reward_model, "version", 0) or 0),
                         "stub": bool(getattr(self.reward_model, "stub", False)), "_R": (R, goal)})
        elif self.reward_mode == "sparse_verified":
            labels = {}
            if self.label_store is not None:
                for lab in self.label_store.iterate():
                    labels[(str(lab.env_id), int(lab.episode))] = lab
            n_term = n_lab = 0
            for i in trainable:
                n = nodes[i]
                if not n.done:
                    continue
                n_term += 1
                env = n.st.env_ids[n.j] if n.st.env_ids is not None else str(n.env_row)
                ordn = self._register_episode(env, n.episode)
                lab = labels.get((str(env), ordn))
                if lab is not None and lab.utility is not None:
                    r[i] = self.sparse_beta * float(lab.utility)
                    n_lab += 1
            info.update({"terminal": n_term, "labelled": n_lab, "pending_or_missing": n_term - n_lab})
        else:                                             # legacy shaped evaluator reward (pre-seam actor reward)
            lp_coef = self.cfg.lp_coef if self.cfg.use_learned_exploration else 0.0
            for i in trainable:
                n = nodes[i]
                lp = float(n.st.lp[n.j]) if n.st.lp is not None else 0.0
                r[i] = float(n.st.reward[n.j]) + lp_coef * lp
        rt = r[torch.tensor(trainable, dtype=torch.long, device=dev)] if trainable else r[:0]
        info.update({"n": len(trainable), "mean": float(rt.mean()) if rt.numel() else 0.0,
                     "std": float(rt.std()) if rt.numel() > 1 else 0.0,
                     "min": float(rt.min()) if rt.numel() else 0.0, "max": float(rt.max()) if rt.numel() else 0.0})
        return r, info

    def policy_batch(self, fresh: list) -> Optional[PolicyBatch]:
        """Nodes, rewards and the segment Trajectory of the fresh decisions (None without a trainable decision)."""
        from .trajectory import Trajectory
        nodes, segs, drops = self._nodes(fresh)
        if not segs:
            return None
        dev = self.model.device
        for n in nodes:
            if n.st.env_ids is not None:
                self._register_episode(n.st.env_ids[n.j], n.episode)
        inputs = StepInputs.cat([n.st.inputs.select([n.j]) for n in nodes])
        actions = [n.st.actions[n.j] for n in nodes]
        rew, rinfo = self._rewards(nodes, segs, inputs)
        S = len(segs)
        T = max(len(ids) for ids, _, _ in segs)
        col = torch.zeros(S, T, dtype=torch.long, device=dev)
        mask = torch.zeros(S, T, dtype=torch.bool, device=dev)
        boot = torch.full((S,), -1, dtype=torch.long, device=dev)
        done = torch.zeros(S, T, dtype=torch.bool, device=dev)
        for s, (ids, b, term) in enumerate(segs):
            L = len(ids)
            col[s, :L] = torch.tensor(ids, device=dev)
            mask[s, :L] = True
            boot[s] = b
            done[s, L - 1] = bool(term)
        val_old = torch.stack([n.st.value_old[n.j].float() for n in nodes]).to(dev)
        logp_old = torch.stack([n.st.logp_old[n.j].float() for n in nodes]).to(dev)
        dt = torch.tensor([float(n.st.dt[n.j]) for n in nodes], device=dev)
        ver = torch.tensor([int(n.st.version) for n in nodes], dtype=torch.long, device=dev)
        z = torch.zeros(S, 1, device=dev)
        value = torch.cat([val_old[col], z], 1)
        lens = mask.long().sum(1)
        bv = torch.where(boot >= 0, val_old[boot.clamp_min(0)], torch.zeros(S, device=dev))
        value[torch.arange(S, device=dev), lens] = bv
        traj = Trajectory(reward=rew[col], dt=dt[col], done=done, behavior_logp=logp_old[col], model_version=ver[col],
                          mask=mask, value=value,
                          reward_int=rew[col] if self.reward_mode == OBSERVATION_ONLY else None,
                          reward_source={"observation_only": "r_psi", "sparse_verified": "sparse_verified",
                                         "legacy": "legacy_shaped"}[self.reward_mode],
                          reward_version=rinfo.get("psi_version"),
                          event_ids=[[nodes[i].st.event_ids[nodes[i].j] if nodes[i].st.event_ids else None
                                      for i in ids] for ids, _, _ in segs])
        rinfo["drops"] = drops
        cons, cons_ev = [], []
        for ids, _, _ in segs:
            for i in ids:
                n = nodes[i]
                cons.append(n.st.decision_ids[n.j])
                if n.st.event_ids is not None and n.st.env_ids is not None:
                    cons_ev.append((n.st.event_ids[n.j], n.st.env_ids[n.j]))
        from .returns import semi_markov_discount
        nxt = torch.full((len(nodes),), -1, dtype=torch.long, device=dev)
        for ids, b, _ in segs:
            chain = ids + ([b] if b >= 0 else [])
            for a_, b_ in zip(chain, chain[1:]):
                nxt[a_] = b_
        term = torch.tensor([n.done for n in nodes], dtype=torch.bool, device=dev)
        gam = semi_markov_discount(dt, self.cfg.kappa)
        v_next = torch.where(nxt >= 0, val_old[nxt.clamp_min(0)], torch.zeros_like(val_old))
        tgt = rew + gam * (~term).float() * v_next
        trainable = torch.zeros(len(nodes), dtype=torch.bool, device=dev)
        trainable[torch.tensor(sorted({i for ids, _, _ in segs for i in ids}), dtype=torch.long, device=dev)] = True
        eval_target = torch.where(trainable & ((nxt >= 0) | term), tgt, torch.full_like(tgt, float("nan")))
        return PolicyBatch(nodes, inputs, actions, traj, col, boot, rinfo, consumed=cons, consumed_events=cons_ev,
                           eval_target=eval_target.detach())

    def _node_eval(self, pb: PolicyBatch, idx: Optional[list] = None) -> dict:
        """policy_evaluate over the nodes (all, or the listed node indices) -> per-node {"logp", "entropy", "tool"
        (tool-pointer entropy or None), "value", "value_logits" (or None), "ponder" (or None)}."""
        m = self.model
        inp = pb.inputs if idx is None else pb.inputs.select(list(idx))
        acts = pb.actions if idx is None else [pb.actions[i] for i in idx]
        with _ToolEntropyCapture(m.action_generator) as cap:
            out = m.policy_evaluate(inp, acts)
        tool = None
        parts = out.get("entropy_parts")
        if isinstance(parts, dict) and parts.get("tool") is not None:
            tool = parts["tool"]
        elif cap.parts is not None and cap.parts.get("tool") is not None:
            tool = cap.parts["tool"]
        vl = out.get("value_logits")
        return {"logp": out["logp"].float(), "entropy": out["entropy"].float(),
                "tool": None if tool is None else tool.float(), "value": out["value"].float(),
                "value_logits": None if vl is None else vl.float(), "ponder": out.get("ponder")}

    @staticmethod
    def _merge(base: dict, part: dict, idx: list) -> dict:
        """Per-node tensors of `base` (detached) with the rows `idx` replaced by `part` (graph-carrying)."""
        out = {}
        it = None
        for k, v in base.items():
            if k == "ponder" or v is None or part.get(k) is None:
                out[k] = None if k == "ponder" else (None if v is None else v.detach())
                continue
            if it is None:
                it = torch.tensor(list(idx), dtype=torch.long, device=v.device)
            out[k] = v.detach().index_put((it,), part[k].to(v.dtype))
        out["ponder"] = part.get("ponder")
        return out

    def _grid(self, pb: PolicyBatch, ne: dict) -> dict:
        """Per-node evaluations laid out on the segment grid: logp / entropy / tool_entropy [S,T], value [S,T+1]
        (or value_logits [S,T+1,bins] + value_head), the bootstrap state in column len(segment)."""
        m = self.model
        S, T = pb.col.shape
        lens = pb.traj.mask.long().sum(1)
        ar = torch.arange(S, device=pb.col.device)
        res = {"logp": ne["logp"][pb.col], "entropy": ne["entropy"][pb.col], "ponder": ne.get("ponder"),
               "tool_entropy": None if ne.get("tool") is None else ne["tool"][pb.col]}
        vl = ne.get("value_logits")
        vh = m.value_head
        if vl is not None and hasattr(vh, "loss") and hasattr(vh, "value_from_logits"):
            grid = torch.cat([vl[pb.col], torch.zeros_like(vl[:1]).expand(S, 1, -1)], 1).clone()
            bl = vl[pb.boot.clamp_min(0)]
            grid[ar, lens] = torch.where((pb.boot >= 0).view(-1, 1), bl, grid[ar, lens])
            res["value_logits"], res["value_head"] = grid, vh
        else:
            v = ne["value"]
            grid = torch.cat([v[pb.col], torch.zeros(S, 1, device=v.device)], 1).clone()
            grid[ar, lens] = torch.where(pb.boot >= 0, v[pb.boot.clamp_min(0)], torch.zeros(S, device=v.device))
            res["value"] = grid
        return res

    def _microbatches(self, N: int) -> Optional[list]:
        """Node index chunks of cfg.microbatch decisions (None: one full-batch recompute)."""
        mb = int(getattr(self.cfg, "microbatch", 0) or 0)
        if mb <= 0 or N <= mb:
            return None
        return [list(range(i, min(N, i + mb))) for i in range(0, N, mb)]

    def _policy_losses(self, pb: PolicyBatch, idx: Optional[list] = None, base: Optional[dict] = None,
                       ne: Optional[dict] = None) -> dict:
        """Actor + value losses of the policy batch. Full batch (default), or microbatch `idx` over the detached
        full-batch evaluation `base`: the other nodes enter as constants, so the loss value is the full-batch loss
        and its gradient is exactly microbatch idx's share of the full-batch gradient (every per-step term is
        separable and the V-trace / routing targets use detached values); summing the microbatches' gradients gives
        the full-batch gradient."""
        if ne is None:
            ne = self._node_eval(pb) if idx is None else self._merge(base, self._node_eval(pb, idx), idx)
        r = self._grid(pb, ne)
        d = self.objective.loss(pb.traj, pb.prep, r["logp"], r["entropy"], r.get("value"), r.get("value_logits"),
                                r.get("value_head"), tool_entropy=r.get("tool_entropy"))
        with torch.no_grad():                # acting vs recomputed log-probability under identical weights (lag 0)
            lag0 = pb.traj.mask & (pb.prep.lag == 0)
            dev_ = (r["logp"].detach().float() - pb.traj.behavior_logp.float()).abs()
            d["logp_maxdev_lag0"] = float(dev_[lag0].max()) if bool(lag0.any()) else None
            d["logp_meandev_lag0"] = float(dev_[lag0].mean()) if bool(lag0.any()) else None
        ponder = r.get("ponder")
        cfg = self.cfg
        if ponder is not None and cfg.use_reasoning and cfg.adaptive_reasoning:
            frac = 1.0 if idx is None else len(idx) / max(1, len(pb.nodes))
            rc = cfg.reason_compute_cost * ponder.mean() * frac
            d["actor"] = d["actor"] + rc
            d["reason_compute"] = float(rc.detach())
        return d

    # ------------------------------------------------------------------ replay objectives
    def _replay_losses(self, batch) -> tuple:
        losses, mets = OrderedDict(), {}
        cfg = self.cfg
        if batch.seq is not None:
            if self.reward_mode == OBSERVATION_ONLY:
                for st in batch.seq.steps:
                    if st.reward is not None and bool((st.reward != 0).any()):
                        self.integrity["replay_reward_nonzero"] += int((st.reward != 0).sum())
                        raise RewardIntegrityError("a replay window carries environment rewards in observation_only "
                                                   "(world-model targets must not see them)")
            out = self.model.world_loss(batch.seq, batch.filtered())
            if "world" not in out:
                raise KeyError(f"WorldModelP.loss must return the total under 'world'; got keys {sorted(out)}")
            losses["world"] = out["world"]
            mets.update({f"world/{k}": float(v.detach()) for k, v in out.items()
                         if torch.is_tensor(v) and v.numel() == 1 and k != "world"})
        seq = batch.adapt_seq if batch.adapt_seq is not None else batch.seq
        if seq is not None and cfg.use_fast_adaptation:
            T = len(seq.steps)
            ks = [k for k in SUPPORT_SIZES if k < T]
            if ks:
                k = self.py_rng.choice(ks)
                self.adapt_k_hist[k] += 1
                loss, am = self.model.adapt_loss(seq, batch.adapt_filtered(), k)
                losses["adapt"] = loss
                mets.update({f"adapt/{n}": v for n, v in am.items()})
                mets["adapt/k"] = k
        if self.agent is not None and cfg.use_memory:
            out = objective_memory(self.agent, batch)
            if out:
                losses.update(out.get("losses", {}))
                mets.update(out.get("metrics", {}))
        return losses, mets

    def _label_batch(self, batch, pb: Optional[PolicyBatch]):
        if self.label_batch_fn is not None:
            return self.label_batch_fn(self, batch, pb)
        labels = list(self.label_store.iterate()) if self.label_store is not None else []
        new = labels[self._labels_seen:]
        self._labels_seen = len(labels)
        events, R, goal, enc = [], None, None, None
        if pb is not None:
            for n in pb.nodes:
                env = n.st.env_ids[n.j] if n.st.env_ids is not None else None
                events.append({"event_id": n.st.event_ids[n.j] if n.st.event_ids else None,
                               "decision_id": n.st.decision_ids[n.j], "env_id": env, "episode_id": n.episode,
                               "episode": self.episode_ordinal.get((str(env), n.episode)), "done": n.done,
                               "t": float(n.st.inputs.now[n.j]), "dt": float(n.st.dt[n.j])})
            got = pb.reward_info.get("_R")
            if got is not None:
                R, goal = got[0].detach(), got[1]

            def enc(idx, _pb=pb):
                sub = _pb.inputs.select(list(idx))
                R_, g_, _ = self._encode_all(sub, grad=True)
                return R_, g_
        return LabelBatch(labels, new, events, R, goal, enc)

    # ------------------------------------------------------------------ the update
    def _probe_due(self, n: int) -> bool:
        return n <= self.probe_warmup or (self.probe_every > 0 and n % self.probe_every == 0)

    def _balance_due(self, n: int) -> bool:
        return n <= self.probe_warmup or (self.balance_every > 0 and n % self.balance_every == 0)

    def _hard_violation(self, table) -> bool:
        """A probe between balancing points that finds a share far outside its range (above 1.5 x its upper bound or
        below half its lower bound) rebalances at once instead of waiting for the next balance_every update."""
        if self.balancer is None:
            return False
        return bool(self.balancer.violated(table.share(), slack=1.5))

    def base_weights(self, names) -> dict:
        lw = self.cfg.loss_weights
        out = {}
        for k in names:
            if k == "value":
                out[k] = float(self.trainer.weights.get("value", lw.get("value", self.cfg.value_coef)))
            else:
                out[k] = float(lw.get(k, 1.0))
        return out

    def step(self, batch=None) -> dict:
        """One integrated update (module docstring); returns the UpdateInfo dict ({"updated": bool, ...})."""
        from .grad_probe import objective_gradients
        t0 = time.perf_counter()
        extras = {}
        own = batch is None
        if own:
            batch, extras = self.collect()
            if batch is None:
                return {"updated": False, "reason": extras, "version": int(self.model.version)}
        m = self.model
        m.train()
        try:
            losses, mets = self._replay_losses(batch)
            pb = self.policy_batch(batch.fresh) if batch.fresh else None
            pol, mbs, pbase = {}, None, None
            if pb is not None:
                pb.prep = self.objective.prepare(pb.traj, int(m.version))
                mbs = self._microbatches(len(pb.nodes))
                if mbs is None:
                    pol = self._policy_losses(pb)
                else:                        # microbatch 0's graph joins the combined backward, the rest follow
                    with torch.no_grad():
                        pbase = self._node_eval(pb)
                    pol = self._policy_losses(pb, mbs[0], pbase)
                losses["actor"], losses["value"] = pol["actor"], pol["value"]
            lb = self._label_batch(batch, pb)
            pred = self.reward_model.losses(lb) if hasattr(self.reward_model, "losses") else {}
            pred = {k: v for k, v in (pred or {}).items() if torch.is_tensor(v) and v.requires_grad}
            if pred:
                losses["pred"] = sum(pred.values())
                mets.update({f"pred/{k}": float(v.detach()) for k, v in pred.items()})
            for name, fn in self.extra_objectives.items():
                for k, v in (fn(self, batch, pb) or {}).items():
                    if torch.is_tensor(v) and v.requires_grad:
                        losses[k] = v
            skipped = {}
            for k, v in list(losses.items()):
                if not torch.is_tensor(v) or not torch.isfinite(v).all():
                    raise FloatingPointError(f"non-finite loss {k!r}: {v}")
                if not v.requires_grad:
                    skipped[k] = True
                    del losses[k]
            if not losses:
                return {"updated": False, "reason": "no training data yet", "version": int(m.version)}
            self.n_updates += 1
            n = self.n_updates
            base = self.base_weights(losses)
            weights = self.balancer.apply(base) if self.balancer is not None else dict(base)
            used = {k: (self.balancer.factor(k) if self.balancer is not None else 1.0) for k in base}
            weighted = OrderedDict((k, weights[k] * v.float()) for k, v in losses.items() if weights[k] != 0)
            raw = {k: float(v.detach()) for k, v in losses.items()}
            self.trainer.zero_grad()
            probe = self._probe_due(n)
            table, bal = None, {}
            if probe:
                table = objective_gradients(weighted, self.groups, accumulate=True, _sample_cache=self._probe_cache)
            else:
                sum(weighted.values()).backward()
            if mbs is not None and "actor" in weights and "value" in weights:
                for idx in mbs[1:]:
                    d_ = self._policy_losses(pb, idx, pbase)
                    (weights["actor"] * d_["actor"].float() + weights["value"] * d_["value"].float()).backward()
                if table is not None:        # the probe saw microbatch 0 only: scale its norms to the batch (estimate)
                    sc = len(pb.nodes) / len(mbs[0])
                    for k in ("actor", "value"):
                        if k in table.norms:
                            table.norms[k] = {g: v * sc for g, v in table.norms[k].items()}
                            table.core_norm[k] *= sc
            if table is not None and self.balancer is not None and (self._balance_due(n)
                                                                    or self._hard_violation(table)):
                bal = self.balancer.update(n, table, calibrate=n <= self.probe_warmup)
            res = self.trainer.apply("slow")
            epochs = [{"epoch": 0, "kl": pol.get("kl"), "stepped": not res["skipped"]}] if pb is not None else []
            stop_reason = None
            if pb is not None and not res["skipped"] and "actor" in weights and "value" in weights:
                for e in range(1, self.epochs):
                    if mbs is None:
                        d = self._policy_losses(pb)
                    else:
                        with torch.no_grad():
                            pbase = self._node_eval(pb)
                            d = self._policy_losses(pb, ne=pbase)
                    if self.target_kl is not None and d["kl"] > 1.5 * self.target_kl:
                        stop_reason = f"kl {d['kl']:.4f} > 1.5 * target_kl"
                        epochs.append({"epoch": e, "kl": d["kl"], "stepped": False})
                        break
                    self.trainer.zero_grad()
                    if mbs is None:
                        (weights["actor"] * d["actor"].float() + weights["value"] * d["value"].float()).backward()
                    else:
                        for idx in mbs:
                            d_ = self._policy_losses(pb, idx, pbase)
                            (weights["actor"] * d_["actor"].float()
                             + weights["value"] * d_["value"].float()).backward()
                    r_e = self.trainer.apply("slow")
                    epochs.append({"epoch": e, "kl": d["kl"], "stepped": not r_e["skipped"],
                                   "policy_loss": d["policy_loss"], "clipped": r_e["clipped"]})
                    if r_e["skipped"]:
                        break
        finally:
            m.eval()
        if res["skipped"]:
            return {"updated": False, "reason": "non-finite gradient (step skipped)", "version": int(m.version),
                    "losses": raw}
        info = {"updated": True, "update": n, "losses": raw, "weights": weights, "base_weights": base,
                "weighted": {k: float(v.detach()) for k, v in weighted.items()}, "skipped_losses": skipped,
                "grad_norm": res["grad_norm"], "group_grad_norm": res["group_grad_norm"], "clipped": res["clipped"],
                "clip": res["clip"], "clip_rate": self.trainer.clip_rate(), "balance": bal,
                "balance_factors": dict(self.balancer.factors) if self.balancer is not None else {},
                "balance_factors_used": used,
                "probe": bool(probe), "metrics": mets, "epochs": epochs, "stop": stop_reason,
                "microbatches": None if mbs is None else [len(x) for x in mbs],
                "probe_policy": None if (table is None or mbs is None) else
                "actor/value norms from microbatch 0 scaled by N / n_mb0 (estimate)",
                "reward_mode": self.reward_mode}
        if table is not None:
            info["grad_table"] = table.to_dict()
            info["core_share"] = table.share()
            info["dom08"] = table.dom08()
            psi = [g for g in ("psi",) if table.sizes.get(g)]
            info["actor_psi_grad_norm"] = (float(table.norms.get("actor", {}).get("psi", 0.0)) if psi else None)
        if pb is not None:
            rinfo = {k: v for k, v in pb.reward_info.items() if not k.startswith("_")}
            info["reward"] = rinfo
            info["policy"] = {k: v for k, v in pol.items() if not torch.is_tensor(v)}
            info["policy"]["n_nodes"] = len(pb.nodes)
            info["policy"]["n_segments"] = int(pb.col.shape[0])
            info["policy"]["max_lag"] = max(nd.lag for nd in pb.nodes)
        if own:
            info.update(self._commit(batch, extras, pb, info))
        else:
            m.version = m.version + 1
            info["version"] = int(m.version)
        info["integrity"] = dict(self.integrity)
        info["wall_s"] = time.perf_counter() - t0
        self.history.append({k: info[k] for k in ("update", "losses", "weights", "clip_rate") if k in info})
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(info, default=_json_default) + "\n")
        return info

    def _commit(self, batch, extras: dict, pb: Optional[PolicyBatch], info: dict) -> dict:
        """The bound agent's update bookkeeping (as CamUnifiedAgent.update): version bump, trace / archive
        provenance of consumed decisions and replayed events, memory sync, survivors for the next bootstrap."""
        a = self.agent
        a.version = a.version + 1
        sem = self.model.semantic
        lora = list(sem.lora_parameters()) if sem is not None and callable(getattr(sem, "lora_parameters", None)) \
            else []
        if lora and any(self.trainer.group_of(p) is not None for p in lora) and \
                callable(getattr(sem, "mark_updated", None)):
            sem.mark_updated()
        update_id = f"upd-w7-{self.trainer.run_id}-{self.n_updates:08d}"
        seqs, aseqs = extras.get("seqs") or [], extras.get("aseqs") or []
        replay_env, replay_dec = {}, set()
        for s_ in list(seqs) + list(aseqs):
            for r in s_:
                replay_env[r.event_id] = r.env_id
                if r.extra.get("decision_id"):
                    replay_dec.add(r.extra["decision_id"])
        consumed = list(pb.consumed) if pb is not None else []
        consumed_events = list(pb.consumed_events) if pb is not None else []
        linked = sorted(set(consumed) | replay_dec)
        events = sorted(set(consumed_events) | set(replay_env.items()))
        a._link_traces(linked, events, update_id)
        if a.archive is not None and hasattr(a.archive, "mark_used") and events:
            a.provenance_stats["marked_events"] += int(a.archive.mark_used([e for e, _ in events], update_id))
        sync = a._memory_sync()
        keep: dict = {}
        for st in a.fresh:
            for j, r in enumerate(st.rows):
                keep[r] = (st, j)
        survivors = []
        for r, (st, j) in sorted(keep.items()):
            if not bool(st.done[j]):
                sub = st.subset([j])
                survivors.append(sub)
                a._last_fresh[r] = (sub, 0)
            else:
                a._last_fresh.pop(r, None)
        a.fresh = survivors
        entry = {"update_id": update_id, "version": a.version, "losses": info["losses"],
                 "consumed_decisions": consumed, "consumed_events": sorted({e for e, _ in consumed_events}),
                 "replay_events": sorted(replay_env), "replay_decisions": sorted(replay_dec),
                 "adapt_k_hist": {int(k): v for k, v in sorted(self.adapt_k_hist.items())}, "learner": "w7"}
        a.update_log.append(entry)
        return {"update_id": update_id, "version": int(a.version), "n_consumed": len(consumed),
                "n_replay_events": len(replay_env), "n_survivors": len(survivors), "memory_sync": sync,
                "consolidated": extras.get("consolidated")}

    # ------------------------------------------------------------------ meta step / state
    def meta_step(self) -> dict:
        """One outer step on psi, delegated to W8's meta trainer (`meta_trainer.meta_step(learner)`)."""
        if self.meta_trainer is None:
            return {"meta": False, "reason": "no meta trainer attached (W8 MetaRewardTrainer)"}
        out = self.meta_trainer.meta_step(self)
        self.n_meta += 1
        return dict(out or {}, meta=True, n_meta=self.n_meta)

    def state_dict(self) -> dict:
        return {"trainer": self.trainer.state_dict(),
                "balancer": None if self.balancer is None else self.balancer.state_dict(),
                "n_updates": self.n_updates, "n_meta": self.n_meta,
                "adapt_k_hist": dict(self.adapt_k_hist), "episode_ordinal": [[k[0], k[1], v] for k, v in
                                                                             self.episode_ordinal.items()],
                "labels_seen": self._labels_seen, "integrity": dict(self.integrity),
                "py_rng": self.py_rng.getstate(),
                "reward_model": self.reward_model.state_dict() if isinstance(self.reward_model, torch.nn.Module)
                else None,
                "reward_version": getattr(self.reward_model, "version", None)}

    def load_state_dict(self, s: dict) -> None:
        self.trainer.load_state_dict(s["trainer"])
        if self.balancer is not None and s.get("balancer"):
            self.balancer.load_state_dict(s["balancer"])
        self.n_updates, self.n_meta = int(s.get("n_updates", 0)), int(s.get("n_meta", 0))
        self.adapt_k_hist = Counter({int(k): v for k, v in (s.get("adapt_k_hist") or {}).items()})
        self.episode_ordinal = {(a, b): int(v) for a, b, v in (s.get("episode_ordinal") or [])}
        self._episodes_seen = Counter()
        for (env, _), _v in self.episode_ordinal.items():
            self._episodes_seen[env] += 1
        self._labels_seen = int(s.get("labels_seen", 0))
        self.integrity.update(s.get("integrity") or {})
        if s.get("py_rng") is not None:
            st = s["py_rng"]
            self.py_rng.setstate((st[0], tuple(st[1]), st[2]) if isinstance(st, list) else st)
        if s.get("reward_model") is not None and isinstance(self.reward_model, torch.nn.Module):
            self.reward_model.load_state_dict(s["reward_model"])
        if s.get("reward_version") is not None and hasattr(self.reward_model, "version"):
            try:
                self.reward_model.version = int(s["reward_version"])
            except AttributeError:
                pass


def _json_default(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)
