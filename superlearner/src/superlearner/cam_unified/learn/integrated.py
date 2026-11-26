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
name here, so `from superlearner.cam_unified.agent import objective_policy` keeps working. The planned
`IntegratedLearner(model, reward_model, label_store, cfg).step(batch)` (LearnerHook) is added here by W7.

Import rule: agent.py imports this module at import time, so this module never imports `..agent` at module level.
"""
from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING

import torch

from ..model import StepInputs

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
