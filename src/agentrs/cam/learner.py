"""CAM learning: three separately logged objectives over the same shared network.

1. Predictive (world model / representation): for every executed transition and every horizon
   (+1 action, within +3 actions, end of task), with the real elapsed time of the window as input:
   Gaussian NLL of observable features and verified progress, BCE of events / termination / success,
   cosine loss of the predicted next latent belief against the (stop-gradient) actual next belief,
   and reconstruction of the current observable features from the belief (anti-collapse). Episodes
   are complete, so every label is observed (matured); nothing unobserved is labelled as failure.
2. Goal-directed RL: PPO over the factored decision π(term)·π(option)·π(action tokens) with GAE
   advantages from the value head; the joint log-prob is re-computed with exactly the recorded
   grammar masks; samples from another policy version are rejected.
3. Goal-value: return regression, P(verified success) BCE, remaining-steps Gaussian NLL.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from agentrs.cam.envs import ENVS
from agentrs.cam.model import CAM
from agentrs.cam.runtime import CamEpisode


def gae(eps: list[CamEpisode], gamma: float, lam: float):
    for e in eps:
        nxt_v, nxt_a = 0.0, 0.0
        for s in reversed(e.steps):
            delta = s.reward + gamma * nxt_v * (0.0 if s.done else 1.0) - s.v_old
            nxt_a = delta + gamma * lam * nxt_a * (0.0 if s.done else 1.0)
            s.adv = nxt_a
            s.ret = nxt_a + s.v_old
            nxt_v = s.v_old


def _fam_masks(model: CAM, fams: list[str]):
    fm = torch.zeros(len(fams), model.cfg.n_feat)
    em = torch.zeros(len(fams), model.cfg.n_events)
    for i, f in enumerate(fams):
        fm[i, :ENVS[f].n_features] = 1
        em[i, :len(ENVS[f].events)] = 1
    return fm.to(model.device), em.to(model.device)


def unroll(model: CAM, eps: list[CamEpisode]):
    """Recompute beliefs with gradients over complete episodes (batched across episodes)."""
    dev = model.device
    T = max(len(e.steps) for e in eps)
    B = len(eps)
    flat = [(i, t) for i, e in enumerate(eps) for t in range(len(e.steps))]
    obs = torch.zeros(len(flat), model.cfg.max_obs, dtype=torch.long)
    for k, (i, t) in enumerate(flat):
        o = eps[i].steps[t].obs
        obs[k, :len(o)] = torch.tensor(o)
    obs = obs.to(dev)
    prev = model.ids([eps[i].steps[t].prev_act for i, t in flat], model.cfg.max_act + 2)
    o_e = model.obs_enc(obs)
    a_e = model.act_enc(prev)
    goal = model.encode_goal(model.ids([[f"FAM_{e.family}"] + e.goal for e in eps], 48))
    el = torch.tensor([eps[i].steps[t].elapsed_prev for i, t in flat], device=dev)
    tf = model.time(__import__("agentrs.cam.model", fromlist=["time_feats"]).time_feats(el))
    pos = {(i, t): k for k, (i, t) in enumerate(flat)}
    b = torch.zeros(B, model.cfg.belief, device=dev)
    beliefs = [None] * len(flat)
    for t in range(T):
        rows = [i for i in range(B) if t < len(eps[i].steps)]
        ks = [pos[(i, t)] for i in rows]
        x = torch.cat([o_e[ks], a_e[ks], goal[rows], tf[ks]], -1)
        nb = model.belief_norm(model.belief_cell(x, b[rows]))
        b = b.clone()
        b[rows] = nb
        for i, k in zip(rows, ks):
            beliefs[k] = nb[rows.index(i)]
    return flat, torch.stack(beliefs), goal


def wm_targets(model: CAM, eps: list[CamEpisode], flat):
    """Matured multi-horizon targets for every (episode, step, horizon)."""
    H = model.cfg.horizons
    rows = []
    for k, (i, t) in enumerate(flat):
        e = eps[i]
        T = len(e.steps)
        for hi, h in enumerate(H):
            end = T - 1 if h == 0 else min(t + h - 1, T - 1)
            win = e.steps[t:end + 1]
            rows.append({"k": k, "hi": hi, "el": float(sum(s.elapsed_h for s in win)),
                         "feat": e.steps[end].feat_after, "ev": np.max(np.stack([s.events for s in win]), 0),
                         "prog": e.steps[end].prog_after, "term": float(e.steps[end].done),
                         "succ": float(e.success), "z_next": (pos_next(flat, k, i, t, T) if h == 1 else -1)})
    return rows


def pos_next(flat, k, i, t, T):
    return k + 1 if t + 1 < T else -1  # flat is ordered by episode then step


def wm_loss(model: CAM, eps, flat, beliefs, sample: Optional[int] = 2048, rng: Optional[random.Random] = None,
            with_grad: bool = True) -> dict:
    rows = wm_targets(model, eps, flat)
    if sample and len(rows) > sample:
        rows = (rng or random).sample(rows, sample)
    dev = model.device
    k = torch.tensor([r["k"] for r in rows], device=dev)
    fams = [eps[flat[r["k"]][0]].family for r in rows]
    fm, em = _fam_masks(model, fams)
    act = model.ids([eps[flat[r["k"]][0]].steps[flat[r["k"]][1]].act_tokens for r in rows], model.cfg.max_act + 2)
    pred = model.predict_effects(beliefs[k], act, torch.tensor([r["el"] for r in rows], device=dev),
                                 torch.tensor([r["hi"] for r in rows], device=dev))
    feat = torch.tensor(np.stack([r["feat"] for r in rows]), device=dev)
    ev = torch.tensor(np.stack([r["ev"] for r in rows]), device=dev)
    prog = torch.tensor([r["prog"] for r in rows], device=dev, dtype=torch.float32)
    term = torch.tensor([r["term"] for r in rows], device=dev)
    succ = torch.tensor([r["succ"] for r in rows], device=dev)
    nll_f = 0.5 * (pred["feat_logvar"] + (feat - pred["feat_mu"]) ** 2 / pred["feat_logvar"].exp())
    L_feat = (nll_f * fm).sum() / fm.sum()
    L_ev = (F.binary_cross_entropy_with_logits(pred["events"], ev, reduction="none") * em).sum() / em.sum()
    L_prog = (0.5 * (pred["prog_logvar"] + (prog - pred["prog_mu"]) ** 2 / pred["prog_logvar"].exp())).mean()
    L_term = F.binary_cross_entropy_with_logits(pred["term"], term)
    tm = term > 0.5
    L_succ = F.binary_cross_entropy_with_logits(pred["success"][tm], succ[tm]) if tm.any() else torch.zeros((), device=dev)
    zi = [(j, r["z_next"]) for j, r in enumerate(rows) if r["z_next"] >= 0]
    if zi:
        a = torch.tensor([j for j, _ in zi], device=dev)
        bt = torch.tensor([z for _, z in zi], device=dev)
        L_z = (1 - F.cosine_similarity(pred["z"][a], beliefs[bt].detach(), dim=-1)).mean()
    else:
        L_z = torch.zeros((), device=dev)
    out = {"feat_nll": L_feat, "event_bce": L_ev, "progress_nll": L_prog, "term_bce": L_term, "success_bce": L_succ,
           "latent_cos": L_z}
    if not with_grad:
        with torch.no_grad():
            p_ev = torch.sigmoid(pred["events"])
            base = (ev * em).sum(0) / em.sum(0).clamp(min=1)
            out["event_brier"] = ((p_ev - ev) ** 2 * em).sum() / em.sum()
            out["event_brier_baserate"] = ((base[None] - ev) ** 2 * em).sum() / em.sum()
            out["feat_mse"] = ((pred["feat_mu"] - feat) ** 2 * fm).sum() / fm.sum()
            mu = (feat * fm).sum(0) / fm.sum(0).clamp(min=1)
            out["feat_mse_const"] = ((mu[None] - feat) ** 2 * fm).sum() / fm.sum()
            ps = torch.sigmoid(pred["success"][tm]) if tm.any() else torch.zeros(1, device=dev)
            sy = succ[tm] if tm.any() else torch.zeros(1, device=dev)
            out["success_brier"] = ((ps - sy) ** 2).mean()
            out["success_brier_baserate"] = ((sy.mean() - sy) ** 2).mean()
            out["success_ece"] = ece(ps, sy)
            # per horizon
            for hi, h in enumerate(model.cfg.horizons):
                sel = torch.tensor([r["hi"] == hi for r in rows], device=dev)
                if sel.any():
                    out[f"event_brier_h{h or 'end'}"] = ((p_ev[sel] - ev[sel]) ** 2 * em[sel]).sum() / em[sel].sum()
    return out


def ece(p: torch.Tensor, y: torch.Tensor, bins: int = 10) -> torch.Tensor:
    tot = torch.zeros((), device=p.device)
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins + (1e-9 if b == bins - 1 else 0))
        if m.any():
            tot = tot + m.float().mean() * (p[m].mean() - y[m].mean()).abs()
    return tot


def policy_terms(model: CAM, eps, flat, beliefs, goal):
    dev = model.device
    rows = [eps[i].steps[t] for i, t in flat]
    gi = torch.tensor([i for i, _ in flat], device=dev)
    g = goal[gi]
    prev_opt = torch.tensor([s.prev_opt for s in rows], device=dev)
    opt = torch.tensor([s.opt for s in rows], device=dev)
    logits, beta = model.option_dists(beliefs, g, prev_opt)
    first = torch.tensor([t == 0 for _, t in flat], device=dev)
    term = torch.tensor([s.term_sample for s in rows], device=dev).float()
    new = torch.tensor([s.new_opt for s in rows], device=dev)
    lpb = F.logsigmoid(beta)
    lpnb = F.logsigmoid(-beta)
    t_lp = torch.where(first, torch.zeros_like(beta), term * lpb + (1 - term) * lpnb)
    olp = torch.log_softmax(logits.float(), -1)
    o_lp = torch.where(new, olp.gather(-1, opt[:, None]).squeeze(-1), torch.zeros_like(beta))
    a_lp, a_ent = model.action_logprob(beliefs, g, opt, [s.act_ids for s in rows], [s.allowed for s in rows])
    o_ent = -(olp.exp() * olp).sum(-1)
    v = model.estimate_value(beliefs, g)
    return t_lp + o_lp + a_lp, a_ent + 0.1 * o_ent, v


def update(model: CAM, opt: torch.optim.Optimizer, eps: list[CamEpisode], version: str, cfg: dict,
           rng: random.Random) -> dict:
    eps = [e for e in eps if e.version == version and not e.planned]   # on-policy only
    gae(eps, cfg["gamma"], cfg["lam"])
    advs = np.array([s.adv for e in eps for s in e.steps])
    mu, sd = advs.mean(), advs.std() + 1e-8
    stats = {k: 0.0 for k in ("pg", "entropy", "value", "succ_bce", "steps_nll", "clipfrac", "kl", "mismatch",
                              "feat_nll", "event_bce", "progress_nll", "term_bce", "success_bce", "latent_cos", "recon",
                              "grad_norm")}
    n_mb = 0
    model.train()
    for epoch in range(cfg["epochs"]):
        order = list(range(len(eps)))
        rng.shuffle(order)
        for s0 in range(0, len(order), cfg["mb_episodes"]):
            mb = [eps[i] for i in order[s0:s0 + cfg["mb_episodes"]]]
            flat, beliefs, goal = unroll(model, mb)
            rows = [mb[i].steps[t] for i, t in flat]
            new_lp, ent, v = policy_terms(model, mb, flat, beliefs, goal)
            old = torch.tensor([s.logp_old for s in rows], device=model.device)
            A = torch.tensor([(s.adv - mu) / sd for s in rows], device=model.device, dtype=torch.float32)
            R = torch.tensor([s.ret for s in rows], device=model.device, dtype=torch.float32)
            ratio = torch.exp(new_lp - old)
            if epoch == 0 and n_mb == 0:  # before any update: recorded vs re-scored log-probs (exactness)
                stats["mismatch"] = float((new_lp.detach() - old).abs().mean())
            L_pg = -torch.min(ratio * A, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * A).mean()
            L_ent = -ent.mean()
            L_v = F.mse_loss(v["ret"], R)
            succ = torch.tensor([float(mb[i].success) for i, _ in flat], device=model.device)
            L_s = F.binary_cross_entropy_with_logits(v["succ_logit"], succ)
            rem = torch.tensor([float(len(mb[i].steps) - t) for i, t in flat], device=model.device)
            L_steps = (0.5 * (v["steps_logvar"] + (rem / 10 - v["steps_mu"]) ** 2 / v["steps_logvar"].exp())).mean()
            loss = cfg.get("pg_coef", 1.0) * (L_pg + cfg["ent_coef"] * L_ent) + cfg["v_coef"] * (L_v + 0.5 * L_s + 0.1 * L_steps)
            if cfg["wm_coef"] > 0:
                W = wm_loss(model, mb, flat, beliefs, sample=cfg["wm_sample"], rng=rng)
                fo = torch.tensor(np.stack([s.feat_obs for s in rows]), device=model.device)
                fams = [mb[i].family for i, _ in flat]
                fm, _ = _fam_masks(model, fams)
                L_rec = (((model.recon(beliefs) - fo) ** 2) * fm).sum() / fm.sum()
                L_wm = W["feat_nll"] + W["event_bce"] + W["progress_nll"] + W["term_bce"] + W["success_bce"] + W["latent_cos"]
                loss = loss + cfg["wm_coef"] * (L_wm + L_rec)
                for kk in ("feat_nll", "event_bce", "progress_nll", "term_bce", "success_bce", "latent_cos"):
                    stats[kk] += float(W[kk])
                stats["recon"] += float(L_rec)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            with torch.no_grad():
                stats["clipfrac"] += float(((ratio - 1).abs() > cfg["clip"]).float().mean())
                kl_mb = float((old - new_lp.detach()).mean())
                stats["kl"] += kl_mb
            stats["pg"] += float(L_pg)
            stats["entropy"] += float(-L_ent)
            stats["value"] += float(L_v)
            stats["succ_bce"] += float(L_s)
            stats["steps_nll"] += float(L_steps)
            stats["grad_norm"] += float(gn)
            n_mb += 1
            if kl_mb > cfg.get("target_kl", 0.05):  # stop this update early: the policy moved enough
                stats["early_stop"] = 1.0
                break
        if stats.get("early_stop"):
            break
    for k in stats:
        if k != "mismatch":
            stats[k] /= max(1, n_mb)
    stats["updates"] = n_mb
    stats["episodes"] = len(eps)
    stats["transitions"] = int(sum(len(e.steps) for e in eps))
    model.eval()
    return stats


@torch.no_grad()
def eval_world_model(model: CAM, eps: list[CamEpisode], rng: random.Random) -> dict:
    """Held-out prediction quality on episodes the model was not trained on (+ collapse diagnostics)."""
    model.eval()
    out = {}
    for s0 in range(0, len(eps), 64):
        mb = eps[s0:s0 + 64]
        flat, beliefs, _ = unroll(model, mb)
        W = wm_loss(model, mb, flat, beliefs, sample=4096, rng=rng, with_grad=False)
        for k, v in W.items():
            out.setdefault(k, []).append(float(v))
        out.setdefault("belief_std", []).append(float(beliefs.std(0).mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}
