"""The learning rule: how experience becomes a loss.  [candidate-editable - this is the Meta-Researcher's main target]

These functions are everything that distinguishes one actor-critic update rule from another. `algo/ppo.py` calls
them and nothing else decides what is optimised, so replacing a function here replaces the learning rule.
All tensors are [B, T] (episodes x decisions) with `valid` marking real decisions.

    advantages(reward, value, done, trunc, valid, cfg)        -> (advantage, return)      credit assignment
    policy_loss(logp, logp_old, adv, valid, cfg)              -> (scalar, info)           policy improvement
    value_loss(values, ret, value_old, valid, cfg, boot)      -> scalar                   critic
    entropy_loss(ent, valid, cfg)                             -> scalar                   exploration
    aux_loss(out, batch, cfg)                                 -> (scalar, info)           representation learning
    total_loss(out, batch, adv, ret, cfg, boot)               -> (scalar, info)

The baseline rule is recurrent PPO with generalised advantage estimation (Schulman et al.): clipped surrogate
objective, an ensemble critic regressed on GAE returns (each member on a bootstrap subset of episodes), an entropy
bonus and a success-prediction auxiliary loss.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass
class RuleConfig:
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    succ_coef: float = 0.1
    adv_norm: bool = True
    value_clip: float = 0.0
    huber: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def advantages(reward, value, done, trunc, valid, cfg: RuleConfig):
    """Generalised advantage estimation over whole episodes.

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t);  A_t = delta_t + gamma * lam * A_{t+1}.
    After a terminal decision V(s_{t+1}) = 0. After a truncated one (decision budget reached, the world goes on)
    the bootstrap value is the critic's estimate at the last decision, V(s_t), because the next state was never
    observed; the recursion stops there. Returns are A + V."""
    B, T = reward.shape
    adv = torch.zeros_like(reward)
    last = torch.zeros(B, device=reward.device, dtype=reward.dtype)
    vnext = torch.zeros(B, device=reward.device, dtype=reward.dtype)
    v = value.detach()
    for t in range(T - 1, -1, -1):
        m = valid[:, t].to(reward.dtype)
        is_last = (done[:, t] + trunc[:, t]).clamp(max=1.0)
        nv = torch.where(trunc[:, t] > 0, v[:, t], vnext) * (1.0 - done[:, t])
        delta = reward[:, t] + cfg.gamma * nv - v[:, t]
        last = (delta + cfg.gamma * cfg.lam * (1.0 - is_last) * last) * m
        adv[:, t] = last
        vnext = torch.where(m > 0, v[:, t], vnext)
    return adv, adv + v * valid.to(reward.dtype)


def _mean(x, valid):
    w = valid.to(x.dtype)
    return (x * w).sum() / w.sum().clamp(min=1.0)


def policy_loss(logp, logp_old, adv, valid, cfg: RuleConfig):
    ratio = torch.exp((logp - logp_old).clamp(-20, 20))
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv
    loss = -_mean(torch.min(s1, s2), valid)
    with torch.no_grad():
        info = {"kl": float(_mean(logp_old - logp, valid)),
                "clip_frac": float(_mean(((ratio - 1.0).abs() > cfg.clip).to(logp.dtype), valid))}
    return loss, info


def value_loss(values, ret, value_old, valid, cfg: RuleConfig, boot=None):
    """`values` [B, T, K] ensemble; `boot` [B, K] in {0, 1}: which episodes each member trains on."""
    target = ret.unsqueeze(-1)
    err = values - target
    if cfg.value_clip > 0:
        vc = value_old.unsqueeze(-1) + (values - value_old.unsqueeze(-1)).clamp(-cfg.value_clip, cfg.value_clip)
        per = torch.max(err ** 2, (vc - target) ** 2)
    elif cfg.huber > 0:
        per = F.huber_loss(values, target.expand_as(values), reduction="none", delta=cfg.huber)
    else:
        per = err ** 2
    w = valid.to(values.dtype).unsqueeze(-1).expand_as(per)
    if boot is not None:
        w = w * boot.to(values.dtype).unsqueeze(1)
    return 0.5 * (per * w).sum() / w.sum().clamp(min=1.0)


def entropy_loss(ent, valid, cfg: RuleConfig):
    return -_mean(ent, valid)


def aux_loss(out, batch, cfg: RuleConfig):
    """Auxiliary prediction: will this episode end in verified success? (labels come from the evaluator)"""
    t = batch.t
    y = t["success"].unsqueeze(1).expand_as(out["succ_logit"])
    per = F.binary_cross_entropy_with_logits(out["succ_logit"], y, reduction="none")
    return _mean(per, t["valid"]), {}


def total_loss(out, batch, adv, ret, cfg: RuleConfig, boot=None):
    t = batch.t
    pl, info = policy_loss(out["logp"], t["logp_old"], adv, t["valid"], cfg)
    vl = value_loss(out["values"], ret, t["value_old"], t["valid"], cfg, boot)
    el = entropy_loss(out["ent"], t["valid"], cfg)
    al, _ = aux_loss(out, batch, cfg)
    loss = pl + cfg.vf_coef * vl + cfg.ent_coef * el + cfg.succ_coef * al
    info.update(pol_loss=float(pl.detach()), v_loss=float(vl.detach()), entropy=float(-el.detach()), aux_loss=float(al.detach()))
    return loss, info
