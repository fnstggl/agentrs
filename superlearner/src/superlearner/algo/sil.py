"""PPO with replay of the agent's own verified-successful episodes (self-imitation).  [candidate-editable]

The replay mechanism lives in `algo/ppo.py` (`PPO.replay`) and its loss in the learning rule
(`rule.replay_loss`); this class only switches it on by configuration instead of by the rule's constants:

``sil_mode: filtered``   reward-filtered imitation (the rule's `replay_loss`): -log pi(a|s) on every decision of a
                         successful episode whose action the world accepted, plus the critic's regression;
``sil_mode: advantage``  Oh et al. (2018): -log pi(a|s) * max(R - V(s), 0) with the one-sided value loss.

Episodes are drawn so that every task arm in the buffer is equally likely. This is reinforcement learning from the
agent's own experience: the only supervision is the evaluator's verified outcome of episodes the agent itself
generated; no demonstration is involved.
"""
from __future__ import annotations

from .base import register
from .ppo import PPO


@register("ppo_sil")
class PPOSIL(PPO):
    uses_replay = True

    def __init__(self, model, cfg=None, device="cpu"):
        cfg = dict(cfg or {})
        cfg.setdefault("sil_steps", 2)
        cfg.setdefault("sil_coef", 0.5)
        cfg["sil_mode"] = {"filtered": "rule"}.get(cfg.get("sil_mode", "advantage"), cfg.get("sil_mode", "advantage"))
        super().__init__(model, cfg, device)
