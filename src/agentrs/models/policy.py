"""Goal-conditioned action policy pi_theta(a | belief, goal).

The policy is a causal language model restricted to a finite, validated
candidate set C(s) (the action mask, optionally extended with free-form LM
proposals that parse and validate)::

    pi(a | s) = exp(log p_LM(a | prompt(s)) / T) / sum_{a' in C(s)} exp(log p_LM(a' | prompt(s)) / T)

With T = 1 this is exactly the LM's distribution conditioned on a in C(s).
Log-probabilities, entropies and KL divergences over C(s) are therefore exact
and differentiable, which is what the RL and imitation objectives need.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import ConfigDict, BaseModel, Field

from agentrs.actions.schema import Action, try_parse_action
from agentrs.agent.belief import BeliefState, action_text_for_lm
from agentrs.agent.runtime import Decision
from agentrs.models.backbone import BackboneConfig, LMBackbone


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    temperature: float = 1.0
    prefix_cache: bool = True
    vocab_restrict: bool = True  # normalise token distributions over the state's candidate vocabulary
    # score = sum log p / n_tokens**length_norm. 0 = exact LM sequence likelihood; 1 = mean per-token
    # log-prob (removes the strong preference of untrained models for short action strings)
    length_norm: float = 0.0
    generative_proposals: int = 0  # LM-sampled free-form actions added to the candidate set
    max_candidates: int = 40


class LMPolicy(nn.Module):
    component = "policy"

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = LMBackbone(cfg.backbone)

    # ---------------------------------------------------------------- distribution
    def log_probs(self, prompts: Sequence[str], candidates: Sequence[Sequence[str]]) -> list[torch.Tensor]:
        conts = [[action_text_for_lm(c) for c in cs] for cs in candidates]
        scores = self.backbone.score(prompts, conts, prefix_cache=self.cfg.prefix_cache,
                                     vocab_restrict=self.cfg.vocab_restrict)
        if self.cfg.length_norm > 0:
            scores = [s / torch.tensor([float(len(self.backbone._encode(c))) for c in cs],
                                       device=s.device) ** self.cfg.length_norm if len(cs) else s
                      for s, cs in zip(scores, conts)]
        return [F.log_softmax(s / self.cfg.temperature, dim=-1) for s in scores]

    def truncate_candidates(self, cands: list[Action]) -> list[Action]:
        return cands[: self.cfg.max_candidates]

    @torch.no_grad()
    def proposals(self, prompt: str, k: int, obs=None) -> list[Action]:
        """Free-form generative proposals (parsed + structurally validated)."""
        out = []
        for text, _ in self.backbone.generate(prompt, num_samples=k, max_new_tokens=32):
            a = try_parse_action(text.strip())
            if a is not None and (a.target is None or obs is None or a.target in obs.element_ids()):
                out.append(a)
        return out

    @torch.no_grad()
    def distribution(self, belief: BeliefState, candidates: list[Action]) -> tuple[list[Action], torch.Tensor]:
        cands = self.truncate_candidates(candidates)
        prompt = belief.prompt()
        if self.cfg.generative_proposals > 0:
            seen = {c.to_str() for c in cands}
            for a in self.proposals(prompt, self.cfg.generative_proposals, belief.observation):
                if a.to_str() not in seen:
                    cands.append(a)
                    seen.add(a.to_str())
        lp = self.log_probs([prompt], [[c.to_str() for c in cands]])[0]
        return cands, lp


class PolicyActor:
    """Acts with the policy. Records exact behaviour log-probs.

    mode: "sample" (on-policy), "greedy" (evaluation), "epsilon" (epsilon-mixture exploration).
    """

    def __init__(self, policy: LMPolicy, mode: str = "sample", epsilon: float = 0.0, seed: int = 0,
                 name: str = "policy"):
        self.policy = policy
        self.mode = mode
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.name = name

    def begin_episode(self, belief):
        pass

    def decide(self, belief: BeliefState, candidates: list[Action]) -> Decision:
        cands, lp = self.policy.distribution(belief, candidates)
        p = lp.exp()
        n = len(cands)
        if self.mode == "greedy":
            i = int(torch.argmax(lp))
            beh = 0.0
        elif self.mode == "epsilon" and self.epsilon > 0:
            mix = (1 - self.epsilon) * p + self.epsilon / n
            i = int(torch.multinomial(mix, 1, generator=self.gen))
            beh = float(torch.log(mix[i]))
        else:
            i = int(torch.multinomial(p, 1, generator=self.gen))
            beh = float(lp[i])
        return Decision(action=cands[i], candidates=[c.to_str() for c in cands], chosen_index=i,
                        policy_logprob=float(lp[i]), behavior_logprob=beh, selection="policy",
                        info={"entropy": float(-(p * lp).sum()), "num_candidates": n})


def entropy_from_logp(lp: torch.Tensor) -> torch.Tensor:
    return -(lp.exp() * lp).sum()


def kl_from_logp(lp: torch.Tensor, ref_lp: torch.Tensor) -> torch.Tensor:
    """KL(pi || ref) over the same candidate set."""
    return (lp.exp() * (lp - ref_lp)).sum()

