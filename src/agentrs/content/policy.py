"""Content-generating policy: chooses the operation AND generates the content it carries.

    pi(a | s) = pi(op | s) * prod_i pi(c_i | s, op, c_<i)

Both factors come from ONE network — the existing ``LMPolicy`` (shared weights with the
high-level action policy; no duplicate model):

* operation: the candidate-restricted distribution over the environment's operation set
  (``submit(<op>, "`` prefixes and ``finish()``), exactly as for primitive actions;
* content: autoregressive sampling after ``<belief prompt> submit(<op>, "`` until the closing
  ``")`` terminator (the terminator tokens are part of the sampled content sequence, so the
  model also learns when to stop). The exact sampled token ids and their log-probabilities
  under the sampling distribution are recorded; training re-scores THOSE ids (re-tokenising the
  decoded text can give different tokens).

Collection for RL uses temperature 1 and no truncation of the distribution, so the recorded
behaviour log-probabilities equal the policy's own log-probabilities (on-policy). Drafts
sampled at another temperature are labelled as such and treated as off-policy.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict

from agentrs.actions.schema import Action, ActionType
from agentrs.agent.belief import BeliefState
from agentrs.agent.runtime import Decision
from agentrs.content.semantic_action import SemanticAction, content_from_generation
from agentrs.environments.content.env import MODALITY
from agentrs.models.policy import LMPolicy

TERMINATOR = '")'


class ContentGenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_content_tokens: int = 48
    temperature: float = 1.0


def op_text(a: Action) -> str:
    """Continuation scored for the operation choice."""
    if a.type == ActionType.SUBMIT:
        return f' submit({a.name}, "'
    return " " + a.to_str() + "\n"


def generation_prompt(belief_prompt: str, operation: str) -> str:
    return belief_prompt + f' submit({operation}, "'


def escape_content(c: str) -> str:
    return c.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class ContentGeneratingPolicy:
    def __init__(self, policy: LMPolicy, cfg: Optional[ContentGenConfig] = None, checkpoint_id: str = "init"):
        self.policy = policy
        self.bb = policy.backbone
        self.cfg = cfg or ContentGenConfig()
        self.checkpoint_id = checkpoint_id

    # ------------------------------------------------------------------ operation factor
    def op_log_probs(self, prompts: Sequence[str], candidates: Sequence[Sequence[Action]]) -> list[torch.Tensor]:
        conts = [[op_text(a) for a in cs] for cs in candidates]
        scores = self.bb.score(list(prompts), conts, prefix_cache=self.policy.cfg.prefix_cache,
                               vocab_restrict=self.policy.cfg.vocab_restrict)
        return [F.log_softmax(s, dim=-1) for s in scores]

    # ------------------------------------------------------------------ content factor
    @torch.no_grad()
    def sample_content(self, gen_prompts: Sequence[str], temperature: Optional[float] = None,
                       generator: Optional[torch.Generator] = None, greedy: bool = False) -> list[dict]:
        T = self.cfg.temperature if temperature is None else temperature
        outs = self.bb.generate_tokens(list(gen_prompts), max_new_tokens=self.cfg.max_content_tokens,
                                       temperature=T, stop_text=TERMINATOR, greedy=greedy, generator=generator)
        for o in outs:
            o["content"], o["terminated"] = content_from_generation(o["text"], TERMINATOR)
            o["temperature"] = T
        return outs

    def content_log_probs(self, gen_prompts: Sequence[str], token_ids: Sequence[Sequence[int]],
                          return_logdists: bool = False):
        """Differentiable per-token log-probs of the given (exact) token ids, full vocabulary, T=1.
        Returns (logps (N, Tmax), mask (N, Tmax)[, logdists])."""
        return self.bb.token_logprobs_from_ids(gen_prompts, token_ids, return_logdists=return_logdists)

    def demo_token_ids(self, content: str) -> list[int]:
        """Token ids a demonstration's content corresponds to (escaped content + terminator)."""
        return self.bb._encode(escape_content(content) + TERMINATOR)

    # ------------------------------------------------------------------ acting
    def semantic_for(self, belief: BeliefState, op_cands: list[Action], op_idx: int, op_lp: float,
                     gp: str, sample: dict, env_id: str, family: str, source: str = "policy") -> SemanticAction:
        a = op_cands[op_idx]
        modality, fmt = MODALITY.get(family, ("text", "plain"))
        return SemanticAction(
            operation=a.name or "", content=sample["content"], modality=modality, format=fmt, target_env=env_id,
            goal=belief.task.goal, context=belief.prompt(), generation_prompt=gp,
            policy_checkpoint=self.checkpoint_id, op_candidates=[op_text(c) for c in op_cands], op_index=op_idx,
            op_logprob=op_lp, token_ids=list(sample["ids"]), token_logprobs=list(sample["logprobs"]),
            sampling_temperature=sample["temperature"], source=source,
            truncated_generation=not sample["terminated"])


class ContentPolicyActor:
    """Actor that picks an operation and generates its content with the policy.
    mode='sample' (on-policy, T=1) or 'greedy' (evaluation)."""

    def __init__(self, cp: ContentGeneratingPolicy, mode: str = "sample", seed: int = 0, name: str = "content-policy"):
        self.cp, self.mode, self.name = cp, mode, name
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = random.Random(seed)

    def begin_episode(self, belief):
        pass

    @torch.no_grad()
    def decide(self, belief: BeliefState, candidates: list[Action]) -> Decision:
        prompt = belief.prompt()
        lp = self.cp.op_log_probs([prompt], [candidates])[0]
        if self.mode == "greedy":
            i = int(torch.argmax(lp))
        else:
            i = int(torch.multinomial(lp.exp(), 1, generator=self.gen))
        a = candidates[i]
        cand_strs = [c.to_str() for c in candidates]
        if a.type != ActionType.SUBMIT:
            return Decision(action=a, candidates=cand_strs, chosen_index=i, policy_logprob=float(lp[i]),
                            behavior_logprob=0.0 if self.mode == "greedy" else float(lp[i]), selection="policy")
        gp = generation_prompt(prompt, a.name or "")
        s = self.cp.sample_content([gp], temperature=1.0, generator=self.gen, greedy=self.mode == "greedy")[0]
        sa = self.cp.semantic_for(belief, candidates, i, float(lp[i]), gp, s, belief.task.env_id, belief.task.family)
        act = sa.to_action()
        total = float(lp[i]) + sum(s["logprobs"])
        return Decision(action=act, candidates=cand_strs, chosen_index=i, policy_logprob=total,
                        behavior_logprob=None if self.mode == "greedy" else total,
                        selection="policy" if self.mode != "greedy" else "policy_greedy", semantic=sa)


class CandidateContentGenerator:
    """Multiple content drafts for one decision (for planning). Drafts are sampled at
    ``temperature``; drafts sampled at T != 1 or chosen by a planner are off-policy data."""

    def __init__(self, cp: ContentGeneratingPolicy, num_drafts: int = 4, temperature: float = 1.0):
        self.cp, self.num_drafts, self.temperature = cp, num_drafts, temperature

    @torch.no_grad()
    def drafts(self, gen_prompt: str, n: Optional[int] = None, generator: Optional[torch.Generator] = None) -> list[dict]:
        outs = self.cp.sample_content([gen_prompt] * (n or self.num_drafts), temperature=self.temperature,
                                      generator=generator)
        seen, uniq = set(), []
        for o in outs:
            if o["content"] not in seen:
                seen.add(o["content"])
                uniq.append(o)
        return uniq
