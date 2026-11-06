"""Content-conditioned world model.

    P(observed effects, next observation, reward | belief, complete action incl. content)

Extends the existing ``WorldModel`` (one shared trunk; no second network):

* the complete action — operation AND exact content — is part of the input text
  (``ACTION: submit(send_email, "Hi Dana, can we meet Mon 9am?")``), so two actions that differ
  only in content are different inputs and can get different predictions;
* the existing generative diff head predicts the environment's textual response (a reply,
  ``Tests: 2/3 passed``, a moderator verdict) — i.e. *possible* responses, sampled;
* the existing ensemble heads predict events, P(verified reward) and P(terminal);
* NEW content-outcome ensemble heads predict, for content actions, P(format valid),
  P(objective met), P(partial credit >= 0.5) and P(positive response), with ensemble
  disagreement as epistemic uncertainty. Transitions without a content outcome (e.g. MiniWeb
  clicks) simply contribute no loss to these heads, so the same model learns both.
* content longer than the encoder's chunk size is read chunk by chunk and the content heads use
  the mean of the chunks' pooled states (all of the content is read, none is summarised away).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from pydantic import ConfigDict

from agentrs.content.semantic_action import ContentActionEncoder
from agentrs.environments.base import Observation
from agentrs.models.world_model import N_EVENTS, EnsembleHeads, WorldModel, WorldModelConfig, bootstrap_mask

CONTENT_OUTCOMES = ["format_valid", "objective_met", "partial_ge_half", "response_positive"]
N_CONTENT = len(CONTENT_OUTCOMES)
OBJECTIVE_IDX = 1


class ContentWorldModelConfig(WorldModelConfig):
    model_config = ConfigDict(extra="forbid")
    content_head_weight: float = 1.0
    chunk_tokens: int = 160


def content_targets(outcome: Optional[dict]) -> Optional[list[float]]:
    if not outcome:
        return None
    return [float(bool(outcome.get("format_valid"))), float(bool(outcome.get("objective_met"))),
            float((outcome.get("partial") or 0.0) >= 0.5), float(bool(outcome.get("response_positive")))]


@dataclass
class ContentPrediction:
    p_reward: float
    p_terminal: float
    content_probs: list[float]
    content_std: list[float]
    diffs: list[tuple[str, float]] = field(default_factory=list)

    @property
    def p_objective(self) -> float:
        return self.content_probs[OBJECTIVE_IDX]

    @property
    def epistemic(self) -> float:
        return float(sum(self.content_std) / len(self.content_std))


class ContentConditionedWorldModel(WorldModel):
    component = "content_world_model"

    def __init__(self, cfg: ContentWorldModelConfig):
        super().__init__(cfg)
        self.content_heads = EnsembleHeads(self.backbone.hidden_size, cfg.head_hidden, cfg.n_heads, N_CONTENT)
        with torch.no_grad():
            self.content_heads.b2.fill_(0.0)
        self.encoder = ContentActionEncoder(self.backbone.tokenizer, cfg.chunk_tokens)

    # ---------------------------------------------------------------- representations
    def action_prompts(self, goal: str, obs: Observation, action_str: str,
                       history: Sequence[tuple[str, str]] = ()) -> list[str]:
        from agentrs.actions.schema import try_parse_action

        a = try_parse_action(action_str)
        texts = self.encoder.chunked_action_texts(a) if a is not None else [action_str]
        return [self.prompt(goal, obs, t, history) for t in texts]

    def content_hidden(self, prompt_groups: Sequence[Sequence[str]]) -> torch.Tensor:
        flat = [p for g in prompt_groups for p in g]
        h = self.backbone.pooled(flat)
        out, k = [], 0
        for g in prompt_groups:
            out.append(h[k: k + len(g)].mean(0))
            k += len(g)
        return torch.stack(out)

    # ---------------------------------------------------------------- training
    def content_loss(self, prompts: Sequence[str], diff_texts: Sequence[str], targets: torch.Tensor,
                     keys: Sequence[str], content_tgts: Sequence[Optional[list[float]]],
                     chunk_groups: Optional[Sequence[Optional[Sequence[str]]]] = None) -> tuple[torch.Tensor, dict]:
        """Base world-model loss (diff NLL + event/reward/terminal heads) + content-outcome heads
        on examples that have a content outcome. One backbone forward serves all heads for
        single-chunk actions; long content adds chunk forwards for the content heads."""
        nll, counts, h = self.backbone.nll_and_prompt_hidden(prompts, [self.diff_target(d) for d in diff_texts])
        logits = self.heads(h)
        tgt = targets.to(logits.device).unsqueeze(0).expand_as(logits)
        mask = bootstrap_mask(keys, self.cfg.n_heads, self.cfg.bootstrap_prob, logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none").mean(-1)
        head_loss = (bce * mask).sum() / mask.sum()
        diff_loss = (nll / counts).mean()
        loss = self.cfg.head_loss_weight * head_loss + self.cfg.diff_loss_weight * diff_loss
        stats = {"head_bce": float(head_loss.detach()), "diff_nll_per_token": float(diff_loss.detach())}
        idx = [i for i, t in enumerate(content_tgts) if t is not None]
        if idx:
            hc = h[idx]
            if chunk_groups is not None:
                long_rows = [j for j, i in enumerate(idx) if chunk_groups[i] and len(chunk_groups[i]) > 1]
                if long_rows:
                    hl = self.content_hidden([chunk_groups[idx[j]] for j in long_rows])
                    hc = hc.clone()
                    hc[long_rows] = hl
            cl = self.content_heads(hc)
            ct = torch.tensor([content_tgts[i] for i in idx], dtype=torch.float32, device=cl.device)
            cmask = bootstrap_mask([keys[i] for i in idx], self.cfg.n_heads, self.cfg.bootstrap_prob, cl.device)
            cbce = F.binary_cross_entropy_with_logits(cl, ct.unsqueeze(0).expand_as(cl), reduction="none").mean(-1)
            c_loss = (cbce * cmask).sum() / cmask.sum()
            loss = loss + self.cfg.content_head_weight * c_loss
            stats["content_bce"] = float(c_loss.detach())
        return loss, stats

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()
    def predict_content(self, goal: str, obs: Observation, action_strs: Sequence[str],
                        history: Sequence[tuple[str, str]] = (), sample_responses: bool = False,
                        generator: Optional[torch.Generator] = None) -> list[ContentPrediction]:
        groups = [self.action_prompts(goal, obs, a, history) for a in action_strs]
        main_prompts = [g[0] if len(g) == 1 else self.prompt(goal, obs, a, history) for g, a in zip(groups,
                                                                                                    action_strs)]
        hm = self.backbone.pooled(main_prompts)
        pm = torch.sigmoid(self.heads(hm))  # (H, B, K)
        hc = self.content_hidden(groups) if any(len(g) > 1 for g in groups) else hm
        pc = torch.sigmoid(self.content_heads(hc))
        mean_m = pm.mean(0)
        mean_c, std_c = pc.mean(0), (pc.std(0) if pc.shape[0] > 1 else torch.zeros_like(pc[0]))
        preds = []
        for i in range(len(action_strs)):
            preds.append(ContentPrediction(p_reward=float(mean_m[i, N_EVENTS]), p_terminal=float(mean_m[i, N_EVENTS + 1]),
                                           content_probs=mean_c[i].tolist(), content_std=std_c[i].tolist()))
        if sample_responses:
            diffs = self.sample_diffs_batch(main_prompts, greedy=generator is None, generator=generator)
            for p, d in zip(preds, diffs):
                p.diffs = [d]
        return preds
