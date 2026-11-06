"""Action-conditioned world model  P_phi(dState, events, r, done | belief, action).

Architecture
------------
A causal LM backbone (initialised from a small pretrained LM) reads::

    GOAL: ...
    RECENT: <last k actions and outcomes>
    OBSERVATION:
    <canonical observation>
    ACTION: <canonical action>
    EFFECT:

and produces two kinds of programmatically usable predictions:

1. **Generative state-difference model**: the LM continues with the canonical
   ``StateDiff`` text terminated by an ``END`` line. Trained by token-level
   negative log-likelihood (a proper likelihood over structured outcomes).
   Sampling several diffs gives a distribution over possible next observations,
   which ``StateDiff.apply`` turns into imagined next observations.
2. **Event / reward / termination heads**: an ensemble of ``n_heads`` small MLPs
   on the pooled last hidden state predicts Bernoulli probabilities for the 12
   observable events in ``data.diff.EVENT_NAMES`` plus P(verified reward > 0)
   and P(episode terminates). Each head is trained on its own bootstrap mask of
   the data; disagreement between heads estimates epistemic uncertainty.

The model never sees hidden simulator state: inputs are observations, goal and
actions; targets are the next *observation* (via the diff), observable events,
and the environment-reported reward/termination.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import ConfigDict, BaseModel, Field

from agentrs.data.diff import EVENT_NAMES, N_EVENTS, StateDiff
from agentrs.environments.base import Observation
from agentrs.models.backbone import BackboneConfig, LMBackbone

N_OUT = N_EVENTS + 2  # events + reward + terminal
REWARD_IDX = N_EVENTS
TERMINAL_IDX = N_EVENTS + 1
END_MARK = "END"


class WorldModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: BackboneConfig = Field(default_factory=lambda: BackboneConfig(max_prompt_tokens=640))
    n_heads: int = 5
    head_hidden: int = 128
    bootstrap_prob: float = 0.8
    diff_loss_weight: float = 1.0
    head_loss_weight: float = 1.0
    history_items: int = 3
    max_obs_elements: int = 30
    max_diff_tokens: int = 96


@dataclass
class WMPrediction:
    event_probs: list[float]
    event_std: list[float]
    reward_prob: float
    terminal_prob: float
    reward_std: float
    terminal_std: float
    diffs: list[tuple[str, float]] = field(default_factory=list)  # (diff text, mean token logprob)
    next_observations: list[Observation] = field(default_factory=list)

    @property
    def epistemic(self) -> float:
        """Mean ensemble std over all predicted probabilities (events, reward, terminal)."""
        vals = self.event_std + [self.reward_std, self.terminal_std]
        return float(sum(vals) / len(vals))

    @property
    def diff_nll(self) -> float:
        if not self.diffs:
            return 0.0
        return float(-sum(lp for _, lp in self.diffs) / len(self.diffs))

    def events(self) -> dict[str, float]:
        return dict(zip(EVENT_NAMES, self.event_probs))


class EnsembleHeads(nn.Module):
    def __init__(self, d: int, hidden: int, n_heads: int, n_out: int):
        super().__init__()
        self.n_heads = n_heads
        self.w1 = nn.Parameter(torch.randn(n_heads, d, hidden) * (1.0 / d**0.5))
        self.b1 = nn.Parameter(torch.zeros(n_heads, hidden))
        self.w2 = nn.Parameter(torch.randn(n_heads, hidden, n_out) * (1.0 / hidden**0.5))
        self.b2 = nn.Parameter(torch.zeros(n_heads, n_out))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: (B, d) -> logits (H, B, K)
        h = self.norm(h)
        z = torch.einsum("bd,hdk->hbk", h, self.w1) + self.b1[:, None]
        z = F.gelu(z)
        return torch.einsum("hbk,hko->hbo", z, self.w2) + self.b2[:, None]


def bootstrap_mask(keys: Sequence[str], n_heads: int, p: float, device) -> torch.Tensor:
    """Deterministic per-(example, head) inclusion mask so each head sees a fixed bootstrap subset."""
    m = torch.zeros((n_heads, len(keys)), device=device)
    for j, k in enumerate(keys):
        for h in range(n_heads):
            u = int(hashlib.md5(f"{k}|{h}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            m[h, j] = 1.0 if u < p else 0.0
    m[:, m.sum(0) == 0] = 1.0  # every example trains at least one head... all heads if none chosen
    return m


class WorldModel(nn.Module):
    component = "world_model"

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = LMBackbone(cfg.backbone)
        self.heads = EnsembleHeads(self.backbone.hidden_size, cfg.head_hidden, cfg.n_heads, N_OUT)
        # sensible priors: most events are rare
        with torch.no_grad():
            self.heads.b2.fill_(-2.0)

    # ---------------------------------------------------------------- inputs
    def prompt(self, goal: str, obs: Observation, action: str, history: Sequence[tuple[str, str]] = ()) -> str:
        lines = [f"GOAL: {goal}"]
        if history:
            lines.append("RECENT:")
            lines += [f"- {a} -> {o}" for a, o in list(history)[-self.cfg.history_items:]]
        lines.append("OBSERVATION:")
        lines.append(obs.to_text(self.cfg.max_obs_elements))
        lines.append(f"ACTION: {action}")
        lines.append("EFFECT:")
        return "\n".join(lines)

    @staticmethod
    def diff_target(diff_text: str) -> str:
        return "\n" + diff_text + "\n" + END_MARK + "\n"

    # ---------------------------------------------------------------- heads
    def head_logits(self, prompts: Sequence[str]) -> torch.Tensor:
        h = self.backbone.pooled(prompts)
        return self.heads(h)  # (H, B, K)

    def loss(self, prompts: Sequence[str], diff_texts: Sequence[str], targets: torch.Tensor,
             keys: Sequence[str]) -> tuple[torch.Tensor, dict]:
        """targets: (B, N_OUT) in {0,1}. keys: stable example ids for bootstrap masks.
        One backbone forward serves both the diff NLL and the pooled-state heads."""
        nll, counts, h = self.backbone.nll_and_prompt_hidden(prompts, [self.diff_target(d) for d in diff_texts])
        logits = self.heads(h)
        tgt = targets.to(logits.device).unsqueeze(0).expand_as(logits)
        mask = bootstrap_mask(keys, self.cfg.n_heads, self.cfg.bootstrap_prob, logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none").mean(-1)  # (H, B)
        head_loss = (bce * mask).sum() / mask.sum()
        diff_loss = (nll / counts).mean()
        loss = self.cfg.head_loss_weight * head_loss + self.cfg.diff_loss_weight * diff_loss
        return loss, {"head_bce": float(head_loss.detach()), "diff_nll_per_token": float(diff_loss.detach())}

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()
    def predict_heads(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean probs (B,K), std across heads (B,K))."""
        p = torch.sigmoid(self.head_logits(prompts))
        return p.mean(0), p.std(0) if p.shape[0] > 1 else torch.zeros_like(p[0])

    @torch.no_grad()
    def sample_diffs(self, prompt: str, n: int = 1, greedy: bool = True, temperature: float = 1.0,
                     generator: Optional[torch.Generator] = None) -> list[tuple[str, float]]:
        return self.sample_diffs_batch([prompt] * n, greedy=greedy, temperature=temperature, generator=generator)

    @torch.no_grad()
    def sample_diffs_batch(self, prompts: Sequence[str], greedy: bool = True, temperature: float = 1.0,
                           generator: Optional[torch.Generator] = None) -> list[tuple[str, float]]:
        """One sampled diff per prompt: (diff text, mean token log-prob)."""
        if not prompts:
            return []
        outs = self.backbone.generate_batch(list(prompts), max_new_tokens=self.cfg.max_diff_tokens,
                                            temperature=temperature, greedy=greedy, stop_text="\n" + END_MARK,
                                            generator=generator)
        res = []
        for text, lp in outs:
            body = text.split("\n" + END_MARK)[0].strip("\n")
            ntok = max(1, len(self.backbone._encode(text)))
            res.append((body if body else "(no change)", lp / ntok))
        return res

    @torch.no_grad()
    def predict(self, goal: str, obs: Observation, action: str, history: Sequence[tuple[str, str]] = (),
                n_samples: int = 1, greedy: bool = True, generate: bool = True,
                generator: Optional[torch.Generator] = None) -> WMPrediction:
        pr = self.prompt(goal, obs, action, history)
        mean, std = self.predict_heads([pr])
        mean, std = mean[0].tolist(), std[0].tolist()
        pred = WMPrediction(event_probs=mean[:N_EVENTS], event_std=std[:N_EVENTS], reward_prob=mean[REWARD_IDX],
                            terminal_prob=mean[TERMINAL_IDX], reward_std=std[REWARD_IDX],
                            terminal_std=std[TERMINAL_IDX])
        if generate:
            pred.diffs = self.sample_diffs(pr, n=n_samples, greedy=greedy and n_samples == 1, generator=generator)
            pred.next_observations = [StateDiff.from_text(d).apply(obs) for d, _ in pred.diffs]
        return pred

    @torch.no_grad()
    def predict_batch_heads(self, goal: str, obs: Observation, actions: Sequence[str],
                            history: Sequence[tuple[str, str]] = ()) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [self.prompt(goal, obs, a, history) for a in actions]
        return self.predict_heads(prompts)


def targets_from_transition(events: Sequence[float], reward: float, terminated: bool) -> list[float]:
    ev = list(events) if events else [0.0] * N_EVENTS
    return ev + [1.0 if reward > 0 else 0.0, 1.0 if terminated else 0.0]
