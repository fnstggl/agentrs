"""cam_live.nn.api — the tensor contract between the neural model (Workstream A) and everything that builds on
it (B: prediction / decision, C: intrinsic reward / meta-learning, D: credit / continual learning, agent).

MASTER-OWNED. Request changes in ``cam_live/docs/change_requests/<workstream>.md``.

Shapes (D = d_model of the experiential core, F = n_features of the fast-adaptation feature map):

    b        [B, D]        integrated belief latent  b_t = F_theta(H_t, m_t)
    h        [B, T, D]     per-token latents of the context H_t (h[:, i] sees only items <= i: causal)
    e_a      [B, N, D]     candidate-action embeddings, with ``a_mask`` [B, N] bool (True = real candidate)
    e_g      [B, D]        goal embedding
    e_q      [B, Q, D]     outcome-query embeddings (metric text + kind + horizon), ``q_mask`` [B, Q]
    z        [B, N, D]     decision latent  z = F_theta(H_t, a_t, g)  (spec 3.2), one per candidate action
    phi      [B, N, Q, F]  learned features  phi = Phi_theta(b_t, a_t, dt)  for the Bayesian fast layer (spec 2.1)

All of the above are float32 on ``model.device`` regardless of the backbone's dtype (the backbone may run in
bf16; double-backward for meta-gradients requires fp32 downstream of it).

Gradient rules:
* ``embed_texts(with_grad=False)`` (default) returns a detached, cacheable tensor: no graph into the backbone.
* ``embed_texts(with_grad=True)`` builds a graph into the trainable semantic adapters (LoRA) only.
* Nothing in this API ever lets a loss reach frozen backbone weights (they have ``requires_grad=False``).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..contracts import ActionSpec, ExperienceItem, OutcomeKind, OutcomeQuery

#: index of each outcome kind in kind-id tensors
KIND_INDEX: Dict[str, int] = {k: i for i, k in enumerate(OutcomeKind.ALL)}

#: parameter-group names every model reports and every optimizer / log / checkpoint uses (spec 1.1, 10.2)
GROUP_BACKBONE_FROZEN = "backbone_frozen"
GROUP_ADAPTERS = "semantic_adapters"          # LoRA (and latent-conditioning bridge into the backbone)
GROUP_EXPERIENTIAL = "experiential"           # tokenizer projections + experiential core + decision trunk + features
GROUP_REWARD = "reward"                       # intrinsic reward generator R_psi (meta-learned only)
GROUP_HEADS = "heads"                         # other trainable output heads (world / evaluator / policy / credit / voi)
GROUPS = (GROUP_BACKBONE_FROZEN, GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_REWARD, GROUP_HEADS)


@dataclass
class TokenBatch:
    """Tokenised context windows. Exact numbers ride along in ``values`` and are never recovered from ``x``."""
    x: Tensor                    # [B, T, D] token embeddings (semantic + type + time + value channels, already fused)
    pad_mask: Tensor             # [B, T] bool, True = real token
    time: Tensor                 # [B, T] float64 seconds (absolute); padded positions hold the last real time
    kind_ids: Tensor             # [B, T] long, index into contracts.ItemKind.ALL
    values: Tensor               # [B, T, V] float64 raw quantity values (first V quantities of the item, 0 padded)
    value_mask: Tensor           # [B, T, V] bool
    refs: List[List[str]] = field(default_factory=list)   # per token: id of the source record


@dataclass
class CoreState:
    """Persistent recurrent latent state carried between calls (part of the fast state m_t)."""
    slots: Tensor                # [B, K, D]
    last_time: Tensor            # [B] float64 seconds
    step: int = 0


@dataclass
class CoreOutput:
    h: Tensor                    # [B, T, D]
    b: Tensor                    # [B, D]
    state: CoreState
    aux: Dict[str, Tensor] = field(default_factory=dict)


class LatentModelAPI(nn.Module, abc.ABC):
    """What Workstream A implements (``cam_live.nn.model.CamLiveModel``). Head modules built by B/C/D are plain
    ``nn.Module`` functions of the tensors above and are registered with :meth:`register_head`."""

    d_model: int
    n_features: int
    d_sem: int                   # backbone embedding width seen by the tokenizer

    # ---- semantic backbone ------------------------------------------------------------------------------
    @abc.abstractmethod
    def embed_texts(self, texts: Sequence[str], *, with_grad: bool = False) -> Tensor:
        """[n, d_sem] pooled semantic embeddings of ``texts`` (multi-layer tap of the pretrained backbone)."""

    @abc.abstractmethod
    def generate(self, prompts: Sequence[str], *, latent: Optional[Tensor] = None, max_new_tokens: int = 256,
                 temperature: float = 0.8, num_return_sequences: int = 1, seed: Optional[int] = None,
                 ) -> List[List[str]]:
        """Sample text continuations from the backbone (+adapters). ``latent`` [n, D] conditions generation on
        CAM's belief through the trainable bridge; ``latent=None`` is the unconditioned backbone."""

    @abc.abstractmethod
    def generation_logprob(self, prompts: Sequence[str], completions: Sequence[str], *,
                           latent: Optional[Tensor] = None) -> Tensor:
        """[n] sum of token log-probabilities of each completion given its prompt; differentiable w.r.t. the
        semantic adapters, the latent bridge and ``latent``."""

    # ---- experience encoding ----------------------------------------------------------------------------
    @abc.abstractmethod
    def tokenize(self, contexts: Sequence[Sequence[ExperienceItem]], *, with_grad: bool = False) -> TokenBatch: ...

    @abc.abstractmethod
    def core_forward(self, tokens: TokenBatch, state: Optional[CoreState] = None) -> CoreOutput: ...

    @abc.abstractmethod
    def init_state(self, batch: int) -> CoreState: ...

    @abc.abstractmethod
    def encode_actions(self, actions: Sequence[Sequence[ActionSpec]], *, with_grad: bool = False
                       ) -> Tuple[Tensor, Tensor]:
        """(e_a [B, N, D], a_mask [B, N])."""

    @abc.abstractmethod
    def encode_goals(self, goals: Sequence[str], *, with_grad: bool = False) -> Tensor:
        """e_g [B, D]."""

    @abc.abstractmethod
    def encode_queries(self, queries: Sequence[Sequence[OutcomeQuery]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """(e_q [B, Q, D], q_mask [B, Q], kind_ids [B, Q] long (KIND_INDEX), horizon_s [B, Q] float64)."""

    @abc.abstractmethod
    def decision_latent(self, b: Tensor, e_a: Tensor, e_g: Tensor) -> Tensor:
        """z [B, N, D]."""

    @abc.abstractmethod
    def features(self, z: Tensor, e_q: Tensor) -> Tensor:
        """phi [B, N, Q, F]: bounded, well-conditioned features for Bayesian last-layer adaptation."""

    # ---- bookkeeping -------------------------------------------------------------------------------------
    @abc.abstractmethod
    def register_head(self, name: str, module: nn.Module, group: str = GROUP_HEADS) -> nn.Module:
        """Attach a head so it is counted, optimised, checkpointed and hashed under ``group``."""

    @abc.abstractmethod
    def head(self, name: str) -> nn.Module: ...

    @abc.abstractmethod
    def param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """group name (GROUPS) -> parameters. Disjoint; their union is every parameter of the model."""

    @abc.abstractmethod
    def count_parameters(self) -> Dict[str, Any]:
        """Exact counts: frozen, trainable experiential, trainable adapters, reward, other heads (per head),
        total persistent trainable (spec 1.1)."""

    @abc.abstractmethod
    def group_hashes(self) -> Dict[str, str]:
        """sha256 over each trainable group's tensors (used for before/after learning records)."""

    @abc.abstractmethod
    def trainable_state(self) -> Dict[str, Tensor]:
        """Everything persistent and trainable (no frozen backbone weights)."""

    @abc.abstractmethod
    def load_trainable_state(self, state: Dict[str, Tensor], *, strict: bool = True) -> Dict[str, List[str]]:
        """Returns {"missing": [...], "unexpected": [...]}; ``strict=False`` tolerates newly added heads."""

    @abc.abstractmethod
    def arch_signature(self) -> Dict[str, Any]:
        """Stable description (scale preset, dims, backbone id + revision, adapter config) stored in checkpoints."""

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...
