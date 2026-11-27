"""cam_live.learning_api — how losses, batches and the continual learner fit together. MASTER-OWNED.

* B, C and D each expose :class:`LossProvider` objects (world-model NLL, evaluator BCE, policy improvement with
  intrinsic rewards, RUDDER regression, hindsight model, retention, adapter objectives ...).
* D's ``ContinualLearner`` owns optimizers, schedules, per-group clipping, logging and checkpoints, and simply
  iterates over the providers registered with it. It contains no loss mathematics of its own.
* The integrated agent registers the providers. The meta-update of R_psi is *not* a provider: it is a separate
  step (``reward.meta``) because it is the only path allowed to change the ``reward`` group.

:func:`encode_decisions` is the single shared way to turn stored experience into the tensors of ``nn.api``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch
from torch import Tensor

from .contracts import ExperienceRecord, ExperienceStore, OutcomeObservation, OutcomeQuery, OutcomeStatus
from .nn.api import LatentModelAPI

TIMESCALE_FREQUENT = "frequent"            # experiential + heads, event-driven
TIMESCALE_SLOW = "slow_adapters"           # semantic adapters (needs with_grad_semantic=True batches)


@dataclass
class DecisionBatch:
    """Tensors for a batch of stored decisions (one row per ExperienceRecord)."""
    records: List[ExperienceRecord]
    b: Tensor                      # [B, D]   belief at decision time
    h: Tensor                      # [B, T, D] per-token latents of H_t
    pad_mask: Tensor               # [B, T]
    e_a: Tensor                    # [B, N, D] all candidates of each decision
    a_mask: Tensor                 # [B, N]
    e_g: Tensor                    # [B, D]
    z: Tensor                      # [B, N, D]
    chosen: Tensor                 # [B] long: index of the executed candidate
    propensity: Tensor             # [B] behaviour-policy probability of the executed candidate
    behaviour_probs: Tensor        # [B, N] logged probabilities of all candidates (0 where masked)
    b_next: Optional[Tensor] = None        # [B, D] belief after the result was observed (None if not requested)
    has_next: Optional[Tensor] = None      # [B] bool
    duration_s: Optional[Tensor] = None    # [B] float64
    ts: Optional[Tensor] = None            # [B] float64 decision time
    with_grad_semantic: bool = False       # the graph reaches the semantic adapters [B-CR-7]

    @property
    def z_chosen(self) -> Tensor:
        idx = self.chosen.view(-1, 1, 1).expand(-1, 1, self.z.shape[-1])
        return self.z.gather(1, idx).squeeze(1)

    @property
    def e_a_chosen(self) -> Tensor:
        idx = self.chosen.view(-1, 1, 1).expand(-1, 1, self.e_a.shape[-1])
        return self.e_a.gather(1, idx).squeeze(1)


def encode_decisions(model: LatentModelAPI, store: ExperienceStore, records: Sequence[ExperienceRecord], *,
                     with_next: bool = False, with_grad_semantic: bool = False,
                     grad: bool = True, grad_last: Optional[int] = None) -> DecisionBatch:
    """Encode stored decisions with the *current* model. ``grad=False`` wraps everything in ``no_grad`` (cached
    latents for the meta unroll); ``with_grad_semantic`` additionally builds a graph into the semantic adapters,
    for the newest ``grad_last`` context items only when it is given [A-3, B-CR-7, D-8]."""
    records = list(records)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    kw = {"grad_last": grad_last} if (with_grad_semantic and grad_last is not None) else {}
    with ctx:
        contexts = [store.items(r.context_item_refs) for r in records]
        tokens = model.tokenize(contexts, with_grad=with_grad_semantic, **kw)
        out = model.core_forward(tokens, None)
        e_a, a_mask = model.encode_actions([[c.action for c in r.decision.candidates] or [r.decision.chosen]
                                            for r in records], with_grad=with_grad_semantic)
        e_g = model.encode_goals([r.goal_text for r in records], with_grad=with_grad_semantic)
        z = model.decision_latent(out.b, e_a, e_g)
        dev = out.b.device
        n = e_a.shape[1]
        chosen = torch.tensor([min(r.decision.chosen_index, n - 1) for r in records], device=dev, dtype=torch.long)
        probs = torch.zeros(len(records), n, device=dev)
        for i, r in enumerate(records):
            p = r.decision.probabilities or [1.0]
            probs[i, :len(p)] = torch.tensor(p[:n], device=dev)
        b_next = has_next = None
        if with_next:
            has = [bool(r.post_item_refs) for r in records]
            nxt = model.tokenize([store.items(r.post_item_refs if h else r.context_item_refs)
                                  for r, h in zip(records, has)], with_grad=with_grad_semantic, **kw)
            b_next = model.core_forward(nxt, None).b
            has_next = torch.tensor(has, device=dev)
    return DecisionBatch(records=records, b=out.b, h=out.h, pad_mask=tokens.pad_mask, e_a=e_a, a_mask=a_mask,
                         e_g=e_g, z=z, chosen=chosen, propensity=probs.gather(1, chosen[:, None]).squeeze(1),
                         behaviour_probs=probs, b_next=b_next, has_next=has_next,
                         duration_s=torch.tensor([r.duration_s for r in records], dtype=torch.float64, device=dev),
                         ts=torch.tensor([r.decision.ts for r in records], dtype=torch.float64, device=dev),
                         with_grad_semantic=bool(with_grad_semantic and grad))


def matured_outcomes(record: ExperienceRecord) -> List[Tuple[OutcomeQuery, OutcomeObservation]]:
    """(query, observation) pairs of a record that are usable as training targets: observed or censored only.
    Pending and unobservable outcomes are never returned (they are masked, not invented)."""
    queries: Dict[str, OutcomeQuery] = {}
    for ev in record.decision.candidates:
        if ev.action.action_id == record.decision.chosen.action_id:
            for p in ev.predictions:
                queries[p.query.query_id] = p.query
    for q in record.pending_queries:
        queries.setdefault(q.query_id, q)
    for q in record.queries:                                     # questions the agent asked itself [B-CR-6]
        queries.setdefault(q.query_id, q)
    if record.result is not None:
        for q in record.result.expected_outcomes:
            queries.setdefault(q.query_id, q)
    out, seen = [], set()
    for o in record.outcomes:
        key = o.dedup_key or o.query_id
        if o.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED) and o.query_id in queries and key not in seen:
            seen.add(key)
            out.append((queries[o.query_id], o))
    return out


@runtime_checkable
class LossProvider(Protocol):
    """One trainable objective. Stateless w.r.t. optimisation: the learner owns optimizers and clipping.

    Optional attributes (defaults in brackets) read by ``credit.learner.ContinualLearner`` [D-9] and by batch
    builders [B-CR-7]: ``uses_batch`` [True] (False: the provider reads its own data and is always called),
    ``needs_latents`` [True] (False: only ``batch.records`` is used, no encoding), ``adapter_grad`` [True]
    (False: not part of slow adapter updates), ``batch_kwargs`` [{}] (``encode_decisions`` keyword arguments the
    provider needs, e.g. ``with_next=True``), ``state_dict()`` / ``load_state_dict()`` (tensor state saved in
    checkpoints)."""
    name: str
    timescale: str                       # TIMESCALE_FREQUENT | TIMESCALE_SLOW
    weight: float

    def eligible(self, record: ExperienceRecord) -> bool:
        """May this record be trained on for this loss right now (matured, real, permitted provenance)?"""

    def loss(self, model: LatentModelAPI, batch: DecisionBatch, store: ExperienceStore
             ) -> Tuple[Optional[Tensor], Dict[str, float]]:
        """Scalar loss (or None when the batch carries no target) and metrics. Must not touch the ``reward``
        parameter group's gradients (R_psi outputs are detached everywhere except inside ``reward.meta``)."""


@dataclass
class UpdatePlan:
    """What the learner should do in one call (built by the agent from the trigger that fired)."""
    trigger: str
    timescales: Tuple[str, ...] = (TIMESCALE_FREQUENT,)
    max_steps: int = 1
    budget_s: float = 30.0
    exp_ids: List[str] = field(default_factory=list)     # newly matured experience that triggered the update
