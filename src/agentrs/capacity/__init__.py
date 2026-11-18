"""Adaptive neural capacity for the shared continual-learning policy (research pathway, opt-in).

Modes (``capacity_mode`` in the run config; default ``fixed``):

* **fixed** — keep training the one shared LoRA (balanced replay, BC-replay regulariser, promotion gate).
* **expand_rank** — :func:`expand_lora_rank` grows the adapter rank of selected layers *function-
  preservingly* (old A/B copied, new B columns zero, scaling kept), transfers AdamW moments into the
  grown tensors, and :func:`match_checkpoint_ranks` rebuilds the right shapes when a checkpoint is
  restored (also from GitHub bundles: shapes are read from the safetensors header).
* **experts** — :class:`ExpertAdapters` adds a bounded set of extra LoRA experts on the same layers as
  the shared adapter, composed per example by a learned router (softmax gate over experts from a
  learned per-domain embedding); the shared core stays always-on, so skills remain shared.
* **consolidation** — :func:`distill_into_shared` trains the shared adapter to reproduce the composed
  (shared + experts) policy on replay prompts (token-level KL), after which an expert whose removal no
  longer changes held-out behaviour can be retired (:meth:`ExpertAdapters.retire`).

Growth is never scheduled: :class:`ExpansionAdvisor` only *proposes* expansion from evidence in the
run state (validation plateau while training reward keeps rising = capacity/generalisation limit;
cross-family interference; forgetting), and the proposal must be confirmed by a matched benchmark
(``scripts/capacity_benchmark.py``) before it is adopted.
"""
from agentrs.capacity.advisor import ExpansionAdvisor  # noqa: F401
from agentrs.capacity.experts import ExpertAdapters, distill_into_shared  # noqa: F401
from agentrs.capacity.lora_rank import (expand_lora_rank, lora_ranks, match_checkpoint_ranks,  # noqa: F401
                                        param_report, transfer_optimizer_state)
