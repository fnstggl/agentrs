"""Experimental condition for the diagnosis of near-input-independent decisions (workstream Y).

Measured in the assembled stack (results/diagnostics.json): the latent state is dominated by a component shared by
all samples - after a real test, flipping only the observed signal changes Z by 0.55% of its norm and R by 0.33%,
and the action generator's launch probability by 0.021 - while the policy objective's gradient at initialisation
goes mostly into input-independent tool preferences (tool embeddings / pointer: 0.236 + 0.073) and hardly into
the latent state (core 0.0015, reasoner 0.0003). The world model's heads already read latent states through a
running per-position standardisation (world._util.StateNorm: "prediction heads read (Z - mean) / std with running
statistics so sample-specific content is visible at unit scale"); the decision heads do not.

`StandardizedReadoutActionGenerator` is A's ActionGenerator reading R through the same StateNorm, i.e. it tests
whether giving the decision heads the readout the world-model heads already have lets policy learning use the
latent state. The statistics change only in `update_readout_stats()`, which the stack's trainer adapter calls after
each optimiser step (from the R of the decisions acted on since the previous step), so a decision's
log-probability is identical when acting and in the learner's recomputation at the same weights. Declared as an
experimental condition wherever it is used; the default stack uses the unmodified ActionGenerator.
"""
from __future__ import annotations

import torch

from ..semantic.actions import ActionGenerator
from ..world._util import StateNorm


class StandardizedReadoutActionGenerator(ActionGenerator):
    def __init__(self, cfg, encoder, tokenizer, *, readout_momentum: float = 0.05, **kw):
        super().__init__(cfg, encoder, tokenizer, **kw)
        self.readout_norm = StateNorm(cfg.n_state, cfg.d_model, readout_momentum).to(self.device)
        self._seen: list = []

    def _forward(self, R, *a, **k):
        if k.get("teacher") is None:                       # acting (sampling): remember R for the statistics
            self._seen.append(R.detach().float().reshape(-1, *R.shape[-2:]))
        return super()._forward(self.readout_norm(R).to(R.dtype), *a, **k)

    @torch.no_grad()
    def update_readout_stats(self) -> int:
        """Fold the R of the decisions acted on since the last call into the running statistics."""
        if not self._seen:
            return 0
        X = torch.cat(self._seen)
        self._seen = []
        self.readout_norm.update(X)
        return int(X.shape[0])


def discard_readout_buffer(model) -> None:
    """Evaluation decisions must not enter the readout statistics: drop the R recorded since the last update."""
    ag = getattr(model, "action_generator", None)
    if isinstance(getattr(ag, "_seen", None), list):
        ag._seen = []
