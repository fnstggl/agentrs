"""Workstream B — unified latent intelligence.

* `backbone.UnifiedCore`   — shared backbone F_theta (UnifiedCoreP): Z_t = F(Z_{t-1}, [O; A; B; M; G])
* `reasoning.Reasoner`     — iterative latent reasoning with adaptive halting (ReasonerP)
* `entities.EntityEncoder` — entity and relation tokens for variable-size worlds (EntityEncoderP)
* `beliefs.BeliefModule`   — mechanism particles: weighting, learned proposal, resampling, misspecification (BeliefModuleP)
* `bayes.BayesLastLayer`   — Kalman / normal-inverse-gamma last layer over learned features (BayesLastLayerP)
* `invariance`             — surface-randomised training pairs and the invariance objective (W02)
* `blocks`                 — RMSNorm, SwiGLU, SDPA attention, pre-norm blocks, time encoding, entity-slot codes
"""
from .backbone import DEFAULT_KIND_PRIORITY, UnifiedCore
from .bayes import BayesLastLayer
from .beliefs import BeliefModule, detach_fast, systematic_resample
from .blocks import (Block, CrossBlock, EntitySlotCodes, KindEmbedding, MultiHeadAttention, RMSNorm, SwiGLU,
                     TimeEncoding, count_parameters)
from .entities import EntityEncoder, entity_type_tokens
from .invariance import InvarianceLoss, SurfaceMap, SurfaceRandomizer, invariance_loss_from_views
from .reasoning import Reasoner

__all__ = ["UnifiedCore", "DEFAULT_KIND_PRIORITY", "Reasoner", "EntityEncoder", "entity_type_tokens", "BeliefModule",
           "detach_fast", "systematic_resample", "BayesLastLayer", "InvarianceLoss", "SurfaceMap", "SurfaceRandomizer",
           "invariance_loss_from_views", "Block", "CrossBlock", "EntitySlotCodes", "KindEmbedding",
           "MultiHeadAttention", "RMSNorm", "SwiGLU", "TimeEncoding", "count_parameters"]
