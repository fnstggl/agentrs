"""Unified, versioned experience schema.

Every real interaction is a ``Transition``. Imagined (world-model-generated)
transitions use the same schema with ``source=Source.IMAGINED`` and are stored
in a separate partition; they can never carry verified rewards.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from agentrs.content.semantic_action import SemanticAction
from agentrs.environments.base import Observation

SCHEMA_VERSION = 3


class Source(str, Enum):
    AGENT = "agent"  # the learning agent's own real interaction
    TEACHER = "teacher"  # another (scripted / AI) agent
    HUMAN = "human"  # consenting human demonstrator
    DATASET = "dataset"  # licensed / permissioned external dataset
    INFERRED = "inferred"  # action labels inferred by inverse dynamics
    IMAGINED = "imagined"  # world-model rollouts (never real)


REAL_SOURCES = {Source.AGENT, Source.TEACHER, Source.HUMAN, Source.DATASET, Source.INFERRED}


class Provenance(BaseModel):
    source: Source
    source_id: str = ""  # demonstrator / agent id
    consent: bool = False  # explicit consent recorded (required for HUMAN)
    license: str = "internal"
    privacy: str = "internal"  # "private" | "internal" | "shareable"
    collected_by: str = ""
    notes: str = ""


class WMPredictionRecord(BaseModel):
    """What the world model predicted *before* the real step was taken."""

    event_probs: list[float] = Field(default_factory=list)
    reward_prob: Optional[float] = None
    terminal_prob: Optional[float] = None
    diff_text: Optional[str] = None
    epistemic: Optional[float] = None
    checkpoint: Optional[str] = None


class Transition(BaseModel):
    schema_version: int = SCHEMA_VERSION
    episode_id: str
    step: int
    task_id: str
    env_id: str
    env_version: str = ""
    family: str = ""
    split: str = "train"
    goal: str
    observation: Observation
    belief_prompt: Optional[str] = None  # serialized belief state given to the policy
    next_belief_prompt: Optional[str] = None  # only on an episode's last transition (for bootstrapping)
    action: str  # canonical action string
    action_type: str = ""
    candidates: list[str] = Field(default_factory=list)  # action-mask candidates offered to the policy
    chosen_index: Optional[int] = None
    policy_logprob: Optional[float] = None  # log pi_theta(a|s) under the acting checkpoint
    behavior_logprob: Optional[float] = None  # log mu(a|s) of the actual selection rule (planner, eps)
    selection: str = "policy"  # policy | planner | teacher | human | random | skill
    wm_prediction: Optional[WMPredictionRecord] = None
    next_observation: Observation
    diff_text: str = ""
    events: list[float] = Field(default_factory=list)
    action_valid: bool = True
    error: Optional[str] = None
    reward: float = 0.0  # environment reward for this step (verified only at termination)
    progress: Optional[float] = None  # objectively verifiable progress after the step
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = ""
    episode_success: Optional[bool] = None  # filled in when the episode ends (verified)
    checkpoints: dict[str, str] = Field(default_factory=dict)  # component -> checkpoint id
    timestamp: float = Field(default_factory=time.time)
    provenance: Provenance
    verification: Optional[dict[str, Any]] = None
    action_confidence: Optional[float] = None  # for inferred labels
    # content actions: complete content, generation log-probs, permissions (schema v3)
    semantic_action: Optional[SemanticAction] = None
    # environment-reported content outcome (format, objective, partial credit, response, engagement,
    # confounders). Engagement is recorded, never used as reward.
    content_outcome: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _grounding(self):
        if self.provenance.source == Source.IMAGINED:
            if self.verification is not None or self.episode_success is not None:
                raise ValueError("imagined transitions cannot carry verified outcomes")
        if self.provenance.source == Source.HUMAN and not self.provenance.consent:
            raise ValueError("human demonstrations require recorded consent")
        return self

    @property
    def is_real(self) -> bool:
        return self.provenance.source in REAL_SOURCES


class EpisodeRecord(BaseModel):
    episode_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    task_id: str
    env_id: str
    family: str
    split: str
    goal: str
    transitions: list[Transition] = Field(default_factory=list)
    success: Optional[bool] = None
    verification: Optional[dict[str, Any]] = None
    termination_reason: str = ""
    num_steps: int = 0
    num_invalid: int = 0
    wall_time: float = 0.0
    provenance: Optional[Provenance] = None
    checkpoints: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
