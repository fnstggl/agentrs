"""Typed actions, results and permission tiers shared by every harness service.

The model never talks to the network: it proposes an ``ActionRequest`` (a provider, an operation
declared by that provider's connector, parameters and optional generated content), and the tool
gateway decides whether it may run.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from enum import IntEnum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Tier(IntEnum):
    """Permission tiers. An agent can never raise its own tier."""

    OBSERVE = 0  # approved public / authorised reads, no external mutation
    SANDBOX = 1  # actions inside owned test environments
    REVERSIBLE = 2  # explicitly permitted operations on owned resources
    EXTERNAL = 3  # publishing / communication through an approved identity (task-level authorisation)
    HIGH_IMPACT = 4  # purchases, ownership / security changes, irreversible operations: human approval


class OperationSpec(BaseModel):
    """An operation a connector supports. ``params`` lists required parameter names."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tier: Tier
    mutates: bool
    params: list[str] = Field(default_factory=list)
    optional_params: list[str] = Field(default_factory=list)
    content_param: Optional[str] = None  # parameter that carries generated content (duplicate / review checks)
    capability: str = ""  # capability key in the provider capability registry (defaults to name)
    rate_key: str = ""  # rate-limit bucket (defaults to "write" for mutations, "read" otherwise)
    reversible: bool = True
    description: str = ""

    def cap(self) -> str:
        return self.capability or self.name

    def bucket(self) -> str:
        return self.rate_key or ("write" if self.mutates else "read")


class ActionRequest(BaseModel):
    action_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    agent_id: str
    task_id: Optional[str] = None
    provider: str
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    account_id: Optional[str] = None
    model_checkpoint: Optional[str] = None
    proposed_at: float = Field(default_factory=time.time)
    # generation record (exact tokens / behaviour log-probs) when the content came from the policy
    generation: Optional[dict[str, Any]] = None
    reason: str = ""

    def content(self, spec: OperationSpec) -> Optional[str]:
        if spec.content_param:
            v = self.params.get(spec.content_param)
            return None if v is None else str(v)
        return None

    def idempotency_key(self) -> str:
        return hashlib.sha256(f"{self.agent_id}|{self.action_id}".encode()).hexdigest()[:32]


Status = Literal["executed", "denied", "pending_approval", "failed", "rejected", "rate_limited"]


class ActionResult(BaseModel):
    action_id: str
    status: Status
    reason: str = ""
    tier: Optional[int] = None
    provider_object_id: Optional[str] = None  # e.g. post URI, PR number, message id (for follow-ups)
    observation: dict[str, Any] = Field(default_factory=dict)  # untrusted provider output
    approval_id: Optional[str] = None
    executed_at: Optional[float] = None


class Untrusted(BaseModel):
    """Text that came from the outside world (pages, posts, emails). It is data for the model, never
    an instruction to the harness: the gateway's decisions never depend on it."""

    source: str
    text: str
    marker: str = "UNTRUSTED_EXTERNAL_CONTENT"

    def render(self) -> str:
        return f"<<{self.marker} source={self.source}>>\n{self.text}\n<<END_{self.marker}>>"
