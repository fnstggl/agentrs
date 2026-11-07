"""Typed, versioned record of an action taken by another actor and observed by a sensor.

Missing or uncertain information is explicit: unknown goals stay ``None`` with ``goal_known=False``;
actions inferred rather than logged carry ``action.inferred=True`` and a confidence; context that
was not fully available is flagged ``context_complete=False``; ``missing`` lists what is unknown and
why. Nothing is filled in by guessing.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

OBSERVED_SCHEMA_VERSION = 1

ActorType = Literal["human", "simulated_user", "software_agent", "system", "unknown"]


class ObservedAction(BaseModel):
    operation: str
    content: str | None = None  # complete content (never a summary); None if not observable
    parameters: dict[str, Any] = Field(default_factory=dict)
    inferred: bool = False  # True if reconstructed (e.g. by inverse dynamics) rather than logged
    confidence: float | None = None

    def to_action_str(self) -> str:
        from agentrs.actions.schema import Action

        if self.content is None:
            return f"{self.operation}()"
        return Action.submit(self.operation, self.content[:65_000]).to_str()


class ObservedActorAction(BaseModel):
    schema_version: int = OBSERVED_SCHEMA_VERSION
    observation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_id: str
    provider_object_id: str | None = None  # stable id at the provider (dedup, follow-ups, deletion)
    actor_type: ActorType = "unknown"
    actor_ref: str | None = None  # pseudonymous (keyed hash), never a raw account name
    action_time: float | None = None
    action_time_known: bool = False
    observation_time: float
    time_base: str
    context_before: dict[str, Any] = Field(default_factory=dict)
    context_complete: bool = False
    action: ObservedAction
    goal: str | None = None
    goal_known: bool = False
    observable_state_after: dict[str, Any] = Field(default_factory=dict)
    outcome_status: Literal["pending", "matured", "censored", "revoked"] = "pending"
    provenance: dict[str, Any] = Field(default_factory=dict)
    permission_metadata: dict[str, Any] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    env_id: str = ""
    env_version: str = ""
    group: str = ""  # stratum: environment/app/action type

    @property
    def content_hash(self) -> str:
        return hashlib.sha256((self.action.operation + "\x00" + (self.action.content or "")).encode()).hexdigest()

    def effective_action_time(self) -> float:
        return self.action_time if self.action_time is not None else self.observation_time

    def context_text(self) -> str:
        """Canonical text of the observed context (shared format with our own actions)."""
        return context_text(self.context_before, self.effective_action_time(), self.time_base)


def context_text(ctx: dict[str, Any], at: float, time_base: str) -> str:
    lines = []
    env = ctx.get("env")
    if env:
        lines.append(f"ENV: {env}" + (f" space={ctx['space']}" if ctx.get("space") else ""))
    if ctx.get("topic"):
        lines.append(f"TOPIC: {ctx['topic']}")
    if "author_followers" in ctx:
        lines.append(f"AUTHOR: followers={ctx['author_followers']}")
    dt = datetime.fromtimestamp(at, tz=UTC)
    lines.append(f"TIME: {dt.strftime('%a %H:%M')} UTC" + (" (simulated clock)" if time_base.startswith("virtual") else ""))
    for k, v in sorted(ctx.items()):
        if k in ("env", "space", "topic", "author_followers") or isinstance(v, (dict, list)):
            continue
        lines.append(f"{k.upper()}: {v}")
    return "\n".join(lines)


def pseudonym(raw: str, key: str) -> str:
    return "actor-" + hashlib.blake2b(raw.encode(), key=key.encode()[:64], digest_size=6).hexdigest()


def canonical_json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))
