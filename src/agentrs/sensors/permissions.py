"""Ingestion permissions: what may enter the training corpus, and how fast it may be collected.

A source is ingested for world-model training only if its declaration says training is
authorised, an authorisation reference exists (except for sources we operate ourselves:
self-hosted simulated environments, our own agent traces, consented demonstrations), the
authorisation has not expired, and the record's data categories are allowed and not prohibited.
Text is PII-redacted when the declaration asks for it.
"""

from __future__ import annotations

import re
import time

from agentrs.sensors.source_adapter import SourceDeclaration

SELF_OPERATED = {"self_hosted_feed", "agent_traces", "consented_demos"}

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def redact_pii(text: str | None) -> str | None:
    if text is None:
        return None
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub("[number]", text)
    return _PHONE.sub("[phone]", text)


class IngestionRefused(PermissionError):
    pass


def check_source(decl: SourceDeclaration, purpose: str = "world_model_training",
                 now_wall: float | None = None) -> tuple[bool, list[str]]:
    reasons = []
    if purpose in ("world_model_training", "policy_training") and not decl.training_authorized:
        reasons.append("training use is not authorised for this source")
    if decl.access_method not in SELF_OPERATED and not decl.authorization_ref:
        reasons.append("external source without an authorisation reference")
    if decl.access_method == "self_hosted_feed" and not decl.simulated and not decl.authorization_ref:
        reasons.append("self-hosted feed with real users needs an authorisation reference (consent / terms)")
    if decl.valid_until_wall is not None and (now_wall or time.time()) > decl.valid_until_wall:
        reasons.append("authorisation expired")
    bad = set(decl.allowed_data_categories) & set(decl.prohibited_categories)
    if bad:
        reasons.append(f"declaration allows prohibited categories {sorted(bad)}")
    return (not reasons), reasons


def check_record(decl: SourceDeclaration, categories: list[str]) -> tuple[bool, list[str]]:
    reasons = []
    for c in categories:
        if c in decl.prohibited_categories:
            reasons.append(f"prohibited category {c}")
        elif c not in decl.allowed_data_categories:
            reasons.append(f"category {c} not allowed by the source declaration")
    return (not reasons), reasons


class TokenBucket:
    """Rate limit in the source's own clock."""

    def __init__(self, per_minute: float, capacity: float | None = None):
        self.rate = per_minute / 60.0
        self.capacity = capacity if capacity is not None else max(1.0, per_minute)
        self.tokens = self.capacity
        self.t: float | None = None
        self.refused = 0

    def take(self, now: float, n: float = 1.0) -> bool:
        if self.t is not None:
            self.tokens = min(self.capacity, self.tokens + max(0.0, now - self.t) * self.rate)
        self.t = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        self.refused += 1
        return False
