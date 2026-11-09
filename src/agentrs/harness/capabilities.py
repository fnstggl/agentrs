"""Provider capability registry (``config/platforms.yaml``).

Every provider defaults to **disabled** with every capability off. Values in the shipped file are
placeholders; an operator enables a provider and a capability only after confirming, from the
provider's current official documentation and their own approved developer access, that the
operation is permitted for their account. Technical API availability is never treated as
authorisation. Anything missing fails closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProviderRequirements(BaseModel):
    model_config = ConfigDict(extra="allow")

    oauth: bool = True
    human_account_creation: bool = True
    developer_approval: bool = True
    bot_disclosure: bool = True
    training_data_authorization: bool = True
    programmatic_provisioning: bool = False  # the provider explicitly offers a permitted provisioning API


class ProviderLimits(BaseModel):
    model_config = ConfigDict(extra="allow")

    requests_per_minute: float = 0
    posts_per_day: float = 0
    concurrent_sessions: int = 0


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    enabled: bool = False
    base_url: Optional[str] = None
    capabilities: dict[str, bool] = Field(default_factory=dict)
    requirements: ProviderRequirements = Field(default_factory=ProviderRequirements)
    limits: ProviderLimits = Field(default_factory=ProviderLimits)
    training_data_authorized: bool = False  # explicit permission to train on data from this provider
    authorization_ref: Optional[str] = None  # agreement / approval id backing ``enabled``
    allowed_domains: list[str] = Field(default_factory=list)
    human_setup_steps: list[str] = Field(default_factory=list)
    notes: str = ""

    def allows(self, capability: str) -> bool:
        return self.enabled and bool(self.capabilities.get(capability, False))


class CapabilityRegistry:
    def __init__(self, providers: dict[str, ProviderConfig]):
        self.providers = providers
        self.overrides: dict[str, bool] = {}  # runtime disable switches from the operator dashboard

    @classmethod
    def load(cls, path: str | Path) -> "CapabilityRegistry":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        provs = {}
        for name, cfg in (raw.get("providers") or {}).items():
            provs[name] = ProviderConfig(provider=name, **(cfg or {}))
        return cls(provs)

    def get(self, provider: str) -> Optional[ProviderConfig]:
        return self.providers.get(provider)

    def enabled(self, provider: str) -> bool:
        p = self.providers.get(provider)
        return bool(p and p.enabled and self.overrides.get(provider, True))

    def allows(self, provider: str, capability: str) -> tuple[bool, str]:
        p = self.providers.get(provider)
        if p is None:
            return False, f"provider {provider!r} is not configured"
        if not p.enabled:
            return False, f"provider {provider!r} is disabled"
        if self.overrides.get(provider) is False:
            return False, f"provider {provider!r} was disabled by an operator"
        if not p.capabilities.get(capability, False):
            return False, f"capability {capability!r} is not enabled for {provider!r}"
        return True, ""

    def disable(self, provider: str) -> None:
        self.overrides[provider] = False

    def enable_override(self, provider: str) -> None:
        self.overrides.pop(provider, None)
