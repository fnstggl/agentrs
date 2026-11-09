"""Platform connector framework.

A connector adapts one provider (or the self-hosted lab, or a generic allowlisted website) to the
harness. It declares exactly which operations it supports (``operations``); everything else is
unsupported. It never decides permissions (the tool gateway does) and never exposes credentials:
it receives a ``CredentialHandle`` that yields auth material only inside its own calls.

Outcome payloads returned by ``fetch_outcome`` use the temporal ledger's format:
``{"counts": {...}, "state": str, "occurred": {event: bool}, "occurred_at": {event: t},
"continuous": {...}, "raw": {...}}`` so that real actions are followed up and learned from exactly
like sandbox actions.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentrs.harness.capabilities import ProviderConfig
from agentrs.harness.net import SafeHTTPClient
from agentrs.harness.types import ActionRequest, ActionResult, OperationSpec


class ConnectorError(RuntimeError):
    def __init__(self, msg: str, status: int = 0, retry_after: Optional[float] = None):
        super().__init__(msg)
        self.status = status
        self.retry_after = retry_after


class CredentialHandle:
    """Resolves the account's credential references on demand (inside connector calls only)."""

    def __init__(self, refs: dict[str, str], resolver: Callable[[str], str], on_update=None):
        self._refs = dict(refs)
        self._resolve = resolver
        self._on_update = on_update  # persist refreshed tokens (e.g. secret store + account record)

    def get(self, name: str) -> str:
        if name not in self._refs:
            raise ConnectorError(f"credential {name!r} is not connected for this account", 401)
        return self._resolve(self._refs[name])

    def has(self, name: str) -> bool:
        return name in self._refs

    def update(self, name: str, value: str) -> None:
        if self._on_update is None:
            raise ConnectorError("this account's credentials cannot be updated at runtime")
        self._refs[name] = self._on_update(name, value)

    def __repr__(self) -> str:  # never print secrets
        return f"CredentialHandle({sorted(self._refs)})"


@dataclass
class ConnectorContext:
    config: ProviderConfig
    http: SafeHTTPClient
    credentials: CredentialHandle
    account: dict[str, Any] = field(default_factory=dict)
    base_url: str = ""


@dataclass
class Subscription:
    mode: str  # webhook | stream | poll
    event_types: list[str]
    detail: dict[str, Any] = field(default_factory=dict)


class PlatformConnector(abc.ABC):
    provider: str = "abstract"
    operations: dict[str, OperationSpec] = {}
    outcome_profile: dict[str, str] = {}  # operation -> horizon profile name

    def __init__(self, ctx: ConnectorContext):
        self.ctx = ctx
        self.base_url = (ctx.base_url or ctx.config.base_url or "").rstrip("/")
        self.last_rate_limits: dict[str, Any] = {}

    # ---------------------------------------------------------------- required interface
    async def capabilities(self) -> dict[str, Any]:
        """Operations this connector implements, and whether the registry currently enables each."""
        return {name: {"tier": int(s.tier), "mutates": s.mutates, "enabled": self.ctx.config.allows(s.cap())}
                for name, s in self.operations.items()}

    @abc.abstractmethod
    async def validate_credentials(self) -> dict[str, Any]:
        """Minimal non-destructive authenticated request (e.g. 'who am I')."""

    async def observe(self, query: dict[str, Any]) -> dict[str, Any]:
        op = query.get("operation")
        if op not in self.operations or self.operations[op].mutates:
            raise ConnectorError(f"{op!r} is not a read operation of {self.provider}")
        return await self._dispatch(op, query.get("params", {}))

    async def execute(self, action: ActionRequest) -> ActionResult:
        if action.operation not in self.operations:
            return ActionResult(action_id=action.action_id, status="rejected",
                                reason=f"{self.provider} does not support {action.operation!r}")
        out = await self._dispatch(action.operation, action.params)
        return ActionResult(action_id=action.action_id, status="executed", provider_object_id=out.pop("object_id", None),
                            observation=out, executed_at=time.time(), tier=int(self.operations[action.operation].tier))

    async def subscribe(self, event_types: list[str]) -> Subscription:
        return Subscription(mode="poll", event_types=event_types)

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        raise ConnectorError(f"{self.provider} cannot fetch outcomes")

    async def get_rate_limits(self) -> dict[str, Any]:
        return {"configured": self.ctx.config.limits.model_dump(), "observed": dict(self.last_rate_limits)}

    async def revoke(self) -> dict[str, Any]:
        return {"revoked_at_provider": False, "reason": "provider has no revocation endpoint; local credentials removed"}

    # ---------------------------------------------------------------- helpers
    async def _dispatch(self, op: str, params: dict[str, Any]) -> dict[str, Any]:
        spec = self.operations[op]
        missing = [p for p in spec.params if p not in params]
        if missing:
            raise ConnectorError(f"{op}: missing parameters {missing}", 400)
        unknown = [p for p in params if p not in spec.params and p not in spec.optional_params]
        if unknown:
            raise ConnectorError(f"{op}: unknown parameters {unknown}", 400)
        fn = getattr(self, f"op_{op}", None)
        if fn is None:
            raise ConnectorError(f"{self.provider}.{op} is declared but not implemented", 501)
        return await fn(**params)

    def _note_rate_limits(self, headers) -> None:
        for k, v in headers.items():
            lk = k.lower()
            if "ratelimit" in lk or lk == "retry-after":
                self.last_rate_limits[lk] = v

    async def _json(self, method: str, path: str, *, headers=None, **kw) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        r = await self.ctx.http.request(method, url, headers=headers or {}, **kw)
        self._note_rate_limits(r.headers)
        if r.status_code == 429:
            ra = r.headers.get("retry-after")
            raise ConnectorError(f"{self.provider}: rate limited", 429, float(ra) if ra and ra.isdigit() else None)
        if r.status_code >= 400:
            raise ConnectorError(f"{self.provider} {method} {path}: HTTP {r.status_code}: {r.text[:300]}", r.status_code)
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {"text": r.text}
