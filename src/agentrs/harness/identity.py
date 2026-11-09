"""Agent identities (internal) and the platform account registry (external).

An agent's internal identity (``agents`` table, loaded from ``config/agents.yaml``) is separate from
the external accounts it may use. Many workers can share one properly authorised organisation or
bot account; an additional account is only registered for a genuinely distinct approved purpose,
never to multiply reach or work around a provider's limits.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update

from agentrs.harness.connectors.base import CredentialHandle
from agentrs.harness.db import Agent, Database, PlatformAccount
from agentrs.harness.secrets import SecretStore


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    role: str
    model_checkpoint: str = "current_approved"
    allowed_capabilities: list[str] = Field(default_factory=list)  # "<provider>:<capability>" or "<provider>:*"
    max_tier: int = 0
    action_budget: int = 100


class AgentRegistry:
    def __init__(self, db: Database):
        self.db = db

    def load_yaml(self, path: str | Path) -> list[str]:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        ids = []
        for a in raw.get("agents", []):
            spec = AgentSpec(**a)
            self.upsert(spec)
            ids.append(spec.id)
        return ids

    def upsert(self, spec: AgentSpec) -> None:
        with self.db.session() as s:
            row = s.get(Agent, spec.id)
            if row is None:
                s.add(Agent(id=spec.id, display_name=spec.display_name, role=spec.role,
                            model_checkpoint=spec.model_checkpoint, allowed_capabilities=spec.allowed_capabilities,
                            max_tier=spec.max_tier, action_budget=spec.action_budget))
            else:  # configuration is the source of truth for permissions; usage counters are kept
                row.display_name, row.role = spec.display_name, spec.role
                row.model_checkpoint = spec.model_checkpoint
                row.allowed_capabilities, row.max_tier, row.action_budget = (spec.allowed_capabilities, spec.max_tier,
                                                                             spec.action_budget)

    def get(self, agent_id: str) -> Optional[Agent]:
        with self.db.session() as s:
            return s.get(Agent, agent_id)

    def all(self) -> list[Agent]:
        with self.db.session() as s:
            return list(s.scalars(select(Agent).order_by(Agent.id)))

    def set_status(self, agent_id: str, status: str) -> None:
        with self.db.session() as s:
            s.execute(update(Agent).where(Agent.id == agent_id).values(status=status))

    def consume_budget(self, agent_id: str, n: int = 1) -> bool:
        """Atomically use ``n`` actions of the agent's budget; False if it would exceed it."""
        with self.db.session() as s:
            res = s.execute(update(Agent).where(Agent.id == agent_id, Agent.actions_used + n <= Agent.action_budget)
                            .values(actions_used=Agent.actions_used + n, last_action_at=time.time()))
            return res.rowcount == 1

    @staticmethod
    def has_capability(agent: Agent, provider: str, capability: str) -> bool:
        caps = set(agent.allowed_capabilities or [])
        return f"{provider}:{capability}" in caps or f"{provider}:*" in caps


class AccountError(PermissionError):
    pass


class PlatformAccountRegistry:
    def __init__(self, db: Database, secrets: SecretStore, runtime_secret_backend: str = "file"):
        self.db = db
        self.secrets = secrets
        self.backend = runtime_secret_backend  # where refreshed OAuth tokens are written

    def register(self, provider: str, external_id: str, account_type: str, owner_org: str, responsible_operator: str,
                 capabilities: list[str], credential_refs: dict[str, str], allowed_agents: list[str],
                 purpose: str, oauth_scopes: Optional[list[str]] = None, bot_disclosure: str = "unknown",
                 training_data_permission: str = "not_authorized", rate_limits: Optional[dict] = None,
                 status: str = "pending", reauth_by: Optional[float] = None, distinct_purpose_ok: bool = False,
                 connection: Optional[dict] = None) -> str:
        for ref in credential_refs.values():
            if not ref.startswith("secret://"):
                raise AccountError("credentials must be secret references, never raw values")
        with self.db.session() as s:
            same = list(s.scalars(select(PlatformAccount).where(PlatformAccount.provider == provider,
                                                                PlatformAccount.status.in_(["pending", "verified",
                                                                                            "active"]))))
            overlap = [a for a in same if set(a.capabilities or []) & set(capabilities) and a.external_id != external_id]
            if overlap and not distinct_purpose_ok:
                raise AccountError(f"{provider}: an account with overlapping capabilities already exists "
                                   f"({overlap[0].id}); register another only for a distinct approved purpose "
                                   "(distinct_purpose_ok=True, recorded by the operator)")
            row = PlatformAccount(provider=provider, external_id=external_id, account_type=account_type,
                                  owner_org=owner_org, responsible_operator=responsible_operator,
                                  capabilities=capabilities, credential_refs=credential_refs,
                                  allowed_agents=allowed_agents, purpose=purpose, oauth_scopes=oauth_scopes or [],
                                  bot_disclosure=bot_disclosure, training_data_permission=training_data_permission,
                                  rate_limits=rate_limits or {}, status=status, reauth_by=reauth_by,
                                  connection=connection or {})
            s.add(row)
            s.flush()
            return row.id

    def get(self, account_id: str) -> Optional[PlatformAccount]:
        with self.db.session() as s:
            return s.get(PlatformAccount, account_id)

    def list(self, provider: Optional[str] = None) -> list[PlatformAccount]:
        with self.db.session() as s:
            q = select(PlatformAccount)
            if provider:
                q = q.where(PlatformAccount.provider == provider)
            return list(s.scalars(q.order_by(PlatformAccount.created_at)))

    def find_for(self, agent_id: str, provider: str, capabilities: list[str],
                 include_unassigned: bool = False) -> Optional[PlatformAccount]:
        for a in self.list(provider):
            if a.status != "active" or a.revoked_at:
                continue
            if a.reauth_by and a.reauth_by < time.time():
                continue
            if not set(capabilities) <= set(a.capabilities or []):
                continue
            if agent_id in (a.allowed_agents or []) or "*" in (a.allowed_agents or []) or include_unassigned:
                return a
        return None

    def update(self, account_id: str, **fields: Any) -> None:
        with self.db.session() as s:
            row = s.get(PlatformAccount, account_id)
            if row is None:
                raise AccountError(f"no account {account_id}")
            for k, v in fields.items():
                setattr(row, k, v)

    def grant_agent(self, account_id: str, agent_id: str) -> None:
        a = self.get(account_id)
        self.update(account_id, allowed_agents=sorted(set(a.allowed_agents or []) | {agent_id}))

    def revoke(self, account_id: str, reason: str) -> None:
        """Immediately unusable: status revoked, secrets deleted from writable stores."""
        a = self.get(account_id)
        if a is None:
            return
        for ref in (a.credential_refs or {}).values():
            try:
                self.secrets.delete(ref)
            except Exception:  # noqa: BLE001 - env-injected secrets are removed by the operator's secret manager
                pass
        self.update(account_id, status="revoked", revoked_at=time.time(), credential_refs={},
                    purpose=(a.purpose + f"\n[revoked: {reason}]"))

    def suspend(self, account_id: str, reason: str) -> None:
        a = self.get(account_id)
        self.update(account_id, status="suspended", purpose=(a.purpose + f"\n[suspended: {reason}]"))

    def credential_handle(self, account: PlatformAccount) -> CredentialHandle:
        def on_update(name: str, value: str) -> str:
            ref = self.secrets.put(self.backend, f"{account.id}.{name}", value)
            refs = dict(self.get(account.id).credential_refs or {})
            refs[name] = ref
            self.update(account.id, credential_refs=refs)
            return ref

        return CredentialHandle(account.credential_refs or {}, self.secrets.get, on_update)
