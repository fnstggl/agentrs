"""Durable account-provisioning workflow.

    REQUESTED -> POLICY_CHECKED -> EXISTING_ACCOUNT_FOUND -> ACTIVE
                               \\-> AUTHORIZATION_REQUIRED -> PENDING_HUMAN_SETUP -> CREDENTIALS_CONNECTED
                                                                                 -> VERIFIED -> ACTIVE
    plus DENIED, FAILED (requests) and SUSPENDED / REVOKED (accounts)

* Minimum permissions: only the capabilities asked for (and allowed for the agent) are requested.
* Automatic provisioning happens ONLY when the provider's configuration declares an explicit,
  permitted programmatic flow (``requirements.programmatic_provisioning``; here: SocialWebLab test
  accounts in our own sandbox). Everything else pauses in ``PENDING_HUMAN_SETUP`` with the steps the
  operator must perform on the provider's official interface (sign-up, email / phone / identity
  verification, CAPTCHAs, developer approval). Nothing here automates around those checks.
* A denial is final for that (provider, capabilities, agent): repeated requests return the denied
  request instead of creating new ones.
* Verification uses the connector's non-destructive ``validate_credentials``.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Optional

from sqlalchemy import select

from agentrs.harness.capabilities import CapabilityRegistry
from agentrs.harness.db import Approval, Database, ProvisioningRequest
from agentrs.harness.identity import AgentRegistry, PlatformAccountRegistry

OPEN = {"REQUESTED", "POLICY_CHECKED", "AUTHORIZATION_REQUIRED", "PENDING_HUMAN_SETUP", "CREDENTIALS_CONNECTED",
        "VERIFIED"}


class ProvisioningError(RuntimeError):
    pass


def fingerprint(provider: str, capabilities: list[str], agent: str) -> str:
    return hashlib.sha256(f"{provider}|{','.join(sorted(capabilities))}|{agent}".encode()).hexdigest()[:32]


class AccountProvisioningOrchestrator:
    def __init__(self, db: Database, registry: CapabilityRegistry, agents: AgentRegistry,
                 accounts: PlatformAccountRegistry,
                 validators: Optional[dict[str, Callable[[Any], Any]]] = None,
                 programmatic: Optional[dict[str, Callable[[ProvisioningRequest], dict]]] = None):
        self.db, self.reg, self.agents, self.accounts = db, registry, agents, accounts
        self.validators = validators or {}  # provider -> async/sync fn(account) -> identity dict (raises on failure)
        self.programmatic = programmatic or {}  # provider -> fn(request) -> account fields (permitted flows only)

    # ------------------------------------------------------------------ helpers
    def _move(self, s, req: ProvisioningRequest, state: str, note: str = "") -> None:
        req.history = list(req.history or []) + [{"t": time.time(), "from": req.state, "to": state, "note": note}]
        req.state = state
        req.updated_at = time.time()
        if note:
            req.decision_reason = note

    def get(self, request_id: str) -> ProvisioningRequest:
        with self.db.session() as s:
            r = s.get(ProvisioningRequest, request_id)
            if r is None:
                raise ProvisioningError(f"no request {request_id}")
            return r

    # ------------------------------------------------------------------ entry point
    def request(self, provider: str, required_capabilities: list[str], purpose: str, requesting_agent: str) -> ProvisioningRequest:
        fp = fingerprint(provider, required_capabilities, requesting_agent)
        with self.db.session() as s:
            prev = list(s.scalars(select(ProvisioningRequest).where(ProvisioningRequest.fingerprint == fp)
                                  .order_by(ProvisioningRequest.created_at.desc())))
            for p in prev:
                if p.state == "DENIED":
                    return p  # a denial is not re-litigated by the agent
                if p.state in OPEN or p.state == "ACTIVE" and p.account_id and self._usable(p.account_id):
                    return p
            req = ProvisioningRequest(provider=provider, required_capabilities=sorted(required_capabilities),
                                      purpose=purpose, requesting_agent=requesting_agent, fingerprint=fp,
                                      history=[{"t": time.time(), "to": "REQUESTED"}])
            s.add(req)
            s.flush()
            rid = req.id
        self._policy_check(rid)
        return self.get(rid)

    def _usable(self, account_id: str) -> bool:
        a = self.accounts.get(account_id)
        return a is not None and a.status == "active"

    def _policy_check(self, rid: str) -> None:
        with self.db.session() as s:
            req = s.get(ProvisioningRequest, rid)
            cfg = self.reg.get(req.provider)
            agent = self.agents.get(req.requesting_agent)
            why = None
            if cfg is None:
                why = f"provider {req.provider} is not configured"
            elif agent is None or agent.status != "active":
                why = f"agent {req.requesting_agent} is not active"
            else:
                unknown = [c for c in req.required_capabilities if c not in cfg.capabilities]
                not_allowed = [c for c in req.required_capabilities
                               if not AgentRegistry.has_capability(agent, req.provider, c)]
                if unknown:
                    why = f"capabilities not offered by {req.provider}: {unknown}"
                elif not_allowed:
                    why = f"agent {agent.id} may not hold {req.provider} capabilities {not_allowed}"
                elif not req.purpose.strip():
                    why = "a purpose is required"
            if why:
                self._move(s, req, "DENIED", f"policy: {why}")
                return
            self._move(s, req, "POLICY_CHECKED")
            existing = self.accounts.find_for(req.requesting_agent, req.provider, req.required_capabilities)
            if existing is not None:
                req.account_id = existing.id
                self._move(s, req, "EXISTING_ACCOUNT_FOUND")
                self._move(s, req, "ACTIVE", "reusing an authorised existing account")
                return
            shared = self.accounts.find_for(req.requesting_agent, req.provider, req.required_capabilities,
                                            include_unassigned=True)
            self._move(s, req, "AUTHORIZATION_REQUIRED")
            if shared is not None:  # an account exists but this agent is not on its list: the operator decides
                req.account_id = shared.id
                s.add(Approval(kind="provisioning", ref_id=req.id, requested_by=req.requesting_agent,
                               summary=f"Grant {req.requesting_agent} use of existing {req.provider} account "
                                       f"{shared.external_id} for: {req.purpose}",
                               payload={"action": "grant_existing", "account_id": shared.id,
                                        "capabilities": req.required_capabilities}))
                self._move(s, req, "PENDING_HUMAN_SETUP", "operator must approve sharing an existing account")
                return
            if cfg.requirements.programmatic_provisioning and req.provider in self.programmatic and cfg.enabled:
                fields = self.programmatic[req.provider](req)
                aid = self.accounts.register(provider=req.provider, capabilities=req.required_capabilities,
                                             allowed_agents=[req.requesting_agent], purpose=req.purpose, **fields)
                req.account_id = aid
                self._move(s, req, "CREDENTIALS_CONNECTED", "provider offers a permitted programmatic flow")
                return
            steps = list(cfg.human_setup_steps) or [f"Set up an authorised {req.provider} identity on the provider's "
                                                    "official interface and connect its credentials."]
            minimal = [f"{req.provider}:{c}" for c in req.required_capabilities]
            req.human_steps = steps + [f"Grant only these permissions: {', '.join(minimal)}"]
            reasons = [k for k, v in cfg.requirements.model_dump().items() if v and k != "programmatic_provisioning"]
            s.add(Approval(kind="provisioning", ref_id=req.id, requested_by=req.requesting_agent,
                           summary=f"{req.requesting_agent} needs {req.provider} access ({', '.join(minimal)}) "
                                   f"for: {req.purpose}",
                           payload={"action": "human_setup", "steps": req.human_steps, "requirements": reasons}))
            self._move(s, req, "PENDING_HUMAN_SETUP", "requires: " + ", ".join(reasons))

    # ------------------------------------------------------------------ operator actions
    def deny(self, rid: str, reason: str, operator: str) -> None:
        with self.db.session() as s:
            req = s.get(ProvisioningRequest, rid)
            self._move(s, req, "DENIED", f"denied by {operator}: {reason}")
            for a in s.scalars(select(Approval).where(Approval.ref_id == rid, Approval.status == "pending")):
                a.status, a.decided_by, a.decided_at, a.note = "denied", operator, time.time(), reason

    def approve_existing(self, rid: str, operator: str) -> None:
        with self.db.session() as s:
            req = s.get(ProvisioningRequest, rid)
            if req.state != "PENDING_HUMAN_SETUP" or not req.account_id:
                raise ProvisioningError("nothing to approve")
            self.accounts.grant_agent(req.account_id, req.requesting_agent)
            for a in s.scalars(select(Approval).where(Approval.ref_id == rid, Approval.status == "pending")):
                a.status, a.decided_by, a.decided_at = "approved", operator, time.time()
            self._move(s, req, "VERIFIED", f"sharing approved by {operator}")
            self._move(s, req, "ACTIVE")

    def connect_credentials(self, rid: str, operator: str, external_id: str, credential_refs: dict[str, str],
                            account_type: str, owner_org: str, bot_disclosure: str, oauth_scopes: Optional[list] = None,
                            training_data_permission: str = "not_authorized", rate_limits: Optional[dict] = None) -> str:
        """The operator finished the provider's official setup and supplies secret *references*."""
        with self.db.session() as s:
            req = s.get(ProvisioningRequest, rid)
            if req.state != "PENDING_HUMAN_SETUP":
                raise ProvisioningError(f"request is {req.state}, not waiting for human setup")
            aid = self.accounts.register(provider=req.provider, external_id=external_id, account_type=account_type,
                                         owner_org=owner_org, responsible_operator=operator,
                                         capabilities=req.required_capabilities, credential_refs=credential_refs,
                                         allowed_agents=[req.requesting_agent], purpose=req.purpose,
                                         oauth_scopes=oauth_scopes, bot_disclosure=bot_disclosure,
                                         training_data_permission=training_data_permission, rate_limits=rate_limits)
            req.account_id = aid
            for a in s.scalars(select(Approval).where(Approval.ref_id == rid, Approval.status == "pending")):
                a.status, a.decided_by, a.decided_at, a.note = "approved", operator, time.time(), "credentials connected"
            self._move(s, req, "CREDENTIALS_CONNECTED", f"connected by {operator}")
            return aid

    async def verify(self, rid: str) -> ProvisioningRequest:
        req = self.get(rid)
        if req.state != "CREDENTIALS_CONNECTED":
            raise ProvisioningError(f"request is {req.state}")
        acc = self.accounts.get(req.account_id)
        fn = self.validators.get(req.provider)
        try:
            if fn is None:
                raise ProvisioningError(f"no credential validator for {req.provider}")
            ident = fn(acc)
            if hasattr(ident, "__await__"):
                ident = await ident
        except Exception as e:  # noqa: BLE001 - any validation failure leaves the account unusable
            with self.db.session() as s:
                r = s.get(ProvisioningRequest, rid)
                self._move(s, r, "FAILED", f"credential validation failed: {type(e).__name__}: {e}")
            self.accounts.update(req.account_id, status="pending")
            return self.get(rid)
        self.accounts.update(req.account_id, status="active", last_auth_at=time.time())
        with self.db.session() as s:
            r = s.get(ProvisioningRequest, rid)
            self._move(s, r, "VERIFIED", f"validated: {str(ident)[:200]}")
            self._move(s, r, "ACTIVE")
        return self.get(rid)

    def pending(self) -> list[ProvisioningRequest]:
        with self.db.session() as s:
            return list(s.scalars(select(ProvisioningRequest).where(ProvisioningRequest.state == "PENDING_HUMAN_SETUP")))
