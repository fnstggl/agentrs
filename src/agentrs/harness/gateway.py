"""Tool gateway: the only path from an agent's proposal to the outside world.

    proposal -> validate schema -> capability -> account authorisation -> policy, rate limits,
    content -> provider policy hook -> approval requirement -> execute -> record the real outcome

The model never receives credentials, cookies or a raw network client; observations it gets back
are wrapped as untrusted content. Every decision (including denials) is written to the audit log.
Executed mutating actions are registered with the outcome tracker for delayed follow-up.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from sqlalchemy import select

from agentrs.harness.capabilities import CapabilityRegistry
from agentrs.harness.connectors.base import ConnectorContext, ConnectorError, PlatformConnector
from agentrs.harness.db import ActionLog, Approval, Database, PlatformAccount, Task
from agentrs.harness.identity import AgentRegistry, PlatformAccountRegistry
from agentrs.harness.net import EgressPolicy, SafeHTTPClient
from agentrs.harness.policy import PermissionEngine, content_hash, summarize_action
from agentrs.harness.secrets import SecretStore
from agentrs.harness.types import ActionRequest, ActionResult, OperationSpec, Untrusted

ConnectorFactory = Callable[[str, ConnectorContext], PlatformConnector]


class ToolGateway:
    def __init__(self, db: Database, registry: CapabilityRegistry, agents: AgentRegistry,
                 accounts: PlatformAccountRegistry, policy: PermissionEngine, secrets: SecretStore,
                 connector_classes: dict[str, type], egress: EgressPolicy, outcomes=None,
                 transports: Optional[dict[str, Any]] = None, metrics=None, tracer=None,
                 context_fn: Optional[Callable[[ActionRequest, Task], dict]] = None):
        self.db, self.reg, self.agents, self.accounts = db, registry, agents, accounts
        self.policy, self.secrets, self.classes, self.egress = policy, secrets, connector_classes, egress
        self.outcomes = outcomes
        self.transports = transports or {}  # provider -> httpx transport (simulators in tests)
        self.metrics, self.tracer = metrics, tracer
        self.context_fn = context_fn  # (action, task) -> {"context_text", "time_base", "action_time", "forecast"}
        self.egress_log: list[dict] = []
        self._connectors: dict[str, PlatformConnector] = {}

    # ------------------------------------------------------------------ connectors
    def spec(self, provider: str, operation: str) -> Optional[OperationSpec]:
        cls = self.classes.get(provider)
        return None if cls is None else cls.operations.get(operation)

    def connector_for(self, account: PlatformAccount) -> PlatformConnector:
        c = self._connectors.get(account.id)
        if c is not None:
            return c
        cfg = self.reg.get(account.provider)
        http = SafeHTTPClient(self.egress, provider=account.provider, provider_domains=cfg.allowed_domains or None,
                              egress_log=self.egress_log, transport=self.transports.get(account.provider))
        conn = dict(account.connection or {})
        ctx = ConnectorContext(config=cfg, http=http, credentials=self.accounts.credential_handle(account),
                               account={**conn, "id": account.id, "external_id": account.external_id,
                                        "capabilities": account.capabilities, "bot_disclosure": account.bot_disclosure},
                               base_url=conn.get("base_url") or cfg.base_url or "")
        c = self.classes[account.provider](ctx)
        self._connectors[account.id] = c
        return c

    def drop_connector(self, account_id: str) -> None:
        self._connectors.pop(account_id, None)

    # ------------------------------------------------------------------ main entry
    def _account(self, action: ActionRequest, spec: OperationSpec) -> Optional[PlatformAccount]:
        if action.account_id:
            a = self.accounts.get(action.account_id)
            return a
        return self.accounts.find_for(action.agent_id, action.provider, [spec.cap()])

    def _log(self, action: ActionRequest, status: str, reason: str, tier: Optional[int] = None,
             account_id: Optional[str] = None, spec: Optional[OperationSpec] = None, **extra: Any) -> None:
        content = action.content(spec) if spec else None
        safe_params = summarize_action(action, spec)["params"] if spec else {}
        with self.db.session() as s:
            row = s.get(ActionLog, action.action_id)
            if row is None:
                row = ActionLog(id=action.action_id, agent_id=action.agent_id, task_id=action.task_id,
                                provider=action.provider, operation=action.operation, params=safe_params,
                                content_hash=content_hash(content) if content else None,
                                model_checkpoint=action.model_checkpoint, generation=action.generation,
                                proposed_at=action.proposed_at, status=status)
                s.add(row)
            row.status, row.reason, row.tier, row.account_id = status, self.secrets.redact(reason), tier, account_id
            for k, v in extra.items():
                setattr(row, k, v)
        if self.metrics is not None:
            self.metrics.action(action.provider, action.operation, status)

    async def submit(self, action: ActionRequest, approved: bool = False, context: Optional[dict] = None) -> ActionResult:
        spec = self.spec(action.provider, action.operation)
        if spec is None:
            self._log(action, "rejected", f"unknown operation {action.provider}:{action.operation}")
            return ActionResult(action_id=action.action_id, status="rejected", reason="unknown provider or operation")
        missing = [p for p in spec.params if p not in action.params]
        unknown = [p for p in action.params if p not in spec.params and p not in spec.optional_params]
        if missing or unknown:
            why = f"invalid parameters (missing {missing}, unknown {unknown})"
            self._log(action, "rejected", why, int(spec.tier), spec=spec)
            return ActionResult(action_id=action.action_id, status="rejected", reason=why, tier=int(spec.tier))
        agent = self.agents.get(action.agent_id)
        with self.db.session() as s:
            task = s.get(Task, action.task_id) if action.task_id else None
        account = self._account(action, spec)
        tier_override = None
        cls = self.classes.get(action.provider)
        if hasattr(cls, "tier_for"):  # connector-specific tiering (e.g. per-site profiles, high-impact controls)
            tier_override = int(cls.tier_for(action, spec))
        dec = self.policy.evaluate(action, spec, agent, task, account, approved=approved, tier_override=tier_override)
        acc_id = account.id if account else None
        if dec.outcome == "deny":
            self._log(action, "denied", dec.reason, dec.tier, acc_id, spec)
            return ActionResult(action_id=action.action_id, status="denied", reason=dec.reason, tier=dec.tier)
        connector = self.connector_for(account)
        hook = getattr(connector, "provider_policy", None)
        if hook is not None:
            with self.db.session() as s:
                recent = [{"operation": r.operation, "content_hash": r.content_hash, "executed_at": r.executed_at,
                           "params": r.params} for r in s.scalars(select(ActionLog).where(
                               ActionLog.account_id == account.id, ActionLog.status == "executed",
                               ActionLog.executed_at > time.time() - 86400))]
            ok, why = hook(action, recent)
            if not ok:
                self._log(action, "denied", f"provider policy: {why}", dec.tier, acc_id, spec)
                return ActionResult(action_id=action.action_id, status="denied", reason=f"provider policy: {why}",
                                    tier=dec.tier)
        if dec.outcome == "needs_approval":
            with self.db.session() as s:
                ap = Approval(kind="high_impact" if dec.tier >= 4 else "action", ref_id=action.action_id,
                              requested_by=action.agent_id, summary=f"{action.agent_id}: {action.provider}:"
                              f"{action.operation} (tier {dec.tier}) for task {action.task_id}",
                              payload={"action": action.model_dump(mode="json"), "summary": summarize_action(action, spec)})
                s.add(ap)
                s.flush()
                apid = ap.id
            self._log(action, "pending_approval", dec.reason, dec.tier, acc_id, spec, approval_id=apid)
            return ActionResult(action_id=action.action_id, status="pending_approval", reason=dec.reason, tier=dec.tier,
                                approval_id=apid)
        return await self._execute(action, spec, account, connector, task, dec.tier, context)

    async def _execute(self, action, spec, account, connector, task, tier, context=None) -> ActionResult:
        if not self.agents.consume_budget(action.agent_id):
            self._log(action, "denied", "agent action budget exhausted", tier, account.id, spec)
            return ActionResult(action_id=action.action_id, status="denied", reason="budget exhausted", tier=tier)
        ctx = dict(self.context_fn(action, task) if (self.context_fn and spec.mutates and task is not None) else {})
        ctx.update(context or {})
        span = self.tracer.start_as_current_span(f"{action.provider}.{action.operation}") if self.tracer else None
        try:
            if span is not None:
                span.__enter__()
            res = await connector.execute(action)
        except ConnectorError as e:
            st = "rate_limited" if e.status == 429 else "failed"
            self._log(action, st, str(e), tier, account.id, spec)
            return ActionResult(action_id=action.action_id, status=st, reason=self.secrets.redact(str(e)), tier=tier)
        except Exception as e:  # noqa: BLE001 - network / provider failures are recorded, never raised to the model
            self._log(action, "failed", f"{type(e).__name__}: {e}", tier, account.id, spec)
            return ActionResult(action_id=action.action_id, status="failed", reason=type(e).__name__, tier=tier)
        finally:
            if span is not None:
                span.__exit__(None, None, None)
        res.tier = tier
        now = time.time()
        ledger_id = None
        if spec.mutates and self.outcomes is not None and res.status == "executed":
            content = action.content(spec)
            ledger_id = self.outcomes.record_action(
                action_id=action.action_id, provider=action.provider, operation=action.operation, agent_id=action.agent_id,
                account_id=account.id, object_ref=res.provider_object_id, content=content,
                action_str=f"{action.operation}({content!r})" if content else action.operation,
                context_text=ctx.get("context_text", f"PROVIDER: {action.provider}"), goal=task.goal if task else None,
                time_base=ctx.get("time_base", "real"), action_time=ctx.get("action_time", action.proposed_at),
                checkpoint=action.model_checkpoint, generation=action.generation, forecast=ctx.get("forecast"),
                split=task.split if task else "train", task_id=action.task_id,
                group_key=(task.budget or {}).get("group") if task else None)
        self._log(action, res.status, res.reason or "executed", tier, account.id, spec,
                  provider_object_id=res.provider_object_id, result={"observation_keys": sorted(res.observation)},
                  executed_at=now, ledger_action_id=ledger_id)
        self.accounts.update(account.id, last_activity_at=now)
        # "untrusted" is what a model may be shown (clearly delimited external content); "data" is the
        # structured provider response for harness code (adapters, verifiers) — never instructions.
        res.observation = {"untrusted": Untrusted(source=action.provider, text=str(res.observation)[:4000]).render(),
                           "data": res.observation}
        return res

    async def execute_approved(self, approval_id: str, operator: str) -> ActionResult:
        with self.db.session() as s:
            ap = s.get(Approval, approval_id)
            if ap is None or ap.status != "pending" or ap.kind not in ("action", "high_impact"):
                raise PermissionError("no pending action approval with that id")
            ap.status, ap.decided_by, ap.decided_at = "approved", operator, time.time()
            payload = ap.payload["action"]
        action = ActionRequest(**payload)
        return await self.submit(action, approved=True)  # every other check runs again at execution time

    def deny_approval(self, approval_id: str, operator: str, note: str = "") -> None:
        with self.db.session() as s:
            ap = s.get(Approval, approval_id)
            ap.status, ap.decided_by, ap.decided_at, ap.note = "denied", operator, time.time(), note
            row = s.get(ActionLog, ap.ref_id)
            if row is not None:
                row.status, row.reason = "denied", f"approval denied by {operator}: {note}"

    async def observe(self, agent_id: str, task_id: str, provider: str, operation: str, params: dict) -> ActionResult:
        """Read operations go through exactly the same checks (tier 0)."""
        return await self.submit(ActionRequest(agent_id=agent_id, task_id=task_id, provider=provider,
                                               operation=operation, params=params))

    async def aclose(self) -> None:
        for c in self._connectors.values():
            await c.ctx.http.aclose()
