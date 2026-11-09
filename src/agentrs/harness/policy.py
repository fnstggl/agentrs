"""Permission and approval engine.

Tiers (``types.Tier``): 0 observe · 1 sandbox · 2 authorised reversible · 3 external publishing /
communication · 4 high impact. Every check fails closed. An agent can never raise its tier: the
effective ceiling is min(agent.max_tier, task.authorized_tier), both set by operators.

Order of checks (first failure wins): emergency stop → agent active → agent budget → task state,
time limit and allowed actions → tier ceilings → provider enabled + capability enabled (registry)
→ agent capability → account active / assigned / capable / not revoked / not expired → provider
rate limits configured and not exceeded → per-account quotas → content checks (length, secret
exfiltration, duplicate content, blocked patterns) → approval requirement (tier 4 always; tier 3
unless the task pre-authorises that exact operation; anything in ``require_approval_for``).
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from agentrs.harness.capabilities import CapabilityRegistry
from agentrs.harness.db import ActionLog, Agent, Database, PlatformAccount, SystemState, Task
from agentrs.harness.identity import AgentRegistry
from agentrs.harness.secrets import SecretStore
from agentrs.harness.types import ActionRequest, OperationSpec, Tier


class PermissionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_tiers: list[int] = Field(default_factory=lambda: [3, 4])
    always_human_tiers: list[int] = Field(default_factory=lambda: [4])
    require_approval_for: list[str] = Field(default_factory=list)  # "<provider>:<operation>"
    forbidden_operations: list[str] = Field(default_factory=list)
    max_content_chars: int = 5000
    duplicate_window_s: float = 7 * 86400
    blocked_patterns: list[str] = Field(default_factory=lambda: [r"(?i)free money", r"(?i)crypto giveaway",
                                                                  r"(?i)click here to claim"])
    account_quota_per_hour: int = 30
    account_writes_per_day: int = 50
    require_rate_limits: bool = True  # an enabled provider must have non-zero configured limits

    @classmethod
    def load(cls, path: str | Path) -> "PermissionsConfig":
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}).get("permissions", {}))


class Decision(BaseModel):
    outcome: str  # allow | deny | needs_approval
    reason: str = ""
    tier: int = 0
    checks: list[str] = Field(default_factory=list)


class RateLimiter:
    """Sliding-window counters. Redis-backed when a client is given (shared by all workers)."""

    def __init__(self, redis_client=None, prefix: str = "agentrs:rl:"):
        self.r = redis_client
        self.prefix = prefix
        self._mem: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: float, window_s: float, now: Optional[float] = None, consume: bool = True) -> bool:
        now = now if now is not None else time.time()
        if limit <= 0:
            return False
        if self.r is not None:
            k = self.prefix + key
            pipe = self.r.pipeline()
            pipe.zremrangebyscore(k, 0, now - window_s)
            pipe.zcard(k)
            _, n = pipe.execute()
            if n >= limit:
                return False
            if consume:
                self.r.zadd(k, {f"{now}:{time.perf_counter_ns()}": now})
                self.r.expire(k, int(window_s) + 1)
            return True
        with self._lock:
            q = self._mem[key]
            while q and q[0] <= now - window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            if consume:
                q.append(now)
            return True


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


class PermissionEngine:
    def __init__(self, db: Database, registry: CapabilityRegistry, cfg: PermissionsConfig, secrets: SecretStore,
                 limiter: Optional[RateLimiter] = None):
        self.db, self.reg, self.cfg, self.secrets = db, registry, cfg, secrets
        self.limiter = limiter or RateLimiter()
        self._blocked = [re.compile(p) for p in cfg.blocked_patterns]

    # ------------------------------------------------------------------ global controls
    def emergency_stop(self, active: bool, by: str, reason: str = "") -> None:
        with self.db.session() as s:
            row = s.get(SystemState, "emergency_stop")
            val = {"active": active, "by": by, "reason": reason, "t": time.time()}
            if row is None:
                s.add(SystemState(key="emergency_stop", value=val))
            else:
                row.value, row.updated_at = val, time.time()

    def stopped(self) -> Optional[dict]:
        with self.db.session() as s:
            row = s.get(SystemState, "emergency_stop")
            return row.value if row is not None and row.value.get("active") else None

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, action: ActionRequest, spec: OperationSpec, agent: Optional[Agent], task: Optional[Task],
                 account: Optional[PlatformAccount], approved: bool = False, consume: bool = True,
                 tier_override: Optional[int] = None) -> Decision:
        checks: list[str] = []
        tier = int(spec.tier) if tier_override is None else max(int(tier_override), 0)

        def deny(why: str) -> Decision:
            return Decision(outcome="deny", reason=why, tier=tier, checks=checks)

        key = f"{action.provider}:{action.operation}"
        if (st := self.stopped()) is not None:
            return deny(f"emergency stop is active ({st.get('reason', '')})")
        checks.append("not_stopped")
        if agent is None or agent.status != "active":
            return deny("agent is not active")
        if agent.actions_used >= agent.action_budget:
            return deny("agent action budget exhausted")
        checks.append("agent_ok")
        if key in self.cfg.forbidden_operations:
            return deny(f"{key} is forbidden by policy")
        if task is None:
            return deny("every action must belong to a task")
        if task.state not in ("running", "waiting_outcome"):
            return deny(f"task is {task.state}")
        if time.time() - task.created_at > task.time_limit_s:
            return deny("task time limit exceeded")
        if task.allowed_actions and key not in task.allowed_actions and f"{action.provider}:*" not in task.allowed_actions:
            return deny(f"{key} is not an allowed action of this task")
        checks.append("task_ok")
        ceiling = min(agent.max_tier, task.authorized_tier)
        if tier > ceiling:
            return deny(f"operation tier {tier} exceeds the authorised tier {ceiling}")
        checks.append("tier_ok")
        ok, why = self.reg.allows(action.provider, spec.cap())
        if not ok:
            return deny(why)
        if not AgentRegistry.has_capability(agent, action.provider, spec.cap()):
            return deny(f"agent {agent.id} lacks capability {action.provider}:{spec.cap()}")
        checks.append("capability_ok")
        if account is None:
            return deny("no authorised account for this action")
        if account.status != "active" or account.revoked_at:
            return deny(f"account {account.id} is {account.status}")
        if account.provider != action.provider:
            return deny("account belongs to another provider")
        if agent.id not in (account.allowed_agents or []) and "*" not in (account.allowed_agents or []):
            return deny("agent is not authorised to use this account")
        if spec.cap() not in (account.capabilities or []):
            return deny(f"account is not authorised for {spec.cap()}")
        if account.reauth_by and account.reauth_by < time.time():
            return deny("account authorisation expired; operator must re-authorise")
        checks.append("account_ok")
        lim = self.reg.get(action.provider).limits
        if self.cfg.require_rate_limits and lim.requests_per_minute <= 0:
            return deny(f"no rate limit configured for {action.provider} (fail closed)")
        if not self.limiter.hit(f"{action.provider}:{account.id}:rpm", lim.requests_per_minute, 60, consume=consume):
            return Decision(outcome="deny", reason="provider rate limit reached", tier=tier, checks=checks + ["rate"])
        if spec.mutates:
            per_day = lim.posts_per_day if lim.posts_per_day > 0 else self.cfg.account_writes_per_day
            if not self.limiter.hit(f"{account.id}:writes:day", min(per_day, self.cfg.account_writes_per_day), 86400,
                                    consume=consume):
                return deny("per-account daily write quota reached")
            if not self.limiter.hit(f"{account.id}:writes:hour", self.cfg.account_quota_per_hour, 3600, consume=consume):
                return deny("per-account hourly quota reached")
        checks.append("rate_ok")
        content = action.content(spec)
        if content is not None:
            if len(content) > self.cfg.max_content_chars:
                return deny("content too long")
            if self.secrets.contains_secret(content) or self.secrets.contains_secret(str(action.params)):
                return deny("content contains a credential (exfiltration blocked)")
            for p in self._blocked:
                if p.search(content):
                    return deny(f"content matches a blocked pattern ({p.pattern})")
            h = content_hash(content)
            with self.db.session() as s:
                dup = s.scalars(select(ActionLog).where(ActionLog.account_id == account.id, ActionLog.content_hash == h,
                                                        ActionLog.status == "executed",
                                                        ActionLog.executed_at > time.time() - self.cfg.duplicate_window_s)
                                ).first()
            if dup is not None:
                return deny("duplicate content already published from this account")
            checks.append("content_ok")
        if not approved:
            pre = set((task.budget or {}).get("pre_approved_operations", []))
            if tier in self.cfg.always_human_tiers:
                return Decision(outcome="needs_approval", reason=f"tier {tier} always needs human approval", tier=tier,
                                checks=checks)
            if key in self.cfg.require_approval_for or (tier in self.cfg.approval_tiers and key not in pre):
                return Decision(outcome="needs_approval", reason=f"tier {tier} operation {key} needs approval",
                                tier=tier, checks=checks)
        return Decision(outcome="allow", reason="authorised", tier=tier, checks=checks + ["approval_ok"])


def task_tier(task: Task) -> Tier:
    return Tier(task.authorized_tier)


def summarize_action(action: ActionRequest, spec: OperationSpec) -> dict[str, Any]:
    c = action.content(spec)
    return {"provider": action.provider, "operation": action.operation, "tier": int(spec.tier),
            "params": {k: (v if k != spec.content_param else (str(v)[:280] + ("…" if len(str(v)) > 280 else "")))
                       for k, v in action.params.items()}, "content_chars": len(c) if c else 0,
            "agent": action.agent_id, "task": action.task_id}
