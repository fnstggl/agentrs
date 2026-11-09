"""Control-plane data model (SQLAlchemy 2.0). PostgreSQL in deployments, SQLite in tests.

Schema changes go through Alembic migrations (``harness/migrations``); ``init_db`` runs them.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from sqlalchemy import JSON, Boolean, Float, Integer, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _id() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Agent(Base):
    __tablename__ = "agents"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(64))
    model_checkpoint: Mapped[str] = mapped_column(String(200), default="current_approved")
    allowed_capabilities: Mapped[list[Any]] = mapped_column(JSON, default=list)
    max_tier: Mapped[int] = mapped_column(Integer, default=0)
    action_budget: Mapped[int] = mapped_column(Integer, default=100)
    actions_used: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="active")  # active | paused | stopped
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    last_action_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class PlatformAccount(Base):
    __tablename__ = "platform_accounts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    external_id: Mapped[str] = mapped_column(String(300))
    account_type: Mapped[str] = mapped_column(String(64))  # bot | app_installation | org_service | test
    owner_org: Mapped[str] = mapped_column(String(200))
    responsible_operator: Mapped[str] = mapped_column(String(200))
    bot_disclosure: Mapped[str] = mapped_column(String(64), default="unknown")  # labelled | profile_text | n/a | unknown
    capabilities: Mapped[list[Any]] = mapped_column(JSON, default=list)
    oauth_scopes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    credential_refs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # name -> secret:// reference
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|verified|active|suspended|revoked
    rate_limits: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    training_data_permission: Mapped[str] = mapped_column(String(32), default="not_authorized")
    last_auth_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_activity_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    revoked_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    allowed_agents: Mapped[list[Any]] = mapped_column(JSON, default=list)
    reauth_by: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    purpose: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    __table_args__ = (UniqueConstraint("provider", "external_id", name="uq_account_provider_external"),)


class ProvisioningRequest(Base):
    __tablename__ = "provisioning_requests"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    required_capabilities: Mapped[list[Any]] = mapped_column(JSON, default=list)
    purpose: Mapped[str] = mapped_column(Text)
    requesting_agent: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(40), default="REQUESTED")
    history: Mapped[list[Any]] = mapped_column(JSON, default=list)
    human_steps: Mapped[list[Any]] = mapped_column(JSON, default=list)
    account_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    decision_reason: Mapped[str] = mapped_column(Text, default="")
    fingerprint: Mapped[str] = mapped_column(String(128), index=True)  # provider + sorted capabilities
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    updated_at: Mapped[float] = mapped_column(Float, default=_now)


class OAuthState(Base):
    __tablename__ = "oauth_states"
    state: Mapped[str] = mapped_column(String(128), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64))
    account_id: Mapped[str] = mapped_column(String(64))
    code_verifier_ref: Mapped[str] = mapped_column(String(300))
    redirect_uri: Mapped[str] = mapped_column(String(500))
    scopes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    used: Mapped[bool] = mapped_column(Boolean, default=False)


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str] = mapped_column(String(64))  # human | external_workflow | benchmark | practice | ...
    goal: Mapped[str] = mapped_column(Text)
    environment: Mapped[str] = mapped_column(String(64))
    required_capabilities: Mapped[list[Any]] = mapped_column(JSON, default=list)
    allowed_actions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    budget: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    initial_state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    verifier: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    failure_conditions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    time_limit_s: Mapped[float] = mapped_column(Float, default=3600.0)
    needs_approval: Mapped[bool] = mapped_column(Boolean, default=False)
    authorized_tier: Mapped[int] = mapped_column(Integer, default=1)  # highest tier this task may use
    state: Mapped[str] = mapped_column(String(32), default="queued")  # queued|running|waiting_outcome|succeeded|failed|cancelled
    assigned_agent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    split: Mapped[str] = mapped_column(String(32), default="train")
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    updated_at: Mapped[float] = mapped_column(Float, default=_now)


class ActionLog(Base):
    """Audit log: every proposal, its decision and its real outcome."""

    __tablename__ = "actions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    provider: Mapped[str] = mapped_column(String(64))
    account_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    operation: Mapped[str] = mapped_column(String(64))
    tier: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text, default="")
    approval_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    provider_object_id: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    model_checkpoint: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    generation: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    ledger_action_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    proposed_at: Mapped[float] = mapped_column(Float, default=_now)
    executed_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Approval(Base):
    __tablename__ = "approvals"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    kind: Mapped[str] = mapped_column(String(40))  # action | provisioning | human_verification | oauth_setup | high_impact
    ref_id: Mapped[str] = mapped_column(String(128), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|approved|denied|expired
    requested_by: Mapped[str] = mapped_column(String(64), default="")
    decided_by: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    decided_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class BrowserSessionRow(Base):
    __tablename__ = "browser_sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    agent_id: Mapped[str] = mapped_column(String(64))
    task_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    storage_ref: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # encrypted state object key
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|closed|revoked|expired
    actions_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    expires_at: Mapped[float] = mapped_column(Float, default=lambda: _now() + 3600)


class EmailMessage(Base):
    __tablename__ = "emails"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    direction: Mapped[str] = mapped_column(String(10))  # in | out
    category: Mapped[str] = mapped_column(String(20))  # transactional | external | notification
    thread_id: Mapped[str] = mapped_column(String(300), index=True)
    message_id: Mapped[str] = mapped_column(String(300), unique=True)
    in_reply_to: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    sender: Mapped[str] = mapped_column(String(300))
    recipients: Mapped[list[Any]] = mapped_column(JSON, default=list)
    subject: Mapped[str] = mapped_column(Text, default="")
    body: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30))  # draft|pending_approval|sent|failed|received|denied
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=_now)


class Job(Base):
    """Durable workflow / follow-up jobs (leases, retries, idempotency)."""

    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    run_at: Mapped[float] = mapped_column(Float, index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=5)
    lease_owner: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    lease_until: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    idem: Mapped[str] = mapped_column(String(200), unique=True)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    done_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Experience(Base):
    __tablename__ = "experiences"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    task_id: Mapped[str] = mapped_column(String(128), index=True)
    agent_id: Mapped[str] = mapped_column(String(64))
    action_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ledger_action_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_checkpoint: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    reward_status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|matured|censored
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    verified_by: Mapped[str] = mapped_column(String(200), default="")
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    split: Mapped[str] = mapped_column(String(32), default="train")
    training_eligible: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    matured_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class SystemState(Base):
    __tablename__ = "system_state"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[float] = mapped_column(Float, default=_now)


# ---------------------------------------------------------------------------- engine / sessions
class Database:
    def __init__(self, url: str):
        self.url = url
        kw: dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine = create_engine(url, **kw)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _pragma(dbapi_conn, _):  # pragma: no cover - sqlite only
                c = dbapi_conn.cursor()
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("PRAGMA busy_timeout=30000")
                c.close()
        self.Session = sessionmaker(self.engine, expire_on_commit=False, future=True)

    @contextmanager
    def session(self) -> Iterator[Any]:
        s = self.Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    @property
    def is_postgres(self) -> bool:
        return self.url.startswith("postgresql")


MIGRATIONS = Path(__file__).with_name("migrations")


def migrate(url: str) -> None:
    """Apply Alembic migrations up to head."""
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


def init_db(url: str) -> Database:
    migrate(url)
    return Database(url)
