"""Harness container: builds every service from configuration and the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from agentrs.harness.capabilities import CapabilityRegistry
from agentrs.harness.db import Database, init_db
from agentrs.harness.gateway import ToolGateway
from agentrs.harness.identity import AgentRegistry, PlatformAccountRegistry
from agentrs.harness.jobs import JobQueue
from agentrs.harness.monitoring import Metrics, setup_tracing
from agentrs.harness.net import EgressPolicy
from agentrs.harness.oauth import OAuthManager
from agentrs.harness.outcomes import OutcomeTracker
from agentrs.harness.policy import PermissionEngine, PermissionsConfig, RateLimiter
from agentrs.harness.provisioning import AccountProvisioningOrchestrator
from agentrs.harness.secrets import SecretStore
from agentrs.harness.tasks import TaskRegistry
from agentrs.harness.verification import VerificationService

ROOT = Path(__file__).resolve().parents[3]


def connector_classes() -> dict[str, type]:
    from agentrs.harness.connectors.generic_web import GenericWebConnector
    from agentrs.harness.connectors.socialweblab import SocialWebLabConnector
    from agentrs.harness.email_service import EmailConnector

    classes: dict[str, type] = {"socialweblab": SocialWebLabConnector, "web": GenericWebConnector,
                                "email": EmailConnector}
    try:  # provider connectors (Bluesky, Mastodon, GitHub, Discord, X, Instagram, Reddit, LinkedIn)
        from agentrs.harness.connectors.registry import CONNECTORS

        classes.update(CONNECTORS)
    except ImportError:  # pragma: no cover
        pass
    return classes


@dataclass
class Harness:
    db: Database
    registry: CapabilityRegistry
    agents: AgentRegistry
    accounts: PlatformAccountRegistry
    secrets: SecretStore
    policy: PermissionEngine
    provisioning: AccountProvisioningOrchestrator
    oauth: OAuthManager
    gateway: ToolGateway
    tasks: TaskRegistry
    verification: VerificationService
    jobs: JobQueue
    outcomes: OutcomeTracker
    metrics: Metrics
    egress: EgressPolicy
    tracer: Any = None
    spans: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


def build_harness(db_url: str, workdir: str | Path, platforms: str | Path = ROOT / "config/platforms.yaml",
                  permissions: str | Path = ROOT / "config/permissions.yaml", agents_yaml: Optional[str | Path] = None,
                  egress: Optional[EgressPolicy] = None, redis_url: Optional[str] = None,
                  secret_file: Optional[str | Path] = None, secret_key: Optional[bytes] = None,
                  runtime_secret_backend: str = "file", transports: Optional[dict] = None,
                  context_fn=None) -> Harness:
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    db = init_db(db_url)
    reg = CapabilityRegistry.load(platforms)
    pcfg = PermissionsConfig.load(permissions) if Path(permissions).exists() else PermissionsConfig()
    if Path(permissions).exists():
        from agentrs.harness.connectors.generic_web import GenericWebConnector

        GenericWebConnector.site_profiles = dict((yaml.safe_load(Path(permissions).read_text()) or {}).get("web_sites", {}))
    secrets = SecretStore(secret_file or wd / "secrets.enc", key=secret_key)
    agents = AgentRegistry(db)
    if agents_yaml and Path(agents_yaml).exists():
        agents.load_yaml(agents_yaml)
    accounts = PlatformAccountRegistry(db, secrets, runtime_secret_backend)
    limiter = RateLimiter(__import__("redis").Redis.from_url(redis_url) if redis_url else None)
    policy = PermissionEngine(db, reg, pcfg, secrets, limiter)
    if egress is None:
        doms = sorted({d for p in reg.providers.values() for d in p.allowed_domains})
        egress = EgressPolicy(allowed_domains=doms)
    metrics = Metrics()
    tracer, spans = setup_tracing("agentrs-harness")
    outcomes = OutcomeTracker(wd / "outcomes")
    gw = ToolGateway(db, reg, agents, accounts, policy, secrets, connector_classes(), egress, outcomes=outcomes,
                     transports=transports, metrics=metrics, tracer=tracer, context_fn=context_fn)
    prov = AccountProvisioningOrchestrator(db, reg, agents, accounts,
                                           validators={p: (lambda a: gw.connector_for(a).validate_credentials())
                                                       for p in reg.providers})
    tasks = TaskRegistry(db)
    ver = VerificationService(db, tasks, ledger=outcomes.ledger)
    tasks.known_verifiers = None
    return Harness(db=db, registry=reg, agents=agents, accounts=accounts, secrets=secrets, policy=policy,
                   provisioning=prov, oauth=OAuthManager(db, secrets, accounts, runtime_secret_backend), gateway=gw,
                   tasks=tasks, verification=ver, jobs=JobQueue(db), outcomes=outcomes, metrics=metrics, egress=egress,
                   tracer=tracer, spans=spans)


def from_env() -> Harness:
    """Build from environment variables (deployments): see .env.example."""
    key = os.environ.get("AGENTRS_SECRET_KEY")
    eg = None
    if os.environ.get("AGENTRS_EGRESS_CONFIG"):
        eg = EgressPolicy(**(yaml.safe_load(Path(os.environ["AGENTRS_EGRESS_CONFIG"]).read_text()) or {}))
    return build_harness(os.environ.get("AGENTRS_DATABASE_URL", "sqlite:///runs/harness/harness.db"),
                         os.environ.get("AGENTRS_DATA_DIR", "runs/harness"),
                         platforms=os.environ.get("AGENTRS_PLATFORMS", ROOT / "config/platforms.yaml"),
                         permissions=os.environ.get("AGENTRS_PERMISSIONS", ROOT / "config/permissions.yaml"),
                         agents_yaml=os.environ.get("AGENTRS_AGENTS", ROOT / "config/agents.yaml"),
                         egress=eg, redis_url=os.environ.get("AGENTRS_REDIS_URL"),
                         secret_key=key.encode() if key else None)
