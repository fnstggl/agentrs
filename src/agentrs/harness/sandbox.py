"""Sandbox wiring for SocialWebLab: the one provider whose account provisioning is programmatic and
permitted, because we operate it (test accounts through the lab's private control channel)."""

from __future__ import annotations

import secrets as pysecrets

from agentrs.harness.secrets import SecretStore


def socialweblab_provisioner(server, secrets: SecretStore, backend: str = "memory", followers: int = 3):
    """Returns fn(provisioning request) -> account fields. Creates a lab test account (disclosed as an
    agent account inside the lab) and stores its session token as a secret reference."""

    def provision(req) -> dict:
        handle = f"agent-{req.requesting_agent}-{pysecrets.token_hex(3)}"
        pw = pysecrets.token_urlsafe(18)
        server.control("create_account", handle=handle, password=pw, kind="agent",
                       spaces=[s["name"] for s in server.lab.spaces()], followers=followers)
        tok = server.control("login", handle=handle, password=pw)["token"]
        ref = secrets.put(backend, f"swl.{handle}.session_token", tok)
        return {"external_id": handle, "account_type": "test", "owner_org": "sandbox",
                "responsible_operator": "system:socialweblab-sandbox", "credential_refs": {"session_token": ref},
                "bot_disclosure": "labelled", "training_data_permission": "authorized_sandbox"}

    return provision


def enable_sandbox_provider(registry, server_url: str, rpm: float = 6000, posts_per_day: float = 50000) -> None:
    """Enable SocialWebLab (our own sandbox) in a loaded capability registry. External providers stay
    as configured (disabled by default)."""
    p = registry.get("socialweblab")
    p.enabled = True
    p.base_url = server_url
    p.authorization_ref = "self-hosted sandbox"
    p.capabilities.update(read_content=True, create_post=True, reply=True, react=True, account_provisioning=True)
    p.limits.requests_per_minute = rpm
    p.limits.posts_per_day = posts_per_day
    host = server_url.split("//", 1)[1].split(":")[0].split("/")[0]
    p.allowed_domains = sorted(set(p.allowed_domains) | {host})


def build_sandbox_harness(workdir, lab, server, db_url: str, agent_ids=("sandbox_agent_001",), redis_url=None,
                          transports=None, extra_domains=(), max_tier: int = 1, budget: int = 10_000,
                          secret_key: bytes | None = None):
    """A complete harness whose only enabled provider is the running SocialWebLab sandbox."""
    from agentrs.harness.app import build_harness
    from agentrs.harness.identity import AgentSpec
    from agentrs.harness.net import EgressPolicy
    from agentrs.harness.verification import socialweblab_help_verifier

    host = server.url.split("//", 1)[1].split(":")[0]
    eg = EgressPolicy(allowed_domains=[host, *extra_domains], allow_private=[host, *extra_domains],
                      allow_http_for=[host, *extra_domains])
    h = build_harness(db_url, workdir, egress=eg, redis_url=redis_url, secret_key=secret_key or SecretStore.generate_key(),
                      transports=transports,
                      context_fn=lambda a, t: ({"time_base": lab.time_base, "action_time": lab.now(),
                                                "context_text": "ENV: socialweblab"} if a.provider == "socialweblab"
                                               else {}))  # real providers use the real clock
    enable_sandbox_provider(h.registry, server.url)
    for aid in agent_ids:
        h.agents.upsert(AgentSpec(id=aid, display_name=aid, role="practice", allowed_capabilities=["socialweblab:*"],
                                  max_tier=max_tier, action_budget=budget))
    h.provisioning.programmatic["socialweblab"] = socialweblab_provisioner(server, h.secrets, backend="file")
    h.verification.register("socialweblab_help_within", socialweblab_help_verifier(server.control, lab.now))
    h.extras["lab"], h.extras["server"] = lab, server
    return h


def sandbox_fetchers(h) -> dict:
    """Follow-up fetchers: every active sandbox account's connector (outcomes read with that account)."""
    out = {}
    for acc in h.accounts.list("socialweblab"):
        if acc.status == "active":
            out["harness:socialweblab"] = h.gateway.connector_for(acc).fetch_outcome
            break
    return out
