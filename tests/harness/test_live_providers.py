"""Credential-gated live tests against the REAL provider APIs. Every test is SKIPPED unless its variables are set.

They are read-only (``validate_credentials`` + one read) unless BOTH
``AGENTRS_LIVE_WRITE_OK=1`` and ``AGENTRS_LIVE_WRITE_PROVIDER=<provider>`` are set; then exactly ONE write
is allowed in the whole run, for that provider only (a module-wide budget enforces it). Reddit and LinkedIn
are always read-only. Writes go through the connector's ``provider_policy`` first, like the gateway would.

Secrets are read through the ``env`` secret backend (``secret://env/<NAME>`` -> ``AGENTRS_SECRET_<NAME>``) and
never printed. Traffic goes through ``SafeHTTPClient`` with the provider's ``allowed_domains`` from
``config/platforms.yaml`` and normal SSRF checks; set ``AGENTRS_LIVE_PROXY_URL`` if egress needs a proxy.

Variables (secrets marked *):

* Bluesky:   ``AGENTRS_LIVE_BLUESKY_HANDLE``, ``AGENTRS_SECRET_BLUESKY_APP_PASSWORD``*,
             optional ``AGENTRS_LIVE_BLUESKY_BASE_URL`` (default https://bsky.social).
             Write: one post.
* Mastodon:  ``AGENTRS_LIVE_MASTODON_BASE_URL`` (e.g. https://mastodon.example; its host is added to the
             egress allowlist), ``AGENTRS_SECRET_MASTODON_ACCESS_TOKEN``*.
             Write: one ``direct``-visibility status with no mentions (visible to the account only).
* GitHub:    ``AGENTRS_LIVE_GITHUB_APP_ID``, ``AGENTRS_LIVE_GITHUB_INSTALLATION_ID``, ``AGENTRS_LIVE_GITHUB_REPO``
             (owner/name), ``AGENTRS_SECRET_GITHUB_APP_KEY``* (App private key PEM).
             Write: one branch ``agentrs/live-test-<epoch>`` from the default branch (delete it afterwards).
* Discord:   ``AGENTRS_SECRET_DISCORD_BOT_TOKEN``*, ``AGENTRS_LIVE_DISCORD_CHANNEL_ID``.
             Write: one message in that channel.
* X:         ``AGENTRS_SECRET_X_ACCESS_TOKEN``* (OAuth 2.0 user context), optional ``AGENTRS_LIVE_X_TWEET_ID``
             (a post to read). Write: one post.
* Instagram: ``AGENTRS_LIVE_INSTAGRAM_USER_ID``, ``AGENTRS_SECRET_INSTAGRAM_ACCESS_TOKEN``*;
             write also needs ``AGENTRS_LIVE_INSTAGRAM_IMAGE_URL`` (public https JPEG). Write: one image post.
* Reddit:    ``AGENTRS_SECRET_REDDIT_ACCESS_TOKEN``*, ``AGENTRS_LIVE_REDDIT_AUTHORIZATION_REF``,
             ``AGENTRS_LIVE_REDDIT_USER_AGENT``, ``AGENTRS_LIVE_REDDIT_SUBREDDIT``,
             ``AGENTRS_LIVE_REDDIT_TRAINING_DATA_AUTHORIZED`` (``true``/``false``, must be stated). Read-only.
* LinkedIn:  ``AGENTRS_SECRET_LINKEDIN_ACCESS_TOKEN``*, ``AGENTRS_LIVE_LINKEDIN_AUTHORIZATION_REF``. Read-only.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from agentrs.harness.capabilities import CapabilityRegistry, ProviderConfig
from agentrs.harness.connectors.base import ConnectorContext, CredentialHandle
from agentrs.harness.connectors.registry import make_connector
from agentrs.harness.net import EgressPolicy, SafeHTTPClient
from agentrs.harness.secrets import SecretStore
from agentrs.harness.types import ActionRequest

REGISTRY = CapabilityRegistry.load(Path(__file__).resolve().parents[2] / "config" / "platforms.yaml")
_WRITES = {"left": 1}
_LOCK = threading.Lock()


def env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None


def need(*names: str) -> None:
    missing = [n for n in names if not env(n)]
    if missing:
        pytest.skip(f"live test needs {', '.join(missing)}")


def write_allowed(provider: str) -> bool:
    """True at most once per run, and only for the provider the operator named."""
    if env("AGENTRS_LIVE_WRITE_OK") != "1" or env("AGENTRS_LIVE_WRITE_PROVIDER") != provider:
        return False
    with _LOCK:
        if _WRITES["left"] <= 0:
            return False
        _WRITES["left"] -= 1
        return True


def connector(provider: str, secrets: dict[str, str], account: dict, base_url: str | None = None,
              extra_domains: tuple[str, ...] = (), config: ProviderConfig | None = None):
    shipped = REGISTRY.get(provider)
    cfg = config or ProviderConfig(provider=provider, enabled=True, base_url=base_url or shipped.base_url,
                                   capabilities=dict.fromkeys(shipped.capabilities, True),
                                   allowed_domains=list(shipped.allowed_domains) + list(extra_domains),
                                   authorization_ref="live-test (operator-supplied credentials)")
    domains = list(cfg.allowed_domains) + list(extra_domains)
    policy = EgressPolicy(allowed_domains=domains, proxy_url=env("AGENTRS_LIVE_PROXY_URL"))
    store = SecretStore()
    refs = {name: f"secret://env/{envname}" for name, envname in secrets.items()}
    creds = CredentialHandle(refs, store.get, lambda n, v: store.put("memory", f"live_{provider}_{n}", v))
    http = SafeHTTPClient(policy, provider=provider, provider_domains=domains)
    return make_connector(provider, ConnectorContext(config=cfg, http=http, credentials=creds, account=account)), http


async def _write(conn, op: str, **params):
    action = ActionRequest(agent_id="live-test", provider=conn.provider, operation=op, params=params,
                           reason="operator-approved single live write")
    ok, why = conn.provider_policy(action, [])
    assert ok, why
    res = await conn.execute(action)
    assert res.status == "executed"
    return res


def _stamp() -> str:
    return f"[agentrs live connector test, automated, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]"


def test_live_bluesky():
    need("AGENTRS_LIVE_BLUESKY_HANDLE", "AGENTRS_SECRET_BLUESKY_APP_PASSWORD")

    async def main():
        conn, http = connector("bluesky", {"app_password": "BLUESKY_APP_PASSWORD"},
                               {"handle": env("AGENTRS_LIVE_BLUESKY_HANDLE")}, env("AGENTRS_LIVE_BLUESKY_BASE_URL"))
        try:
            v = await conn.validate_credentials()
            assert v["ok"] and v["did"].startswith("did:")
            await conn.observe({"operation": "read_author_feed", "params": {"limit": 1}})
            if write_allowed("bluesky"):
                res = await _write(conn, "create_post", text=f"{_stamp()} test post from an automated account")
                assert res.provider_object_id.startswith("at://")
        finally:
            await http.aclose()
    asyncio.run(main())


def test_live_mastodon():
    need("AGENTRS_LIVE_MASTODON_BASE_URL", "AGENTRS_SECRET_MASTODON_ACCESS_TOKEN")
    base = env("AGENTRS_LIVE_MASTODON_BASE_URL").rstrip("/")

    async def main():
        conn, http = connector("mastodon", {"access_token": "MASTODON_ACCESS_TOKEN"}, {}, base,
                               extra_domains=(urlsplit(base).hostname,))
        try:
            v = await conn.validate_credentials()
            assert v["ok"]
            assert isinstance((await conn.observe({"operation": "read_instance", "params": {}}))["rules"], list)
            if write_allowed("mastodon"):
                await _write(conn, "create_post", text=f"{_stamp()} private test status", visibility="direct")
        finally:
            await http.aclose()
    asyncio.run(main())


def test_live_github():
    need("AGENTRS_LIVE_GITHUB_APP_ID", "AGENTRS_LIVE_GITHUB_INSTALLATION_ID", "AGENTRS_LIVE_GITHUB_REPO",
         "AGENTRS_SECRET_GITHUB_APP_KEY")
    repo = env("AGENTRS_LIVE_GITHUB_REPO")

    async def main():
        conn, http = connector("github", {"app_private_key": "GITHUB_APP_KEY"},
                               {"app_id": int(env("AGENTRS_LIVE_GITHUB_APP_ID")),
                                "installation_id": int(env("AGENTRS_LIVE_GITHUB_INSTALLATION_ID")),
                                "repositories": [repo]})
        try:
            v = await conn.validate_credentials()
            assert repo in v["repositories"]
            info = await conn.observe({"operation": "read_repository", "params": {"repo": repo}})
            assert info["repository"]["full_name"].lower() == repo.lower()
            if write_allowed("github"):
                await _write(conn, "create_branch", repo=repo, branch=f"agentrs/live-test-{int(time.time())}")
        finally:
            await conn.revoke()  # revoke the installation token minted for this test
            await http.aclose()
    asyncio.run(main())


def test_live_discord():
    need("AGENTRS_SECRET_DISCORD_BOT_TOKEN", "AGENTRS_LIVE_DISCORD_CHANNEL_ID")
    channel = env("AGENTRS_LIVE_DISCORD_CHANNEL_ID")

    async def main():
        conn, http = connector("discord", {"bot_token": "DISCORD_BOT_TOKEN"}, {"allowed_channel_ids": [channel]})
        try:
            assert (await conn.validate_credentials())["bot"] is True
            await conn.observe({"operation": "read_messages", "params": {"channel_id": channel, "limit": 1}})
            if write_allowed("discord"):
                await _write(conn, "send_message", channel_id=channel, content=f"{_stamp()} test message from a bot")
        finally:
            await http.aclose()
    asyncio.run(main())


def test_live_x():
    need("AGENTRS_SECRET_X_ACCESS_TOKEN")

    async def main():
        conn, http = connector("x", {"access_token": "X_ACCESS_TOKEN"}, {})
        try:
            assert (await conn.validate_credentials())["ok"]
            if env("AGENTRS_LIVE_X_TWEET_ID"):
                await conn.observe({"operation": "read_post", "params": {"tweet_id": env("AGENTRS_LIVE_X_TWEET_ID")}})
            if write_allowed("x"):
                await _write(conn, "create_post", text=f"{_stamp()} test post from an automated account")
        finally:
            await http.aclose()
    asyncio.run(main())


def test_live_instagram():
    need("AGENTRS_LIVE_INSTAGRAM_USER_ID", "AGENTRS_SECRET_INSTAGRAM_ACCESS_TOKEN")

    async def main():
        conn, http = connector("instagram", {"access_token": "INSTAGRAM_ACCESS_TOKEN"},
                               {"ig_user_id": env("AGENTRS_LIVE_INSTAGRAM_USER_ID")})
        try:
            assert (await conn.validate_credentials())["account_type"] in ("BUSINESS", "MEDIA_CREATOR")
            await conn.observe({"operation": "read_publishing_limit", "params": {}})
            if env("AGENTRS_LIVE_INSTAGRAM_IMAGE_URL") and write_allowed("instagram"):
                await _write(conn, "publish_image", image_url=env("AGENTRS_LIVE_INSTAGRAM_IMAGE_URL"),
                             caption=f"{_stamp()} test post from an automated account")
        finally:
            await http.aclose()
    asyncio.run(main())


def test_live_reddit_read_only():
    need("AGENTRS_SECRET_REDDIT_ACCESS_TOKEN", "AGENTRS_LIVE_REDDIT_AUTHORIZATION_REF", "AGENTRS_LIVE_REDDIT_USER_AGENT",
         "AGENTRS_LIVE_REDDIT_SUBREDDIT", "AGENTRS_LIVE_REDDIT_TRAINING_DATA_AUTHORIZED")
    declared = env("AGENTRS_LIVE_REDDIT_TRAINING_DATA_AUTHORIZED").lower()
    assert declared in ("true", "false"), "AGENTRS_LIVE_REDDIT_TRAINING_DATA_AUTHORIZED must be true or false"
    shipped = REGISTRY.get("reddit")
    cfg = ProviderConfig(provider="reddit", enabled=True, base_url=shipped.base_url,
                         capabilities=dict.fromkeys(shipped.capabilities, True), allowed_domains=shipped.allowed_domains,
                         authorization_ref=env("AGENTRS_LIVE_REDDIT_AUTHORIZATION_REF"),
                         training_data_authorized=declared == "true")

    async def main():
        conn, http = connector("reddit", {"access_token": "REDDIT_ACCESS_TOKEN"},
                               {"user_agent": env("AGENTRS_LIVE_REDDIT_USER_AGENT")}, config=cfg)
        try:
            v = await conn.validate_credentials()
            assert v["ok"] and v["data_use"] in ("training_permitted", "non_training_only")
            out = await conn.observe({"operation": "read_content",
                                      "params": {"subreddit": env("AGENTRS_LIVE_REDDIT_SUBREDDIT"), "limit": 1}})
            assert out["data_use"] == v["data_use"]
        finally:
            await http.aclose()
    asyncio.run(main())


def test_live_linkedin_read_only():
    need("AGENTRS_SECRET_LINKEDIN_ACCESS_TOKEN", "AGENTRS_LIVE_LINKEDIN_AUTHORIZATION_REF")
    shipped = REGISTRY.get("linkedin")
    cfg = ProviderConfig(provider="linkedin", enabled=True, base_url=shipped.base_url,
                         capabilities=dict.fromkeys(shipped.capabilities, True), allowed_domains=shipped.allowed_domains,
                         authorization_ref=env("AGENTRS_LIVE_LINKEDIN_AUTHORIZATION_REF"))

    async def main():
        conn, http = connector("linkedin", {"access_token": "LINKEDIN_ACCESS_TOKEN"}, {}, config=cfg)
        try:
            assert (await conn.validate_credentials())["ok"]
            await conn.observe({"operation": "read_profile", "params": {}})
        finally:
            await http.aclose()
    asyncio.run(main())
