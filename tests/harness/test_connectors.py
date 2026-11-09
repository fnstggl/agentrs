"""Connector tests against the local provider simulators (no real credentials, no internet).

Each connector talks to its simulator through ``SafeHTTPClient`` (``httpx.ASGITransport`` as the
transport, an egress policy that allows only the simulator host); one test runs a simulator under a
real uvicorn server on 127.0.0.1 and talks to it over real HTTP.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import socket
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import Request

from agentrs.harness.capabilities import CapabilityRegistry, ProviderConfig
from agentrs.harness.connectors import _common
from agentrs.harness.connectors.base import (
    ConnectorContext,
    ConnectorError,
    CredentialHandle,
)
from agentrs.harness.connectors.mastodon import make_pkce
from agentrs.harness.connectors.registry import CONNECTORS, make_connector
from agentrs.harness.net import EgressDenied, EgressPolicy, SafeHTTPClient
from agentrs.harness.secrets import SecretStore
from agentrs.harness.simulators import bluesky as bsky_sim
from agentrs.harness.simulators import discord as dc_sim
from agentrs.harness.simulators import github as gh_sim
from agentrs.harness.simulators import instagram as ig_sim
from agentrs.harness.simulators import mastodon as md_sim
from agentrs.harness.simulators import x as x_sim
from agentrs.harness.types import ActionRequest, Tier
from agentrs.harness.webhooks import verify_discord_signature, verify_github_signature

ROOT = Path(__file__).resolve().parents[2]
PLATFORMS = ROOT / "config" / "platforms.yaml"
REGISTRY = CapabilityRegistry.load(PLATFORMS)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fresh_account_memory():
    _common._MEMORY.clear()
    yield
    _common._MEMORY.clear()


class PolicyRefused(Exception):
    pass


class Harness:
    """A connector wired to a simulator, plus a miniature gateway (provider_policy, then execute)."""

    def __init__(self, provider: str, app, host: str, *, secrets: dict[str, str], account: dict, base_path: str = "",
                 transport=None, scheme: str = "http", port: str = "", config: ProviderConfig | None = None):
        self.provider = provider
        self.store = SecretStore()
        refs = {k: self.store.put("memory", f"{provider}_{k}", v) for k, v in secrets.items()}
        self.creds = CredentialHandle(refs, self.store.get, lambda n, v: self.store.put("memory", f"{provider}_{n}", v))
        self.egress: list[dict] = []
        policy = EgressPolicy(allowed_domains=[host], allow_private=[host], allow_http_for=[host])
        self.http = SafeHTTPClient(policy, provider=provider, provider_domains=[host], egress_log=self.egress,
                                   transport=transport or httpx.ASGITransport(app=app))
        caps = dict.fromkeys(REGISTRY.get(provider).capabilities, True) if REGISTRY.get(provider) else {}
        caps["update_profile"] = True
        self.cfg = config or ProviderConfig(provider=provider, enabled=True, capabilities=caps,
                                            base_url=f"{scheme}://{host}{port}{base_path}", authorization_ref="test-approval")
        self.ctx = ConnectorContext(config=self.cfg, http=self.http, credentials=self.creds, account=account)
        self.conn = make_connector(provider, self.ctx)
        self.recent: list[dict] = []

    def act(self, op: str, **params) -> ActionRequest:
        return ActionRequest(agent_id="agent-1", provider=self.provider, operation=op, params=params)

    def allowed(self, op: str, **params) -> tuple[bool, str]:
        return self.conn.provider_policy(self.act(op, **params), self.recent)

    async def do(self, op: str, **params):
        a = self.act(op, **params)
        ok, why = self.conn.provider_policy(a, self.recent)
        if not ok:
            raise PolicyRefused(why)
        res = await self.conn.execute(a)
        self.recent.append({"provider": self.provider, "operation": op, "params": params, "status": res.status,
                            "executed_at": time.time()})
        return res

    def secret(self, name: str) -> str:
        return self.creds.get(name)


# ====================================================================== registry / generic
def test_every_connector_declares_known_capabilities_and_sane_tiers():
    known_gaps = {("bluesky", "update_profile")}  # not yet a key in config/platforms.yaml -> fails closed
    for name, cls in CONNECTORS.items():
        if name in ("socialweblab", "web"):
            continue
        keys = set(REGISTRY.get(name).capabilities)
        for op, spec in cls.operations.items():
            assert spec.name == op
            assert spec.cap() in keys or (name, spec.cap()) in known_gaps, (name, op, spec.cap())
            if not spec.mutates:
                assert spec.tier == Tier.OBSERVE, (name, op)
            else:
                assert spec.tier >= Tier.REVERSIBLE, (name, op)
            assert hasattr(cls, f"op_{op}"), (name, op)
    for name in ("bluesky", "mastodon", "x", "discord", "instagram", "reddit", "linkedin"):
        for op, spec in CONNECTORS[name].operations.items():
            if spec.mutates and op not in ("delete_post", "set_bot_label", "like"):
                assert spec.tier == Tier.EXTERNAL and spec.content_param, (name, op)
    for op in ("create_branch", "commit_file", "create_pull_request", "dispatch_workflow"):
        assert CONNECTORS["github"].operations[op].tier == Tier.REVERSIBLE


@pytest.mark.parametrize("mod,path", [(bsky_sim, "/xrpc/com.atproto.server.getSession"), (md_sim, "/api/v1/instance"),
                                      (gh_sim, "/app"), (dc_sim, "/api/v10/users/@me"), (x_sim, "/2/users/me"),
                                      (ig_sim, "/v21.0/123")])
def test_every_simulator_marks_its_responses(mod, path):
    async def main():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mod.create_app()), base_url="http://sim") as c:
            r = await c.get(path)
        assert r.headers["x-agentrs-simulator"] == "1"
    run(main())


def test_egress_policy_still_applies_to_connectors():
    async def main():
        st = bsky_sim.BlueskyState()
        h = Harness("bluesky", bsky_sim.create_app(st), "bsky.sim", secrets={"app_password": "x"},
                    account={"handle": "a.bsky.sim"})
        h.conn.base_url = "http://evil.example"  # not on the egress allowlist
        with pytest.raises(EgressDenied):
            await h.conn.validate_credentials()
        assert h.egress[-1]["outcome"] == "denied"
    run(main())


# ====================================================================== Bluesky
def _bluesky():
    st = bsky_sim.BlueskyState()
    did = st.add_user("agent.bsky.sim", app_password="app-pw-0001")
    st._write(did, "app.bsky.actor.profile", {"$type": "app.bsky.actor.profile", "displayName": "Agentrs Bot",
                                               "description": "Automated account operated by the agentrs lab"}, "self")
    h = Harness("bluesky", bsky_sim.create_app(st), "bsky.sim", secrets={"app_password": "app-pw-0001"},
                account={"handle": "agent.bsky.sim", "id": "bsky-1"})
    return st, h


def test_bluesky_session_post_outcome_and_policies():
    async def main():
        st, h = _bluesky()
        v = await h.conn.validate_credentials()
        assert v["ok"] and v["handle"] == "agent.bsky.sim" and v["bot_label"] is False
        assert h.creds.has("access_jwt") and h.creds.has("refresh_jwt")  # session persisted via credentials.update

        lab = await h.do("set_bot_label")
        assert lab.tier == int(Tier.REVERSIBLE) and lab.observation["self_labels"] == ["bot"]
        prof = (await h.conn.observe({"operation": "read_profile", "params": {}}))["profile"]
        assert [lb["val"] for lb in prof["labels"]] == ["bot"]
        assert prof["description"].startswith("Automated account")  # rest of the profile record preserved

        res = await h.do("create_post", text="Hello from a simulated agent: https://example.org/about")
        uri = res.provider_object_id
        assert res.status == "executed" and uri.startswith(f"at://{v['did']}/app.bsky.feed.post/")
        assert st.records[uri]["value"]["facets"][0]["features"][0]["uri"] == "https://example.org/about"
        feed = await h.conn.observe({"operation": "read_author_feed", "params": {"limit": 5}})
        assert feed["feed"][0]["post"]["uri"] == uri

        before = await h.conn.fetch_outcome(uri)
        assert before["state"] == "visible" and before["occurred"]["first_response"] is False
        assert before["counts"] == {"replies": 0, "reposts": 0, "likes": 0, "quotes": 0}
        st.clock.advance(120)
        st.simulate_reply(uri, "Welcome to Bluesky!", by="carol.bsky.sim")
        st.simulate_like(uri, by="erin.bsky.sim")
        st.simulate_repost(uri, by="frank.bsky.sim")
        after = await h.conn.fetch_outcome(uri)
        assert after["counts"] == {"replies": 1, "reposts": 1, "likes": 1, "quotes": 0}
        assert after["occurred"]["first_response"] and after["occurred"]["positive_feedback"]
        assert after["occurred_at"]["first_response"] >= time.time() + 100  # sim clock advanced 120 s

        # provider policy: identical text (normalised) refused
        ok, why = h.allowed("create_post", text="  hello from a simulated agent:   https://example.org/about ")
        assert not ok and "duplicate" in why

        # replies only to accounts that addressed us (or with explicit human approval)
        other = st.post("dave.bsky.sim", "Unrelated musings about the weather")
        ok, why = h.allowed("reply", parent_uri=other, text="Interesting!")
        assert not ok and "did not mention" in why
        with pytest.raises(ConnectorError) as e:  # the connector re-checks even without the gateway
            await h.conn.op_reply(parent_uri=other, text="Interesting!")
        assert e.value.status == 403
        assert h.allowed("reply", parent_uri=other, text="Interesting!", approved_reply=True)[0]

        mention = st.simulate_mention("hey @agent.bsky.sim what do you think?", by="dave.bsky.sim")
        assert not h.allowed("reply", parent_uri=mention, text="I think it's great")[0]
        notes = await h.conn.observe({"operation": "read_notifications", "params": {}})
        assert {n["reason"] for n in notes["notifications"]} >= {"reply", "like", "repost", "mention"}
        r = await h.do("reply", parent_uri=mention, text="I think it's great")
        rec = st.records[r.provider_object_id]["value"]
        assert rec["reply"]["parent"]["uri"] == mention and rec["reply"]["root"]["uri"] == mention

        # token expiry -> refreshSession, refreshed tokens persisted through the credential handle
        old = h.secret("access_jwt")
        st.expire_access_tokens()
        await h.conn.observe({"operation": "read_profile", "params": {}})
        assert h.secret("access_jwt") != old

        st.simulate_takedown(uri)
        gone = await h.conn.fetch_outcome(uri)
        assert gone["state"] == "removed" and gone["occurred"]["removed"]
        assert "ratelimit-remaining" in h.conn.last_rate_limits

        rv = await h.conn.revoke()
        assert rv["revoked_at_provider"] is True and "App passwords" in rv["reason"]
    run(main())


def test_bluesky_rate_limit_and_bad_password():
    async def main():
        st, h = _bluesky()
        st.limits["write_points"] = (6, 3600)  # two posts (3 points each)
        await h.do("create_post", text="one")
        await h.do("create_post", text="two")
        with pytest.raises(ConnectorError) as e:
            await h.do("create_post", text="three")
        assert e.value.status == 429 and h.conn.last_rate_limits["ratelimit-remaining"] == "0"

        h2 = Harness("bluesky", bsky_sim.create_app(st), "bsky.sim", secrets={"app_password": "wrong"},
                     account={"handle": "agent.bsky.sim", "id": "bsky-2"})
        with pytest.raises(ConnectorError) as e:
            await h2.conn.validate_credentials()
        assert e.value.status == 401
    run(main())


# ====================================================================== Mastodon
REDIRECT = "https://agentrs.example/oauth/callback"


def test_mastodon_oauth_authorization_code_pkce_with_human_approval_and_revoke():
    async def main():
        st = md_sim.MastodonState(token_ttl=3600)
        st.add_account("agentrs_bot", bot=True)
        app = md_sim.create_app(st)
        h = Harness("mastodon", app, "mastodon.sim", secrets={}, account={"username": "agentrs_bot", "id": "m-oauth"})
        reg = await h.conn.register_app("agentrs harness", REDIRECT, "read write", website="https://agentrs.example")
        assert reg["client_id"] and "client_secret" not in reg and h.creds.has("client_secret")

        verifier, challenge = make_pkce()
        url = h.conn.authorization_url(REDIRECT, "read write", "state-123", challenge)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mastodon.sim") as human:
            page = await human.get(url)  # what the operator sees in the browser
            assert page.status_code == 200 and "Authorization required" in page.text and "SIMULATED" in page.text
            denied = await human.get(url + "&simulate_human_approval=0")
            assert denied.status_code == 302 and "error=access_denied" in denied.headers["location"]
            ok = await human.get(url + "&simulate_human_approval=1")  # the operator clicks "Authorize"
            loc = ok.headers["location"]
            assert ok.status_code == 302 and loc.startswith(REDIRECT)
            q = parse_qs(urlsplit(loc).query)
            assert q["state"] == ["state-123"]
            code = q["code"][0]
            code2 = parse_qs(urlsplit((await human.get(url + "&simulate_human_approval=1")).headers["location"]).query)["code"][0]
            bad_scope = await human.get(url.replace("scope=read+write", "scope=admin%3Aread") + "&simulate_human_approval=1")
            assert "error=invalid_scope" in bad_scope.headers["location"]

        with pytest.raises(ConnectorError) as e:  # PKCE: wrong verifier
            await h.conn.exchange_code(code2, REDIRECT, "not-the-verifier")
        assert e.value.status == 400 and e.value.code == "invalid_grant"
        tok = await h.conn.exchange_code(code, REDIRECT, verifier)
        assert tok["refreshable"] and "access_token" not in tok and "read" in tok["scope"]
        with pytest.raises(ConnectorError):  # codes are single use
            await h.conn.exchange_code(code, REDIRECT, verifier)
        v = await h.conn.validate_credentials()
        assert v["acct"] == "agentrs_bot" and v["bot"] is True

        old = h.secret("access_token")
        st.clock.advance(3601)  # access token expires -> refresh_token grant
        assert (await h.conn.validate_credentials())["ok"]
        assert h.secret("access_token") != old

        rv = await h.conn.revoke()
        assert rv["revoked_at_provider"] is True
        with pytest.raises(ConnectorError) as e:
            await h.conn.validate_credentials()
        assert e.value.status == 401
    run(main())


def _mastodon(scopes="read write", username="agentrs_bot", bot=True):
    st = md_sim.MastodonState()
    st.add_account(username, bot=bot)
    tok = st.issue_token(username, scopes)
    return st, Harness("mastodon", md_sim.create_app(st), "mastodon.sim", secrets={"access_token": tok},
                       account={"username": username, "id": f"m-{username}"})


def test_mastodon_posts_outcomes_scopes_and_policies():
    async def main():
        st, h = _mastodon()
        assert (await h.conn.validate_credentials())["bot"] is True
        inst = await h.conn.observe({"operation": "read_instance", "params": {}})
        assert len(inst["rules"]) == 3 and inst["max_characters"] == 500

        res = await h.do("create_post", text="Hello fediverse from a simulated bot")
        sid = res.provider_object_id
        assert res.observation["visibility"] == "unlisted"  # bot default
        a = h.act("create_post", text="Idempotent hello")
        r1, r2 = await h.conn.execute(a), await h.conn.execute(a)  # retried request, same Idempotency-Key
        assert r1.provider_object_id == r2.provider_object_id
        assert sum(1 for s in st.statuses.values() if s["text"] == "Idempotent hello") == 1

        before = await h.conn.fetch_outcome(sid)
        assert before["occurred"]["first_response"] is False and before["counts"]["replies"] == 0
        st.clock.advance(30)
        st.simulate_reply(sid, "nice to meet you", by="carol")
        st.simulate_favourite(sid, by="erin")
        st.simulate_reblog(sid, by="frank")
        after = await h.conn.fetch_outcome(sid)
        assert after["counts"] == {"replies": 1, "reposts": 1, "likes": 1}
        assert after["occurred"]["first_response"] and "first_response" in after["occurred_at"]

        home = await h.conn.observe({"operation": "read_home_timeline", "params": {"limit": 5}})
        assert home["statuses"][0]["id"] == r1.provider_object_id

        assert not h.allowed("create_post", text="hello FEDIVERSE from a simulated bot")[0]  # duplicate
        other = st.simulate_post("dave", "unrelated toot")
        ok, why = h.allowed("reply", in_reply_to_id=other, text="hm")
        assert not ok and "did not mention" in why
        with pytest.raises(ConnectorError):
            await h.conn.op_reply(in_reply_to_id=other, text="hm")
        mention = st.simulate_mention("@agentrs_bot can you summarise the rules?", by="erin")
        notes = await h.conn.observe({"operation": "read_notifications", "params": {"types": ["mention"]}})
        assert {n["type"] for n in notes["notifications"]} == {"mention"}
        rep = await h.do("reply", in_reply_to_id=mention, text="Sure: be kind, no spam.")
        assert st.statuses[rep.provider_object_id]["text"].startswith("@erin ")
        assert st.statuses[rep.provider_object_id]["in_reply_to_id"] == mention
        assert "x-ratelimit-remaining" in h.conn.last_rate_limits

        st.simulate_removal(sid)
        assert (await h.conn.fetch_outcome(sid))["state"] == "removed"

        _st2, ro = _mastodon(scopes="read")
        with pytest.raises(ConnectorError) as e:  # token without write scope
            await ro.conn.op_create_post(text="should fail")
        assert e.value.status == 403 and "outside the authorized scopes" in str(e.value)

        _st3, human = _mastodon(username="not_a_bot", bot=False)
        v = await human.conn.validate_credentials()
        assert v["bot"] is False and v["warning"]
        ok, why = human.allowed("create_post", text="hi")
        assert not ok and "bot" in why
    run(main())


# ====================================================================== GitHub
WRITE_PERMS = {"contents": "write", "pull_requests": "write", "issues": "read", "checks": "read", "actions": "write"}
_KEYS: dict[str, tuple[str, str]] = {}


def _app_key() -> tuple[str, str]:
    if "k" not in _KEYS:
        _KEYS["k"] = gh_sim.generate_app_key()
    return _KEYS["k"]


def _github(perms=None, installed=("acme/widgets",), account_repos=("acme/widgets",), target="receiver",
            transport=None, base_url_port=""):
    secret = "whsec-simulated-0001"
    received: list[tuple[str, bool, dict]] = []

    def receiver(headers: dict, body: bytes):
        ok = verify_github_signature(secret, body, headers.get("X-Hub-Signature-256"))
        received.append((headers["X-GitHub-Event"], ok, json.loads(body)))
        return 200 if ok else 401

    st = gh_sim.GitHubState(webhook_target=receiver if target == "receiver" else target, webhook_secret=secret)
    priv, pub = _app_key()
    st.register_app(4242, pub, "agentrs-sim")
    st.add_repo("acme/widgets", files={"README.md": "# widgets\n", "src/app.py": "print('hi')\n"},
                workflows={"ci.yml": {"id": 77, "name": "CI"}, "release.yml": {"id": 78, "dispatch": False}})
    st.add_repo("acme/other")
    st.add_repo("acme/secret-sauce")
    st.install(4242, 99, list(installed), perms or WRITE_PERMS)
    app = gh_sim.create_app(st)
    host = "127.0.0.1" if transport is not None else "api.github.sim"
    h = Harness("github", app, host, port=base_url_port, transport=transport, secrets={"app_private_key": priv},
                account={"app_id": 4242, "installation_id": 99, "repositories": list(account_repos)})
    return st, h, received, secret, app


async def _github_pr_flow(st, h, received):
    v = await h.conn.validate_credentials()
    assert v["app_slug"] == "agentrs-sim" and v["permissions"]["contents"] == "write"
    assert v["repositories"] == ["acme/widgets"] and v["missing_repositories"] == []
    f = await h.conn.observe({"operation": "read_file", "params": {"repo": "acme/widgets", "path": "README.md"}})
    assert f["text"] == "# widgets\n"
    st.simulate_issue("acme/widgets", "Bug: README is too short", by="reporter")
    issues = await h.conn.observe({"operation": "list_issues", "params": {"repo": "acme/widgets"}})
    assert [i["title"] for i in issues["issues"]] == ["Bug: README is too short"]

    b = await h.do("create_branch", repo="acme/widgets", branch="agentrs/fix-readme")
    assert b.tier == int(Tier.REVERSIBLE) and b.observation["ref"] == "refs/heads/agentrs/fix-readme"
    c = await h.do("commit_file", repo="acme/widgets", branch="agentrs/fix-readme", path="README.md",
                   content="# widgets\n\nA longer README.\n", message="docs: expand README")
    assert c.observation["created"] is False  # existing file: sha looked up and sent
    pr = await h.do("create_pull_request", repo="acme/widgets", head="agentrs/fix-readme", title="Expand README",
                    body="Automated change proposed by an agent.")
    ref = pr.provider_object_id
    assert ref == "acme/widgets#2"
    assert st.repos["acme/widgets"]["issues"][2]["user"] == "agentrs-sim[bot]"

    before = await h.conn.fetch_outcome(ref)
    assert before["state"] == "open"
    assert {k: before["occurred"][k] for k in ("ci_completed", "ci_passed", "reviewed", "merged")} == dict.fromkeys(
        ("ci_completed", "ci_passed", "reviewed", "merged"), False)

    st.run_ci("acme/widgets", "agentrs/fix-readme", conclusion="success")
    st.simulate_review("acme/widgets", 2, "APPROVED", "LGTM")
    st.simulate_merge("acme/widgets", 2)
    after = await h.conn.fetch_outcome(ref)
    assert after["state"] == "merged"
    for ev in ("ci_completed", "ci_passed", "reviewed", "approved", "merged"):
        assert after["occurred"][ev] is True and after["occurred_at"][ev] > 0, ev
    assert after["counts"]["approvals"] == 1 and after["counts"]["check_runs"] == 1
    events = [(e, ok) for e, ok, _ in received]
    assert ("check_run", True) in events and ("pull_request_review", True) in events and ("pull_request", True) in events
    return ref


def test_github_app_jwt_installation_token_branch_commit_pr_ci_outcome():
    async def main():
        st, h, received, _secret, _app = _github()
        await _github_pr_flow(st, h, received)
        assert received[0][2]["installation"]["id"] == 99

        d = await h.do("dispatch_workflow", repo="acme/widgets", workflow="ci.yml", ref="agentrs/fix-readme")
        assert d.observation["dispatched"] and st.repos["acme/widgets"]["workflow_runs"][0]["event"] == "workflow_dispatch"
        with pytest.raises(ConnectorError) as e:
            await h.conn.op_dispatch_workflow(repo="acme/widgets", workflow="release.yml", ref="agentrs/fix-readme")
        assert e.value.status == 422
        st.run_ci("acme/widgets", "agentrs/fix-readme", conclusion="failure", name="lint")
        ci = await h.conn.fetch_outcome("acme/widgets@agentrs/fix-readme")
        assert ci["occurred"]["ci_completed"] and not ci["occurred"]["ci_passed"] and ci["counts"]["check_failures"] == 1

        # protected default branch at the provider (bypassing the connector's own refusal)
        tok = h.conn._inst_token
        r = await h.http.request("PUT", "http://api.github.sim/repos/acme/widgets/contents/NEW.md",
                                 headers={"Authorization": f"Bearer {tok}"},
                                 json={"message": "x", "content": "eA==", "branch": "main"})
        assert r.status_code == 409 and r.headers["x-agentrs-simulator"] == "1"

        rl = await h.conn.get_rate_limits()
        assert rl["provider"]["core"]["limit"] == 5000 and "x-ratelimit-remaining" in rl["observed"]

        st.expire_tokens()  # 1-hour installation tokens: re-minted transparently on 401
        assert (await h.conn.observe({"operation": "list_repositories", "params": {}}))["repositories"]
        current = h.conn._inst_token
        rv = await h.conn.revoke()
        assert rv["revoked_at_provider"] is True and st.tokens[current]["revoked"]
        assert (await h.conn.observe({"operation": "read_repository", "params": {"repo": "acme/widgets"}}))  # re-minted
    run(main())


def test_github_permissions_repository_selection_and_policy_refusals():
    async def main():
        st, h, _, _, _ = _github(perms={"contents": "read", "pull_requests": "read", "checks": "read"})
        v = await h.conn.validate_credentials()
        assert v["permissions"]["contents"] == "read"
        with pytest.raises(ConnectorError) as e:  # contents:read token cannot create refs
            await h.do("create_branch", repo="acme/widgets", branch="agentrs/x")
        assert e.value.status == 403 and "Resource not accessible by integration" in str(e.value)
        repo = st.repos["acme/widgets"]
        repo["branches"]["agentrs/x"] = repo["branches"]["main"]
        with pytest.raises(ConnectorError) as e:  # ... nor commit
            await h.do("commit_file", repo="acme/widgets", branch="agentrs/x", path="a.txt", content="a", message="m")
        assert e.value.status == 403
        with pytest.raises(ConnectorError) as e:  # actions: none
            await h.do("dispatch_workflow", repo="acme/widgets", workflow="ci.yml", ref="agentrs/x")
        assert e.value.status == 403

        # repository selection: the installation token cannot see repositories outside the installation
        r = await h.http.get("http://api.github.sim/repos/acme/other", headers={"Authorization": f"Bearer {h.conn._inst_token}"})
        assert r.status_code == 404
        _st2, h2, _, _, _ = _github(account_repos=("acme/widgets", "acme/other"))
        with pytest.raises(ConnectorError) as e:  # down-scoping to a repo the installation lacks fails
            await h2.conn.validate_credentials()
        assert e.value.status == 422

        # connector / provider-policy refusals (before any request)
        _st3, h3, _, _, _ = _github()
        for op, params, needle in [
            ("create_branch", {"repo": "acme/secret-sauce", "branch": "agentrs/y"}, "not in this account"),
            ("create_branch", {"repo": "acme/widgets", "branch": "feature/y"}, "agentrs/"),
            ("commit_file", {"repo": "acme/widgets", "branch": "main", "path": "a", "content": "a", "message": "m"}, "agentrs/"),
            ("commit_file", {"repo": "acme/widgets", "branch": "agentrs/y", "path": ".github/workflows/x.yml",
                             "content": "a", "message": "m"}, ".github"),
        ]:
            ok, why = h3.allowed(op, **params)
            assert not ok and needle in why, (op, why)
        with pytest.raises(ConnectorError) as e:
            await h3.conn.op_commit_file(repo="acme/widgets", branch="main", path="a", content="a", message="m")
        assert e.value.status == 403
        with pytest.raises(ConnectorError) as e:
            await h3.conn.op_read_repository(repo="acme/secret-sauce")
        assert e.value.status == 403
        h3.conn.branch_prefix = ""  # even without a prefix rule the default branch is refused at execution time
        with pytest.raises(ConnectorError) as e:
            await h3.conn.op_commit_file(repo="acme/widgets", branch="main", path="a", content="a", message="m")
        assert e.value.status == 403 and "default branch" in str(e.value)
    run(main())


def test_github_jwt_validation_at_simulator():
    async def main():
        st, h, _, _, _app = _github()
        priv, _ = _app_key()
        import jwt as pyjwt
        now = int(time.time())
        too_long = pyjwt.encode({"iat": now, "exp": now + 3600, "iss": "4242"}, priv, algorithm="RS256")
        r = await h.http.post("http://api.github.sim/app/installations/99/access_tokens",
                              headers={"Authorization": f"Bearer {too_long}"})
        assert r.status_code == 401 and "too far in the future" in r.json()["message"]
        _, other_pub = gh_sim.generate_app_key()
        st.register_app(4242, other_pub)  # key rotated: old private key no longer valid
        with pytest.raises(ConnectorError) as e:
            await h.conn.validate_credentials()
        assert e.value.status == 401
    run(main())


def test_github_webhook_signature_verification():
    secret = "whsec-abc"
    body = b'{"action":"completed","check_run":{"conclusion":"success"}}'
    good = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_github_signature(secret, body, good)
    assert verify_github_signature(secret.encode(), body, good.upper().replace("SHA256=", "sha256="))
    assert not verify_github_signature(secret, body.replace(b"success", b"failure"), good)  # tampered body
    assert not verify_github_signature("other-secret", body, good)
    assert not verify_github_signature(secret, body, good[:-2] + "00")
    assert not verify_github_signature(secret, body, "sha1=" + hmac.new(secret.encode(), body, hashlib.sha1).hexdigest())
    for bad in (None, "", "sha256=", "garbage"):
        assert not verify_github_signature(secret, body, bad)
    assert not verify_github_signature("", body, good)


# ====================================================================== Discord
def _discord():
    st = dc_sim.DiscordState()
    general = st.add_channel("general")
    play = st.add_channel("bot-playground")
    mods = st.add_channel("mods-only", bot_can_view=False)
    news = st.add_channel("announcements", bot_can_send=False)
    h = Harness("discord", dc_sim.create_app(st), "discord.sim", base_path="/api/v10", secrets={"bot_token": st.bot_token},
                account={"id": "dc-1", "allowed_channel_ids": [play, mods, news], "public_key": st.public_key_hex})
    return st, h, general, play, mods, news


def test_discord_messages_channel_permissions_outcomes_and_interactions():
    async def main():
        st, h, general, play, mods, news = _discord()
        v = await h.conn.validate_credentials()
        assert v["bot"] and v["id"] == st.bot_user["id"]
        st.simulate_message(play, "hello everyone, who is this bot?", by="carol")
        msgs = (await h.conn.observe({"operation": "read_messages", "params": {"channel_id": play}}))["messages"]
        assert msgs[0]["content"].startswith("hello everyone")

        res = await h.do("send_message", channel_id=play, content="Hi! I'm an automated helper (bot).")
        ref = res.provider_object_id
        assert ref.startswith(f"{play}/")
        sent = st.find(play, ref.split("/")[1])
        assert sent["mention_everyone"] is False and len(sent["nonce"]) <= 25

        # only configured channels: policy refusal, connector refusal, and the provider's own 403s
        ok, why = h.allowed("send_message", channel_id=general, content="hi")
        assert not ok and "configured channel" in why
        with pytest.raises(ConnectorError) as e:
            await h.conn.op_send_message(channel_id=general, content="hi")
        assert e.value.status == 403
        with pytest.raises(ConnectorError) as e:
            await h.conn.op_read_messages(channel_id=mods)
        assert e.value.status == 403 and e.value.code == "50001"
        with pytest.raises(ConnectorError) as e:
            await h.conn.op_send_message(channel_id=news, content="hi")
        assert e.value.status == 403 and e.value.code == "50013"
        assert not h.allowed("send_message", channel_id=play, content="@everyone look at this")[0]
        assert not h.allowed("send_message", channel_id=play, content="hi! i'm an automated helper (bot).")[0]

        before = await h.conn.fetch_outcome(ref)
        assert before["counts"] == {"replies": 0, "reactions": 0} and not before["occurred"]["first_response"]
        st.clock.advance(10)
        st.simulate_message(play, "thanks bot!", by="dave", reply_to=ref.split("/")[1])
        st.simulate_reaction(play, ref.split("/")[1], by="erin")
        after = await h.conn.fetch_outcome(ref)
        assert after["counts"] == {"replies": 1, "reactions": 1} and after["occurred"]["first_response"]
        assert after["occurred_at"]["first_response"] > 0

        # interactions: Ed25519-signed delivery to the endpoint, then a callback response
        got: list[dict] = []

        def receiver(headers: dict, body: bytes):
            if not verify_discord_signature(st.public_key_hex, headers.get("X-Signature-Timestamp"), body,
                                            headers.get("X-Signature-Ed25519")):
                return 401
            p = json.loads(body)
            if p["type"] == 1:
                return 200, b'{"type":1}'
            got.append(p)
            return 202, b""

        st.interactions_endpoint = receiver
        assert st.verify_endpoint() == {"ping_ok": True, "rejects_bad_signature": True}
        st.interactions_endpoint = lambda headers, body: (200, b'{"type":1}')  # a receiver that skips verification
        assert st.verify_endpoint()["rejects_bad_signature"] is False
        st.interactions_endpoint = receiver
        payload = st.emit_interaction(play, "ask", {"question": "what can you do?"}, by="carol")
        assert got and got[0]["id"] == payload["id"] and got[0]["data"]["options"][0]["value"] == "what can you do?"
        r = await h.do("respond_interaction", interaction_id=payload["id"], interaction_token=payload["token"],
                       channel_id=play, content="I can answer questions about this server.")
        assert r.provider_object_id.startswith(f"{play}/")
        assert st.find(play, r.provider_object_id.split("/")[1])["interaction_metadata"]["id"] == payload["id"]
        with pytest.raises(ConnectorError) as e:
            await h.conn.op_respond_interaction(interaction_id=payload["id"], interaction_token=payload["token"],
                                                channel_id=play, content="again")
        assert e.value.status == 400 and e.value.code == "40060"
        late = st.emit_interaction(play, "ask", by="carol")
        st.clock.advance(4)  # the initial response must come within 3 s
        with pytest.raises(ConnectorError) as e:
            await h.conn.op_respond_interaction(interaction_id=late["id"], interaction_token=late["token"],
                                                channel_id=play, content="too late")
        assert e.value.status == 404

        st.limits["post_message"] = (1, 60)
        st.clock.advance(61)  # new window
        await h.conn.op_send_message(channel_id=play, content="first in window")
        with pytest.raises(ConnectorError) as e:
            await h.conn.op_send_message(channel_id=play, content="second in window")
        assert e.value.status == 429 and e.value.retry_after and e.value.retry_after >= 1
        assert (await h.conn.revoke())["revoked_at_provider"] is False
    run(main())


def test_discord_signature_verification():
    from nacl.signing import SigningKey
    sk = SigningKey.generate()
    pk = sk.verify_key.encode().hex()
    ts, body = "1767225600", b'{"type":2,"data":{"name":"ask"}}'
    sig = sk.sign(ts.encode() + body).signature.hex()
    assert verify_discord_signature(pk, ts, body, sig)
    assert not verify_discord_signature(pk, ts, body + b" ", sig)  # tampered body
    assert not verify_discord_signature(pk, "1767225601", body, sig)  # tampered timestamp
    assert not verify_discord_signature(SigningKey.generate().verify_key.encode().hex(), ts, body, sig)
    assert not verify_discord_signature(pk, ts, body, "zz" * 64)
    assert not verify_discord_signature(pk, ts, body, None)
    assert not verify_discord_signature("not-hex", ts, body, sig)
    assert verify_discord_signature(pk, ts, body, sig, max_age_s=300, now=1767225700)
    assert not verify_discord_signature(pk, ts, body, sig, max_age_s=300, now=1767226000)  # replayed


# ====================================================================== X
def _x(**account):
    st = x_sim.XState()
    tok = st.issue_token("agentrs_bot")
    h = Harness("x", x_sim.create_app(st), "x.sim", base_path="/2",
                secrets={"access_token": tok["access_token"], "refresh_token": tok["refresh_token"]},
                account={"username": "agentrs_bot", "client_id": "sim-client", **account})
    return st, h


def test_x_posts_outcomes_duplicates_bulk_likes_and_replies():
    async def main():
        st, h = _x(max_likes_per_hour=2)
        v = await h.conn.validate_credentials()
        assert v["username"] == "agentrs_bot"
        res = await h.do("create_post", text="Testing the simulated X API v2")
        tid = res.provider_object_id
        before = await h.conn.fetch_outcome(tid)
        assert before["counts"] == {"replies": 0, "reposts": 0, "likes": 0, "quotes": 0}
        assert before["occurred"]["first_response"] is False
        st.clock.advance(45)
        st.simulate_reply(tid, "@agentrs_bot neat", by="carol")
        st.simulate_like(tid, by="erin")
        st.simulate_like(tid, by="gina")
        st.simulate_retweet(tid, by="frank")
        after = await h.conn.fetch_outcome(tid)
        assert after["counts"] == {"replies": 1, "reposts": 1, "likes": 2, "quotes": 0}
        assert after["occurred"]["first_response"] and after["occurred_at"]["first_response"] > 0
        assert "x-rate-limit-remaining" in h.conn.last_rate_limits

        ok, why = h.allowed("create_post", text="testing the simulated x api v2")
        assert not ok and "duplicate" in why
        with pytest.raises(ConnectorError) as e:  # and the provider itself refuses identical text
            await h.conn.op_create_post(text="Testing the simulated X API v2")
        assert e.value.status == 403 and "duplicate content" in str(e.value)

        t1, t2, t3 = (st.simulate_mention(f"post number {i}", by="carol") for i in range(3))
        await h.do("like", tweet_id=t1)
        assert not h.allowed("like", tweet_id=t1)[0]  # same object twice
        await h.do("like", tweet_id=t2)
        ok, why = h.allowed("like", tweet_id=t3)
        assert not ok and "bulk" in why

        plain = st.simulate_mention("just my own thoughts", by="dave")
        ok, why = h.allowed("reply", in_reply_to_tweet_id=plain, text="I agree")
        assert not ok and "did not mention" in why
        assert h.allowed("reply", in_reply_to_tweet_id=plain, text="I agree", approved_reply=True)[0]
        with pytest.raises(ConnectorError) as e:  # the API's own reply restriction still applies
            await h.do("reply", in_reply_to_tweet_id=plain, text="I agree", approved_reply=True)
        assert e.value.status == 403 and "not been mentioned" in str(e.value)
        m = st.simulate_mention("@agentrs_bot what is this account?", by="dave")
        await h.conn.observe({"operation": "read_post", "params": {"tweet_id": m}})
        r = await h.do("reply", in_reply_to_tweet_id=m, text="A research bot (simulated).")
        assert st.tweets[r.provider_object_id]["in_reply_to_user_id"] == st.by_username["dave"]

        old = h.secret("access_token")
        st.expire_tokens()
        assert (await h.conn.validate_credentials())["ok"]
        assert h.secret("access_token") != old

        st.search_enabled = False  # lower access tiers: no search -> counts only
        partial = await h.conn.fetch_outcome(tid)
        assert partial["occurred"]["first_response"] and "first_response" not in partial["occurred_at"]
        assert partial["raw"]["search_unavailable"] == 403

        d = await h.do("delete_post", tweet_id=tid)
        assert d.observation["deleted"] and d.tier == int(Tier.REVERSIBLE)
        assert (await h.conn.fetch_outcome(tid))["state"] == "removed"

        app_only = Harness("x", x_sim.create_app(st), "x.sim", base_path="/2",
                           secrets={"access_token": st.issue_app_only_token()}, account={"username": "agentrs_bot"})
        with pytest.raises(ConnectorError) as e:
            await app_only.conn.op_create_post(text="app-only tokens cannot post")
        assert e.value.status == 403 and "Unsupported Authentication" in str(e.value)
    run(main())


# ====================================================================== Instagram
def test_instagram_publish_outcomes_quota_and_personal_account_refusal():
    async def main():
        st = ig_sim.InstagramState(publish_quota=2)
        uid, tok = st.add_account("acme_studio", "BUSINESS")
        puid, ptok = st.add_account("jane_personal", "PERSONAL")
        app = ig_sim.create_app(st)
        h = Harness("instagram", app, "graph.sim", base_path="/v21.0", secrets={"access_token": tok},
                    account={"ig_user_id": uid})
        ok, why = h.allowed("publish_image", image_url="https://cdn.example/p.jpg", caption="hi")
        assert not ok and "validate_credentials" in why
        v = await h.conn.validate_credentials()
        assert v["account_type"] == "BUSINESS"

        res = await h.do("publish_image", image_url="https://cdn.example/photo.jpg",
                         caption="Hello from a simulated professional account #test")
        mid = res.provider_object_id
        assert (await h.conn.observe({"operation": "read_publishing_limit", "params": {}}))["quota_usage"] == 1
        before = await h.conn.fetch_outcome(mid)
        assert before["counts"] == {"likes": 0, "comments": 0} and not before["occurred"]["first_response"]
        st.clock.advance(300)
        st.simulate_comment(mid, "Lovely!", by="carol")
        st.simulate_like(mid, 3)
        after = await h.conn.fetch_outcome(mid)
        assert after["counts"] == {"likes": 3, "comments": 1} and after["occurred"]["first_response"]
        comments = await h.conn.observe({"operation": "read_comments", "params": {"media_id": mid}})
        assert comments["comments"][0]["text"] == "Lovely!"
        assert "x-business-use-case-usage" in h.conn.last_rate_limits

        with pytest.raises(ConnectorError) as e:  # only JPEG images can be published
            await h.conn.op_publish_image(image_url="https://cdn.example/photo.png", caption="png")
        assert e.value.status == 422
        await h.conn.op_publish_image(image_url="https://cdn.example/two.jpg", caption="second")
        with pytest.raises(ConnectorError) as e:  # 24 h publishing quota
            await h.conn.op_publish_image(image_url="https://cdn.example/three.jpg", caption="third")
        assert e.value.status == 429
        st.simulate_removal(mid)
        assert (await h.conn.fetch_outcome(mid))["state"] == "removed"

        hp = Harness("instagram", app, "graph.sim", base_path="/v21.0", secrets={"access_token": ptok},
                     account={"ig_user_id": puid})
        with pytest.raises(ConnectorError) as e:
            await hp.conn.validate_credentials()
        assert e.value.status == 403 and "PERSONAL" in str(e.value)
        ok, why = hp.allowed("publish_image", image_url="https://cdn.example/p.jpg", caption="hi")
        assert not ok and "not a professional account" in why
        with pytest.raises(ConnectorError) as e:
            await hp.conn.op_publish_image(image_url="https://cdn.example/p.jpg", caption="hi")
        assert e.value.status == 403
        r = await hp.http.post(f"http://graph.sim/v21.0/{puid}/media", headers={"Authorization": f"Bearer {ptok}"},
                               data={"image_url": "https://cdn.example/p.jpg", "caption": "hi"})
        assert r.status_code == 403 and r.json()["error"]["code"] == 10  # the provider refuses too
    run(main())


# ====================================================================== Reddit / LinkedIn
def _gated(provider: str, cfg: ProviderConfig, handler=None):
    calls: list[str] = []

    def default(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(500)

    host = urlsplit(cfg.base_url).hostname
    hosts = [host, "www.reddit.com"]
    http = SafeHTTPClient(EgressPolicy(allowed_domains=hosts, allow_private=hosts), provider=provider,
                          transport=httpx.MockTransport(handler or default))
    store = SecretStore()
    creds = CredentialHandle({"access_token": store.put("memory", "tok", "token-123456")}, store.get)
    ctx = ConnectorContext(config=cfg, http=http, credentials=creds,
                           account={"user_agent": "test:agentrs:0.1 (by /u/agentrs_test)"})
    return make_connector(provider, ctx), calls


def test_reddit_and_linkedin_refuse_by_default():
    async def main():
        for provider, ops in [("reddit", {"read_content": {"subreddit": "test"},
                                          "create_post": {"subreddit": "test", "title": "t", "text": "x"},
                                          "reply": {"parent_fullname": "t3_abc", "text": "x"}}),
                              ("linkedin", {"read_profile": {}, "create_post": {"text": "x"}})]:
            shipped = REGISTRY.get(provider)
            assert shipped.enabled is False
            no_ref = ProviderConfig(provider=provider, enabled=True, base_url=shipped.base_url,
                                    training_data_authorized=False)
            for cfg in (shipped, no_ref):
                conn, calls = _gated(provider, cfg)
                with pytest.raises(ConnectorError) as e:
                    await conn.validate_credentials()
                assert e.value.status == 403
                for op, params in ops.items():
                    with pytest.raises(ConnectorError) as e:
                        await conn.execute(ActionRequest(agent_id="a", provider=provider, operation=op, params=params))
                    assert e.value.status == 403, (provider, op)
                    ok, _why = conn.provider_policy(ActionRequest(agent_id="a", provider=provider, operation=op,
                                                                 params=params), [])
                    assert not ok
                    with pytest.raises(ConnectorError):  # even with bad parameters the gate answers first
                        await conn.execute(ActionRequest(agent_id="a", provider=provider, operation=op, params={}))
                assert calls == []  # nothing ever left the process

        # Reddit additionally needs training_data_authorized to be explicitly declared
        undeclared = ProviderConfig(provider="reddit", enabled=True, authorization_ref="DPA-1",
                                    base_url="https://oauth.reddit.com")
        conn, _ = _gated("reddit", undeclared)
        with pytest.raises(ConnectorError) as e:
            await conn.op_read_content(subreddit="test")
        assert e.value.status == 403 and "explicitly declared" in str(e.value)

        def reddit_api(request: httpx.Request):
            assert request.headers["authorization"] == "bearer token-123456"
            assert request.headers["user-agent"].startswith("test:agentrs")
            if request.url.path == "/api/v1/me":
                return httpx.Response(200, json={"name": "agentrs_test", "id": "u1"})
            return httpx.Response(200, json={"data": {"after": None, "children": [
                {"kind": "t3", "data": {"name": "t3_1", "title": "Hello", "selftext": "", "author": "someone",
                                        "score": 3, "num_comments": 0, "created_utc": 1.0, "permalink": "/r/test/1"}}]}})

        declared_false = ProviderConfig(provider="reddit", enabled=True, authorization_ref="DPA-1",
                                        training_data_authorized=False, base_url="https://oauth.reddit.com")
        conn, _ = _gated("reddit", declared_false, reddit_api)
        v = await conn.validate_credentials()
        assert v["ok"] and v["data_use"] == "non_training_only"
        out = await conn.observe({"operation": "read_content", "params": {"subreddit": "test"}})
        assert out["posts"][0]["title"] == "Hello" and out["data_use"] == "non_training_only"
        assert out["training_data_authorized"] is False
        with pytest.raises(ConnectorError) as e:  # outcomes would be training signals
            await conn.fetch_outcome("t3_1")
        assert e.value.status == 403

        li = ProviderConfig(provider="linkedin", enabled=True, authorization_ref="LI-APP-1", base_url="https://api.linkedin.com")
        conn, _ = _gated("linkedin", li, lambda r: httpx.Response(200, json={"sub": "abc", "name": "Agentrs Lab"}))
        assert (await conn.validate_credentials())["sub"] == "abc"
    run(main())


# ====================================================================== real HTTP (uvicorn)
def test_github_simulator_over_real_http_with_uvicorn_and_webhook_delivery():
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    hook_log: list[tuple[str, bool]] = []
    st, h, _received, secret, app = _github(target=f"http://127.0.0.1:{port}/_test/webhook",
                                           transport=httpx.AsyncHTTPTransport(), base_url_port=f":{port}")

    async def webhook(request: Request):
        body = await request.body()
        ok = verify_github_signature(secret, body, request.headers.get("x-hub-signature-256"))
        hook_log.append((request.headers.get("x-github-event"), ok))
        return {"ok": ok}

    app.add_api_route("/_test/webhook", webhook, methods=["POST"])
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    t = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    t.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        assert server.started

        async def main():
            v = await h.conn.validate_credentials()
            assert v["repositories"] == ["acme/widgets"]
            r = await h.http.get(f"http://127.0.0.1:{port}/installation/repositories",
                                 headers={"Authorization": f"Bearer {h.conn._inst_token}"})
            assert r.status_code == 200 and r.headers["x-agentrs-simulator"] == "1" and "date" in r.headers
            await h.do("create_branch", repo="acme/widgets", branch="agentrs/real-http")
            await h.do("commit_file", repo="acme/widgets", branch="agentrs/real-http", path="docs/notes.md",
                       content="notes\n", message="docs: add notes")
            pr = await h.do("create_pull_request", repo="acme/widgets", head="agentrs/real-http", title="Add notes")
            st.run_ci("acme/widgets", "agentrs/real-http", "success")  # webhook over real HTTP to the receiver route
            out = await h.conn.fetch_outcome(pr.provider_object_id)
            assert out["occurred"]["ci_passed"] and out["state"] == "open"
            await h.http.aclose()
        run(main())
        assert hook_log == [("check_run", True)]
        assert st.deliveries[-1]["status"] == 200
        assert all(e["url_host"] == "127.0.0.1" for e in h.egress)
    finally:
        server.should_exit = True
        t.join(5)
