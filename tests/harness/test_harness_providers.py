"""Provider integrations through the full harness (gateway, policy, accounts, ledger) against the local
provider simulators: test 3 (OAuth connection with human consent), test 6 (authorised external API
action with task-level authorisation and the approval queue), test 7 (signed third-party / provider
events ingested through the control API's webhook endpoint)."""

import asyncio
import json

import httpx

from agentrs.harness.types import ActionRequest

DAY = 86400.0


def _enable(h, provider, base_url, host, caps, rpm=120, posts=20):
    p = h.registry.get(provider)
    p.enabled, p.base_url, p.authorization_ref = True, base_url, "test approval (simulator)"
    for c in caps:
        p.capabilities[c] = True
    p.limits.requests_per_minute, p.limits.posts_per_day = rpm, posts
    p.allowed_domains = [host]


def test_3_oauth_connection_with_simulated_human_consent(tmp_path, scratch_bb):
    from harness_helpers import make_sandbox

    from agentrs.harness.identity import AgentSpec
    from agentrs.harness.net import SafeHTTPClient
    from agentrs.harness.simulators import mastodon as md_sim

    st = md_sim.MastodonState(token_ttl=3600)
    st.add_account("agentrs_bot", bot=True)
    app = md_sim.create_app(st)
    host = "mastodon.sim"
    s = make_sandbox(tmp_path, scratch_bb, transports={"mastodon": httpx.ASGITransport(app=app)}, extra_domains=[host])
    h = s.h
    try:
        _enable(h, "mastodon", f"http://{host}", host, ["read_content", "read_profile", "create_post"])
        h.agents.upsert(AgentSpec(id="comms", display_name="c", role="c", max_tier=3,
                                  allowed_capabilities=["mastodon:read_content", "mastodon:create_post"]))
        req = h.provisioning.request("mastodon", ["create_post", "read_content"], "post approved updates", "comms")
        assert req.state == "PENDING_HUMAN_SETUP"  # the operator creates the account on the instance themselves
        aid = h.provisioning.connect_credentials(req.id, "operator@example.org", "agentrs_bot@mastodon.sim", {},
                                                 "bot", "example org", "labelled", oauth_scopes=["read", "write"])
        h.accounts.update(aid, connection={"username": "agentrs_bot"})
        acc = h.accounts.get(aid)
        conn = h.gateway.connector_for(acc)

        async def flow():
            redirect = "https://agents.example.org/oauth/callback"
            reg = await conn.register_app("agentrs harness", redirect, "read write", website="https://agents.example.org")
            client_id = reg["client_id"]
            url = h.oauth.start("mastodon", aid, f"http://{host}/oauth/authorize", client_id, redirect, ["read", "write"])
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://{host}") as human:
                r = await human.get(url + "&simulate_human_approval=1")  # the operator clicks "Authorize"
            q = httpx.URL(r.headers["location"]).params
            http = SafeHTTPClient(h.egress, transport=httpx.ASGITransport(app=app))
            sec_ref = h.accounts.get(aid).credential_refs["client_secret"]
            res = await h.oauth.complete(q["state"], q["code"], f"http://{host}/oauth/token", client_id, http,
                                         client_secret_ref=sec_ref)
            assert "access_token" not in json.dumps(res)  # no token values returned anywhere
            h.gateway.drop_connector(aid)
            return await h.provisioning.verify(req.id)

        final = asyncio.run(flow())
        assert final.state == "ACTIVE", final.decision_reason
        acc = h.accounts.get(aid)
        assert acc.status == "active" and acc.credential_refs["access_token"].startswith("secret://")
        assert "read" in acc.oauth_scopes
        # replaying the same OAuth state is refused
        import pytest

        from agentrs.harness.oauth import OAuthError

        with h.db.session() as db:
            from agentrs.harness.db import OAuthState

            used = db.query(OAuthState).first().state
        with pytest.raises(OAuthError):
            asyncio.run(h.oauth.complete(used, "x", f"http://{host}/oauth/token", "c",
                                         SafeHTTPClient(h.egress, transport=httpx.ASGITransport(app=app))))
    finally:
        s.srv.stop()
        s.lab.close()


def test_6_authorised_external_action_task_preauthorisation_and_approval_queue(tmp_path, scratch_bb):
    from harness_helpers import make_sandbox

    from agentrs.harness.identity import AgentSpec
    from agentrs.harness.simulators import bluesky as bsky_sim

    st = bsky_sim.BlueskyState()
    st.add_user("agent.bsky.sim", app_password="app-pw-0001")
    app = bsky_sim.create_app(st)
    host = "bsky.sim"
    s = make_sandbox(tmp_path, scratch_bb, transports={"bluesky": httpx.ASGITransport(app=app)}, extra_domains=[host])
    h = s.h
    try:
        _enable(h, "bluesky", f"http://{host}", host, ["read_content", "read_profile", "create_post", "reply"])
        h.agents.upsert(AgentSpec(id="comms", display_name="c", role="c", max_tier=3,
                                  allowed_capabilities=["bluesky:create_post", "bluesky:read_content", "bluesky:reply"]))
        ref = h.secrets.put("memory", "bsky.app_password", "app-pw-0001")
        aid = h.accounts.register("bluesky", "agent.bsky.sim", "bot", "example org", "operator@example.org",
                                  ["create_post", "read_content", "reply"], {"app_password": ref}, ["comms"],
                                  "publish approved announcements", bot_disclosure="labelled", status="active",
                                  connection={"handle": "agent.bsky.sim"})
        h.tasks.create(task_id="pub1", source="human", goal="publish the approved release note", environment="bluesky",
                       required_capabilities=["bluesky:create_post"], allowed_actions=["bluesky:create_post"],
                       verifier={"kind": "manual"}, authorized_tier=3,
                       budget={"pre_approved_operations": ["bluesky:create_post"]})
        h.tasks.set_state("pub1", "running")
        sub = lambda tid, op, **p: asyncio.run(h.gateway.submit(ActionRequest(  # noqa: E731
            agent_id="comms", task_id=tid, provider="bluesky", operation=op, params=p)))
        r = sub("pub1", "create_post", text="Release 1.2 of the agentrs harness is out (automated account).")
        assert r.status == "executed" and r.provider_object_id.startswith("at://")
        uri = r.provider_object_id
        # without task-level pre-authorisation a tier-3 post goes to the approval queue
        h.tasks.create(task_id="pub2", source="human", goal="post", environment="bluesky", required_capabilities=[],
                       allowed_actions=["bluesky:create_post", "bluesky:reply"], verifier={"kind": "manual"},
                       authorized_tier=3)
        h.tasks.set_state("pub2", "running")
        r2 = sub("pub2", "create_post", text="A second update that a human must approve first.")
        assert r2.status == "pending_approval" and r2.approval_id
        n_posts = len(st.records)
        done = asyncio.run(h.gateway.execute_approved(r2.approval_id, "operator@example.org"))
        assert done.status == "executed" and len(st.records) == n_posts + 1
        # unsolicited reply to someone who did not address us: refused by provider policy
        other = st.post("dave.bsky.sim", "Unrelated musings about the weather")
        r3 = sub("pub2", "reply", parent_uri=other, text="Interesting!")
        assert r3.status in ("denied", "pending_approval")
        # the executed post is followed up through the same account; another user's reply appears later
        st.clock.advance(120)
        st.simulate_reply(uri, "Congrats on the release!", by="carol.bsky.sim")
        acc = h.accounts.get(aid)
        fetch = {"harness:bluesky": h.gateway.connector_for(acc).fetch_outcome}
        out = asyncio.run(h.outcomes.run_due(fetch, {"real": 10 ** 11}))
        assert out["done"] >= 1
        obs = h.outcomes.ledger.observations(r.action_id)
        assert obs and obs[-1]["payload"]["occurred"]["first_response"] is True
        assert h.outcomes.ledger.action(r.action_id)["source_type"] == "self"
    finally:
        s.srv.stop()
        s.lab.close()


def test_7_signed_webhook_events_reach_the_ledger_through_the_control_api(tmp_path, scratch_bb):
    from fastapi.testclient import TestClient
    from harness_helpers import make_sandbox

    from agentrs.harness.control_api import create_control_app
    from agentrs.harness.identity import AgentSpec
    from agentrs.harness.simulators import github as gh_sim
    from agentrs.harness.webhooks import verify_github_signature

    secret = "whsec-simulated-0001"
    st = gh_sim.GitHubState(webhook_target=None, webhook_secret=secret)
    priv, pub = gh_sim.generate_app_key()
    st.register_app(4242, pub, "agentrs-sim")
    st.add_repo("acme/widgets", files={"README.md": "# widgets\n"}, workflows={"ci.yml": {"id": 77, "name": "CI"}})
    st.install(4242, 99, ["acme/widgets"], {"contents": "write", "pull_requests": "write", "checks": "read",
                                             "metadata": "read", "issues": "read", "actions": "write"})
    app = gh_sim.create_app(st)
    host = "api.github.sim"
    s = make_sandbox(tmp_path, scratch_bb, transports={"github": httpx.ASGITransport(app=app)}, extra_domains=[host])
    h = s.h
    try:
        _enable(h, "github", f"http://{host}", host,
                ["read_repository", "read_issues", "create_branch", "commit", "create_pull_request", "read_checks"])
        h.agents.upsert(AgentSpec(id="coder", display_name="c", role="dev", max_tier=2,
                                  allowed_capabilities=["github:*"]))
        k = h.secrets.put("memory", "gh.key", priv)
        caps = [c for c, on in h.registry.get("github").capabilities.items() if on]
        aid = h.accounts.register("github", "app:4242/inst:99", "app_installation", "acme", "operator@acme.example", caps,
                                  {"app_private_key": k}, ["coder"], "fix README issues", bot_disclosure="app_bot",
                                  status="active", connection={"app_id": 4242, "installation_id": 99,
                                                               "repositories": ["acme/widgets"]})
        h.tasks.create(task_id="fix1", source="human", goal="expand the README", environment="github",
                       required_capabilities=[], allowed_actions=["github:*"], verifier={"kind": "manual"},
                       authorized_tier=2)
        h.tasks.set_state("fix1", "running")

        def act(op, **p):
            return asyncio.run(h.gateway.submit(ActionRequest(agent_id="coder", task_id="fix1", provider="github",
                                                              operation=op, params=p)))
        assert act("create_branch", repo="acme/widgets", branch="agentrs/readme").status == "executed"
        assert act("commit_file", repo="acme/widgets", branch="agentrs/readme", path="README.md",
                   content="# widgets\n\nMore.\n", message="docs").status == "executed"
        pr = act("create_pull_request", repo="acme/widgets", head="agentrs/readme", title="Expand README",
                 body="Automated change proposed by an agent.")
        assert pr.status == "executed" and pr.provider_object_id == "acme/widgets#1"
        # pushing to the default branch is refused by provider policy before any request leaves
        assert act("commit_file", repo="acme/widgets", branch="main", path="README.md", content="x",
                   message="x").status == "denied"

        async def gh_webhook(headers: dict, body: bytes):
            hd = {k.lower(): v for k, v in headers.items()}
            if not verify_github_signature(secret, body, hd.get("x-hub-signature-256")):
                return False, {}
            ev = json.loads(body)
            num = (ev.get("pull_request") or {}).get("number") or next(
                iter((ev.get("check_run") or {}).get("pull_requests") or [{}]), {}).get("number")
            ref = f"{ev['repository']['full_name']}#{num}" if num else None
            rows = [a for a in h.outcomes.ledger.actions("source_type='self'") if (a["context"] or {}).get("object_ref") == ref]
            for a in rows:
                h.outcomes.record_event(a["action_id"], hd["x-github-event"], None, __import__("time").time(),
                                        hd.get("x-github-delivery", ""), "harness:github",
                                        payload={"action": ev.get("action")})
            return True, {"matched": len(rows)}

        api = TestClient(create_control_app(h, "op", extras={"webhooks": {"github": gh_webhook}}))
        st.webhook_target = lambda headers, body: api.post("/webhooks/github", content=body, headers=headers).status_code
        st.run_ci("acme/widgets", "agentrs/readme", conclusion="success")
        evs = h.outcomes.ledger.events(pr.action_id)
        assert evs and evs[0]["type"] == "check_run"
        # a forged delivery is rejected
        bad = api.post("/webhooks/github", content=b'{"x":1}', headers={"X-Hub-Signature-256": "sha256=00",
                                                                         "X-GitHub-Event": "ping"})
        assert bad.status_code == 401
        out = asyncio.run(h.gateway.connector_for(h.accounts.get(aid)).fetch_outcome(pr.provider_object_id))
        assert out["occurred"]["ci_passed"] is True
    finally:
        s.srv.stop()
        s.lab.close()
