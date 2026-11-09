"""Browser isolation (test 4), real browser execution through the gateway (test 5), email,
S3-compatible artifact storage, and the control API / dashboard."""

import asyncio
import json

import pytest

from agentrs.harness.types import ActionRequest

DAY = 86400.0


def _lab_cookie_state(srv, handle, password):
    tok = srv.control("login", handle=handle, password=password)["token"]
    host = srv.url.split("//")[1].split(":")[0]
    return {"cookies": [{"name": "swl_session", "value": tok, "domain": host, "path": "/", "expires": -1,
                         "httpOnly": True, "secure": False, "sameSite": "Lax"}], "origins": []}, tok


def _pool(tmp_path, srv):
    from cryptography.fernet import Fernet

    from agentrs.harness.browser_pool import BrowserWorkerPool
    from agentrs.harness.net import EgressPolicy
    from agentrs.harness.object_store import LocalObjectStore

    host = srv.url.split("//")[1].split(":")[0]
    eg = EgressPolicy(allowed_domains=[host], allow_private=[host], allow_http_for=[host])
    key = Fernet.generate_key()
    return BrowserWorkerPool(LocalObjectStore(tmp_path / "objects"), key, eg, max_sessions=3), key


def test_4_browser_sessions_are_isolated_and_state_is_encrypted(sandbox, tmp_path):
    from agentrs.environments.base import TaskSpec
    from agentrs.environments.web.playwright_env import BrowserUnavailable

    s = sandbox
    for h in ("alice-bot", "bob-bot"):
        s.srv.control("create_account", handle=h, password=f"{h}-pw", spaces=["linux"], followers=0)
    pool, key = _pool(tmp_path, s.srv)
    from cryptography.fernet import Fernet

    tokens = {}
    for acc, h in (("acc-A", "alice-bot"), ("acc-B", "bob-bot")):
        st, tokens[acc] = _lab_cookie_state(s.srv, h, f"{h}-pw")
        pool.store.put(f"browser-state/{acc}.enc", Fernet(key).encrypt(json.dumps(st).encode()))
    try:
        sa = pool.open("acc-A", "agent1", "t1", s.srv.url + "/notifications", [s.srv.url + "/"])
        sb = pool.open("acc-B", "agent2", "t2", s.srv.url + "/notifications", [s.srv.url + "/"])
        for sess in (sa, sb):
            sess.env.reset(TaskSpec(task_id="x", env_id="web", family="web", goal="", max_steps=50,
                                    private={"start_url": s.srv.url + "/notifications"}))
    except BrowserUnavailable as e:
        pytest.skip(f"Chromium cannot be launched: {e}")
    try:
        ta = sa.env.observe().to_text(200)
        tb = sb.env.observe().to_text(200)
        assert "alice-bot" in ta and "bob-bot" not in ta
        assert "bob-bot" in tb and "alice-bot" not in tb
        ca = {c["value"] for c in sa.env.storage_state()["cookies"]}
        cb = {c["value"] for c in sb.env.storage_state()["cookies"]}
        assert tokens["acc-A"] in ca and tokens["acc-A"] not in cb and not (ca & cb)  # no cross-account cookies
        pool.save_state(sa.id)
        blob = pool.store.get("browser-state/acc-A.enc")
        assert tokens["acc-A"].encode() not in blob  # encrypted at rest
        # requests outside the allowlist / to private addresses are blocked inside the page
        assert not sa.env._request_allowed("http://169.254.169.254/latest/meta-data/")
        assert not sa.env._request_allowed("https://example.com/")
        assert pool.revoke_account("acc-A") == 1 and not pool.store.exists("browser-state/acc-A.enc")
        with pytest.raises(PermissionError):
            pool.open("acc-A", "agent1", "t3", s.srv.url, [s.srv.url + "/"])
    finally:
        pool.close_all()


def test_5_real_browser_execution_through_the_gateway(sandbox, tmp_path):
    from cryptography.fernet import Fernet

    from agentrs.environments.web.playwright_env import BrowserUnavailable
    from agentrs.harness.connectors.generic_web import GenericWebConnector
    from agentrs.harness.identity import AgentSpec

    s = sandbox
    h = s.h
    s.srv.control("create_account", handle="web-bot", password="web-bot-pw", spaces=["linux"], followers=0)
    author = s.lab._q("SELECT id FROM users WHERE kind='simulated' AND role='regular' LIMIT 1")[0]["id"]
    iid = s.lab.create_item(author, "linux", "How do I fix ssh when cron hangs?", "How do I fix ssh? error E1234")
    pool, key = _pool(tmp_path, s.srv)
    st, tok = _lab_cookie_state(s.srv, "web-bot", "web-bot-pw")
    GenericWebConnector.pool = pool
    GenericWebConnector.site_profiles = {"lab": {"start_url": s.srv.url + "/", "allowed_prefixes": [s.srv.url + "/"],
                                                 "tier": 1}}
    web = h.registry.get("web")
    web.enabled, web.authorization_ref = True, "local test site"
    web.capabilities.update(navigate=True, read_page=True, click=True, fill=True, screenshot=True)
    web.limits.requests_per_minute = 600
    host = s.srv.url.split("//")[1].split(":")[0]
    web.allowed_domains = [host]
    h.agents.upsert(AgentSpec(id="browser_agent", display_name="b", role="b", max_tier=1,
                              allowed_capabilities=["web:navigate", "web:read_page", "web:click", "web:screenshot"]))
    aid = h.accounts.register("web", "web-bot@lab", "test", "sandbox", "operator", ["navigate", "read_page", "click",
                                                                                    "screenshot"], {}, ["browser_agent"],
                              "browser workflow tests", status="active")
    pool.store.put(f"browser-state/{aid}.enc", Fernet(key).encrypt(json.dumps(st).encode()))
    h.tasks.create(task_id="w1", source="human", goal="upvote the ssh question", environment="web",
                   required_capabilities=[], allowed_actions=["web:*"], verifier={"kind": "reacted"}, authorized_tier=1)
    h.tasks.set_state("w1", "running")

    def go(op, **params):
        return asyncio.run(h.gateway.submit(ActionRequest(agent_id="browser_agent", task_id="w1", provider="web",
                                                          operation=op, params={"site": "lab", **params})))
    try:
        r = go("navigate", url=f"{s.srv.url}/item/{iid}")
    except BrowserUnavailable as e:
        pytest.skip(f"Chromium cannot be launched: {e}")
    try:
        assert r.status == "executed", r.reason
        page = r.observation["data"]["page"]
        assert "How do I fix ssh" in page and "UNTRUSTED_EXTERNAL_CONTENT" in r.observation["untrusted"]
        up = next(line.split("]")[0].strip("[") for line in page.splitlines() if "Upvote" in line)
        r2 = go("click", element=up)
        assert r2.status == "executed", r2.reason
        # independent verification from the application's database, not from the page
        rows = s.lab._q("SELECT r.kind FROM reactions r JOIN users u ON u.id=r.user_id WHERE r.item_id=? AND u.handle=?",
                        (iid, "web-bot"))
        assert [x["kind"] for x in rows] == ["up"]
        shot = go("screenshot")
        assert shot.status == "executed" and shot.observation["data"]["screenshot_ref"].startswith("file://")
        # an unknown site is never routine; a navigation outside the allowlist is blocked
        assert go("click", element=up, site="elsewhere").status == "denied"
        off = go("navigate", url="https://example.com/")
        assert off.status == "failed" and s.lab.stats()["reactions"] >= 1  # blocked by the session allowlist
    finally:
        pool.close_all()


def test_email_transactional_external_approval_replies_and_bounces(sandbox):
    from email.message import EmailMessage

    from agentrs.harness.email_service import EmailConnector, LocalMailServer, SmtpImapBackend
    from agentrs.harness.identity import AgentSpec

    h = sandbox.h
    mail = LocalMailServer()
    try:
        cfg = h.registry.get("email")
        cfg.enabled, cfg.authorization_ref = True, "operator mailbox"
        cfg.capabilities.update(read_messages=True, send_transactional=True, send_external=True)
        cfg.limits.requests_per_minute = 60
        EmailConnector.db = h.db
        EmailConnector.internal_domains = ["agents.example.org"]
        EmailConnector.backend_factory = staticmethod(lambda ctx: SmtpImapBackend(
            mail.host, mail.port, None, ctx.account["external_id"], "", "", None, starttls=False, local=mail))
        h.agents.upsert(AgentSpec(id="mailer", display_name="m", role="m", max_tier=3,
                                  allowed_capabilities=["email:read_messages", "email:send_transactional",
                                                        "email:send_external"]))
        h.accounts.register("email", "agent@agents.example.org", "org_mailbox", "example org", "ops",
                            ["read_messages", "send_transactional", "send_external"], {}, ["mailer"], "agent mail",
                            status="active", bot_disclosure="sender_name")
        h.tasks.create(task_id="m1", source="human", goal="report", environment="email", required_capabilities=[],
                       allowed_actions=["email:*"], verifier={"kind": "manual"}, authorized_tier=3)
        h.tasks.set_state("m1", "running")
        send = lambda op, **p: asyncio.run(h.gateway.submit(ActionRequest(  # noqa: E731
            agent_id="mailer", task_id="m1", provider="email", operation=op, params=p)))
        r = send("send_transactional", to="ops@agents.example.org", subject="daily report", body="All tasks done.")
        assert r.status == "executed" and len(mail.inbox("ops@agents.example.org")) == 1
        assert send("send_transactional", to="someone@outside.example", subject="x", body="hello").status == "failed"
        r = send("send_external", to="person@outside.example", subject="Question", body="Could you review the draft?")
        assert r.status == "pending_approval" and not mail.inbox("person@outside.example")
        done = asyncio.run(h.gateway.execute_approved(r.approval_id, "operator@example.org"))
        assert done.status == "executed" and mail.inbox("person@outside.example")
        mid = done.provider_object_id
        reply = EmailMessage()
        reply["From"], reply["To"], reply["Subject"] = "person@outside.example", "agent@agents.example.org", "Re: Question"
        reply["In-Reply-To"], reply["Message-ID"] = mid, "<reply-1@outside.example>"
        reply.set_content("Sure, send it.")
        mail.deliver(reply, "agent@agents.example.org")
        ver = EmailMessage()
        ver["From"], ver["To"], ver["Subject"] = "no-reply@provider.example", "agent@agents.example.org", "Verify your email"
        ver["Message-ID"] = "<v-1@provider.example>"
        ver.set_content("Your verification code is 123456")
        mail.deliver(ver, "agent@agents.example.org")
        acc = h.accounts.find_for("mailer", "email", ["read_messages"])
        conn = h.gateway.connector_for(acc)
        out = asyncio.run(conn.fetch_outcome(mid))
        assert out["occurred"]["first_response"] and out["counts"]["responses"] == 1
        new = conn.ingest(0)
        assert not new or all(n["category"] != "external" or n["message_id"] != "<v-1@provider.example>" for n in new)
        from agentrs.harness.db import EmailMessage as Row

        with h.db.session() as db:
            v = db.query(Row).filter_by(message_id="<v-1@provider.example>").one()
            assert v.category == "notification"  # surfaced to the operator, never acted on automatically
        r = send("send_external", to="x@bounce.invalid", subject="hi", body="bounce test text")
        assert r.status == "pending_approval"
        bounced = asyncio.run(h.gateway.execute_approved(r.approval_id, "operator@example.org"))
        assert bounced.status == "failed"
        with h.db.session() as db:
            assert db.query(Row).filter_by(status="failed").count() >= 1
    finally:
        mail.stop()


def test_s3_object_store_against_moto_server():
    moto = pytest.importorskip("moto.server")
    srv = moto.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    srv.start()
    try:
        import boto3

        from agentrs.harness.object_store import S3ObjectStore

        host, port = srv.get_host_and_port()
        c = boto3.client("s3", endpoint_url=f"http://{host}:{port}", region_name="us-east-1", aws_access_key_id="test",
                         aws_secret_access_key="test")
        st = S3ObjectStore("agentrs-artifacts", client=c)
        ref = st.put("screenshots/t1/a.png", b"\x89PNG data", "image/png")
        assert ref == "s3://agentrs-artifacts/screenshots/t1/a.png" and st.get("screenshots/t1/a.png") == b"\x89PNG data"
        st.delete("screenshots/t1/a.png")
        assert not st.exists("screenshots/t1/a.png")
    finally:
        srv.stop()


def test_control_api_dashboard_and_operator_controls(sandbox):
    from fastapi.testclient import TestClient

    from agentrs.harness.control_api import create_control_app

    s = sandbox
    asyncio.run(s.h.provisioning.verify(s.h.provisioning.request("socialweblab", ["create_post"], "practice",
                                                                 "sandbox_agent_001").id))
    app = create_control_app(s.h, "op-token-123", extras={"inference": s.inf})
    c = TestClient(app)
    assert c.get("/health").json()["ok"]
    assert c.get("/api/agents").status_code == 401
    H = {"Authorization": "Bearer op-token-123"}
    assert c.get("/api/agents", headers=H).json()[0]["id"] == "sandbox_agent_001"
    accs = c.get("/api/accounts", headers=H).json()
    assert accs and "session_token" in accs[0]["credentials"] and "secret" not in json.dumps(accs).lower().replace(
        "credentials", "")
    r = c.post("/api/provisioning/x/connect", headers=H, json={"external_id": "e", "credential_refs": {"t": "raw-value"}})
    assert r.status_code == 400  # secret values are never accepted, only references
    assert c.post("/api/agents/sandbox_agent_001/pause", headers=H).json()["status"] == "paused"
    assert s.h.agents.get("sandbox_agent_001").status == "paused"
    c.post("/api/agents/sandbox_agent_001/resume", headers=H)
    assert c.post("/api/connectors/socialweblab/disable", headers=H).json()["enabled"] is False
    c.post("/api/connectors/socialweblab/enable", headers=H)
    assert c.post("/api/emergency_stop", headers=H, data={"active": "true", "reason": "drill"}).json()["emergency_stop"]
    assert s.h.policy.stopped()
    c.post("/api/emergency_stop", headers=H, data={"active": "false"})
    assert c.post(f"/api/accounts/{accs[0]['id']}/revoke", headers=H).json()["revoked"] == accs[0]["id"]
    assert "agentrs_actions_total" in c.get("/metrics", headers=H).text
    assert c.get("/dashboard", follow_redirects=False).status_code == 303
    c.post("/login", data={"token": "op-token-123"})
    page = c.get("/dashboard").text
    for section in ("Agents", "Approvals", "Accounts", "Connectors", "Learning", "Infrastructure", "Emergency stop"):
        assert section in page
    assert "op-token-123" not in page
    assert c.get("/api/learning", headers=H).json()["experiences"] == 0
    assert "jobs" in c.get("/api/infra", headers=H).json()
