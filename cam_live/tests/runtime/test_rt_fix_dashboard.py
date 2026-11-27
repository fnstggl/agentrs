"""D1 regression tests (G_rt.md): the dashboard renders learner- and environment-written content as text only, sends a
strict CSP, and operator actions need an operator credential distinct from the viewing token, CSRF protection, a
same-origin request and a peer that is not a service user."""
import json
import os
import pwd
import re
import socket
import stat
import subprocess
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from cam_live.contracts import (AuthorizationDecision, AuthVerdict, ExperimentRecord, LearningUpdateRecord,
                                MetaUpdateRecord, ObstacleRecord, Observation, OutcomeObservation, Provenance, RunFiles,
                                SimClock)
from cam_live.runtime import dashboard as D
from cam_live.runtime.logs import RunLog, atomic_write_json

TOKEN = "view-token-" + "v" * 24
OPER = "operator-credential-" + "o" * 24
X = lambda tag: f'<img src=x onerror="window.__xss=(window.__xss||[]).concat([\'{tag}\'])">'  # noqa: E731
APPROVER = ("<img src=x onerror=\"window.__xss=(window.__xss||[]).concat(['approver']);"
            "fetch('/api/approvals/apr_g1/approve',{method:'POST',headers:{'content-type':'application/json'},body:'{}'})\">")
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class Pol:
    """Operator handle on an authorization service stand-in: records every call that would change something."""

    def __init__(self):
        self.resolved, self.resets, self.stops = [], [], []

    def approvals(self):
        return [{"approval_id": "apr_g1", "status": "pending"}]

    def resolve_approval(self, approval_id, approve, by, note=""):
        self.resolved.append((approval_id, approve, by))
        return AuthorizationDecision(AuthVerdict.ALLOW if approve else AuthVerdict.DENY, "", ["ok"], grant="g",
                                     approval_id=approval_id)

    def reset_stop(self, by):
        self.resets.append(by)
        return True

    def emergency_stop(self, reason, by):
        self.stops.append((reason, by))

    def stopped(self):
        return bool(self.stops)


def hostile_run(run_dir: str) -> str:
    """A run directory whose every learner- or environment-written field carries markup (also fields that are
    numbers by contract), written with the runtime's own log writer."""
    clock = SimClock()
    log = RunLog(run_dir, clock, provenance="diagnostic", durable=False)
    log.meta_update(MetaUpdateRecord(meta_id="m1", ts=0.0, reward_version_before="a", reward_version_after=X("meta_version"),
                                     applied=False, skipped_reason=X("meta_reason"), estimator=X("meta_estimator"),
                                     bootstrap_mix=APPROVER, meta_grad_norm=X("meta_grad_norm")))
    log.learning_update(LearningUpdateRecord(update_id="u1", ts=0.0, trigger=X("lu_trigger"), timescale=X("lu_timescale"),
                                             param_groups=[X("lu_group")], weight_delta={X("lu_wd"): 1.0},
                                             prediction_change={"p": 0.1}, losses={X("lu_loss"): 1.0}, notes=X("lu_notes")))
    log.experiment(ExperimentRecord(experiment_id="e1", created_at=0.0, question=X("exp_question"),
                                    hypothesis=X("exp_hypothesis"), status=X("exp_status"), supported=X("exp_supported"),
                                    actual=[OutcomeObservation("q_never_issued", "observed", 999.0, verified=True,
                                                               source="forged-by-learner")]))
    log.experiment_outcome("e1", "dec1", {"query_id": "q1", "status": "observed", "value": 3.0, "verified": True,
                                          "outcome_rid": "ev-0000000009"})
    log.event("internal_rewards", {"r": 1.0, X("reward_key"): 2.0})
    log.observation(Observation(obs_id="o1", ts=0.0, kind="page", text=X("observation_text"),
                                provenance=Provenance(source="env")))
    log.obstacle(ObstacleRecord(obstacle_id="ob1", ts=0.0, action_id="a1", failure_class=X("failure_class"),
                                description=X("obstacle_description"), category=X("category"), outcome=X("outcome"),
                                learning_update_ids=X("ids_not_a_list")))
    log.close()
    state = {"version": 1, "run_id": X("run_id"), "status": "running", "status_reason": X("status_reason"),
             "task": {"objective_id": X("objective_id"), "objective": X("objective"), "mode": X("mode")},
             "activity": {"phase": X("phase"), "detail": X("detail")}, "wake": {"reason": X("wake")},
             "limits": {"money": {"used": X("used"), "limit": 5.0}, X("limit_name"): {"used": 1.0, "limit": X("limit")}},
             "approvals": {"apr_g1": {"approval_id": "apr_g1", "status": "pending", "operation": X("operation"),
                                      "preview": X("approval_preview"), "reasons": [X("approval_reason")]},
                           "apr_old": {"approval_id": X("old_id"), "status": X("old_status"), "operation": X("old_op")}},
             "errors": [{"error": X("error_text")}],
             "last_verified": {"rid": "ev-1", "outcome": {"status": X("v_status"), "value": X("v_value"), "notes": X("v_notes"),
                                                          "secondary": [{"name": X("secondary"), "value": 1}],
                                                          "excluded": {X("excluded"): 1}}},
             "next_tasks": [{"task": {"objective_id": X("queued")}}],
             "committed": {"ckpt": {"id": X("ckpt"), "generation": X("generation")}},
             "live": {"pending": {"items": []}}}
    atomic_write_json(os.path.join(run_dir, RunFiles.STATE), state)
    with open(os.path.join(run_dir, RunFiles.JOURNAL), "w") as f:
        f.write("# journal\n" + X("journal") + "\n")
    return run_dir


@pytest.fixture
def hostile(tmp_path):
    return hostile_run(str(tmp_path / "run"))


def operator_client(app):
    """A test client that loaded the page (cookies: viewing token + CSRF) and knows the operator credential."""
    c = TestClient(app)
    assert c.get("/", params={"token": TOKEN}).status_code == 200
    return c, {D.OPERATOR_HEADER: OPER, D.CSRF_HEADER: c.cookies[D.CSRF_COOKIE]}


def routes(app):
    return [(m, r.path) for r in app.routes for m in getattr(r, "methods", []) if m in ("GET", "POST")]


def concrete(path):
    return path.replace("{name}", "events.jsonl").replace("{approval_id}", "apr_g1").replace("{command_id}", "x")


ACTIONS = ["/api/control/pause", "/api/control/resume", "/api/control/stop", "/api/control/reset-stop",
           "/api/control/export-checkpoint", "/api/control/assign-objective", "/api/approvals/apr_g1/approve",
           "/api/approvals/apr_g1/reject"]


# ------------------------------------------------------------------------------------------------- headers / page
def test_strict_csp_and_security_headers_on_every_response(hostile):
    app = D.create_app(hostile, TOKEN, policy=Pol(), operator_token=OPER)
    c = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    anon = TestClient(app)
    seen = 0
    for m, path in routes(app):
        for client in (c, anon):
            r = client.request(m, concrete(path), json={})
            csp = r.headers.get("content-security-policy", "")
            assert "script-src 'self'" in csp and "default-src 'none'" in csp and "'unsafe-inline'" not in csp, path
            assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp and "base-uri 'none'" in csp
            assert r.headers.get("x-content-type-options") == "nosniff" and r.headers.get("x-frame-options") == "DENY"
            assert r.headers.get("referrer-policy") == "no-referrer" and r.headers.get("cache-control") == "no-store"
            seen += 1
    assert seen >= 2 * 25
    assert c.get("/app.js").headers["content-type"].startswith("text/javascript")
    assert c.get("/app.css").headers["content-type"].startswith("text/css")
    assert c.get("/api/journal").headers["content-type"].startswith("text/plain")
    assert c.get("/api/status").headers["content-type"].startswith("application/json")


def test_page_has_no_inline_script_or_style_and_the_client_never_parses_markup(hostile):
    app = D.create_app(hostile, TOKEN)
    c = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    html = c.get("/").text
    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.S)
    assert scripts == [(' src="/app.js" defer', "")]                       # one same-origin script, no inline code
    assert "<style" not in html and not re.search(r"\sstyle\s*=", html)
    assert not re.search(r"\son[a-z]+\s*=", html, re.I)                      # no event-handler attributes
    assert "http://" not in html and "https://" not in html
    js = c.get("/app.js").text
    assert js == D.APP_JS
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
                 "setAttribute('on", "srcdoc"):
        assert sink not in js, sink
    assert "textContent" in js and "history.replaceState" in js
    assert c.get("/app.css").text == D.APP_CSS and "url(" not in D.APP_CSS


def test_experiments_show_only_runtime_measured_outcomes(hostile):
    c = TestClient(D.create_app(hostile, TOKEN), headers={"Authorization": f"Bearer {TOKEN}"})
    (row,) = c.get("/api/experiments").json()
    assert row["actual"] == [{"record": row["actual"][0]["record"], "decision_id": "dec1", "value": 3.0,
                              "status": "observed", "verified": True, "outcome_record": "ev-0000000009"}]
    assert 999.0 not in [a["value"] for a in row["actual"]]                 # the learner's own 'actual' is not shown
    assert "supported" not in row and row["supported_according_to_learner"] == X("exp_supported")


# ------------------------------------------------------------------------------------------------- operator actions
def test_operator_actions_need_operator_credential_csrf_and_same_origin(hostile):
    pol = Pol()
    app = D.create_app(hostile, TOKEN, policy=pol, operator_token=OPER)
    control = os.path.join(hostile, RunFiles.CONTROL)
    c, good = operator_client(app)
    viewer = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    # the operator credential is never taken from the URL or from a cookie
    via_url_or_cookie = TestClient(app, cookies={D.COOKIE: TOKEN, D.CSRF_COOKIE: good[D.CSRF_HEADER],
                                                 "cam_live_operator": OPER, D.OPERATOR_HEADER: OPER})
    for path in ACTIONS:
        body = {"task": "lab_staging.yaml"} if path.endswith("assign-objective") else {}
        assert viewer.post(path, json=body).status_code == 401, path                     # viewing token only
        assert c.post(path, json=body, headers={**good, D.OPERATOR_HEADER: OPER[:-1] + "x"}).status_code == 401, path
        assert c.post(path, json=body, headers={D.CSRF_HEADER: good[D.CSRF_HEADER]}).status_code == 401, path
        assert via_url_or_cookie.post(path, json=body, params={"operator": OPER, "operator_token": OPER},
                                      headers={D.CSRF_HEADER: good[D.CSRF_HEADER]}).status_code == 401, path
        assert c.post(path, json=body, headers={D.OPERATOR_HEADER: OPER}).status_code == 403, path       # no CSRF
        assert c.post(path, json=body, headers={**good, D.CSRF_HEADER: "forged-" + "x" * 30}).status_code == 403, path
        assert c.post(path, json=body, headers={**good, "Origin": "http://evil.example"}).status_code == 403, path
        assert c.post(path, json=body, headers={**good, "Origin": "null"}).status_code == 403, path
        assert c.post(path, json=body, headers={**good, "Sec-Fetch-Site": "cross-site"}).status_code == 403, path
    assert pol.resolved == [] and pol.resets == [] and pol.stops == [] and not os.path.exists(control)
    # with all three, the action goes through (same-origin headers as a browser sends them)
    r = c.post("/api/approvals/apr_g1/approve", json={"note": "ok"},
               headers={**good, "Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 200 and pol.resolved == [("apr_g1", True, "operator")]
    assert c.post("/api/control/reset-stop", headers=good).json() == {"reset": True, "by": "operator"}
    assert c.post("/api/control/stop", json={"emergency": True}, headers=good).status_code == 200 and pol.stops
    assert c.post("/api/control/pause", json={}, headers=good).status_code == 200 and os.path.exists(control)
    assert c.post("/api/approvals/nope/approve", json={}, headers=good).status_code == 404


def test_without_an_operator_credential_the_dashboard_is_view_only(hostile):
    pol = Pol()
    app = D.create_app(hostile, TOKEN, policy=pol)
    c = TestClient(app)
    assert c.get("/", params={"token": TOKEN}).status_code == 200 and c.get("/api/status").status_code == 200
    h = {D.OPERATOR_HEADER: OPER, D.CSRF_HEADER: c.cookies[D.CSRF_COOKIE]}
    for path in ACTIONS:
        r = c.post(path, json={"task": "lab_staging.yaml"} if path.endswith("assign-objective") else {}, headers=h)
        assert r.status_code == 403 and "no operator credential" in r.json()["detail"], path
    assert c.get("/api/operator").json()["actions_enabled"] is False
    assert pol.resolved == [] and not os.path.exists(os.path.join(hostile, RunFiles.CONTROL))
    with pytest.raises(ValueError, match="distinct"):
        D.create_app(hostile, TOKEN, operator_token=TOKEN)
    # reset without a handle on the authorization service: refused with an explanation
    c2, good = operator_client(D.create_app(hostile, TOKEN, operator_token=OPER))
    r = c2.post("/api/control/reset-stop", headers=good)
    assert r.status_code == 409 and "authorization service" in r.json()["detail"]


def test_operator_credential_file(tmp_path):
    home = str(tmp_path / "home")
    view = D.load_or_create_token(home)
    op = D.load_or_create_operator_token(home)
    p = os.path.join(home, D.OPERATOR_TOKEN_FILE)
    assert op != view and len(op) >= 32 and D.load_or_create_operator_token(home) == op
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    os.chmod(p, 0o640)
    with pytest.raises(PermissionError, match="0600"):
        D.load_or_create_operator_token(home)
    os.chmod(p, 0o600)
    if os.geteuid() == 0:
        os.chown(p, 65534, -1)
        with pytest.raises(PermissionError, match="owned by uid 65534"):
            D.load_or_create_operator_token(home)
    home2 = str(tmp_path / "home2")
    os.makedirs(home2)
    D.load_or_create_token(home2)
    os.symlink(os.path.join(home2, D.TOKEN_FILE), os.path.join(home2, D.OPERATOR_TOKEN_FILE))
    with pytest.raises(PermissionError):
        D.load_or_create_operator_token(home2)                             # a link to the viewing token is refused


# ------------------------------------------------------------------------------------------------- peer uid
def test_peer_uid_of_a_real_loopback_connection():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    cli = socket.create_connection(srv.getsockname())
    acc, _ = srv.accept()
    try:
        client, server = cli.getsockname(), srv.getsockname()
        assert D.socket_owner_uid(client[0], client[1], server[0], server[1]) == os.geteuid()
        assert D.peer_problem(client, server, refused_uids={}) == ""
        why = D.peer_problem(client, server, refused_uids={os.geteuid(): "camlearn"})
        assert "service user camlearn" in why
        assert "could not be determined" in D.peer_problem(("127.0.0.1", 1), server, refused_uids={})
        assert "not a network address" in D.peer_problem(("testclient", 0), server, refused_uids={})   # spoofed peer
        assert D.peer_problem(("testclient", 50000), ("testserver", 80), refused_uids={}) == ""        # no socket
    finally:
        for s in (cli, acc, srv):
            s.close()


def test_peer_uid_parses_ipv4_and_ipv6_tables(tmp_path):
    t4, t6 = tmp_path / "tcp", tmp_path / "tcp6"
    hdr = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    # 127.0.0.1:40000 -> 127.0.0.1:8787 owned by uid 1234 (little-endian word, as the kernel prints it)
    t4.write_text(hdr + "   0: 0100007F:9C40 0100007F:2253 01 00000000:00000000 00:00000000 00000000  1234        0 1\n")
    # [::1]:40001 -> [::1]:8787 owned by uid 4321
    t6.write_text(hdr + "   0: 00000000000000000000000001000000:9C41 00000000000000000000000001000000:2253 01 "
                        "00000000:00000000 00:00000000 00000000  4321        0 2\n")
    tables = (str(t4), str(t6))
    if sys.byteorder == "little":
        assert D.socket_owner_uid("127.0.0.1", 40000, "127.0.0.1", 8787, tables) == 1234
        assert D.socket_owner_uid("::1", 40001, "::1", 8787, tables) == 4321
        assert "service user camlearn" in D.peer_problem(("::1", 40001), ("::1", 8787), {4321: "camlearn"}, tables)
    assert D.socket_owner_uid("127.0.0.1", 40002, "127.0.0.1", 8787, tables) is None


def _serve(app):
    import uvicorn
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", proxy_headers=False))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    return server, th, port


@pytest.mark.privileged
def test_a_learner_uid_process_cannot_approve_even_with_every_credential(hostile):
    from cam_live.harness import SERVICE_USERS, users_exist
    if os.geteuid() != 0 or not users_exist():
        pytest.skip("needs root and the service users")
    pol = Pol()
    server, th, port = _serve(D.create_app(hostile, TOKEN, policy=pol, operator_token=OPER))
    code = (
        "import json, sys, urllib.request\n"
        "port, tok, op = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "req = urllib.request.Request(f'http://127.0.0.1:{port}/api/approvals/apr_g1/approve', data=b'{}', method='POST',"
        " headers={'Authorization': 'Bearer ' + tok, 'X-CAM-Operator': op, 'X-CSRF-Token': 'c' * 40,"
        " 'Cookie': 'cam_live_csrf=' + 'c' * 40, 'content-type': 'application/json'})\n"
        "try:\n"
        "    r = urllib.request.urlopen(req, timeout=10); print(r.status, r.read().decode())\n"
        "except urllib.error.HTTPError as e:\n"
        "    print(e.code, e.read().decode())\n")
    try:
        pw = pwd.getpwnam(SERVICE_USERS["learner"])

        def drop():
            os.setgroups([])
            os.setgid(pw.pw_gid)
            os.setuid(pw.pw_uid)
        p = subprocess.run([sys.executable, "-c", code, str(port), TOKEN, OPER], capture_output=True, text=True,
                           timeout=60, preexec_fn=drop, env={"PATH": os.environ.get("PATH", "")}, cwd="/")
        status, _, body = p.stdout.strip().partition(" ")
        assert status == "403" and "service user camlearn" in body, p.stdout + p.stderr
        assert pol.resolved == []
        # the same request from the operator's own uid is accepted
        p = subprocess.run([sys.executable, "-c", code, str(port), TOKEN, OPER], capture_output=True, text=True, timeout=60)
        assert p.stdout.startswith("200"), p.stdout + p.stderr
        assert pol.resolved == [("apr_g1", True, "operator")]
    finally:
        server.should_exit = True
        th.join(timeout=10)


def _poll(pg, expr, timeout_s=15.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if pg.evaluate(expr):
            return True
        pg.wait_for_timeout(100)
    return False


@pytest.mark.browser
def test_hostile_run_content_never_executes_in_the_operators_browser(hostile):
    """p08's attack: markup in learner-written records (also in number fields). Rendered in Chromium, nothing
    executes and nothing reaches the authorization service; the operator's own approval still works."""
    from playwright.sync_api import sync_playwright
    pol = Pol()
    server, th, port = _serve(D.create_app(hostile, TOKEN, policy=pol, operator_token=OPER))
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(args=["--no-sandbox"])
            pg = b.new_page()
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(f"http://127.0.0.1:{port}/?token={TOKEN}")
            # (the strict CSP also forbids evaluating strings, which Playwright's wait_for_function needs: poll instead)
            assert _poll(pg, "document.getElementById('meta').textContent.length > 0")
            for name in ("decisions.jsonl", "experiments.jsonl", "learning_updates.jsonl", "meta_updates.jsonl",
                         "obstacles.jsonl", "journal"):
                pg.select_option("#log-name", name)
                pg.wait_for_timeout(300)
            fired = pg.evaluate("window.__xss || []")
            imgs = pg.evaluate("document.querySelectorAll('img').length")
            status_text = pg.evaluate("document.getElementById('st').textContent")
            meta_text = pg.evaluate("document.getElementById('meta').textContent")
            url = pg.url
            assert fired == [] and imgs == 0 and not errors, (fired, imgs, errors)
            assert not status_text.startswith("error:") and "onerror" in meta_text      # shown as text, not run
            assert pol.resolved == [] and "token=" not in url
            # the operator's approval through the page itself: credential typed in, CSRF cookie echoed
            pg.fill("#op-cred", OPER)
            pg.dispatch_event("#op-cred", "change")
            pg.click("#approvals button:has-text('Approve')")
            assert _poll(pg, "document.getElementById('op-out').textContent.startsWith('accepted')"), \
                pg.evaluate("document.getElementById('op-out').textContent")
            b.close()
        assert pol.resolved == [("apr_g1", True, "operator")]
    finally:
        server.should_exit = True
        th.join(timeout=10)
