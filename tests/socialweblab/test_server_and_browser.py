"""SocialWebLab web front end: socket-free dispatch, HTTP server, and the real-browser env.

Browser tests skip (rather than fail) when Chromium cannot be launched.
"""

from __future__ import annotations

import json
import re
import urllib.request

import pytest

from agentrs.actions.schema import Action
from agentrs.socialweblab.engine import SocialWebLab
from agentrs.socialweblab.server import (
    HttpTransport,
    InProcessTransport,
    LabApp,
    LabClient,
    LabHTTPError,
    SocialWebLabServer,
)


@pytest.fixture
def lab(tmp_path):
    lab = SocialWebLab(tmp_path / "lab.db", seed=3, n_users=20)
    lab.advance(4 * 3600)
    lab.create_account("alice", "pw-alice", spaces=["python-help", "linux"])
    lab.create_account("bob", "pw-bob", spaces=["python-help"])
    yield lab
    lab.close()


def _ctl(app: LabApp, op: str, **payload):
    status, _, body = app.dispatch("POST", f"/__control/{app.control_token}/{op}", {}, {},
                                   json.dumps(payload).encode())
    assert status == 200, body
    return json.loads(body)


def _user(app, handle, pw) -> LabClient:
    c = LabClient(InProcessTransport(app))
    c.login(handle, pw)
    return c


# =========================================================================== dispatch level (no browser)
def test_inprocess_login_create_respond_react(lab):
    app = LabApp(lab)
    alice, bob = _user(app, "alice", "pw-alice"), _user(app, "bob", "pw-bob")
    iid = alice.create_item("python-help", "How do I pin numpy in a venv?", "Traceback: ImportError, version 1.2.3")
    assert alice.create_item("python-help", "dup", "dup", client_ref="r1") == \
        alice.create_item("python-help", "dup", "dup", client_ref="r1")  # idempotent retry
    rid = bob.respond(iid, "Solution: reinstall numpy inside the venv.", kind="answer")
    assert bob.react(iid, "up") is True and bob.react(iid, "up") is False  # one reaction per user
    it = alice.item(iid)
    assert it["author"] == "alice" and it["counts"]["answers"] == 1 and it["counts"]["reactions"] == 1
    assert [r["id"] for r in it["responses"]] == [rid]
    alice.accept(iid, rid)
    assert bob.item(iid)["status"] == "resolved"
    n = alice.notifications()
    assert n["unread"] >= 1 and any(x["kind"] == "response" and x["item_id"] == iid for x in n["notifications"])
    assert alice.mark_read() >= 1 and alice.notifications()["unread"] == 0
    types = [e["type"] for e in alice.events() if e["item_id"] == iid]
    assert types[:1] == ["item_created"] and {"response_created", "reaction_up", "item_resolved"} <= set(types)
    assert any(x["id"] == iid for x in alice.search("pin numpy"))
    assert {"name": "python-help", "app": "forum"} in alice.spaces()
    assert alice.user("alice")["items"] == 2


def test_auth_refusals(lab):
    app = LabApp(lab)
    alice, bob = _user(app, "alice", "pw-alice"), _user(app, "bob", "pw-bob")
    iid = alice.create_item("python-help", "Alice's question", "How do I fix pip? version 2.0")
    anon = InProcessTransport(app)
    assert anon.get("/api/events")[0] == 401
    assert anon.post("/api/items", {"space": "linux", "title": "t", "body": "b"})[0] == 401
    assert InProcessTransport(app, session_token="forged").get("/api/feed")[0] == 401
    assert LabClient(InProcessTransport(app)).transport.post("/api/login", {"handle": "alice", "password": "x"})[0] == 401
    for call in (lambda: bob.edit(iid, "hijacked"), lambda: bob.delete(iid), lambda: alice.react(iid, "up")):
        with pytest.raises(LabHTTPError) as ei:
            call()
        assert ei.value.status == 403
    with pytest.raises(LabHTTPError) as ei:
        bob.respond(iid, "CI passed", kind="ci")  # internal response kinds are not for users
    assert ei.value.status == 400
    with pytest.raises(LabHTTPError) as ei:
        bob.item(10_000)
    assert ei.value.status == 404
    assert lab.items("id=?", (iid,))[0]["body"].startswith("How do I fix pip")
    # cookie-authenticated JSON POSTs need a JSON content type (no cross-site form posts)
    tok = alice.transport.session_token
    st, _, body = app.dispatch("POST", "/api/items", {}, {"Cookie": f"swl_session={tok}",
                                                          "Content-Type": "text/plain"}, b'{"space":"linux"}')
    assert st == 403 and b"cross-site" in body
    # HTML pages redirect anonymous visitors to the login form
    st, hd, _ = app.dispatch("GET", f"/item/{iid}", {}, {}, None)
    assert st == 303 and hd["Location"].startswith("/login?next=")


def test_observer_key_reads_events_but_cannot_post(lab):
    app = LabApp(lab)
    alice = _user(app, "alice", "pw-alice")
    iid = alice.create_item("linux", "Observer test post", "ssh keeps timing out, version 9.1")
    key = _ctl(app, "issue_api_key", source_id="observer-1", requests_per_minute=100)["api_key"]
    obs = LabClient(InProcessTransport(app, api_key=key))
    evs = obs.events(since=0, limit=1000)
    assert any(e["type"] == "item_created" and e["item_id"] == iid for e in evs)
    views = lab.items("id=?", (iid,))[0]["views"]
    assert obs.item(iid)["title"] == "Observer test post"
    assert lab.items("id=?", (iid,))[0]["views"] == views  # observer reads are not views
    assert obs.feed(space="linux", sort="new")[0]["id"] == iid
    for call in (lambda: obs.create_item("linux", "x", "y"), lambda: obs.react(iid, "up"),
                 lambda: obs.respond(iid, "hi"), lambda: obs.notifications()):
        with pytest.raises(LabHTTPError) as ei:
            call()
        assert ei.value.status == 403
    assert _ctl(app, "revoke_api_key", api_key=key) == {"revoked": True}
    assert obs.transport.get("/api/events")[0] == 401


def test_rate_limit_uses_lab_clock(lab):
    app = LabApp(lab, session_requests_per_minute=2)
    key = _ctl(app, "issue_api_key", source_id="obs", requests_per_minute=3)["api_key"]
    t = InProcessTransport(app, api_key=key)
    assert [t.get("/api/events", limit=1)[0] for _ in range(3)] == [200, 200, 200]
    st, hd, body = app.dispatch("GET", "/api/events", {}, {"X-Api-Key": key}, None)
    assert st == 429 and int(hd["Retry-After"]) >= 1 and json.loads(body)["retry_after"] >= 1
    assert t.get("/api/events")[0] == 429  # real time passing does not help: the bucket runs on lab time
    lab.advance(60)
    assert t.get("/api/events")[0] == 200
    # sessions get their own bucket (default 120/min; 2/min here)
    alice = _user(app, "alice", "pw-alice")
    assert [alice.transport.get("/api/feed")[0] for _ in range(3)] == [200, 200, 429]
    lab.advance(30)
    assert alice.transport.get("/api/feed")[0] == 200


def test_control_channel(lab, tmp_path):
    app = LabApp(lab)
    for path in (f"/__control/{'x' * 32}/stats", "/__control/stats", f"/__control/{app.control_token[:-1]}/now"):
        st, hd, body = app.dispatch("POST", path, {}, {}, b"{}")
        assert st == 404 and hd["Content-Type"].startswith("text/html") and b"error" not in body
    assert app.dispatch("GET", f"/__control/{app.control_token}/now", {}, {}, None)[0] == 404
    t0 = _ctl(app, "now")["now"]
    assert _ctl(app, "advance", seconds=120)["now"] == pytest.approx(t0 + 120)
    uid = _ctl(app, "create_account", handle="carol", password="pw-c", spaces=["linux"], followers=3)["uid"]
    assert lab.followers(uid) == 3 and lab.user_for_token(_ctl(app, "login", handle="carol", password="pw-c")["token"]) == uid
    carol = _user(app, "carol", "pw-c")
    iid = carol.create_item("linux", "How do I debug cron?", "cron job fails with error E1234, version 1.0")
    assert "features" in _ctl(app, "item_truth", item_id=iid)
    assert _ctl(app, "verify_help", item_id=iid, within_s=3600)["success"] is False
    snap = _ctl(app, "snapshot", path=str(tmp_path / "snap.db"))["path"]
    carol.create_item("linux", "after snapshot", "will be rolled back")
    n_items = _ctl(app, "stats")["items"]
    _ctl(app, "restore", path=snap)
    assert _ctl(app, "stats")["items"] == n_items - 1
    st, _, body = app.dispatch("POST", f"/__control/{app.control_token}/nope", {}, {}, b"{}")
    assert st == 404 and b"unknown control op" in body
    assert app.control_token not in app.dispatch("GET", "/login", {}, {}, None)[2].decode()


def test_http_transport_matches_in_process(lab):
    with SocialWebLabServer(lab) as server:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(f"{server.url}/__control/{server.control_token}/issue_api_key",
                                     data=b'{"source_id": "obs", "requests_per_minute": 100}', method="POST")
        key = json.loads(opener.open(req, timeout=10).read())["api_key"]
        http_alice = LabClient(HttpTransport(server.url))
        http_alice.login("alice", "pw-alice")
        iid = http_alice.create_item("linux", "Over HTTP", "systemd unit fails, version 2.3")
        http, inproc = LabClient(HttpTransport(server.url, api_key=key)), LabClient(InProcessTransport(server, api_key=key))
        assert http.item(iid) == inproc.item(iid)
        assert http.events(0, 500) == inproc.events(0, 500)
        assert http.feed(space="linux") == inproc.feed(space="linux")
        assert http.search("Over HTTP") == inproc.search("Over HTTP")
        assert HttpTransport(server.url).get("/api/events") == InProcessTransport(server).get("/api/events")
        bad = urllib.request.Request(f"{server.url}/__control/wrong/stats", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as ei:
            opener.open(bad, timeout=10)
        assert ei.value.code == 404


# =========================================================================== real browser
@pytest.fixture(scope="module")
def blab(tmp_path_factory):
    lab = SocialWebLab(tmp_path_factory.mktemp("swl") / "lab.db", seed=5, n_users=20)
    lab.advance(6 * 3600)
    yield lab
    lab.close()


@pytest.fixture(scope="module")
def benv(blab):
    pytest.importorskip("playwright.sync_api")
    from agentrs.socialweblab.browser_env import SocialWebLabBrowserEnv

    env = SocialWebLabBrowserEnv(SocialWebLabServer(blab))
    try:
        env.start()
    except Exception as exc:  # noqa: BLE001 - missing browser / system libraries
        env.close()
        pytest.skip(f"Chromium cannot be launched: {exc}")
    yield env
    env.close()


def _no_control_exposure(env, run_obs):
    assert all("/__control/" not in o.url for o in run_obs)
    assert not any("/__control/" in u for u in env.blocked_requests)  # the agent never even tried
    assert env.server.control_token not in env.run_in_browser(lambda p: p.content())


@pytest.mark.browser
def test_teacher_upvote_post_via_async_search(benv, blab):
    from agentrs.socialweblab.tasks import make_task, quoted, run_teacher

    task = make_task(blab, "upvote_post", "train", 0)
    title, iid = quoted(task.goal)[0], task.private["target_item"]
    run = run_teacher(benv, task)
    assert run.success, [(a.to_str(), r.error) for a, r in zip(run.actions, run.results)]
    assert run.verification.verifier_id == "socialweblab/reacted/v1"
    row = blab.items("id=?", (iid,))[0]
    assert row["ups"] == 1 and blab.all_events(iid)[-1]["type"] == "reaction_up"  # lab state, not the page
    assert run.observations[0].url == "/" and not any(o.url.startswith("/login") for o in run.observations)
    # async search: results show up only once loading has finished
    on_search = [o for o in run.observations if o.url.startswith("/search?q=")]
    loading = [any(e.role == "status" and "Loading" in e.name for e in o.elements) for o in on_search]
    has_result = [any(e.role == "link" and e.name == title for e in o.elements) for o in on_search]
    assert loading[0] and not has_result[0]  # first look at the results page: still loading
    assert any(has_result) and not any(lo and hr for lo, hr in zip(loading, has_result))
    assert any(a.type.value == "wait" for a in run.actions)
    _no_control_exposure(benv, run.observations)


@pytest.mark.browser
def test_teacher_create_post_is_persistent(benv, blab):
    from agentrs.socialweblab.tasks import make_task, run_teacher

    task = make_task(blab, "create_post", "train", 0)
    run = run_teacher(benv, task)
    assert run.success, [(a.to_str(), r.error) for a, r in zip(run.actions, run.results)]
    iid = run.verification.evidence["item_id"]
    assert benv.last_created_item == iid and run.observations[-1].url == f"/item/{iid}"
    assert any(o.url == "/new" and any(e.role == "status" and "Publishing" in e.name for e in o.elements)
               for o in run.observations)  # the async publish state was observable
    _no_control_exposure(benv, run.observations)
    other = SocialWebLab(blab.path)  # a new process-level view of the same database file
    try:
        row = other.items("id=?", (iid,))
        assert row and other.handle(row[0]["author"]) == task.private["account"]
        assert row[0]["space"] == task.private["space"]
        assert all(w in (row[0]["title"] + row[0]["body"]).lower() for w in task.private["verifier"]["must_contain"])
    finally:
        other.close()


@pytest.mark.browser
@pytest.mark.parametrize("family", ["reply_to_post", "edit_own_post"])
def test_teacher_content_families(benv, blab, family):
    from agentrs.socialweblab.tasks import make_task, run_teacher

    run = run_teacher(benv, make_task(blab, family, "val", 0))
    assert run.success, [(a.to_str(), r.error) for a, r in zip(run.actions, run.results)]


@pytest.mark.browser
def test_submit_macros_and_help_within(benv, blab):
    from agentrs.socialweblab.tasks import make_task

    task = make_task(blab, "create_post", "test", 1)
    k1, k2 = task.private["verifier"]["must_contain"]
    benv.reset(task)
    content = f"Why does {k1} break my {k2}?\nWhen I combine {k1} and {k2} I get error E4242 on version 1.4.2. Thanks!"
    r = benv.step(Action.submit("create_post", content))
    assert r.action_valid, r.error
    iid = r.info["created_item_id"]
    assert benv.last_created_item == iid and r.observation.url == f"/item/{iid}"
    row = blab.items("id=?", (iid,))[0]
    assert (row["title"], row["body"], row["space"]) == (content.splitlines()[0], content, task.private["space"])
    assert benv.verify_outcome().success
    r = benv.step(Action.submit("post_reply", f"Self-answer: pinning {k1} fixed it."))
    assert r.action_valid and r.info["item_id"] == iid and r.observation.url == f"/item/{iid}"
    resp = blab.public_item(iid)["responses"]
    assert [x["id"] for x in resp] == [r.info["created_response_id"]] and resp[0]["kind"] == "answer"
    assert resp[0]["body"].startswith("Self-answer") and resp[0]["by_author"]
    # explicit space line; refused operations are observable errors
    r = benv.step(Action.submit("create_post", "space: linux\nHow do I read journald logs?\nsystemd question, v2.1"))
    assert r.action_valid and blab.items("id=?", (r.info["created_item_id"],))[0]["space"] == "linux"
    r = benv.step(Action.submit("create_post", "space: no-such-space\nTitle\nBody"))
    assert not r.action_valid and "not available" in r.error
    r = benv.step(Action.submit("send_email", "hi"))
    assert not r.action_valid and "unsupported content operation" in r.error
    # the control channel is unreachable from the agent
    r = benv.step(Action.navigate(f"/__control/{benv.server.control_token}/stats"))
    assert not r.action_valid and "not permitted" in r.error
    fetched = benv.run_in_browser(lambda p: p.evaluate(
        "async (u) => { try { await fetch(u, {method: 'POST', body: '{}'}); return 'reached'; }"
        " catch (e) { return 'blocked'; } }", f"/__control/{benv.server.control_token}/stats"))
    assert fetched == "blocked"
    # delayed verification: undecidable until the lab clock passes the window
    help_task = task.model_copy(update={"private": {**task.private, "verifier": {"kind": "help_within",
                                                                                 "within_s": 86400}}})
    benv.reset(help_task)
    r = benv.step(Action.submit("create_post", f"How do I fix {k1} with {k2}?\nGetting error E1234 with {k1} "
                                               f"and {k2} on version 3.2.1. Please help, thanks!"))
    assert r.action_valid
    v = benv.verify_outcome()
    assert v.success is False and v.evidence["pending"] is True
    blab.advance(86400)
    v = benv.verify_outcome()
    assert v.evidence["pending"] is False and v.verifier_id == "socialweblab/help_within/v1"
    assert v.success == blab.helpful_answer_within(r.info["created_item_id"], 86400)["success"]


@pytest.mark.browser
def test_snapshot_restore_reloads_browser(benv, blab, tmp_path):
    from agentrs.socialweblab.tasks import make_task

    task = make_task(blab, "create_post", "train", 2)
    benv.reset(task)
    snap = benv.snapshot(tmp_path / "before.db")
    r = benv.step(Action.submit("create_post", "Temporary post about pandas\npandas and venv question, v1.0"))
    iid = r.info["created_item_id"]
    assert blab.items("id=?", (iid,))
    benv.restore(snap)
    assert not blab.items("id=?", (iid,))
    obs = benv.observe()
    assert obs.url == f"/item/{iid}" and any(e.role == "alert" or "not found" in e.name.lower() for e in obs.elements)
    r = benv.step(Action.navigate("/"))
    assert r.action_valid and r.observation.url == "/"  # still logged in after the restore
    assert re.search(r"Signed in as", " ".join(e.name for e in r.observation.elements))
    assert benv.get_observable_events(0)[-1]["id"] <= blab.public_events(0, 100000)[-1]["id"]
