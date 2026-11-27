"""The real-world validation bridge is off by default, cannot be reached from inside a world, and when authorised
does exactly what the operator's document allows. The only server these tests ever contact is a loopback HTTP
server started here as the "user-owned staging service" and listed explicitly in the test's authorisation file."""
import copy
import http.server
import json
import os
import pickle
import socket
import subprocess
import sys
import threading
import time

import pytest

import crucible
from crucible.realworld import (AuthorizationError, BridgeRefused, ContaminationError, ExternalConnectionAttempt,
                                HoldoutBudgetExceeded, LogCorrupt, NetworkGuard, RealWorldBridge, RealWorldResult,
                                SealedHoldoutGuard, assert_no_real_world, export_results, forbid_in_this_process,
                                is_real_world, load_authorization, verify_log)
from crucible.realworld import bridge as bridge_mod
from crucible.realworld.authorization import ACK
from crucible.realworld.log import read_log


@pytest.fixture(autouse=True)
def _validation_process(monkeypatch):
    """These tests exercise the bridge itself, i.e. they play an operator's real-world validation process; an earlier
    test of the same session may have run a world-hosting entry point in-process (crucible.hosting latch)."""
    monkeypatch.delenv("CRUCIBLE_HOSTS_WORLDS", raising=False)
    monkeypatch.setattr(bridge_mod, "_FORBIDDEN_HERE", [])

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


# ---------------------------------------------------------------------------- the staging service (loopback only)
class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", headers=()):
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        srv = self.server
        srv.hits.append((self.command, self.path))
        if srv.role == "other":
            return self._send(200, b"this server is not in scope")
        if self.path == "/api/items":
            return self._send(200, json.dumps({"items": srv.items}).encode(), [("Content-Type", "application/json")])
        if self.path == "/.well-known/crucible-realworld-bridge.txt" and srv.proof:
            return self._send(200, srv.proof.encode())
        if self.path == "/api/redirect-in":
            return self._send(302, b"", [("Location", "/api/items")])
        if self.path == "/api/redirect-out":
            return self._send(302, b"", [("Location", f"http://127.0.0.1:{srv.other_port}/steal")])
        if self.path == "/api/redirect-name":
            return self._send(302, b"", [("Location", f"http://localhost:{srv.server_address[1]}/api/items")])
        if self.path == "/api/big":
            return self._send(200, b"x" * 50_000)
        if self.path == "/admin":
            return self._send(200, b"admin area")
        return self._send(404, b"not found")

    do_HEAD = do_GET

    def do_POST(self):
        srv = self.server
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        srv.hits.append((self.command, self.path))
        if srv.role == "other":
            return self._send(200, b"this server is not in scope")
        if self.path == "/api/redirect-post":
            return self._send(307, b"", [("Location", "/api/items")])
        if self.path == "/api/items":
            srv.items.append(json.loads(body or b"{}"))
            return self._send(201, json.dumps({"n": len(srv.items)}).encode())
        return self._send(404, b"not found")

    do_DELETE = do_POST


def _serve(role):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.role, srv.hits, srv.items, srv.other_port, srv.proof = role, [], [{"id": 1}], 0, None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture()
def staging():
    a, b = _serve("staging"), _serve("other")
    a.other_port = b.server_address[1]
    yield a, b
    for s in (a, b):
        s.shutdown()
        s.server_close()


class Clock:
    def __init__(self):
        self.t = time.time()

    def __call__(self):
        return self.t


def _doc(port, **over):
    d = {"acknowledgement": ACK, "authorised_by": "Test Operator <operator@localhost>", "experiment": "loopback-test",
         "purpose": "unit test against a loopback staging service",
         "base_urls": [f"http://127.0.0.1:{port}/api/"], "methods": ["GET", "HEAD", "POST"],
         "limits": {"max_requests": 50, "max_rate_per_s": 1000.0, "max_bytes": 1_000_000, "max_wall_s": 600},
         "expires": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600)),
         "approval_required": True, "allow_private_addresses": [f"127.0.0.1:{port}"], "holdouts": {"holdout-a": 2}}
    lim = over.pop("limits", None)
    d.update(over)
    if lim:
        d["limits"] = {**d["limits"], **lim}
    return d


def _write(tmp_path, doc, name="authorisation.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    os.chmod(p, 0o600)
    return str(p)


def _bridge(tmp_path, port, approve=None, clock=None, log="bridge.log", **over):
    return RealWorldBridge.from_file(_write(tmp_path, _doc(port, **over)), log_path=str(tmp_path / log), approve=approve,
                                     **({"clock": clock} if clock else {}))


# ---------------------------------------------------------------------------- off by default
def test_bridge_refuses_everything_without_an_authorisation_document(staging):
    srv, _ = staging
    port = srv.server_address[1]
    b = RealWorldBridge()
    assert not b.enabled
    for method in ("GET", "POST", "DELETE"):
        with pytest.raises(BridgeRefused, match="no authorisation document"):
            b.request(method, f"http://127.0.0.1:{port}/api/items")
    with pytest.raises(BridgeRefused):
        b.measure("m", lambda r: 1)
    assert srv.hits == []                                     # nothing left the process
    with pytest.raises(BridgeRefused, match="load_authorization"):
        RealWorldBridge({"base_urls": ["http://127.0.0.1/"]}, log_path="/dev/null")   # a dict is not an authorisation
    # there is no environment switch: variables that look like one change nothing
    os.environ["CRUCIBLE_REALWORLD"] = "1"
    os.environ["CRUCIBLE_MODE_C_APPROVAL"] = "loopback-test:token"
    try:
        with pytest.raises(BridgeRefused):
            RealWorldBridge().request("GET", f"http://127.0.0.1:{port}/api/items")
    finally:
        del os.environ["CRUCIBLE_REALWORLD"], os.environ["CRUCIBLE_MODE_C_APPROVAL"]
    assert srv.hits == []


def test_authorisation_document_is_validated_strictly(tmp_path, monkeypatch):
    good = _doc(8080)
    assert load_authorization(_write(tmp_path, good)).experiment == "loopback-test"
    bad = [
        ("missing fields", {k: v for k, v in good.items() if k != "expires"}),
        ("unknown fields", dict(good, scope="everything")),
        ("acknowledgement must be exactly", dict(good, acknowledgement="ok")),
        ("has expired", dict(good, expires="2020-01-01T00:00:00Z")),
        ("at most 90 days", dict(good, expires="2099-01-01T00:00:00Z")),
        ("ISO-8601", dict(good, expires="tomorrow")),
        ("wildcards", dict(good, base_urls=["http://*.example.org/"])),
        ("credentials in URLs", dict(good, base_urls=["http://user:pw@127.0.0.1:8080/api/"])),
        ("scheme must be http", dict(good, base_urls=["ftp://127.0.0.1/"])),
        ("query or fragment", dict(good, base_urls=["http://127.0.0.1:8080/api/?x=1"])),
        ("1-20 base URLs", dict(good, base_urls=[])),
        ("methods not supported", dict(good, methods=["GET", "CONNECT"])),
        ("limits must have exactly", dict(good, limits={"max_requests": 5})),
        ("must be a positive number", dict(good, limits=dict(good["limits"], max_requests=0))),
        ("beyond what this bridge accepts", dict(good, limits=dict(good["limits"], max_requests=10**9))),
        ("approval_required must be true or false", dict(good, approval_required="yes")),
        ("not a literal IP address", dict(good, allow_private_addresses=["localhost:8080"])),
        ("holdouts must map", dict(good, holdouts={"h": "many"})),
        ("must be a JSON object", ["not", "an", "object"]),
    ]
    for i, (why, doc) in enumerate(bad):
        with pytest.raises(AuthorizationError, match=why):
            load_authorization(_write(tmp_path, doc, name=f"bad-{i}.json"))
    with pytest.raises(AuthorizationError, match="cannot read"):
        load_authorization(str(tmp_path / "does-not-exist.json"))
    with pytest.raises(AuthorizationError, match="local file"):
        load_authorization("https://example.org/authorisation.json")
    p = _write(tmp_path, good, name="loose.json")
    os.chmod(p, 0o666)
    with pytest.raises(AuthorizationError, match="writable by group or others"):
        load_authorization(p)
    # a document inside a sandbox working directory (where agent code can write) is refused
    sbx = tmp_path / "sbx"
    (sbx / "job-1").mkdir(parents=True)
    monkeypatch.setenv("CRUCIBLE_SANDBOX_DIR", str(sbx))
    inside = sbx / "job-1" / "authorisation.json"
    inside.write_text(json.dumps(good))
    os.chmod(inside, 0o600)
    with pytest.raises(AuthorizationError, match="sandbox directory"):
        load_authorization(str(inside))
    if os.getuid() == 0:                                      # a file written by the sandbox user is not the operator's
        p2 = _write(tmp_path, good, name="foreign.json")
        os.chown(p2, 65534, 65534)
        with pytest.raises(AuthorizationError, match="owned by the operator"):
            load_authorization(p2)


# ---------------------------------------------------------------------------- allowed and refused requests
def test_allowed_get_works_is_logged_and_only_the_checked_address_is_contacted(tmp_path, staging):
    srv, other = staging
    port = srv.server_address[1]
    with NetworkGuard(allow=[("127.0.0.1", port)]) as guard:
        with _bridge(tmp_path, port) as b:
            r = b.get(f"http://127.0.0.1:{port}/api/items")
            assert r.status == 200 and json.loads(r.body) == {"items": [{"id": 1}]} and r.hops == 0
            r2 = b.get(f"http://127.0.0.1:{port}/api/redirect-in")          # a redirect that stays in scope is followed
            assert r2.status == 200 and r2.hops == 1 and r2.url.endswith("/api/items")
            assert b.request("HEAD", f"http://127.0.0.1:{port}/api/items").body == b""
    assert {a[:1] + a[1:] for a in guard.attempts if a[0] == "connect"} == {("connect", "127.0.0.1", port)}
    assert other.hits == []
    log = str(tmp_path / "bridge.log")
    v = verify_log(log)
    assert v["ok"] and v["records"] == 10                        # open, 4 x (request, response), close
    recs = read_log(log)
    assert [x["kind"] for x in recs] == ["open", "request", "response", "request", "response", "request", "response",
                                         "request", "response", "close"]
    assert recs[0]["authorization"]["base_urls"] == [f"http://127.0.0.1:{port}/api/"]
    assert recs[1]["method"] == "GET" and recs[1]["pinned_address"] == "127.0.0.1" and recs[2]["status"] == 200
    import base64
    assert json.loads(base64.b64decode(recs[2]["body_b64"])) == {"items": [{"id": 1}]}   # reproducible from the log


def test_requests_outside_the_allow_list_are_refused_before_any_connection(tmp_path, staging):
    srv, other = staging
    port, oport = srv.server_address[1], other.server_address[1]
    resolved = []

    def resolver(host, p, **kw):
        resolved.append(host)
        return socket.getaddrinfo(host, p, **kw)
    doc = _write(tmp_path, _doc(port))
    b = RealWorldBridge(load_authorization(doc), log_path=str(tmp_path / "b.log"), resolver=resolver)
    refused = {
        f"http://127.0.0.1:{oport}/api/items": "not under any authorised base URL",       # other port
        f"http://localhost:{port}/api/items": "not under any authorised base URL",        # a name that is not listed
        "http://example.invalid/api/items": "not under any authorised base URL",          # another host entirely
        f"https://127.0.0.1:{port}/api/items": "not under any authorised base URL",       # other scheme
        f"http://127.0.0.1:{port}/admin": "not under any authorised base URL",            # outside the path prefix
        f"http://127.0.0.1:{port}/apix/items": "not under any authorised base URL",       # prefix is not a segment
        f"http://127.0.0.1:{port}/api/../admin": "dot segments",
        f"http://127.0.0.1:{port}/api/%2e%2e/admin": "dot segments",
        f"http://127.0.0.1:{port}/api/..%2fadmin": "encoded separators",
        f"http://127.0.0.1:{port}/api//items": "empty segments",
        f"http://user:s3cr3tpw@127.0.0.1:{port}/api/items": "credentials in URLs",
        f"http://127.0.0.1:{port}/api/items?api_key=tok3nvalue": "credential-like query parameter",
        f"http://127.0.0.1:{port}/api/items?x=1&Token=tok3nvalue": "credential-like query parameter",
        f"file:///etc/passwd": "scheme",
        f"http://127.0.0.1:{port}\\@example.invalid/api/items": "malformed URL",
    }
    for url, why in refused.items():
        with pytest.raises(BridgeRefused, match=why):
            b.get(url)
    with pytest.raises(BridgeRefused, match="Host header"):
        b.get(f"http://127.0.0.1:{port}/api/items", headers={"Host": "example.invalid"})
    with pytest.raises(BridgeRefused, match="method DELETE is not authorised"):
        b.request("DELETE", f"http://127.0.0.1:{port}/api/items")
    assert resolved == [] and srv.hits == [] and other.hits == []          # no name was even resolved
    # a redirect to a host that is not listed is refused; the other server is never contacted
    with pytest.raises(BridgeRefused, match="not under any authorised base URL"):
        b.get(f"http://127.0.0.1:{port}/api/redirect-out")
    with pytest.raises(BridgeRefused, match="not under any authorised base URL"):
        b.get(f"http://127.0.0.1:{port}/api/redirect-name")                # same server under an unlisted name
    assert other.hits == [] and [h[1] for h in srv.hits] == ["/api/redirect-out", "/api/redirect-name"]
    b.close()
    recs = read_log(str(tmp_path / "b.log"))
    assert sum(1 for r in recs if r["kind"] == "refusal") == len(refused) + 4      # every refusal is on record
    assert not any("s3cr3tpw" in json.dumps(r) or "tok3nvalue" in json.dumps(r) for r in recs)   # no credentials logged


def test_private_and_metadata_addresses_need_an_explicit_listing(tmp_path, staging):
    srv, _ = staging
    port = srv.server_address[1]
    # loopback is refused when the document does not list it explicitly, although the base URL names it
    b = _bridge(tmp_path, port, allow_private_addresses=[], log="p1.log")
    with pytest.raises(BridgeRefused, match="does not list explicitly"):
        b.get(f"http://127.0.0.1:{port}/api/items")
    assert srv.hits == []
    b.close()
    # names are resolved and every address is checked: a staging name that resolves to an internal address
    # (DNS rebinding, split-horizon DNS, the cloud metadata service) is refused
    for name, answers in {"staging.test": ["10.0.0.5"], "meta.test": ["169.254.169.254"],
                          "mixed.test": ["93.184.216.34", "192.168.1.10"], "v6.test": ["::1"],
                          "mapped.test": ["::ffff:127.0.0.1"], "cgnat.test": ["100.64.0.1"]}.items():
        doc = _doc(8080, base_urls=[f"http://{name}:8080/api/"], allow_private_addresses=[])
        bb = RealWorldBridge(load_authorization(_write(tmp_path, doc, name=f"{name}.json")),
                             log_path=str(tmp_path / f"{name}.log"),
                             resolver=lambda h, p, answers=answers, **kw: [(0, 0, 0, "", (a, p)) for a in answers])
        with pytest.raises(BridgeRefused, match="does not list explicitly"):
            bb.get(f"http://{name}:8080/api/items")
        bb.close()
    # an unresolvable name is a refusal, not a crash
    doc = _doc(8080, base_urls=["http://nowhere.test:8080/api/"])

    def nx(*a, **k):
        raise socket.gaierror("no such host")
    bb = RealWorldBridge(load_authorization(_write(tmp_path, doc, name="nx.json")), log_path=str(tmp_path / "nx.log"), resolver=nx)
    with pytest.raises(BridgeRefused, match="cannot resolve"):
        bb.get("http://nowhere.test:8080/api/items")
    bb.close()


def test_consequential_actions_need_live_approval_and_fail_closed(tmp_path, staging):
    srv, _ = staging
    port = srv.server_address[1]
    url = f"http://127.0.0.1:{port}/api/items"
    body = json.dumps({"id": 2}).encode()
    b = _bridge(tmp_path, port, log="a1.log")                                # no approval callback
    with pytest.raises(BridgeRefused, match="no approval callback"):
        b.request("POST", url, body=body)
    b.close()
    asked = []
    for decision, log in ((False, "a2.log"), (None, "a3.log"), ("yes", "a4.log")):   # only `True` approves
        b = _bridge(tmp_path, port, approve=lambda d, decision=decision: asked.append(d) or decision, log=log)
        with pytest.raises(BridgeRefused, match="declined"):
            b.request("POST", url, body=body)
        b.close()

    def boom(d):
        raise RuntimeError("approver crashed")
    b = _bridge(tmp_path, port, approve=boom, log="a5.log")
    with pytest.raises(BridgeRefused, match="approval callback failed"):
        b.request("POST", url, body=body)
    b.close()
    assert not any(m == "POST" for m, _ in srv.hits) and srv.items == [{"id": 1}]   # nothing consequential happened
    # with approval it goes through, and the approval is on record
    b = _bridge(tmp_path, port, approve=lambda d: asked.append(d) or True, log="a6.log")
    r = b.request("POST", url, body=body, note="create item 2")
    assert r.status == 201 and srv.items == [{"id": 1}, {"id": 2}]
    assert asked[-1]["method"] == "POST" and asked[-1]["url"] == url and asked[-1]["body_bytes"] == len(body)
    r = b.request("POST", f"http://127.0.0.1:{port}/api/redirect-post", body=body)
    assert r.status == 307 and len(srv.items) == 2            # a consequential request is never re-sent to a redirect target
    assert b.get(url).status == 200                           # reads need no approval
    b.close()
    kinds = [x["kind"] for x in read_log(str(tmp_path / "a6.log"))]
    assert kinds.count("approval") == 2 and kinds.index("approval") < kinds.index("request")
    # the document may waive approval explicitly; that is recorded too
    b = _bridge(tmp_path, port, approval_required=False, log="a7.log")
    assert b.request("POST", url, body=json.dumps({"id": 3}).encode()).status == 201
    b.close()
    assert any(x["kind"] == "approval" and "waived" in x["decision"] for x in read_log(str(tmp_path / "a7.log")))


def test_limits_and_expiry_are_enforced(tmp_path, staging):
    srv, _ = staging
    port = srv.server_address[1]
    url = f"http://127.0.0.1:{port}/api/items"
    # request count (and it survives a restart, because it is recomputed from the log)
    b = _bridge(tmp_path, port, log="l1.log", limits={"max_requests": 3})
    for _ in range(3):
        assert b.get(url).status == 200
    with pytest.raises(BridgeRefused, match="request budget of 3"):
        b.get(url)
    b.close()
    b = _bridge(tmp_path, port, log="l1.log", limits={"max_requests": 3})
    with pytest.raises(BridgeRefused, match="request budget of 3"):
        b.get(url)
    assert b.n_bytes == sum(r["body_bytes"] for r in read_log(str(tmp_path / "l1.log")) if r["kind"] in ("request", "response")) > 0
    b.close()
    assert len(srv.hits) == 3
    # ... and so does the wall-time budget, which runs from the first use of the same document
    clock = Clock()
    doc = _write(tmp_path, _doc(port), name="wall.json")
    b = RealWorldBridge(load_authorization(doc), log_path=str(tmp_path / "l0.log"), clock=clock)
    assert b.get(url).status == 200
    b.close()
    clock.t += 601
    b = RealWorldBridge(load_authorization(doc), log_path=str(tmp_path / "l0.log"), clock=clock)
    with pytest.raises(BridgeRefused, match="wall-time budget of 600"):
        b.get(url)
    b.close()
    # rate
    clock = Clock()
    b = _bridge(tmp_path, port, clock=clock, log="l2.log", limits={"max_rate_per_s": 2.0})
    assert b.get(url).status == 200
    with pytest.raises(BridgeRefused, match="rate limit of 2"):
        b.get(url)
    clock.t += 0.3
    with pytest.raises(BridgeRefused, match="rate limit"):
        b.get(url)
    clock.t += 0.3                                             # 0.6 s after the last request: one token again
    assert b.get(url).status == 200
    # wall time
    clock.t += 700
    with pytest.raises(BridgeRefused, match="wall-time budget of 600"):
        b.get(url)
    b.close()
    # bytes: a response larger than the remaining budget is cut off and refused, and the budget is then used up
    b = _bridge(tmp_path, port, log="l3.log", limits={"max_bytes": 10_000})
    with pytest.raises(BridgeRefused, match="byte budget of 10000 exceeded"):
        b.get(f"http://127.0.0.1:{port}/api/big")
    with pytest.raises(BridgeRefused, match="byte budget"):
        b.request("POST", url, body=b"x" * 10)
    assert b.n_bytes == 10_000
    b.close()
    # expiry while running
    clock = Clock()
    b = _bridge(tmp_path, port, clock=clock, log="l4.log")
    assert b.enabled and b.get(url).status == 200
    clock.t += 3601
    assert not b.enabled
    with pytest.raises(BridgeRefused, match="expired"):
        b.get(url)
    b.close()
    for name in ("l0.log", "l1.log", "l2.log", "l3.log", "l4.log"):
        assert verify_log(str(tmp_path / name))["ok"]


def test_log_verifies_and_detects_tampering(tmp_path, staging):
    srv, _ = staging
    port = srv.server_address[1]
    log = str(tmp_path / "t.log")
    with _bridge(tmp_path, port, approve=lambda d: True, log="t.log") as b:
        b.get(f"http://127.0.0.1:{port}/api/items")
        b.request("POST", f"http://127.0.0.1:{port}/api/items", body=b'{"id": 9}')
    v = verify_log(log)
    assert v["ok"] and v["records"] == 7 and v["error"] is None
    lines = open(log, "rb").read().split(b"\n")
    # 1. change one recorded response (status 201 -> 200)
    t1 = tmp_path / "t1.log"
    t1.write_bytes(b"\n".join(lines).replace(b'"status":201', b'"status":200'))
    v1 = verify_log(str(t1))
    assert not v1["ok"] and "line 6" in v1["error"]
    # 2. delete a record
    t2 = tmp_path / "t2.log"
    t2.write_bytes(b"\n".join(lines[:2] + lines[3:]))
    assert not verify_log(str(t2))["ok"]
    # 3. reorder two records
    t3 = tmp_path / "t3.log"
    t3.write_bytes(b"\n".join([lines[0], lines[2], lines[1]] + lines[3:]))
    assert not verify_log(str(t3))["ok"]
    # 4. a torn last line
    t4 = tmp_path / "t4.log"
    t4.write_bytes(b"\n".join(lines)[:-20])
    assert not verify_log(str(t4))["ok"]
    # a corrupt log is never extended
    doc = _write(tmp_path, _doc(port))
    with pytest.raises(LogCorrupt):
        RealWorldBridge(load_authorization(doc), log_path=str(t1))
    # an intact log is extended and still verifies
    with _bridge(tmp_path, port, log="t.log") as b:
        b.get(f"http://127.0.0.1:{port}/api/items")
    v = verify_log(log)
    assert v["ok"] and v["records"] == 11


# ---------------------------------------------------------------------------- outcomes, holdouts, separation
def test_outcomes_come_only_from_the_experimenters_measurement(tmp_path, staging):
    srv, _ = staging
    port = srv.server_address[1]
    url = f"http://127.0.0.1:{port}/api/items"
    b = _bridge(tmp_path, port, approve=lambda d: True, log="m.log")
    b.request("POST", url, body=b'{"id": 2}')                 # what the "agent" did
    b.get(url, headers={"Authorization": "Bearer hunter2token", "X-Trace": "trace-17"})

    def count_items(reader):                                   # the experimenter's independent measurement
        return {"items": len(json.loads(reader.get(url).body)["items"])}
    res = b.measure("items_after_run", count_items)
    assert isinstance(res, RealWorldResult) and res.value == {"items": 2} and res.origin == "real-world"
    assert res.experiment == "loopback-test" and res.authorization_sha256 and res.log_head
    # an agent's claim is not a measurement: only a function supplied by the experimenter produces a result
    for claim in ("the agent says it created 100 items", {"items": 100}, 100, None):
        with pytest.raises(BridgeRefused, match="function supplied by the experimenter"):
            b.measure("claimed", claim)
    with pytest.raises(TypeError):
        RealWorldResult(object(), value=100)                  # cannot be constructed by hand
    with pytest.raises(AttributeError):
        res.value = {"items": 100}                            # cannot be edited afterwards
    reader = b.reader()                                        # measurement functions can only observe
    assert not hasattr(reader, "post") and not hasattr(reader, "request")
    with pytest.raises(BridgeRefused, match="plain JSON"):
        b.measure("bad", lambda r: object())
    b.close()
    recs = read_log(str(tmp_path / "m.log"))
    m = [r for r in recs if r["kind"] == "measurement"]
    assert len(m) == 1 and m[0]["value"] == {"items": 2} and "count_items" in m[0]["function"]
    raw = open(str(tmp_path / "m.log")).read()
    assert "hunter2token" not in raw and "trace-17" in raw and "sha256:" in raw    # secret headers are logged as hashes


def test_sealed_holdout_budget_is_counted_and_enforced(tmp_path, staging):
    srv, _ = staging
    port = srv.server_address[1]
    url = f"http://127.0.0.1:{port}/api/items"
    calls = []

    def score(reader):
        calls.append(1)
        return len(json.loads(reader.get(url).body)["items"])
    b = _bridge(tmp_path, port, log="h.log")
    assert b.holdout_remaining("holdout-a") == 2
    assert b.evaluate_holdout("holdout-a", "score", score).holdout == "holdout-a"
    b.close()
    b = _bridge(tmp_path, port, log="h.log")                   # a restart does not reset the count
    assert b.holdout_remaining("holdout-a") == 1
    b.evaluate_holdout("holdout-a", "score", score)
    with pytest.raises(HoldoutBudgetExceeded, match="budget of 2"):
        b.evaluate_holdout("holdout-a", "score", score)
    with pytest.raises(HoldoutBudgetExceeded, match="no stated budget"):
        b.evaluate_holdout("holdout-unlisted", "score", score)
    assert len(calls) == 2 and b.holdout_remaining("holdout-a") == 0
    b.close()
    assert verify_log(str(tmp_path / "h.log"))["ok"]
    # the guard also works on its own (e.g. for sealed synthetic evaluation sets)
    g = SealedHoldoutGuard({"sealed-set": 1}, ledger_path=str(tmp_path / "ledger.log"))
    assert g.evaluate("sealed-set", lambda: 42) == 42
    with pytest.raises(HoldoutBudgetExceeded):
        g.evaluate("sealed-set", lambda: 43)
    with pytest.raises(HoldoutBudgetExceeded):
        SealedHoldoutGuard({"sealed-set": 1}, ledger_path=str(tmp_path / "ledger.log")).charge("sealed-set")


def test_real_world_results_cannot_enter_synthetic_stores(tmp_path, staging):
    from crucible.core.canonical import canonical_bytes, to_jsonable
    from crucible.core.kernel import World
    from crucible.storage.memory import MemoryStore
    from crucible.storage.sqlite import SQLiteStore
    from crucible.worldgen import generate
    srv, _ = staging
    port = srv.server_address[1]
    b = _bridge(tmp_path, port, log="r.log")
    res = b.measure("m", lambda r: {"status": r.get(f"http://127.0.0.1:{port}/api/items").status})
    b.close()
    # generic serialisers, the kernel's encoders, copying
    for fn in (json.dumps, pickle.dumps, copy.copy, copy.deepcopy, to_jsonable, canonical_bytes):
        with pytest.raises((TypeError, ContaminationError)):
            fn(res)
    for fn in (json.dumps, pickle.dumps, copy.deepcopy, to_jsonable, canonical_bytes):
        with pytest.raises((TypeError, ContaminationError)):
            fn({"synthetic": 1, "mixed_in": [res]})
    # event stores
    crucible.load_all_modules()
    ev = {"seq": 1, "t": 0, "type": "note", "actor": "sys", "cmd": None, "payload": {"result": res}, "vis": "internal", "hash": "x"}
    ms = MemoryStore()
    ms.create_world("w", {}, None)
    with pytest.raises((TypeError, ContaminationError)):
        ms.append_events("w", [ev])
    ss = SQLiteStore(str(tmp_path / "synthetic.db"))
    ss.create_world("w", {}, None)
    with pytest.raises((TypeError, ContaminationError)):
        ss.append_events("w", [ev])
    assert ms.read_events("w") == [] and ss.read_events("w") == []
    # a world: it cannot arrive as a tool argument or be snapshotted with one inside
    w = World.create(generate({"family": "tiny", "drop_modules": ["consumers", "shocks"]}, 1))
    w.system_call("admin.join", {"handle": "ag", "endowment": 1000})
    seq = w.seq
    r = w.execute("p:ag", {"tool": "docs.create", "args": {"title": "real", "body": res}})
    assert not r.ok and r.error["code"] == "invalid_action"
    assert all(e.type == "cmd.rejected" for e in w.events_range(seq))
    # export only to a dedicated, tagged file
    with pytest.raises(ContaminationError, match="realworld.json"):
        export_results(str(tmp_path / "ow_results.json"), [res])
    synthetic = tmp_path / "x.realworld.json"
    synthetic.write_text(json.dumps({"schema": "crucible-redteam/1", "probes": []}))
    with pytest.raises(ContaminationError, match="not a real-world results file"):
        export_results(str(synthetic), [res])                  # will not overwrite a synthetic file
    with pytest.raises(ContaminationError, match="no synthetic rows"):
        export_results(str(tmp_path / "y.realworld.json"), [{"success": True}])
    out = export_results(str(tmp_path / "run1.realworld.json"), [res])
    doc = json.load(open(out))
    assert doc["origin"] == "real-world" and doc["results"][0]["origin"] == "real-world" and doc["results"][0]["value"] == {"status": 200}
    # and the check a synthetic writer runs on its own rows catches exported rows too
    assert is_real_world(res) and is_real_world(doc) and is_real_world({"rows": doc["results"]})
    assert not is_real_world({"success": True, "origin": "synthetic"})
    with pytest.raises(ContaminationError):
        assert_no_real_world({"runs": [{"success": True}, doc["results"][0]]})
    assert_no_real_world({"runs": [{"success": True}]})


# ---------------------------------------------------------------------------- unreachable from inside a world
def test_no_tool_reaches_the_bridge_and_a_world_cannot_enable_it():
    crucible.load_all_modules()
    from crucible.core.kernel import MODULES
    from crucible.redteam import default_world_factory, run_probe
    from crucible.redteam.independence import StaticReachability
    w = default_world_factory(open_world=True)
    w.system_call("admin.join", {"handle": "inside", "endowment": 100_000})
    # no endpoint is implemented by, named after, or documented as the bridge
    for name, ep in w.endpoints.items():
        mod = getattr(ep.handler, "__module__", "") or ""
        assert not mod.startswith("crucible.realworld"), name
        assert "realworld" not in name and "real-world" not in ep.doc.lower() and "real world" not in ep.doc.lower(), name
    assert not any(cls.__module__.startswith("crucible.realworld") for cls in MODULES.values())
    # an agent's attempts to name it get the ordinary unknown-tool answer
    for tool in ("realworld.request", "bridge.request", "crucible.realworld.request", "admin.realworld", "world.realworld"):
        r = w.execute("p:inside", {"tool": tool, "args": {"method": "GET", "url": "http://example.org/"}})
        assert not r.ok and r.error["code"] in ("unknown_tool", "forbidden"), (tool, r.error)
    r = w.execute("p:inside", {"tool": "world.describe", "args": {"host": "realworld"}})
    assert not r.ok
    # the import graph of all world mechanics does not contain the bridge (static, includes function-level imports)
    rec = run_probe(default_world_factory, StaticReachability())
    assert rec["result"] == "held", rec
    assert rec["evidence"]["modules_reachable"] > 30
    # and a process that only runs worlds never even loads the package
    code = ("import sys; sys.path.insert(0, %r); import crucible; crucible.load_all_modules();"
            "from crucible.core.kernel import World; from crucible.worldgen import generate;"
            "w = World.create(generate({'family': 'open_digital', 'overrides': {'n_consumers': 100}}, 2));"
            "w.system_call('admin.join', {'handle': 'ag', 'endowment': 1000});"
            "w.execute('p:ag', {'tool': 'net.fetch', 'args': {'url': 'http://example.org/'}});"
            "w.advance_time(w.now + 6 * 3600);"
            "print(sorted(m for m in sys.modules if m.startswith(('crucible.realworld', 'crucible.redteam'))))" % SRC)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"
    # operator code in a world-hosting process can latch the bridge shut for good
    try:
        forbid_in_this_process("this process hosts synthetic worlds")
        with pytest.raises(BridgeRefused, match="forbidden in this process"):
            RealWorldBridge()
    finally:
        bridge_mod._FORBIDDEN_HERE.clear()


def test_network_guard_blocks_before_the_operating_system_is_asked():
    with NetworkGuard() as g:
        with pytest.raises(ExternalConnectionAttempt):
            socket.create_connection(("192.0.2.1", 80), timeout=0.01)      # TEST-NET-1: never contacted
        with pytest.raises(ExternalConnectionAttempt):
            socket.getaddrinfo("example.invalid", 80)
        with pytest.raises(ExternalConnectionAttempt):
            socket.gethostbyname("example.invalid")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with pytest.raises(ExternalConnectionAttempt):
            s.sendto(b"x", ("192.0.2.1", 53))
        with pytest.raises(ExternalConnectionAttempt):
            s.connect(("127.0.0.1", 9))                                    # loopback is not allowed unless asked for
        s.close()
        a, b = socket.socketpair()                                         # local pipes are not network connections
        a.sendall(b"ok")
        assert b.recv(2) == b"ok"
        a.close()
        b.close()
    assert [x[0] for x in g.attempts] == ["connect", "resolve", "resolve", "sendto", "connect"]
    assert socket.getaddrinfo("127.0.0.1", 80)                             # restored afterwards


def test_default_crucible_process_makes_no_network_connection():
    """Create and run a full open-world economy with random and hostile agent actions under a socket-level guard
    that records every resolve / connect / sendto attempt and raises on any of them."""
    crucible.load_all_modules()
    from crucible.agents.base import Runner
    from crucible.agents.scripted import RandomPolicy
    from crucible.core.kernel import World
    from crucible.worldgen import generate
    with NetworkGuard() as g:
        w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 200}}, 5))
        actors = [w.system_call("admin.join", {"handle": f"agent{i}", "endowment": 500_000})["principal"] for i in range(2)]
        run = Runner(w, [(a, RandomPolicy()) for a in actors], dt=3600, seed=1, record=False)
        run.run(12)
        for url in ("http://example.org/", "https://169.254.169.254/latest/meta-data/", "http://127.0.0.1:22/",
                    "http://freecredits.crucible/", "ftp://example.org/x"):
            w.execute(actors[0], {"tool": "net.fetch", "args": {"url": url}})
        w.execute(actors[0], {"tool": "mail.send", "args": {"to": ["someone@example.org"], "subject": "x", "body": "x"}})
        w.execute(actors[0], {"tool": "search.query", "args": {"q": "http://example.org"}})
        w.advance_time(w.now + 12 * 3600)
        assert w.check_invariants() == []
        blob = w.snapshot_blob()
        World.from_blob(blob, world_id="copy").close()
        w.close()
    assert g.attempts == [], g.attempts


def test_public_hosts_need_the_operators_ownership_file(tmp_path, staging):
    """Wave-3b verifier: a denylist of platforms cannot be complete (other Mastodon/Lemmy instances, Amazon SES, URL
    shorteners, an IP literal of a listed platform were all accepted). A host that is not this machine must serve
    the ownership file naming the authorisation's hash before any request of the experiment reaches it; checked
    once per host, logged, never following a redirect. A 'public' host is simulated by a resolver answering a public
    address and a connection hook that sends it to a loopback server (nothing leaves the machine)."""
    import socket as _s
    import crucible.realworld.bridge as BR
    srv, other = staging
    port = srv.server_address[1]
    doc = _write(tmp_path, _doc(port, base_urls=[f"http://shop.operator.example:{port}/api/"], allow_private_addresses=[]))
    auth = load_authorization(doc)
    resolver = lambda h, p, type=None: [(_s.AF_INET, _s.SOCK_STREAM, 6, "", ("93.184.215.14", p))]  # noqa: E731
    real = BR.socket.create_connection
    BR.socket.create_connection = lambda addr, *a, **k: real(("127.0.0.1", addr[1]), *a, **k)
    try:
        b = RealWorldBridge(auth, log_path=str(tmp_path / "own.log"), resolver=resolver)
        n0 = len(srv.hits)
        with pytest.raises(BridgeRefused, match="does not show that the operator controls it"):
            b.get(f"http://shop.operator.example:{port}/api/items")
        assert [h[1] for h in srv.hits[n0:]] == [BR.OWNERSHIP_PATH]          # only the ownership file was asked for
        b.close()
        srv.proof = f"{BR.OWNERSHIP_LINE} {auth.sha256}\n"
        b = RealWorldBridge(auth, log_path=str(tmp_path / "own2.log"), resolver=resolver)
        n1 = len(srv.hits)
        assert b.get(f"http://shop.operator.example:{port}/api/items").status == 200
        assert [h[1] for h in srv.hits[n1:]] == [BR.OWNERSHIP_PATH, "/api/items"]
        b.close()
        other_doc = _write(tmp_path, _doc(port, base_urls=[f"http://shop.operator.example:{port}/api/"],
                                          allow_private_addresses=[], purpose="another experiment"), name="o.json")
        b = RealWorldBridge(load_authorization(other_doc), log_path=str(tmp_path / "own3.log"), resolver=resolver)
        with pytest.raises(BridgeRefused, match="does not show"):    # the file names another document's hash
            b.get(f"http://shop.operator.example:{port}/api/items")
        b.close()
    finally:
        BR.socket.create_connection = real
        srv.proof = None
