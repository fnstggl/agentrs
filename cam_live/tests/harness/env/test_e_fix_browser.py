"""Regression tests of the E-fix workstream for the browser on real Chromium (run with ``-m browser``): every
request the browser makes is checked before it leaves (M1), a click that opens a tab is reported as what it was
(M2), pages with unparseable numbers and huge fields are observable and bounded (M2, M8), and a stored secret is
typed by reference (M5)."""
import json
import socket
import time

import pytest

from cam_live.contracts import Modality, ResultStatus, TaskSpec
from cam_live.harness.browser import BrowserExecutor, executable_by, find_chromium
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.internal import InternalExecutor

from e_fix_support import CountingServer, FourArgBroker, html
from eb_support import Driver, GrantPolicy

pytestmark = pytest.mark.browser
SECRET = "Tr0ub4dor-e-fix-BROWSER-41q!"


@pytest.fixture(scope="module")
def servers():
    a, b = CountingServer(), CountingServer()
    for p in ("/secret", "/pixel", "/hop", "/beacon", "/landing", "/sw-target"):
        b.page(p, "B-ONLY-CONTENT")
    yield a, b
    a.close()
    b.close()


@pytest.fixture(scope="module")
def w(tmp_path_factory, servers):
    a, b = servers
    run_dir = str(tmp_path_factory.mktemp("efix_browser"))
    policy = GrantPolicy(deny_urls=[b.host])
    broker = FourArgBroker({"site_pw": SECRET})
    env = ComputerEnvironment(run_dir, policy, secrets=broker)
    env.register(InternalExecutor(run_dir, clock=env.clock))
    env.register(BrowserExecutor(run_dir, clock=env.clock))
    d = Driver(env, policy, TaskSpec("obj_efix", "exercise the network guard", mode="staging"))
    d.a, d.b, d.broker, d.run_dir = a, b, broker, run_dir
    yield d
    env.close()


def checked(w, res):
    return [(r["op"]["op"], r["op"]["arguments"].get("url", ""), r["verdict"])
            for r in w.env.records("env.primitive_check", 500, res.action_id)]


def test_redirect_hops_are_checked_before_they_are_sent(w):
    a, b = w.a, w.b
    a.routes["/to-b"] = lambda *x: (302, {"Location": b.url + "/secret"}, "")
    a.routes["/to-a"] = lambda *x: (302, {"Location": "/to-a2"}, "")
    a.routes["/to-a2"] = lambda *x: (307, {"Location": a.url + "/fine"}, "")
    a.page("/fine", html("fine", "<h1>fine</h1>"))
    a.routes["/sub-hop"] = lambda *x: (302, {"Location": b.url + "/hop"}, "")
    a.page("/fetcher", html("fetcher", "<div id=o></div><script>fetch('/sub-hop').then(()=>o.innerText='done')"
                                       ".catch(()=>o.innerText='failed')</script>"))
    n0 = b.count("/secret")
    res = w.do("browser.navigate", url=a.url + "/to-b")
    assert res.status == ResultStatus.DENIED and b.count("/secret") == n0          # the hop never reached B
    assert any(op == "browser.navigate" and b.host in u and v == "deny" for op, u, v in checked(w, res))
    ok = w.ok("browser.navigate", url=a.url + "/to-a")                             # a chain of allowed hops works
    assert ok.observations[0].data["url"] == a.url + "/fine"
    n1 = b.count("/hop")
    w.ok("browser.navigate", url=a.url + "/fetcher")
    time.sleep(0.8)
    assert b.count("/hop") == n1                                                   # a subresource's hop as well
    assert "failed" in w.inspect().text


def test_websockets_and_unseen_tunnels_are_checked(w):
    a, b = w.a, w.b
    a.page("/ws", html("ws", f"""<div id=o></div><script>
      const out = (s) => document.getElementById('o').innerText += s + ' ';
      const bad = new WebSocket('ws://{b.host}/sock'); bad.onopen = () => out('bad-open'); bad.onclose = () => out('bad-closed');
      const good = new WebSocket('ws://{a.host}/sock'); good.onopen = () => out('good-open'); good.onerror = () => out('good-error');
      </script>"""))
    b.ws.clear()
    a.ws.clear()
    res = w.ok("browser.navigate", url=a.url + "/ws")
    time.sleep(1.0)
    assert b.ws == [] and a.ws == ["/sock"]                     # forbidden refused, allowed connected
    text = w.inspect().text
    assert "bad-closed" in text and "bad-open" not in text and "good-open" in text
    assert any(u.startswith("ws://" + b.host) and v == "deny" for _, u, v in checked(w, res))
    # the proxy itself: no credentials, no service; a tunnel to a forbidden host is refused before it is opened
    sess = w.env.executor(Modality.STRUCTURED_BROWSER)._sessions["default"]
    port = sess.guard.port
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(f"GET {b.url}/secret HTTP/1.1\r\nHost: {b.host}\r\n\r\n".encode())
        assert s.recv(200).startswith(b"HTTP/1.1 407")
    n = b.count("/secret")
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(f"CONNECT {b.host} HTTP/1.1\r\nProxy-Authorization: {sess.guard._auth}\r\n\r\n".encode())
        assert s.recv(200).startswith(b"HTTP/1.1 403")
    assert b.count("/secret") == n and sess.guard.stats["unauthenticated"] >= 1


def test_writes_of_scripts_are_reported_and_service_workers_are_blocked(w):
    a, b = w.a, w.b
    a.routes["/api/delete"] = lambda *x: (200, {"Content-Type": "application/json"}, "{}")
    a.routes["/sw.js"] = lambda *x: (200, {"Content-Type": "application/javascript"},
                                     f"self.addEventListener('install', () => fetch('{b.url}/sw-target'));")
    a.page("/scripts", html("scripts", f"""<div id=o></div><script>
      const out = (s) => document.getElementById('o').innerText += s + ' ';
      fetch('/api/delete', {{method: 'POST', body: 'id=1'}}).then(() => out('post-done'));
      navigator.sendBeacon('{b.url}/beacon', 'x');
      if (navigator.serviceWorker) navigator.serviceWorker.register('/sw.js').then(() => out('sw-registered'))
        .catch((e) => out('sw-refused')); else out('sw-absent');
      out(typeof WebTransport === 'undefined' ? 'no-webtransport' : 'webtransport');
      </script>"""))
    res = w.ok("browser.navigate", url=a.url + "/scripts")
    time.sleep(1.0)
    ins = w.do("browser.inspect")
    text = ins.observations[0].text
    reported = [(op, u, v) for op, u, v in checked(w, res) + checked(w, ins) if "/api/delete" in u]
    assert a.count("/api/delete", "POST") == 1 and reported and reported[0][0] == "browser.request"
    assert b.count("/beacon") == 0 and b.count("/sw-target") == 0 and a.count("/sw.js") == 0
    assert "sw-refused" in text and "sw-registered" not in text and "no-webtransport" in text


def test_a_click_that_opens_a_tab_is_reported_ok(w):
    a = w.a
    a.page("/pop", html("pop", "<button id=pa onclick=\"window.open('/fine')\">popup</button>"))
    a.page("/fine", html("fine", "<h1>fine</h1>"))
    w.ok("browser.navigate", url=a.url + "/pop")
    outcomes = []
    for _ in range(8):
        n = a.count("/fine")
        r = w.do("browser.click", selector="#pa")
        time.sleep(0.2)
        outcomes.append((r.status, r.failure_detail[:60], a.count("/fine") - n))
        w.ok("browser.select_tab", tab="t1")
    assert all(o[0] == ResultStatus.OK for o in outcomes), outcomes


def test_pages_with_unparseable_numbers_and_huge_fields_are_observable_and_bounded(w):
    a = w.a
    a.routes["/api/count"] = lambda *x: (200, {"Content-Type": "application/json"}, "{}")
    a.page("/boom", html("boom", "<p>Members: 1e999999999999</p><p>Visible: 3</p>"
                                 "<dl style='display:none'><dt>Hidden</dt><dd>77</dd></dl>"))
    a.page("/boom2", html("boom2", "<button id=buy onclick=\"fetch('/api/count',{method:'POST'});"
                                   "document.getElementById('o').innerText='Order total: 1e999999999999'\">Buy</button>"
                                   "<p id=o></p>"))
    o = w.ok("browser.navigate", url=a.url + "/boom").observations[0]
    q = {x.name: x.value for x in o.quantities}
    assert q == {"page.boom.visible": 3.0}                     # the huge exponent is no number; hidden <dl> unread
    w.ok("browser.navigate", url=a.url + "/boom2")
    r = w.do("browser.click", selector="#buy")
    assert r.status == ResultStatus.OK and a.count("/api/count", "POST") == 1
    a.page("/bigtable", html("bigtable", "<table><tr>" + "".join(f"<td>{'x' * 79}{i}</td>" for i in range(20000))
                                         + "</tr></table>"))
    a.page("/bighref", html("bighref", "<a href='/t?" + "q" * 2_000_000 + "'>long link</a><form action='/f?"
                                       + "z" * 1_000_000 + "'><input name=a></form>"))
    a.page("/bigtitle", html("T" * 3_000_000, "<p>x</p>"))
    for path in ("/bigtable", "/bighref", "/bigtitle"):
        o = w.ok("browser.navigate", url=a.url + path).observations[0]
        assert len(json.dumps(o.to_dict())) < 200_000, path
    links = w.inspect().data
    assert links is not None


def test_a_stored_secret_is_typed_by_reference(w):
    a = w.a
    a.page("/login", html("login", "<label>Password <input id=pw type=password></label><div id=e></div><script>"
                                   "pw.addEventListener('input', (e) => document.getElementById('e').innerText = "
                                   "e.target.value.toUpperCase().split('').join(' '))</script>"))
    w.ok("browser.navigate", url=a.url + "/login")
    res = w.do("browser.fill", selector="#pw", secret="secret:site_pw")
    assert res.status == ResultStatus.OK and w.broker.asked[-1] == ("site_pw", a.url + "/login")
    blob = json.dumps(res.to_dict()) + json.dumps(w.inspect().to_dict()) + open(w.run_dir + "/events.jsonl").read()
    assert SECRET not in blob and " ".join(SECRET.upper()) not in blob        # also its transformed echo
    plain = w.do("browser.fill", selector="#pw", secret="no_such")
    assert plain.status == ResultStatus.DENIED


def test_a_service_uid_finds_a_chromium_it_can_execute():
    import os
    import pwd
    try:
        p = pwd.getpwnam("cambrowser")
    except KeyError:
        pytest.skip("the cambrowser user does not exist")
    exe = find_chromium(p.pw_uid, p.pw_gid, BrowserExecutor._bundled_chromium())
    if not exe:
        pytest.fail("no Chromium the browser uid can execute: run python -m cam_live.harness.browser --install-shared")
    assert executable_by(exe, p.pw_uid, p.pw_gid) and os.path.isfile(exe)


def test_the_browser_itself_sends_nothing_a_page_did_not_cause(w):
    """Chromium's own background requests (autofill form queries, network time, search preconnect) are switched
    off; anything else would still have to pass the network guard."""
    from urllib.parse import urlsplit
    a = w.a
    a.page("/form", html("form", "<form><input name=email autocomplete=email><input name=pw type=password>"
                                 "<input type=submit></form>"))
    w.policy.primitives.clear()
    for _ in range(2):
        w.ok("browser.navigate", url=a.url + "/form")
        w.ok("browser.click", selector="input[name=email]")
        time.sleep(0.8)
    hosts = {urlsplit(p[1].get("url") or p[1].get("page_url") or "").hostname for p in w.policy.primitives}
    assert hosts == {"127.0.0.1"}, hosts
    pids = w.env.executor(Modality.STRUCTURED_BROWSER).chromium_pids()
    cmd = ""
    for pid in pids:
        try:
            c = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if any(x.startswith("--proxy-server=") for x in c):
            cmd = c
            break
    feats = [x for x in cmd if x.startswith("--disable-features=")]
    assert len(feats) == 1 and "AutofillServerCommunication" in feats[0] and "PreconnectToSearch" in feats[0]
