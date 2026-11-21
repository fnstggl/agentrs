"""The synthetic internet fabric: agent-built web applications with routes, sessions, Crucible-ID login and
persistent storage; name registration; rate limits and failures; crawling, link graph and search; real-browser and
API equivalence."""
import json
import re

import pytest

import crucible

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400

NOTES_APP = r'''
import html
import crucible_sdk as sdk


def _page(title, body):
    return ("<!doctype html><html><head><title>" + html.escape(title) + "</title></head><body>" + body +
            "</body></html>")


def index(req, caller):
    notes = sdk.call("db.query", collection="notes", order_by="t", desc=True, limit=20)
    items = "".join('<li><a href="/note/%s">%s</a> by %s</li>' % (n["id"], html.escape(n["text"][:40]),
                                                                  html.escape(n["by"])) for n in notes)
    if req["user"]:
        form = ('<form method="post" action="/notes"><label for="text">Note</label>'
                '<input id="text" name="text" required><button type="submit">Post</button></form>')
    else:
        form = '<a href="/_login?return=/">Log in to post</a>'
    seen = int(req["session"].get("visits", 0)) + 1
    links = '<p>See also <a href="http://%s.crucible/">our friends</a>.</p>' % req["query"].get("friend", "shopping")
    return {"html": _page("Notes board", "<h1>Notes board</h1><ul>" + items + "</ul>" + form + links +
                          "<p id='visits'>visits: %d</p>" % seen), "session": {"visits": seen}}


def create(req, caller):
    if not req["user"]:
        return {"status": 401, "text": "log in first"}
    text = (req["form"].get("text") or "").strip()
    if not text:
        return {"status": 400, "text": "empty note"}
    r = sdk.call("db.put", collection="notes", doc={"text": text[:500], "by": req["user"], "t": req["now"]})
    return {"redirect": "/note/" + r["id"]}


def note(req, caller):
    n = sdk.call("db.get", collection="notes", id=req["params"]["id"])
    if n is None:
        return {"status": 404, "html": _page("Not found", "<p>No such note.</p>")}
    return {"html": _page("Note", "<p id='note'>" + html.escape(n["text"]) + "</p><a href='/'>back</a>")}


def api_notes(req, caller):
    return {"json": sdk.call("db.query", collection="notes", limit=100)}


def secret(req, caller):
    return {"html": _page("private", '<meta name="robots" content="noindex"><p>staff only</p>')}
'''


def manifest(rate=120):
    return json.dumps({"entry": "service.py", "web": {
        "title": "Notes board", "rate_per_min": rate,
        "routes": {"GET /": "index", "POST /notes": "create", "GET /note/{id}": "note", "GET /api/notes": "api_notes",
                   "GET /staff": "secret"}}})


def _world(seed=1):
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 200}}, seed))
    w.system_call("admin.join", {"handle": "alice", "endowment": 3_000_000, "password": "pw-alice"})
    w.system_call("admin.join", {"handle": "bob", "endowment": 500_000, "password": "pw-bob"})
    return w


def _ex(w, who):
    return lambda t, **k: w.execute(who, {"tool": t, "args": k})  # noqa: E731


def _deploy(w, host="notes", rate=120):
    ex = _ex(w, "p:alice")
    me = w.modules["bank"].primary_account(w.state, "p:alice")
    if not w.state.has("orgs", "o:aliceweb"):
        assert ex("id.create_org", handle="aliceweb", name="Alice Web", kind="company").ok
        assert ex("biz.register", org="o:aliceweb", account=me).ok
        acct = ex("bank.open_account", owner="o:aliceweb").result["account"]
        ex("bank.transfer", from_account=me, to=acct, amount=1_000_000)
    acct = w.modules["bank"].primary_account(w.state, "o:aliceweb")
    repo = ex("code.create_repo", name=host, owner="o:aliceweb").result["repo"]
    assert ex("code.commit", repo=repo, files={"crucible.json": manifest(rate), "service.py": NOTES_APP}).ok
    inst = ex("cloud.rent", provider="o:cloud1", cores=1, account=acct, owner="o:aliceweb").result["instance"]
    r = ex("cloud.deploy", repo=repo, host=host, instance=inst)
    assert r.ok and r.result["status"] == "live", (r.error, r.result)
    return ex


def test_agent_web_app_routes_sessions_login_and_storage_through_the_api():
    w = _world()
    _deploy(w)
    bob = _ex(w, "p:bob")
    r = bob("net.fetch", url="http://notes.crucible/").result
    assert r["status"] == 200 and r["content_type"] == "text/html" and "Log in to post" in r["body"]
    r2 = bob("net.fetch", url="http://notes.crucible/").result
    assert "visits: 2" in r2["body"]                                         # a persistent server-side session
    r = bob("net.fetch", url="http://notes.crucible/notes", method="POST", form={"text": "hello"}).result
    assert r["status"] == 401
    assert bob("net.login", site="notes").ok
    r = bob("net.fetch", url="http://notes.crucible/notes", method="POST", form={"text": "hello world"}).result
    assert r["status"] == 303 and r["location"].startswith("/note/")
    r = bob("net.fetch", url="http://notes.crucible" + r["location"]).result
    assert "hello world" in r["body"]
    api = json.loads(bob("net.fetch", url="http://notes.crucible/api/notes").result["body"])
    assert api[0]["by"] == "p:bob" and api[0]["text"] == "hello world"     # state persisted in the app database
    assert bob("net.fetch", url="http://notes.crucible/nope").result["status"] == 404
    assert bob("net.fetch", url="http://nowhere.crucible/").result["status"] == 404      # DNS failure
    assert not w.execute("p:bob", {"tool": "db.put", "args": {"collection": "notes", "doc": {"x": 1}}}).ok
    w2 = World.replay(w.spec, [e.to_dict() for e in w.log])
    assert w2.digest() == w.digest()


def test_names_rate_limits_and_outages():
    w = _world()
    alice = _deploy(w, rate=3)
    assert alice("net.register", name="bestnotes", org="o:aliceweb").ok
    assert not _ex(w, "p:bob")("net.register", name="bestnotes").ok                      # taken
    assert alice("net.point", name="bestnotes", host="notes").ok
    bob = _ex(w, "p:bob")
    codes = [bob("net.fetch", url="http://bestnotes.crucible/").result["status"] for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    w.advance_time(w.now + 120)
    assert bob("net.fetch", url="http://bestnotes.crucible/").result["status"] == 200
    prov = w.state.tables["instances"][w.state.tables["deployments"][w.state.tables["hosts"]["notes"]["deployment"]]
                                       ["instance"]]["provider"]
    w._run_system("outage", lambda ctx: ctx.emit("cloud.outage", {"id": prov, "start": ctx.now, "until": ctx.now + 3600},
                                                 vis="public"))
    assert bob("net.fetch", url="http://notes.crucible/").result["status"] == 503


def test_crawler_indexes_public_pages_follows_links_and_ranks_by_links():
    w = _world()
    _deploy(w)
    _deploy(w, host="shopping")                                   # the site the notes board links to
    bob = _ex(w, "p:bob")
    bob("net.login", site="notes")
    bob("net.fetch", url="http://notes.crucible/notes", method="POST", form={"text": "a guide to sorting records"})
    w.advance_time(w.now + DAY + 3600)
    pages = {k for k, _ in w.state.scan("web_pages")}
    assert "notes.crucible/" in pages and any(p.startswith("notes.crucible/note/") for p in pages)
    assert "notes.crucible/staff" not in pages                                  # not linked: not discovered
    res = w.query("p:bob", "search.query", {"q": "guide sorting records", "kind": "page"})
    hits = res["results"] if isinstance(res, dict) else res
    assert any("notes.crucible/note/" in (h.get("ref") or "") for h in hits), hits
    assert any(lk["dst"] == "shopping" and lk["src_host"] == "notes" for _, lk in w.state.scan("web_links"))
    rank = w.hooks["rank_signal"]
    shop_home = w.state.get("search_docs", "page:shopping.crucible/")
    assert shop_home is not None and rank(w, shop_home) > 1.0             # inbound links raise the ranking signal


def test_noindex_and_login_required_pages_are_not_indexed():
    w = _world()
    alice = _deploy(w)
    del alice
    w._run_system("link", lambda ctx: None)
    st = w.state
    net = w.modules["net"]
    r = w.execute("p:bob", {"tool": "net.fetch", "args": {"url": "http://notes.crucible/staff"}}).result
    assert r["status"] == 200 and "noindex" in r["body"]
    w._run_system("crawl", lambda ctx: net.pr_crawl(ctx))
    assert not any(k.endswith("/staff") for k, _ in st.scan("web_pages"))


@pytest.mark.browser
def test_real_chromium_browses_an_agent_web_app_with_the_same_effects_as_the_api():
    pytest.importorskip("playwright")
    from crucible.web.app import make_web_app
    from crucible.web.browser import BrowserSession, playwright_available, serve_in_thread
    if not playwright_available():
        pytest.skip("Chromium unavailable")
    wb = _world(3)
    _deploy(wb)
    wa = World.from_blob(wb.snapshot_blob(), world_id="api-twin")
    url, stop = serve_in_thread(make_web_app(wb))
    b = BrowserSession(url, timeout_ms=20_000)
    try:
        b.act({"action": "goto", "url": "id.crucible"})
        b.act({"action": "fill", "target": "login-handle", "value": "bob"})
        b.act({"action": "fill", "target": "Password", "value": "pw-bob"})
        b.act({"action": "click", "target": "Sign in"})
        seq0 = wb.seq
        o = b.act({"action": "goto", "url": "http://notes.crucible/"})
        assert o["ok"] and o["title"] == "Notes board", o
        o = b.act({"action": "click", "target": "Log in to post"})
        assert "wants to know who you are" in o["text"]
        o = b.act({"action": "click", "target": "Log in as p:bob"})
        assert o["url"].startswith("http://notes.crucible/") and any(e["id"] == "text" for e in o["elements"]), o
        b.act({"action": "fill", "target": "text", "value": "from a real browser"})
        o = b.act({"action": "click", "target": "Post"})
        assert "from a real browser" in o["text"] and re.search(r"/note/\w+", o["url"])
        o = b.act({"action": "click", "target": "back"})
        assert "from a real browser" in o["text"]
    finally:
        b.close()
        stop()
    # the browser's commands, replayed through the API on the twin world, give the same state
    cmds = [e for e in wb.log if e.seq > seq0 and e.type == "cmd" and e.payload.get("n") is not None]
    assert {c.payload["tool"] for c in cmds} <= {"net.fetch", "net.login"}
    for c in cmds:
        assert wa.execute(c.actor, {"tool": c.payload["tool"], "args": c.payload["args"]}).ok
    assert wa.digest() == wb.digest()


def _site_world(n=2, seed=5):
    from crucible.worldgen.sitegen import generate_spec
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 200, "module_config": {
        "webgen": {"sites": n}}}}, seed))
    rng = w.rng_for("webgen")
    return w, [generate_spec(rng, i) for i in range(n)]


def test_generated_sites_pass_the_critic_and_are_crawled():
    from crucible.web.critic import api_critic
    w, specs = _site_world(3)
    for spec in specs:
        rep = api_critic(w, spec["name"], spec)
        assert rep["accepted"], [c for c in rep["checks"] if not c["ok"]]
    w.advance_time(w.now + DAY + 3600)
    pages = {v["host"] for _, v in w.state.scan("web_pages")}
    assert {s["name"] for s in specs} <= pages


def test_critic_rejects_a_site_that_fakes_success():
    from crucible.web.critic import api_critic
    from crucible.worldgen.sitegen import app_files
    w, specs = _site_world(1, seed=9)
    spec = dict(specs[0], name="fakeshop")
    w.system_call("admin.join", {"handle": "alice", "endowment": 3_000_000})
    ex = _ex(w, "p:alice")
    me = w.modules["bank"].primary_account(w.state, "p:alice")
    ex("id.create_org", handle="fakeco", name="Fake", kind="company")
    ex("biz.register", org="o:fakeco", account=me)
    acct = ex("bank.open_account", owner="o:fakeco").result["account"]
    ex("bank.transfer", from_account=me, to=acct, amount=500_000)
    files = app_files(spec, acct)
    # the "create" handler pretends to save and shows a success page, but writes nothing
    files["service.py"] = files["service.py"].replace(
        '    r = sdk.call("db.put", collection=e["name"], doc=doc)', '    r = {"id": "saved"}')
    repo = ex("code.create_repo", name="fakeshop", owner="o:fakeco").result["repo"]
    ex("code.commit", repo=repo, files=files)
    inst = ex("cloud.rent", provider="o:cloud1", cores=1, account=acct, owner="o:fakeco").result["instance"]
    assert ex("cloud.deploy", repo=repo, host="fakeshop", instance=inst).result["status"] == "live"
    rep = api_critic(w, "fakeshop", spec)
    assert not rep["accepted"]
    assert any("writes exactly one record" in c["check"] and not c["ok"] for c in rep["checks"])


@pytest.mark.browser
def test_browser_critic_validates_a_generated_site_in_chromium():
    from crucible.web.app import make_web_app
    from crucible.web.browser import playwright_available, serve_in_thread
    from crucible.web.critic import browser_critic
    if not playwright_available():
        pytest.skip("Chromium unavailable")
    w, specs = _site_world(1, seed=11)
    w.system_call("admin.join", {"handle": "qa", "endowment": 100_000, "password": "pw-qa"})
    url, stop = serve_in_thread(make_web_app(w))
    try:
        rep = browser_critic(url, w, specs[0]["name"], specs[0], "qa", "pw-qa")
    finally:
        stop()
    assert rep["accepted"], [c for c in rep["checks"] if not c["ok"]]


def test_async_runner_interleaves_agents_on_independent_clocks_reproducibly():
    from crucible.agents.base import AsyncRunner
    from crucible.agents.scripted import RandomPolicy

    def run():
        w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 200}}, 4))
        agents = []
        for k in range(12):
            w.system_call("admin.join", {"handle": f"r{k}", "endowment": 10_000})
            agents.append((f"p:r{k}", RandomPolicy()))
        ar = AsyncRunner(w, agents, seed=3, think_mean_s=900)
        n = ar.run_until(w.now + 6 * 3600)
        return w, ar, n
    w1, ar1, n1 = run()
    w2, ar2, n2 = run()
    assert n1 > 12 and ar1.schedule == ar2.schedule and w1.digest() == w2.digest()
    times = [t for t, _, _ in ar1.schedule]
    assert len(set(times)) > len(times) // 3                       # actions spread over time, not in lockstep rounds
    per_agent = {}
    for t, a, _ in ar1.schedule:
        per_agent.setdefault(a, []).append(t)
    assert len({len(v) for v in per_agent.values()}) > 1             # agents act at different rates
