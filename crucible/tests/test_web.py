"""Browser mode (crucible.web): HTML pages over World.query, forms over World.execute.

(a) TestClient tests: login/SSO, permissions, forms -> state changes, error display, XSS escaping, the generic
    page of a deployed host, form -> argument conversion, a WorldServer backend.
(b)+(c) a real Chromium session (marked ``browser``) plays a multi-step workflow through the forms; the same
    scenario is played headlessly through World.execute on a second world from the same spec and seed, and the two
    worlds must have identical digests, identical command sequences and identical event-hash chains.
(d) the controlled browser network blocks every non-*.crucible URL.
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

import pytest

import crucible
from crucible.core.kernel import World

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")
from fastapi.testclient import TestClient  # noqa: E402

from crucible.web.app import FormError, form_to_args, make_web_app  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

crucible.load_all_modules()

ALICE, BOB = "p:alice", "p:bob"
ALICE_ACCT = "acct_000017"


def new_world() -> World:
    w = World.create(generate("tiny", 1))
    w.system_call("admin.join", {"handle": "alice", "endowment": 1_000_000, "password": "pw-alice"})
    w.system_call("admin.join", {"handle": "bob", "endowment": 200_000, "password": "pw-bob"})
    return w


def commands(w: World, since: int = 0) -> list[tuple]:
    """(actor, event type, tool, args) of every actor command after `since` (accepted and rejected)."""
    out = []
    for e in w.log:
        if e.seq <= since:
            continue
        if e.type == "cmd" and e.payload.get("n") is not None:
            out.append((e.actor, "cmd", e.payload["tool"], e.payload["args"]))
        elif e.type == "cmd.rejected":
            out.append((e.payload.get("actor"), "rejected", e.payload["tool"], e.payload["args"]))
    return out


# ---------------------------------------------------------------------- a tiny "browser" for TestClient
class _Forms(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms: dict[str, dict] = {}
        self.cur: dict | None = None
        self.select: str | None = None
        self.first_opt: str | None = None
        self.textarea: str | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            self.cur = {"action": a.get("action", ""), "method": a.get("method", "get").lower(), "fields": {},
                        "buttons": []}
            self.forms[a.get("id") or f"form{len(self.forms)}"] = self.cur
        elif self.cur is None:
            return
        elif tag == "input":
            t = a.get("type", "text")
            if t == "checkbox":
                if "checked" in a:
                    self.cur["fields"][a["name"]] = a.get("value", "on")
            elif t == "submit":
                self.cur["buttons"].append((a.get("name"), a.get("value"), a.get("id")))
            elif a.get("name"):
                self.cur["fields"][a["name"]] = a.get("value", "")
        elif tag == "select":
            self.select, self.first_opt = a.get("name"), None
        elif tag == "option" and self.select:
            v = a.get("value", "")
            if self.first_opt is None:
                self.first_opt = v
                self.cur["fields"][self.select] = v
            if "selected" in a:
                self.cur["fields"][self.select] = v
        elif tag == "textarea":
            self.textarea = a.get("name")
            self.cur["fields"][self.textarea] = ""
        elif tag == "button" and a.get("type", "submit") == "submit":
            self.cur["buttons"].append((a.get("name"), a.get("value"), a.get("id")))

    def handle_endtag(self, tag):
        if tag == "form":
            self.cur = None
        elif tag == "select":
            self.select = None
        elif tag == "textarea":
            self.textarea = None

    def handle_data(self, data):
        if self.textarea and self.cur is not None:
            self.cur["fields"][self.textarea] += data


def forms_of(html: str) -> dict[str, dict]:
    p = _Forms()
    p.feed(html)
    return p.forms


def submit(c: TestClient, page_url: str, form_id: str, values: dict | None = None, button: str | None = None):
    """GET the page, fill the form like a user (defaults + `values`), press a button, follow redirects."""
    page = c.get(page_url)
    assert page.status_code == 200, (page_url, page.status_code, page.text[:500])
    forms = forms_of(page.text)
    assert form_id in forms, f"no form {form_id!r} on {page_url}: {sorted(forms)}"
    f = forms[form_id]
    data = dict(f["fields"])
    for k, v in (values or {}).items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    if button is not None:
        b = next(b for b in f["buttons"] if b[2] == button or b[1] == button)
        data[b[0]] = b[1]
    url = urljoin(str(page.url), f["action"])
    if f["method"] == "get":
        return c.get(url, params=data)
    return c.post(url, data=data)


def login(c: TestClient, handle: str, pw: str, start: str = "http://id.crucible/"):
    r = c.get(start)
    assert "login-form" in r.text
    return submit(c, str(r.url), "login-form", {"handle": handle, "password": pw})


def alerts(html: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", m).strip() for m in re.findall(r'role="alert"[^>]*>(.*?)</div>', html, re.S)]


@pytest.fixture
def world():
    return new_world()


@pytest.fixture
def alice(world):
    c = TestClient(make_web_app(world), base_url="http://id.crucible")
    r = login(c, "alice", "pw-alice")
    assert r.status_code == 200 and "Signed in as" in r.text
    return c


ALL_PAGES = ["http://id.crucible/", "http://id.crucible/directory", "http://id.crucible/directory?kind=persons",
             "http://id.crucible/grants", "http://id.crucible/notifications", "http://id.crucible/notifications?all=1",
             "http://id.crucible/profile?p=o:standards", "http://id.crucible/profile?p=p:bob",
             "http://bank.crucible/", f"http://bank.crucible/account?id={ALICE_ACCT}", "http://bank.crucible/invoices",
             "http://bank.crucible/mandates", "http://mail.crucible/", "http://mail.crucible/?folder=sent",
             "http://mail.crucible/compose", "http://mail.crucible/search?q=hello", "http://forum.crucible/",
             "http://forum.crucible/b?board=general&sort=top", "http://search.crucible/",
             "http://search.crucible/doc?id=doc_000003", "http://docs.crucible/", "http://docs.crucible/?scope=public",
             "http://docs.crucible/d?id=doc_000003", "http://code.crucible/", "http://code.crucible/?owner=o:recordco1",
             "http://issues.crucible/", "http://cal.crucible/", "http://support.crucible/",
             "http://support.crucible/stats?owner=o:recordco1", "http://ads.crucible/", "http://market.crucible/",
             "http://market.crucible/?kind=api&sort=price", "http://market.crucible/listing?id=lst_000001",
             "http://market.crucible/orders", "http://market.crucible/orders?role=seller",
             "http://market.crucible/inventory", "http://cloud.crucible/", "http://cloud.crucible/provider",
             "http://biz.crucible/", "http://biz.crucible/company?org=o:cloud1", "http://contracts.crucible/",
             "http://jobs.crucible/", "http://jobs.crucible/bounty?id=bounty_000001", "http://research.crucible/",
             "http://record1.crucible/", "http://bank.crucible/_api", "http://forum.crucible/_api"]


# ====================================================================== (a) TestClient level
def test_login_sso_and_logout(world):
    c = TestClient(make_web_app(world), base_url="http://id.crucible")
    r = c.get("http://bank.crucible/account?id=" + ALICE_ACCT)
    assert r.status_code == 200 and "login-form" in r.text and str(r.url).startswith("http://id.crucible/login")
    bad = submit(c, str(r.url), "login-form", {"handle": "alice", "password": "nope"})
    assert bad.status_code == 401 and "wrong password" in bad.text
    r = submit(c, str(r.url), "login-form", {"handle": "alice", "password": "pw-alice"})
    # single sign-on: id.crucible issued a one-time ticket, bank.crucible set its own cookie, we are back
    assert r.status_code == 200 and str(r.url) == f"http://bank.crucible/account?id={ALICE_ACCT}"
    assert "$10,000.00" in r.text and "Signed in as" in r.text
    r = c.get("http://forum.crucible/")                                 # another host: SSO hop again
    assert r.status_code == 200 and any(h.url.path == "/_sso" for h in r.history)
    # logout ends the session on every host
    r = submit(c, "http://forum.crucible/", "logout-form")
    assert "login-form" in r.text
    assert "login-form" in c.get(f"http://bank.crucible/account?id={ALICE_ACCT}").text
    # path-prefix mode for clients that cannot set Host
    p = TestClient(make_web_app(world), base_url="http://127.0.0.1")
    r = p.get("/_h/bank/")
    assert str(r.url).startswith("http://127.0.0.1/_h/id/login")
    r = submit(p, str(r.url), "login-form", {"handle": "alice", "password": "pw-alice"})
    assert str(r.url) == "http://127.0.0.1/_h/bank/" and 'href="/_h/bank/invoices"' in r.text


def test_every_page_renders_without_recording(world, alice):
    d0, s0, h0 = world.digest(), world.seq, world.last_hash
    for u in ALL_PAGES:
        r = alice.get(u)
        assert r.status_code == 200, (u, r.status_code, alerts(r.text))
        assert not alerts(r.text), (u, alerts(r.text))
        assert "<script" not in r.text and "script-src 'none'" in r.headers["content-security-policy"]
    assert (world.digest(), world.seq, world.last_hash) == (d0, s0, h0)   # renders are World.query only


def test_forms_execute_recorded_commands(world, alice):
    s0 = world.seq
    r = submit(alice, "http://id.crucible/", "id-create_org", {"handle": "acme", "name": "Acme Inc", "kind": "company"})
    assert r.status_code == 200 and str(r.url).startswith("http://id.crucible/profile?p=o%3Aacme")
    assert "Acme Inc" in r.text and 'role="status"' in r.text
    r = submit(alice, "http://biz.crucible/", "biz-register", {"org": "o:acme", "account": ALICE_ACCT})
    assert "registered" in r.text
    r = submit(alice, "http://bank.crucible/", "bank-open_account", {"name": "ops", "owner": "o:acme"})
    acct = world.state.find("accounts", "owner", "o:acme")[0]
    org_acct = next(a for a in world.state.find_ids("accounts", "owner", "o:acme"))
    assert org_acct in r.text and acct["name"] == "ops"
    r = submit(alice, "http://bank.crucible/", "bank-transfer",
               {"from_account": ALICE_ACCT, "to": org_acct, "amount": "2500", "memo": "seed"})
    assert world.modules["bank"].balance(world.state, org_acct) == 2500
    assert commands(world, s0) == [
        (ALICE, "cmd", "id.create_org", {"handle": "acme", "name": "Acme Inc", "kind": "company"}),
        (ALICE, "cmd", "biz.register", {"org": "o:acme", "account": ALICE_ACCT}),
        (ALICE, "cmd", "bank.open_account", {"name": "ops", "owner": "o:acme"}),
        (ALICE, "cmd", "bank.transfer", {"from_account": ALICE_ACCT, "to": org_acct, "amount": 2500, "memo": "seed"}),
    ]
    # the statement page shows it; the forum: post, reply (as bob), vote (enum int via buttons), read
    st = alice.get(f"http://bank.crucible/account?id={org_acct}")
    assert "$25.00" in st.text and "seed" in st.text
    r = submit(alice, "http://forum.crucible/", "forum-post", {"board": "dev", "title": "Hello devs", "body": "Hi"})
    post = world.state.find_ids("forum_posts", "author", ALICE)[0]
    assert str(r.url).startswith(f"http://forum.crucible/post?id={post}") and "Hello devs" in r.text
    r = submit(alice, f"http://forum.crucible/post?id={post}", "forum-vote", button="forum-vote-1")
    assert any("cannot vote on your own" in x for x in alerts(r.text))        # rejected (and recorded)
    bobc = TestClient(make_web_app(world), base_url="http://id.crucible")
    login(bobc, "bob", "pw-bob")
    r = submit(bobc, f"http://forum.crucible/post?id={post}", "forum-vote", button="forum-vote-1")
    assert world.state.get("forum_posts", post)["score"] == 1 and "score 1" in r.text
    r = submit(bobc, f"http://forum.crucible/post?id={post}", "forum-reply", {"body": "first!"})
    assert "first!" in r.text
    assert commands(world, s0)[-4:] == [
        (ALICE, "cmd", "forum.post", {"board": "dev", "title": "Hello devs", "body": "Hi"}),
        (ALICE, "rejected", "forum.vote", {"post": post, "value": 1}),
        (BOB, "cmd", "forum.vote", {"post": post, "value": 1}),
        (BOB, "cmd", "forum.reply", {"post": post, "body": "first!"})]


def test_more_services_through_forms(world, alice):
    """Docs, code (commit editor + run), issues, calendar (buttons), contracts (JSON clauses), market, mail."""
    s0 = world.seq
    r = submit(alice, "http://docs.crucible/", "docs-create", {"title": "Plan", "body": "line 1\nline 2"})
    doc = re.search(r"/d\?id=(doc_\d+)", str(r.url)).group(1)
    r = submit(alice, f"http://docs.crucible/d?id={doc}", "docs-edit", {"body": "line 1\nline 2\nline 3"})
    assert "line 3" in r.text and world.state.get("documents", doc)["version"] == 2
    submit(alice, f"http://docs.crucible/d?id={doc}", "docs-share", {"principal": "bob", "perm": "read"})
    r = submit(alice, "http://code.crucible/", "code-create_repo", {"name": "tools", "visibility": "public"})
    assert str(r.url).startswith("http://code.crucible/repo?id=alice%2Ftools")
    src = "def main(inp, caller):\n    return {'echo': inp}\n"
    r = submit(alice, "http://code.crucible/repo?id=alice/tools", "code-commit",
               {"files.path": "main.py", "files.content": src, "message": "init"})
    assert "main.py" in r.text
    r = alice.get("http://code.crucible/file?repo=alice/tools&path=main.py")
    assert "return {&#39;echo&#39;: inp}" in r.text
    r = submit(alice, "http://code.crucible/repo?id=alice/tools", "code-run",
               {"target": "main.py:main", "input": '{"a": 1}'})
    assert 'role="status"' in r.text and "echo" in r.text
    r = submit(alice, "http://issues.crucible/", "issues-create_project", {"name": "tracker", "public": "true"})
    proj = re.search(r"id=(proj_\d+)", str(r.url)).group(1)
    r = submit(alice, f"http://issues.crucible/project?id={proj}", "issues-create",
               {"title": "Bug", "body": "broken", "labels": "bug, ui"})
    assert world.state.get("iss_issues", re.search(r"id=(iss_\d+)", str(r.url)).group(1))["labels"] == ["bug", "ui"]
    r = submit(alice, "http://cal.crucible/", "cal-create", {"title": "Sync", "start": "3600", "duration": "1800",
                                                             "attendees": "bob"})
    ev = re.search(r"id=(cev_\d+)", str(r.url)).group(1)
    clauses = json.dumps([{"type": "payment", "from": BOB, "to": ALICE, "amount": 100}])
    r = submit(alice, "http://contracts.crucible/", "contracts-propose",
               {"parties": "p:alice, p:bob", "clauses": clauses, "title": "Deal"})
    ctr = re.search(r"id=(ctr_\d+)", str(r.url)).group(1)
    assert "Deal" in r.text and world.state.get("contracts", ctr)["status"] == "proposed"
    r = submit(alice, "http://market.crucible/", "market-list", {"kind": "service", "title": "Consulting",
                                                                 "price_cents": "300", "description": "an hour"})
    assert "Consulting" in r.text
    r = submit(alice, "http://mail.crucible/compose", "mail-send", {"to": "bob", "subject": "Hi", "body": "see you"})
    got = [x for x in commands(world, s0)]
    tools = [t for _, kind, t, _ in got if kind == "cmd"]
    assert tools == ["docs.create", "docs.edit", "docs.share", "code.create_repo", "code.commit", "code.run",
                     "issues.create_project", "issues.create", "cal.create", "contracts.propose", "market.list",
                     "mail.send"], got
    assert got[4][3] == {"repo": "alice/tools", "files": {"main.py": src}, "message": "init"}
    assert got[5][3]["input"] == {"a": 1} and got[9][3]["clauses"][0]["amount"] == 100
    # bob: calendar buttons (required boolean), mail.read (a recorded open), contract signing
    world.advance_time(600)
    bobc = TestClient(make_web_app(world), base_url="http://id.crucible")
    login(bobc, "bob", "pw-bob")
    s1 = world.seq
    r = submit(bobc, f"http://cal.crucible/event?id={ev}", "cal-respond", button="cal-respond-1")
    assert world.query(ALICE, "cal.get", {"event": ev})["responses"][BOB] == "accepted", alerts(r.text)
    inbox = bobc.get("http://mail.crucible/")
    assert "Hi" in inbox.text
    msg = re.search(r'id="mail-read-(mail_\d+)"', inbox.text).group(1)
    r = submit(bobc, "http://mail.crucible/", f"mail-read-{msg}")
    assert "see you" in r.text and str(r.url).startswith(f"http://mail.crucible/msg?id={msg}")
    assert bobc.get(f"http://mail.crucible/msg?id={msg}").text.count("see you") >= 1   # re-render from session
    r = submit(bobc, f"http://contracts.crucible/contract?id={ctr}", "contracts-sign",
               {"accounts": json.dumps({"0": "acct_000018"})})
    assert world.state.get("contracts", ctr)["status"] in ("active", "completed"), alerts(r.text)
    assert [(t, a) for _, k, t, a in commands(world, s1)] == [
        ("cal.respond", {"event": ev, "accept": True, "allow_conflict": False}),
        ("mail.read", {"msg": msg}), ("contracts.sign", {"contract": ctr, "accounts": {"0": "acct_000018"}})]
    assert world.check_invariants() == []


def test_marketplace_jobs_and_inline_forms(world, alice):
    """Row buttons (invites, orders, instances, campaigns, invoices) and the remaining services' forms."""
    bob = TestClient(make_web_app(world), base_url="http://id.crucible")
    login(bob, "bob", "pw-bob")
    s0 = world.seq

    def ok(r):
        assert r.status_code == 200 and 'role="status"' in r.text, (str(r.url), alerts(r.text))
        return r

    def row_form(c, url, prefix):
        return next(k for k in forms_of(c.get(url).text) if k.startswith(prefix))
    ok(submit(alice, "http://id.crucible/", "id-create_org", {"handle": "acme", "name": "Acme", "kind": "company"}))
    ok(submit(alice, "http://biz.crucible/", "biz-register", {"org": "o:acme", "account": ALICE_ACCT}))
    ok(submit(alice, "http://id.crucible/profile?p=o:acme", "id-invite", {"person": "bob"}))
    f = row_form(bob, "http://id.crucible/", "id-respond_invite")
    ok(submit(bob, "http://id.crucible/", f, button=f + "-1"))                       # Accept
    assert world.modules["identity"].role(world.state, "o:acme", BOB) == "member"
    r = ok(submit(alice, "http://jobs.crucible/", "jobs-post", {"org": "o:acme", "title": "Dev", "role": "developer",
                                                                "salary_cents_month": "100000"}))
    posting = re.search(r"id=(job_\d+)", str(r.url)).group(1)
    ok(submit(bob, f"http://jobs.crucible/posting?id={posting}", "jobs-apply", {"note": "hire me"}))
    page = alice.get(f"http://jobs.crucible/posting?id={posting}").text            # org admin: applications
    assert 'id="s-applications"' in page and "hire me" in page.split('id="s-applications"')[1].split("</section>")[0]
    assert 'id="s-applications"' not in bob.get(f"http://jobs.crucible/posting?id={posting}").text
    r = ok(submit(alice, "http://market.crucible/", "market-list", {"kind": "service", "title": "Consult",
                                                                    "price_cents": "300"}))
    lid = re.search(r"id=(lst_\d+)", str(r.url)).group(1)
    ok(submit(bob, f"http://market.crucible/listing?id={lid}", "market-buy", {}))
    ok(submit(alice, "http://market.crucible/orders?role=seller",
              row_form(alice, "http://market.crucible/orders?role=seller", "market-deliver")))
    ok(submit(bob, "http://market.crucible/orders", row_form(bob, "http://market.crucible/orders", "market-confirm")))
    ok(submit(bob, f"http://market.crucible/listing?id={lid}", "market-rate", {"stars": "5", "text": "great"}))
    ok(submit(alice, "http://cloud.crucible/", "cloud-rent", {"cores": "1"}))
    ok(submit(alice, "http://cloud.crucible/", row_form(alice, "http://cloud.crucible/", "cloud-release")))
    ok(submit(alice, "http://ads.crucible/", "ads-create_campaign", {"keywords": "acme, widgets", "bid_cents": "5",
                                                                     "daily_budget_cents": "500", "title": "Ad"}))
    ok(submit(alice, "http://ads.crucible/", row_form(alice, "http://ads.crucible/", "ads-pause")))
    ok(submit(alice, "http://bank.crucible/invoices", "bank-invoice", {"payer": "bob", "amount": "100"}))
    ok(submit(bob, "http://bank.crucible/invoices", row_form(bob, "http://bank.crucible/invoices", "bank-pay_invoice")))
    ok(submit(bob, "http://support.crucible/?host=record1", "support-open", {"subject": "Slow", "body": "x"}))
    ok(submit(alice, "http://research.crucible/", "research-survey", {"need": "sort.records"}))
    ok(submit(alice, "http://forum.crucible/b?board=dev", "forum-follow-board"))
    ok(submit(alice, "http://id.crucible/grants", "id-grant", {"grantee": "bob", "cap": "bank.spend", "limit": "1000"}))
    tools = [t for _, kind, t, _ in commands(world, s0) if kind == "cmd"]
    assert len(tools) == 21 and tools[-1] == "id.grant" and world.check_invariants() == []
    assert "research.survey" in tools and "market.confirm" in tools
    statuses = [o["status"] for o in world.query(BOB, "market.orders", {"role": "buyer"})]
    assert statuses == ["completed"], statuses


def test_errors_are_displayed(world, alice):
    s0 = world.seq
    r = submit(alice, "http://bank.crucible/", "bank-transfer", {"to": "bob", "amount": str(10 ** 11)})
    a = alerts(r.text)
    assert any("insufficient_funds" in x for x in a), a
    assert commands(world, s0) == [(ALICE, "rejected", "bank.transfer",
                                    {"from_account": ALICE_ACCT, "to": "bob", "amount": 10 ** 11})]
    assert forms_of(r.text)["bank-transfer"]["fields"]["amount"] == str(10 ** 11)   # input kept for correction
    # input that cannot be converted is not sent to the world at all
    s1 = world.seq
    r = submit(alice, "http://bank.crucible/", "bank-transfer", {"to": "bob", "amount": "ten dollars"})
    assert any("invalid_input" in x and "whole number" in x for x in alerts(r.text))
    assert "Nothing was sent to the world" in r.text and world.seq == s1
    # a kernel validation error (pattern) and a handler error (unknown post) come back as error codes
    r = submit(alice, "http://id.crucible/", "id-create_org", {"handle": "Bad Handle!", "name": "x"})
    assert any("invalid_action" in x for x in alerts(r.text))
    r = alice.get("http://forum.crucible/post?id=fpost_999999")
    assert r.status_code == 404 and any("not_found" in x for x in alerts(r.text))
    # notifications: bob pays alice; alice sees it on every page
    world.execute(BOB, {"tool": "bank.transfer", "args": {"from_account": "acct_000018", "to": "alice",
                                                          "amount": 777, "memo": "for lunch"}})
    r = alice.get("http://docs.crucible/")
    note = r.text.split('id="notifications"', 1)[1]
    assert "bank.txn" in note and "for lunch" in note


def test_xss_is_escaped(world, alice):
    evil = '<script>alert("pwned")</script><img src=x onerror=alert(1)>'
    world.execute(BOB, {"tool": "forum.post", "args": {"board": "general", "title": "T " + evil,
                                                       "body": evil + " visit record1.crucible"}})
    post = world.state.find_ids("forum_posts", "author", BOB)[0]
    world.execute(BOB, {"tool": "docs.create", "args": {"title": evil, "body": evil, "visibility": "public"}})
    world.execute(BOB, {"tool": "mail.send", "args": {"to": ["alice"], "subject": evil, "body": evil}})
    world.execute(BOB, {"tool": "id.update_profile", "args": {"bio": evil}})
    world.advance_time(600)
    pages = [f"http://forum.crucible/post?id={post}", "http://forum.crucible/b?board=general",
             "http://docs.crucible/?scope=public", "http://mail.crucible/", "http://id.crucible/profile?p=p:bob"]
    for u in pages:
        html = alice.get(u).text
        assert "<script>" not in html and "<img src=x" not in html, u
        assert "&lt;script&gt;" in html, u
    r = submit(alice, "http://search.crucible/", "search-query", {"q": "pwned"})
    assert "<script>" not in r.text and "&lt;script&gt;" in r.text
    # host mentions become real links, everything else stays text
    html = alice.get(f"http://forum.crucible/post?id={post}").text
    assert '<a href="http://record1.crucible/">record1.crucible</a>' in html
    assert alice.get("http://forum.crucible/").headers["x-content-type-options"] == "nosniff"


def test_permissions_and_admin_tools(world, alice):
    s0 = world.seq
    r = alice.get("http://bank.crucible/account?id=acct_000018")      # bob's account
    assert r.status_code == 400 and any("forbidden" in x for x in alerts(r.text))
    assert "$2,000.00" not in r.text
    assert "admin." not in alice.get("http://id.crucible/_api").text
    assert alice.get("http://admin.crucible/").status_code == 404
    csrf = forms_of(alice.get("http://bank.crucible/").text)["bank-transfer"]["fields"]["_csrf"]
    # admin tools and tools of another host are refused without being sent to the world
    for host, tool in (("bank", "admin.endow"), ("bank", "id.create_org"), ("id", "admin.join")):
        r = alice.post(f"http://{host}.crucible/_act", data={"_tool": tool, "_csrf": csrf, "_back": "/",
                                                            "principal": ALICE, "amount": "1000000"})
        assert any("forbidden" in x for x in alerts(r.text)), (tool, alerts(r.text))
    # a forged form without the session's CSRF token is refused
    r = alice.post("http://bank.crucible/_act", data={"_tool": "bank.transfer", "_back": "/", "to": "bob",
                                                      "from_account": ALICE_ACCT, "amount": "5"})
    assert r.status_code == 403
    assert world.seq == s0
    # bob's private mail never shows up for alice
    world.execute(BOB, {"tool": "mail.send", "args": {"to": ["bob"], "subject": "bob-secret", "body": "s3cret"}})
    world.advance_time(600)
    assert "bob-secret" not in alice.get("http://mail.crucible/").text
    assert "bob-secret" not in alice.get("http://mail.crucible/?mailbox=p:bob").text
    # anonymous requests never see a page
    s1 = world.seq
    anon = TestClient(make_web_app(world), base_url="http://id.crucible")
    assert "login-form" in anon.get("http://forum.crucible/").text
    r = anon.post("http://bank.crucible/_act", data={"_tool": "bank.transfer", "to": "bob", "amount": "5"})
    assert "login-form" in r.text and world.seq == s1                       # no command from anonymous users


def test_deployed_host_generic_page(world, alice):
    r = alice.get("http://record1.crucible/")
    assert r.status_code == 200 and "record1.solve" in r.text and "o:recordco1" in r.text
    fs = forms_of(r.text)
    assert "record1-solve" in fs and "_args" in fs["record1-solve"]["fields"]       # schema without properties
    assert not world.knows(ALICE, "host", "record1")
    s0 = world.seq
    r = submit(alice, "http://record1.crucible/", "world-describe")              # "Add to my tools"
    assert world.knows(ALICE, "host", "record1")
    assert any(t["name"] == "record1.solve" for t in world.available_tools(ALICE))
    args = {"records": [{"id": "a", "key": 3}, {"id": "b", "key": 1}], "descending": False}
    r = submit(alice, "http://record1.crucible/", "record1-solve", {"_args": json.dumps(args)})
    assert commands(world, s0) == [(ALICE, "cmd", "world.describe", {"host": "record1"}),
                                   (ALICE, "cmd", "record1.solve", args)]
    assert 'role="status"' in r.text and "status" in r.text
    # unknown hosts look like a DNS failure
    assert alice.get("http://nosuch-host.crucible/").status_code == 404


def test_form_to_args_conversion():
    params = {"type": "object", "properties": {
        "s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": "number"}, "b": {"type": "boolean"},
        "e": {"type": "integer", "enum": [-1, 1]}, "tags": {"type": "array", "items": {"type": "string"}},
        "o": {"type": "object"}, "any": {}, "files": {"type": "object"}, "req": {"type": "string"}},
        "required": ["req"]}
    got = form_to_args(params, {"s": [""], "i": ["42"], "n": ["0.5"], "b": ["true"], "e": ["-1"],
                                "tags": ["a, b\nc"], "o": ['{"k": 1}'], "any": ["plain text"],
                                "files.path": ["x.py"], "files.content": ["print(1)\n"], "req": [""]})
    assert got == {"i": 42, "n": 0.5, "b": True, "e": -1, "tags": ["a", "b", "c"], "o": {"k": 1},
                   "any": "plain text", "files": {"x.py": "print(1)\n"}, "req": ""}
    assert form_to_args(params, {"any": ['{"x": [1]}'], "_args": ['{"extra": true}']}) == {"any": {"x": [1]},
                                                                                         "extra": True}
    for bad in ({"i": ["1.5"]}, {"e": ["2"]}, {"o": ["[1]"]}, {"b": ["maybe"]}):
        with pytest.raises(FormError):
            form_to_args(params, bad)


def test_worldserver_backend_and_token_login():
    from crucible.api import WorldServer
    srv = WorldServer("memory://")
    try:
        wid = srv.create_world(generate("tiny", 1))
        carol = srv.join_world(wid, "carol", endowment=5_000, password="pw-carol")
        dave = srv.join_world(wid, "dave", endowment=1_000)                 # no password: log in with the token
        app = make_web_app(srv, wid)
        c = TestClient(app, base_url="http://id.crucible")
        r = login(c, "dave", dave.token)
        assert "Signed in as" in r.text and "p:dave" in r.text
        r = submit(c, "http://bank.crucible/", "bank-transfer", {"to": "carol", "amount": "250"})
        assert 'role="status"' in r.text
        assert srv.observe(wid, carol).summary["bank"]                         # same world, through the server
        evs = srv.replay_events(wid)
        assert any(e["type"] == "cmd" and e["payload"]["tool"] == "bank.transfer" for e in evs)   # flushed
        c2 = TestClient(app, base_url="http://id.crucible")
        assert login(c2, "dave", "cru_wrong").status_code == 401
    finally:
        srv.close()


# ====================================================================== (b)-(d) real Chromium
def _chromium_ok() -> bool:
    try:
        from crucible.web.browser import playwright_available
    except ImportError:
        return False
    return playwright_available()


browser = pytest.mark.skipif(not _chromium_ok(), reason="Playwright/Chromium not available")
POST_TITLE = "Acme launches widgets"
POST_BODY = "Acme Inc is open for business. <b>Not bold</b> - see record1.crucible"


def headless_scenario(w: World, org_acct: str) -> list[dict]:
    """The scenario as plain tool calls (what an API agent sends)."""
    calls = [
        ("id.create_org", {"handle": "acme", "name": "Acme Inc", "kind": "company"}),
        ("biz.register", {"org": "o:acme", "account": ALICE_ACCT, "sector": "software"}),
        ("bank.open_account", {"name": "ops", "owner": "o:acme"}),
        ("bank.transfer", {"from_account": ALICE_ACCT, "to": org_acct, "amount": 5000, "memo": "seed money"}),
        ("bank.transfer", {"from_account": ALICE_ACCT, "to": org_acct, "amount": 10 ** 11}),     # rejected
        ("forum.post", {"board": "general", "title": POST_TITLE, "body": POST_BODY}),
        ("search.query", {"q": "Acme launches"}),
    ]
    return [w.execute(ALICE, {"tool": t, "args": a}).to_dict() for t, a in calls]


@pytest.fixture(scope="module")
def live():
    from crucible.web.browser import BrowserSession, serve_in_thread
    w = new_world()
    url, stop = serve_in_thread(make_web_app(w))
    b = None
    try:
        b = BrowserSession(url, timeout_ms=20_000)
        yield w, b, url
    finally:
        if b is not None:
            b.close()
        stop()


@pytest.mark.browser
@browser
def test_browser_workflow_equals_headless(live):
    wb, b, _ = live
    seq0 = wb.seq
    o = b.act({"action": "goto", "url": "id.crucible"})
    assert o["ok"] and "login" in o["url"] and any(e["id"] == "login-handle" for e in o["elements"])
    b.act({"action": "fill", "target": "login-handle", "value": "alice"})
    b.act({"action": "fill", "target": "Password", "value": "pw-alice"})             # by label
    o = b.act({"action": "click", "target": "Sign in"})                             # by button text
    assert o["ok"] and o["url"] == "http://id.crucible/" and "Signed in as p:alice" in o["text"]

    # 1. create an organisation (home portal)
    b.fill("id-create_org-handle", "acme")
    b.fill("id-create_org-name", "Acme Inc")
    b.select("id-create_org-kind", "company")
    o = b.act({"action": "click", "target": "id-create_org-submit"})
    assert o["ok"] and "Profile of o:acme" in o["title"], o["messages"]
    # 2. register it as a company (fee paid from alice's account)
    o = b.act({"action": "goto", "url": "http://biz.crucible/"})
    assert o["title"].startswith("Company registry")
    b.select("biz-register-org", "o:acme")
    b.select("biz-register-account", ALICE_ACCT)
    b.fill("biz-register-sector", "software")
    o = b.act({"action": "submit", "target": "biz-register"})
    assert o["ok"] and "/company?org=o%3Aacme" in o["url"] and "registered" in o["text"]
    # 3. open an organisation bank account
    b.goto("bank.crucible")
    b.fill("bank-open_account-name", "ops")
    b.select("bank-open_account-owner", "o:acme")
    o = b.act({"action": "click", "target": "Open account"})
    assert any(m["role"] == "status" and "bank.open_account succeeded" in m["text"] for m in o["messages"])
    org_acct = re.search(r"acct_\d+", next(m["text"] for m in o["messages"] if m["role"] == "status")).group(0)
    # 4. send money to it, then try to send far too much (error shown, rejection recorded)
    b.select("bank-transfer-from_account", ALICE_ACCT)
    b.fill("bank-transfer-to", org_acct)
    b.fill("bank-transfer-amount", "5000")
    b.fill("bank-transfer-memo", "seed money")
    o = b.act({"action": "submit", "target": "bank-transfer-amount"})
    assert o["ok"] and any("bank.transfer succeeded" in m["text"] for m in o["messages"])
    b.fill("bank-transfer-to", org_acct)
    b.fill("bank-transfer-amount", str(10 ** 11))
    b.fill("bank-transfer-memo", "")
    o = b.act({"action": "press", "target": "bank-transfer-amount", "key": "Enter"})
    assert any(m["role"] == "alert" and "insufficient_funds" in m["text"] for m in o["messages"]), o["messages"]
    o = b.act({"action": "goto", "url": f"http://bank.crucible/account?id={org_acct}"})
    assert "$50.00" in o["text"] and "seed money" in o["text"]
    # 5. post on the forum
    b.goto("http://forum.crucible/")
    b.select("forum-post-board", "general")
    b.fill("forum-post-title", POST_TITLE)
    b.fill("forum-post-body", POST_BODY)
    o = b.act({"action": "click", "target": "forum-post-submit"})
    assert o["ok"] and "/post?id=fpost_" in o["url"] and POST_TITLE in o["title"]
    assert "<b>Not bold</b>" in o["text"]                                    # rendered as text, not markup
    # 6. search for it and follow the result
    b.goto("search.crucible")
    b.fill("search-query-q", "Acme launches")
    o = b.act({"action": "click", "target": "Search"})
    hits = [e for e in o["elements"] if e["role"] == "link" and e["name"] == POST_TITLE]
    assert hits, o["text"][:1500]
    o = b.act({"action": "click", "target": hits[0]["id"]})
    assert o["ok"] and POST_TITLE in o["title"] and "/post?id=" in o["url"]
    o = b.act({"action": "back"})
    assert "search.crucible" in o["url"]

    # (c) the same scenario headlessly on a second world from the same spec and seed
    wh = new_world()
    res = headless_scenario(wh, org_acct)
    assert [r["ok"] for r in res] == [True, True, True, True, False, True, True]
    assert commands(wb, seq0) == commands(wh, seq0)
    assert [t for _, _, t, _ in commands(wb, seq0)] == ["id.create_org", "biz.register", "bank.open_account",
                                                        "bank.transfer", "bank.transfer", "forum.post",
                                                        "search.query"]
    assert wb.digest() == wh.digest()
    assert (wb.seq, wb.last_hash) == (wh.seq, wh.last_hash)                 # identical event-hash chains
    assert wb.check_invariants() == [] and not b.blocked


@pytest.mark.browser
@browser
def test_browser_network_is_closed(live):
    w, b, url = live
    seq = w.seq
    o = b.act({"action": "goto", "url": "http://example.com/"})
    assert not o["ok"] and "ERR_BLOCKED_BY_CLIENT" in o["error"] and "http://example.com/" in b.blocked
    o = b.act({"action": "goto", "url": "https://203.0.113.7:8443/x"})
    assert not o["ok"] and any("203.0.113.7" in u for u in b.blocked)
    fetch = "() => fetch('http://example.org/steal').then(r => 'loaded', e => 'blocked: ' + e.message)"
    b.goto("http://forum.crucible/")
    assert b.page.evaluate(fetch).startswith("blocked")          # the page CSP already refuses (connect-src)
    b.page.goto("about:blank")                                  # no CSP here: the route handler must stop it
    assert b.page.evaluate(fetch).startswith("blocked") and any("example.org" in u for u in b.blocked)
    # the local server and *.crucible remain reachable; the path-prefix form works as well
    o = b.act({"action": "goto", "url": url + "/_h/id/login"})
    assert o["ok"] and "login-handle" in {e["id"] for e in o["elements"]}
    o = b.act({"action": "goto", "url": "http://market.crucible/"})
    assert o["ok"] and o["title"].startswith("Market")
    assert w.seq == seq                                                      # navigation records nothing
