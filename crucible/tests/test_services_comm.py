"""Communication / information services: search, ads, forum, mail, docs, calendar, issues, support."""
from __future__ import annotations

import json

import pytest

import crucible.services.admin  # noqa: F401  (registers modules)
import crucible.services.bank  # noqa: F401
import crucible.services.identity  # noqa: F401
from crucible.core import schema as S
from crucible.core.errors import CrucibleError, InvalidAction
from crucible.core.kernel import MODULES, Module, World, WorldSpec, endpoint, register_module
from crucible.services import ads as ads_mod  # noqa: F401
from crucible.services import calendar as cal_mod  # noqa: F401
from crucible.services import docs as docs_mod  # noqa: F401
from crucible.services import forum as forum_mod  # noqa: F401
from crucible.services import issues as issues_mod  # noqa: F401
from crucible.services import mail as mail_mod  # noqa: F401
from crucible.services import search as search_mod
from crucible.services import support as support_mod  # noqa: F401

DAY = 86400
COMM = ["search", "ads", "forum", "mail", "docs", "cal", "issues", "support"]


@register_module
class ProbeComm(Module):
    """Test-only module: installs deterministic is_host / host_owner hooks from its config (so that replayed
    and regenerated worlds have them too) and exposes probes for transaction rollback and internal APIs."""
    name = "probe_comm"
    host = "probe"
    depends = ("search",)

    def install(self, world):
        hosts = frozenset(self.config.get("hosts", ()))
        owners = dict(self.config.get("owners", {}))
        if hosts:
            world.hooks["is_host"] = lambda st, name: name in hosts
        if owners:
            world.hooks["host_owner"] = lambda st, host: owners.get(host)

    @endpoint("index_then_fail", S.obj({"word": S.s_str(maxLength=40), "query_first": S.s_bool()}, ["word"]))
    def ep_index_then_fail(self, ctx, word, query_first=False):
        search_mod.index(ctx, f"probe_{word}", "doc", word, f"{word} probe text", ctx.actor)
        if query_first and not search_mod.query(ctx.world, word):
            raise RuntimeError("index not visible inside its own transaction")
        raise InvalidAction("probe failure after indexing")

    @endpoint("nested", S.obj({"word": S.s_str(maxLength=40)}, ["word"]))
    def ep_nested(self, ctx, word):
        try:
            ctx.call("probe.index_then_fail", {"word": word, "query_first": True})
        except CrucibleError:
            pass
        return {"n": len(search_mod.query(ctx.world, word))}

    @endpoint("ticket", S.obj({"host": S.s_str(maxLength=40), "owner": S.s_str(maxLength=40),
                               "customer": S.s_str(maxLength=40), "subject": S.s_str(maxLength=200),
                               "severity": S.s_int(minimum=1, maximum=4)}, ["host", "owner", "customer", "subject"]))
    def ep_ticket(self, ctx, host, owner, customer, subject, severity=1):
        return {"ticket": ctx.world.modules["support"].open_ticket(ctx, host, owner, customer, subject,
                                                                   "simulated failure report", severity)}

    @endpoint("npc_mail", S.obj({"to": S.s_str(maxLength=40), "body": S.s_str(maxLength=400)}, ["to", "body"]))
    def ep_npc_mail(self, ctx, to, body):
        return {"msg": ctx.world.modules["mail"].send(ctx, "sys", [to], "Notice", body)}


# ---------------------------------------------------------------------- helpers
def make_world(params=None, probe=None, handles=("alice", "bob", "carol"), endowment=100_000, ads_cfg=None):
    mods = [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}]
    for m in COMM:
        mods.append({"name": m, "config": ads_cfg or {}} if m == "ads" else {"name": m})
    mods.append({"name": "probe_comm", "config": probe or {}})
    spec = WorldSpec("comm-test", 11, mods, params or {})
    w = World.create(spec, strict=True)
    accts = {}
    for h in handles:
        accts[h] = w.system_call("admin.join", {"handle": h, "endowment": endowment})["account"]
    return w, accts


def act(w, actor, tool, **args):
    return w.execute(actor, {"tool": tool, "args": args})


def ok(w, actor, tool, **args):
    r = act(w, actor, tool, **args)
    assert r.ok, (tool, r.error)
    return r.result


def fails(w, actor, tool, code, **args):
    r = act(w, actor, tool, **args)
    assert not r.ok, (tool, r.result)
    assert r.error["code"] == code, (tool, r.error)
    return r.error


def make_org(w, owner="p:alice", handle="acme", members=("p:bob",)):
    org = ok(w, owner, "id.create_org", handle=handle, name=handle.title(), kind="company")["org"]
    for m in members:
        inv = ok(w, owner, "id.invite", org=handle, person=m)["invite"]
        ok(w, m, "id.respond_invite", invite=inv, accept=True)
    return org


def visible_payloads(w, actor) -> str:
    """Everything an actor can observe in the event stream, minus the echo of its own commands' arguments."""
    evs, _ = w.visible_events(actor, 0, limit=10**6)
    return json.dumps([e.payload for e in evs if not (e.type in ("cmd", "cmd.rejected") and e.actor == actor)])


def clean(w):
    errs = w.check_invariants()
    assert errs == [], errs


def balances_sum_zero(w):
    assert sum(a["balance"] for a in w.state.tables["accounts"].values()) == 0


# ---------------------------------------------------------------------- search
def test_search_bm25_ranking_kind_filter_browse_and_signal():
    w, _ = make_world()
    a = ok(w, "p:alice", "forum.post", board="dev", title="Rust compiler",
           body="A fast rust compiler for rust programs.")["post"]
    b = ok(w, "p:bob", "forum.post", board="general", title="Gardening tools", body="Rust on old tools.")["post"]
    d1 = ok(w, "p:carol", "docs.create", title="Compiler notes", body="notes about a compiler",
            visibility="public")["doc"]
    w.advance_time(100)
    d2 = ok(w, "p:carol", "docs.create", title="Compiler notes", body="notes about a compiler",
            visibility="public")["doc"]
    res = ok(w, "p:bob", "search.query", q="rust compiler")["results"]
    assert res[0]["id"] == a and res[0]["kind"] == "post" and not res[0]["sponsored"]
    assert b in [r["id"] for r in res]
    # kind filter + recency tie-break between identical documents
    docs = ok(w, "p:bob", "search.query", q="compiler", kind="doc")["results"]
    assert [r["id"] for r in docs] == [d2, d1]
    assert all(r["kind"] == "doc" for r in docs)
    # empty query browses most recent
    assert ok(w, "p:bob", "search.query", q="", kind="post")["results"][0]["id"] == b
    # the pure function equals the endpoint and a from-scratch rebuild of the cache
    pure = search_mod.query(w, "rust compiler")
    assert [r["id"] for r in pure] == [r["id"] for r in res]
    w.runtime.pop(search_mod.RUNTIME_KEY)
    assert search_mod.query(w, "rust compiler") == pure
    # rank signal multiplies relevance; <= 0 delists
    w.hooks["rank_signal"] = lambda world, rec: 0.0 if rec["owner"] == "p:alice" else 1.0
    assert a not in [r["id"] for r in search_mod.query(w, "rust compiler")]
    w.hooks["rank_signal"] = lambda world, rec: 100.0 if rec["owner"] == "p:bob" else 1.0
    assert search_mod.query(w, "rust compiler")[0]["id"] == b
    del w.hooks["rank_signal"]
    fails(w, "p:bob", "search.query", "invalid_action", q="rust", limit=21)
    fails(w, "p:bob", "search.query", "invalid_action", q="x" * 201)
    fails(w, "p:bob", "search.get", "not_found", id="nope")
    assert ok(w, "p:bob", "search.get", id=a)["text"].startswith("A fast rust")
    clean(w)


def test_search_cache_survives_rollbacks():
    w, _ = make_world()
    ok(w, "p:alice", "forum.post", board="dev", title="baseline", body="baseline words")
    search_mod.query(w, "baseline")  # build the runtime cache
    fails(w, "p:alice", "probe.index_then_fail", "invalid_action", word="ghosta")
    assert search_mod.query(w, "ghosta") == []
    fails(w, "p:alice", "probe.index_then_fail", "invalid_action", word="ghostb", query_first=True)
    assert search_mod.query(w, "ghostb") == []
    assert ok(w, "p:alice", "probe.nested", word="ghostc") == {"n": 0}   # savepoint rollback inside one command
    assert search_mod.query(w, "ghostc") == []
    # cache built for the first time inside a failing transaction
    w.runtime.pop(search_mod.RUNTIME_KEY)
    fails(w, "p:alice", "probe.index_then_fail", "invalid_action", word="ghostd", query_first=True)
    assert search_mod.query(w, "ghostd") == []
    assert not w.state.tables["search_docs"].keys() & {"probe_ghosta", "probe_ghostb", "probe_ghostc",
                                                       "probe_ghostd"}
    clean(w)   # includes cache == from-scratch rebuild


def test_search_after_replay_and_fork():
    w, _ = make_world()
    ok(w, "p:alice", "forum.post", board="market", title="Selling lamps", body="bright lamps cheap lamps")
    ok(w, "p:bob", "docs.create", title="Lamp manual", body="how to fix a lamp", visibility="public")
    before = search_mod.query(w, "lamps")
    r = World.replay(w.spec, [e.to_dict() for e in w.log])
    assert r.digest() == w.digest()
    assert search_mod.query(r, "lamps") == before
    f = w.fork("fork-1")
    assert search_mod.query(f, "lamps") == before
    ok(f, "p:carol", "forum.post", board="market", title="More lamps", body="lamps lamps lamps")
    assert len(search_mod.query(f, "lamps")) == len(before) + 1
    assert search_mod.query(w, "lamps") == before
    clean(f)
    clean(w)


def test_search_discovery_learns_hosts():
    w, _ = make_world()
    w.hooks["is_host"] = lambda st, name: name == "widgets"
    pid = ok(w, "p:alice", "forum.post", board="market", title="Review",
             body="widgets.crucible is great, nothere.crucible does not exist")["post"]
    assert w.state.tables["forum_posts"][pid]["hosts"] == ["widgets"]
    assert w.state.tables["search_docs"][pid]["hosts"] == ["widgets"]
    assert not w.knows("p:bob", "host", "widgets")
    res = ok(w, "p:bob", "search.query", q="review")["results"]
    assert res and res[0]["hosts"] == ["widgets"]
    assert w.knows("p:bob", "host", "widgets")
    assert not w.knows("p:bob", "host", "nothere")
    # reading the post also teaches the host; so does a public doc read
    assert not w.knows("p:carol", "host", "widgets")
    ok(w, "p:carol", "forum.read", post=pid)
    assert w.knows("p:carol", "host", "widgets")
    doc = ok(w, "p:alice", "docs.create", title="Guide", body="use widgets.crucible", visibility="public")["doc"]
    w.system_call("admin.join", {"handle": "dave"})
    ok(w, "p:dave", "docs.get", doc=doc)
    assert w.knows("p:dave", "host", "widgets")
    clean(w)


# ---------------------------------------------------------------------- forum
def test_forum_endpoints_votes_follow_feed_delete():
    w, _ = make_world()
    boards = {b["board"] for b in ok(w, "p:alice", "forum.boards")}
    assert boards == {"general", "dev", "market", "jobs", "help"}
    fails(w, "p:alice", "forum.post", "not_found", board="nosuch", title="t", body="b")
    fails(w, "p:alice", "forum.post", "invalid_action", board="dev", title="", body="b")
    p = ok(w, "p:alice", "forum.post", board="dev", title="My tool", body="It parses logs", tags=["tools"])["post"]
    r = ok(w, "p:bob", "forum.reply", post=p, body="Nice, it works for me")["reply"]
    rd = ok(w, "p:carol", "forum.read", post=p)
    assert rd["n_replies"] == 1 and rd["replies"][0]["reply"] == r
    assert r in [x["id"] for x in search_mod.query(w, "works for me")]
    # votes
    fails(w, "p:alice", "forum.vote", "forbidden", post=p, value=1)
    assert ok(w, "p:bob", "forum.vote", post=p, value=1)["score"] == 1
    fails(w, "p:bob", "forum.vote", "conflict", post=p, value=1)
    assert ok(w, "p:bob", "forum.vote", post=p, value=-1)["score"] == -1
    assert ok(w, "p:carol", "forum.vote", post=p, value=1)["score"] == 0
    fails(w, "p:carol", "forum.vote", "invalid_action", post=p, value=2)
    fails(w, "p:carol", "forum.vote", "not_found", post="fpost_999999", value=1)
    # org posts: members cannot vote on them, posting as org needs forum.post (owner/admin)
    make_org(w)
    fails(w, "p:bob", "forum.post", "forbidden", board="market", title="Acme", body="buy", as_org="acme")
    op = ok(w, "p:alice", "forum.post", board="market", title="Acme launch", body="buy", as_org="acme")["post"]
    assert w.state.tables["forum_posts"][op]["author"] == "o:acme"
    fails(w, "p:bob", "forum.vote", "forbidden", post=op, value=1)
    ok(w, "p:carol", "forum.vote", post=op, value=1)
    # follow + feed
    ok(w, "p:carol", "forum.follow", target="alice")
    fails(w, "p:carol", "forum.follow", "conflict", target="alice")
    fails(w, "p:carol", "forum.follow", "invalid_action", target="p:carol")
    ok(w, "p:carol", "forum.follow", target="board:jobs")
    w.advance_time(3600)
    j = ok(w, "p:bob", "forum.post", board="jobs", title="Hiring", body="We hire")["post"]
    feed = ok(w, "p:carol", "forum.feed", limit=30)
    ids = [x["post"] for x in feed]
    assert p in ids and j in ids
    assert next(x for x in feed if x["post"] == p)["followed"] is True
    fails(w, "p:carol", "forum.feed", "invalid_action", limit=31)
    ok(w, "p:carol", "forum.unfollow", target="alice")
    fails(w, "p:carol", "forum.unfollow", "not_found", target="alice")
    lst = ok(w, "p:carol", "forum.list", board="dev", sort="top")
    assert lst[0]["post"] == p
    # delete
    fails(w, "p:bob", "forum.delete", "forbidden", post=p)
    ok(w, "p:alice", "forum.delete", post=p)
    fails(w, "p:carol", "forum.read", "not_found", post=p)
    assert p not in w.state.tables["search_docs"] and r not in w.state.tables["search_docs"]
    assert w.observe("p:alice")["forum"]["my_posts"] == 1
    clean(w)


# ---------------------------------------------------------------------- mail
def test_mail_delay_read_hidden_info_and_hook():
    w, _ = make_world(probe={"hosts": ["widgets"]})
    delivered = []
    w.hooks["on_mail_delivered"] = lambda ctx, msg, to: delivered.append((msg, to, ctx.now))
    secret = "the launch code is quokka"
    res = ok(w, "p:alice", "mail.send", to=["bob"], subject="Plan", body=secret + " see widgets.crucible")
    mid = res["msg"]
    eta = res["eta"]["p:bob"]
    assert 30 <= eta <= 90
    assert ok(w, "p:bob", "mail.inbox") == []
    fails(w, "p:bob", "mail.read", "not_found", msg=mid)
    w.advance_time(eta - 1)
    assert ok(w, "p:bob", "mail.inbox") == [] and delivered == []
    w.advance_time(eta)
    assert delivered == [(mid, "p:bob", eta)]
    inbox = ok(w, "p:bob", "mail.inbox", unread_only=True)
    assert [m["msg"] for m in inbox] == [mid] and inbox[0]["from"] == "p:alice"
    assert w.observe("p:bob")["mail"] == {"unread": 1}
    # hidden information: carol can neither read it nor see it in her observable events
    fails(w, "p:carol", "mail.read", "not_found", msg=mid)
    assert ok(w, "p:carol", "mail.search", q="quokka") == []
    assert "quokka" not in visible_payloads(w, "p:carol")
    assert "quokka" not in visible_payloads(w, "p:bob")    # the recipient's notification carries no body
    assert not w.knows("p:bob", "host", "widgets")
    m = ok(w, "p:bob", "mail.read", msg=mid)
    assert m["body"].startswith(secret) and m["delivered"] == eta
    assert w.knows("p:bob", "host", "widgets")
    assert w.observe("p:bob")["mail"] == {"unread": 0}
    assert ok(w, "p:bob", "mail.search", q="launch quokka")[0]["msg"] == mid
    sent = ok(w, "p:alice", "mail.read", msg=mid)
    assert sent["folder"] == "sent" and sent["delivery"]["p:bob"]["status"] == "delivered"
    # reply threading, schema limits, unknown recipients, deletion
    ok(w, "p:bob", "mail.send", to=["alice"], subject="Re: Plan", body="ok", in_reply_to=mid)
    fails(w, "p:carol", "mail.send", "not_found", to=["alice"], subject="x", body="y", in_reply_to=mid)
    fails(w, "p:alice", "mail.send", "not_found", to=["nobody"], subject="x", body="y")
    fails(w, "p:alice", "mail.send", "invalid_action", to=[f"u{i}" for i in range(21)], subject="x", body="y")
    fails(w, "p:alice", "mail.send", "invalid_action", to=[], subject="x", body="y")
    ok(w, "p:bob", "mail.delete", msg=mid)
    fails(w, "p:bob", "mail.read", "not_found", msg=mid)
    w.advance_time(eta + 200)
    clean(w)


def test_mail_bounces():
    w, _ = make_world(params={"mail_failure_p": 1.0, "mail_latency_s": 10})
    mid = ok(w, "p:alice", "mail.send", to=["bob", "carol"], subject="Hello", body="hi")["msg"]
    w.advance_time(100)
    assert ok(w, "p:bob", "mail.inbox") == []
    notices = ok(w, "p:alice", "mail.inbox")
    assert len(notices) == 2 and all(n["kind"] == "bounce" and n["subject"] == "Undeliverable: Hello"
                                     for n in notices)
    note = ok(w, "p:alice", "mail.read", msg=notices[0]["msg"])
    assert "transient_failure" in note["body"] and note["in_reply_to"] == mid
    status = ok(w, "p:alice", "mail.read", msg=mid)["delivery"]
    assert {s["status"] for s in status.values()} == {"bounced"}
    clean(w)
    # inactive recipient bounces even without random failures
    w2, _ = make_world(params={"mail_latency_s": 5})
    mid = ok(w2, "p:alice", "mail.send", to=["bob"], subject="S", body="B")["msg"]
    w2.system_call("admin.set_status", {"principal": "p:bob", "status": "suspended"})
    w2.advance_time(50)
    m = w2.state.tables["mail_msgs"][mid]
    assert m["status"]["p:bob"] == "bounced"
    assert "recipient_inactive" in ok(w2, "p:alice", "mail.read", msg=ok(w2, "p:alice", "mail.inbox")[0]["msg"])["body"]
    clean(w2)


def test_mail_org_mailbox_and_system_sender():
    w, _ = make_world(params={"mail_latency_s": 0})
    make_org(w)      # alice owner, bob member
    mid = ok(w, "p:carol", "mail.send", to=["acme"], subject="Order", body="I want 3 widgets")["msg"]
    w.advance_time(1)
    box = ok(w, "p:alice", "mail.inbox")
    assert box[0]["msg"] == mid and box[0]["mailbox"] == "o:acme"
    assert ok(w, "p:alice", "mail.read", msg=mid)["body"] == "I want 3 widgets"
    fails(w, "p:bob", "mail.read", "not_found", msg=mid)                       # plain member: no org mail
    fails(w, "p:bob", "mail.inbox", "forbidden", mailbox="acme")
    fails(w, "p:bob", "mail.send", "forbidden", to=["carol"], subject="s", body="b", from_org="acme")
    ok(w, "p:alice", "mail.send", to=["carol"], subject="Re: Order", body="shipped", from_org="acme")
    ok(w, "p:alice", "id.grant", grantee="bob", cap="mail.read", owner="acme")
    w.advance_time(2)
    assert ok(w, "p:bob", "mail.inbox", mailbox="acme")[0]["msg"] == mid
    assert ok(w, "p:carol", "mail.inbox")[0]["from"] == "o:acme"
    # internal API: a simulated actor (system) sends mail
    sysmsg = w.system_call("probe.npc_mail", {"to": "p:carol", "body": "your invoice is due"})["msg"]
    w.advance_time(3)
    assert ok(w, "p:carol", "mail.read", msg=sysmsg)["from"] == "sys"
    clean(w)


# ---------------------------------------------------------------------- docs
def test_docs_visibility_shares_versions_and_search():
    w, _ = make_world()
    d = ok(w, "p:alice", "docs.create", title="Secret plan", body="zyxwv strategy")["doc"]
    fails(w, "p:bob", "docs.get", "not_found", doc=d)
    assert search_mod.query(w, "zyxwv") == [] and d not in w.state.tables["search_docs"]
    assert ok(w, "p:bob", "search.query", q="zyxwv strategy")["results"] == []
    assert d not in [x["doc"] for x in ok(w, "p:bob", "docs.list", scope="public")]
    assert "zyxwv" not in visible_payloads(w, "p:bob")
    fails(w, "p:bob", "docs.share", "not_found", doc=d, principal="bob", perm="read")
    ok(w, "p:alice", "docs.share", doc=d, principal="bob", perm="read")
    assert ok(w, "p:bob", "docs.get", doc=d)["access"] == "read"
    fails(w, "p:bob", "docs.edit", "forbidden", doc=d, body="hacked")
    fails(w, "p:bob", "docs.share", "forbidden", doc=d, principal="carol", perm="read")
    ok(w, "p:alice", "docs.share", doc=d, principal="bob", perm="write")
    assert ok(w, "p:bob", "docs.edit", doc=d, body="zyxwv strategy v2", base_version=1)["version"] == 2
    fails(w, "p:alice", "docs.edit", "conflict", doc=d, body="stale", base_version=1)
    assert ok(w, "p:alice", "docs.get", doc=d, version=1)["body"] == "zyxwv strategy"
    hist = ok(w, "p:alice", "docs.history", doc=d)
    assert [h["version"] for h in hist] == [2, 1] and hist[0]["author"] == "p:bob"
    assert [x["doc"] for x in ok(w, "p:bob", "docs.list", scope="shared")] == [d]
    fails(w, "p:carol", "docs.get", "not_found", doc=d)
    # publish -> indexed; edit -> reindexed; unpublish -> removed
    ok(w, "p:alice", "docs.set_visibility", doc=d, visibility="public")
    assert ok(w, "p:carol", "docs.get", doc=d)["access"] == "read"
    assert [x["id"] for x in search_mod.query(w, "zyxwv")] == [d]
    ok(w, "p:alice", "docs.edit", doc=d, body="new words qwertz")
    assert search_mod.query(w, "zyxwv") == [] and search_mod.query(w, "qwertz")[0]["id"] == d
    ok(w, "p:alice", "docs.set_visibility", doc=d, visibility="private")
    assert search_mod.query(w, "qwertz") == []
    ok(w, "p:alice", "docs.share", doc=d, principal="bob", perm="none")
    fails(w, "p:bob", "docs.get", "not_found", doc=d)
    # organisation documents
    fails(w, "p:alice", "docs.create", "invalid_action", title="t", body="b", visibility="org")
    make_org(w)
    od = ok(w, "p:alice", "docs.create", title="Handbook", body="team rules", visibility="org", org="acme")["doc"]
    pd = ok(w, "p:alice", "docs.create", title="Board minutes", body="private", org="acme")["doc"]
    assert ok(w, "p:bob", "docs.get", doc=od)["access"] == "write"
    fails(w, "p:bob", "docs.get", "not_found", doc=pd)
    fails(w, "p:carol", "docs.get", "not_found", doc=od)
    fails(w, "p:carol", "docs.create", "forbidden", title="t", body="b", org="acme")
    assert {x["doc"] for x in ok(w, "p:bob", "docs.list", scope="org", org="acme")} == {od}
    # explicit identity grant to an outsider
    ok(w, "p:alice", "id.grant", grantee="carol", cap="docs.read", owner="acme")
    assert ok(w, "p:carol", "docs.get", doc=pd)["access"] == "read"
    assert w.observe("p:alice")["docs"]["docs"] == 1
    clean(w)


# ---------------------------------------------------------------------- calendar
def test_calendar_invites_conflicts_agenda():
    w, _ = make_world()
    invited = []
    w.hooks["on_calendar_invite"] = lambda ctx, eid, a: invited.append((eid, a))
    fails(w, "p:alice", "cal.create", "invalid_action", title="late", start=0, duration=10)
    w.advance_time(1000)
    fails(w, "p:alice", "cal.create", "invalid_action", title="past", start=10, duration=3600)
    e1 = ok(w, "p:alice", "cal.create", title="Kickoff", start=5000, duration=3600, attendees=["bob", "carol"])["event"]
    assert invited == [(e1, "p:bob"), (e1, "p:carol")]
    ag = ok(w, "p:bob", "cal.agenda")
    assert ag[0]["event"] == e1 and ag[0]["response"] == "pending"
    assert w.observe("p:bob")["cal"]["pending_invites"] == 1
    ok(w, "p:bob", "cal.respond", event=e1, accept=True)
    fails(w, "p:bob", "cal.respond", "conflict", event=e1, accept=True)
    e2 = ok(w, "p:carol", "cal.create", title="Overlap", start=6000, duration=3600, attendees=["bob"])["event"]
    err = fails(w, "p:bob", "cal.respond", "conflict", event=e2, accept=True)
    assert err["details"]["conflicts"] == [e1]
    ok(w, "p:bob", "cal.respond", event=e2, accept=True, allow_conflict=True)
    ag = {a["event"]: a for a in ok(w, "p:bob", "cal.agenda")}
    assert ag[e1]["conflicts"] == [e2] and ag[e2]["conflicts"] == [e1]
    # creating an event that overlaps one's own accepted events
    fails(w, "p:bob", "cal.create", "conflict", title="Mine", start=5500, duration=60)
    fails(w, "p:alice", "cal.respond", "not_found", event=e2, accept=True)
    assert ok(w, "p:carol", "cal.get", event=e1)["responses"] == {"p:alice": "accepted", "p:bob": "accepted",
                                                                  "p:carol": "pending"}
    ok(w, "p:carol", "cal.respond", event=e1, accept=False)
    fails(w, "p:carol", "cal.agenda", "invalid_action", from_t=0, to_t=100 * DAY)
    assert e1 not in [a["event"] for a in ok(w, "p:carol", "cal.agenda")]
    # privacy and cancellation
    e3 = ok(w, "p:alice", "cal.create", title="1:1", start=20000, duration=1800, attendees=["bob"])["event"]
    fails(w, "p:carol", "cal.get", "not_found", event=e3)
    fails(w, "p:bob", "cal.cancel", "forbidden", event=e3)
    ok(w, "p:alice", "cal.cancel", event=e3)
    fails(w, "p:bob", "cal.respond", "invalid_action", event=e3, accept=True)
    assert "1:1" not in visible_payloads(w, "p:carol")
    # organisation events need cal.write (members have it)
    make_org(w)
    ok(w, "p:bob", "cal.create", title="Acme standup", start=30000, duration=900, org="acme")
    fails(w, "p:carol", "cal.create", "forbidden", title="x", start=30000, duration=900, org="acme")
    clean(w)


def test_calendar_alias_module_name():
    assert MODULES["calendar"] is MODULES["cal"]
    spec = WorldSpec("alias", 1, [{"name": "identity"}, {"name": "calendar"}])
    w = World.create(spec, strict=True)
    assert "cal" in w.modules and "cal.create" in w.endpoints


# ---------------------------------------------------------------------- issues
def test_issues_projects_permissions_and_flow():
    w, _ = make_world()
    assigned = []
    w.hooks["on_issue_assigned"] = lambda ctx, iid, a: assigned.append((iid, a))
    make_org(w)
    priv = ok(w, "p:bob", "issues.create_project", name="backend", owner="acme", repo="acme/backend")["project"]
    fails(w, "p:bob", "issues.create_project", "conflict", name="backend", owner="acme")
    fails(w, "p:carol", "issues.create_project", "forbidden", name="evil", owner="acme")
    i1 = ok(w, "p:bob", "issues.create", project=priv, title="Crash on start", labels=["bug"], assignee="alice")["issue"]
    assert assigned == [(i1, "p:alice")]
    fails(w, "p:carol", "issues.get", "not_found", issue=i1)
    fails(w, "p:carol", "issues.create", "not_found", project=priv, title="x")
    assert ok(w, "p:carol", "issues.projects", owner="acme") == []
    ok(w, "p:alice", "issues.update", issue=i1, status="in_progress")
    ok(w, "p:alice", "issues.comment", issue=i1, body="looking into it")
    assert ok(w, "p:alice", "issues.assigned")[0]["issue"] == i1
    assert w.observe("p:alice")["issues"] == {"assigned_open": 1}
    ok(w, "p:alice", "issues.close", issue=i1, comment="fixed in abc123")
    fails(w, "p:alice", "issues.close", "conflict", issue=i1)
    got = ok(w, "p:bob", "issues.get", issue=i1)
    assert got["status"] == "closed" and len(got["comments"]) == 2
    ok(w, "p:bob", "issues.update", issue=i1, status="open")           # reopen
    assert w.state.tables["iss_projects"][priv]["open"] == 1
    # public project: outsiders file and comment, but cannot triage
    pub = ok(w, "p:alice", "issues.create_project", name="sdk", owner="acme", public=True)["project"]
    i2 = ok(w, "p:carol", "issues.create", project=pub, title="Docs typo", body="page 3")["issue"]
    fails(w, "p:carol", "issues.create", "forbidden", project=pub, title="x", assignee="carol")
    ok(w, "p:carol", "issues.comment", issue=i2, body="still there")
    fails(w, "p:carol", "issues.assign", "forbidden", issue=i2, assignee="carol")
    fails(w, "p:carol", "issues.update", "forbidden", issue=i2, labels=["wontfix"])
    ok(w, "p:carol", "issues.update", issue=i2, title="Docs typo on page 3")
    ok(w, "p:bob", "issues.assign", issue=i2, assignee="bob")
    assert assigned[-1] == (i2, "p:bob")
    fails(w, "p:bob", "issues.assign", "conflict", issue=i2, assignee="bob")
    ok(w, "p:carol", "issues.close", issue=i2)                          # creator closes own issue
    assert [x["issue"] for x in ok(w, "p:carol", "issues.list", project=pub, status="closed")] == [i2]
    assert ok(w, "p:carol", "issues.list", project=pub) == []
    fails(w, "p:bob", "issues.list", "invalid_action", project=pub, limit=51)
    clean(w)


# ---------------------------------------------------------------------- support
def test_support_tickets_roles_hook_and_daily_stats():
    w, _ = make_world(probe={"hosts": ["widgets"], "owners": {"widgets": "o:acme"}})
    resolved = []
    w.hooks["on_ticket_resolved"] = lambda ctx, tid: resolved.append(tid)
    make_org(w)
    fails(w, "p:carol", "support.open", "not_found", host="nosuch", subject="help")
    t1 = ok(w, "p:carol", "support.open", host="widgets", subject="API down", body="500 errors", severity=4)["ticket"]
    t2 = w.system_call("probe.ticket", {"host": "widgets", "owner": "o:acme", "customer": "seg:12",
                                        "subject": "slow responses", "severity": 2})["ticket"]
    q = ok(w, "p:bob", "support.list", owner="acme")
    assert [x["ticket"] for x in q] == [t1, t2]          # most severe first
    fails(w, "p:carol", "support.list", "forbidden", owner="acme")
    assert [x["ticket"] for x in ok(w, "p:carol", "support.list")] == [t1]
    assert w.observe("p:bob")["support"]["open_tickets"] == 2
    w.advance_time(600)
    ok(w, "p:bob", "support.reply", ticket=t1, body="we are on it")
    tk = ok(w, "p:carol", "support.get", ticket=t1)
    assert tk["status"] == "answered" and tk["first_response"] == 600 and tk["your_role"] == "customer"
    assert w.observe("p:carol")["support"]["answered_for_you"] == 1
    ok(w, "p:carol", "support.reply", ticket=t1, body="still failing")
    assert ok(w, "p:bob", "support.get", ticket=t1)["status"] == "open"
    fails(w, "p:carol", "support.resolve", "forbidden", ticket=t1)
    w.system_call("admin.join", {"handle": "dave"})
    fails(w, "p:dave", "support.get", "not_found", ticket=t1)
    fails(w, "p:dave", "support.get", "not_found", ticket=t2)
    assert "500 errors" not in visible_payloads(w, "p:dave")
    ok(w, "p:bob", "support.resolve", ticket=t1, note="deployed fix")
    assert resolved == [t1]
    fails(w, "p:bob", "support.resolve", "conflict", ticket=t1)
    # daily SLA metrics
    w.advance_time(DAY)
    st = ok(w, "p:dave", "support.stats", owner="acme")
    assert st["live"] == {"open": 1, "answered": 0, "resolved": 1, "total": 2}
    day = st["daily"][0]
    assert day["day"] == 1 and day["unresolved"] == 1 and day["median_age_s"] == DAY
    assert day["opened"] == 2 and day["resolved"] == 1 and day["mean_first_response_s"] == 600
    assert w.modules["support"].stats(w.state, "o:acme")["day"] == 1
    w.advance_time(2 * DAY)
    assert ok(w, "p:dave", "support.stats", owner="acme")["daily"][0]["opened"] == 0
    clean(w)


# ---------------------------------------------------------------------- ads
def test_ads_billing_budget_reset_and_pausing():
    w, accts = make_world(probe={"hosts": ["widgets"]})
    fails(w, "p:bob", "ads.create_campaign", "forbidden", account=accts["alice"], keywords=["widgets"], bid_cents=100,
          daily_budget_cents=250, title="Widgets!")
    fails(w, "p:alice", "ads.create_campaign", "invalid_action", account=accts["alice"], keywords=["widgets"],
          bid_cents=100, daily_budget_cents=50, title="Widgets!")
    fails(w, "p:alice", "ads.create_campaign", "invalid_action", account=accts["alice"], keywords=["widgets"],
          bid_cents=100, daily_budget_cents=500, title="Widgets!", hosts=["unknownhost"])
    cid = ok(w, "p:alice", "ads.create_campaign", account=accts["alice"], keywords=["Widgets", "cheap gadgets"],
             bid_cents=100, daily_budget_cents=250, title="Widgets!", text="best widgets",
             hosts=["widgets"])["campaign"]
    assert w.state.tables["ads_campaigns"][cid]["keywords"] == ["cheap", "gadgets", "widgets"]
    ok(w, "p:carol", "forum.post", board="market", title="widgets review", body="widgets are ok")
    bal = lambda a: w.state.tables["accounts"][a]["balance"]  # noqa: E731
    fees0 = bal("acct_fees")
    res = ok(w, "p:bob", "search.query", q="buy widgets")["results"]
    assert res[0]["sponsored"] is True and res[0]["campaign"] == cid and res[0]["title"] == "Widgets!"
    assert not res[1]["sponsored"]
    assert w.knows("p:bob", "host", "widgets")
    assert bal(accts["alice"]) == 100_000 - 100 and bal("acct_fees") == fees0 + 100
    balances_sum_zero(w)
    assert ok(w, "p:bob", "search.query", q="widgets")["results"][0]["sponsored"]
    # budget: 2 impressions = 200 of 250; the third would exceed it
    assert not ok(w, "p:bob", "search.query", q="widgets")["results"][0]["sponsored"]
    s = ok(w, "p:alice", "ads.stats")[0]
    assert (s["impressions"], s["spent_today"], s["total_spent"]) == (2, 200, 200)
    fails(w, "p:bob", "ads.stats", "forbidden", campaign=cid)
    # no match for other words or other kinds
    assert not any(r["sponsored"] for r in search_mod.query(w, "lamps"))
    w.advance_time(DAY)
    assert w.state.tables["ads_campaigns"][cid]["spent_today"] == 0
    assert ok(w, "p:carol", "search.query", q="cheap stuff")["results"][0]["sponsored"]
    # manual pause/resume, authorisation
    fails(w, "p:bob", "ads.pause", "forbidden", campaign=cid)
    ok(w, "p:alice", "ads.pause", campaign=cid)
    assert not any(r["sponsored"] for r in ok(w, "p:bob", "search.query", q="widgets")["results"])
    ok(w, "p:alice", "ads.resume", campaign=cid)
    # insufficient funds: the impression is not served and the campaign pauses itself
    ok(w, "p:alice", "bank.transfer", from_account=accts["alice"], to="carol", amount=bal(accts["alice"]) - 50)
    spent = w.state.tables["ads_campaigns"][cid]["total_spent"]
    res = ok(w, "p:bob", "search.query", q="widgets")["results"]
    assert not any(r["sponsored"] for r in res)
    c = w.state.tables["ads_campaigns"][cid]
    assert c["status"] == "paused" and c["reason"] == "insufficient_funds" and c["total_spent"] == spent
    fails(w, "p:alice", "ads.resume", "insufficient_funds", campaign=cid)
    balances_sum_zero(w)
    clean(w)


def test_ads_targeting_one_ad_per_advertiser_and_operator_account():
    w, accts = make_world()
    op = w.system_call("admin.join", {"handle": "adnet"})["account"]
    w2, accts2 = make_world(ads_cfg={"operator_account": "acct_000001"})
    c1 = ok(w, "p:alice", "ads.create_campaign", account=accts["alice"], keywords=["hosting"], bid_cents=10,
            daily_budget_cents=1000, title="Alice hosting", kind="service")["campaign"]
    ok(w, "p:alice", "ads.create_campaign", account=accts["alice"], keywords=["hosting"], bid_cents=20,
       daily_budget_cents=1000, title="Alice hosting 2")
    c3 = ok(w, "p:bob", "ads.create_campaign", account=accts["bob"], keywords=["hosting"], bid_cents=15,
            daily_budget_cents=1000, title="Bob hosting")["campaign"]
    sp = [r for r in ok(w, "p:carol", "search.query", q="hosting")["results"] if r["sponsored"]]
    assert [r["owner"] for r in sp] == ["p:alice", "p:bob"] and sp[1]["campaign"] == c3
    sp = [r for r in ok(w, "p:carol", "search.query", q="hosting", kind="doc")["results"] if r["sponsored"]]
    assert c1 not in [r["campaign"] for r in sp]
    assert op  # an ordinary account exists; default operator remains acct_fees
    # configurable operator account receives the charges
    a0 = w2.state.tables["accounts"]["acct_000001"]["balance"]
    ok(w2, "p:bob", "ads.create_campaign", account=accts2["bob"], keywords=["hosting"], bid_cents=7,
       daily_budget_cents=100, title="Bob")
    ok(w2, "p:carol", "search.query", q="hosting")
    assert w2.state.tables["accounts"]["acct_000001"]["balance"] == a0 + 7
    clean(w)
    clean(w2)


# ---------------------------------------------------------------------- observation
def test_observe_summaries():
    w, _ = make_world(params={"mail_latency_s": 1})
    ok(w, "p:alice", "mail.send", to=["bob"], subject="hi", body="there")
    w.advance_time(10)
    ok(w, "p:alice", "cal.create", title="Sync", start=100, duration=600, attendees=["bob"])
    o = w.observe("p:bob")
    assert o["mail"] == {"unread": 1}
    assert o["cal"]["pending_invites"] == 1
    assert "ads" not in o and "support" not in o


# ---------------------------------------------------------------------- determinism of a scripted session
def scripted_session():
    w, accts = make_world(params={"mail_failure_p": 0.3, "mail_latency_s": 45},
                          probe={"hosts": ["widgets", "lampco"], "owners": {"widgets": "o:acme"}},
                          handles=("alice", "bob", "carol", "dave"))
    make_org(w)
    oacct = ok(w, "p:alice", "bank.open_account", name="acme ops", owner="acme")["account"]
    ok(w, "p:alice", "bank.transfer", from_account=accts["alice"], to=oacct, amount=20_000)
    ok(w, "p:alice", "ads.create_campaign", account=oacct, keywords=["widgets", "api"], bid_cents=50,
       daily_budget_cents=120, title="Acme Widgets API", text="fast widgets", hosts=["widgets"])
    p = ok(w, "p:alice", "forum.post", board="market", title="Launching widgets.crucible",
           body="Our API at widgets.crucible, docs soon.", as_org="acme")["post"]
    d = ok(w, "p:bob", "docs.create", title="Widgets API reference", body="GET /widgets on widgets.crucible",
           visibility="org", org="acme")["doc"]
    proj = ok(w, "p:bob", "issues.create_project", name="widgets-api", owner="acme", public=True)["project"]
    for who, q in [("p:carol", "widgets api"), ("p:dave", "cheap widgets"), ("p:carol", "api"),
                   ("p:dave", "lamps")]:
        ok(w, who, "search.query", q=q)
    ok(w, "p:carol", "forum.read", post=p)
    ok(w, "p:carol", "forum.vote", post=p, value=1)
    ok(w, "p:dave", "forum.reply", post=p, body="tried it, lampco.crucible is better")
    fails(w, "p:bob", "forum.vote", "forbidden", post=p, value=1)
    fails(w, "p:carol", "probe.index_then_fail", "invalid_action", word="ghost", query_first=True)
    for i in range(6):
        ok(w, "p:carol", "mail.send", to=["acme", "bob"], subject=f"question {i}", body="does widgets.crucible scale?")
    t = ok(w, "p:carol", "support.open", host="widgets", subject="timeouts", severity=3)["ticket"]
    w.system_call("probe.ticket", {"host": "widgets", "owner": "o:acme", "customer": "seg:7", "subject": "errors"})
    iss = ok(w, "p:carol", "issues.create", project=proj, title="Timeouts on GET")["issue"]
    ev = ok(w, "p:alice", "cal.create", title="Incident review", start=7200, duration=1800,
            attendees=["bob"])["event"]
    w.advance_time(300)
    ok(w, "p:bob", "cal.respond", event=ev, accept=True)
    ok(w, "p:bob", "support.reply", ticket=t, body="investigating")
    ok(w, "p:bob", "issues.assign", issue=iss, assignee="bob")
    ok(w, "p:bob", "docs.edit", doc=d, body="GET /widgets and POST /widgets", base_version=1)
    ok(w, "p:alice", "docs.set_visibility", doc=d, visibility="public")
    for m in ok(w, "p:alice", "mail.inbox"):
        ok(w, "p:alice", "mail.read", msg=m["msg"])
    w.advance_time(DAY + 10)
    ok(w, "p:dave", "search.query", q="widgets reference")
    ok(w, "p:bob", "support.resolve", ticket=t)
    ok(w, "p:bob", "issues.close", issue=iss, comment="fixed")
    ok(w, "p:dave", "support.stats", owner="acme")
    w.advance_time(2 * DAY + 5)
    return w


def test_scripted_session_replay_and_regenerate():
    w = scripted_session()
    clean(w)
    kinds = {e.type for e in w.log}
    assert {"mail.delivered", "mail.bounced", "ads.impressions", "ads.reset", "support.stats", "search.index",
            "cmd.rejected"} <= kinds
    events = [e.to_dict() for e in w.log]
    r = World.replay(w.spec, events)
    assert r.digest() == w.digest()
    assert search_mod.query(r, "widgets api") == search_mod.query(w, "widgets api")
    clean(r)
    g = World.regenerate(w.spec, events)
    assert g.last_hash == w.last_hash
    assert g.digest() == w.digest()
    w2 = scripted_session()
    assert w2.last_hash == w.last_hash
