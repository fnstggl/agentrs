"""The authorized publishing adapter (spec 12.4; requirements IH-04, IH-06)."""
import functools
import hashlib
import json
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cam_live.contracts import FailureClass, Modality, ModalityExecutor, ResultStatus, WallClock
from cam_live.harness.discord import DiscordExecutor
from cam_live.harness.http_api import ApiModality, HttpApiExecutor
from cam_live.harness.publishing import (DISCORD_CHANNEL, STATIC_SITE, Destination, PublishingExecutor,
                                         aggregate_access_log)
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer
from es_support import ScopedPolicy, act, make_ctx

SCOPES = ("publish.draft", "publish.write", "publish.read", "discord.write", "discord.read", "http.read")
PAGE = "<!doctype html>\n<title>Notes</title>\n<h1>Tide tables</h1>\n<p>High water 06:12 and 18:40.</p>\n"


def pact(name, args):
    scope = {"draft": "publish.draft", "preview": "publish.draft", "update": "publish.draft", "submit": "publish.write",
             "status": "publish.read", "analytics": "publish.read"}[name]
    return act(f"publish.{name}", args, scope=scope)


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    pub = PublishingExecutor(str(tmp_path / "state"), [Destination("site", STATIC_SITE, root=str(root),
                                                                   base_url="https://owned.example",
                                                                   access_log=str(tmp_path / "access.log"))])
    return pub, root, tmp_path


def staged(pub, ctx, content=PAGE, path="guides/tides.html"):
    d = pub.execute(pact("draft", {"destination": "site", "path": path, "content": content}), ctx)
    assert d.status == ResultStatus.OK, d.failure_detail
    draft_id = d.artifacts["draft_id"]
    p = pub.execute(pact("preview", {"draft_id": draft_id}), ctx)
    assert p.status == ResultStatus.OK
    return draft_id, p


def test_draft_preview_submit_publishes_exactly_what_was_written(site):
    pub, root, _ = site
    assert isinstance(pub, ModalityExecutor) and pub.modality == Modality.API
    assert {t.name for t in pub.tools()} == {"publish.draft", "publish.preview", "publish.submit", "publish.update",
                                              "publish.status", "publish.analytics"}
    pol = ScopedPolicy(scopes=SCOPES)
    ctx = make_ctx(pol)
    d = pub.execute(pact("draft", {"destination": "site", "path": "guides/tides.html", "content": PAGE}), ctx)
    draft_id = d.artifacts["draft_id"]
    assert d.observations[0].data["state"] == "draft" and not (root / "guides").exists()        # nothing published
    assert pol.checks == []                                                                     # a draft is local
    # submitting before previewing is refused
    early = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    assert early.failure_class == FailureClass.INVALID_ARGUMENTS and "previewed" in early.failure_detail
    # the preview is the exact rendering and the exact place
    p = pub.execute(pact("preview", {"draft_id": draft_id}), ctx).observations[0]
    assert p.data["content"] == PAGE and p.data["content_complete"] and p.data["bytes"] == len(PAGE.encode())
    assert p.data["where"] == {"kind": "static_site", "destination": "site", "location": str(root / "guides" / "tides.html"),
                               "url": "https://owned.example/guides/tides.html", "replaces_sha256": None}
    assert p.provenance.content_sha256 == hashlib.sha256(PAGE.encode()).hexdigest() and PAGE in p.text
    assert not (root / "guides").exists()
    s = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    assert s.status == ResultStatus.OK and s.verified is True
    # exactly the agent's bytes: no template, header, footer or tracking was added
    assert (root / "guides" / "tides.html").read_bytes() == PAGE.encode()
    op = pol.checks[-1]
    assert op.op == "publish.write" and op.arguments["content"] == PAGE and op.arguments["url"] == "https://owned.example/guides/tides.html"
    assert op.arguments["content_sha256"] == hashlib.sha256(PAGE.encode()).hexdigest() and len(pol.checks) == 1
    st = pub.execute(pact("status", {"draft_id": draft_id}), ctx).observations[0]
    row = st.data["drafts"][0]
    assert row["state"] == "published" and row["live_matches_published"] is True and row["published"]["revision"] == 1
    assert {x.name: x.value for x in st.quantities}["publish.drafts.published"] == 1.0
    assert not any("reward" in k or "bonus" in k for o in s.observations for k in list(o.data) + [q.name for q in o.quantities])
    # submitting the same revision again does nothing more
    again = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    assert again.status == ResultStatus.OK and again.observations[0].data["already_published"] and len(pol.checks) == 1


def test_update_makes_a_new_revision_that_needs_preview_and_submit(site):
    pub, root, tmp = site
    pol = ScopedPolicy(scopes=SCOPES)
    ctx = make_ctx(pol)
    draft_id, _ = staged(pub, ctx)
    pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    newer = PAGE.replace("06:12", "06:31")
    u = pub.execute(pact("update", {"draft_id": draft_id, "content": newer}), ctx).observations[0]
    assert u.data["revision"] == 2 and u.data["state"] == "draft"
    assert (root / "guides" / "tides.html").read_text() == PAGE                    # the live page is unchanged so far
    assert pub.execute(pact("submit", {"draft_id": draft_id}), ctx).failure_class == FailureClass.INVALID_ARGUMENTS
    p = pub.execute(pact("preview", {"draft_id": draft_id}), ctx).observations[0]
    assert p.data["where"]["replaces_sha256"] == hashlib.sha256(PAGE.encode()).hexdigest() and p.data["content"] == newer
    assert pub.execute(pact("submit", {"draft_id": draft_id}), ctx).status == ResultStatus.OK
    assert (root / "guides" / "tides.html").read_text() == newer
    # someone edits the live file by hand: status reports that what is live is no longer what was published
    (root / "guides" / "tides.html").write_text("changed outside")
    assert pub.execute(pact("status", {}), ctx).observations[0].data["drafts"][0]["live_matches_published"] is False
    # drafts survive a restart of the adapter
    again = PublishingExecutor(str(tmp / "state"), list(pub.destinations.values()))
    assert again.drafts()[0]["revision"] == 2 and again.drafts()[0]["state"] == "published"


def test_approval_required_then_approved(site):
    pub, root, _ = site
    pol = ScopedPolicy(scopes=SCOPES, approval_rule=lambda op, a: op.op == "publish.write")
    ctx = make_ctx(pol)
    draft_id, _ = staged(pub, ctx)
    res = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    assert res.status == ResultStatus.NEEDS_APPROVAL and res.artifacts["approval_id"] and res.artifacts["draft_id"] == draft_id
    assert not (root / "guides").exists()                                          # nothing was published
    queue = pol.approvals()
    assert len(queue) == 1 and queue[0]["op"]["arguments"]["content"] == PAGE      # the operator sees the exact content
    assert queue[0]["op"]["arguments"]["location"] == str(root / "guides" / "tides.html")
    st = pub.execute(pact("status", {"draft_id": draft_id}), ctx).observations[0].data
    assert st["drafts"][0]["state"] == "pending_approval" and st["drafts"][0]["approval_id"] == queue[0]["approval_id"]
    assert st["counts"]["pending_approval"] == 1
    # still waiting: submitting again does not publish and does not create a second request
    res = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    assert res.status == ResultStatus.NEEDS_APPROVAL and len(pol.approvals()) == 1 and not (root / "guides").exists()
    pol.resolve_approval(queue[0]["approval_id"], True, by="operator")
    res = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    assert res.status == ResultStatus.OK and (root / "guides" / "tides.html").read_text() == PAGE
    assert pub.drafts()[0]["state"] == "published" and pol.approvals() == []
    # the approval was for that content: a changed revision has to be approved again
    pub.execute(pact("update", {"draft_id": draft_id, "content": PAGE + "<p>more</p>\n"}), ctx)
    pub.execute(pact("preview", {"draft_id": draft_id}), ctx)
    res = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    assert res.status == ResultStatus.NEEDS_APPROVAL and (root / "guides" / "tides.html").read_text() == PAGE


def test_approval_required_then_rejected(site):
    pub, root, _ = site
    pol = ScopedPolicy(scopes=SCOPES, approval_rule=lambda op, a: op.op == "publish.write")
    ctx = make_ctx(pol)
    draft_id, _ = staged(pub, ctx)
    first = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    pol.resolve_approval(first.artifacts["approval_id"], False, by="operator", note="not this")
    res = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION and res.retries == 0
    assert not (root / "guides").exists()
    d = pub.drafts()[0]
    assert d["state"] == "rejected" and "rejected by the operator" in d["reasons"][0]
    assert pub.execute(pact("status", {}), ctx).observations[0].data["counts"]["rejected"] == 1
    # the same content stays rejected however often it is submitted
    for _ in range(3):
        assert pub.execute(pact("submit", {"draft_id": draft_id}), ctx).status == ResultStatus.DENIED
    assert not (root / "guides").exists()


def test_denied_destination_and_invalid_requests(site, tmp_path):
    pub, root, _ = site
    ctx = make_ctx(ScopedPolicy(scopes=("publish.draft",)))                        # no publish.write scope
    draft_id, _ = staged(pub, ctx)
    res = pub.execute(pact("submit", {"draft_id": draft_id}), ctx)
    assert res.status == ResultStatus.DENIED and not (root / "guides").exists() and pub.drafts()[0]["state"] == "rejected"
    outside = tmp_path / "outside.html"
    os.symlink(str(tmp_path), str(root / "link"))
    bad = [{"destination": "elsewhere", "path": "a.html", "content": "x"},
           {"destination": "site", "path": "../outside.html", "content": "x"},
           {"destination": "site", "path": "/etc/passwd", "content": "x"},
           {"destination": "site", "path": "a//b.html", "content": "x"},
           {"destination": "site", "path": ".htaccess", "content": "x"},
           {"destination": "site", "path": "link/outside.html", "content": "x"},
           {"destination": "site", "path": "", "content": "x"},
           {"destination": "site", "path": "a.html", "content": ""}]
    for args in bad:
        res = pub.execute(pact("draft", args), ctx)
        assert res.failure_class == FailureClass.INVALID_ARGUMENTS, args
    assert not outside.exists() and len(pub.drafts()) == 1
    assert pub.execute(pact("preview", {"draft_id": "draft_none"}), ctx).failure_class == FailureClass.INVALID_ARGUMENTS
    empty = PublishingExecutor(str(tmp_path / "s2"), [])
    assert all(not t.available and "no owned publishing destination" in t.unavailable_reason for t in empty.tools())


# ---- owned Discord channel as a destination -------------------------------------------------------------------
def test_publishing_to_an_owned_channel_goes_through_the_discord_adapter(tmp_path):
    with StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0))) as staging:
        st = staging.state
        discord = DiscordExecutor(staging.base_url, guild_id=st.guild_id, secrets={"discord_bot_token": st.token})
        pub = PublishingExecutor.from_resources({"discord": {"channels": {"announcements": st.general_id}}},
                                                str(tmp_path), discord=discord)
        assert pub.destinations["announcements"].kind == DISCORD_CHANNEL
        pol = ScopedPolicy(scopes=SCOPES, approval_rule=lambda op, a: op.op in ("discord.send_message", "discord.edit_message"))
        ctx = make_ctx(pol)
        text = "The tide table for January is in the guides channel."
        d = pub.execute(pact("draft", {"destination": "announcements", "content": text}), ctx).artifacts["draft_id"]
        p = pub.execute(pact("preview", {"draft_id": d}), ctx).observations[0]
        assert p.data["content"] == text and p.data["where"]["channel_id"] == st.general_id
        assert p.data["where"]["location"] == f"discord channel {st.general_id}" and st.request_log == []
        res = pub.execute(pact("submit", {"draft_id": d}), ctx)
        assert res.status == ResultStatus.NEEDS_APPROVAL and st.messages[st.general_id] == []
        assert pol.approvals()[0]["op"]["arguments"]["content"] == text
        pol.resolve_approval(res.artifacts["approval_id"], True, by="operator")
        res = pub.execute(pact("submit", {"draft_id": d}), ctx)
        assert res.status == ResultStatus.OK and [m["content"] for m in st.messages[st.general_id]] == [text]
        published = pub.drafts()[0]["published"]
        assert published["environment"] == "staging" and published["message_id"] == st.messages[st.general_id][0]["id"]
        # an update replaces the same message after a new approval; it does not post a second one
        pub.execute(pact("update", {"draft_id": d, "content": text + " Corrected."}), ctx)
        pub.execute(pact("preview", {"draft_id": d}), ctx)
        res = pub.execute(pact("submit", {"draft_id": d}), ctx)
        assert res.status == ResultStatus.NEEDS_APPROVAL
        pol.resolve_approval(res.artifacts["approval_id"], True, by="operator")
        assert pub.execute(pact("submit", {"draft_id": d}), ctx).status == ResultStatus.OK
        assert [m["content"] for m in st.messages[st.general_id]] == [text + " Corrected."]
        # too long for a channel message; no Discord adapter configured
        assert pub.execute(pact("draft", {"destination": "announcements", "content": "x" * 2001}), ctx).failure_class \
            == FailureClass.INVALID_ARGUMENTS
        lonely = PublishingExecutor(str(tmp_path / "l"), [Destination("ch", DISCORD_CHANNEL, channel_id=st.general_id)])
        d2 = lonely.execute(pact("draft", {"destination": "ch", "content": "x"}), ctx).artifacts["draft_id"]
        lonely.execute(pact("preview", {"draft_id": d2}), ctx)
        assert lonely.execute(pact("submit", {"draft_id": d2}), ctx).status == ResultStatus.DENIED
        # everything is reachable through one api-modality executor
        api = ApiModality([pub, discord])
        assert len(api.tools()) == 16 and api.execute(pact("status", {}), ctx).status == ResultStatus.OK
        api.close()


# ---- aggregate analytics --------------------------------------------------------------------------------------
class _LoggingSite:
    """Serves a directory over HTTP and writes a combined-format access log, like an ordinary web server."""

    def __init__(self, root, log_path):
        log = open(log_path, "a", buffering=1)

        class Handler(SimpleHTTPRequestHandler):
            disable_nagle_algorithm = True

            def log_request(self, code="-", size="-"):
                stamp = time.strftime("%d/%b/%Y:%H:%M:%S +0000", time.gmtime())
                log.write(f'{self.client_address[0]} - - [{stamp}] "{self.requestline}" {getattr(code, "value", code)} '
                          f'{size} "{self.headers.get("Referer", "-")}" "{self.headers.get("User-Agent", "-")}"\n')

            def log_message(self, *a):
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(root)))
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self._log = log

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self._log.close()


def test_analytics_from_the_access_log_of_the_owned_site(site):
    pub, root, tmp = site
    pol = ScopedPolicy(scopes=SCOPES)
    ctx = make_ctx(pol)
    for path, content in (("index.html", "<h1>home</h1>"), ("guides/tides.html", PAGE), ("style.css", "h1{}")):
        draft_id, _ = staged(pub, ctx, content, path)
        assert pub.execute(pact("submit", {"draft_id": draft_id}), ctx).status == ResultStatus.OK
    server = _LoggingSite(root, tmp / "access.log")
    http = HttpApiExecutor(user_agent="Mozilla/5.0 (X11; Linux x86_64) reader")
    try:
        for path, n in (("/index.html", 3), ("/guides/tides.html", 5), ("/style.css", 2), ("/missing.html", 1)):
            for _ in range(n):
                http.execute(act("http.get", {"url": server.url + path}), ctx)
        crawler = HttpApiExecutor(user_agent="ExampleBot/2.1 (+https://crawler.example/bot)")
        crawler.execute(act("http.get", {"url": server.url + "/index.html"}), ctx)
        crawler.close()
    finally:
        http.close()
        server.close()
    res = pub.execute(pact("analytics", {"destination": "site"}), ctx)
    assert res.status == ResultStatus.OK and pol.checks[-1].op == "publish.analytics"
    obs = res.observations[0]
    q = {x.name: x for x in obs.quantities}
    assert q["site.site.requests"].value == 12.0 and q["site.site.page_views"].value == 8.0
    assert q["site.site.visitors"].value == 1.0                              # every request came from this machine
    assert q["site.site.page_views.declared_automated"].value == 1.0 and q["site.site.errors"].value == 1.0
    assert q["site.site.views.guides_tides_html"].value == 5.0 and q["site.site.views.index_html"].value == 3.0
    assert q["site.site.page_views"].kind == "count" and q["site.site.page_views"].exposure >= 1.0
    # provenance: which log, when, and the hash of exactly what was read; only aggregates, no addresses
    assert obs.provenance.source == "publishing.analytics" and obs.provenance.uri == str(tmp / "access.log")
    assert obs.provenance.content_sha256 == hashlib.sha256((tmp / "access.log").read_bytes()).hexdigest()
    assert obs.provenance.retrieved_at > 0 and "127.0.0.1" not in json.dumps(obs.to_dict())
    # denied by policy -> nothing is read; no log -> a classified failure
    strict = ScopedPolicy(scopes=("publish.draft",))
    assert pub.execute(pact("analytics", {"destination": "site"}), make_ctx(strict)).status == ResultStatus.DENIED
    os.remove(tmp / "access.log")
    assert pub.execute(pact("analytics", {"destination": "site"}), ctx).failure_class == FailureClass.WRONG_SITE_ASSUMPTION


def test_access_log_aggregation_rules():
    now = 1_800_000_000.0
    def line(ip, ago, req, status, ref="-", ua="Mozilla/5.0"):
        stamp = time.strftime("%d/%b/%Y:%H:%M:%S +0000", time.gmtime(now - ago))
        return f'{ip} - - [{stamp}] "{req} HTTP/1.1" {status} 512 "{ref}" "{ua}"'
    log = "\n".join([
        line("203.0.113.5", 500, "GET /", 200, "https://search.example/q"),
        line("203.0.113.5", 490, "GET /guide.html?x=1", 200),
        line("203.0.113.9", 480, "GET /guide.html", 304, "https://owned.example/"),
        line("198.51.100.7", 470, "GET /guide.html", 200, ua="Mozilla/5.0 (compatible; SomeBot/1.0)"),
        line("198.51.100.8", 460, "GET /app.js", 200),
        line("198.51.100.8", 450, "POST /form", 200),
        line("198.51.100.8", 440, "GET /nothing", 404),
        line("192.0.2.44", 7200, "GET /old.html", 200),
        "this line is not a log line",
    ])
    a = aggregate_access_log(log, now=now, own_host="owned.example")
    assert (a["requests"], a["page_views"], a["visitors"], a["automated_page_views"], a["errors"], a["lines_unparsed"]) \
        == (8, 4, 3, 1, 1, 1)
    assert a["by_path"] == {"/guide.html": 2, "/": 1, "/old.html": 1} and a["referrer_hosts"] == {"search.example": 1}
    assert a["window_s"] == pytest.approx(7200 - 440)
    recent = aggregate_access_log(log, now=now, since_s=3600.0, own_host="owned.example")
    assert (recent["requests"], recent["page_views"], recent["visitors"], recent["window_s"]) == (7, 3, 2, 3600.0)
    assert "203.0.113.5" not in json.dumps(a)
