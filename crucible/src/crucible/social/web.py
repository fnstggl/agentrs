"""Browser pages for the social internet (on the 4.0 browser mode, crucible.web.app).

`make_social_web_app(world_or_server, world_id=None)` is `make_web_app` plus the pages registered here; or call
`register_pages(app.state.web)` on an existing app. The 4.0 rules hold unchanged: every page render is a
`World.query` (rolled back, nothing recorded), every form submission is ONE recorded `World.execute` of the tool
the form names, as the signed-in participant - so a browser session and the same tool calls made headlessly give
the same event log.

Pages:

    social.crucible/          your timeline, the platforms, people to follow, your profile
    social.crucible/u?p=      a profile (follow / unfollow, the person's items)
    studio.crucible/          create artifacts (text, HTML, image, audio, page, course); `/made?ref=` publish links
    community.crucible/       directory, found a community; /c?id= a community (join, leave, post, events,
                              proposals); /item?id= a post with comments
    <site>.crucible/          shorts, tube, posts, pics, blog, pods, launch, creators, letters, dev, kb: latest
                              items and a publish form; /item?id= an item with comments; dev/kb /q?id= a question
    msg.crucible/             conversations; /conv?id= one conversation
    content.crucible/raw?id=  the bytes of an item you may open, served under a sandbox CSP (an interactive page
                              runs in an opaque origin: it cannot read your session or act as you)
    <host>/_api               every endpoint of the host as a form
"""
from __future__ import annotations

import base64

from fastapi.responses import Response

from ..web.app import Field, Link, PageError, Pre, _redirect, make_web_app
from . import content as C
from .platforms import KIND_FORMATS, PLATFORMS

SITE_HOSTS = ("shorts", "tube", "posts", "pics", "blog", "pods", "launch", "creators", "letters", "dev", "kb")
ALL_HOSTS = ("social", "studio", "community", "msg", "content") + SITE_HOSTS
RAW_CSP = {"text/html": "sandbox allow-scripts allow-forms; default-src 'none'; script-src 'unsafe-inline'; "
                        "style-src 'unsafe-inline'; img-src data:; form-action 'none'"}


def make_social_web_app(server_or_world, world_id: str | None = None, **kw):
    app = make_web_app(server_or_world, world_id, **kw)
    register_pages(app.state.web)
    return app


def _present(c, host: str) -> bool:
    return any(e.host == host for e in c.w.endpoints.values())


def register_pages(app) -> None:  # noqa: C901 - one place that lists every page
    r = app.route

    # ------------------------------------------------------------------ shared builders
    def item_link(c, it: dict, host: str | None = None):
        h = host or PLATFORMS.get(it.get("platform"), {}).get("host") or "content"
        if it.get("platform") == "forum":
            h = "content"
        return Link(it.get("title") or it.get("id"), c.url(h if _present(c, h) or h == "content" else "content",
                                                           "/item", id=it["id"]), id=f"item-{it['id']}")

    def item_table(c, title: str, items: list[dict], host: str | None = None, empty: str = "Nothing here yet."):
        c.table(title, items, [("title", "Title", lambda it: item_link(c, it, host)), ("kind", "Kind"),
                               ("creator", "By"), ("published_at", "Published"), ("comments", "Comments"),
                               ("tags", "Tags")], empty=empty, id=f"tbl-{title.lower().replace(' ', '-')}")

    def item_page(c, host: str):
        cid = c.qp.get("id", "")
        it = c.q("content.get", id=cid)
        c.title = it["title"]
        meta = [f"{it['kind']} on {it['platform']}", c.who(it["creator"]), f"version {it['version']}"]
        if it.get("duration_s"):
            meta.append(f"{it['duration_s']:g} s")
        body = None
        if it.get("format") in ("text", "html", "page", "document", "course", "app", "package"):
            try:
                body = Pre(c.q("content.open", id=cid).get("text", "")[:20_000], id="item-text")
            except PageError:
                body = None
        forms = []
        if _present(c, host) and host != "content" and f"{host}.comment" in c.w.endpoints:
            forms.append(c.form(f"{host}.comment", title="Comment", hide={"id": cid}, fields=["text"],
                                submit="Comment"))
        elif host == "content":
            forms.append(c.form("content.comment", title="Comment", hide={"id": cid}, fields=["text"],
                                submit="Comment"))
        comments = c.tq("content.comments", id=cid, limit=100) or []
        children = [{"label": f"comment by {x['author']}", "id": x["comment"], "meta": [c.who(x["author"])],
                     "body": c.rich(x["text"])} for x in comments]
        c.add("article", it["title"], id="item", meta=meta, body=body or c.rich(it.get("description") or ""),
              forms=[f for f in forms if f is not None], children=children)
        c.kv("About", it, ["id", "kind", "platform", "format", "mime", "size", "creator", "published_at", "tags",
                           "description", "community", "access", "comments"], id="item-about",
             extra=[("media", Link("open the artifact", c.url("content", "/raw", id=cid), id="item-raw"))])
        c.links("More", [Link(f"Profile of {it['creator']}", c.url("social", "/u", p=it["creator"]),
                              id="link-creator"),
                         Link("Share or send it", c.url("social", "/share", id=cid), id="link-share")])

    def api_console(c, host: str):
        c.title = f"All endpoints of {host}.crucible"
        for name, ep in sorted(c.w.endpoints.items()):
            if ep.host == host:
                f = c.form(name, title=name, submit=f"Call {name.split('.', 1)[1]}")
                if f is not None:
                    c.add("form", id="ep-" + name.replace(".", "-"), form=f)

    def publish_form(c, host: str):
        if f"{host}.publish" not in c.w.endpoints:
            return None
        return c.form(f"{host}.publish", title=f"Publish on {host}.crucible", values={"ref": c.qp.get("ref", "")},
                      next="/item?id={id}")

    def platform_links(c):
        return [Link(f"{p['host']}.crucible - {p['label']}", c.url(p["host"], "/"), id=f"plat-{name}", desc=p["doc"])
                for name, p in PLATFORMS.items() if _present(c, p["host"])]

    # ------------------------------------------------------------------ social.crucible
    @r("social", "/")
    def social_home(c):
        c.title = "Social network"
        tl = c.tq("social.timeline", limit=30) or {"entries": []}
        c.table("Your timeline", tl["entries"], [
            ("item", "Item", lambda e: item_link(c, e["item"])), ("type", "How"), ("by", "By"), ("t", "When")],
            empty="Follow people to see what they publish and share.", id="timeline")
        c.forms("Follow", [c.form("social.follow", title="Follow someone", fields=["target"], submit="Follow")])
        sug = c.tq("social.suggestions", limit=10) or []
        c.table("People you may know", sug, [("principal", "Who", lambda s: Link(s["principal"],
                                                                                   c.url("social", "/u",
                                                                                         p=s["principal"]))),
                                              ("followed_by_people_you_follow", "Followed by people you follow")],
                empty="No suggestions yet.", id="suggestions")
        c.links("Platforms", platform_links(c))
        c.links("Create", [Link("Studio: create text, images, audio, video, pages, courses",
                                c.url("studio", "/"), id="link-studio")])
        c.forms("Profile", [c.form("social.set_profile", title="Your profile")])

    @r("social", "/u")
    def social_user(c):
        p = c.qp.get("p") or c.p
        prof = c.q("social.profile", principal=p, items=30)
        c.title = f"{prof['display_name']} ({prof['principal']})"
        c.kv("Profile", prof, ["principal", "display_name", "bio", "followers", "following", "you_follow"],
             id="profile")
        if prof["principal"] != c.p:
            tool = "social.unfollow" if prof["you_follow"] else "social.follow"
            c.forms("Follow", [c.form(tool, hide={"target": prof["principal"]}, inline=True,
                                      submit="Unfollow" if prof["you_follow"] else "Follow",
                                      submit_aria=f"{'Unfollow' if prof['you_follow'] else 'Follow'} {p}")],
                    id="follow-box")
        item_table(c, "Items", prof["items"])

    @r("social", "/share")
    def social_share(c):
        cid = c.qp.get("id", "")
        it = c.q("content.get", id=cid)
        c.title = f"Share {it['title']}"
        c.forms("Share", [c.form("social.share", title="Repost to your followers", hide={"id": cid}, fields=["note"],
                                 submit="Repost"),
                          c.form("social.share", title="Send to people", hide={"id": cid}, fields=["to", "note"],
                                 submit="Send", key="to")])

    # ------------------------------------------------------------------ studio.crucible
    @r("studio", "/")
    def studio_home(c):
        c.title = "Creator studio"
        c.note("Make an artifact, then publish it on a platform or post it in a community.", id="studio-note")
        nxt = "/made?ref={ref}"
        c.forms("Create", [
            c.form("studio.write", title="Write text, markdown, HTML or a PDF document", next=nxt,
                   fields=["format", "title", "body"]),
            c.form("studio.image", title="Draw a text card", fields=["text", "title", "width", "background"],
                   next=nxt),
            c.form("studio.audio", title="Synthesise audio", fields=["kind", "seconds", "freq", "notes", "bpm"],
                   next=nxt),
            c.form("studio.page", title="Make an interactive page", fields=["title", "html", "script"], next=nxt),
        ])
        drafts = c.tq("studio.drafts", limit=30) or []
        c.table("Your artifacts", drafts, [("ref", "Artifact", lambda d: Link(d["ref"][:19] + "...",
                                                                             c.url("studio", "/made", ref=d["ref"]))),
                                           ("format", "Format"), ("title", "Title"), ("size", "Bytes"),
                                           ("published", "Published")], id="drafts")

    @r("studio", "/made")
    def studio_made(c):
        ref = c.qp.get("ref", "")
        info = c.q("studio.probe", ref=ref)
        c.title = f"Artifact {ref[:19]}..."
        c.kv("Artifact", info, ["ref", "kind", "mime", "size", "facts", "parents"], id="artifact")
        fmt = info["kind"]
        targets = []
        for name, p in PLATFORMS.items():
            if p.get("publish") is False or p.get("bridged") or not _present(c, p["host"]):
                continue
            kinds = [k for k in p["kinds"] if fmt in KIND_FORMATS[k]]
            if not kinds:
                continue
            path = "/" if name != "community" else "/"
            targets.append(Link(f"Publish on {p['host']}.crucible ({', '.join(kinds)})",
                                c.url(p["host"], path, ref=ref), id=f"publish-{name}"))
        c.links("Publish it", targets, empty="No platform carries this format.")

    # ------------------------------------------------------------------ community.crucible
    @r("community", "/")
    def community_home(c):
        c.title = "Communities"
        mine = c.tq("community.mine") or []
        c.table("Your communities", mine, [("community", "Community", lambda x: Link(x["name"], c.url(
            "community", "/c", id=x["community"], ref=c.qp.get("ref")), id=f"comm-{x['community']}")),
            ("your_status", "You"), ("role", "Role"), ("members", "Members"), ("active_members", "Active")],
            empty="You are not in any community yet.", id="mine")
        rows = c.tq("community.directory", limit=50) or []
        c.table("Directory", rows, [("community", "Community", lambda x: Link(x["name"], c.url(
            "community", "/c", id=x["community"]), id=f"dir-{x['community']}")), ("subject", "Subject"),
            ("members", "Members"), ("active_members", "Active"), ("join", "Joining")], id="directory")
        c.forms("Found a community", [c.form("community.found", title="Found a community on a subject of your choice",
                                             fields=["id", "name", "subject", "description", "tags"],
                                             next="/c?id={community}")])

    @r("community", "/c")
    def community_page(c):
        cid = c.qp.get("id", "")
        v = c.q("community.view", community=cid)
        c.title = v["name"]
        c.kv("About", v, ["community", "subject", "description", "status", "members", "active_members", "join",
                          "visibility", "founder", "moderators", "rules_version"], id="about")
        you = v.get("you")
        if you:
            c.kv("You here", you, id="you")
        if not you or you["status"] in ("left", "invited"):
            c.forms("Join", [c.form("community.join", hide={"community": cid}, fields=["message"],
                                    submit="Join", submit_aria=f"Join {cid}")], id="join-box")
        else:
            c.forms("Post", [c.form("community.post", title="Post in this community", hide={"community": cid},
                                    values={"ref": c.qp.get("ref", "")}, fields=["ref", "title", "kind", "tags"],
                                    next=f"/item?id={{id}}")], id="post-box",
                    note="Make the artifact in the studio first (studio.crucible).")
            c.forms("Take part", [c.form("community.event", title="Schedule an event", hide={"community": cid}),
                                  c.form("community.propose", title="Make a proposal", hide={"community": cid}),
                                  c.form("community.leave", hide={"community": cid}, inline=True, submit="Leave",
                                         submit_aria=f"Leave {cid}")])
        feed = c.tq("community.feed", community=cid, limit=30)
        item_table(c, "Posts", (feed or {}).get("items", []), host="community")
        c.kv("Health", v["health"], id="health")
        c.table("Proposals", c.tq("community.proposals", community=cid, status="any") or [],
                [("proposal", "Proposal"), ("action", "Action"), ("by", "By"), ("status", "Status"), ("yes", "For"),
                 ("no", "Against"), ("vote", "Vote", lambda p: c.form(
                     "community.vote", hide={"proposal": p["proposal"]}, key=p["proposal"], inline=True,
                     buttons=[("approve", "true", "For", f"Vote for {p['proposal']}"),
                              ("approve", "false", "Against", f"Vote against {p['proposal']}")])
                  if p["status"] == "open" else None)], empty="No proposals.", id="proposals")
        c.table("Events", c.tq("community.events", community=cid) or [], [
            ("title", "Event"), ("at", "Starts"), ("going", "Going"), ("attended", "Attended")],
            empty="No upcoming events.", id="events")
        c.links("More", [Link("Members", c.url("community", "/members", id=cid), id="link-members"),
                         Link("Moderation log", c.url("community", "/_api"), id="link-api")])

    @r("community", "/members")
    def community_members(c):
        cid = c.qp.get("id", "")
        m = c.q("community.members", community=cid, limit=500)
        c.title = f"Members of {cid}"
        c.table("Members", m["members"], [("member", "Member"), ("role", "Role"), ("joined", "Joined"),
                                          ("last_active", "Last active"), ("posts", "Posts"),
                                          ("comments", "Comments"), ("reputation", "Reputation"),
                                          ("trust_level", "Trust level"), ("inactive", "Inactive")], id="members")

    @r("community", "/item")
    def community_item(c):
        item_page(c, "community")

    # ------------------------------------------------------------------ content.crucible
    @r("content", "/")
    def content_home(c):
        c.title = "Catalogue"
        res = c.q("content.list", limit=50)
        item_table(c, "Newest items", res["items"])
        c.links("Platforms", platform_links(c))

    @r("content", "/item")
    def content_item(c):
        item_page(c, "content")

    @r("content", "/raw")
    def content_raw(c):
        out = c.q("content.open", id=c.qp.get("id", ""))
        mime = out["mime"]
        csp = RAW_CSP.get(mime, "default-src 'none'; sandbox")
        return Response(content=base64.b64decode(out["data_b64"]), media_type=mime,
                        headers={"Content-Security-Policy": csp, "X-Content-Type-Options": "nosniff",
                                 "Cache-Control": "no-store", "Cross-Origin-Resource-Policy": "same-origin"})

    # ------------------------------------------------------------------ the content sites
    for h in SITE_HOSTS:
        def site_home(c, h=h):
            if not _present(c, h):
                raise PageError(f"no {h} platform in this world", 404)
            p = PLATFORMS[h]
            c.title = p["label"]
            c.note(p["doc"], id="site-doc")
            if h in ("dev", "kb"):
                qs = c.tq(f"{h}.questions", limit=30) or []
                c.table("Questions", qs, [("title", "Question", lambda q: Link(q["title"], c.url(h, "/q", id=q["id"]),
                                                                               id=f"q-{q['id']}")),
                                          ("creator", "By"), ("score", "Score"), ("answers", "Answers"),
                                          ("accepted", "Accepted")], id="questions")
                c.forms("Ask", [c.form(f"{h}.ask", title="Ask a question", fields=["title", "body", "tags"],
                                       next="/q?id={question}")])
            if h == "launch":
                t = c.tq("launch.today") or {"launches": []}
                c.table("Today's launches", t["launches"], [
                    ("title", "Launch", lambda x: item_link(c, x, "launch")), ("upvotes", "Upvotes"),
                    ("link", "Links to"), ("target", "Product"), ("creator", "By")], id="launches")
                c.forms("Launch", [c.form("launch.post", title="Launch a product",
                                          values={"ref": c.qp.get("ref", "")}, next="/item?id={id}")])
            if f"{h}.feed" in c.w.endpoints:
                feed = c.tq(f"{h}.feed", limit=30) or {"items": []}
                item_table(c, "Your feed", feed["items"], host=h)
            if f"{h}.latest" in c.w.endpoints:
                latest = c.tq(f"{h}.latest", limit=30) or {"items": []}
            else:
                latest = c.tq("content.list", platform=h, limit=30) or {"items": []}
            item_table(c, "Latest", latest["items"], host=h if f"{h}.comment" in c.w.endpoints else "content")
            if h not in ("dev", "kb", "launch"):
                c.forms("Publish", [publish_form(c, h)])
            if h == "creators":
                c.forms("Memberships", [c.form("creators.create_tier", title="Offer a membership tier"),
                                        c.form("creators.join", title="Join a tier")])
            if h == "letters":
                c.table("Newsletters", c.tq("letters.list") or [], [("newsletter", "Newsletter"), ("name", "Name"),
                                                                     ("owner", "By"), ("subscribers", "Subscribers"),
                                                                     ("issues", "Issues")], id="newsletters")
                c.forms("Newsletters", [c.form("letters.subscribe", title="Subscribe"),
                                        c.form("letters.create", title="Start a newsletter")])
        r(h, "/")(site_home)

        def site_item(c, h=h):
            item_page(c, h)
        r(h, "/item")(site_item)

    for h in ("dev", "kb"):
        def qa_page(c, h=h):
            q = c.q(f"{h}.question", id=c.qp.get("id", ""))
            c.title = q["title"]
            kids = [{"label": f"answer by {a['creator']}", "id": a["id"],
                     "meta": [c.who(a["creator"]), f"score {a['score']}", "accepted" if a["id"] == q["accepted"]
                              else ""],
                     "body": Pre(a.get("text", "")[:20_000])} for a in q["answers_list"]]
            c.add("article", q["title"], id="question", meta=[c.who(q["creator"]), f"score {q['score']}"],
                  body=Pre(q.get("text", "")[:20_000]),
                  forms=[f for f in (c.form(f"{h}.answer", title="Your answer", hide={"question": q["id"]},
                                            fields=["body"], submit="Answer"),) if f is not None],
                  children=kids)
        r(h, "/q")(qa_page)

    # ------------------------------------------------------------------ msg.crucible
    @r("msg", "/")
    def msg_home(c):
        c.title = "Messages"
        conv = c.tq("msg.conversations") or []
        c.table("Conversations", conv, [("conversation", "Conversation", lambda x: Link(
            ", ".join(m for m in x["members"] if m != c.p) or x["conversation"],
            c.url("msg", "/conv", id=x["conversation"]), id=f"conv-{x['conversation']}")), ("kind", "Kind"),
            ("messages", "Messages"), ("unread", "Unread"), ("last_t", "Last")], empty="No conversations.",
            id="conversations")
        c.forms("New message", [c.form("msg.send", title="Send a message", fields=["to", "text", "item"],
                                       next="/conv?id={conversation}")])

    @r("msg", "/conv")
    def msg_conv(c):
        cid = c.qp.get("id", "")
        conv = c.tq("msg.conversations") or []
        rec = next((x for x in conv if x["conversation"] == cid), None)
        if rec is None:
            raise PageError(f"no conversation {cid}", 404)
        c.title = f"Conversation with {', '.join(m for m in rec['members'] if m != c.p)}"
        h = c.q("msg.history", conversation=cid, limit=100)
        c.table("Messages", h["messages"], [("sender", "From"), ("text", "Message", lambda m: c.rich(m["text"])),
                                            ("item", "Shared", lambda m: Link(m["item"], c.url("content", "/item",
                                                                                               id=m["item"]))
                                             if m["item"] else None), ("t", "When")], id="messages")
        c.forms("Reply", [c.form("msg.send", hide={"conversation": cid}, fields=["text", "item"], submit="Send"),
                          c.form("msg.read", hide={"conversation": cid}, fields=[], submit="Mark read",
                                 key="read")])

    # ------------------------------------------------------------------ endpoint consoles
    for h in ALL_HOSTS:
        def console(c, h=h):
            api_console(c, h)
        r(h, "/_api")(console)


__all__ = ["make_social_web_app", "register_pages", "Field", "_redirect"]
