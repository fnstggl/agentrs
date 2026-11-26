"""Gate SP-4 (MASTER_SPEC Part 4): interoperable platforms - short and long video, text posts, images,
articles/blogs, forums, social networks, communities, messaging, search, creator subscriptions, newsletters,
product discovery, developer communities, knowledge sharing (plus audio) - sharing one population, one content
catalogue and one identity space.

One world holds every platform. For each platform kind its characteristic use is exercised through tools by
ordinary participants and checked in shared state: the item is a row of the one catalogue, the same principal is
the creator everywhere, a second participant reaches it through that platform, and cross-platform paths work (the
social timeline and search span every platform; a message carries an item of another platform; a launch links to
a market listing; a forum post is a catalogue thread). `platform_kinds` counts the platform kinds whose checks
all passed.
"""
from __future__ import annotations

import base64

import pytest

from crucible.acceptance.v5 import evidence
from crucible.social.platforms import PLATFORMS
from social_helpers import Actor, social_world, assert_met

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4.py"


@pytest.mark.timeout(900)
def test_SP_4_interoperable_platforms():
    w, P = social_world(seed=4, extra=["market"], people=("alice", "bob", "carol", "dave"))
    A, B, Cc, D = (Actor(w, P[h]) for h in ("alice", "bob", "carol", "dave"))
    st = w.state
    res: dict[str, dict] = {}

    def item_ok(cid: str, platform: str, creator: str) -> bool:
        r = st.tables["content"][cid]
        return r["platform"] == platform and r["creator"] == creator and st.tables["artifacts"][r["ref"]]["published"]

    # --- content sites: publish as alice, reach as bob through the site
    img = A.ok("studio.image", text="Lighthouse", width=480)["ref"]
    snd = A.ok("studio.audio", kind="chord", notes=["C4", "E4", "G4"], seconds=65.0)["ref"]
    short = A.ok("studio.video", images=[img], audio=A.ok("studio.audio", kind="tone", seconds=4.0)["ref"],
                 seconds_per_image=4)["ref"]
    long_ = A.ok("studio.video", images=[img, img], audio=snd, seconds_per_image=33)["ref"]
    txt = A.ok("studio.write", format="text", title="Lighthouse keeping", body="Lamps, lenses and fog horns.")["ref"]
    art = A.ok("studio.write", format="html", title="A history of lighthouses", body="From **Pharos** to LEDs.")["ref"]
    sites = {"shorts": (short, "video_short"), "tube": (long_, "video_long"), "posts": (txt, "post"),
             "pics": (img, "image"), "blog": (art, "article"), "pods": (snd, "audio")}
    ids = {}
    for site, (ref, kind) in sites.items():
        cid = A.ok(f"{site}.publish", ref=ref, title=f"Lighthouse on {site}", kind=kind)["id"]
        ids[site] = cid
        seen = [it["id"] for it in B.ok(f"{site}.latest", limit=20)["items"]]
        view = B.ok(f"{site}.view", id=cid)
        B.ok(f"{site}.comment", id=cid, text="Lovely")
        res[site] = {"catalogue": item_ok(cid, site, P["alice"]), "reached": cid in seen and view["id"] == cid,
                     "kind": st.tables["content"][cid]["kind"] == kind,
                     "comment": st.tables["content"][cid]["comments"] == 1}
    # --- forum (4.0): a forum post enters the catalogue as a thread
    fp = Cc.ok("forum.post", board="general", title="Lighthouse tours?", body="Anyone visited the north light?")
    fl = st.get("content_forum", fp["post"])
    res["forum"] = {"bridged": fl is not None and item_ok(fl["id"], "forum", P["carol"]) and
                    st.tables["content"][fl["id"]]["kind"] == "thread",
                    "listed": any(it["id"] == (fl or {}).get("id") for it in B.ok("content.list", platform="forum")["items"])}
    # --- social network: follow alice; bob's timeline spans her items on every platform
    B.ok("social.follow", target="alice")
    tl = B.ok("social.timeline", limit=50)["entries"]
    plats = {e["item"]["platform"] for e in tl}
    B.ok("social.share", id=ids["pics"])
    res["social"] = {"timeline_spans_platforms": len(plats & set(sites)) == len(sites),
                     "followers": B.ok("social.followers", principal="alice")["count"] == 1,
                     "public_share": B.ok("social.shares", id=ids["pics"])["public_shares"] == 1}
    # --- communities: found, join, post an image item, members' feed
    A.ok("community.found", id="lightkeepers", name="Lightkeepers", subject="keeping and restoring lighthouses")
    B.ok("community.join", community="lightkeepers")
    cp = B.ok("community.post", community="lightkeepers", ref=B.ok("studio.write", format="text", title="Lens",
                                                                    body="Fresnel lens care")["ref"], title="Lens care")
    feed = A.ok("community.feed", community="lightkeepers")["items"]
    res["community"] = {"post": item_ok(cp["id"], "community", P["bob"]),
                        "feed": [it["id"] for it in feed] == [cp["id"]],
                        "members": A.ok("community.view", community="lightkeepers")["members"] == 2}
    # --- messaging: a private message carrying an item of another platform
    m = Cc.ok("msg.send", to="dave", text="watch this", item=ids["tube"])
    hist = D.ok("msg.history", conversation=m["conversation"])
    res["msg"] = {"delivered": hist["messages"][-1]["item"] == ids["tube"] and hist["unread"] == 1,
                  "private": B("msg.history", conversation=m["conversation"]).ok is False}
    # --- search: every public catalogue item is findable, by its own kind (filled in after the Q&A below)
    search_checks = {k: ("lighthouse", ids[s]) for s, (_, k) in sites.items()}
    # --- creator subscriptions: paid from the fan's own account, members-only item
    tier = A.ok("creators.create_tier", name="Keeper", price_cents=300, perks="behind the scenes")["tier"]
    mo = A.ok("creators.publish", ref=A.ok("studio.write", format="text", title="BTS", body="Inside the lamp room")
              ["ref"], kind="post", title="Lamp room")["id"]
    before = D("content.open", id=mo).ok
    acct = w.modules["bank"].primary_account(st, P["dave"])
    bal0 = st.tables["accounts"][acct]["balance"]
    D.ok("creators.join", tier=tier)
    after = D.ok("content.open", id=mo)
    res["creators"] = {"gated": before is False and base64.b64decode(after["data_b64"]) == b"Inside the lamp room",
                       "paid_by_fan": st.tables["accounts"][acct]["balance"] == bal0 - 300}
    # --- newsletters: subscribe, send, mail delivered
    nl = A.ok("letters.create", name="Lamp letters")["newsletter"]
    Cc.ok("letters.subscribe", newsletter=nl)
    iss = A.ok("letters.send", newsletter=nl, ref=art, title="Issue one")
    w.advance_time(w.now + 3600)
    inbox = Cc.ok("mail.inbox", limit=10)
    res["letters"] = {"published": st.tables["content"][iss["item"]]["kind"] == "newsletter",
                      "mailed": iss["mailed"] == 1 and any("Issue one" in x["subject"] for x in inbox)}
    # --- product discovery: a launch links a real market listing; upvote; visit reaches the listing page
    lst = D.ok("market.list", kind="service", title="Lighthouse tours", description="guided tours", price_cents=2500)
    L = D.ok("launch.post", ref=D.ok("studio.image", text="Tours")["ref"], title="Lighthouse tours", kind="image",
             listing=lst["listing"])
    B.ok("launch.upvote", id=L["id"])
    v = B.ok("launch.visit", id=L["id"])
    res["launch"] = {"linked": L["link"] == "listing" and L["maker"],
                     "ranked": B.ok("launch.today")["launches"][0]["upvotes"] == 1,
                     "visit": v["page"]["title"] == "Lighthouse tours" and v["next"]["buy"] == "market.buy"}
    # --- developer community: Q&A and a package registry
    q = B.ok("dev.ask", title="Parsing tide tables in python?", body="Which library?", tags=["python"])
    an = Cc.ok("dev.answer", question=q["question"], body="Write a small parser; see tidecalc.")
    B.ok("dev.accept", question=q["question"], answer=an["answer"])
    pkg = Cc.ok("studio.package", name="tidecalc", version="0.1.0", files={"tidecalc.py": "def f():\n    return 1\n"})
    pid = Cc.ok("dev.publish", ref=pkg["ref"], title="tidecalc 0.1.0", kind="package")["id"]
    Cc.ok("dev.register", id=pid)
    res["dev"] = {"qa": D.ok("dev.question", id=q["question"])["accepted"] == an["answer"],
                  "registry": D.ok("dev.package", name="tidecalc")["latest"] == "0.1.0"}
    # --- knowledge sharing: Q&A with votes, an educational product
    kq = D.ok("kb.ask", title="Why do tides vary?", body="Springs and neaps?")
    ka = A.ok("kb.answer", question=kq["question"], body="The moon and the sun.")
    B.ok("kb.vote", id=ka["answer"], value=1)
    D.ok("kb.accept", question=kq["question"], answer=ka["answer"])
    course = A.ok("studio.course", title="Tides", lessons=[{"title": "Moon", "body": "Gravity."}])
    A.ok("kb.publish", ref=course["ref"], title="Tides course", kind="article")
    res["kb"] = {"qa": A.ok("kb.reputation_of")["reputation"] == 25,
                 "course": len(B.ok("kb.courses")) == 1}
    search_checks["thread"] = ("tides vary", kq["question"])
    search_checks["package"] = ("tidecalc", pid)
    search_checks["newsletter"] = ("issue", iss["item"])
    found = {k: any(r["id"] == cid for r in D.ok("search.query", q=q, kind=k, limit=20)["results"])
             for k, (q, cid) in search_checks.items()}
    res["search"] = {f"kind_{k}": v for k, v in found.items()}
    # --- one identity space: alice is the same principal (and social node) everywhere she published
    creators_alice = {r["platform"] for r in st.tables["content"].values() if r["creator"] == P["alice"]}
    assert w.check_invariants() == []
    ok = {p: all(v.values()) for p, v in res.items()}
    present = {p for p in PLATFORMS if p in res}
    metrics = {"platform_kinds": sum(1 for p in ok if ok[p]), "platform_kinds_registered": len(PLATFORMS),
               "platform_kinds_tested": len(present), "catalogue_items": len(st.tables["content"]),
               "platforms_one_creator_published_on": len(creators_alice)}
    for p, v in res.items():
        metrics[f"{p}_ok"] = bool(ok[p])
    evidence.record("SP-4", metrics, command=CMD, notes=(
        "Every platform kind in one world (crucible.social.modules.social_internet_modules + market); each "
        "platform's characteristic use exercised by participants through tools and checked in the shared "
        "catalogue, follow graph, identity and bank state."))
    assert_met('SP-4', metrics)
    assert all(ok.values()), {p: v for p, v in res.items() if not ok[p]}
