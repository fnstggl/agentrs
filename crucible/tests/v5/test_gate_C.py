"""Gate C (MASTER_SPEC Part 17): rich social ecosystem - measured on the real social internet modules.

C.1  actual text, image, audio and video artifacts are made with the studio tools, published, and verified by
     another participant with independent decoders (UTF-8, Pillow, the WAV header, ffprobe).
C.2  participants discover content through several channels: reference participants (crucible.social.reference)
     browse for 14 simulated days; every logged discovery is checked against recorded world state (a follow edge
     for the timeline, a membership for a community feed, a delivered message for a share, a delivered mail for a
     newsletter, an index entry for search, the platform for its latest list). Population-scale channels (the
     audience module's impression sources) are added when workstream C2's modules are in this tree.
C.3  consumption events produce analytics: needs workstream C2's audience module (the only way audience state
     changes); measured on it when it is in this tree, recorded as unmeasured otherwise.
C.4  communities form and persist: participants found communities on subjects of their own and still take part in
     them after 30 simulated days (at least 3 distinct contributors in the last week of a community 30+ days old).
C.5  sharing, following and subscriptions affect future interactions: three randomised experiments through the
     ordinary mechanisms (timeline, direct shares, newsletter mail); outcomes are the participants' recorded
     content.open commands; exact permutation / Fisher tests.
C.6  an end-to-end browser workflow (Playwright Chromium against the browser mode) changes real simulated state,
     verified in the world and against the same calls made headlessly (identical event-hash chain).
C.n1 no direct counter mutation: every counter of the social modules equals its recount from recorded events, and
     probes that try to write counters are refused.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from collections import Counter

import numpy as np
import pytest

from crucible.acceptance.v5 import evidence
from social_helpers import Actor, social_world

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_C.py"
DAY = 86400


def opens(w, principal: str | None = None) -> list[tuple[str, str, int]]:
    """(actor, content id, time) of every successful content.open command in the log (the evaluator's record of
    what participants opened)."""
    out = []
    for e in w.log:
        if e.type == "cmd" and e.payload.get("tool") == "content.open" and (principal is None or e.actor == principal):
            out.append((e.actor, e.payload["args"]["id"], e.t))
    return out


# ============================================================================================ C.1
def _ffprobe(data: bytes) -> dict:
    exe = shutil.which("ffprobe")
    assert exe, "ffprobe is needed to verify videos independently"
    with tempfile.NamedTemporaryFile(suffix=".mp4") as fh:
        fh.write(data)
        fh.flush()
        out = subprocess.run([exe, "-v", "error", "-show_streams", "-show_format", "-of", "json", fh.name],
                             capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr[-300:]
    return json.loads(out.stdout)


def test_C_1_media_artifacts_published():
    import base64
    from PIL import Image
    w, P = social_world(seed=11)
    A, B = Actor(w, P["alice"]), Actor(w, P["bob"])
    text = A.ok("studio.write", format="markdown", title="Notes on tides", body="# Tides\nThe tide turns twice a day.")
    page = A.ok("studio.write", format="html", title="Harbour guide", body="Moorings are on the **east** quay.")
    img = A.ok("studio.image", text="Harbour at dawn", title="Dawn", width=640)
    scene = A.ok("studio.image", shapes=[{"shape": "rect", "xy": [0, 0, 640, 360], "fill": "navy"},
                                          {"shape": "ellipse", "xy": [260, 120, 380, 240], "fill": "yellow"}],
                 width=640, height=360)
    edited = A.ok("studio.edit_image", ref=scene["ref"], ops=[{"op": "text", "text": "Sunrise", "xy": [20, 20]}])
    snd = A.ok("studio.audio", kind="melody", notes=["C4", "E4", "G4", None, "C5", "G4"], bpm=96,
               effects=[{"effect": "fade", "in_s": 0.2, "out_s": 0.5}])
    vid = A.ok("studio.video", images=[img["ref"], edited["ref"]], audio=snd["ref"], seconds_per_image=4)
    long_imgs = [A.ok("studio.image", text=f"Chapter {k}: rigging a dinghy")["ref"] for k in range(1, 5)]
    long_snd = A.ok("studio.audio", kind="chord", notes=["A3", "C4", "E4"], seconds=72.0)
    lvid = A.ok("studio.video", images=long_imgs, audio=long_snd["ref"], seconds_per_image=18)
    items = {
        "text": [A.ok("posts.publish", ref=text["ref"], title="Notes on tides")["id"],
                 A.ok("blog.publish", ref=page["ref"], title="Harbour guide")["id"]],
        "image": [A.ok("pics.publish", ref=img["ref"], title="Dawn")["id"],
                  A.ok("pics.publish", ref=edited["ref"], title="Sunrise")["id"]],
        "audio": [A.ok("pods.publish", ref=snd["ref"], title="Harbour tune")["id"]],
        "video": [A.ok("shorts.publish", ref=vid["ref"], title="Harbour short")["id"],
                  A.ok("tube.publish", ref=lvid["ref"], title="Rigging a dinghy", kind="video_long")["id"]],
    }
    verified = Counter()
    details = {}
    for kind, cids in items.items():
        for cid in cids:
            got = B.ok("content.open", id=cid)              # another participant opens the published item
            data = base64.b64decode(got["data_b64"])
            info = w.state.tables["artifacts"][w.state.tables["content"][cid]["ref"]]
            assert info["published"] and not w.state.tables["content"][cid]["removed"]
            if kind == "text":
                s = data.decode("utf-8")
                assert ("tide" in s.lower()) or ("moorings" in s.lower())
            elif kind == "image":
                im = Image.open(io.BytesIO(data))
                im.load()
                assert im.size[0] == 640
                details[f"{cid}_size"] = f"{im.size[0]}x{im.size[1]}"
            elif kind == "audio":
                with wave.open(io.BytesIO(data)) as wf:
                    dur = wf.getnframes() / wf.getframerate()
                assert dur > 1.0
                details[f"{cid}_seconds"] = round(dur, 3)
            else:
                pr = _ffprobe(data)
                vs = [s for s in pr["streams"] if s["codec_type"] == "video"]
                assert vs and float(pr["format"]["duration"]) > 1.0
                details[f"{cid}_seconds"] = round(float(pr["format"]["duration"]), 2)
                details[f"{cid}_codec"] = vs[0]["codec_name"]
            verified[kind] += 1
    assert w.state.tables["content"][items["video"][1]]["kind"] == "video_long"
    assert w.check_invariants() == []
    metrics = {f"published_{k}": int(verified[k]) for k in ("text", "image", "audio", "video")}
    metrics.update({"catalogue_items": len(w.state.tables["content"]), **details})
    evidence.record("C.1", metrics, command=CMD, notes=(
        "Artifacts made with studio.write/image/edit_image/audio/video (Pillow, numpy+SoX, FFmpeg), published on "
        "posts/blog/pics/pods/shorts/tube, opened by another participant through content.open and decoded "
        "independently (UTF-8, Pillow, wave, ffprobe)."))


# ============================================================================================ reference run
@pytest.fixture(scope="module")
def reference_run():
    from crucible.social.reference import make_participants, run_days
    w, P = social_world(seed=7, people=())
    ps = make_participants(w, 60, seed=7)
    snaps = {}

    def hook(d):
        if d == 13:
            snaps["day14"] = (w.seq, [list(p.log) for p in ps])
    run_days(w, ps, 36, hook=hook)
    return w, ps, snaps


def _verify_discovery(w, p, day: int, channel: str, cid: str) -> bool:
    from crucible.social import communities as CM
    from crucible.social import graph as G
    from crucible.social import nodes
    st = w.state
    rec = st.tables["content"].get(cid)
    if rec is None:
        return False
    t_end = (day + 1) * DAY
    me = nodes.node_of(w, p.principal)
    if channel == "following":
        v = G.FOLLOW.view(w)
        ok = (v["src"] == me) & (v["dst"] == rec["creator_node"]) & (v["t"] < t_end)
        if ok.any():
            return True
        sh = G.SHARE.view(w)
        fol = v["dst"][(v["src"] == me) & (v["t"] < t_end)]
        return bool(((sh["item"] == rec["n"]) & np.isin(sh["sharer"], fol) & (sh["to"] == -1)).any())
    if channel == "community":
        c = st.tables["communities"].get(rec["community"] or "")
        return c is not None and CM.member_row(w, me, c["n"]) is not None
    if channel == "share":
        for k in st.find_ids("msg_members", "member", p.principal):
            conv = st.tables["msg_members"][k]["conversation"]
            for mid in st.find_ids("msg_messages", "conversation", conv):
                m = st.tables["msg_messages"][mid]
                if m["item"] == cid and m["sender"] != p.principal and m["t"] < t_end:
                    return True
        return False
    if channel == "newsletter":
        for mid, m in st.tables["mail_msgs"].items():
            if p.principal in m["to"] and f"content {cid}" in m["body"] and m["t"] < t_end:
                return True
        return False
    if channel == "search":
        return st.has("search_docs", cid)
    if channel in ("latest", "launch"):
        return not rec["removed"] or rec["removed_at"] >= day * DAY
    return False


def test_C_2_discovery_channels(reference_run):
    w, ps, snaps = reference_run
    seq14, logs14 = snaps["day14"]
    used, checked, failed = Counter(), 0, 0
    opened = {(a, c) for a, c, _ in opens(w)}
    for p, log in zip(ps, logs14):
        for day, channel, cid, _rel in log:
            checked += 1
            if (p.principal, cid) in opened and _verify_discovery(w, p, day, channel, cid):
                used[channel] += 1
            else:
                failed += 1
    pop_sources = 0
    if "aud" in w.modules:
        from crucible.social import audience as AUD
        pop_sources = len({s for b in AUD.batches(w, kind="impress") for s in b.get("sources", [])})
    metrics = {"discovery_channels_used": int(sum(1 for c in used if used[c] > 0)) + pop_sources,
               "participant_channels_verified": int(sum(1 for c in used if used[c] > 0)),
               "population_impression_sources": pop_sources, "discoveries_checked": checked,
               "discoveries_unverified": failed, "participants": len(ps), "days": 14}
    for c, n in sorted(used.items()):
        metrics[f"opened_via_{c}"] = int(n)
    assert checked > 0 and failed == 0, f"{failed} logged discoveries could not be verified in the world"
    evidence.record("C.2", metrics, command=CMD, notes=(
        "60 reference participants (scripted, interest-driven, acting only through tools) for the first 14 days of "
        "a 36-day run; a channel counts when participants opened items they found there and the discovery is "
        "confirmed by recorded state. Population channels are added from the audience module's impression sources "
        "when workstream C2's modules are present (not in this tree when the count is 0)."))


# ============================================================================================ C.3
def test_C_3_consumption_produces_analytics():
    try:
        from crucible.social import audience as AUD            # workstream C2
    except ImportError:
        AUD = None
    if AUD is None:
        evidence.record("C.3", {"consumption_events": 0, "analytics_mismatches": -1, "measured": False},
                        command=CMD, notes=(
                            "NOT MEASURED in this tree: consumption is recorded only by workstream C2's audience "
                            "module (crucible.social.audience), which is not merged here. analytics_mismatches = -1 "
                            "means 'not measured'. C1's catalogue indexes every public item for it "
                            "(crucible.social.integration) and content.inspect reports its creator analytics."))
        pytest.fail("C.3 needs workstream C2's audience module (crucible.social.audience), not in this tree")
    from crucible.social.reference import make_participants, run_days
    w, P = social_world(seed=3, extra=["pop", "aud", "feed"], people=("alice",))
    ps = make_participants(w, 30, seed=3)
    run_days(w, ps, 7)
    rep = AUD.recount(w)
    n_cons = sum(int(b.get("n", 0)) for b in AUD.batches(w, kind="consume"))
    mism = int(rep.get("mismatches", 0)) if isinstance(rep, dict) else -1
    creator_view = 0
    for cid, rec in w.state.tables["content"].items():
        if rec.get("aud_item") is not None:
            got = w.query(rec["creator"], "content.inspect", {"id": cid})
            creator_view += int(got["audience"] is not None)
    evidence.record("C.3", {"consumption_events": n_cons, "analytics_mismatches": mism, "measured": True,
                            "creator_analytics_items": creator_view}, command=CMD)


# ============================================================================================ C.4
def test_C_4_communities_form_and_persist(reference_run):
    w, ps, _ = reference_run
    st = w.state
    cm = w.modules["community"]
    by_participants = {p.principal for p in ps}
    formed = [c for c in st.tables["communities"].values() if c["founder"] in by_participants]
    active, rows = 0, []
    for c in formed:
        h = cm.health(w, dict(c))
        alive = h["age_days"] >= 30 and h["weekly_contributors"][0] >= 3 and c["status"] == "active"
        active += int(alive)
        rows.append((c["id"], h["age_days"], h["members"], h["weekly_contributors"][0], h["contributors_every_week_4w"]))
    subjects = {c["subject"] for c in formed}
    # persistence: the communities and memberships survive a snapshot round trip and a replay of the log
    from crucible.core.kernel import World
    w2 = World.from_blob(w.snapshot_blob())
    same = all(w2.modules["community"].health(w2, dict(w2.state.tables["communities"][c["id"]])) ==
               cm.health(w, dict(c)) for c in formed)
    assert same and w.check_invariants() == []
    metrics = {"communities_formed": len(formed), "communities_active_after_30_days": active,
               "distinct_subjects": len(subjects), "run_days": 36, "participants": len(ps),
               "memberships_ever": int(sum(cm.health(w, dict(c))["ever_members"] for c in formed)),
               "snapshot_roundtrip_identical": bool(same)}
    metrics["largest_community_members"] = max((r[2] for r in rows), default=0)
    evidence.record("C.4", metrics, command=CMD, notes=(
        "Communities founded by reference participants (subjects written by the founders) during a 36-day run; "
        "active after 30 days = age >= 30 days and >= 3 distinct members posting or commenting in the last 7 days."))


# ============================================================================================ C.5
def _permutation_p(x: np.ndarray, y: np.ndarray, rng, n: int = 20000) -> float:
    """One-sided exact-ish permutation test of mean(x) > mean(y)."""
    obs = x.mean() - y.mean()
    pool = np.concatenate([x, y])
    k = x.size
    cnt = 0
    for _ in range(n):
        p = rng.permutation(pool)
        cnt += (p[:k].mean() - p[k:].mean()) >= obs - 1e-12
    return (cnt + 1) / (n + 1)


def _cohort(w, n: int, topic: str, seed: int, prefix: str):
    from crucible.social.reference import ReferenceParticipant
    out = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        h = f"{prefix}{i}"
        p = w.system_call("admin.join", {"handle": h, "endowment": 20_000, "kind": "npc"})["principal"]
        out.append(ReferenceParticipant(w, p, h, [topic], float(rng.uniform(0.2, 0.9)), False, seed * 100 + i,
                                        attention=4))
    return out


def test_C_5_social_mechanisms_affect_future_interactions():
    from scipy.stats import fisher_exact
    from crucible.social.reference import TOPICS, run_days
    rng = np.random.default_rng(55)
    w, P = social_world(seed=55, people=("cre", "sharer"))
    C, S = Actor(w, P["cre"]), Actor(w, P["sharer"])
    words = TOPICS["sailing"]

    def article(k: int, where: str = "posts"):
        body = " ".join(words[(k + j) % len(words)] for j in range(30))
        a = C.ok("studio.write", format="text", title=f"Sailing log {k}", body=body)
        return C.ok(f"{where}.publish", ref=a["ref"], title=f"Sailing log {k}", tags=["sailing"])["id"]
    # competing creators publish many other sailing items every day: attention is scarce, content is not
    others = [Actor(w, w.system_call("admin.join", {"handle": f"bg{k}", "endowment": 1000})["principal"])
              for k in range(4)]
    bg = [0]

    def background(d):
        for k in range(14):
            o = others[k % len(others)]
            body = " ".join(words[int(j)] for j in rng.integers(0, len(words), 30))
            a = o.ok("studio.write", format="text", title=f"Harbour chat {bg[0]}", body=body)
            o.ok(["posts", "blog"][k % 2] + ".publish", ref=a["ref"], title=f"Harbour chat {bg[0]}",
                 tags=["sailing"])
            bg[0] += 1
    background(-1)
    out = {}

    # 1. following: a random half follows the creator; the creator then publishes for 8 days
    g1 = _cohort(w, 40, "sailing", 1, "fo")
    treat = set(rng.choice(40, 20, replace=False).tolist())
    for i, p in enumerate(g1):
        if i in treat:
            assert p.call("social.follow", target=P["cre"], via="profile")
    items1 = []

    def publish_daily(d):
        background(d)
        items1.append(article(d))
    run_days(w, g1, 8, hook=publish_daily)
    cnt = {p.principal: sum(1 for a, c, _ in opens(w, p.principal) if c in set(items1)) for p in g1}
    x = np.array([cnt[g1[i].principal] for i in sorted(treat)], float)
    y = np.array([cnt[g1[i].principal] for i in range(40) if i not in treat], float)
    out["follow_effect_pvalue"] = _permutation_p(x, y, np.random.default_rng(1))
    out["follow_treated_mean_opens"] = round(float(x.mean()), 3)
    out["follow_control_mean_opens"] = round(float(y.mean()), 3)

    # 2. sharing: one item is sent to a random half of a fresh cohort; did they open it in the next 3 days?
    g2 = _cohort(w, 40, "sailing", 2, "sh")
    target = article(100, "blog")
    treat2 = set(rng.choice(40, 20, replace=False).tolist())
    S.ok("social.share", id=target, to=[g2[i].principal for i in sorted(treat2)][:20], note="look at this")
    run_days(w, g2, 3, hook=background)
    o2 = {a for a, c, _ in opens(w) if c == target}
    a_ = sum(1 for i in treat2 if g2[i].principal in o2)
    c_ = sum(1 for i in range(40) if i not in treat2 and g2[i].principal in o2)
    out["share_effect_pvalue"] = float(fisher_exact([[a_, 20 - a_], [c_, 20 - c_]], alternative="greater")[1])
    out["share_treated_opened"], out["share_control_opened"] = a_, c_

    # 3. subscriptions: a random half subscribes to the creator's newsletter; four issues are sent
    g3 = _cohort(w, 40, "sailing", 3, "nl")
    nl = C.ok("letters.create", name="Sailing letters", description="weekly sailing notes")["newsletter"]
    treat3 = set(rng.choice(40, 20, replace=False).tolist())
    for i in sorted(treat3):
        assert g3[i].call("letters.subscribe", newsletter=nl)
    issues = []

    def send(d):
        background(d)
        if d % 2 == 0:
            body = "<p>" + " ".join(words * 3) + "</p>"
            a = C.ok("studio.write", format="html", title=f"Issue {len(issues) + 1}", html=body)
            issues.append(C.ok("letters.send", newsletter=nl, ref=a["ref"], title=f"Sailing letter {len(issues) + 1}")
                          ["item"])
    run_days(w, g3, 8, hook=send)
    cnt3 = {p.principal: sum(1 for a, c, _ in opens(w, p.principal) if c in set(issues)) for p in g3}
    x3 = np.array([cnt3[g3[i].principal] for i in sorted(treat3)], float)
    y3 = np.array([cnt3[g3[i].principal] for i in range(40) if i not in treat3], float)
    out["subscription_effect_pvalue"] = _permutation_p(x3, y3, np.random.default_rng(3))
    out["subscription_treated_mean_opens"] = round(float(x3.mean()), 3)
    out["subscription_control_mean_opens"] = round(float(y3.mean()), 3)
    assert w.check_invariants() == []
    for k in ("follow_effect_pvalue", "share_effect_pvalue", "subscription_effect_pvalue"):
        assert 0.0 < out[k] <= 1.0
    out["participants"] = 120
    evidence.record("C.5", out, command=CMD, notes=(
        "Three randomised experiments with reference participants (scripted, acting through tools): following a "
        "creator (timeline), receiving a direct share (messaging), subscribing to a newsletter (mail delivery). "
        "Outcome: recorded content.open commands of the creator's later items / the shared item / the issues. "
        "One-sided permutation tests (follow, subscription) and Fisher's exact test (share)."))


# ============================================================================================ C.6
def _chromium_ok() -> bool:
    try:
        from crucible.web.browser import playwright_available
    except ImportError:
        return False
    return playwright_available()


@pytest.mark.browser
@pytest.mark.timeout(600)
def test_C_6_browser_workflow_against_real_state():
    assert _chromium_ok(), "Playwright/Chromium is required for gate C.6 (run through crucible.scale.slots)"
    from crucible.social import graph as G
    from crucible.social import nodes
    from crucible.social.web import make_social_web_app
    from crucible.web.browser import BrowserSession, serve_in_thread
    w, P = social_world(seed=21)
    Actor(w, P["bob"]).ok("posts.publish", ref=Actor(w, P["bob"]).ok("studio.write", format="text", title="b",
                                                                     body="Bob on knots")["ref"], title="Knots")
    seq0 = w.seq
    url, stop = serve_in_thread(make_social_web_app(w))
    steps = []
    b = None
    try:
        b = BrowserSession(url, timeout_ms=30_000)
        o = b.act({"action": "goto", "url": "id.crucible"})
        b.act({"action": "fill", "target": "login-handle", "value": "alice"})
        b.act({"action": "fill", "target": "Password", "value": "pw-alice"})
        o = b.act({"action": "click", "target": "Sign in"})
        steps.append(("login", o["ok"] and "Signed in as p:alice" in o["text"]))
        # 1. found a community
        b.goto("http://community.crucible/")
        b.fill("community-found-id", "knot-tyers")
        b.fill("community-found-name", "Knot tyers")
        b.fill("community-found-subject", "tying, testing and naming sailing knots")
        o = b.act({"action": "click", "target": "community-found-submit"})
        steps.append(("found community", o["ok"] and "/c?id=knot-tyers" in o["url"]))
        # 2. write a post in the studio
        b.goto("http://studio.crucible/")
        b.select("studio-write-format", "text")
        b.fill("studio-write-title", "The bowline")
        b.fill("studio-write-body", "Make a loop, the rabbit comes out of the hole and goes round the tree.")
        o = b.act({"action": "click", "target": "studio-write-submit"})
        steps.append(("create artifact", o["ok"] and "/made?ref=sha256" in o["url"]))
        # 3. follow the publish link to the community and post it there
        o = b.act({"action": "click", "target": "publish-community"})
        o = b.act({"action": "click", "target": "comm-knot-tyers"})
        b.fill("community-post-title", "The bowline")
        o = b.act({"action": "click", "target": "community-post-submit"})
        steps.append(("post in community", o["ok"] and "/item?id=c" in o["url"] and "The bowline" in o["title"]))
        cid = re.search(r"id=(c\d+)", o["url"]).group(1)
        # 4. comment on it
        b.fill("community-comment-text", "Remember to dress the knot.")
        o = b.act({"action": "click", "target": "community-comment-submit"})
        steps.append(("comment", o["ok"] and "dress the knot" in o["text"]))
        # 5. follow another creator from their profile
        b.goto("http://social.crucible/u?p=p:bob")
        o = b.act({"action": "click", "target": "social-follow-submit"})
        steps.append(("follow", o["ok"] and any("social.follow succeeded" in m["text"] for m in o["messages"])))
        # 6. the timeline shows the followed creator's item; search finds the new post
        b.goto("http://social.crucible/")
        o = b.observe()
        steps.append(("timeline", "Knots" in o["text"]))
        b.goto("http://search.crucible/")
        b.fill("search-query-q", "bowline rabbit")
        o = b.act({"action": "click", "target": "Search"})
        steps.append(("search", "The bowline" in o["text"]))
        blocked = list(b.blocked)
    finally:
        if b is not None:
            b.close()
        stop()
    # state changes, verified in the world itself
    st = w.state
    rec = st.tables["content"].get(cid)
    changes = {
        "community_founded": st.has("communities", "knot-tyers") and
        st.tables["communities"]["knot-tyers"]["founder"] == P["alice"],
        "post_in_catalogue": rec is not None and rec["community"] == "knot-tyers" and rec["creator"] == P["alice"],
        "comment_recorded": rec is not None and rec["comments"] == 1,
        "follow_edge": bool(G.is_following(w, [nodes.node_of(w, P["alice"])], [nodes.node_of(w, P["bob"])])[0]),
        "search_indexed": st.has("search_docs", cid),
    }
    # the same commands headlessly on a twin world give the identical event-hash chain
    w2, _ = social_world(seed=21)
    Actor(w2, P["bob"]).ok("posts.publish", ref=Actor(w2, P["bob"]).ok("studio.write", format="text", title="b",
                                                                       body="Bob on knots")["ref"], title="Knots")
    replayed = 0
    for e in w.log:
        if e.seq > seq0 and ((e.type == "cmd" and e.payload.get("n") is not None) or e.type == "cmd.rejected"):
            actor = e.payload.get("actor") or e.actor if e.type == "cmd.rejected" else e.actor
            w2.execute(actor, {"tool": e.payload["tool"], "args": e.payload["args"]})
            replayed += 1
    twin = (w2.seq, w2.last_hash, w2.digest()) == (w.seq, w.last_hash, w.digest())
    assert w.check_invariants() == []
    passed = sum(1 for _, ok in steps if ok)
    metrics = {"browser_workflow_steps_passed": passed, "browser_workflow_steps": len(steps),
               "browser_state_changes_verified": sum(1 for v in changes.values() if v),
               "headless_twin_identical": bool(twin), "blocked_requests": len(blocked),
               "recorded_commands": replayed}
    metrics.update({f"step_{k.replace(' ', '_')}": bool(ok) for k, ok in steps})
    metrics.update({f"state_{k}": bool(v) for k, v in changes.items()})
    evidence.record("C.6", metrics, command=CMD, notes=(
        "Playwright Chromium (confined to *.crucible) drives the browser mode (crucible.social.web): login, found "
        "a community, create an artifact in the studio, post it in the community, comment, follow a creator, see "
        "the timeline, search. Every form post is one recorded command; the same commands headlessly reproduce "
        "the event-hash chain."))
    assert passed == len(steps), steps
    assert all(changes.values()) and twin, (changes, twin)


# ============================================================================================ C.n1
def recount_counters(w) -> dict[str, int]:
    """Recompute every counter kept by the C1 modules from recorded events and recorded batches; returns the number
    of mismatches per family (0 = every counter is exactly what the recorded decisions imply)."""
    from crucible.core.kernel import World
    from crucible.social import communities as CM
    st = w.state
    bad = Counter()
    # comments per item, from content.comment / content.comment_removed events
    com = Counter()
    removed = {e.payload["id"] for e in w.log if e.type == "content.comment_removed"}
    for e in w.log:
        if e.type == "content.comment" and e.payload["id"] not in removed:
            com[e.payload["item"]] += 1
    for cid, r in st.tables["content"].items():
        bad["content.comments"] += int(r["comments"] != com.get(cid, 0))
    # community participation counters, from the cols.batch blobs of table cmember (add mode) applied in order
    from crucible.substrate import artifacts as A
    adds: dict[tuple[int, str], int] = Counter()
    for e in w.log:
        if e.type == "cols.batch" and e.payload["table"] == "cmember" and e.payload["mode"] == "add":
            arr = A.get_array_blob(w, e.payload["ref"])
            for c in e.payload["cols"]:
                for r_, v in zip(arr["idx"].tolist(), arr[f"v.{c}"].tolist()):
                    adds[(int(r_), c)] += int(v)
    v = CM.CMEMBER.view(w)
    for c in ("posts", "comments", "kudos_recv", "kudos_given", "attended", "removed", "warnings"):
        for r_ in range(v["node"].size):
            bad["community." + c] += int(int(v[c][r_]) != adds.get((r_, c), 0))
    # launch upvotes from launch.vote events
    up = Counter()
    for e in w.log:
        if e.type == "launch.vote":
            up[e.payload["id"]] += 1 if e.payload["up"] else -1
    for cid, L in st.tables["launches"].items():
        bad["launch.upvotes"] += int(L["upvotes"] != up.get(cid, 0))
    # Q&A scores from vote events
    for plat in ("dev", "kb"):
        sc = Counter()
        for e in w.log:
            if e.type == f"{plat}.vote":
                sc[e.payload["id"]] += e.payload["value"] - e.payload["prev"]
        for pid, p in st.tables[f"{plat}_posts"].items():
            bad[f"{plat}.score"] += int(p["score"] != sc.get(pid, 0))
    # newsletter issue counts, conversation message counts
    iss = Counter(e.payload["newsletter"] for e in w.log if e.type == "letters.issue")
    for nid, n in st.tables["newsletters"].items():
        bad["letters.issues"] += int(n["issues"] != iss.get(nid, 0))
    msgs = Counter(e.payload["conversation"] for e in w.log if e.type == "msg.message")
    for cid, c in st.tables["msg_conversations"].items():
        bad["msg.messages"] += int(c["n"] != msgs.get(cid, 0))
    # and the whole state is a function of the log
    bad["replay_digest"] = int(World.replay(w.spec, [e.to_dict() for e in w.log]).digest() != w.digest())
    return dict(bad)


def test_C_n1_no_direct_counter_mutation(reference_run):
    w, ps, _ = reference_run
    rc = recount_counters(w)
    # probes: attempts to write audience/social counters through the tools are refused and change nothing
    w2, P = social_world(seed=31, extra=["market"])
    A, B = Actor(w2, P["alice"]), Actor(w2, P["bob"])
    t = A.ok("studio.write", format="text", title="x", body="hello world of sailing")
    item = A.ok("posts.publish", ref=t["ref"], title="Hello")["id"]
    before = recount_counters(w2)
    snap = (dict(w2.state.tables["content"][item]),)
    probes = {
        "content.update with a comment count": A("content.update", id=item, ref=t["ref"], comments=10 ** 6),
        "publish with a views figure": A("posts.publish", ref=t["ref"], title="Hello again", views=10 ** 6),
        "artifact meta claiming views/quality is never a counter": None,
        "self-upvote of a launch": None,
        "self-vote on a Q&A answer": None,
        "self-kudos": None,
        "repeat upvote": None,
        "admin tool from an agent": A("admin.endow", principal=P["alice"], amount=10 ** 9),
        "operator column write from an agent": A("cols.batch", table="cmember", kind="x", ref="sha256:" + "0" * 64),
    }
    m = A.ok("artifact.put", data_b64="aGk=", mime="text/plain", kind="text", meta={"views": 10 ** 6, "quality": 1.0})
    claimed = A("posts.publish", ref=m["ref"], title="hi")
    # accepted as a publication, but the claimed figures reach no catalogue record, listing or counter
    rec_c = w2.state.tables["content"][claimed.result["id"]] if claimed.ok else {}
    inert = claimed.ok and "views" not in json.dumps(dict(rec_c)) and "quality" not in json.dumps(dict(rec_c)) and \
        "views" not in json.dumps(B.ok("posts.latest", limit=50))
    probes["artifact meta claiming views/quality is never a counter"] = type(claimed)(ok=not inert)
    lst = A.ok("market.list", kind="service", title="Lessons", description="sailing lessons", price_cents=500)
    img = A.ok("studio.image", text="launch")
    L = A.ok("launch.post", ref=img["ref"], title="Lessons", kind="image", listing=lst["listing"])
    probes["self-upvote of a launch"] = A("launch.upvote", id=L["id"])
    q = B.ok("kb.ask", title="Which knot?", body="for mooring")
    an = A.ok("kb.answer", question=q["question"], body="a cleat hitch")
    probes["self-vote on a Q&A answer"] = A("kb.vote", id=an["answer"], value=1)
    A.ok("community.found", id="harbour", name="Harbour", subject="harbour life")
    probes["self-kudos"] = A("community.kudos", community="harbour", member="alice")
    B.ok("launch.upvote", id=L["id"])
    probes["repeat upvote"] = B("launch.upvote", id=L["id"])
    blocked = {k: (r is not None and not r.ok) for k, r in probes.items()}
    after = recount_counters(w2)
    assert w2.state.tables["content"][item]["comments"] == snap[0]["comments"]
    assert w2.state.tables["launches"][L["id"]]["upvotes"] == 1
    assert all(blocked.values()), {k: (r.error if r is not None and not r.ok else None) for k, r in probes.items()}
    direct = int(sum(rc.values())) + int(sum(after.values())) + int(sum(before.values()))
    metrics = {"direct_counter_writes": direct, "counter_write_probes_blocked": int(sum(blocked.values())),
               "counter_write_probes": len(probes)}
    metrics.update({f"recount_{k.replace('.', '_')}": int(v) for k, v in rc.items()})
    evidence.record("C.n1", metrics, command=CMD, notes=(
        "Every counter of the C1 modules (comments, community participation counters, launch upvotes, Q&A scores, "
        "newsletter issues, conversation lengths) is recounted from recorded events / cols.batch blobs of a 36-day "
        "run with 60 participants and compared with the state; the state is replayed from the log. Probes try to "
        "write counters through tools (extra arguments, artifact metadata, self-votes, repeats, operator tools) "
        "and must be refused."))
    assert direct == 0, (rc, before, after)
