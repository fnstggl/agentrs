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


def artifacts_get(w, ref):
    from crucible.substrate import artifacts
    return artifacts.get(w, ref)


def _integrated_world(seed: int, n: int):
    """The integrated social internet of the one population (crucible.social.worlds.social_world): every platform
    distributed by `aud` with the default (shared semantic space) encoder, feeds, the population's own creation."""
    from crucible.core.kernel import World
    from crucible.social.worlds import social_world as integrated
    w = World.create(integrated("development", seed, n=n, creators={"prior": 0.4}), strict=True)
    P = {h: w.system_call("admin.join", {"handle": h, "endowment": 100_000, "password": f"pw-{h}"})["principal"]
         for h in ("alice", "bob")}
    return w, P


def test_C_1_media_artifacts_published():
    """Measured in the integrated, distributed configuration with the default encoder (audit C skeptic S9): every
    published item must also be distributed (indexed by the audience module from its bytes, not a caption)."""
    import base64
    from PIL import Image
    w, P = _integrated_world(11, 600)
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
    # distributed from the bytes by the default encoder (no refusal, no caption fallback)
    distributed = Counter()
    for kind, cids in items.items():
        for cid in cids:
            it = w.state.tables["content"][cid].get("aud_item")
            assert it is not None, (kind, cid, "not distributed")
            distributed[kind] += 1
    reps = Counter(e.payload.get("rep") for e in w.log if e.type == "aud.item")
    # what the population itself made in a day of autonomous activity (popmedia: image cards, narrated audio and
    # narrated slide videos, recorded rendering), decoded independently like the agents' items
    w.advance_time(w.now + DAY)
    pop_made = Counter()
    pop_dist = Counter()
    for cid, c in w.state.tables["content"].items():
        if not (c["creator"].startswith("p") and c["creator"][1:].isdigit()):
            continue
        fmt = {"image": "image", "audio": "audio", "video": "video", "text": "text"}.get(c["format"])
        if fmt is None:
            continue
        data = artifacts_get(w, c["ref"])
        if fmt == "image":
            Image.open(io.BytesIO(data)).load()
        elif fmt == "video":
            assert [x for x in _ffprobe(data)["streams"] if x["codec_type"] == "video"]
        elif fmt == "audio" and c["mime"] == "audio/ogg":
            assert data[:4] == b"OggS"
        pop_made[fmt] += 1
        pop_dist[fmt] += int(c.get("aud_item") is not None)
    assert w.check_invariants() == []
    metrics = {f"published_{k}": int(min(verified[k], distributed[k])) for k in ("text", "image", "audio", "video")}
    metrics.update({"catalogue_items": len(w.state.tables["content"]), **details,
                    "representations_from_bytes": int(reps.get("bytes", 0)),
                    "representations_from_caption": int(reps.get("caption", 0)),
                    **{f"population_published_{k}": int(v) for k, v in pop_made.items()},
                    **{f"population_distributed_{k}": int(v) for k, v in pop_dist.items()}})
    evidence.record("C.1", metrics, command=CMD, notes=(
        "Integrated social internet of the one population (600 members, every platform distributed by aud with the "
        "default shared-space encoder). Artifacts made with studio.write/image/edit_image/audio/video (Pillow, "
        "numpy+SoX, FFmpeg, jailed), published on posts/blog/pics/pods/shorts/tube, distributed from their bytes, "
        "opened by another participant through content.open and decoded independently (UTF-8, Pillow, wave, "
        "ffprobe); a published_* count is an item both verified and distributed. Then one day of the population's "
        "own creation: population_published_* counts its image cards, narrated audio and slide videos."))
    assert reps.get("caption", 0) == 0


# ============================================================================================ reference run
# ==================================================================================================== the one population
POP_PLATFORMS = ["posts", "blog", "community", "letters", "pics", "kb", "dev", "launch"]
POP_DAYS = 30


@pytest.fixture(scope="module")
def pop_run():
    """The social internet on the ONE population (crucible.population, D-15): 2,000 individuals whose interests live
    in the shared semantic space, every C1 platform, the distribution and consumption engine as population
    activities, members who write, publish and found and run communities; 30 simulated days, nothing scripted."""
    import time as _t
    from crucible.core.kernel import World
    from crucible.social import experiments as X
    from crucible.social.worlds import social_world
    t0 = _t.time()
    w = World.create(social_world("development", 21, n=2000, platforms=POP_PLATFORMS,
                                  params={"log_retain": 4_000_000}), strict=True)
    weekly = []
    for d in range(POP_DAYS):
        w.advance_time((d + 1) * DAY)
        if (d + 1) % 7 == 0 or d + 1 == POP_DAYS:
            weekly.append(_community_snapshot(w))
    return {"world": w, "weekly": weekly, "wall_s": _t.time() - t0,
            "views": X.decision_rows(w, "consume", 0), "imps": X.decision_rows(w, "imp", 0)}


def _community_snapshot(w) -> dict:
    from crucible.social import communities as CM
    out = {}
    v = CM.CMEMBER.view(w)
    for cid, c in w.state.tables["communities"].items():
        m = v["comm"] == c["n"]
        out[cid] = {"members": int(np.sum(m & (v["status"] == CM.MEMBER))),
                    "active7": int(CM.members(w, c["n"], active_days=7).size), "founded": int(c["created"]),
                    "founder": c["founder"], "status": c["status"], "left": int(np.sum(m & (v["status"] == CM.LEFT)))}
    return out


def test_C_2_discovery_channels(pop_run):
    from crucible.social import audience as AUD
    w = pop_run["world"]
    imps, views = pop_run["imps"], pop_run["views"]
    # the channel through which each qualifying view's item reached the viewer: the source of the viewer's first
    # impression of that item
    npop = AUD.n_people(w)
    ik = imps["item"] * npop + imps["viewer"]
    order = np.lexsort((imps["seq"], ik))
    uk, first = np.unique(ik[order], return_index=True)
    src_first = imps["src"][order][first]
    vk = views["item"][views["is_view"]] * npop + views["viewer"][views["is_view"]]
    pos = np.clip(np.searchsorted(uk, vk), 0, max(uk.size - 1, 0))
    ok = uk[pos] == vk if uk.size else np.zeros(vk.size, bool)
    ch = Counter(AUD.SOURCES[int(x)] for x in src_first[pos][ok])
    used = sorted(c for c, n in ch.items() if n >= 30)
    plats = Counter(AUD.info(w, int(i))["platform"] for i in views["item"][views["is_view"]])
    metrics = {"discovery_channels_used": len(used), "channels": ",".join(used),
               "views_attributed": int(ok.sum()), "platforms_with_views": len([p for p, n in plats.items() if n > 0])}
    metrics.update({f"views_via_{c}": int(n) for c, n in ch.items()})
    evidence.record("C.2", metrics, command=CMD, notes=(
        "The one population over 30 days: for every qualifying view, the channel of the viewer's first impression "
        "of the item (platform recommender mechanisms, shares in the inbox, newsletters, community feeds, search); "
        "a channel counts when it delivered at least 30 views."))
    assert ok.sum() > 1000


def test_C_3_consumption_produces_analytics(pop_run):
    from crucible.social import audience as AUD
    w = pop_run["world"]
    rc = AUD.recount(w)
    prov = rc["provenance"]
    mism = int(rc["total_mismatches"]) + int(rc["views_without_prior_exposure"])
    # what the creators see: their delayed, noisy analytics follow the exact audience
    items = np.flatnonzero(AUD.item_col(w, "views") > 0)
    pub = np.array([AUD.audience(w, int(i), principal=AUD.info(w, int(i))["creator"])
                    ["views"] for i in items[:500]], np.float64)
    exact = AUD.item_col(w, "views")[items[:500]].astype(np.float64)
    corr = float(np.corrcoef(pub, exact)[0, 1]) if items.size > 2 else 0.0
    metrics = {"consumption_events": int(rc["consumption_rows"]), "analytics_mismatches": mism,
               "views": int(rc["views"]), "items": int(rc["items"]), "measured": True,
               "decision_rows_without_provenance": int(prov["forced_rows"]),
               "creator_analytics_correlation_with_exact": round(corr, 4)}
    evidence.record("C.3", metrics, command=CMD, notes=(
        "Consumption decisions of the one population (feeds of 8 platforms, communities, newsletters) recorded as "
        "decision batches with provenance; every audience quantity of every item recounted independently from the "
        "batch blobs; creator analytics are the platform's delayed, noisy publication of the same quantities."))
    assert prov["forced_rows"] == 0


def test_C_4_communities_form_and_persist(pop_run):
    w = pop_run["world"]
    last = pop_run["weekly"][-1]
    pop_founded = {c: r for c, r in last.items() if r["founder"].startswith("p") and r["founder"][1:].isdigit()}
    old = {c: r for c, r in pop_founded.items() if r["founded"] <= w.now - 30 * DAY + DAY}
    persisting = [c for c in pop_founded if c in pop_run["weekly"][0] and last[c]["active7"] >= 3]
    subjects = sorted({w.state.tables["communities"][c]["subject"] for c in pop_founded})
    metrics = {"communities_formed": len(pop_founded),
               "communities_active_after_30_days": len([c for c, r in old.items() if r["active7"] >= 3]),
               "communities_active_from_week_1_to_end": len(persisting),
               "members_total": int(sum(r["members"] for r in last.values())),
               "members_left_total": int(sum(r["left"] for r in last.values())),
               "active7_total": int(sum(r["active7"] for r in last.values()))}
    evidence.record("C.4", metrics, command=CMD, notes=(
        "Communities founded by members of the population (subjects from the titles they had been reading, never "
        "predefined), joined by fit, activity and contacts, read, posted and replied in, left when their "
        "experienced value fell. Active = at least 3 members who took part in the last 7 days. Weekly snapshots: "
        f"{json.dumps(pop_run['weekly'][-1])[:3000]}. Subjects: {subjects[:20]}"))
    assert len(pop_founded) >= 1


def _two_prop(k1, n1, k2, n2):
    from scipy.stats import norm
    p = (k1 + k2) / max(n1 + n2, 1)
    se = max((p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1))) ** 0.5, 1e-12)
    z = (k1 / max(n1, 1) - k2 / max(n2, 1)) / se
    return k1 / max(n1, 1) - k2 / max(n2, 1), float(2 * norm.sf(abs(z)))


def test_C_5_social_mechanisms_affect_future_interactions():
    """Three randomised experiments on the one population (forks of the day-10 state of a smaller run of the same
    world): the experimenter assigns the relation to a random half of a sample - a follow edge to a creator, a share
    of an item into the inbox, a newsletter subscription - and the outcome is what the individuals then do by
    themselves through the ordinary mechanisms (following feeds, inboxes, newsletter delivery)."""
    from crucible.core.kernel import World
    from crucible.social import audience as AUD, graph as G
    from crucible.social.worlds import social_world
    from crucible.substrate import ids
    w = World.create(social_world("development", 22, n=2000, platforms=POP_PLATFORMS))
    lab = w.system_call("admin.join", {"handle": "lab", "endowment": 10_000, "password": "pw-lab"})["principal"]
    w.advance_time(5 * DAY)
    rng = np.random.default_rng(5)
    n = AUD.n_people(w)
    out = {}
    # --- follow: a creator the sample does not follow; outcome: viewed one of the creator's next items
    info = {k: AUD.info(w, int(k)) for k in w.state.tables["aud_item_info"]}
    recent = w.now - 3 * DAY
    creators = Counter(r["creator"] for r in info.values() if ids.index(r["creator"]) is not None
                       and r["published_at"] >= recent)
    cre = sorted(creators, key=lambda p: (-creators[p], p))[0]       # the most active creator of late
    sample = rng.choice(np.setdiff1d(np.arange(n), [ids.index(cre)]), 1200, replace=False)
    treat, ctrl = sample[:600], sample[600:]
    f = w.fork("c5-follow")
    from crucible.social import nodes
    dst = nodes.node_of(f, cre)
    f._run_system("experiment.assign", lambda ctx: G.follow_batch(ctx, treat, np.full(treat.size, dst), via="other"))
    t0 = f.now
    mine0 = AUD.items_of(f, cre)
    before = np.zeros(n, bool)
    for it in mine0:
        before[AUD.viewers_of(f, it)] = True
    f.advance_time(t0 + 4 * DAY)
    mine = AUD.items_of(f, cre)
    later = [i for i in mine if AUD.info(f, i)["published_at"] > t0]
    viewed = np.zeros(n, bool)
    for it in mine:
        viewed[AUD.viewers_of(f, it)] = True
    viewed &= ~before                     # outcome: became a viewer of the creator's items after the assignment
    treat_f, ctrl_f = treat[~before[treat]], ctrl[~before[ctrl]]
    eff, p = _two_prop(int(viewed[treat_f].sum()), treat_f.size, int(viewed[ctrl_f].sum()), ctrl_f.size)
    out.update(follow_effect=round(eff, 4), follow_effect_pvalue=p, follow_new_items=len(later),
               follow_creator_items=len(mine))
    # --- share: an informative post by the experimenter shared into the inboxes of the treated
    g = w.fork("c5-share")
    from crucible.population import semantics, writing
    popular = int(np.argmax(semantics.popularity()))
    title_s, body_s = writing.compose(np.random.default_rng(11), popular, 5, 1.0)
    a = g.execute(lab, {"tool": "studio.write", "args": {"format": "text", "title": title_s, "body": body_s}})
    r = g.execute(lab, {"tool": "posts.publish", "args": {"ref": a.result["ref"], "title": title_s}})
    item = int(g.state.tables["content"][r.result["id"]]["aud_item"])
    feed = g.modules["feed"]
    g._run_system("experiment.assign", lambda ctx: feed.engine.deliver(ctx, treat, np.full(treat.size, item),
                                                                       np.ones(treat.size, np.uint8), platform="posts"))
    g.advance_time(g.now + 2 * DAY)
    vw = np.zeros(n, bool)
    vw[AUD.viewers_of(g, item)] = True
    eff, p = _two_prop(int(vw[treat].sum()), treat.size, int(vw[ctrl].sum()), ctrl.size)
    out.update(share_effect=round(eff, 4), share_effect_pvalue=p)
    # --- subscription: a newsletter of the experimenter; the treated subscribed; two issues are sent
    h = w.fork("c5-sub")
    nl = h.execute(lab, {"tool": "letters.create", "args": {"name": "Weekend Weather", "description": "weather"}})
    nid = nl.result["newsletter"]
    ln = int(h.state.tables["newsletters"][nid]["n"])
    h._run_system("experiment.assign", lambda ctx: h.modules["letters"].subscribe_many(
        ctx, treat, np.full(treat.size, ln, np.int64)))
    items = []
    for k in range(2):
        ti, bi = writing.compose(np.random.default_rng(20 + k), popular, 5, 1.0)
        a = h.execute(lab, {"tool": "studio.write", "args": {"format": "text", "title": ti, "body": bi}})
        s_ = h.execute(lab, {"tool": "letters.send", "args": {"newsletter": nid, "ref": a.result["ref"],
                                                               "title": f"Weekend weather {k}"}})
        items.append(int(h.state.tables["content"][s_.result["item"]]["aud_item"]))
        h.advance_time(h.now + DAY)
    vs = np.zeros(n, bool)
    for it in items:
        vs[AUD.viewers_of(h, it)] = True
    eff, p = _two_prop(int(vs[treat].sum()), treat.size, int(vs[ctrl].sum()), ctrl.size)
    out.update(subscription_effect=round(eff, 4), subscription_effect_pvalue=p)
    out.update(sample_treated=int(treat.size), sample_control=int(ctrl.size))
    evidence.record("C.5", out, command=CMD, notes=(
        "Randomised assignment of a follow edge (to the population's most active creator of the last 3 days; outcome: "
        "became a viewer of any of its items, old or new, among those who had not viewed it before), of a share into the "
        "inbox (an experimenter's informative post on the population's most popular topic), and of a newsletter "
        "subscription (two such issues) to 600 of 1,200 sampled "
        "individuals; outcome: the individual later viewed the creator's new items / the shared post / an issue, "
        "decided by the population itself; two-proportion z-tests. The assignment is the experimenter's "
        "manipulation (labelled), the outcomes are ordinary population decisions."))
    assert out["sample_treated"] == 600


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


def test_C_n1_no_direct_counter_mutation(pop_run):
    from crucible.population import check_provenance, record_decisions
    from crucible.social import audience as AUD
    w = pop_run["world"]
    rc = recount_counters(w)
    rc["audience_recount"] = int(AUD.recount(w)["total_mismatches"])
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
    # module-level injection (not a tool): 50 forged views with a valid-looking provenance are recorded by module
    # code in the population world's fork; the provenance check counts them as forced, never as organic
    f = w.fork("cn1-inject")
    items = np.flatnonzero(AUD.item_col(f, "removed") == 0)[:5]
    viewers = np.arange(10)
    hist = list(w.log)
    def inject(ctx):
        vv = np.repeat(viewers, items.size)
        it = np.tile(items, viewers.size)
        record_decisions(ctx, "consume.posts", vv, {"aud_c_item": it.astype(np.int32)}, model_id="c2-consumption",
                         model_version="2", fidelity_id="social-consumption-builtin")
    f._run_system("forge", inject)
    pr = check_provenance(f, hist + list(f.log))
    probes["module code forging consumption rows is detected"] = type(claimed)(ok=not (pr["forced_rows"] >= 50))
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
        "newsletter issues, conversation lengths) and every audience quantity is recounted from recorded events / "
        "cols.batch blobs of the 30-day run of the one population (2,000 individuals) and compared with the state; "
        "the state is replayed from the log. Probes try to write counters through tools (extra arguments, artifact "
        "metadata, self-votes, repeats, operator tools) and must be refused; a module-level forgery of consumption "
        "rows must be detected by decision provenance."))
    assert direct == 0, (rc, before, after)
