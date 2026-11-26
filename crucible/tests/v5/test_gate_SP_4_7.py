"""Gate SP-4.7 (MASTER_SPEC Part 4.7): communities with persistent membership, individual interests,
participation history, member-to-member interactions, content, shared activities, governance, moderation, trust,
entry and exit, inactive members, organizational change, and participant-founded subjects.

Each feature is exercised through the community tools by ordinary participants and verified in recorded state
(membership rows, the catalogue, the moderation log, proposals and votes). Individual interests are measured on a
run of reference participants (interest-driven, acting only through tools): members' private interests overlap
their community's subject more than non-members' do. `community_features` counts the verified features.
"""
from __future__ import annotations

import numpy as np
import pytest

from crucible.acceptance.v5 import evidence
from social_helpers import Actor, social_world, assert_met

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_7.py"
DAY = 86400


def _row(w, community: str, principal: str) -> dict | None:
    from crucible.social import communities as CM
    from crucible.social import nodes
    c = w.state.tables["communities"][community]
    n = nodes.node_of(w, principal)
    r = CM.member_row(w, n, c["n"]) if n is not None else None
    return None if r is None else CM.row_dict(w, r)


@pytest.mark.timeout(900)
def test_SP_4_7_community_features():
    from crucible.core.kernel import World
    w, P = social_world(seed=47, people=("ann", "ben", "cat", "dan", "eve", "fay"))
    ann, ben, cat, dan, eve, fay = (Actor(w, P[h]) for h in ("ann", "ben", "cat", "dan", "eve", "fay"))
    f: dict[str, bool] = {}

    def text(a, body, title="note"):
        return a.ok("studio.write", format="text", title=title, body=body)["ref"]

    # participant-founded subject: free text, nothing predefined
    subject = "restoring hand-built wooden rowing skiffs from the 1930s"
    ann.ok("community.found", id="skiff-restorers", name="Skiff restorers", subject=subject,
           rules={"governance": "vote", "quorum": 0.4, "vote_threshold": 0.5, "min_trust_propose": 1,
                  "inactive_days": 7, "prune_days": 20})
    found = ann.ok("community.directory", q="wooden skiffs")
    f["participant_founded_subjects"] = w.state.tables["communities"]["skiff-restorers"]["subject"] == subject and \
        found[0]["community"] == "skiff-restorers" and found[0]["founder"] == P["ann"]
    # entry: open join, then an approval-gated community and an invite-only one
    for a in (ben, cat, dan):
        a.ok("community.join", community="skiff-restorers")
    ann.ok("community.found", id="varnish-club", name="Varnish club", subject="varnish and oils", rules={"join": "approval"})
    ben.ok("community.join", community="varnish-club")
    pending = _row(w, "varnish-club", P["ben"])["status"] == 4
    ann.ok("community.approve", community="varnish-club", member="ben", accept=True)
    ann.ok("community.found", id="masters", name="Master builders", subject="invited master builders",
           rules={"join": "invite"})
    refused = cat("community.join", community="masters").ok is False
    ann.ok("community.invite", community="masters", member="cat")
    cat.ok("community.join", community="masters")
    # exit and re-entry keep the history (spells)
    dan.ok("community.leave", community="skiff-restorers")
    left = _row(w, "skiff-restorers", P["dan"])["status"] == 2
    dan.ok("community.join", community="skiff-restorers")
    rj = _row(w, "skiff-restorers", P["dan"])
    f["entry_exit"] = pending and _row(w, "varnish-club", P["ben"])["status"] == 1 and refused and \
        _row(w, "masters", P["cat"])["status"] == 1 and left and rj["status"] == 1 and rj["spells"] == 2
    # content: posts are catalogue items of the community; private communities hide posts from non-members
    p1 = ben.ok("community.post", community="skiff-restorers", ref=text(ben, "Steam-bending oak ribs", "Ribs"),
                title="Steam-bending ribs")["id"]
    eve.ok("community.found", id="secret-yard", name="Secret yard", subject="our private yard",
           rules={"visibility": "private"})
    sp = eve.ok("community.post", community="secret-yard", ref=text(eve, "Yard plans"), title="Yard plans")["id"]
    feed = cat.ok("community.feed", community="skiff-restorers")["items"]
    f["content"] = [i["id"] for i in feed] == [p1] and w.state.tables["content"][p1]["community"] == "skiff-restorers" \
        and cat("content.open", id=sp).ok is False and eve.ok("content.open", id=sp)["id"] == sp and \
        cat("community.feed", community="secret-yard").ok is False
    # member-to-member interactions: replies to another member's post, kudos
    c1 = cat.ok("community.comment", id=p1, text="Soak them a day first.")
    ben.ok("community.comment", id=p1, text="Thanks!", parent=c1["comment"])
    cat.ok("community.kudos", community="skiff-restorers", member="ben", note="great ribs")
    rb = _row(w, "skiff-restorers", P["ben"])
    f["member_interactions"] = rb["kudos_recv"] == 1 and _row(w, "skiff-restorers", P["cat"])["kudos_given"] == 1 \
        and w.state.tables["content"][p1]["comments"] == 2 and cat("community.kudos", community="skiff-restorers",
                                                                     member="cat").ok is False
    # participation history: counters equal what was recorded
    f["participation_history"] = rb["posts"] == 1 and rb["comments"] == 1 and \
        _row(w, "skiff-restorers", P["cat"])["comments"] == 1 and rb["last_active"] == w.now
    # shared activities: an event, RSVPs, attendance only while it runs
    ev = ben.ok("community.event", community="skiff-restorers", title="Launch day", in_s=2 * 3600, duration_s=3600)
    cat.ok("community.rsvp", event=ev["event"])
    early = cat("community.attend", event=ev["event"]).ok is False
    w.advance_time(w.now + 2 * 3600 + 60)
    cat.ok("community.attend", event=ev["event"])
    ben.ok("community.attend", event=ev["event"])
    evs = ann.ok("community.events", community="skiff-restorers", upcoming=False)
    f["shared_activities"] = early and evs[0]["going"] == 1 and evs[0]["attended"] == 2 and \
        _row(w, "skiff-restorers", P["cat"])["attended"] == 1
    # trust: a newcomer cannot propose (trust level 0 < 1); after contributing and being endorsed, it can
    eve.ok("community.join", community="skiff-restorers")
    blocked = eve("community.propose", community="skiff-restorers", action="describe",
                  args={"description": "x"}).ok is False
    for k in range(3):
        eve.ok("community.post", community="skiff-restorers", ref=text(eve, f"Plank {k} copper riveting"),
               title=f"Riveting {k}")
    ben.ok("community.kudos", community="skiff-restorers", member="eve")
    view = eve.ok("community.view", community="skiff-restorers")["you"]
    f["trust"] = blocked and view["trust_level"] >= 1 and view["reputation"] >= 5
    # governance: a member vote changes the rules (quorum and threshold), and a proposal without votes expires
    pr = eve.ok("community.propose", community="skiff-restorers", action="set_rules",
                args={"rules": {"inactive_days": 10}}, reason="people travel")
    ben.ok("community.vote", proposal=pr["proposal"], approve=True)
    cat.ok("community.vote", proposal=pr["proposal"], approve=True)
    decided = w.state.tables["community_proposals"][pr["proposal"]]["status"]
    pr2 = ben.ok("community.propose", community="skiff-restorers", action="rename", args={"name": "Skiffs!"})
    w.advance_time(w.now + 4 * DAY)
    c_ = w.state.tables["communities"]["skiff-restorers"]
    f["governance"] = decided == "approved" and c_["rules"]["inactive_days"] == 10 and c_["rules_version"] == 2 and \
        w.state.tables["community_proposals"][pr2["proposal"]]["status"] in ("expired", "rejected") and \
        c_["name"] == "Skiff restorers"
    # moderation: a report, a moderator removes the post, warns and bans; public log; banned cannot post
    pr3 = ann.ok("community.propose", community="skiff-restorers", action="add_moderator", args={"member": "cat"})
    for a in (ben, eve):
        a("community.vote", proposal=pr3["proposal"], approve=True)
    spam = dan.ok("community.post", community="skiff-restorers", ref=text(dan, "BUY CHEAP BOATS"), title="Cheap")["id"]
    rep = ben.ok("community.report", community="skiff-restorers", target=spam, reason="advertising")
    cat.ok("community.moderate", community="skiff-restorers", action="remove", target=spam, report=rep["report"])
    cat.ok("community.moderate", community="skiff-restorers", action="ban", target="dan", reason="repeat spam")
    log = ben.ok("community.modlog", community="skiff-restorers")
    f["moderation"] = w.state.tables["content"][spam]["removed"] and \
        w.state.tables["content"][spam]["removal_role"] == "moderator" and \
        {e["action"] for e in log} >= {"remove", "ban"} and _row(w, "skiff-restorers", P["dan"])["status"] == 3 and \
        dan("community.post", community="skiff-restorers", ref=text(dan, "again"), title="again").ok is False and \
        _row(w, "skiff-restorers", P["dan"])["removed"] == 1
    # inactive members: fay joins and never takes part; after inactive_days she is inactive; prune removes her
    fay.ok("community.join", community="skiff-restorers")
    for d in range(12):
        w.advance_time(w.now + DAY)
        for a in (ann, ben, cat, eve):
            a.ok("community.visit", community="skiff-restorers")
    h = ann.ok("community.health", community="skiff-restorers")
    inactive_now = h["inactive_members"] >= 1
    w.advance_time(w.now + 10 * DAY)
    f["inactive_members"] = inactive_now and _row(w, "skiff-restorers", P["fay"])["status"] == 2
    # organizational change: ownership transfer, a spin-off by a trusted member, a merge that moves members
    ann.ok("community.transfer", community="skiff-restorers", member="ben")
    roles = (_row(w, "skiff-restorers", P["ben"])["role"], _row(w, "skiff-restorers", P["ann"])["role"])
    for k in range(20):
        eve.ok("community.post", community="skiff-restorers", ref=text(eve, f"Oar making {k}"), title=f"Oars {k}")
    spin = eve.ok("community.found", id="oar-makers", name="Oar makers", subject="making oars", parent="skiff-restorers")
    ann.ok("community.join", community="oar-makers")
    ann("community.propose", community="oar-makers", action="merge", args={"into": "varnish-club"})
    refused_merge = w.state.tables["communities"]["oar-makers"]["status"] == "active"   # ann does not govern it
    eve.ok("community.propose", community="oar-makers", action="merge", args={"into": "secret-yard"})
    merged = w.state.tables["communities"]["oar-makers"]
    f["organizational_change"] = roles == (2, 1) and spin["community"] == "oar-makers" and refused_merge and \
        merged["parent"] == "skiff-restorers" and merged["status"] == "merged" and \
        merged["merged_into"] == "secret-yard" and _row(w, "secret-yard", P["ann"])["status"] == 1 and \
        len([k for k, x in w.state.tables["community_history"].items() if x["community"] == "skiff-restorers"]) >= 2
    # persistent membership: memberships survive a snapshot round trip and a replay of the whole log
    w2 = World.from_blob(w.snapshot_blob())
    w3 = World.replay(w.spec, [e.to_dict() for e in w.log])
    f["persistent_membership"] = w2.digest() == w.digest() == w3.digest() and \
        _row(w2, "skiff-restorers", P["cat"]) == _row(w, "skiff-restorers", P["cat"]) and \
        _row(w2, "skiff-restorers", P["cat"])["first_joined"] == 0
    assert w.check_invariants() == []

    # individual interests: reference participants join communities whose subject matches their own interests
    from crucible.social.communities import members
    from crucible.social.reference import TOPICS, make_participants, run_days, tokens
    wr, _ = social_world(seed=48, people=())
    ps = make_participants(wr, 40, seed=48)
    run_days(wr, ps, 14)
    by_node = {}
    from crucible.social import nodes
    for p in ps:
        by_node[nodes.node_of(wr, p.principal)] = p
    inside, outside = [], []
    for cid, c in wr.state.tables["communities"].items():
        sub = tokens(c["subject"] + " " + " ".join(c["tags"]))
        mem = {int(x) for x in members(wr, c["n"])}
        for node, p in by_node.items():
            if p.principal == c["founder"]:
                continue
            s = len(sub & (set(p.words) | set(p.interests)))
            (inside if node in mem else outside).append(s)
    f["individual_interests"] = bool(inside) and float(np.mean(inside)) > 2 * max(1e-9, float(np.mean(outside)))
    metrics = {"community_features": sum(1 for v in f.values() if v), "community_features_tested": len(f),
               "interest_overlap_members": round(float(np.mean(inside)), 3) if inside else 0.0,
               "interest_overlap_non_members": round(float(np.mean(outside)), 3) if outside else 0.0}
    metrics.update({f"feature_{k}": bool(v) for k, v in f.items()})
    evidence.record("SP-4.7", metrics, command=CMD, notes=(
        "13 community features exercised through the community tools and verified in recorded state; individual "
        "interests measured on 40 reference participants over 14 days (members' interests overlap the subject of "
        "their community more than non-members')."))
    assert_met('SP-4.7', metrics)
    assert all(f.values()), {k: v for k, v in f.items() if not v}
