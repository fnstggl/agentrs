"""Evaluator-side measurements of semantic participants (Gate H, SP-3.2d, SP-3.3c). Never callable in-world.

Everything here reads the world's state and event log after the fact; nothing is taken from what a participant says
about itself:

* `actions(world)` - every call a participant made (from its persistent history), joined with the events the call
  emitted in the log: whether it changed shared state and whether anyone else can see the change (public events, or
  events visible to another principal). *Consequential* = succeeded, not read-only, and emitted at least one event
  that another principal (or the public) can see.
* `provenance(world)` - SP-3.2d: every event in the log whose actor is a semantic participant must lie inside a
  call the participant's brain chose (the recorded step shortlist and pick); any other event acted as a participant
  is a scripted path.
* `productive(world)` - content a participant published (catalogue items it created) that independent individuals of
  the population (not class 4) viewed, from the exact audience sets (`crucible.social.audience.viewers_of`).
* `collaborative(world)` - (a) answered requests: a message from one participant to another and the recipient's
  reply to it (`in_reply_to`), with the reply on the request's subject (MiniLM cosine of request and reply above
  ``RELEVANCE``); (b) documents edited by at least two participants; (c) payments between participants (a bank
  transfer from one to another) that follow a message between them.
"""
from __future__ import annotations

import numpy as np

from ...substrate import columns, ids
from .module import CLASS, TABLE, rows

RELEVANCE = 0.3


def principals(world) -> dict[str, int]:
    return {r["principal"]: int(r["i"]) for r in rows(world)}


def _visible_to_others(ev, actor: str) -> bool:
    if ev.vis == "public":
        return True
    if isinstance(ev.vis, (tuple, list)):
        return any(p != actor for p in ev.vis)
    return False


def actions(world) -> list[dict]:
    out = []
    for r in rows(world):
        for h in r["history"]:
            rec = dict(h, principal=r["principal"], i=int(r["i"]))
            seq = h.get("seq")
            evs = world.events_range(int(seq[0]) - 1, int(seq[1])) if seq else []
            rec["events"] = len(evs)
            rec["shared"] = any(_visible_to_others(e, r["principal"]) for e in evs)
            rec["consequential"] = bool(h["ok"] and not h.get("read_only") and rec["shared"])
            out.append(rec)
    return out


def provenance(world) -> dict:
    """Events acted as a semantic participant outside every model-chosen call (SP-3.2d scripted paths)."""
    ps = principals(world)
    covered: dict[str, list[tuple[int, int]]] = {p: [] for p in ps}
    for r in rows(world):
        for h in r["history"]:
            if h.get("seq"):
                covered[r["principal"]].append((int(h["seq"][0]), int(h["seq"][1])))
    outside, inside, kinds = 0, 0, {}
    for ev in world.events_range(0, world.seq):
        if ev.actor not in ps:
            continue
        if any(a <= ev.seq <= b for a, b in covered[ev.actor]):
            inside += 1
        else:
            outside += 1
            kinds[ev.type] = kinds.get(ev.type, 0) + 1
    return {"events_as_participants": inside + outside, "inside_model_chosen_calls": inside,
            "outside_model_chosen_calls": outside, "outside_by_type": kinds}


def productive(world, min_viewers: int = 3) -> list[dict]:
    if "aud" not in world.modules:
        return []
    from ...social import audience as A
    ps = principals(world)
    cls = columns.col(world, "pop", "class")
    out = []
    for key, rec in world.state.scan("aud_item_info"):
        if rec.get("creator") not in ps:
            continue
        v = A.viewers_of(world, int(key))
        indep = v[(v < cls.shape[0]) & (cls[np.minimum(v, cls.shape[0] - 1)] != CLASS)] if v.size else v
        ex = A.audience(world, int(key))
        out.append({"item": int(key), "creator": rec["creator"], "platform": rec.get("platform"),
                    "unique_viewers": int(ex["unique_viewers"]), "independent_viewers": int(indep.size),
                    "views": int(ex["views"]), "shares": int(ex["shares"]),
                    "verified": bool(indep.size >= min_viewers)})
    return out


def _cos(a: str, b: str) -> float:
    from . import catalog
    v = catalog.embed([a, b])
    return float(v[0] @ v[1])


def collaborative(world) -> list[dict]:
    ps = principals(world)
    out = []
    st = world.state
    if "mail_msgs" in st.tables:
        msgs = dict(st.scan("mail_msgs"))
        for mid, m in sorted(msgs.items()):
            src = msgs.get(m.get("in_reply_to") or "")
            if src is None or m["from"] not in ps or src["from"] not in ps or m["from"] == src["from"]:
                continue
            if m["from"] not in src["to"]:
                continue
            rel = _cos(f"{src['subject']}\n{src['body']}", f"{m['subject']}\n{m['body']}")
            out.append({"kind": "answered_request", "request": m["in_reply_to"], "reply": mid, "from": src["from"],
                        "by": m["from"], "relevance": round(rel, 4), "verified": bool(rel >= RELEVANCE)})
    if "doc_versions" in st.tables:
        editors: dict[str, set] = {}
        for _, v in st.scan("doc_versions"):
            who = v.get("author") or v.get("by")
            if who in ps:
                editors.setdefault(v["doc"], set()).add(who)
        for doc, es in sorted(editors.items()):
            if len(es) >= 2:
                out.append({"kind": "co_edited_document", "doc": doc, "editors": sorted(es), "verified": True})
    talked = set()
    if "mail_msgs" in st.tables:
        for _, m in st.scan("mail_msgs"):
            if m["from"] in ps:
                for t in m["to"]:
                    if t in ps:
                        talked.add(frozenset((m["from"], t)))
    if "txns" in st.tables:
        acct = {r["account"]: r["principal"] for r in rows(world) if r.get("account")}
        for tid, tx in st.scan("txns"):
            post = tx.get("postings") or []
            src = [acct.get(a) for a, d in post if d < 0 and a in acct]
            dst = [acct.get(a) for a, d in post if d > 0 and a in acct]
            for a in src:
                for b in dst:
                    if a != b and frozenset((a, b)) in talked:
                        out.append({"kind": "payment_after_conversation", "tx": tid, "from": a, "to": b,
                                    "amount": int(sum(d for x, d in post if d > 0 and acct.get(x) == b)),
                                    "verified": True})
    return out


def bound(world) -> dict:
    """SP-3.3c: participants whose state row belongs to a class-4 individual of the population with its principal."""
    cls = columns.col(world, "pop", "class")
    members = np.flatnonzero(cls == CLASS)
    rs = {int(r["i"]): r for r in rows(world)}
    ok = [int(i) for i in members if int(i) in rs and rs[int(i)]["principal"] == ids.pid(int(i))
          and world.state.has("principals", ids.pid(int(i)))]
    return {"class4_rows": int(members.size), "bound": len(ok), "state_rows": len(rs)}


def summary(world) -> dict:
    acts = actions(world)
    rs = rows(world)
    prov = provenance(world)
    prod = productive(world)
    coll = collaborative(world)
    goals = [g for r in rs for g in r["goals"]]
    return {
        "participants": len(rs),
        "acting": len({a["principal"] for a in acts if a["ok"]}),
        "calls": len(acts), "ok_calls": sum(a["ok"] for a in acts),
        "rejected_calls": sum(not a["ok"] for a in acts),
        "consequential": sum(a["consequential"] for a in acts),
        "consequential_by_tool": _count(a["tool"] for a in acts if a["consequential"]),
        "tools_used": len({a["tool"] for a in acts if a["ok"]}),
        "goals": len(goals), "distinct_goals": len({g["text"] for g in goals}),
        "goals_done": sum(g["status"] == "done" for g in goals),
        "goals_abandoned": sum(g["status"] == "abandoned" for g in goals),
        "participants_with_own_goals": sum(1 for r in rs if r["goals"]),
        "provenance": prov,
        "productive_verified": sum(p["verified"] for p in prod), "productive": prod,
        "collaborative_verified": sum(c["verified"] for c in coll), "collaborative": coll,
        "bound": bound(world),
    }


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
