"""Evaluator-side measurements of semantic participants (Gate H, SP-3.2d, SP-3.3c). Never callable in-world.

Everything here reads the world's state and event log after the fact; nothing is taken from what a participant says
about itself:

* `actions(world)` - every call a participant made (from its persistent history), joined with the events the call
  emitted in the log: whether it changed shared state and whether anyone else can see the change (public events, or
  events visible to another principal). *Consequential* = succeeded, not read-only, and emitted at least one event
  that another principal (or the public) can see. What the call wrote is read from the log too (the text fields of
  its events' payloads and the text artifacts it stored): a call whose main written content is junk - a tool or
  service name, a line of the participants' prompt scaffolding, the participant's own goal sentence restated, a
  text that only announces what its writer will do ("I will publish an article on ..."), or under two letters - is
  *junk*, and `summary` counts consequential calls that are not junk once per participant,
  tool and content (`signature`: a calendar entry titled 'cal' made again later, or the same text mailed to other
  people, counts once; audit D skeptic "junk repetitions").
* `provenance(world)` - SP-3.2d: every event in the log whose actor is a semantic participant must lie inside a
  call the participant's brain chose (the recorded step shortlist and pick); any other event acted as a participant
  is a scripted path.
* `productive(world)` - content a participant published (catalogue items it created) that independent individuals of
  the population (not class 4) viewed, from the exact audience sets (`crucible.social.audience.viewers_of`);
  `summary` counts distinct published contents (the same artifact published again is one result).
* `collaborative(world)` - (a) answered requests: a message from one participant to another and the recipient's
  reply to it (`in_reply_to`), with the reply on the request's subject (MiniLM cosine of request and reply above
  ``RELEVANCE``); (b) documents edited by at least two participants; (c) payments between participants (a bank
  transfer from one to another) that follow a message between them.
"""
from __future__ import annotations

import json
import re

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


TEXT_KEYS = ("title", "subject", "body", "text", "message", "description", "name", "content", "bio", "summary", "note",
             "comment", "answer", "q", "query", "label", "topic", "headline", "caption", "about", "question")


def _texts(x, out: list, depth: int = 0) -> None:
    if depth > 5:
        return
    if isinstance(x, dict):
        for k, v in sorted(x.items()):
            if isinstance(v, str) and k in TEXT_KEYS and v.strip():
                out.append(v[:6000])
            elif isinstance(v, (dict, list)):
                _texts(v, out, depth + 1)
    elif isinstance(x, list):
        for v in x[:20]:
            _texts(v, out, depth + 1)


def written(world, evs, principal: str) -> tuple[list[str], list[str]]:
    """(text fields of the call's event payloads, texts of the text artifacts the call stored in the participant's
    name), from the event log and the artifact store."""
    from ...substrate import artifacts
    vals, arts = [], []
    for e in evs:
        _texts(e.payload, vals)
        p = e.payload if isinstance(e.payload, dict) else {}
        if e.type == "artifact.stored" and p.get("owner") == principal and str(p.get("mime", "")).startswith("text/"):
            try:
                arts.append(artifacts.get(world, p["ref"]).decode("utf-8", "replace")[:6000])
            except Exception:  # noqa: BLE001 - bytes not in this store: judged by the other fields
                pass
    return list(dict.fromkeys(vals)), arts


def _norm(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", t.lower()))


def junk_text(text: str, names: set, goal: str, publication: bool = False) -> bool:
    """Evaluator-side: a written text that says nothing of its own (see the module docstring). A publication (the
    text of an artifact it published, or a publication's title) is also junk when it only announces what its writer
    will do or prefaces itself; a message may well say what its writer will do."""
    from .mind import ANNOUNCE, SCAFFOLD, announcement, restates
    flat = _norm(text)
    if len(re.findall(r"[A-Za-z]", text)) < 2 or SCAFFOLD.search(text):
        return True
    if publication and (announcement(text) if len(flat.split()) >= 12 else bool(ANNOUNCE.match(text))):
        return True
    if flat in names or text.strip().lower() in names:
        return True
    return bool(goal) and restates(text, goal)


def _job_runs(world, seeds: set) -> dict:
    """Events of the runs of the jobs in `seeds` and of the jobs those runs schedule (a run lies between its ``proc``
    event and the next ``proc`` or ``cmd`` event, as `provenance` delimits them), by job id; one pass over the log
    that keeps only the tracked jobs' events."""
    runs: dict = {}
    tracked = set(seeds)
    cur = None
    if not tracked:
        return runs
    for ev in world.events_range(0, world.seq):
        if ev.type == "proc":
            cur = ev.payload.get("id")
        elif ev.type == "cmd":
            cur = None
        if cur is not None and cur in tracked:
            runs.setdefault(cur, []).append(ev)
            if ev.type == "sched.add" and isinstance(ev.payload, dict):
                tracked.add(ev.payload.get("id"))
    return runs


def _consequences(evs, runs: dict, depth: int = 3) -> list:
    """Events of the jobs a call scheduled (a delivery it sent, a renewal it ordered), and of the jobs those
    scheduled, to `depth` levels."""
    out = []
    jobs = [e.payload.get("id") for e in evs if e.type == "sched.add" and isinstance(e.payload, dict)]
    for _ in range(depth):
        nxt = []
        for j in jobs:
            r = runs.get(j) or []
            out += r
            nxt += [e.payload.get("id") for e in r if e.type == "sched.add" and isinstance(e.payload, dict)
                    and e.payload.get("id") != j]
        jobs = nxt
        if not jobs:
            break
    return out


def actions(world) -> list[dict]:
    """Every call of every participant with its events and consequences (see the module docstring). The events of
    jobs the call scheduled count as its consequences (a mail is delivered to its recipient by a job one minute
    after mail.send)."""
    out = []
    tools = {}
    calls = []
    for r in rows(world):
        for h in r["history"]:
            seq = h.get("seq")
            calls.append((r, h, world.events_range(int(seq[0]) - 1, int(seq[1])) if seq else []))
    runs = _job_runs(world, {e.payload.get("id") for _, h, evs in calls if h["ok"] for e in evs
                             if e.type == "sched.add" and isinstance(e.payload, dict)})
    for r, h, evs in calls:
        p = r["principal"]
        if p not in tools:
            ns = {t["name"].lower() for t in world.available_tools(p)}
            tools[p] = ns | {n.split(".", 1)[0] for n in ns}
        goals = {g["id"]: g["text"] for g in r["goals"]}
        rec = dict(h, principal=p, i=int(r["i"]))
        later = _consequences(evs, runs) if h["ok"] else []
        rec["events"] = len(evs)
        rec["scheduled_events"] = len(later)
        rec["shared"] = any(_visible_to_others(e, p) for e in evs + later)
        rec["consequential"] = bool(h["ok"] and not h.get("read_only") and rec["shared"])
        vals, arts = written(world, evs, p) if rec["consequential"] else ([], [])
        main = max(arts, key=len) if arts else (max(vals, key=len) if vals else None)
        rec["written"] = bool(main)
        rec["junk"] = bool(main is not None and junk_text(main, tools[p], goals.get(h.get("goal"), ""),
                                                          publication=bool(arts)))
        rec["signature"] = json.dumps([p, h["tool"], _norm(main)[:400] if main else h.get("args")])
        out.append(rec)
    return out


def provenance(world) -> dict:
    """Events acted as a semantic participant outside every model-chosen call (SP-3.2d scripted paths).

    An event acted as a participant counts as inside when it lies in the seq range of one of its brain-chosen calls,
    or when it is emitted by a scheduled job that one of its chosen calls scheduled (a renewal, a workflow run, a
    delivery it ordered) - directly or through the job's own periodic rescheduling: the job's ``sched.add`` lies in
    a chosen call of that participant, or in a run of a job that does. Job runs are delimited by their ``proc``
    event and the next ``proc`` or ``cmd`` event."""
    ps = principals(world)
    covered: dict[str, list[tuple[int, int]]] = {p: [] for p in ps}
    for r in rows(world):
        for h in r["history"]:
            if h.get("seq"):
                covered[r["principal"]].append((int(h["seq"][0]), int(h["seq"][1])))

    def chosen_by(seq: int) -> str | None:
        for p, rs in covered.items():
            if any(a <= seq <= b for a, b in rs):
                return p
        return None
    origin: dict[str, str] = {}            # job id -> participant whose chosen call scheduled it
    cur = None                             # the job whose run the current event belongs to
    outside, inside, consequence, kinds = 0, 0, 0, {}
    for ev in world.events_range(0, world.seq):
        if ev.type == "proc":
            cur = ev.payload.get("id")
        elif ev.type == "cmd":
            cur = None
        if ev.type == "sched.add":
            who = chosen_by(ev.seq) or (origin.get(cur) if cur is not None else None)
            if who is not None:
                origin[ev.payload.get("id")] = who
        if ev.actor not in ps:
            continue
        if any(a <= ev.seq <= b for a, b in covered[ev.actor]):
            inside += 1
        elif cur is not None and origin.get(cur) == ev.actor:
            consequence += 1
        else:
            outside += 1
            kinds[ev.type] = kinds.get(ev.type, 0) + 1
    return {"events_as_participants": inside + consequence + outside, "inside_model_chosen_calls": inside,
            "scheduled_consequences_of_chosen_calls": consequence, "outside_model_chosen_calls": outside,
            "outside_by_type": kinds}


def productive(world, min_viewers: int = 3) -> list[dict]:
    if "aud" not in world.modules:
        return []
    from ...social import audience as A
    from ...substrate import artifacts
    ps = principals(world)
    cls = columns.col(world, "pop", "class")
    goals = {r["principal"]: [(int(g["t"]), g["text"]) for g in r["goals"]] for r in rows(world)}
    out = []
    for key, rec in world.state.scan("aud_item_info"):
        if rec.get("creator") not in ps:
            continue
        v = A.viewers_of(world, int(key))
        indep = v[(v < cls.shape[0]) & (cls[np.minimum(v, cls.shape[0] - 1)] != CLASS)] if v.size else v
        ex = A.audience(world, int(key))
        text, title = str(rec.get("title") or ""), str(rec.get("title") or "")
        try:
            info = artifacts.info(world, rec["ref"]) if rec.get("ref") else None
            if info and str(info.get("mime", "")).startswith("text/"):
                text = artifacts.get(world, rec["ref"]).decode("utf-8", "replace")[:6000]
        except Exception:  # noqa: BLE001 - no text artifact: judged by its title
            pass
        ns = {t["name"].lower() for t in world.available_tools(rec["creator"])}
        names = ns | {n.split(".", 1)[0] for n in ns}
        # restating a goal: only goals the creator had set by the time it published (a later goal copied from its
        # own article - small models form the next goal from what they remember doing - says nothing about it)
        before = [g for t, g in goals[rec["creator"]] if t <= int(rec.get("published_at", 0))]
        junk = any(junk_text(text, names, g, publication=True) for g in before or [""]) or \
            (bool(title) and title != text and junk_text(title, names, "", publication=True))
        out.append({"item": int(key), "creator": rec["creator"], "platform": rec.get("platform"), "ref": rec.get("ref"),
                    "unique_viewers": int(ex["unique_viewers"]), "independent_viewers": int(indep.size),
                    "views": int(ex["views"]), "shares": int(ex["shares"]), "junk_content": junk,
                    # independent viewers of content that says something of its own (not a tool name, not the
                    # creator's goal sentence restated)
                    "verified": bool(indep.size >= min_viewers and not junk)})
    return out


def distinct_results(prod: list[dict]) -> int:
    """Verified productive results counted once per creator and published content (artifact ref)."""
    return len({(p["creator"], p.get("ref") or p["item"]) for p in prod if p["verified"]})


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
                        "by": m["from"], "relevance": round(rel, 4),  # not-a-content-score: evaluator-side MiniLM cosine of request and reply, never in-world
                        "verified": bool(rel >= RELEVANCE)})
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
        # consequential calls whose written content is not junk, once per (participant, tool, content)
        "consequential": len({a["signature"] for a in acts if a["consequential"] and not a["junk"]}),
        "consequential_raw": sum(a["consequential"] for a in acts),
        "consequential_junk": sum(a["consequential"] and a["junk"] for a in acts),
        "consequential_repeats": sum(a["consequential"] and not a["junk"] for a in acts) -
        len({a["signature"] for a in acts if a["consequential"] and not a["junk"]}),
        "consequential_by_tool": _count(a["tool"] for a in acts if a["consequential"] and not a["junk"]),
        "tools_used": len({a["tool"] for a in acts if a["ok"]}),
        "goals": len(goals), "distinct_goals": len({g["text"] for g in goals}),
        "goals_done": sum(g["status"] == "done" for g in goals),
        "goals_abandoned": sum(g["status"] == "abandoned" for g in goals),
        "participants_with_own_goals": sum(1 for r in rs if r["goals"]),
        "provenance": prov,
        # distinct published contents with >= 3 independent viewers (the same artifact published again, e.g. as a
        # reply to other items, is one result)
        "productive_verified": distinct_results(prod),
        "productive_items_verified": sum(p["verified"] for p in prod), "productive": prod,
        "collaborative_verified": sum(c["verified"] for c in coll), "collaborative": coll,
        "bound": bound(world),
    }


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
