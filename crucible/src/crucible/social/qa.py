"""Developer communities (`dev`) and knowledge sharing (`kb`): question-and-answer sites over the catalogue, plus a
package registry (dev) and educational products (kb).

Questions and answers are catalogue items of kind `thread` (an answer has `reply_to` = its question); articles,
apps and packages are published as on any site. On top of the catalogue each site keeps:

* votes (+1/-1, one per person per post, never on one's own posts) and the post scores they sum to;
* accepted answers (chosen by the person who asked);
* reputation of each participant on the site, derived from votes and acceptances (+10 per upvote on an answer,
  +5 per upvote on a question, -2 per downvote, +15 per accepted answer) - computed from the recorded votes;
* dev only: a package registry (name -> versions, each a catalogue item of kind `package`; a name belongs to its
  first publisher), dependency information read from the packages themselves, and test runs in the sandbox.

Every score, count and reputation is recomputable from the recorded votes and posts.
"""
from __future__ import annotations

import json

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import Ctx, applier, endpoint, register_module
from ..substrate import artifacts, bus, ids
from . import content as C
from .sites import Site, _site_endpoints

_ID = S.s_str("content id c<n>", pattern=C.CID_RE.pattern, maxLength=14)
_S, _I = {"type": "string"}, {"type": "integer"}
for _p in ("dev", "kb"):
    bus.event_type(f"{_p}.qa", S.obj({"id": _S, "type": {"type": "string", "enum": ["question", "answer"]},
                                      "question": {"type": ["string", "null"]}, "by": _S},
                                     ["id", "type", "by"]), owner=_p, doc="a question or an answer was posted")
    bus.event_type(f"{_p}.vote", S.obj({"id": _S, "voter": _S, "value": {"type": "integer", "enum": [-1, 0, 1]},
                                        "prev": {"type": "integer"}}, ["id", "voter", "value", "prev"]), owner=_p,
                   doc="a vote on a question or an answer")
    bus.event_type(f"{_p}.accept", S.obj({"question": _S, "answer": {"type": ["string", "null"]}, "by": _S},
                                         ["question", "by"]), owner=_p, doc="the asker accepted an answer")
bus.event_type("dev.package", S.obj({"name": _S, "version": _S, "item": _S, "owner": _S,
                                     "deps": {"type": "array", "items": _S}}, ["name", "version", "item", "owner"]),
               owner="dev", doc="a package version was registered")


class QASite(Site):
    """A Q&A platform: questions, answers, votes, accepted answers, reputation."""

    def install(self, world):
        super().install(world)
        st = world.state
        st.declare_table(f"{self.platform}_posts", index=("question", "author", "type"))
        st.declare_table(f"{self.platform}_votes", index=("post", "voter"))
        bus.enforce(world, owner=self.platform)

    @property
    def T(self) -> str:
        return f"{self.platform}_posts"

    @property
    def V(self) -> str:
        return f"{self.platform}_votes"

    def _text_ref(self, ctx: Ctx, title: str, body: str | None, ref: str | None) -> str:
        if (body is None) == (ref is None):
            raise InvalidAction("give either body (text) or ref (an artifact you own)")
        if ref is not None:
            return ref
        data = (f"{title}\n\n{body}" if title else body).encode("utf-8")
        return artifacts.put(ctx, data, "text/markdown", owner=ctx.actor, meta={"title": title[:200]}, kind="text")

    def _post(self, ctx: Ctx, pid: str) -> dict:
        p = ctx.state.get(self.T, pid)
        rec = ctx.state.get("content", pid)
        if p is None or rec is None or rec["removed"] or not C.readable(ctx.world, ctx.actor, rec):
            raise NotFound(f"no question or answer {pid} on {self.platform}")
        return dict(p)

    def reputation(self, world, principal: str) -> dict:
        st = world.state
        rep = up = down = accepted = 0
        for pid in st.find_ids(self.T, "author", principal):
            p = st.tables[self.T][pid]
            w = 10 if p["type"] == "answer" else 5
            rep += w * p["up"] - 2 * p["down"]
            up += p["up"]
            down += p["down"]
            if p["type"] == "answer" and p["accepted"]:
                rep += 15
                accepted += 1
        return {"principal": principal, "reputation": rep, "upvotes": up, "downvotes": down, "accepted": accepted}

    def _view_post(self, world, pid: str, actor: str) -> dict:
        p = world.state.tables[self.T][pid]
        out = C.module(world).view(world, world.state.tables["content"][pid], actor)
        out.update(type=p["type"], score=p["score"], up=p["up"], down=p["down"], answers=p["answers"],
                   accepted=p["accepted"], question=p["question"])
        return out

    @endpoint("ask", S.obj({"title": S.s_str(minLength=5, maxLength=C.MAX_TITLE),
                            "body": S.s_str("the question (markdown)", minLength=1, maxLength=50_000),
                            "ref": S.s_str("or an artifact you own", maxLength=71), "tags": C._TAGS},
                           ["title"]), cost_s=120)
    def ep_ask(self, ctx, title, body=None, ref=None, tags=()):
        """Ask a question (a public thread)."""
        r = self._text_ref(ctx, title, body, ref)
        cid = C.publish(ctx, ctx.actor, r, "thread", self.platform, title, tags=tags)
        bus.emit(ctx, f"{self.platform}.qa", {"id": cid, "type": "question", "question": None, "by": ctx.actor},
                 vis="public")
        return {"question": cid}

    @endpoint("answer", S.obj({"question": _ID, "body": S.s_str(minLength=1, maxLength=50_000),
                               "ref": S.s_str(maxLength=71)}, ["question"]), cost_s=300)
    def ep_answer(self, ctx, question, body=None, ref=None):
        """Answer a question."""
        q = self._post(ctx, question)
        if q["type"] != "question":
            raise InvalidAction("answer questions, not answers")
        qrec = ctx.state.tables["content"][question]
        r = self._text_ref(ctx, "", body, ref)
        cid = C.publish(ctx, ctx.actor, r, "thread", self.platform, f"Re: {qrec['title']}"[:C.MAX_TITLE],
                        tags=qrec["tags"], reply_to=question)
        bus.emit(ctx, f"{self.platform}.qa", {"id": cid, "type": "answer", "question": question, "by": ctx.actor},
                 vis="public")
        return {"answer": cid, "question": question}

    @endpoint("vote", S.obj({"id": _ID, "value": {"type": "integer", "enum": [-1, 0, 1],
                                                  "description": "+1, -1, or 0 to withdraw"}}, ["id", "value"]),
              cost_s=2)
    def ep_vote(self, ctx, id, value):
        """Vote on a question or an answer (not your own)."""
        p = self._post(ctx, id)
        if p["author"] in ctx.world.principals_of(ctx.actor):
            raise Forbidden("you cannot vote on your own posts")
        old = ctx.state.get(self.V, f"{id}|{ctx.actor}")
        prev = old["value"] if old else 0
        if prev == value:
            raise Conflict("you already cast this vote")
        bus.emit(ctx, f"{self.platform}.vote", {"id": id, "voter": ctx.actor, "value": int(value), "prev": prev},
                 vis=(ctx.actor,))
        return {"id": id, "score": ctx.state.tables[self.T][id]["score"]}

    @endpoint("accept", S.obj({"question": _ID, "answer": S.s_str("an answer to it, or empty to clear",
                                                                  maxLength=14)}, ["question"]), cost_s=2)
    def ep_accept(self, ctx, question, answer=None):
        """Accept the answer that solved your question."""
        q = self._post(ctx, question)
        if q["type"] != "question" or q["author"] != ctx.actor:
            raise Forbidden("only the asker accepts an answer")
        if answer:
            a = self._post(ctx, answer)
            if a["question"] != question:
                raise InvalidAction(f"{answer} does not answer {question}")
            if a["author"] == ctx.actor:
                raise Forbidden("you cannot accept your own answer")
        bus.emit(ctx, f"{self.platform}.accept", {"question": question, "answer": answer or None, "by": ctx.actor},
                 vis="public")
        return {"question": question, "accepted": answer or None}

    @endpoint("questions", S.obj({"tag": S.s_str(pattern=C.TAG_RE, maxLength=32),
                                  "unanswered": S.s_bool(default=False),
                                  "sort": S.s_enum(["new", "score", "active"]),
                                  "limit": S.s_int(minimum=1, maximum=50, default=20)}), read_only=True, cost_s=2)
    def ep_questions(self, ctx, tag=None, unanswered=False, sort="new", limit=20):
        """Questions (optionally one tag, or only those without an accepted answer)."""
        st = ctx.state
        rows = []
        for pid in st.find_ids(self.T, "type", "question"):
            p = st.tables[self.T][pid]
            rec = st.tables["content"][pid]
            if rec["removed"] or (tag and tag not in rec["tags"]) or (unanswered and p["accepted"]):
                continue
            rows.append((pid, p, rec))
        key = {"new": lambda x: (-x[2]["published_at"], -x[2]["n"]), "score": lambda x: (-x[1]["score"], -x[2]["n"]),
               "active": lambda x: (-x[1]["last_activity"], -x[2]["n"])}[sort]
        rows.sort(key=key)
        return [self._view_post(ctx.world, pid, ctx.actor) for pid, _, _ in rows[:limit]]

    @endpoint("question", S.obj({"id": _ID}, ["id"]), read_only=True, cost_s=5)
    def ep_question(self, ctx, id):
        """A question with its answers (accepted first, then by score) and the question's text."""
        q = self._post(ctx, id)
        if q["type"] != "question":
            id = q["question"]
        st = ctx.state
        out = self._view_post(ctx.world, id, ctx.actor)
        data = artifacts.get(ctx.world, st.tables["content"][id]["ref"])
        rec = st.tables["content"][id]
        from . import formats as F
        out["text"] = F.extract_text(data, rec["format"], rec["mime"], limit=20_000)
        ans = []
        for aid in st.find_ids(self.T, "question", id):
            arec = st.tables["content"][aid]
            if arec["removed"]:
                continue
            a = self._view_post(ctx.world, aid, ctx.actor)
            a["text"] = F.extract_text(artifacts.get(ctx.world, arec["ref"]), arec["format"], arec["mime"],
                                       limit=20_000)
            ans.append(a)
        acc = st.tables[self.T][id]["accepted"]
        ans.sort(key=lambda a: (a["id"] != acc, -a["score"], a["n"]))
        out["answers_list"] = ans
        return out

    @endpoint("reputation_of", S.obj({"principal": S.s_str(maxLength=60)}), read_only=True)
    def ep_reputation_of(self, ctx, principal=None):
        """Reputation on this site, from the votes and acceptances people gave."""
        from ..services.identity import ident
        p = ctx.actor if not principal else (principal if ids.index(principal) is not None else
                                             ident(ctx.world).resolve_handle(ctx.state, principal))
        return self.reputation(ctx.world, p)

    # ------------------------------------------------------------------ appliers (registered per subclass)
    def ap_qa(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert(self.T, p["id"], {"type": p["type"], "question": p.get("question"), "author": p["by"], "score": 0,
                                    "up": 0, "down": 0, "answers": 0, "accepted": None, "t": ev.t,
                                    "last_activity": ev.t})
        if p["type"] == "answer":
            q = st.tables[self.T][p["question"]]
            st.patch(self.T, p["question"], answers=q["answers"] + 1, last_activity=ev.t)

    def ap_vote(self, ev):
        p = ev.payload
        st = self.world.state
        key = f"{p['id']}|{p['voter']}"
        if p["value"] == 0:
            st.delete(self.V, key)
        else:
            st.put(self.V, key, {"post": p["id"], "voter": p["voter"], "value": p["value"], "t": ev.t})
        post = st.tables[self.T][p["id"]]
        up = post["up"] + (p["value"] == 1) - (p["prev"] == 1)
        down = post["down"] + (p["value"] == -1) - (p["prev"] == -1)
        st.patch(self.T, p["id"], up=up, down=down, score=up - down)

    def ap_accept(self, ev):
        p = ev.payload
        st = self.world.state
        q = st.tables[self.T][p["question"]]
        if q["accepted"]:
            st.patch(self.T, q["accepted"], accepted=False)
        st.patch(self.T, p["question"], accepted=p.get("answer"), last_activity=ev.t)
        if p.get("answer"):
            st.patch(self.T, p["answer"], accepted=True)

    def check_invariants(self, world):
        errs = []
        st = world.state
        tally: dict[str, list[int]] = {}
        for k, v in st.tables[self.V].items():
            if k != f"{v['post']}|{v['voter']}" or v["value"] not in (-1, 1):
                errs.append(f"bad vote record {k}")
            t = tally.setdefault(v["post"], [0, 0])
            t[0 if v["value"] == 1 else 1] += 1
            if st.tables[self.T].get(v["post"], {}).get("author") == v["voter"]:
                errs.append(f"self-vote {k}")
        answers: dict[str, int] = {}
        for pid, p in st.tables[self.T].items():
            if p["type"] == "answer":
                answers[p["question"]] = answers.get(p["question"], 0) + 1
        for pid, p in st.tables[self.T].items():
            up, down = tally.get(pid, [0, 0])
            if (p["up"], p["down"], p["score"]) != (up, down, up - down):
                errs.append(f"{self.platform} post {pid} score {p['score']} != votes {up}-{down}")
            if p["type"] == "question" and p["answers"] != answers.get(pid, 0):
                errs.append(f"{self.platform} question {pid} counts {p['answers']} answers")
        return errs


def _qa_appliers(cls):
    for what, fn in (("qa", QASite.ap_qa), ("vote", QASite.ap_vote), ("accept", QASite.ap_accept)):
        def bound(self, ev, _fn=fn):
            return _fn(self, ev)
        bound.__crucible_applier__ = f"{cls.platform}.{what}"
        setattr(cls, f"ap_{what}_{cls.platform}", bound)
    return cls


class DevSite(QASite):
    """Developer community: Q&A, technical articles, apps and a package registry."""

    def install(self, world):
        super().install(world)
        world.state.declare_table("dev_registry", index=("owner",))

    @endpoint("register", S.obj({"id": _ID}, ["id"]), cost_s=30)
    def ep_register(self, ctx, id):
        """Register a published package item (kind package on dev) under its name and version. A name belongs to
        whoever registered it first; later versions must be higher."""
        rec = C.module(ctx.world)._rec(ctx, id)
        if rec["platform"] != "dev" or rec["kind"] != "package" or rec["removed"]:
            raise InvalidAction("register a package published on dev")
        if rec["creator"] not in ctx.world.principals_of(ctx.actor):
            raise Forbidden("only the package's publisher registers it")
        doc = json.loads(artifacts.get(ctx.world, rec["ref"]).decode("utf-8"))
        name, ver = doc["name"], doc["version"]
        reg = ctx.state.get("dev_registry", name)
        if reg is not None:
            if reg["owner"] != rec["creator"]:
                raise Forbidden(f"package name {name} belongs to {reg['owner']}")
            if tuple(int(x) for x in ver.split(".")) <= tuple(int(x) for x in reg["latest"].split(".")):
                raise Conflict(f"{name} {ver} is not newer than {reg['latest']}")
        bus.emit(ctx, "dev.package", {"name": name, "version": ver, "item": id, "owner": rec["creator"],
                                      "deps": sorted(set(doc.get("deps", [])))}, vis="public")
        return {"package": name, "version": ver}

    @endpoint("package", S.obj({"name": S.s_str(maxLength=41)}, ["name"]), read_only=True)
    def ep_package(self, ctx, name):
        """A registered package: versions, dependencies, and the registered packages that depend on it."""
        reg = ctx.state.get("dev_registry", name)
        if reg is None:
            raise NotFound(f"no package {name}")
        dependents = sorted(n for n, r in ctx.state.scan("dev_registry") if name in r["deps"])
        return {"name": name, "owner": reg["owner"], "latest": reg["latest"], "versions": list(reg["versions"]),
                "deps": list(reg["deps"]), "dependents": dependents}

    @endpoint("packages", S.obj({"limit": S.s_int(minimum=1, maximum=200, default=50)}), read_only=True)
    def ep_packages(self, ctx, limit=50):
        """Registered packages, by number of registered dependents."""
        reg = dict(ctx.state.scan("dev_registry"))
        dep = {n: 0 for n in reg}
        for n, r in reg.items():
            for d in r["deps"]:
                if d in dep:
                    dep[d] += 1
        rows = [{"name": n, "owner": r["owner"], "latest": r["latest"], "dependents": dep[n]} for n, r in reg.items()]
        return sorted(rows, key=lambda x: (-x["dependents"], x["name"]))[:limit]

    @applier("dev.package")
    def ap_package(self, ev):
        p = ev.payload
        st = self.world.state
        old = st.tables["dev_registry"].get(p["name"])
        vers = (list(old["versions"]) if old else []) + [{"version": p["version"], "item": p["item"], "t": ev.t}]
        st.put("dev_registry", p["name"], {"owner": p["owner"], "latest": p["version"], "versions": vers,
                                           "deps": list(p.get("deps", [])), "t": ev.t})


class KbSite(QASite):
    """Knowledge sharing: Q&A and educational products (courses, explainers)."""

    @endpoint("courses", S.obj({"limit": S.s_int(minimum=1, maximum=50, default=20)}), read_only=True)
    def ep_courses(self, ctx, limit=20):
        """Educational products published here (course-format articles and apps), newest first."""
        cat = C.catalogue(ctx.world)
        m = (cat["platform"] == C.PLATFORM_CODE["kb"]) & ~cat["removed"] & \
            (cat["format"] == C.FORMAT_CODE["course"])
        ids_ = self._items(ctx.world, ctx.actor, m)
        return C.module(ctx.world).listing(ctx.world, ctx.actor, ids_, limit)


def _make(base, platform: str):
    from .platforms import PLATFORMS
    spec = PLATFORMS[platform]
    cls = type(f"Site_{platform}", (base,), {"name": platform, "host": spec["host"], "platform": platform,
                                            "__doc__": f"{spec['label']}: {spec['doc']}."})
    return register_module(_qa_appliers(_site_endpoints(cls)))


Dev = _make(DevSite, "dev")
Kb = _make(KbSite, "kb")
__all__ = ["Dev", "Kb", "np"]
