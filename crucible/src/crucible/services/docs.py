"""Documents (host `docs`): versioned documents with owner, visibility and explicit shares.

Visibility: `private` (owner, explicit shares, explicit grants), `org` (organisation-owned documents only: every
member reads and writes, per identity MEMBER_CAPS docs.read/docs.write), `public` (anyone reads; indexed in
search as kind `doc`). Access levels: read < write < admin. Owners (and owners/admins of an owning
organisation) are admin; shares give read or write (a share to an organisation covers its members); outsiders
may hold identity grants `docs.read` / `docs.write` / `docs.share` from the owner. Members of an owning
organisation do *not* see its private documents unless shared with them.

Every edit appends a version record (`doc_versions`), so history is complete. Events about non-public
documents are visible only to the owner, the acting principal and share holders; reads of a document you
cannot access fail with not_found (existence is not revealed). Reading a document makes you learn the service
hosts it mentions.

Internal API: `create_doc(ctx, owner, title, body, visibility="public") -> doc_id` (no authorisation check).

Invariants: version count equals the current version and the latest version matches the document; `org`
visibility only on organisation-owned documents; a document is in the search index iff it is public.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, register_module
from .identity import ident
from .search import extract_hosts, index as search_index, unindex as search_unindex

VISIBILITY = ("private", "org", "public")
MAX_BODY = 100_000
NONE, READ, WRITE, ADMIN = 0, 1, 2, 3
LEVEL_NAMES = {READ: "read", WRITE: "write", ADMIN: "admin"}


@register_module
class Docs(Module):
    name = "docs"
    host = "docs"
    depends = ("identity",)

    def install(self, world):
        st = world.state
        st.declare_table("documents", index=("owner", "visibility"))
        st.declare_table("doc_versions", index=("doc",))
        st.declare_table("doc_shares", index=("doc", "principal"))

    # ------------------------------------------------------------------ internal API
    def create_doc(self, ctx: Ctx, owner: str, title: str, body: str, visibility: str = "public") -> str:
        if visibility not in VISIBILITY:
            raise InvalidAction(f"bad visibility {visibility}")
        if visibility == "org" and not owner.startswith("o:"):
            raise InvalidAction("org visibility needs an organisation-owned document")
        title, body = str(title)[:200], str(body)[:MAX_BODY]
        did = ctx.state.new_id("doc")
        hosts = extract_hosts(ctx.world, title + "\n" + body)
        ctx.emit("docs.create", {"id": did, "owner": owner, "title": title, "body": body, "visibility": visibility,
                                 "hosts": hosts, "by": ctx.actor},
                 vis="public" if visibility == "public" else {owner, ctx.actor})
        if visibility == "public":
            search_index(ctx, did, "doc", title, body, owner, ref=did, hosts=hosts)
        return did

    def access(self, ctx: Ctx, actor: str, did: str, d) -> int:
        """Access level of `actor` on document `did` (record `d`)."""
        st = ctx.state
        owner, vis = d["owner"], d["visibility"]
        if actor == owner or actor == SYSTEM:
            return ADMIN
        idm = ident(ctx.world)
        level = READ if vis == "public" else NONE
        role = None
        if owner.startswith("o:"):
            o = st.get("orgs", owner)
            if o is not None and o["status"] == "active":
                role = idm.role(st, owner, actor)
            if role in ("owner", "admin"):
                return ADMIN
            if role == "member" and vis != "private":
                level = max(level, WRITE)
        for p in [actor] + [m["org"] for m in st.find("members", "person", actor)]:
            s = st.get("doc_shares", f"{did}|{p}")
            if s is not None:
                level = max(level, WRITE if s["perm"] == "write" else READ)
        if role is None:  # outsiders: explicit identity grants from the owner
            if idm.allowed(ctx, actor, owner, "docs.share", scope=did):
                return ADMIN
            if level < WRITE and idm.allowed(ctx, actor, owner, "docs.write", scope=did):
                level = WRITE
            if level < READ and idm.allowed(ctx, actor, owner, "docs.read", scope=did):
                level = READ
        return level

    def _doc(self, ctx, did: str, need: int):
        d = ctx.state.get("documents", did)
        if d is None:
            raise NotFound(f"no document {did}")
        lvl = self.access(ctx, ctx.actor, did, d)
        if lvl < READ:
            raise NotFound(f"no document {did}")
        if lvl < need:
            raise Forbidden(f"you need {LEVEL_NAMES[need]} access to {did}")
        return d, lvl

    def _vis(self, ctx, did: str, d, *extra: str):
        if d["visibility"] == "public":
            return "public"
        shares = [ctx.state.tables["doc_shares"][s]["principal"] for s in ctx.state.find_ids("doc_shares", "doc", did)]
        return {d["owner"], ctx.actor, *shares, *extra}

    # ------------------------------------------------------------------ endpoints
    @endpoint("create", S.obj({"title": S.s_str(minLength=1, maxLength=200), "body": S.s_str(maxLength=MAX_BODY),
                               "visibility": S.s_enum(list(VISIBILITY), "private (default), org or public"),
                               "org": S.s_str("create for an organisation (needs docs.write on it)", maxLength=40)},
                              ["title", "body"]), cost_s=30)
    def ep_create(self, ctx, title, body, visibility="private", org=None):
        """Create a document. Public documents are indexed by search; org visibility means every member of the
        owning organisation can read and edit it."""
        owner = ctx.actor
        if org:
            owner = ident(ctx.world).resolve_handle(ctx.state, org)
            if not owner.startswith("o:"):
                raise InvalidAction("org must be an organisation")
            ident(ctx.world).authorize(ctx, owner, "docs.write")
        return {"doc": self.create_doc(ctx, owner, title, body, visibility)}

    @endpoint("get", S.obj({"doc": S.s_str(maxLength=40),
                            "version": S.s_int("a past version (default: current)", minimum=1)}, ["doc"]),
              cost_s=5, read_only=True)
    def ep_get(self, ctx, doc, version=None):
        """Read a document (or one of its past versions). You learn the service hosts it mentions."""
        d, lvl = self._doc(ctx, doc, READ)
        title, body, v, t, hosts = d["title"], d["body"], d["version"], d["updated"], list(d["hosts"])
        if version is not None and version != d["version"]:
            rec = ctx.state.get("doc_versions", f"{doc}|{version:06d}")
            if rec is None:
                raise NotFound(f"{doc} has no version {version}")
            title, body, v, t = rec["title"], rec["body"], rec["v"], rec["t"]
            hosts = []
        for h in hosts:
            ctx.world.learn(ctx, ctx.actor, "host", h)
        out = {"doc": doc, "owner": d["owner"], "title": title, "body": body, "visibility": d["visibility"],
               "version": v, "latest_version": d["version"], "created": d["created"], "updated": t,
               "access": LEVEL_NAMES[lvl], "hosts": hosts}
        if lvl >= ADMIN:
            out["shares"] = [{"principal": s["principal"], "perm": s["perm"]}
                             for s in ctx.state.find("doc_shares", "doc", doc)]
        return out

    @endpoint("history", S.obj({"doc": S.s_str(maxLength=40), "limit": S.s_int(minimum=1, maximum=50, default=20)},
                               ["doc"]), read_only=True)
    def ep_history(self, ctx, doc, limit=20):
        """Version history (newest first): version, author, time, title."""
        self._doc(ctx, doc, READ)
        rows = [ctx.state.tables["doc_versions"][vid] for vid in ctx.state.find_ids("doc_versions", "doc", doc)]
        rows.sort(key=lambda r: -r["v"])
        return [{"version": r["v"], "author": r["author"], "t": r["t"], "title": r["title"]} for r in rows[:limit]]

    @endpoint("list", S.obj({"scope": S.s_enum(["mine", "org", "shared", "public"], "which documents"),
                             "org": S.s_str("organisation for scope=org", maxLength=40),
                             "limit": S.s_int(minimum=1, maximum=50, default=20)}), read_only=True)
    def ep_list(self, ctx, scope="mine", org=None, limit=20):
        """List documents you own, your organisation's documents, documents shared with you, or recent public
        documents (most recently updated first)."""
        st = ctx.state
        if scope == "mine":
            ids = st.find_ids("documents", "owner", ctx.actor)
        elif scope == "org":
            if not org:
                raise InvalidAction("scope=org needs org")
            oid = ident(ctx.world).resolve_handle(st, org)
            ids = st.find_ids("documents", "owner", oid)
        elif scope == "shared":
            ids = sorted({st.tables["doc_shares"][s]["doc"] for s in st.find_ids("doc_shares", "principal", ctx.actor)})
        else:
            ids = st.find_ids("documents", "visibility", "public")
        rows = []
        for did in ids:
            d = st.tables["documents"][did]
            if scope in ("mine", "public") or self.access(ctx, ctx.actor, did, d) >= READ:
                rows.append((-d["updated"], did, d))
        rows.sort(key=lambda x: (x[0], x[1]))
        return [{"doc": did, "owner": d["owner"], "title": d["title"], "visibility": d["visibility"],
                 "version": d["version"], "updated": d["updated"]} for _, did, d in rows[:limit]]

    @endpoint("edit", S.obj({"doc": S.s_str(maxLength=40), "title": S.s_str(minLength=1, maxLength=200),
                             "body": S.s_str(maxLength=MAX_BODY),
                             "base_version": S.s_int("fail if the document changed since this version", minimum=1)},
                            ["doc"]), cost_s=30)
    def ep_edit(self, ctx, doc, title=None, body=None, base_version=None):
        """Edit a document (needs write access). Creates a new version; pass base_version to avoid overwriting
        someone else's concurrent edit."""
        d, _ = self._doc(ctx, doc, WRITE)
        if title is None and body is None:
            raise InvalidAction("nothing to change")
        if base_version is not None and base_version != d["version"]:
            raise Conflict(f"document is at version {d['version']}", version=d["version"])
        title = d["title"] if title is None else title
        body = d["body"] if body is None else body
        hosts = extract_hosts(ctx.world, title + "\n" + body)
        v = d["version"] + 1
        ctx.emit("docs.edit", {"id": doc, "v": v, "title": title, "body": body, "hosts": hosts, "by": ctx.actor},
                 vis=self._vis(ctx, doc, d))
        if d["visibility"] == "public":
            search_index(ctx, doc, "doc", title, body, d["owner"], ref=doc, hosts=hosts)
        return {"doc": doc, "version": v}

    @endpoint("share", S.obj({"doc": S.s_str(maxLength=40), "principal": S.s_str(maxLength=60),
                              "perm": S.s_enum(["read", "write", "none"], "none removes the share")},
                             ["doc", "principal", "perm"]), cost_s=5)
    def ep_share(self, ctx, doc, principal, perm):
        """Share a document with a person or organisation (needs admin access: owner, org owner/admin, or a
        docs.share grant)."""
        d, _ = self._doc(ctx, doc, ADMIN)
        pid = ident(ctx.world).resolve_handle(ctx.state, principal)
        if pid == d["owner"]:
            raise InvalidAction("the owner already has full access")
        cur = ctx.state.get("doc_shares", f"{doc}|{pid}")
        if (cur["perm"] if cur else "none") == perm:
            raise Conflict("share unchanged")
        ctx.emit("docs.share", {"doc": doc, "principal": pid, "perm": perm, "by": ctx.actor},
                 vis={d["owner"], pid, ctx.actor})
        return {"ok": True}

    @endpoint("set_visibility", S.obj({"doc": S.s_str(maxLength=40), "visibility": S.s_enum(list(VISIBILITY))},
                                      ["doc", "visibility"]), cost_s=5)
    def ep_set_visibility(self, ctx, doc, visibility):
        """Change who can see a document (needs admin access). Making it public indexes it in search; making it
        non-public removes it from search."""
        d, _ = self._doc(ctx, doc, ADMIN)
        if visibility == d["visibility"]:
            raise Conflict("visibility unchanged")
        if visibility == "org" and not d["owner"].startswith("o:"):
            raise InvalidAction("org visibility needs an organisation-owned document")
        ctx.emit("docs.visibility", {"doc": doc, "visibility": visibility, "by": ctx.actor},
                 vis={d["owner"], ctx.actor})
        if visibility == "public":
            search_index(ctx, doc, "doc", d["title"], d["body"], d["owner"], ref=doc, hosts=d["hosts"])
        else:
            search_unindex(ctx, doc)
        return {"ok": True}

    # ------------------------------------------------------------------ appliers
    @applier("docs.create")
    def ap_create(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("documents", p["id"], {"owner": p["owner"], "title": p["title"], "body": p["body"],
                                         "visibility": p["visibility"], "version": 1, "created": ev.t,
                                         "updated": ev.t, "hosts": list(p["hosts"]), "by": p["by"]})
        st.insert("doc_versions", f"{p['id']}|{1:06d}", {"doc": p["id"], "v": 1, "title": p["title"],
                                                         "body": p["body"], "author": p["by"], "t": ev.t})

    @applier("docs.edit")
    def ap_edit(self, ev):
        p = ev.payload
        st = self.world.state
        st.patch("documents", p["id"], title=p["title"], body=p["body"], version=p["v"], updated=ev.t,
                 hosts=list(p["hosts"]))
        st.insert("doc_versions", f"{p['id']}|{p['v']:06d}", {"doc": p["id"], "v": p["v"], "title": p["title"],
                                                              "body": p["body"], "author": p["by"], "t": ev.t})

    @applier("docs.share")
    def ap_share(self, ev):
        p = ev.payload
        sid = f"{p['doc']}|{p['principal']}"
        if p["perm"] == "none":
            self.world.state.delete("doc_shares", sid)
        else:
            self.world.state.put("doc_shares", sid, {"doc": p["doc"], "principal": p["principal"], "perm": p["perm"],
                                                     "t": ev.t})

    @applier("docs.visibility")
    def ap_visibility(self, ev):
        self.world.state.patch("documents", ev.payload["doc"], visibility=ev.payload["visibility"], updated=ev.t)

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        versions: dict[str, list[int]] = {}
        for vid, v in st.tables["doc_versions"].items():
            versions.setdefault(v["doc"], []).append(v["v"])
        indexed = st.tables.get("search_docs")
        for did, d in st.tables["documents"].items():
            vs = sorted(versions.get(did, []))
            if vs != list(range(1, d["version"] + 1)):
                errs.append(f"{did} versions {vs[:5]}... do not match version {d['version']}")
            else:
                last = st.tables["doc_versions"][f"{did}|{d['version']:06d}"]
                if (last["title"], last["body"]) != (d["title"], d["body"]):
                    errs.append(f"{did} latest version differs from the document")
            if d["visibility"] not in VISIBILITY:
                errs.append(f"{did} has bad visibility")
            if d["visibility"] == "org" and not d["owner"].startswith("o:"):
                errs.append(f"{did} has org visibility but a non-organisation owner")
            if indexed is not None:
                rec = indexed.get(did)
                if d["visibility"] == "public" and rec is None:
                    errs.append(f"public {did} missing from search")
                if d["visibility"] != "public" and rec is not None:
                    errs.append(f"non-public {did} is in the search index")
        for sid, s in st.tables["doc_shares"].items():
            if s["doc"] not in st.tables["documents"] or s["perm"] not in ("read", "write"):
                errs.append(f"bad share {sid}")
        return errs

    def observe(self, world, actor):
        st = world.state
        mine = len(st.find_ids("documents", "owner", actor))
        shared = len(st.find_ids("doc_shares", "principal", actor))
        return {"docs": mine, "shared_with_me": shared} if mine or shared else None
