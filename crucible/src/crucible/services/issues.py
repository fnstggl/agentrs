"""Issue tracker (host `issues`): how teams and their users coordinate work on software and services.

A project is owned by a principal (person or organisation) and may reference a repository. Issues have a
title, body, status (`open` -> `in_progress` -> `closed`, reopenable), labels, an assignee and comments.

Authorisation: writing (creating issues, commenting, assigning, changing status/labels) needs `issues.write` on
the project owner (organisation members hold it via MEMBER_CAPS; outsiders via grants). On a `public` project
anyone may read, open issues and comment, and an issue's creator may edit or close their own issue. Private
projects are invisible to outsiders (not_found). Events of public projects are public; otherwise they are
visible to the owner, the creator, the assignee and the actor.

Hook: `world.hooks["on_issue_assigned"](ctx, issue_id, assignee)` whenever an issue gets an assignee.

Invariants: per-project open counters equal a recount; closed issues (and only those) carry a closing time;
comment counters match; issues reference existing projects.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import Ctx, Module, applier, endpoint, register_module
from .identity import ident

STATUSES = ("open", "in_progress", "closed")
NAME_RE = r"[a-z0-9][a-z0-9._-]{0,60}"
LABEL_RE = r"[a-z0-9][a-z0-9:._-]{0,30}"
MAX_BODY = 20_000


@register_module
class Issues(Module):
    name = "issues"
    host = "issues"
    depends = ("identity",)

    def install(self, world):
        st = world.state
        st.declare_table("iss_projects", index=("owner",))
        st.declare_table("iss_issues", index=("project", "assignee", "creator"))
        st.declare_table("iss_comments", index=("issue",))

    # ------------------------------------------------------------------ authorisation helpers
    def can_write(self, ctx: Ctx, proj) -> bool:
        return ident(ctx.world).allowed(ctx, ctx.actor, proj["owner"], "issues.write")

    def can_read(self, ctx: Ctx, proj) -> bool:
        if proj["public"]:
            return True
        idm = ident(ctx.world)
        return (idm.allowed(ctx, ctx.actor, proj["owner"], "issues.write")
                or idm.allowed(ctx, ctx.actor, proj["owner"], "issues.read"))

    def _project(self, ctx, pid: str):
        p = ctx.state.get("iss_projects", pid)
        if p is None or not self.can_read(ctx, p):
            raise NotFound(f"no project {pid}")
        return p

    def _issue(self, ctx, iid: str):
        i = ctx.state.get("iss_issues", iid)
        if i is None:
            raise NotFound(f"no issue {iid}")
        return i, self._project(ctx, i["project"])

    def _vis(self, ctx, proj, issue=None, *extra):
        if proj["public"]:
            return "public"
        s = {proj["owner"], ctx.actor, *[x for x in extra if x]}
        if issue is not None:
            s.add(issue["creator"])
            if issue["assignee"]:
                s.add(issue["assignee"])
        return s

    # ------------------------------------------------------------------ endpoints: projects
    @endpoint("create_project", S.obj({"name": S.s_str(pattern=NAME_RE, maxLength=61),
                                       "owner": S.s_str("organisation owning it (default: you)", maxLength=40),
                                       "repo": S.s_str("repository reference", maxLength=120),
                                       "public": S.s_bool("anyone may read and file issues", default=False),
                                       "description": S.s_str(maxLength=2000)}, ["name"]), cost_s=20)
    def ep_create_project(self, ctx, name, owner=None, repo=None, public=False, description=""):
        """Create an issue-tracking project for yourself or an organisation (needs issues.write on it)."""
        st = ctx.state
        own = ident(ctx.world).resolve_handle(st, owner) if owner else ctx.actor
        if own != ctx.actor:
            ident(ctx.world).authorize(ctx, own, "issues.write")
        if any(st.tables["iss_projects"][p]["name"] == name for p in st.find_ids("iss_projects", "owner", own)):
            raise Conflict(f"{own} already has a project named {name}")
        pid = st.new_id("proj")
        ctx.emit("issues.project", {"id": pid, "owner": own, "name": name, "repo": repo, "public": public,
                                    "description": description, "by": ctx.actor},
                 vis="public" if public else {own, ctx.actor})
        return {"project": pid}

    @endpoint("projects", S.obj({"owner": S.s_str("principal or handle (default: you)", maxLength=40)}),
              read_only=True)
    def ep_projects(self, ctx, owner=None):
        """Projects of a principal that you can see."""
        own = ident(ctx.world).resolve_handle(ctx.state, owner) if owner else ctx.actor
        out = []
        for pid in ctx.state.find_ids("iss_projects", "owner", own):
            p = ctx.state.tables["iss_projects"][pid]
            if self.can_read(ctx, p):
                out.append({"project": pid, "name": p["name"], "owner": p["owner"], "repo": p["repo"],
                            "public": p["public"], "open_issues": p["open"], "description": p["description"]})
        return out

    # ------------------------------------------------------------------ endpoints: issues
    @endpoint("create", S.obj({"project": S.s_str(maxLength=40), "title": S.s_str(minLength=1, maxLength=200),
                               "body": S.s_str(maxLength=MAX_BODY),
                               "labels": S.s_arr(S.s_str(pattern=LABEL_RE, maxLength=31), maxItems=10),
                               "assignee": S.s_str("principal to assign (needs issues.write)", maxLength=60)},
                              ["project", "title"]), cost_s=30)
    def ep_create(self, ctx, project, title, body="", labels=(), assignee=None):
        """File an issue (needs issues.write on the project owner, or a public project)."""
        proj = self._project(ctx, project)
        writer = self.can_write(ctx, proj)
        if not writer and not proj["public"]:
            raise Forbidden(f"{ctx.actor} lacks issues.write on {proj['owner']}")
        asg = None
        if assignee:
            if not writer:
                raise Forbidden("assigning needs issues.write")
            asg = ident(ctx.world).resolve_handle(ctx.state, assignee)
        iid = ctx.state.new_id("iss")
        issue = {"creator": ctx.actor, "assignee": asg}
        ctx.emit("issues.create", {"id": iid, "project": project, "title": title, "body": body,
                                   "labels": sorted(set(labels)), "assignee": asg, "creator": ctx.actor},
                 vis=self._vis(ctx, proj, issue))
        if asg:
            self._assigned(ctx, iid, asg)
        return {"issue": iid}

    def _assigned(self, ctx, iid: str, assignee: str) -> None:
        hook = ctx.world.hooks.get("on_issue_assigned")
        if hook is not None:
            hook(ctx, iid, assignee)

    @endpoint("list", S.obj({"project": S.s_str(maxLength=40), "status": S.s_enum(list(STATUSES) + ["any"]),
                             "assignee": S.s_str(maxLength=60), "label": S.s_str(pattern=LABEL_RE, maxLength=31),
                             "limit": S.s_int(minimum=1, maximum=50, default=20)}, ["project"]), read_only=True)
    def ep_list(self, ctx, project, status="open", assignee=None, label=None, limit=20):
        """Issues of a project, newest first (default: open ones)."""
        self._project(ctx, project)
        st = ctx.state
        asg = ident(ctx.world).resolve_handle(st, assignee) if assignee else None
        rows = []
        for iid in st.find_ids("iss_issues", "project", project):
            i = st.tables["iss_issues"][iid]
            if status != "any" and i["status"] != status:
                continue
            if asg is not None and i["assignee"] != asg:
                continue
            if label is not None and label not in i["labels"]:
                continue
            rows.append((-i["t"], iid, i))
        rows.sort(key=lambda x: (x[0], x[1]))
        return [_summary(iid, i) for _, iid, i in rows[:limit]]

    @endpoint("assigned", S.obj({"limit": S.s_int(minimum=1, maximum=50, default=20)}), read_only=True)
    def ep_assigned(self, ctx, limit=20):
        """Open issues assigned to you, across projects."""
        st = ctx.state
        rows = [(iid, st.tables["iss_issues"][iid]) for iid in st.find_ids("iss_issues", "assignee", ctx.actor)]
        rows = [(iid, i) for iid, i in rows if i["status"] != "closed"]
        rows.sort(key=lambda x: (-x[1]["updated"], x[0]))
        return [_summary(iid, i) for iid, i in rows[:limit]]

    @endpoint("get", S.obj({"issue": S.s_str(maxLength=40),
                            "comments": S.s_int(minimum=0, maximum=100, default=50)}, ["issue"]), read_only=True)
    def ep_get(self, ctx, issue, comments=50):
        """An issue with its comments (oldest first)."""
        i, _ = self._issue(ctx, issue)
        st = ctx.state
        cids = sorted(st.find_ids("iss_comments", "issue", issue),
                      key=lambda c: (st.tables["iss_comments"][c]["t"], c))
        cs = [{"comment": c, "author": st.tables["iss_comments"][c]["author"],
               "body": st.tables["iss_comments"][c]["body"], "t": st.tables["iss_comments"][c]["t"]}
              for c in cids[:comments]]
        return dict(_summary(issue, i), body=i["body"], comments=cs)

    @endpoint("comment", S.obj({"issue": S.s_str(maxLength=40), "body": S.s_str(minLength=1, maxLength=MAX_BODY)},
                               ["issue", "body"]), cost_s=20)
    def ep_comment(self, ctx, issue, body):
        """Comment on an issue (needs issues.write, or a public project)."""
        i, proj = self._issue(ctx, issue)
        if not proj["public"] and not self.can_write(ctx, proj):
            raise Forbidden(f"{ctx.actor} lacks issues.write on {proj['owner']}")
        cid = ctx.state.new_id("icmt")
        ctx.emit("issues.comment", {"id": cid, "issue": issue, "author": ctx.actor, "body": body},
                 vis=self._vis(ctx, proj, i))
        return {"comment": cid}

    @endpoint("assign", S.obj({"issue": S.s_str(maxLength=40),
                               "assignee": S.s_str("principal; omit to unassign", maxLength=60)}, ["issue"]),
              cost_s=5)
    def ep_assign(self, ctx, issue, assignee=None):
        """Assign (or unassign) an issue (needs issues.write)."""
        i, proj = self._issue(ctx, issue)
        if not self.can_write(ctx, proj):
            raise Forbidden(f"{ctx.actor} lacks issues.write on {proj['owner']}")
        asg = ident(ctx.world).resolve_handle(ctx.state, assignee) if assignee else None
        if asg == i["assignee"]:
            raise Conflict("assignee unchanged")
        ctx.emit("issues.update", {"id": issue, "changes": {"assignee": asg}, "by": ctx.actor},
                 vis=self._vis(ctx, proj, i, asg))
        if asg:
            self._assigned(ctx, issue, asg)
        return {"issue": issue, "assignee": asg}

    @endpoint("update", S.obj({"issue": S.s_str(maxLength=40), "status": S.s_enum(list(STATUSES)),
                               "labels": S.s_arr(S.s_str(pattern=LABEL_RE, maxLength=31), maxItems=10),
                               "title": S.s_str(minLength=1, maxLength=200), "body": S.s_str(maxLength=MAX_BODY)},
                              ["issue"]), cost_s=10)
    def ep_update(self, ctx, issue, status=None, labels=None, title=None, body=None):
        """Change status, labels, title or body. Needs issues.write; an issue's creator may edit its title/body
        and close it."""
        i, proj = self._issue(ctx, issue)
        writer = self.can_write(ctx, proj)
        creator = i["creator"] == ctx.actor
        changes = {}
        if status is not None and status != i["status"]:
            if not writer and not (creator and status == "closed"):
                raise Forbidden("changing status needs issues.write")
            changes["status"] = status
        if labels is not None and sorted(set(labels)) != list(i["labels"]):
            if not writer:
                raise Forbidden("changing labels needs issues.write")
            changes["labels"] = sorted(set(labels))
        for k, v in (("title", title), ("body", body)):
            if v is not None and v != i[k]:
                if not writer and not creator:
                    raise Forbidden(f"editing {k} needs issues.write")
                changes[k] = v
        if not changes:
            raise InvalidAction("nothing to change")
        ctx.emit("issues.update", {"id": issue, "changes": changes, "by": ctx.actor}, vis=self._vis(ctx, proj, i))
        return {"issue": issue, "changed": sorted(changes)}

    @endpoint("close", S.obj({"issue": S.s_str(maxLength=40), "comment": S.s_str(maxLength=MAX_BODY)}, ["issue"]),
              cost_s=10)
    def ep_close(self, ctx, issue, comment=None):
        """Close an issue, optionally with a final comment (issues.write, or the issue's creator)."""
        i, proj = self._issue(ctx, issue)
        if not self.can_write(ctx, proj) and i["creator"] != ctx.actor:
            raise Forbidden("closing needs issues.write")
        if i["status"] == "closed":
            raise Conflict("already closed")
        if comment:
            cid = ctx.state.new_id("icmt")
            ctx.emit("issues.comment", {"id": cid, "issue": issue, "author": ctx.actor, "body": comment},
                     vis=self._vis(ctx, proj, i))
        ctx.emit("issues.update", {"id": issue, "changes": {"status": "closed"}, "by": ctx.actor},
                 vis=self._vis(ctx, proj, i))
        return {"issue": issue, "status": "closed"}

    # ------------------------------------------------------------------ appliers
    @applier("issues.project")
    def ap_project(self, ev):
        p = ev.payload
        self.world.state.insert("iss_projects", p["id"], {"owner": p["owner"], "name": p["name"], "repo": p["repo"],
                                                          "public": p["public"], "description": p["description"],
                                                          "open": 0, "issues": 0, "created": ev.t, "by": p["by"]})

    @applier("issues.create")
    def ap_create(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("iss_issues", p["id"], {"project": p["project"], "title": p["title"], "body": p["body"],
                                          "status": "open", "labels": list(p["labels"]), "assignee": p["assignee"],
                                          "creator": p["creator"], "t": ev.t, "updated": ev.t, "closed_t": None,
                                          "comments": 0})
        proj = st.tables["iss_projects"][p["project"]]
        st.patch("iss_projects", p["project"], open=proj["open"] + 1, issues=proj["issues"] + 1)

    @applier("issues.update")
    def ap_update(self, ev):
        p = ev.payload
        st = self.world.state
        i = st.tables["iss_issues"][p["id"]]
        ch = dict(p["changes"])
        if "status" in ch:
            was_closed, now_closed = i["status"] == "closed", ch["status"] == "closed"
            if was_closed != now_closed:
                proj = st.tables["iss_projects"][i["project"]]
                st.patch("iss_projects", i["project"], open=proj["open"] + (1 if was_closed else -1))
            ch["closed_t"] = ev.t if now_closed else None
        if "labels" in ch:
            ch["labels"] = list(ch["labels"])
        st.patch("iss_issues", p["id"], updated=ev.t, **ch)

    @applier("issues.comment")
    def ap_comment(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("iss_comments", p["id"], {"issue": p["issue"], "author": p["author"], "body": p["body"], "t": ev.t})
        i = st.tables["iss_issues"][p["issue"]]
        st.patch("iss_issues", p["issue"], comments=i["comments"] + 1, updated=ev.t)

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        open_n: dict[str, int] = {}
        all_n: dict[str, int] = {}
        for iid, i in st.tables["iss_issues"].items():
            if i["project"] not in st.tables["iss_projects"]:
                errs.append(f"issue {iid} in a missing project")
                continue
            if i["status"] not in STATUSES:
                errs.append(f"issue {iid} has bad status {i['status']}")
            if (i["status"] == "closed") != (i["closed_t"] is not None):
                errs.append(f"issue {iid} status/closing time mismatch")
            all_n[i["project"]] = all_n.get(i["project"], 0) + 1
            if i["status"] != "closed":
                open_n[i["project"]] = open_n.get(i["project"], 0) + 1
        for pid, p in st.tables["iss_projects"].items():
            if p["open"] != open_n.get(pid, 0) or p["issues"] != all_n.get(pid, 0):
                errs.append(f"project {pid} counters {p['open']}/{p['issues']} != {open_n.get(pid, 0)}/"
                            f"{all_n.get(pid, 0)}")
        cn: dict[str, int] = {}
        for cid, c in st.tables["iss_comments"].items():
            cn[c["issue"]] = cn.get(c["issue"], 0) + 1
        for iid, i in st.tables["iss_issues"].items():
            if i["comments"] != cn.get(iid, 0):
                errs.append(f"issue {iid} comment counter {i['comments']} != {cn.get(iid, 0)}")
        return errs

    def observe(self, world, actor):
        st = world.state
        n = sum(1 for i in st.find("iss_issues", "assignee", actor) if i["status"] != "closed")
        return {"assigned_open": n} if n else None


def _summary(iid: str, i) -> dict:
    return {"issue": iid, "project": i["project"], "title": i["title"], "status": i["status"],
            "labels": list(i["labels"]), "assignee": i["assignee"], "creator": i["creator"], "t": i["t"],
            "updated": i["updated"], "comments": i["comments"]}
