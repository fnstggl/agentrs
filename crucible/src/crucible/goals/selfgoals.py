"""Self-generated goals (module `goals`, host `goals`).

An agent may register its own experiments, hypotheses, subgoals or projects as private predicates over metrics it
is allowed to observe (its own cash, its organisations' independent revenue, its own services' requests and
availability). The kernel checks them at their due time and tells only the agent the outcome. They never change
the independent success criteria (the evaluator ignores this table), and an optional intrinsic reward for them is
computed only by a research-side RewardContract that explicitly enables it.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Forbidden, InvalidAction, NotFound
from ..core.kernel import Module, applier, endpoint, process, register_module
from .metrics import check_predicate, evaluate_predicate

ALLOWED = {"cash": "principal", "independent_revenue": "principal", "service_requests_ok": "host",
           "service_availability": "host", "repo_commits": "repo", "independent_members": "org",
           "org_status_active": "org"}


@register_module
class SelfGoals(Module):
    name = "goals"
    depends = ("identity",)

    def install(self, world):
        world.state.declare_table("self_goals", index=("owner", "status"))

    def _check_scope(self, ctx, p):
        if "all" in p or "any" in p:
            for x in p.get("all", p.get("any", [])):
                self._check_scope(ctx, x)
            return
        if "not" in p:
            return self._check_scope(ctx, p["not"])
        m = p["metric"]
        if m not in ALLOWED:
            raise Forbidden(f"metric {m} is not observable by agents")
        arg = (p.get("args") or {}).get(ALLOWED[m])
        mine = ctx.world.principals_of(ctx.actor)
        st = ctx.state
        if ALLOWED[m] in ("principal", "org") and arg not in mine and not st.has("members", f"{arg}|{ctx.actor}"):
            raise Forbidden("self-goals may only reference your own principals and organisations")
        if ALLOWED[m] == "host":
            h = st.get("hosts", arg or "") if "hosts" in st.tables else None
            if h is None or h["owner"] not in mine:
                raise Forbidden("self-goals may only reference services you operate")
        if ALLOWED[m] == "repo":
            r = st.get("repos", arg or "") if "repos" in st.tables else None
            if r is None or r["owner"] not in mine:
                raise Forbidden("self-goals may only reference your repositories")

    @endpoint("propose", S.obj({"title": S.s_str(maxLength=200), "predicate": S.s_any_obj("metric predicate"),
                                "due_in_s": S.s_int(minimum=3600, maximum=365 * 86400),
                                "note": S.s_str("hypothesis / plan, for your own records", maxLength=2000)},
                               ["title", "predicate", "due_in_s"]), cost_s=30)
    def ep_propose(self, ctx, title, predicate, due_in_s, note=""):
        """Register a private subgoal / experiment: a predicate over your own observable metrics, checked at its
        due time. Example: {"metric": "independent_revenue", "args": {"principal": "o:me", "window_s": 604800},
        "op": ">=", "value": 10000}. Metrics: cash, independent_revenue, service_requests_ok, service_availability,
        repo_commits, independent_members, org_status_active."""
        errs = check_predicate(predicate)
        if errs:
            raise InvalidAction("; ".join(errs))
        self._check_scope(ctx, predicate)
        if len([g for g in ctx.state.find("self_goals", "owner", ctx.actor) if g["status"] == "open"]) >= 20:
            raise InvalidAction("at most 20 open self-goals")
        gid = ctx.state.new_id("sg")
        ctx.emit("goals.proposed", {"id": gid, "owner": ctx.actor, "title": title, "predicate": predicate,
                                    "due": ctx.now + due_in_s, "note": note}, vis=(ctx.actor,))
        ctx.schedule(ctx.now + due_in_s, "goals.check", {"id": gid}, prio=80)
        return {"goal": gid, "due": ctx.now + due_in_s}

    @endpoint("mine", S.obj({"status": S.s_enum(["open", "met", "missed"])}), read_only=True)
    def ep_mine(self, ctx, status=None):
        return [dict(g, id=gid) for gid in ctx.state.find_ids("self_goals", "owner", ctx.actor)
                for g in [ctx.state.tables["self_goals"][gid]] if status is None or g["status"] == status]

    @endpoint("check_now", S.obj({"goal": S.s_str(maxLength=40)}, ["goal"]), read_only=True)
    def ep_check_now(self, ctx, goal):
        g = ctx.state.get("self_goals", goal)
        if g is None or g["owner"] != ctx.actor:
            raise NotFound("no such goal")
        return {"goal": goal, "currently_true": evaluate_predicate(ctx.world, g["predicate"])}

    @process("check")
    def pr_check(self, ctx, id):
        g = ctx.state.get("self_goals", id)
        if g is None or g["status"] != "open":
            return
        okv = evaluate_predicate(ctx.world, g["predicate"])
        ctx.emit("goals.checked", {"id": id, "met": okv}, vis=(g["owner"],))

    @applier("goals.proposed")
    def ap_proposed(self, ev):
        p = ev.payload
        self.world.state.insert("self_goals", p["id"], {"owner": p["owner"], "title": p["title"],
                                                       "predicate": p["predicate"], "due": p["due"], "note": p["note"],
                                                       "status": "open", "t": ev.t})

    @applier("goals.checked")
    def ap_checked(self, ev):
        self.world.state.patch("self_goals", ev.payload["id"], status="met" if ev.payload["met"] else "missed",
                               checked=ev.t)
