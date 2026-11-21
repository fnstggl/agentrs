"""Workflows, triggers and asynchronous jobs (module `wf`, host `wf`).

Composition primitives for building automation out of any tools in the world - built-in services and services
deployed by agents alike. Nothing here knows what a workflow is *for*.

* **Triggers** run one tool call on a schedule (`every_s`) or when events visible to the owner occur
  (`on_event`: an event-type prefix such as "mail.", "support.", "ops.loss", "cloud.outage"). A trigger acts as
  its owner (a person, or an organisation the creator administers), with exactly the owner's permissions, and pays
  whatever the called endpoint costs. Ownership is re-checked at every run.
* **Flows** are typed step graphs: each step is a tool call whose arguments may reference the flow input and the
  results of earlier steps (`"${input.x}"`, `"${s1.result.items.0.id}"`, `"${item}"` inside `foreach`), with
  optional conditions and iteration. `wf.run` executes a flow atomically inside the caller's transaction (all steps
  or none); `wf.run_later` runs it asynchronously as a scheduled job. Flows can call flows (bounded depth), and
  a flow can be published so others can run it (as themselves) - a reusable, shareable software artifact.

Failures are recorded on the trigger (consecutive failures pause it after 10) and as a `dependency_failure` loss
of the owner when the called tool is an agent-deployed service that did not answer (503/5xx): reliance on other
people's software is a measured, real dependency.
"""
from __future__ import annotations

import re
from typing import Any

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from .identity import ident

REF = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)*)\}")
MAX_STEPS = 16
MAX_FOREACH = 50
MAX_TRIGGERS = 30
STEP = {"type": "object", "additionalProperties": True}


def _lookup(env: dict, path: str) -> Any:
    cur: Any = env
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise InvalidAction(f"reference ${{{path}}}: no field {part!r}")
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                raise InvalidAction(f"reference ${{{path}}}: bad index {part!r}")
        else:
            raise InvalidAction(f"reference ${{{path}}}: cannot index into {type(cur).__name__}")
    return cur


def substitute(x: Any, env: dict) -> Any:
    """Replace ${...} references. A string that is exactly one reference takes the referenced value's type."""
    if isinstance(x, str):
        m = REF.fullmatch(x)
        if m:
            return _lookup(env, m.group(1))
        return REF.sub(lambda mm: str(_lookup(env, mm.group(1))), x)
    if isinstance(x, dict):
        return {k: substitute(v, env) for k, v in x.items()}
    if isinstance(x, list):
        return [substitute(v, env) for v in x]
    return x


def _cond(c: dict | None, env: dict) -> bool:
    if not c:
        return True
    v = substitute(c.get("ref"), env)
    op, w = c.get("op", "truthy"), substitute(c.get("value"), env)
    try:
        return {"truthy": lambda: bool(v), "falsy": lambda: not v, "eq": lambda: v == w, "ne": lambda: v != w,
                "gt": lambda: v > w, "ge": lambda: v >= w, "lt": lambda: v < w, "le": lambda: v <= w,
                "in": lambda: v in w, "contains": lambda: w in v}[op]()
    except (KeyError, TypeError):
        raise InvalidAction(f"bad condition {c!r}")


@register_module
class Workflows(Module):
    name = "wf"
    depends = ("identity",)

    def install(self, world):
        st = world.state
        st.declare_table("wf_triggers", index=("owner", "status"))
        st.declare_table("wf_flows", index=("owner",))
        st.declare_table("wf_runs", index=("flow",))
        world.hooks["wf"] = self

    def genesis(self, ctx):
        ctx.schedule(ctx.now + 600, "wf.poll", {}, prio=60, period=600, key="wf.poll")

    # ------------------------------------------------------------------ principals
    def _run_as(self, ctx, as_org: str | None) -> str:
        if not as_org:
            return ctx.actor
        org = ident(ctx.world).resolve_handle(ctx.state, as_org)
        if not (ctx.actor == SYSTEM or ctx.world.may_act_as(ctx.actor, org) or ctx.actor == org):
            raise Forbidden(f"{ctx.actor} cannot act for {org}")
        return org

    def _still_allowed(self, ctx, t) -> bool:
        if t["owner"] == t["created_by"] or t["created_by"] == SYSTEM:
            return ctx.state.has("principals", t["owner"])
        return ctx.world.may_act_as(t["created_by"], t["owner"])

    # ------------------------------------------------------------------ triggers
    def create_trigger(self, ctx: Ctx, owner: str, created_by: str, tool: str, args: dict, every_s: int | None,
                       on_event: str | None, start_in_s: int, max_runs: int | None, label: str) -> str:
        st = ctx.state
        if every_s is None and not on_event:
            raise InvalidAction("a trigger needs every_s or on_event")
        if len([1 for t in st.find("wf_triggers", "owner", owner) if t["status"] != "deleted"]) >= MAX_TRIGGERS:
            raise Conflict(f"at most {MAX_TRIGGERS} triggers per owner")
        ctx.world.resolve(tool, owner)       # unknown tools are rejected now, not at run time
        if tool.startswith("admin."):
            raise Forbidden("admin tools are reserved for the simulator operator")
        tid = st.new_id("trg")
        sched = None
        if every_s is not None:
            sched = ctx.schedule(ctx.now + max(60, int(start_in_s or every_s)), "wf.fire", {"id": tid}, prio=55,
                                 period=int(every_s), key=f"wf.fire:{tid}")
        ctx.emit("wf.trigger", {"id": tid, "owner": owner, "created_by": created_by, "tool": tool, "args": args,
                                "every_s": every_s, "on_event": on_event, "max_runs": max_runs, "label": label[:80],
                                "sched": sched, "cursor": ctx.world.seq}, vis=(owner, created_by))
        return tid

    @endpoint("trigger", S.obj({
        "tool": S.s_str("tool to call, e.g. myhost.tick or wf.run", maxLength=80),
        "args": S.s_any_obj("arguments (passed verbatim)"),
        "every_s": S.s_int("run periodically (seconds, >= 600)", minimum=600, maximum=90 * 86400),
        "on_event": S.s_str("run when an event of this type prefix becomes visible to the owner "
                            "(e.g. 'mail.', 'support.ticket', 'ops.loss'); ${event} holds the event", maxLength=40),
        "as_org": S.s_str("run as an organisation you administer", maxLength=40),
        "start_in_s": S.s_int(minimum=0, maximum=90 * 86400), "max_runs": S.s_int(minimum=1, maximum=100_000),
        "label": S.s_str(maxLength=80)}, ["tool"]), cost_s=20)
    def ep_trigger(self, ctx, tool, args=None, every_s=None, on_event=None, as_org=None, start_in_s=0, max_runs=None,
                   label=""):
        """Create a trigger that calls `tool` as you (or as an organisation you administer) on a schedule or on
        events. The called endpoint's price is paid by the owner on every run."""
        owner = self._run_as(ctx, as_org)
        tid = self.create_trigger(ctx, owner, ctx.actor, tool, args or {}, every_s, on_event, start_in_s, max_runs,
                                  label)
        return {"trigger": tid, "owner": owner}

    @endpoint("triggers", S.obj({"as_org": S.s_str(maxLength=40)}), read_only=True)
    def ep_triggers(self, ctx, as_org=None):
        """Your triggers (or an organisation's)."""
        owner = self._run_as(ctx, as_org)
        return [{"trigger": tid, **{k: t[k] for k in ("tool", "args", "every_s", "on_event", "status", "runs",
                                                      "fails", "last", "label")}}
                for tid in ctx.state.find_ids("wf_triggers", "owner", owner)
                for t in [ctx.state.tables["wf_triggers"][tid]] if t["status"] != "deleted"]

    def set_status(self, ctx: Ctx, tid: str, status: str, reason: str = "") -> None:
        t = ctx.state.get("wf_triggers", tid)
        if t is not None and t["status"] != status and t["status"] != "deleted":
            ctx.emit("wf.trigger_status", {"id": tid, "status": status, "reason": reason[:200]},
                     vis=(t["owner"], t["created_by"]))

    def _own_trigger(self, ctx, tid):
        t = ctx.state.get("wf_triggers", tid)
        if t is None or t["status"] == "deleted":
            raise NotFound("no trigger " + tid)
        if not (ctx.actor in (t["owner"], t["created_by"]) or ctx.world.may_act_as(ctx.actor, t["owner"])):
            raise Forbidden("not your trigger")
        return t

    @endpoint("pause", S.obj({"trigger": S.s_str(maxLength=40)}, ["trigger"]))
    def ep_pause(self, ctx, trigger):
        """Pause a trigger."""
        self._own_trigger(ctx, trigger)
        self.set_status(ctx, trigger, "paused", "by owner")
        return {"ok": True}

    @endpoint("resume", S.obj({"trigger": S.s_str(maxLength=40)}, ["trigger"]))
    def ep_resume(self, ctx, trigger):
        """Resume a paused trigger."""
        self._own_trigger(ctx, trigger)
        self.set_status(ctx, trigger, "active", "by owner")
        return {"ok": True}

    @endpoint("delete", S.obj({"trigger": S.s_str(maxLength=40)}, ["trigger"]))
    def ep_delete(self, ctx, trigger):
        """Delete a trigger."""
        t = self._own_trigger(ctx, trigger)
        self.delete_trigger(ctx, trigger, t)
        return {"ok": True}

    def delete_trigger(self, ctx, tid, t=None):
        t = t or ctx.state.get("wf_triggers", tid)
        if t is None or t["status"] == "deleted":
            return
        if t.get("sched"):
            ctx.unschedule(t["sched"])
        ctx.emit("wf.trigger_status", {"id": tid, "status": "deleted", "reason": ""}, vis=(t["owner"], t["created_by"]))

    def _fire(self, ctx: Ctx, tid: str, extra: dict | None = None) -> None:
        st = ctx.state
        t = st.get("wf_triggers", tid)
        if t is None or t["status"] != "active":
            return
        if not self._still_allowed(ctx, t):
            self.set_status(ctx, tid, "paused", "creator no longer administers the owner")
            return
        ok, err, status = True, None, None
        try:
            args = substitute(t["args"], extra or {}) if extra else t["args"]
            res = ctx.call(t["tool"], args, as_principal=t["owner"])
            if isinstance(res, dict) and isinstance(res.get("status"), int) and res["status"] >= 400:
                ok, status, err = False, res["status"], str(res.get("error", ""))[:200]
        except CrucibleError as e:
            ok, err = False, e.message[:200]
        n_fail = 0 if ok else t["fails_in_row"] + 1
        ctx.emit("wf.fired", {"id": tid, "ok": ok, "error": err, "status": status, "fails_in_row": n_fail},
                 vis=(t["owner"],))
        if not ok:
            host = t["tool"].split(".", 1)[0]
            ops = ctx.world.hooks.get("ops")
            if ops is not None and status in (402, 500, 503) and host not in ctx.world.modules:
                ops.record_loss(ctx, t["owner"], "dependency_failure", 0, 1, resource=host)
            if n_fail >= 10:
                self.set_status(ctx, tid, "paused", "10 consecutive failures")
        if t["max_runs"] is not None and t["runs"] + 1 >= t["max_runs"]:
            self.delete_trigger(ctx, tid)

    @process("fire")
    def pr_fire(self, ctx, id):
        self._fire(ctx, id)

    @process("poll")
    def pr_poll(self, ctx):
        """Event triggers: scan events visible to each owner since the trigger's cursor (bounded per poll)."""
        st = ctx.state
        evs = None
        for tid in st.find_ids("wf_triggers", "status", "active"):
            t = st.tables["wf_triggers"][tid]
            if not t["on_event"]:
                continue
            if evs is None:
                lo = min(x["cursor"] for x in st.tables["wf_triggers"].values() if x["on_event"])
                evs = ctx.world.events_range(max(lo, ctx.world.log_base))
            ps = ctx.world.principals_of(t["owner"]) | {t["owner"]}
            fired = 0
            cursor = t["cursor"]
            for e in evs:
                if e.seq <= t["cursor"]:
                    continue
                cursor = e.seq
                if not e.type.startswith(t["on_event"]) or not ctx.world.visible(e, ps):
                    continue
                if e.actor == t["owner"] and e.type.startswith("wf."):
                    continue
                self._fire(ctx, tid, {"event": {"seq": e.seq, "t": e.t, "type": e.type, "actor": e.actor,
                                                "payload": e.payload}})
                fired += 1
                if fired >= 20 or st.tables["wf_triggers"][tid]["status"] != "active":
                    break
            ctx.emit("wf.cursor", {"id": tid, "cursor": cursor}, vis="internal")

    # ------------------------------------------------------------------ flows
    @endpoint("define", S.obj({"name": S.s_str(maxLength=80),
                               "steps": S.s_arr(STEP, "[{id, tool, args, if?: {ref, op, value}, foreach?: '${..}', "
                                                      "optional?: bool}]", minItems=1, maxItems=MAX_STEPS),
                               "input_schema": S.s_any_obj("JSON schema for the input (validated on every run)"),
                               "output": S.s_str("reference returned as the flow result, e.g. ${s3.result}",
                                                 maxLength=200),
                               "public": S.s_bool("anyone may run it (as themselves)"),
                               "description": S.s_str(maxLength=2000), "as_org": S.s_str(maxLength=40)},
                              ["name", "steps"]), cost_s=30)
    def ep_define(self, ctx, name, steps, input_schema=None, output=None, public=False, description="", as_org=None):
        """Define a reusable flow of tool calls with typed data passing between steps."""
        owner = self._run_as(ctx, as_org)
        ids = set()
        clean = []
        for i, s in enumerate(steps):
            sid = s.get("id", f"s{i + 1}")
            if not isinstance(sid, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,20}", sid) or sid in ids or sid in (
                    "input", "item", "event"):
                raise InvalidAction(f"step {i}: bad or duplicate id {sid!r}")
            ids.add(sid)
            if not isinstance(s.get("tool"), str) or s["tool"].startswith("admin."):
                raise InvalidAction(f"step {sid}: bad tool")
            if not isinstance(s.get("args", {}), dict):
                raise InvalidAction(f"step {sid}: args must be an object")
            clean.append({"id": sid, "tool": s["tool"], "args": s.get("args", {}), "if": s.get("if"),
                          "foreach": s.get("foreach"), "optional": bool(s.get("optional", False))})
        if input_schema is not None:
            errs = S.check_schema(input_schema)
            if errs:
                raise InvalidAction("input_schema: " + "; ".join(errs[:3]))
        fid = ctx.state.new_id("flow")
        ctx.emit("wf.flow", {"id": fid, "owner": owner, "name": name, "steps": clean, "input_schema": input_schema,
                             "output": output, "public": bool(public), "description": description},
                 vis="public" if public else (owner,))
        if public:
            from .search import index
            index(ctx, fid, "flow", name, description + " " + " ".join(s["tool"] for s in clean), owner, ref=fid)
        return {"flow": fid}

    def run_flow(self, ctx: Ctx, fid: str, inp: dict, depth: int = 0) -> dict:
        st = ctx.state
        f = st.get("wf_flows", fid)
        if f is None:
            raise NotFound("no flow " + fid)
        if not f["public"] and not (ctx.actor == f["owner"] or ctx.world.may_act_as(ctx.actor, f["owner"])):
            raise Forbidden("flow is private")
        if depth > 3:
            raise InvalidAction("flow nesting too deep")
        if f["input_schema"]:
            errs = S.validate(f["input_schema"], inp, "input")
            if errs:
                raise InvalidAction("; ".join(errs[:4]))
        env: dict = {"input": inp}
        for s in f["steps"]:
            if not _cond(s["if"], env):
                env[s["id"]] = {"skipped": True, "result": None}
                continue
            items = [None]
            if s["foreach"]:
                items = substitute(s["foreach"], env)
                if not isinstance(items, list):
                    raise InvalidAction(f"step {s['id']}: foreach must reference a list")
                items = items[:MAX_FOREACH]
            results = []
            for it in items:
                e2 = dict(env, item=it)
                args = substitute(s["args"], e2)
                try:
                    if s["tool"] == "wf.run":
                        r = self.run_flow(ctx, args.get("flow"), args.get("input") or {}, depth + 1)
                    else:
                        r = ctx.call(s["tool"], args)
                    results.append({"ok": True, "result": r})
                except CrucibleError as e:
                    if not s["optional"]:
                        raise InvalidAction(f"step {s['id']} ({s['tool']}) failed: {e.message}", step=s["id"])
                    results.append({"ok": False, "error": e.message})
            env[s["id"]] = results[0] if not s["foreach"] else {"ok": all(r["ok"] for r in results),
                                                               "result": [r.get("result") for r in results]}
        out = substitute(f["output"], env) if f["output"] else {k: v for k, v in env.items() if k != "input"}
        return {"flow": fid, "result": out}

    @endpoint("run", S.obj({"flow": S.s_str(maxLength=40), "input": S.s_any_obj()}, ["flow"]), cost_s=10)
    def ep_run(self, ctx, flow, input=None):
        """Run a flow now, as you, atomically (if a non-optional step fails nothing happens)."""
        out = self.run_flow(ctx, flow, input or {})
        ctx.emit("wf.ran", {"flow": flow, "by": ctx.actor}, vis=(ctx.actor,))
        return out

    @endpoint("run_later", S.obj({"flow": S.s_str(maxLength=40), "input": S.s_any_obj(),
                                  "in_s": S.s_int(minimum=60, maximum=90 * 86400)}, ["flow", "in_s"]), cost_s=10)
    def ep_run_later(self, ctx, flow, in_s, input=None):
        """Run a flow asynchronously after `in_s` seconds, as you. The result is recorded in `wf.jobs`."""
        if not ctx.state.has("wf_flows", flow):
            raise NotFound("no flow " + flow)
        jid = ctx.state.new_id("wfjob")
        ctx.schedule(ctx.now + in_s, "wf.job", {"id": jid, "flow": flow, "input": input or {}, "as": ctx.actor},
                     prio=56)
        return {"job": jid}

    @process("job")
    def pr_job(self, ctx, id, flow, input, **kw):
        who = kw.get("as")
        try:
            ok, res = True, ctx.run_as(who, lambda sub: self.run_flow(sub, flow, input))["result"]
        except CrucibleError as e:
            ok, res = False, e.message
        ctx.emit("wf.job_done", {"id": id, "flow": flow, "by": who, "ok": ok, "result": res}, vis=(who,))

    @endpoint("jobs", S.obj(), read_only=True)
    def ep_jobs(self, ctx):
        """Results of your asynchronous flow runs."""
        return [dict(r, job=k) for k, r in ctx.state.scan("wf_runs") if r["by"] == ctx.actor][-50:]

    @endpoint("flows", S.obj({"mine": S.s_bool()}), read_only=True)
    def ep_flows(self, ctx, mine=False):
        """Public flows (or yours)."""
        out = []
        for fid, f in ctx.state.scan("wf_flows"):
            if (mine and f["owner"] in ctx.world.principals_of(ctx.actor)) or (not mine and f["public"]):
                out.append({"flow": fid, "name": f["name"], "owner": f["owner"], "description": f["description"],
                            "steps": [s["tool"] for s in f["steps"]], "input_schema": f["input_schema"]})
        return out

    # ------------------------------------------------------------------ appliers
    @applier("wf.trigger")
    def ap_trigger(self, ev):
        p = dict(ev.payload)
        p.update(status="active", runs=0, fails=0, fails_in_row=0, last=None, t=ev.t)
        self.world.state.insert("wf_triggers", p.pop("id"), p)

    @applier("wf.trigger_status")
    def ap_trigger_status(self, ev):
        self.world.state.patch("wf_triggers", ev.payload["id"], status=ev.payload["status"])

    @applier("wf.fired")
    def ap_fired(self, ev):
        p = ev.payload
        t = self.world.state.tables["wf_triggers"][p["id"]]
        self.world.state.patch("wf_triggers", p["id"], runs=t["runs"] + 1, fails=t["fails"] + (0 if p["ok"] else 1),
                               fails_in_row=p["fails_in_row"], last={"t": ev.t, "ok": p["ok"], "error": p["error"]})

    @applier("wf.cursor")
    def ap_cursor(self, ev):
        self.world.state.patch("wf_triggers", ev.payload["id"], cursor=ev.payload["cursor"])

    @applier("wf.flow")
    def ap_flow(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("wf_flows", p.pop("id"), p)

    @applier("wf.ran")
    def ap_ran(self, ev):
        return None

    @applier("wf.job_done")
    def ap_job_done(self, ev):
        p = ev.payload
        self.world.state.insert("wf_runs", p["id"], {"flow": p["flow"], "by": p["by"], "ok": p["ok"],
                                                    "result": p["result"], "t": ev.t})
